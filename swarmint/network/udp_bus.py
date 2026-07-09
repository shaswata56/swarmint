"""Real asyncio UDP transport implementing the Bus protocol.

One UdpBus = one node = one bound socket (unlike SimBus, which is a single
shared instance for the whole in-process simulation). Encoders/decoders are
registered per message kind so later tasks (T5 discovery, T6 inference RPC)
can add new kinds without editing this file.

Windows quirk (decision #023 D10): sending UDP to a peer with nothing
listening on the destination port can trigger an ICMP port-unreachable that
Windows surfaces as a socket reset, which can tear down the datagram
transport (WinError 10054, historically via ProactorEventLoop). Two
mitigations, belt-and-suspenders: (1) best-effort disable via the
SIO_UDP_CONNRESET socket ioctl; (2) if the transport dies anyway,
connection_lost() rebinds it at the same port automatically.
"""

import asyncio
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import wire
from .bus import Message
from .identity import Identity, ReplayGuard, verify_envelope

_SIO_UDP_CONNRESET = 0x9800000C


def _disable_connreset(sock) -> None:
    if sys.platform != "win32" or sock is None:
        return
    try:
        sock.ioctl(_SIO_UDP_CONNRESET, False)
    except (AttributeError, OSError):
        pass  # best-effort; connection_lost rebind is the fallback


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, on_datagram: Callable, on_lost: Callable):
        self._on_datagram = on_datagram
        self._on_lost = on_lost
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        _disable_connreset(transport.get_extra_info("socket"))

    def datagram_received(self, data: bytes, addr) -> None:
        self._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        pass  # non-fatal transport error (e.g. ICMP port-unreachable) — ignore

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if exc is not None:
            self._on_lost(exc)


@dataclass
class UdpBus:
    identity: Identity
    peer_addrs: dict = field(default_factory=dict)   # node_id -> (host, port)
    replay_guard: ReplayGuard = field(default_factory=ReplayGuard)
    max_clock_skew: float = 30.0
    loss_rate: float = 0.0   # simulated outbound packet loss (chaos testing, T8).
                             # Real UDP already loses packets; this lets us inject
                             # a controlled rate to prove convergence survives it.
    rx_bytes: int = 0
    tx_bytes: int = 0
    dropped_oversized: int = 0
    dropped_unverified: int = 0
    dropped_loss: int = 0

    _encoders: dict = field(default_factory=lambda: {"protos": wire.pack_gossip_protos})
    _decoders: dict = field(default_factory=lambda: {"protos": wire.unpack_gossip_protos})
    _handler: Optional[Callable] = None
    _transport = None
    _protocol = None
    _loop = None
    _host: str = "127.0.0.1"
    bound_port: int = 0

    def register_kind(self, kind: str, encoder: Callable, decoder: Callable) -> None:
        """Extension point for T5 (advert/pex) and T6 (query/response/calib_*)."""
        self._encoders[kind] = encoder
        self._decoders[kind] = decoder

    def register(self, node_id, handler: Callable[[Message], None]) -> None:
        if node_id != self.identity.node_id:
            raise ValueError("UdpBus is single-node: node_id must match its own identity")
        self._handler = handler

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self._loop = asyncio.get_running_loop()
        self._host = host
        self._transport, self._protocol = await self._loop.create_datagram_endpoint(
            lambda: _Protocol(self._on_datagram, self._on_transport_lost),
            local_addr=(host, port),
        )
        self.bound_port = self._transport.get_extra_info("sockname")[1]
        return self.bound_port

    async def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()

    GOSSIP_CHUNK_SIZE = 12  # decision #023 D9: measured budget for a 12-proto envelope

    def send(self, msg: Message) -> None:
        addr = self.peer_addrs.get(msg.recipient)
        if addr is None or self._transport is None:
            return  # unknown peer or not started — silent drop, matches UDP semantics
        encoder = self._encoders.get(msg.kind)
        if encoder is None:
            return
        # SwarmNode.gossip_step() samples up to 30 prototypes per call — that
        # figure predates the 1200B wire budget and is untouched by design
        # (node logic invariant). Without chunking here, any batch over ~12
        # prototypes silently exceeds MAX_DATAGRAM_BYTES and gets dropped
        # every time once a node's model grows past that size, which looked
        # like "gossip just stops working" rather than an obvious size error.
        if isinstance(msg.payload, list) and len(msg.payload) > self.GOSSIP_CHUNK_SIZE:
            chunks = [msg.payload[i:i + self.GOSSIP_CHUNK_SIZE]
                     for i in range(0, len(msg.payload), self.GOSSIP_CHUNK_SIZE)]
        else:
            chunks = [msg.payload]
        for chunk in chunks:
            body = encoder(chunk)
            envelope = self.identity.sign_envelope(body, time.time())
            if len(envelope) > wire.MAX_DATAGRAM_BYTES:
                self.dropped_oversized += 1  # a non-list or unchunkable payload that's still too big
                continue
            if self.loss_rate > 0.0 and random.random() < self.loss_rate:
                self.dropped_loss += 1  # injected loss (chaos); silent, like real UDP
                continue
            self._transport.sendto(envelope, addr)
            self.tx_bytes += len(envelope)

    def _on_datagram(self, data: bytes, addr) -> None:
        self.rx_bytes += len(data)
        result = verify_envelope(data, self.replay_guard, self.max_clock_skew, now=time.time())
        if not result.ok:
            self.dropped_unverified += 1
            return
        # Learn the sender's reachable address from ANY verified message — this
        # is how a rendezvous node (which has no way to initiate contact with a
        # joiner) ever learns a joining peer's return address. Safe because
        # `result.sender` is cryptographically verified, not attacker-chosen.
        self.peer_addrs[result.sender] = addr
        kind = wire.peek_kind(result.body)
        decoder = self._decoders.get(kind)
        if decoder is None:
            return
        payload = decoder(result.body)
        if self._handler is not None:
            self._handler(Message(sender=result.sender, recipient=self.identity.node_id,
                                  kind=kind, payload=payload))

    def _on_transport_lost(self, exc: Exception) -> None:
        if self._loop is not None:
            self._loop.create_task(self._rebind())

    async def _rebind(self) -> None:
        self._transport, self._protocol = await self._loop.create_datagram_endpoint(
            lambda: _Protocol(self._on_datagram, self._on_transport_lost),
            local_addr=(self._host, self.bound_port),
        )
