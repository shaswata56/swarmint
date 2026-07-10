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

    # Relay fallback (T4). All off by default so existing runs are byte-for-byte
    # unchanged. When enabled, a message to a peer in `relay_only` is wrapped and
    # sent to `relay_id` (a mutually-reachable node, e.g. the rendezvous), which
    # forwards it on. A node that RECEIVES a relayed message auto-learns to reply
    # the same way. See wire.pack_relay/pack_relayed and decision #033.
    enable_relay: bool = False
    relay_id: object = None            # primary relay (back-compat); tried first
    relay_ids: list = field(default_factory=list)  # additional relay candidates (T6):
                                       # more public/rendezvous nodes to fail over to,
                                       # so no single relay is a point of failure
    relay_only: set = field(default_factory=set)   # peers reachable only via relay
    relayed_tx: int = 0                # relay wraps we sent (as origin)
    relayed_fwd: int = 0               # messages we forwarded (as relay)
    relayed_rx: int = 0                # relayed messages we received (as destination)

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
    RELAY_GOSSIP_CHUNK_SIZE = 8  # smaller when relaying: the outer relay envelope
                                 # wraps the inner one, so the inner must leave room
                                 # to keep the wrapped datagram under MAX_DATAGRAM_BYTES

    def _relay_candidates(self):
        """Ordered, de-duplicated relay node_ids to try: the primary first, then
        the extra candidates (T6). None entries dropped."""
        seen = set()
        out = []
        for rid in [self.relay_id, *self.relay_ids]:
            if rid is not None and rid not in seen:
                seen.add(rid)
                out.append(rid)
        return out

    def _relay_target_addr(self, recipient):
        """If `recipient` should be reached via a relay, return the address of
        the first candidate relay we can actually reach (else None). Failing
        over across candidates means one relay going down doesn't partition the
        peer — the swarm keeps a path as long as ANY shared relay is up."""
        if not (self.enable_relay and recipient in self.relay_only):
            return None
        for rid in self._relay_candidates():
            if rid == recipient:
                continue  # a peer can't relay to itself
            addr = self.peer_addrs.get(rid)
            if addr is not None:
                return addr
        return None

    def _raw_send(self, data: bytes, addr) -> bool:
        """Put one datagram on the wire, honoring injected loss. Returns True if
        actually sent. Shared by direct, relay-wrap, and relay-forward paths."""
        if len(data) > wire.MAX_DATAGRAM_BYTES:
            self.dropped_oversized += 1
            return False
        if self.loss_rate > 0.0 and random.random() < self.loss_rate:
            self.dropped_loss += 1  # injected loss (chaos); silent, like real UDP
            return False
        self._transport.sendto(data, addr)
        self.tx_bytes += len(data)
        return True

    def send(self, msg: Message) -> None:
        if self._transport is None:
            return
        encoder = self._encoders.get(msg.kind)
        if encoder is None:
            return
        relay_addr = self._relay_target_addr(msg.recipient)
        direct_addr = self.peer_addrs.get(msg.recipient)
        if relay_addr is None and direct_addr is None:
            return  # unknown peer, no relay — silent drop, matches UDP semantics
        relaying = relay_addr is not None
        # SwarmNode.gossip_step() samples up to 30 prototypes per call — that
        # figure predates the 1200B wire budget and is untouched by design
        # (node logic invariant). Without chunking here, any batch over the
        # budget silently exceeds MAX_DATAGRAM_BYTES and gets dropped. When
        # relaying we chunk smaller so the wrapped datagram still fits.
        chunk_size = self.RELAY_GOSSIP_CHUNK_SIZE if relaying else self.GOSSIP_CHUNK_SIZE
        if isinstance(msg.payload, list) and len(msg.payload) > chunk_size:
            chunks = [msg.payload[i:i + chunk_size]
                     for i in range(0, len(msg.payload), chunk_size)]
        else:
            chunks = [msg.payload]
        for chunk in chunks:
            body = encoder(chunk)
            inner = self.identity.sign_envelope(body, time.time())
            if len(inner) > wire.MAX_DATAGRAM_BYTES:
                self.dropped_oversized += 1  # a non-list or unchunkable payload that's still too big
                continue
            if relaying:
                wrapped = wire.pack_relay({"target": msg.recipient, "blob": inner})
                outer = self.identity.sign_envelope(wrapped, time.time())
                if self._raw_send(outer, relay_addr):
                    self.relayed_tx += 1
            else:
                self._raw_send(inner, direct_addr)

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
        # Relay fallback (T4): handled entirely in the transport so SwarmNode
        # never sees a relay/relayed message — it only ever receives the inner.
        if kind == "relay":
            if self.enable_relay:
                self._forward_relay(result.sender, result.body)
            return
        if kind == "relayed":
            self._receive_relayed(result.sender, result.body)
            return
        decoder = self._decoders.get(kind)
        if decoder is None:
            return
        payload = decoder(result.body)
        if self._handler is not None:
            self._handler(Message(sender=result.sender, recipient=self.identity.node_id,
                                  kind=kind, payload=payload))

    # ---- relay fallback (T4) ----

    def _forward_relay(self, origin, body: bytes) -> None:
        """We are the relay: `origin` asked us to deliver an inner blob to a
        target. Forward it (untouched, still origin-signed) wrapped so the
        target knows it came via us. Drop silently if we can't reach the target
        directly — a relay only bridges peers it can itself reach."""
        info = wire.unpack_relay(body)
        target_addr = self.peer_addrs.get(info["target"])
        if target_addr is None:
            return
        wrapped = wire.pack_relayed({"origin": origin, "blob": info["blob"]})
        outer = self.identity.sign_envelope(wrapped, time.time())
        if self._raw_send(outer, target_addr):
            self.relayed_fwd += 1

    def _receive_relayed(self, relay_sender, body: bytes) -> None:
        """We are the destination: unwrap a relayed blob, verify the ORIGINAL
        sender's signature on it, remember to reply via this same relay, and
        deliver the inner message as if it had arrived directly."""
        info = wire.unpack_relayed(body)
        origin, blob = info["origin"], info["blob"]
        res = verify_envelope(blob, self.replay_guard, self.max_clock_skew, now=time.time())
        if not res.ok or res.sender != origin:
            self.dropped_unverified += 1
            return
        self.relayed_rx += 1
        # Learn the return path: the origin is only reachable via a relay, so
        # our own sends back to it must wrap the same way (symmetric relay). Also
        # remember this relay as a candidate for future failover (T6) — any node
        # that can deliver to us is a usable relay.
        if self.enable_relay:
            if self.relay_id is None:
                self.relay_id = relay_sender
            elif relay_sender != self.relay_id and relay_sender not in self.relay_ids:
                self.relay_ids.append(relay_sender)
            self.relay_only.add(origin)
        inner_kind = wire.peek_kind(res.body)
        decoder = self._decoders.get(inner_kind)
        if decoder is None:
            return
        payload = decoder(res.body)
        if self._handler is not None:
            self._handler(Message(sender=origin, recipient=self.identity.node_id,
                                  kind=inner_kind, payload=payload))

    def _on_transport_lost(self, exc: Exception) -> None:
        if self._loop is not None:
            self._loop.create_task(self._rebind())

    async def _rebind(self) -> None:
        self._transport, self._protocol = await self._loop.create_datagram_endpoint(
            lambda: _Protocol(self._on_datagram, self._on_transport_lost),
            local_addr=(self._host, self.bound_port),
        )
