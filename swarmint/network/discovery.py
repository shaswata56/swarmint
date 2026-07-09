"""Kademlia-backed rendezvous + gossip-piggybacked peer-exchange (T5).

Decision #023 D6 / plan review gap #2: the `kademlia` pip library's
Server.set/get is a single-value last-writer-wins key-value store. Storing
every peer's address under one shared topic key (the original T5 design)
would mean each advertisement clobbers the last, leaving exactly ONE peer
discoverable — not flaky, structurally broken.

The fix: every node publishes its OWN signed advert under a key derived from
its OWN node_id (so multiple nodes never collide on one key), and the peer
SET grows via peer-exchange (PEX) once at least one peer is known — the DHT
is used for rendezvous (finding a first peer / a specific peer), not as a
peer-list store.

The library's built-in value republish is an hourly origin-refresh only,
independent of our own staleness policy, so TTL refresh/expiry here are
custom loops, not something the library does for us.
"""

import asyncio
import time
from dataclasses import dataclass, field

from kademlia.network import Server

from . import wire
from .identity import Identity, ReplayGuard, verify_envelope

ADVERT_TTL_S = 60.0        # decision #023 D6
ADVERT_REFRESH_S = 30.0    # refresh at half the TTL so a missed cycle doesn't expire us
STALE_AFTER_S = ADVERT_TTL_S * 2  # expire a peer we haven't refreshed in this long
FAST_RETRY_S = 1.0         # retry cadence BEFORE the first successful publish — a lone
                           # bootstrap node has no DHT neighbors yet (kademlia's set()
                           # silently no-ops), so without this it would stay unpublished
                           # for up to ADVERT_REFRESH_S after the first peer joins


def _advert_key(node_id: bytes) -> bytes:
    return b"swarmint:advert:" + node_id


@dataclass
class Discovery:
    identity: Identity
    peer_addrs: dict = field(default_factory=dict)     # node_id -> (host, port)
    peer_seen_at: dict = field(default_factory=dict)   # node_id -> last-refresh wall time
    peer_topics: dict = field(default_factory=dict)    # node_id -> set[int]  (D2)
    topics: set = field(default_factory=lambda: {1})   # this node's topic set (D2)
    replay_guard: ReplayGuard = field(default_factory=ReplayGuard)
    server: Server = field(default_factory=Server)
    last_publish_ok: bool = field(default=False, init=False)
    _refresh_task: asyncio.Task = field(default=None, init=False)
    _my_addr: tuple = field(default=None, init=False)

    def peers_for_topic(self, topic: int) -> list:
        """Peer ids that advertise membership in `topic` (D2 topic-scoped
        routing). Peers whose topic set we don't know yet are included as a
        fallback so routing degrades to broadcast rather than silence."""
        return [nid for nid in self.peer_addrs
                if topic in self.peer_topics.get(nid, set()) or nid not in self.peer_topics]

    async def start(self, dht_host: str, dht_port: int, bootstrap_nodes: list,
                    advertised_host: str, advertised_port: int) -> None:
        """bootstrap_nodes: list of (host, port) DHT contacts — pass 2-3 for
        resilience (decision #023 D10: a single bootstrap is a chaos-test SPOF).
        advertised_(host,port) is the node's GOSSIP address (UdpBus), which is
        what peers actually need — the DHT listen port is a separate concern."""
        await self.server.listen(dht_port, interface=dht_host)
        if bootstrap_nodes:
            await self.server.bootstrap(bootstrap_nodes)
        self._my_addr = (advertised_host, advertised_port)
        await self._publish_advert()
        self._refresh_task = asyncio.get_running_loop().create_task(self._refresh_loop())

    async def stop(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        self.server.stop()

    async def _publish_advert(self) -> bool:
        """Returns False (not raises) if the DHT has no known neighbors yet —
        kademlia's set() silently no-ops in that case (e.g. this is the very
        first node in the network before anyone has bootstrapped to it). Not
        fatal: _refresh_loop retries every ADVERT_REFRESH_S, so this self-heals
        once a peer joins. last_publish_ok makes the state observable instead
        of silently swallowed (T7's metrics should surface it per-node)."""
        host, port = self._my_addr
        body = wire.pack_advert(self.identity.node_id, f"{host}:{port}", self.topics, ts=time.time())
        envelope = self.identity.sign_envelope(body, time.time())
        self.last_publish_ok = bool(await self.server.set(_advert_key(self.identity.node_id), envelope))
        return self.last_publish_ok

    async def _refresh_loop(self) -> None:
        try:
            while not self.last_publish_ok:
                await asyncio.sleep(FAST_RETRY_S)
                await self._publish_advert()
            while True:
                await asyncio.sleep(ADVERT_REFRESH_S)
                await self._publish_advert()
                self._expire_stale()
        except asyncio.CancelledError:
            raise

    def _expire_stale(self) -> None:
        now = time.time()
        stale = [nid for nid, seen in self.peer_seen_at.items() if now - seen > STALE_AFTER_S]
        for nid in stale:
            self.peer_addrs.pop(nid, None)
            self.peer_seen_at.pop(nid, None)

    async def lookup(self, node_id: bytes):
        """Rendezvous: resolve a SPECIFIC node's address via the DHT (e.g. the
        first contact with a peer whose id you know but not its address)."""
        raw = await self.server.get(_advert_key(node_id))
        if raw is None:
            return None
        return self._ingest_advert(raw)

    def _ingest_advert(self, envelope_bytes: bytes):
        result = verify_envelope(envelope_bytes, self.replay_guard)
        if not result.ok:
            return None  # forged/expired/replayed advert — anti-eclipse
        advert = wire.unpack_advert(result.body)
        if advert["node_id"] != result.sender:
            return None  # advert content disagrees with its own verified envelope
        host, port_s = advert["addr"].rsplit(":", 1)
        addr = (host, int(port_s))
        self.peer_addrs[result.sender] = addr
        self.peer_seen_at[result.sender] = time.time()
        self.peer_topics[result.sender] = set(advert.get("topics", []))
        return addr

    # ---- peer-exchange: growth beyond the DHT (decision #023 D6) ----

    def known_peers(self) -> list:
        return list(self.peer_addrs.keys())

    def ingest_pex(self, peers: list) -> None:
        """peers: [(node_id, "host:port", topics), ...] from a received PEX
        message. Unverified (PEX rides on the already-signed gossip envelope
        from a trusted-enough peer, not independently signed per-entry) — an
        untrusted PEX source can only ever suggest addresses to DIAL and topic
        HINTS for routing, never inject authoritative state; the DHT
        `lookup()`/advert verification is the trust boundary for accepting a
        peer's claimed identity."""
        for entry in peers:
            node_id, addr_str = entry[0], entry[1]
            topics = set(entry[2]) if len(entry) > 2 else set()
            if node_id == self.identity.node_id:
                continue
            if topics:  # learn topic hints even for peers we already know
                self.peer_topics.setdefault(node_id, set()).update(topics)
            if node_id in self.peer_addrs:
                continue
            host, port_s = addr_str.rsplit(":", 1)
            self.peer_addrs[node_id] = (host, int(port_s))
            self.peer_seen_at[node_id] = time.time()

    def pex_payload(self) -> list:
        return [(nid, f"{host}:{port}", sorted(self.peer_topics.get(nid, set())))
                for nid, (host, port) in self.peer_addrs.items()]
