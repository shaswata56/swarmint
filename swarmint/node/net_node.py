"""NetNode: wires a SwarmNode to real UDP + Kademlia DHT + peer-exchange so a
worker process (T7 multiproc harness, T8 chaos test) just calls start()/stop().

Peer discovery here is DHT-rendezvous + PEX growth (decision #023 D6), not a
static peer list: a joining node is told only ONE well-known bootstrap
identity (a standard pattern — like a DNS seed node), looks up that specific
node's address via the DHT, and grows its peer set from there via PEX
messages piggybacked on the gossip cadence.
"""

import asyncio
import random
from dataclasses import dataclass, field

from ..core.update_chain import UpdateChain
from ..inference.rpc import wire_up
from ..network.bus import Message
from ..network.discovery import Discovery
from ..network.identity import Identity
from ..network.nat import HolePuncher
from ..network.udp_bus import UdpBus
from ..network.wire import (EMBEDDING_CHUNK_BYTES, pack_chain_request, pack_chain_response,
                            pack_embedding_request, pack_embedding_response, pack_pex,
                            unpack_chain_request, unpack_chain_response, unpack_embedding_request,
                            unpack_embedding_response, unpack_pex)
from .runner import NodeRunner
from .swarm_node import SwarmNode

PEX_EVERY_N_ROUNDS = 5  # send our known-peer list to a random peer this often
CHECKPOINT_EVERY_N_ROUNDS = 3  # append a signed model checkpoint to the hash-chain (D1)


@dataclass
class NetNode:
    identity: Identity
    topic: int                # primary topic (backward compat / SwarmNode)
    rng: random.Random
    topics: set = None        # full topic set this node participates in (D2);
                              # defaults to {topic}. Advertised via DHT + PEX so
                              # peers can route gossip/inference by topic.
    malicious: bool = False
    n_classes: int = 10
    gossip_interval: float = 2.0
    gossip_sample_size: int = 30  # lowered by the harness to fit the bandwidth budget
    loss_rate: float = 0.0        # injected outbound packet loss (chaos testing, T8)
    enable_chain: bool = True     # maintain a tamper-evident model-checkpoint chain (D1)
    enable_nat: bool = True       # attach the NAT hole-punch signaler (D3)
    embedding: object = None      # shared frozen genesis embedding (E1-3). Every node MUST
                                  # use the identical one — provision it here, or a joiner
                                  # fetches it from the rendezvous via fetch_embedding().
    data_feed: object = None  # () -> (x, y) | None

    bus: UdpBus = field(default=None, init=False)
    discovery: Discovery = field(default=None, init=False)
    node: SwarmNode = field(default=None, init=False)
    runner: NodeRunner = field(default=None, init=False)
    inference: object = field(default=None, init=False)
    chain: UpdateChain = field(default=None, init=False)
    nat: HolePuncher = field(default=None, init=False)
    _pex_tick: int = field(default=0, init=False)
    _last_ckpt_counter: int = field(default=-1, init=False)

    async def start(self, host: str, gossip_port: int, dht_port: int,
                    bootstrap_dht_addrs: list, rendezvous_id: bytes = None) -> None:
        self.bus = UdpBus(identity=self.identity, loss_rate=self.loss_rate)
        await self.bus.start(host, gossip_port)
        self.bus.register_kind("pex", pack_pex, unpack_pex)

        self.discovery = Discovery(identity=self.identity)
        self.discovery.topics = set(self.topics) if self.topics else {self.topic}
        await self.discovery.start(host, dht_port, bootstrap_dht_addrs, host, self.bus.bound_port)

        self.node = SwarmNode(node_id=self.identity.node_id, topic=self.topic, bus=self.bus,
                              rng=self.rng, malicious=self.malicious, n_classes=self.n_classes,
                              gossip_sample_size=self.gossip_sample_size)
        self.inference = wire_up(self.node, self.bus)  # installs the base protos+RPC dispatcher

        if self.enable_chain:
            self.chain = UpdateChain(self.identity)
            self.bus.register_kind("chain_req", pack_chain_request, unpack_chain_request)
            self.bus.register_kind("chain_resp", pack_chain_response, unpack_chain_response)

        if self.enable_nat:
            self.nat = HolePuncher(self.identity, self.bus, self.discovery)

        # Genesis embedding distribution: a node that holds the shared frozen
        # projection serves it; a joiner fetches it (chunked) so every node ends
        # up in the identical embedded space (the shared-space invariant).
        self.bus.register_kind("emb_req", pack_embedding_request, unpack_embedding_request)
        self.bus.register_kind("emb_resp", pack_embedding_response, unpack_embedding_response)

        base_dispatch = self.bus._handler

        def dispatch(msg: Message) -> None:
            # Any verified sender is a usable gossip peer, even before PEX
            # spreads further — this is how a rendezvous node (which never
            # initiates contact, and so never calls discovery.lookup() on
            # anyone) learns about the node that just joined it. Recording
            # into discovery.peer_addrs (not just node.peers/bus.peer_addrs)
            # is essential: pex_payload() reads ONLY from discovery.peer_addrs,
            # so without this line the rendezvous node's own PEX payload stays
            # permanently empty and it can never introduce peers to each other.
            if msg.sender != self.identity.node_id:
                if msg.sender not in self.node.peers:
                    self.node.peers.append(msg.sender)
                observed_addr = self.bus.peer_addrs.get(msg.sender)
                if observed_addr is not None:
                    self.discovery.peer_addrs.setdefault(msg.sender, observed_addr)
            if msg.kind == "pex":
                self._on_pex(msg)
            elif msg.kind == "chain_req":
                self._on_chain_request(msg)
            elif msg.kind == "chain_resp":
                self._on_chain_response(msg)
            elif msg.kind == "nat":
                if self.nat is not None:
                    self.nat.handle(msg)
            elif msg.kind == "emb_req":
                self._on_embedding_request(msg)
            elif msg.kind == "emb_resp":
                self._on_embedding_response(msg)
            else:
                base_dispatch(msg)

        self.bus.register(self.identity.node_id, dispatch)

        if rendezvous_id is not None:
            # The bootstrap node may not have published its advert yet (it
            # needed US to become its first DHT neighbor before its own
            # publish could succeed — see Discovery's fast-retry loop). Retry
            # the lookup briefly rather than giving up on a one-shot race.
            addr = None
            for _ in range(20):  # ~4s at 0.2s steps — comfortably covers Discovery.FAST_RETRY_S
                addr = await self.discovery.lookup(rendezvous_id)
                if addr is not None:
                    break
                await asyncio.sleep(0.2)
            if addr is not None:
                self.bus.peer_addrs[rendezvous_id] = addr
                self.node.peers.append(rendezvous_id)

        def feed_and_pex():
            self._pex_tick += 1
            if self._pex_tick % PEX_EVERY_N_ROUNDS == 0 and self.node.peers:
                target = self.rng.choice(self.node.peers)
                self.bus.send(Message(self.identity.node_id, target, "pex",
                                      self.discovery.pex_payload()))
            # Append a signed checkpoint when the model has actually advanced —
            # skip idle rounds so the chain records real state transitions, not
            # heartbeat noise.
            if (self.chain is not None
                    and self._pex_tick % CHECKPOINT_EVERY_N_ROUNDS == 0
                    and self.node.model.version.counter != self._last_ckpt_counter):
                self.chain.checkpoint(self.node.model.version, self.node.model.protos)
                self._last_ckpt_counter = self.node.model.version.counter
            sample = self.data_feed() if self.data_feed is not None else None
            if sample is not None and self.embedding is not None:
                x, y = sample                     # map raw input into the SHARED embedded space
                sample = (self.embedding.transform(x), y)
            return sample

        self.runner = NodeRunner(node=self.node, gossip_interval=self.gossip_interval,
                                 data_feed=feed_and_pex)
        await self.runner.start()

    def _on_pex(self, msg: Message) -> None:
        self.discovery.ingest_pex(msg.payload)
        for nid, addr in self.discovery.peer_addrs.items():
            if nid not in self.node.peers:
                self.node.peers.append(nid)
            self.bus.peer_addrs.setdefault(nid, addr)

    async def infer_topic(self, x, topic: int, top_n: int = 20, timeout: float = 0.1):
        """Topic-scoped distributed inference (D2): query only peers that
        advertise `topic`, instead of the whole peer set. This is the payoff of
        finer topic granularity — as the network grows, a query fans out to the
        handful of relevant experts rather than everyone. Falls back to all
        peers if none are known for the topic yet."""
        candidates = self.discovery.peers_for_topic(topic)
        if not candidates:
            candidates = list(self.node.peers)
        # Embed the query ONCE at the origin so the vector on the wire is already
        # in the shared space — responders (whose models live there) answer directly.
        xq = self.embedding.transform(x) if self.embedding is not None else x
        return await self.inference.infer(xq, candidates, top_n=top_n, timeout=timeout)

    # ---- genesis embedding distribution ----

    def _on_embedding_request(self, msg: Message) -> None:
        """Serve our shared frozen embedding to a joiner, chunked."""
        if self.embedding is None:
            return
        blob = self.embedding.to_bytes()
        chunks = [blob[i:i + EMBEDDING_CHUNK_BYTES] for i in range(0, len(blob), EMBEDDING_CHUNK_BYTES)] or [b""]
        for i, chunk in enumerate(chunks):
            self.bus.send(Message(self.identity.node_id, msg.sender, "emb_resp",
                                  {"chunk_index": i, "total_chunks": len(chunks), "data": chunk}))

    def _on_embedding_response(self, msg: Message) -> None:
        entry = getattr(self, "_pending_emb", {}).get(msg.sender)
        if entry is None:
            return
        p = msg.payload
        entry["chunks"][p["chunk_index"]] = p["data"]
        entry["total"] = p["total_chunks"]
        if len(entry["chunks"]) >= entry["total"] and not entry["fut"].done():
            entry["fut"].set_result(b"".join(entry["chunks"][i] for i in range(entry["total"])))

    async def fetch_embedding(self, peer_id, timeout: float = 3.0):
        """Pull + install the shared frozen embedding from a peer (the genesis /
        rendezvous node). After this, our inputs map into the same space as
        everyone else's — the shared-space invariant, bootstrapped over the wire."""
        from ..core.embedding import Embedding
        fut = asyncio.get_running_loop().create_future()
        self._pending_emb = getattr(self, "_pending_emb", {})
        self._pending_emb[peer_id] = {"fut": fut, "chunks": {}, "total": None}
        self.bus.send(Message(self.identity.node_id, peer_id, "emb_req", {}))
        try:
            blob = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_emb.pop(peer_id, None)
        self.embedding = Embedding.from_bytes(blob)
        return self.embedding

    def _on_chain_request(self, msg: Message) -> None:
        """A peer asked to audit our update history — send our whole chain,
        chunked so each datagram stays within the 1200B budget."""
        if self.chain is None:
            return
        from ..network.wire import CHAIN_ENTRIES_PER_CHUNK
        entries = self.chain.to_wire()
        chunks = [entries[i:i + CHAIN_ENTRIES_PER_CHUNK]
                  for i in range(0, len(entries), CHAIN_ENTRIES_PER_CHUNK)] or [[]]
        for i, chunk in enumerate(chunks):
            self.bus.send(Message(self.identity.node_id, msg.sender, "chain_resp",
                                  {"chunk_index": i, "total_chunks": len(chunks), "entries": chunk}))

    def _on_chain_response(self, msg: Message) -> None:
        entry = getattr(self, "_pending_chain", {}).get(msg.sender)
        if entry is None:
            return
        p = msg.payload
        entry["chunks"][p["chunk_index"]] = p["entries"]
        entry["total"] = p["total_chunks"]
        if len(entry["chunks"]) >= entry["total"] and not entry["fut"].done():
            ordered = [e for i in range(entry["total"]) for e in entry["chunks"][i]]
            entry["fut"].set_result(ordered)

    async def fetch_chain(self, peer_id, timeout: float = 1.5):
        """Request and verify a peer's tamper-evident chain (reassembled from
        chunks). Returns (ok, reason, entries) — entries is the raw wire list so
        the caller can inspect what the peer claimed even on a verify failure."""
        from ..core.update_chain import verify_chain

        fut = asyncio.get_running_loop().create_future()
        self._pending_chain = getattr(self, "_pending_chain", {})
        self._pending_chain[peer_id] = {"fut": fut, "chunks": {}, "total": None}
        self.bus.send(Message(self.identity.node_id, peer_id, "chain_req", {}))
        try:
            entries = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return False, "timeout", None
        finally:
            self._pending_chain.pop(peer_id, None)
        ok, reason = verify_chain(entries, expected_node_id=peer_id)
        return ok, reason, entries

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.stop()
        if self.discovery is not None:
            await self.discovery.stop()
        if self.bus is not None:
            await self.bus.stop()
