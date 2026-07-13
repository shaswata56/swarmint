"""Beacon federation: a decentralized directory of beacons ("brain regions").

Each beacon is a rendezvous/relay hosting its own local swarm. The federation is
the map of WHERE every beacon is and how to reach it — the white-matter tract map
connecting the regions. There is NO master: every beacon is a peer that holds the
FULL directory (links to all beacons and their swarms) and gossips it. The GENESIS
beacon (beacon.swarmint.org) is special in exactly one way — it's the well-known
DNS name a newcomer contacts FIRST to bootstrap into the mesh. It has no authority:
adverts are signed by each beacon's own Ed25519 identity and gossiped peer-to-peer
as soft state, so the directory survives the genesis dying (a restarted genesis
just re-learns it from the others), and a beacon can bootstrap off ANY beacon it
knows — the genesis is only the default entry point.

Two layers, kept apart on purpose:

  BeaconRegistry   PURE soft-state table (no I/O): record signed adverts, track
                   reachability, expire stale entries, produce the gossip set.
  FederationService bus-wired glue: verify incoming advert blobs, probe reachability
                   with a signed ping/pong to the ADVERTISED address (the live proof
                   an operator's public host+port actually accept inbound — this is
                   the "is my beacon setup working?" check, and it doubles as
                   anti-Sybil: only genuinely-reachable beacons are marked up).

An advert carries both ports + the task/space the beacon's swarm runs, i.e. exactly
what a node on beacon A needs to reach the swarm on beacon B — the seam that makes
cross-beacon "one singular system" traffic a later routing extension, not a rewrite.
"""

import hashlib
import os
import time
from dataclasses import dataclass, field

from . import wire
from .bus import Message
from .identity import Identity, ReplayGuard, verify_envelope


def space_fingerprint(task: str, n_classes: int, embedding=None,
                      dim: int = None, world_seed: int = None) -> str:
    """A stable fingerprint of the shared embedding SPACE a beacon's swarm lives
    in. Two beacons cross-learn only if these match — identical genesis space =>
    prototype vectors are mutually meaningful. For an embedded task (digits) the
    frozen projection's bytes fully pin the space; for a raw task (synthetic) the
    (dim, world_seed) do. Anything space-defining goes in; nothing else does, so
    two independently-configured beacons on the same task converge to the same fp."""
    h = hashlib.sha1()
    h.update(task.encode())
    h.update(f"|nc={n_classes}|dim={dim}|ws={world_seed}|".encode())
    if embedding is not None:
        h.update(b"emb:")
        h.update(embedding.to_bytes())
    return h.hexdigest()

BEACON_TTL_S = 90.0          # drop a beacon we haven't re-heard an advert for in this long
REACHABLE_TTL_S = 45.0       # "reachable" decays faster than mere existence: a beacon stays
                             # KNOWN via gossip while its direct pong goes quiet (firewall
                             # closed, host down) — so this is half of BEACON_TTL_S
GOSSIP_FANOUT = 4            # beacons we push the known-set to each federation tick
GOSSIP_BLOB_GROUP = 3        # signed adverts per datagram (~300B each) to stay <1200B budget;
                             # a larger known-set is SPLIT across datagrams, never truncated


@dataclass
class BeaconInfo:
    node_id: bytes
    host: str
    gossip_port: int
    dht_port: int
    task: str
    n_classes: int
    topics: list
    name: str
    space_fp: str                # shared-space fingerprint (see space_fingerprint)
    advert_ts: float             # ts from the signed advert body (newest wins)
    blob: bytes                  # the signed advert envelope, re-gossiped verbatim
    seen_at: float               # wall time we last accepted an advert for it
    reachable: bool = False      # confirmed by a ping/pong to its advertised address
    last_pong_at: float = None

    @property
    def addr(self) -> tuple:
        return (self.host, self.gossip_port)


@dataclass
class BeaconRegistry:
    """PURE (no sockets): the shared soft-state directory. Everything here is a
    deterministic function of the adverts fed in + the clock passed to it, so it
    is fully unit-testable without a network."""
    beacons: dict = field(default_factory=dict)   # node_id -> BeaconInfo
    self_id: bytes = None                          # never list ourselves
    own_fp: str = ""                               # our own space fingerprint (for same_space)

    def record(self, advert: dict, blob: bytes, now: float) -> bool:
        """Insert/refresh a verified beacon advert. Returns True if this is a NEW
        beacon (caller then probes reachability). Older adverts never regress a
        newer entry (gossip can deliver stale blobs out of order)."""
        nid = advert["node_id"]
        if nid == self.self_id:
            return False
        existing = self.beacons.get(nid)
        if existing is not None and advert["ts"] < existing.advert_ts:
            return False  # stale duplicate — keep the newer one
        is_new = existing is None
        self.beacons[nid] = BeaconInfo(
            node_id=nid, host=advert["host"], gossip_port=advert["gossip_port"],
            dht_port=advert["dht_port"], task=advert["task"], n_classes=advert["n_classes"],
            topics=list(advert["topics"]), name=advert.get("name", ""),
            space_fp=advert.get("space_fp", ""), advert_ts=advert["ts"],
            blob=blob, seen_at=now,
            reachable=(existing.reachable if existing else False),
            last_pong_at=(existing.last_pong_at if existing else None))
        return is_new

    def mark_reachable(self, node_id: bytes, now: float) -> None:
        info = self.beacons.get(node_id)
        if info is not None:
            info.reachable = True
            info.last_pong_at = now

    def expire(self, now: float) -> None:
        for nid, info in list(self.beacons.items()):
            if now - info.seen_at > BEACON_TTL_S:
                del self.beacons[nid]
            elif info.reachable and info.last_pong_at is not None \
                    and now - info.last_pong_at > REACHABLE_TTL_S:
                info.reachable = False  # went quiet — stop claiming it's up

    def gossip_blobs(self) -> list:
        return [info.blob for info in self.beacons.values()]

    def snapshot(self, now: float) -> list:
        """Read-only view for the status page / CLI."""
        out = []
        for info in self.beacons.values():
            out.append({
                "id": info.node_id.hex(), "name": info.name,
                "host": info.host, "gossip_port": info.gossip_port, "dht_port": info.dht_port,
                "task": info.task, "n_classes": info.n_classes, "topics": list(info.topics),
                "reachable": info.reachable, "age_s": now - info.seen_at,
                "same_space": info.space_fp == self.own_fp,
            })
        out.sort(key=lambda b: (not b["reachable"], b["age_s"]))
        return out


class FederationService:
    """Bus-wired federation glue for ONE beacon. Verifies advert blobs, replies to
    reachability pings, and drives the periodic re-advert / gossip / probe tick."""

    def __init__(self, identity: Identity, bus, own: dict, *,
                 genesis_id: bytes = None, genesis_addr: tuple = None, on_bridge=None):
        self.identity = identity
        self.bus = bus
        self.own = own                       # host/gossip_port/dht_port/task/n_classes/topics/name/space_fp
        self.genesis_id = genesis_id
        self.genesis_addr = genesis_addr
        self.on_bridge = on_bridge           # called(node_id, addr) to cross-learn with a compatible beacon
        self.registry = BeaconRegistry(self_id=identity.node_id, own_fp=own.get("space_fp", ""))
        self.replay_guard = ReplayGuard()
        self._pending_pings = {}             # nonce -> node_id
        self._bridged = set()                # beacons we've already wired into local gossip
        self._own_blob = None
        if genesis_id is not None and genesis_addr is not None:
            self.bus.peer_addrs.setdefault(genesis_id, genesis_addr)

    # ---- own advert ----

    def build_own_advert(self, now: float) -> bytes:
        """(Re)sign our beacon advert with a fresh nonce + ts so refreshes beat
        stale gossiped copies and never trip the replay guard."""
        body = wire.pack_beacon_advert(
            self.identity.node_id, self.own["host"], self.own["gossip_port"],
            self.own["dht_port"], self.own["task"], self.own["n_classes"],
            self.own["topics"], self.own.get("name", ""), ts=now,
            space_fp=self.own.get("space_fp", ""))
        self._own_blob = self.identity.sign_envelope(body, now)
        return self._own_blob

    # ---- cross-beacon learning ----

    def _maybe_bridge(self, node_id: bytes) -> None:
        """Wire a REACHABLE, SAME-SPACE beacon into local gossip so the two
        swarms learn as one. Guarded: an empty or mismatched fingerprint is never
        bridged (its prototypes would be noise). Idempotent."""
        info = self.registry.beacons.get(node_id)
        if info is None or node_id in self._bridged:
            return
        own_fp = self.own.get("space_fp", "")
        if not own_fp or info.space_fp != own_fp or not info.reachable:
            return
        self._bridged.add(node_id)
        if self.on_bridge is not None:
            self.on_bridge(node_id, info.addr)

    # ---- incoming ----

    def dispatch(self, msg: Message) -> None:
        if msg.kind == "beacon_gossip":
            self._on_gossip(msg)
        elif msg.kind == "beacon_ping":
            self._on_ping(msg)

    def _on_gossip(self, msg: Message) -> None:
        now = time.time()
        for blob in msg.payload["blobs"]:
            res = verify_envelope(blob, self.replay_guard)  # now=None: TTL handles freshness
            if not res.ok:
                continue
            try:
                advert = wire.unpack_beacon_advert(res.body)
            except Exception:
                continue
            if advert["node_id"] != res.sender:
                continue  # advert content disagrees with its signed envelope
            if advert["node_id"] == self.identity.node_id:
                continue
            is_new = self.registry.record(advert, blob, now)
            # Make the beacon dial-able and probe its ADVERTISED address for real
            # reachability (fresh probe on first sighting).
            self.bus.peer_addrs[advert["node_id"]] = (advert["host"], advert["gossip_port"])
            if is_new:
                self.ping(advert["node_id"])

    def _on_ping(self, msg: Message) -> None:
        op = msg.payload["op"]
        if op == "ping":
            self.bus.send(Message(self.identity.node_id, msg.sender, "beacon_ping",
                                  {"op": "pong", "nonce": msg.payload["nonce"]}))
        elif op == "pong":
            nid = self._pending_pings.pop(msg.payload["nonce"], None)
            if nid is not None:
                self.registry.mark_reachable(nid, time.time())
                self._maybe_bridge(nid)   # confirmed reachable => cross-learn if same-space

    # ---- outgoing ----

    def ping(self, node_id: bytes) -> None:
        nonce = os.urandom(8)
        self._pending_pings[nonce] = node_id
        self.bus.send(Message(self.identity.node_id, node_id, "beacon_ping",
                              {"op": "ping", "nonce": nonce}))

    def _send_blobs(self, target: bytes, blobs: list) -> None:
        """Push signed adverts to `target`, SPLIT into datagrams of at most
        GOSSIP_BLOB_GROUP so we never exceed the UDP budget (which the bus would
        silently drop). Splitting, not truncating — the whole set still goes."""
        for i in range(0, len(blobs), GOSSIP_BLOB_GROUP):
            group = blobs[i:i + GOSSIP_BLOB_GROUP]
            self.bus.send(Message(self.identity.node_id, target, "beacon_gossip", {"blobs": group}))

    def announce_to_genesis(self, now: float) -> None:
        """Introduce ourselves to our bootstrap beacon (the genesis by default,
        but any known beacon works): send our signed advert. It verifies and pings
        us back — that pong is the operator's confirmation their beacon is reachable
        — then gossips us to the rest of the mesh, and us the rest."""
        if self.genesis_id is None:
            return
        self.build_own_advert(now)
        self._send_blobs(self.genesis_id, [self._own_blob])

    def tick(self, rng, now: float) -> None:
        """One federation round (driven off the gossip cadence): refresh our own
        advert, push the known set to a fanout of beacons (+ the genesis), re-probe
        reachability, and expire stale entries."""
        self.build_own_advert(now)
        blobs = [self._own_blob] + self.registry.gossip_blobs()
        targets = list(self.registry.beacons.keys())
        if self.genesis_id is not None:
            targets.append(self.genesis_id)
        if targets:
            rng.shuffle(targets)
            for tid in targets[:GOSSIP_FANOUT]:
                self._send_blobs(tid, blobs)
        for nid in list(self.registry.beacons.keys()):
            self.ping(nid)
            self._maybe_bridge(nid)   # re-assert bridges (idempotent) as reachability firms up
        self.registry.expire(now)
