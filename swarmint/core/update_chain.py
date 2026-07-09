"""Per-node tamper-evident hash-chain of model checkpoints (deferred item D1).

The lightweight alternative to the full blockchain the original spec called for
(decision #023 D2/D11 fallback: "for tamper evidence, add a per-node hash chain
like Git; that's trivial and needs no consensus"). Each node keeps an
append-only chain of its OWN model-version checkpoints; every link commits to
the previous link's hash and is Ed25519-signed, so any third party can verify:

  * the chain is contiguous (each entry's prev_hash == hash of the prior entry),
  * every entry is signed by the claimed node, and
  * the node_id binds to the signing key (H(pubkey) == node_id),

which makes it impossible to silently rewrite, reorder, or forge a node's update
history after the fact. It does NOT provide global consensus (a node could keep
two divergent chains and show different peers different ones) — detecting THAT
is what gossip + reputation already handle; this just makes each node's own
claimed history auditable and non-repudiable.

This is OPT-IN: nothing in the learning path calls it, so SwarmNode behaviour
(and the deterministic simulator's 0.959) is unchanged. A networked node
(NetNode) checkpoints into its chain once per round when its model version
advances.
"""

import hashlib
from dataclasses import dataclass

import msgpack

GENESIS_HASH = b"\x00" * 32


def _entry_digest(prev_hash: bytes, node_id: bytes, counter: int, ctr_node,
                  model_digest: bytes, seq: int) -> bytes:
    """Content hash of an entry (everything except its own signature). ctr_node
    is the Version.node_id (the node that last bumped the vector clock), kept
    distinct from the chain owner's node_id."""
    blob = msgpack.packb({
        "p": prev_hash, "id": node_id, "c": counter, "cn": ctr_node,
        "m": model_digest, "s": seq,
    }, use_bin_type=True)
    return hashlib.sha256(blob).digest()


def model_digest(protos) -> bytes:
    """Order-independent digest of a prototype set — commits to WHAT the model
    knew at a checkpoint without shipping the weights. Order-independent because
    the prototype set is a set (merge is order-invariant), so two nodes that
    converged to the same knowledge get the same digest."""
    h = hashlib.sha256()
    parts = sorted(
        hashlib.sha256(msgpack.packb(
            {"l": int(p.label), "v": p.vector.tobytes()}, use_bin_type=True
        )).digest()
        for p in protos
    )
    for part in parts:
        h.update(part)
    return h.digest()


@dataclass
class ChainEntry:
    seq: int
    prev_hash: bytes
    node_id: bytes            # chain owner (== H(pubkey))
    counter: int             # Version.counter at this checkpoint
    ctr_node: bytes          # Version.node_id at this checkpoint
    model_digest: bytes
    pubkey: bytes
    signature: bytes
    entry_hash: bytes        # = _entry_digest(...) — the value the NEXT entry chains to

    def to_wire(self) -> dict:
        return {"seq": self.seq, "prev": self.prev_hash, "id": self.node_id,
                "c": self.counter, "cn": self.ctr_node, "m": self.model_digest,
                "pk": self.pubkey, "sig": self.signature, "h": self.entry_hash}

    @classmethod
    def from_wire(cls, d: dict) -> "ChainEntry":
        return cls(seq=d["seq"], prev_hash=d["prev"], node_id=d["id"], counter=d["c"],
                   ctr_node=d["cn"], model_digest=d["m"], pubkey=d["pk"],
                   signature=d["sig"], entry_hash=d["h"])


class UpdateChain:
    """Append-only, signed checkpoint chain for one node's model."""

    def __init__(self, identity):
        self.identity = identity
        self.entries: list[ChainEntry] = []

    @property
    def head_hash(self) -> bytes:
        return self.entries[-1].entry_hash if self.entries else GENESIS_HASH

    def checkpoint(self, version, protos) -> ChainEntry:
        """Append a signed checkpoint of the current model version + digest."""
        seq = len(self.entries)
        prev = self.head_hash
        md = model_digest(protos)
        ctr_node = version.node_id if isinstance(version.node_id, bytes) else \
            str(version.node_id).encode()
        digest = _entry_digest(prev, self.identity.node_id, version.counter, ctr_node, md, seq)
        sig = self.identity.signing_key.sign(digest).signature
        entry = ChainEntry(seq=seq, prev_hash=prev, node_id=self.identity.node_id,
                           counter=version.counter, ctr_node=ctr_node, model_digest=md,
                           pubkey=self.identity.pubkey, signature=sig, entry_hash=digest)
        self.entries.append(entry)
        return entry

    def to_wire(self) -> list:
        return [e.to_wire() for e in self.entries]


def verify_chain(entries: list, expected_node_id: bytes = None):
    """Verify a chain (list of ChainEntry or wire dicts). Returns (ok, reason).

    Checks, in order: id<->pubkey binding, signature on each entry's content
    hash, contiguity (prev_hash links + monotonic seq), and — if given —
    that the whole chain belongs to expected_node_id."""
    import nacl.exceptions
    import nacl.signing

    from .. network.identity import derive_node_id  # local import avoids cycle

    parsed = [e if isinstance(e, ChainEntry) else ChainEntry.from_wire(e) for e in entries]
    prev = GENESIS_HASH
    for i, e in enumerate(parsed):
        if e.seq != i:
            return False, f"seq gap at {i}: got {e.seq}"
        if derive_node_id(e.pubkey) != e.node_id:
            return False, f"entry {i}: node_id != H(pubkey)"
        if expected_node_id is not None and e.node_id != expected_node_id:
            return False, f"entry {i}: belongs to a different node"
        if e.prev_hash != prev:
            return False, f"entry {i}: broken link (prev_hash mismatch)"
        recomputed = _entry_digest(e.prev_hash, e.node_id, e.counter, e.ctr_node,
                                   e.model_digest, e.seq)
        if recomputed != e.entry_hash:
            return False, f"entry {i}: content hash does not match (tampered)"
        try:
            nacl.signing.VerifyKey(e.pubkey).verify(e.entry_hash, e.signature)
        except nacl.exceptions.BadSignatureError:
            return False, f"entry {i}: bad signature"
        prev = e.entry_hash
    return True, "ok"
