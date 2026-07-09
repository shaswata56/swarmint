"""Ed25519 identity, envelope signing, and the pubkey<->node_id binding check.

Decision #023 D4 / plan review gap #5: node_id is a hash of the pubkey, so a
receiver cannot recover the signing key from an id alone (chicken-and-egg).
The envelope therefore always carries the pubkey, and verify_envelope() checks
sender == H(pubkey) BEFORE any signature check or trust-table lookup — this is
the check that blocks an attacker from replaying a high-trust node's id under
their own keypair and inheriting that trust (or prototype-eviction rights via
PrototypeModel.OVERRIDE_TRUST).
"""

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import nacl.exceptions
import nacl.signing

from . import wire

NONCE_CACHE_SIZE = 10_000  # FIFO cap (decision #023 D10) — bounds memory under churn


def derive_node_id(pubkey: bytes) -> bytes:
    return hashlib.sha1(pubkey).digest()[:20]


def new_nonce() -> bytes:
    return os.urandom(16)


@dataclass
class Identity:
    """A node's keypair + derived id."""
    signing_key: nacl.signing.SigningKey
    pubkey: bytes = field(init=False)
    node_id: bytes = field(init=False)

    def __post_init__(self):
        self.pubkey = bytes(self.signing_key.verify_key)
        self.node_id = derive_node_id(self.pubkey)

    @classmethod
    def generate(cls) -> "Identity":
        return cls(nacl.signing.SigningKey.generate())

    def sign_envelope(self, body: bytes, ts: float, nonce: bytes = None) -> bytes:
        nonce = nonce if nonce is not None else new_nonce()
        signable = wire.signable_bytes(self.pubkey, self.node_id, ts, nonce, body)
        sig = self.signing_key.sign(signable).signature
        return wire.pack_envelope(self.pubkey, self.node_id, ts, nonce, sig, body)


class ReplayGuard:
    """FIFO-capped seen-nonce set. Not keyed per-sender: a nonce is 16 random
    bytes, so cross-sender collision is cryptographically negligible."""

    def __init__(self, capacity: int = NONCE_CACHE_SIZE):
        self.capacity = capacity
        self._seen: "OrderedDict[bytes, None]" = OrderedDict()

    def seen_before(self, nonce: bytes) -> bool:
        return nonce in self._seen

    def record(self, nonce: bytes) -> None:
        if nonce in self._seen:
            return
        self._seen[nonce] = None
        if len(self._seen) > self.capacity:
            self._seen.popitem(last=False)  # evict oldest


@dataclass
class VerifyResult:
    ok: bool
    reason: str = ""
    sender: bytes = b""
    body: bytes = b""


def verify_envelope(data: bytes, replay_guard: ReplayGuard,
                    max_clock_skew: float = 30.0, now: float = None) -> VerifyResult:
    """Order matters: id-binding, then signature, then freshness, then replay.
    Only a fully-verified message's `sender` may ever reach a trust table."""
    try:
        env = wire.unpack_envelope(data)
        pubkey, sender, ts, nonce, sig, body = (
            env["pk"], env["s"], env["ts"], env["n"], env["sig"], env["b"]
        )
    except Exception as e:
        return VerifyResult(False, f"malformed envelope: {e}")

    if derive_node_id(pubkey) != sender:
        return VerifyResult(False, "sender id does not match H(pubkey) (impersonation)")

    signable = wire.signable_bytes(pubkey, sender, ts, nonce, body)
    try:
        nacl.signing.VerifyKey(pubkey).verify(signable, sig)
    except nacl.exceptions.BadSignatureError:
        return VerifyResult(False, "bad signature")

    if now is not None and abs(now - ts) > max_clock_skew:
        return VerifyResult(False, "clock skew exceeds tolerance")

    if replay_guard.seen_before(nonce):
        return VerifyResult(False, "replay: nonce already seen")
    replay_guard.record(nonce)

    return VerifyResult(True, sender=sender, body=body)
