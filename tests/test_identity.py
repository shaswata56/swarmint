"""T3: Ed25519 identity, envelope signing, pubkey<->node_id binding, replay guard.

The impersonation test is the load-bearing one: without the sender==H(pubkey)
check, an attacker could replay a high-trust node's claimed id under their own
keypair and inherit that trust (or PrototypeModel.OVERRIDE_TRUST eviction
rights) — see decision #023 D4 / plan review gap #5.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.prototype_model import Prototype
from swarmint.network import wire
from swarmint.network.identity import (
    Identity, ReplayGuard, derive_node_id, new_nonce, verify_envelope,
)


def _signed_gossip(identity: Identity, n=3, ts=None):
    body = wire.pack_gossip_protos([
        Prototype(vector=np.zeros(4, dtype=np.float32), label=i, weight=1.0) for i in range(n)
    ])
    return identity.sign_envelope(body, ts if ts is not None else time.time())


def test_node_id_derivation_matches_pubkey():
    a = Identity.generate()
    assert a.node_id == derive_node_id(a.pubkey)
    assert len(a.node_id) == 20


def test_valid_envelope_verifies():
    a = Identity.generate()
    env = _signed_gossip(a)
    result = verify_envelope(env, ReplayGuard())
    assert result.ok
    assert result.sender == a.node_id
    assert wire.unpack_gossip_protos(result.body)


def test_bad_signature_rejected():
    a, b = Identity.generate(), Identity.generate()
    body = wire.pack_gossip_protos([Prototype(np.zeros(4, dtype=np.float32), 0, 1.0)])
    ts, nonce = time.time(), new_nonce()
    # sign under b's key but claim a's pubkey/id in the envelope (a forged envelope)
    forged_sig = b.signing_key.sign(
        wire.signable_bytes(a.pubkey, a.node_id, ts, nonce, body)
    ).signature
    env = wire.pack_envelope(a.pubkey, a.node_id, ts, nonce, forged_sig, body)
    result = verify_envelope(env, ReplayGuard())
    assert not result.ok
    assert "signature" in result.reason


def test_impersonation_wrong_pubkey_for_claimed_id_rejected():
    """The core anti-impersonation check: attacker signs correctly with THEIR
    OWN key but claims a victim's node_id as sender. Must be rejected before
    any signature check even runs, since a valid self-consistent signature
    would otherwise pass."""
    victim, attacker = Identity.generate(), Identity.generate()
    body = wire.pack_gossip_protos([Prototype(np.zeros(4, dtype=np.float32), 0, 1.0)])
    ts, nonce = time.time(), new_nonce()
    # attacker signs with their OWN key/pubkey, correctly and self-consistently,
    # but claims the victim's node_id as sender.
    signable = wire.signable_bytes(attacker.pubkey, victim.node_id, ts, nonce, body)
    sig = attacker.signing_key.sign(signable).signature
    env = wire.pack_envelope(attacker.pubkey, victim.node_id, ts, nonce, sig, body)
    result = verify_envelope(env, ReplayGuard())
    assert not result.ok
    assert "impersonation" in result.reason
    assert result.sender != victim.node_id  # victim's id never validated as sender


def test_high_trust_impersonation_gains_no_trust():
    """Integration-flavored: simulate a trust table keyed by VERIFIED sender
    only. An attacker impersonating a trusted peer's id must never get their
    envelope's claimed id inserted into the trust table."""
    victim, attacker = Identity.generate(), Identity.generate()
    trust = {victim.node_id: 0.9}  # victim is highly trusted

    body = wire.pack_gossip_protos([Prototype(np.zeros(4, dtype=np.float32), 0, 1.0)])
    ts, nonce = time.time(), new_nonce()
    signable = wire.signable_bytes(attacker.pubkey, victim.node_id, ts, nonce, body)
    sig = attacker.signing_key.sign(signable).signature
    forged = wire.pack_envelope(attacker.pubkey, victim.node_id, ts, nonce, sig, body)

    result = verify_envelope(forged, ReplayGuard())
    assert not result.ok
    # a correct dispatcher never reaches the trust-table lookup on failure:
    if result.ok:
        trust.setdefault(result.sender, 0.5)
    assert trust[victim.node_id] == 0.9  # unchanged; attacker inherited nothing


def test_replay_rejected_second_time():
    a = Identity.generate()
    env = _signed_gossip(a)
    guard = ReplayGuard()
    first = verify_envelope(env, guard)
    second = verify_envelope(env, guard)
    assert first.ok and not second.ok
    assert "replay" in second.reason


def test_clock_skew_rejected():
    a = Identity.generate()
    stale = _signed_gossip(a, ts=1000.0)
    result = verify_envelope(stale, ReplayGuard(), max_clock_skew=30.0, now=2000.0)
    assert not result.ok
    assert "skew" in result.reason


def test_replay_guard_fifo_cap_bounds_memory():
    guard = ReplayGuard(capacity=5)
    nonces = [new_nonce() for _ in range(10)]
    for n in nonces:
        guard.record(n)
    assert len(guard._seen) == 5
    # oldest 5 evicted; a repeat of an evicted nonce is (incorrectly but
    # boundedly) treated as fresh — documented tradeoff, not a bug
    assert not guard.seen_before(nonces[0])
    assert guard.seen_before(nonces[-1])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
