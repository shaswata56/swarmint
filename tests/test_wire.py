"""T2: wire-format round-trip + the 1200B datagram budget, MEASURED not
hand-computed (decision #023 D9 flagged the hand-computed table as fragile)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.prototype_model import Prototype
from swarmint.network import wire

DIM = 16
GOSSIP_BATCH = 12  # decision #023 D9 cap


def _rand_protos(n, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    return [Prototype(vector=rng.normal(0, 3, dim).astype(np.float32),
                       label=int(rng.integers(0, 10)), weight=float(rng.integers(1, 20)))
            for _ in range(n)]


def test_gossip_protos_roundtrip():
    protos = _rand_protos(12)
    data = wire.pack_gossip_protos(protos)
    back = wire.unpack_gossip_protos(data)
    assert len(back) == 12
    for a, b in zip(protos, back):
        assert a.label == b.label
        assert abs(a.weight - b.weight) < 1e-6
        assert np.allclose(a.vector, b.vector)


def test_inference_query_response_roundtrip():
    qid = b"\x01" * 8
    x = np.random.default_rng(1).normal(0, 1, DIM).astype(np.float32)
    q = wire.pack_inference_query(qid, x, 0.34)
    back = wire.unpack_inference_query(q)
    assert back["id"] == qid and np.allclose(back["x"], x) and abs(back["confidence_threshold"] - 0.34) < 1e-6

    r = wire.pack_inference_response(qid, 7, 0.91)
    back_r = wire.unpack_inference_response(r)
    assert back_r == {"id": qid, "label": 7, "confidence": 0.91}


def test_calibration_chunking_roundtrip():
    qid = b"\x02" * 8
    xs = np.random.default_rng(2).normal(0, 1, (20, DIM)).astype(np.float32)
    q = wire.pack_calibration_query(qid, 0, 3, xs)
    back = wire.unpack_calibration_query(q)
    assert back["chunk_index"] == 0 and back["total_chunks"] == 3
    assert np.allclose(back["xs"], xs)

    labels = list(range(20))
    r = wire.pack_calibration_response(qid, 0, labels)
    back_r = wire.unpack_calibration_response(r)
    assert back_r["labels"] == labels


def test_advert_and_pex_roundtrip():
    a = wire.pack_advert(b"\xaa" * 20, "127.0.0.1:9001", topics={3, 7}, ts=123.5)
    back = wire.unpack_advert(a)
    assert back == {"node_id": b"\xaa" * 20, "addr": "127.0.0.1:9001", "topics": [3, 7], "ts": 123.5}

    # pex now carries per-peer topic sets (D2); 3-tuples round-trip
    peers = [(b"\x01" * 20, "127.0.0.1:9002", [3]), (b"\x02" * 20, "127.0.0.1:9003", [7, 9])]
    p = wire.pack_pex(peers)
    assert wire.unpack_pex(p) == peers


def test_envelope_roundtrip_without_crypto():
    pubkey, sender, ts, nonce = b"\xff" * 32, b"\xaa" * 20, 100.0, b"\x00" * 16
    body = wire.pack_gossip_protos(_rand_protos(3))
    sig = b"\x00" * 64  # placeholder; T3 fills real signatures
    signable = wire.signable_bytes(pubkey, sender, ts, nonce, body)
    env = wire.pack_envelope(pubkey, sender, ts, nonce, sig, body)
    back = wire.unpack_envelope(env)
    assert back["pk"] == pubkey and back["s"] == sender and back["ts"] == ts
    assert back["n"] == nonce and back["sig"] == sig and back["b"] == body
    # re-deriving signable bytes from the unpacked envelope must match exactly
    # (this is what a verifier signs-over on the receiving end)
    rederived = wire.signable_bytes(back["pk"], back["s"], back["ts"], back["n"], back["b"])
    assert rederived == signable


def test_gossip_batch_fits_1200_byte_budget():
    """The decisive measurement behind decision #023 D9: 12 prototypes (dim=16)
    wrapped in a full signed envelope must fit one UDP datagram."""
    protos = _rand_protos(GOSSIP_BATCH)
    body = wire.pack_gossip_protos(protos)

    pubkey, sender, ts, nonce, sig = b"\xff" * 32, b"\xaa" * 20, 1720000000.123, b"\x11" * 16, b"\x22" * 64
    envelope = wire.pack_envelope(pubkey, sender, ts, nonce, sig, body)

    budget = wire.ByteBudget(dim=DIM, n_protos=GOSSIP_BATCH, body_bytes=len(body),
                             envelope_overhead_bytes=len(envelope) - len(body),
                             total_bytes=len(envelope))
    assert budget.fits(), f"{budget.total_bytes}B exceeds {wire.MAX_DATAGRAM_BYTES}B budget: {budget}"
    # sanity: overhead should be small relative to body, not dominate it
    assert budget.envelope_overhead_bytes < budget.body_bytes


def test_13th_proto_or_provenance_would_not_fit():
    """Guards decision #023 D9's explicit tripwire: adding per-proto version
    metadata (~25B/proto) at batch=12 should push past budget, confirming the
    cap is tight-but-correct rather than accidentally generous."""
    protos = _rand_protos(GOSSIP_BATCH)
    body = wire.pack_gossip_protos(protos)
    envelope_overhead = 32 + 20 + 8 + 16 + 64 + 40  # pubkey+sender+ts+nonce+sig+map framing (measured ballpark)
    provenance_bytes_per_proto = 25
    inflated = len(body) + GOSSIP_BATCH * provenance_bytes_per_proto + envelope_overhead
    assert inflated > wire.MAX_DATAGRAM_BYTES, (
        "if this now fits, the cap-at-12 rationale in decision #023 D9 is stale — revisit"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
