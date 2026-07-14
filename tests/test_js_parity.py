"""Python<->JS wire/crypto/embedding PARITY (opt-in, real `node` execution).

The riskiest unknown in browser P2P inference isn't the network, it's whether the
JS client's signed envelopes and embedded vectors are byte-identical to what
Python produces for the same inputs — a mismatch means every JS-signed query is
silently rejected by verify_envelope() or mapped into the wrong embedded space.
This test drives the ACTUAL JS client (js/swarmint-client) via a real `node`
subprocess (not a JS-in-Python re-implementation guess) and compares against
Python's own primitives.

Gated by SWARM_JS_TESTS=1 (needs `node` + the npm deps installed once via
`npm install` in js/swarmint-client) so it doesn't fail the fast suite on a box
without Node.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
JS_DIR = ROOT / "js" / "swarmint-client"


def _run_js(job: dict) -> dict:
    with tempfile.TemporaryDirectory() as td:
        job_path = Path(td) / "job.json"
        out_path = Path(td) / "out.json"
        job_path.write_text(json.dumps(job))
        subprocess.run(["node", str(JS_DIR / "parity_check.mjs"), str(job_path), str(out_path)],
                       cwd=JS_DIR, check=True, capture_output=True, text=True)
        return json.loads(out_path.read_text())


def test_js_sign_verifies_in_python():
    if os.environ.get("SWARM_JS_TESTS") != "1":
        print("skip test_js_sign_verifies_in_python (set SWARM_JS_TESTS=1, needs `node` + npm install)")
        return
    from swarmint.network.identity import ReplayGuard, derive_node_id, verify_envelope
    from swarmint.network.wire import pack_gossip_protos

    priv = os.urandom(32)
    body = pack_gossip_protos([])  # any real body encoder; content doesn't matter here
    nonce = os.urandom(16)
    ts = time.time()  # real timestamp -> naturally non-integral, avoids the msgpack int/float edge case

    result = _run_js({"sign": {"privateKeyHex": priv.hex(), "bodyHex": body.hex(),
                               "nonceHex": nonce.hex(), "ts": ts}})["sign"]
    envelope = bytes.fromhex(result["envelopeHex"])

    # node_id derivation must match: sha1(pubkey)[:20]
    pubkey = bytes.fromhex(result["pubkeyHex"])
    assert derive_node_id(pubkey) == bytes.fromhex(result["nodeIdHex"])

    # the JS-signed envelope must verify under PYTHON's verify_envelope, unmodified
    res = verify_envelope(envelope, ReplayGuard())
    assert res.ok, res.reason
    assert res.sender == bytes.fromhex(result["nodeIdHex"])
    assert res.body == body


def test_python_sign_verifies_in_js():
    if os.environ.get("SWARM_JS_TESTS") != "1":
        print("skip test_python_sign_verifies_in_js (set SWARM_JS_TESTS=1)")
        return
    from nacl.signing import SigningKey

    from swarmint.network.identity import Identity
    from swarmint.network.wire import pack_gossip_protos

    sk = SigningKey.generate()
    ident = Identity(sk)
    body = pack_gossip_protos([])
    envelope = ident.sign_envelope(body, time.time())

    result = _run_js({"verify": {"envelopeHex": envelope.hex()}})["verify"]
    assert result["ok"], result["reason"]
    assert bytes.fromhex(result["senderHex"]) == ident.node_id


def test_js_inference_query_unpacks_in_python():
    if os.environ.get("SWARM_JS_TESTS") != "1":
        print("skip test_js_inference_query_unpacks_in_python (set SWARM_JS_TESTS=1)")
        return
    import numpy as np

    from swarmint.network import wire

    x = np.array([0.1, -0.2, 0.3, 0.4, -0.5], dtype=np.float32)
    qid = os.urandom(8)
    result = _run_js({"query": {"xFloat32Hex": x.tobytes().hex(), "qidHex": qid.hex(),
                                "confidenceThreshold": 0.34}})["query"]
    packed = bytes.fromhex(result["packedHex"])
    decoded = wire.unpack_inference_query(packed)
    assert decoded["id"] == qid
    assert np.allclose(decoded["x"], x)
    assert decoded["confidence_threshold"] == 0.34


def test_python_inference_response_unpacks_in_js():
    if os.environ.get("SWARM_JS_TESTS") != "1":
        print("skip test_python_inference_response_unpacks_in_js (set SWARM_JS_TESTS=1)")
        return
    from swarmint.network import wire

    qid = os.urandom(8)
    packed = wire.pack_inference_response(qid, 7, 0.912)
    result = _run_js({"unpackResponse": {"dataHex": packed.hex()}})["unpackResponse"]
    assert bytes.fromhex(result["idHex"]) == qid
    assert result["label"] == 7
    assert abs(result["confidence"] - 0.912) < 1e-6


def test_js_verify_then_unpack_response_uses_inner_body_not_envelope():
    """Regression test for a real bug found in infer.html: the browser passed
    the raw signed ENVELOPE to unpackInferenceResponse instead of the verified
    INNER body (res.body). The envelope has no "k" discriminator (only the
    inner body does) -> "expected 'response', got 'undefined'" at runtime,
    which reads as "no response from node" even though a valid response
    arrived. Drives the exact channel.onmessage pattern end-to-end."""
    if os.environ.get("SWARM_JS_TESTS") != "1":
        print("skip test_js_verify_then_unpack_response_uses_inner_body_not_envelope (set SWARM_JS_TESTS=1)")
        return
    from nacl.signing import SigningKey

    from swarmint.network.identity import Identity
    from swarmint.network.wire import pack_inference_response

    sk = SigningKey.generate()
    ident = Identity(sk)
    qid = os.urandom(8)
    body = pack_inference_response(qid, 4, 0.87)
    envelope = ident.sign_envelope(body, time.time())

    result = _run_js({"verifyThenUnpackResponse": {"envelopeHex": envelope.hex()}})["verifyThenUnpackResponse"]
    assert result["ok"], result.get("reason")
    assert bytes.fromhex(result["idHex"]) == qid
    assert result["label"] == 4
    assert abs(result["confidence"] - 0.87) < 1e-6


def test_js_correction_claim_unpacks_in_python():
    if os.environ.get("SWARM_JS_TESTS") != "1":
        print("skip test_js_correction_claim_unpacks_in_python (set SWARM_JS_TESTS=1)")
        return
    import numpy as np

    from swarmint.network import wire

    x = np.array([0.1, -0.2, 0.3, 0.4, -0.5], dtype=np.float32)
    qid = os.urandom(8)
    result = _run_js({"correction": {"xFloat32Hex": x.tobytes().hex(), "qidHex": qid.hex(),
                                     "label": 6}})["correction"]
    packed = bytes.fromhex(result["packedHex"])
    decoded = wire.unpack_correction_claim(packed)
    assert decoded["id"] == qid
    assert np.allclose(decoded["x"], x)
    assert decoded["label"] == 6


def test_python_correction_ack_unpacks_in_js():
    if os.environ.get("SWARM_JS_TESTS") != "1":
        print("skip test_python_correction_ack_unpacks_in_js (set SWARM_JS_TESTS=1)")
        return
    from swarmint.network import wire

    qid = os.urandom(8)
    packed = wire.pack_correction_ack(qid, promoted=True, corroborated=True)
    result = _run_js({"unpackCorrectionAck": {"dataHex": packed.hex()}})["unpackCorrectionAck"]
    assert bytes.fromhex(result["idHex"]) == qid
    assert result["promoted"] is True
    assert result["corroborated"] is True


def test_js_embedding_matches_python_for_a_real_digit():
    if os.environ.get("SWARM_JS_TESTS") != "1":
        print("skip test_js_embedding_matches_python_for_a_real_digit (set SWARM_JS_TESTS=1)")
        return
    import numpy as np

    from swarmint.sim.live_task import build_task
    from swarmint.sim.real_data import normalize

    task = build_task("digits", n_classes=10, world_seed=0)

    from sklearn.datasets import load_digits
    raw_arr = load_digits().data[0].astype(np.float32)
    raw = raw_arr.astype(float).tolist()   # what we send the JS side, raw 64 pixels

    # the SAME pipeline infer_query.py's --image path uses: normalize -> embedding.transform
    py_embedded = task.embedding.transform(normalize(raw_arr.reshape(1, -1)))[0]

    genesis_json = json.loads((JS_DIR / "genesis_embedding.json").read_text())
    result = _run_js({"embed": {"embeddingJson": genesis_json, "rawPixels": raw}})["embed"]
    js_embedded = np.asarray(result["z"], dtype=np.float32)

    assert js_embedded.shape == py_embedded.shape
    assert np.allclose(js_embedded, py_embedded, atol=1e-4), \
        f"embedding mismatch: py={py_embedded} js={js_embedded}"


if __name__ == "__main__":
    test_js_sign_verifies_in_python()
    test_python_sign_verifies_in_js()
    test_js_inference_query_unpacks_in_python()
    test_python_inference_response_unpacks_in_js()
    test_js_verify_then_unpack_response_uses_inner_body_not_envelope()
    test_js_embedding_matches_python_for_a_real_digit()
    print("ok")
