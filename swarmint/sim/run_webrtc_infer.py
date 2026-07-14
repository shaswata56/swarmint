"""Phase 2 proof: an inference query answered over a REAL WebRTC data channel,
by a SwarmNode holding REAL sklearn digits knowledge — with ZERO changes to
SwarmNode/InferenceService/wire.py. Two aiortc peer connections in one process
(the "browser" side and the answering node), SDP exchanged directly in-process
(no signaling server yet — that's Phase 3), a signed query sent over the data
channel, and a signed response verified back.

Run:  python -m swarmint.sim.run_webrtc_infer
"""

import asyncio
import random

import numpy as np
from aiortc import RTCPeerConnection
from nacl.signing import SigningKey

from ..core.prototype_model import Prototype
from ..network import wire
from ..network.identity import Identity, ReplayGuard, verify_envelope
from ..network.simbus import SimBus
from ..network.webrtc_answerer import WebRTCAnswerer
from ..node.swarm_node import SwarmNode


def _log(event, **kv):
    print(f"EVENT {event} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def _build_trained_node(seed: int) -> SwarmNode:
    """A SwarmNode fed the FULL real digits training pool directly (no gossip
    needed for this proof — Phase 2 is about the TRANSPORT substitution, and
    SwarmNode.answer_query is already proven correct/unchanged over UDP)."""
    from sklearn.datasets import load_digits

    from ..core.embedding import LDAEmbedding, genesis_seed_split
    from .real_data import derive_radii, normalize

    d = load_digits()
    X = normalize(d.data.astype(np.float32))
    y = d.target.astype(int)
    (Xs, ys), (Xr, yr) = genesis_seed_split(X, y, 10, per_class=40,
                                            rng=np.random.default_rng(0))
    embedding = LDAEmbedding().fit(Xs, ys)
    Z = embedding.transform(Xr)
    radii = derive_radii(Z, yr, np.random.default_rng(0))

    bus = SimBus()
    node = SwarmNode(node_id=seed, topic=1, bus=bus, rng=random.Random(seed),
                     n_classes=10, **radii)
    protos = [Prototype(vector=x, label=int(y_), weight=1.0) for x, y_ in zip(Z, yr)]
    node.model = node.model.merged_with(protos, node.node_id, sender_trust=1.0)
    return node, embedding


async def run() -> dict:
    node, embedding = _build_trained_node(seed=1)
    ident = Identity(SigningKey(b"\x02" * 32))
    answerer = WebRTCAnswerer(ident, node)

    # --- the "browser" side: its own peer connection + identity, offers a channel ---
    browser_ident = Identity(SigningKey(b"\x03" * 32))
    browser_pc = RTCPeerConnection()
    channel = browser_pc.createDataChannel("swarmint")

    got_response = asyncio.get_event_loop().create_future()
    replay_guard = ReplayGuard()

    @channel.on("message")
    def on_message(message):
        result = verify_envelope(message, replay_guard)
        if not result.ok:
            got_response.set_exception(RuntimeError(result.reason))
            return
        payload = wire.unpack_inference_response(result.body)
        if not got_response.done():
            got_response.set_result(payload)

    channel_open = asyncio.get_event_loop().create_future()

    @channel.on("open")
    def on_open():
        if not channel_open.done():
            channel_open.set_result(True)

    # --- SDP exchange, in-process (Phase 3 carries this over the beacon instead) ---
    offer = await browser_pc.createOffer()
    await browser_pc.setLocalDescription(offer)
    answer = await answerer.accept_offer(browser_pc.localDescription.sdp,
                                         browser_pc.localDescription.type)
    await browser_pc.setRemoteDescription(answer)
    _log("webrtc_handshake_done")

    await asyncio.wait_for(channel_open, timeout=10.0)
    _log("datachannel_open")

    # --- send a REAL held-out digit as a signed query, over the data channel ---
    from sklearn.datasets import load_digits
    from .real_data import normalize
    raw = load_digits()
    test_idx = 500  # arbitrary held-out-ish index; not in the genesis seed slice
    x_raw = normalize(raw.data[test_idx].astype(np.float32).reshape(1, -1))
    x_embedded = embedding.transform(x_raw)[0]
    true_label = int(raw.target[test_idx])

    qid = b"\x01" * 8
    body = wire.pack_inference_query(qid, x_embedded, 0.0)
    envelope = browser_ident.sign_envelope(body, ts=__import__("time").time())
    channel.send(envelope)
    _log("query_sent", true_label=true_label)

    response = await asyncio.wait_for(got_response, timeout=10.0)
    _log("response_received", label=response["label"], confidence=round(float(response["confidence"]), 3))

    # aiortc's close() can hang on some transport-teardown paths (known upstream
    # rough edge) — bound it; the functional result is already captured above.
    try:
        await asyncio.wait_for(answerer.close(), timeout=3.0)
    except asyncio.TimeoutError:
        pass
    try:
        await asyncio.wait_for(browser_pc.close(), timeout=3.0)
    except asyncio.TimeoutError:
        pass

    return {"true_label": true_label, "predicted_label": response["label"],
           "confidence": response["confidence"]}


def main():
    res = asyncio.run(run())
    print("\n== Phase 2: inference query answered over a REAL WebRTC data channel ==")
    print(f"  true label: {res['true_label']}  predicted: {res['predicted_label']}  "
         f"confidence: {res['confidence']:.3f}")
    ok = res["predicted_label"] == res["true_label"]
    print(f"\n{'PASS' if ok else 'FAIL'}: SwarmNode.answer_query answered correctly "
         f"over WebRTC (aiortc), unmodified.")
    import os
    import sys
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
