"""PURE tests for the browser-client activity registry (webrtc_answerer.py).

Gated with a lazy, guarded import (not a top-level module import) because
aiortc is an OPTIONAL extra (`pip install -e .[web]`) -- a machine without it
should see this file skip cleanly, not fail the fast suite. The registry
functions themselves (_record_browser_query/recent_browser_clients) don't
touch aiortc at all; only importing the MODULE does (it imports RTCPeerConnection
at top level), so the guard lives here, once, rather than sprinkled through the
production module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _import_or_skip():
    try:
        from swarmint.network import webrtc_answerer
        return webrtc_answerer
    except ImportError as e:
        print(f"skip test_webrtc_answerer (aiortc not installed: {e})")
        return None


def test_recent_browser_clients_records_and_windows():
    wa = _import_or_skip()
    if wa is None:
        return
    wa._recent_browser_clients.clear()

    a, b = b"\x01" * 20, b"\x02" * 20
    wa._record_browser_query(a, 7, 0.92)
    wa._record_browser_query(a, 3, 0.51)   # same client, second query -> n_queries=2, updates "last"
    wa._record_browser_query(b, None, 0.0)  # abstained

    snap = wa.recent_browser_clients(now=__import__("time").time())
    by_id = {c["id"]: c for c in snap}
    assert by_id[a.hex()]["n_queries"] == 2
    assert by_id[a.hex()]["last_label"] == 3      # most recent query wins
    assert by_id[b.hex()]["n_queries"] == 1
    assert by_id[b.hex()]["last_label"] is None   # abstention preserved, not coerced

    # outside the activity window -> excluded
    old = wa.recent_browser_clients(now=__import__("time").time() + 10_000, window_s=300.0)
    assert old == []


def test_recent_browser_clients_bounded_fifo():
    wa = _import_or_skip()
    if wa is None:
        return
    wa._recent_browser_clients.clear()
    for i in range(wa.RECENT_BROWSER_CLIENTS_MAX + 10):
        wa._record_browser_query(i.to_bytes(20, "big"), 0, 0.5)
    assert len(wa._recent_browser_clients) == wa.RECENT_BROWSER_CLIENTS_MAX  # oldest evicted, never unbounded


def test_correction_message_routes_through_corroboration_gate_not_direct_learn():
    """A browser's "correction" message must dispatch to submit_correction_claim
    (quarantine + N-distinct-sender corroboration), never straight to
    node.observe()/model.learn() -- see swarm_node.py's submit_correction_claim
    docstring for why an anonymous per-query keypair can't skip that gate. This
    drives _handle_message end-to-end: sign a real envelope, dispatch it, and
    verify the ack that comes back is itself a validly signed correction_ack
    reporting "not yet corroborated" for a lone claim."""
    wa = _import_or_skip()
    if wa is None:
        return
    import random
    import numpy as np

    from swarmint.network import wire
    from swarmint.network.identity import Identity, ReplayGuard, verify_envelope
    from swarmint.network.simbus import SimBus
    from swarmint.node.swarm_node import SwarmNode

    node_identity = Identity.generate()
    bus = SimBus(rng=random.Random(0))
    node = SwarmNode(node_id=0, topic=1, bus=bus, rng=random.Random(1), n_classes=10)

    answerer = wa.WebRTCAnswerer(node_identity, node)

    browser_identity = Identity.generate()
    x = np.zeros(9, dtype=np.float32)
    qid = b"\x01" * 8
    body = wire.pack_correction_claim(qid, x, label=6)
    envelope = browser_identity.sign_envelope(body, ts=__import__("time").time())

    sent = []

    class FakeChannel:
        def send(self, data):
            sent.append(data)

    answerer._handle_message(FakeChannel(), envelope)

    assert len(sent) == 1, "a correction message must get exactly one signed ack back"
    guard = ReplayGuard()
    result = verify_envelope(sent[0], guard)
    assert result.ok, result.reason
    ack = wire.unpack_correction_ack(result.body)
    assert ack["id"] == qid
    assert ack["promoted"] is False       # a single, uncorroborated claim never commits
    assert ack["corroborated"] is False


if __name__ == "__main__":
    test_recent_browser_clients_records_and_windows()
    test_recent_browser_clients_bounded_fifo()
    test_correction_message_routes_through_corroboration_gate_not_direct_learn()
    print("ok")
