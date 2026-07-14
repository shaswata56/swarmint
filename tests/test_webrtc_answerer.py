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


if __name__ == "__main__":
    test_recent_browser_clients_records_and_windows()
    test_recent_browser_clients_bounded_fifo()
    print("ok")
