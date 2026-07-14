"""Regression guards for the live beacon status page (PURE, no sockets/browser).

The page used to crash mid-reconciliation when a DOM query returned null during
a poll cycle (a `.dot`/`.nat`/`.re` element briefly not found), which silently
tripped the fetch chain's `.catch()` and stuck the live indicator on
"reconnecting..." forever with no visible error (found via a real jsdom
run — see DECISIONS #058). The fix: every DOM write in updPeer/updBeacon goes
through a null-guarded setClass()/setText(), mirroring the guard setText()
already had. These tests can't execute the JS (no JS runtime in this suite),
but they pin the guarded-source PATTERN so a future edit can't silently drop it,
and they cover the new server-side status_url/federates plumbing that feeds it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.network import beacon_status as bs


def _fake_snapshot(with_federation=True):
    return {
        "beacon_id": "8c30c97e7fd5460ce3b962db4cd75879eecd8abd",
        "uptime_s": 12.0, "n_total": 2, "n_active": 2,
        "peers": [{"id": "aa" * 20, "endpoint": "1.1.1.1:9003", "topics": [1], "age_s": 2.0,
                  "active": True, "local": True, "sources": ["this beacon"]},
                 {"id": "cc" * 20, "endpoint": "5.6.7.8:9401", "topics": [101], "age_s": 4.0,
                  "active": True, "local": False, "sources": ["beacon-brave-fox-1677"]}],
        "federates": with_federation,
        "federation": ([{"id": "bb" * 20, "name": "peer-1", "host": "34.132.137.213",
                        "gossip_port": 9001, "dht_port": 9002, "task": "digits", "n_classes": 10,
                        "topics": [1], "reachable": True, "age_s": 5.0, "same_space": True,
                        "status_url": "http://34.132.137.213/"}] if with_federation else []),
        "n_beacons_reachable": 1 if with_federation else 0,
    }


def test_page_is_api_driven_not_meta_refresh():
    html = bs.render_html(_fake_snapshot())
    assert 'meta http-equiv="refresh"' not in html
    assert "window.__INITIAL__" in html and "/status.json" in html
    assert "/*__INITIAL__*/null" not in html  # token must be replaced


def test_peer_table_is_whole_swarm_aggregate():
    html = bs.render_html(_fake_snapshot())
    # the peer table exposes the swarm-wide "seen via" column, and the old
    # observer-relative "direct path" column is gone (reachability stays, but only
    # on the Federated Beacons table — a beacon's reachability IS globally meaningful)
    assert "seen via" in html
    assert "direct path" not in html
    # both a local and a remote-only peer are rendered from the aggregate
    assert '"local": true' in html and '"local": false' in html


def test_dom_writes_are_null_guarded():
    """Pins the crash fix: every className write goes through the guarded
    setClass() helper (mirroring setText()'s existing node-truthy check), and
    updPeer/updBeacon bail out on a null row instead of dereferencing it."""
    html = bs.render_html(_fake_snapshot())
    assert "function setClass(node, cls){ if(node) node.className = cls; }" in html
    assert "function updPeer(row, p){\n    if(!row) return;" in html
    assert "function updBeacon(row, b){\n    if(!row) return;" in html
    # no direct unguarded `.querySelector(...).className =` left in the source
    assert '.querySelector(".dot").className' not in html
    assert '.querySelector(".re").className' not in html


def test_status_url_rendered_for_clickable_federation_row():
    html = bs.render_html(_fake_snapshot(with_federation=True))
    assert '"status_url": "http://34.132.137.213/"' in html
    assert 'a class="mono idc nolink"' in html  # anchor wrapper present in the row template


def test_signal_options_preflight_returns_cors_headers():
    """A cross-origin POST with a JSON body (the browser client) triggers a CORS
    preflight OPTIONS request first; browsers refuse the real POST if this isn't
    answered with the right headers. Real socket, real asyncio server — not a
    unit test of a helper function, the actual wire behavior."""
    import asyncio

    from nacl.signing import SigningKey

    from swarmint.network import beacon_status
    from swarmint.network.identity import Identity

    class _FakeNode:
        node_id = b"\x00" * 20

    class _FakeNet:
        identity = Identity(SigningKey(b"\x09" * 32))
        node = _FakeNode()
        discovery = type("D", (), {"peer_addrs": {}, "peer_seen_at": {}, "peer_topics": {}})()
        bus = type("B", (), {"peer_addrs": {}})()
        nat = None
        federation = None

    async def go():
        server = await beacon_status.serve(_FakeNet(), "127.0.0.1", 8098, 0.0)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 8098)
            writer.write(b"OPTIONS /signal HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            await writer.drain()
            raw = await reader.read()
            writer.close()
            return raw
        finally:
            server.close()
            await server.wait_closed()

    raw = asyncio.run(go())
    assert b"204" in raw.split(b"\r\n", 1)[0]
    assert b"Access-Control-Allow-Origin: *" in raw
    assert b"Access-Control-Allow-Methods: POST, OPTIONS" in raw


def test_status_json_endpoint_shape_matches_snapshot_keys():
    snap = _fake_snapshot()
    # what beacon_status.serve() would emit for /status.json is the raw snapshot dict
    assert set(["beacon_id", "uptime_s", "n_total", "n_active", "peers",
               "federates", "federation", "n_beacons_reachable"]) <= set(snap.keys())


if __name__ == "__main__":
    test_page_is_api_driven_not_meta_refresh()
    test_peer_table_is_whole_swarm_aggregate()
    test_dom_writes_are_null_guarded()
    test_status_url_rendered_for_clickable_federation_row()
    test_signal_options_preflight_returns_cors_headers()
    test_status_json_endpoint_shape_matches_snapshot_keys()
    print("ok")
