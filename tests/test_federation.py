"""Beacon-federation PURE logic (no sockets): the signed-advert wire round-trip
and the BeaconRegistry soft-state machine (record / newest-wins / reachability /
TTL expiry / self-exclusion). The bus-wired FederationService reachability path is
exercised end-to-end by the live two-beacon run, not here."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nacl.signing import SigningKey

from swarmint.network import wire
from swarmint.network.federation import (BEACON_TTL_S, REACHABLE_TTL_S,
                                         BeaconRegistry, space_fingerprint)
from swarmint.network.identity import Identity, ReplayGuard, verify_envelope


def _ident(seed: int) -> Identity:
    return Identity(SigningKey(seed.to_bytes(4, "big").rjust(32, b"\0")))


def _advert(idn, host="1.2.3.4", gp=9001, task="digits", ts=100.0):
    body = wire.pack_beacon_advert(idn.node_id, host, gp, gp + 1, task, 10, {1}, "beacon-x", ts)
    blob = idn.sign_envelope(body, ts)
    return wire.unpack_beacon_advert(body), blob


def test_beacon_advert_wire_roundtrip():
    idn = _ident(1000)
    body = wire.pack_beacon_advert(idn.node_id, "34.41.60.102", 9001, 9002,
                                   "digits", 10, {1, 2}, "genesis", 123.5)
    d = wire.unpack_beacon_advert(body)
    assert d["node_id"] == idn.node_id
    assert d["host"] == "34.41.60.102" and d["gossip_port"] == 9001 and d["dht_port"] == 9002
    assert d["task"] == "digits" and d["n_classes"] == 10 and d["topics"] == [1, 2]
    assert d["name"] == "genesis" and d["ts"] == 123.5
    # signed blob verifies + sender matches the advert's own node_id
    blob = idn.sign_envelope(body, 123.5)
    res = verify_envelope(blob, ReplayGuard())
    assert res.ok and res.sender == idn.node_id
    assert wire.unpack_beacon_advert(res.body)["node_id"] == idn.node_id


def test_gossip_and_ping_wire_roundtrip():
    g = wire.pack_beacon_gossip({"blobs": [b"aa", b"bb"]})
    assert wire.unpack_beacon_gossip(g)["blobs"] == [b"aa", b"bb"]
    p = wire.pack_beacon_ping({"op": "ping", "nonce": b"n0"})
    assert wire.unpack_beacon_ping(p) == {"op": "ping", "nonce": b"n0"}


def test_registry_record_newest_wins_and_self_exclusion():
    me = _ident(1)
    other = _ident(2)
    reg = BeaconRegistry(self_id=me.node_id)

    # never list ourselves
    self_adv, self_blob = _advert(me)
    assert reg.record(self_adv, self_blob, now=100.0) is False
    assert len(reg.beacons) == 0

    adv1, blob1 = _advert(other, host="1.1.1.1", ts=100.0)
    assert reg.record(adv1, blob1, now=100.0) is True          # new
    assert reg.beacons[other.node_id].host == "1.1.1.1"

    adv2, blob2 = _advert(other, host="2.2.2.2", ts=200.0)
    assert reg.record(adv2, blob2, now=200.0) is False         # refresh, not new
    assert reg.beacons[other.node_id].host == "2.2.2.2"        # newer advert applied

    adv_stale, blob_stale = _advert(other, host="9.9.9.9", ts=150.0)
    assert reg.record(adv_stale, blob_stale, now=250.0) is False
    assert reg.beacons[other.node_id].host == "2.2.2.2"        # stale advert did NOT regress


def test_registry_reachability_and_expiry():
    me, other = _ident(1), _ident(2)
    reg = BeaconRegistry(self_id=me.node_id)
    adv, blob = _advert(other, ts=100.0)
    reg.record(adv, blob, now=100.0)
    assert reg.beacons[other.node_id].reachable is False

    reg.mark_reachable(other.node_id, now=100.0)
    assert reg.beacons[other.node_id].reachable is True
    # a refreshed advert preserves prior reachability
    adv2, blob2 = _advert(other, ts=110.0)
    reg.record(adv2, blob2, now=110.0)
    assert reg.beacons[other.node_id].reachable is True

    # reachable flips off after the pong goes quiet, entry still present
    reg.expire(now=110.0 + REACHABLE_TTL_S + 1)
    assert reg.beacons[other.node_id].reachable is False
    # entry dropped entirely once the advert itself goes stale
    reg.expire(now=110.0 + BEACON_TTL_S + 1)
    assert other.node_id not in reg.beacons


def test_beacon_registry_status_url():
    # explicit url override always wins (the genesis's domain+TLS case)
    assert BeaconRegistry.status_url("1.2.3.4", 8080, "https://beacon.swarmint.org/") \
        == "https://beacon.swarmint.org/"
    # no http_port -> no status page -> no link
    assert BeaconRegistry.status_url("1.2.3.4", 0, "") == ""
    # bare IP + port 80 -> clean http URL with no port suffix
    assert BeaconRegistry.status_url("34.132.137.213", 80, "") == "http://34.132.137.213/"
    # bare IP + non-80 port -> port included; NEVER guesses https for an IP
    assert BeaconRegistry.status_url("34.132.137.213", 8080, "") == "http://34.132.137.213:8080/"


def test_registry_snapshot_orders_reachable_first():
    me = _ident(1)
    reg = BeaconRegistry(self_id=me.node_id)
    a, ba = _advert(_ident(2), ts=100.0)
    b, bb = _advert(_ident(3), ts=100.0)
    reg.record(a, ba, now=100.0)
    reg.record(b, bb, now=101.0)
    reg.mark_reachable(b["node_id"], now=101.0)
    snap = reg.snapshot(now=102.0)
    assert len(snap) == 2 and snap[0]["reachable"] is True   # reachable sorts first


def test_space_fingerprint_matches_only_on_same_space():
    # raw (synthetic) space: pinned by (task, n_classes, dim, world_seed)
    a = space_fingerprint("synthetic", 6, dim=16, world_seed=0)
    b = space_fingerprint("synthetic", 6, dim=16, world_seed=0)
    assert a == b                                             # same params => same fp (cross-learn)
    assert a != space_fingerprint("synthetic", 6, dim=16, world_seed=1)   # diff seed
    assert a != space_fingerprint("synthetic", 10, dim=16, world_seed=0)  # diff n_classes
    assert a != space_fingerprint("digits", 6, dim=16, world_seed=0)      # diff task

    # embedded space: pinned by the frozen projection's bytes
    class _Emb:
        def __init__(self, blob):
            self._b = blob
        def to_bytes(self):
            return self._b
    e1 = space_fingerprint("digits", 10, embedding=_Emb(b"AAAA"))
    e2 = space_fingerprint("digits", 10, embedding=_Emb(b"AAAA"))
    e3 = space_fingerprint("digits", 10, embedding=_Emb(b"BBBB"))
    assert e1 == e2 and e1 != e3                              # different genesis => no cross-learn


def test_registry_snapshot_flags_same_space():
    own = "fp-own"
    reg = BeaconRegistry(self_id=_ident(1).node_id, own_fp=own)
    same = _ident(2)
    diff = _ident(3)
    body_s = wire.pack_beacon_advert(same.node_id, "1.1.1.1", 9001, 9002, "synthetic", 6,
                                     {1}, "s", 100.0, space_fp=own)
    body_d = wire.pack_beacon_advert(diff.node_id, "2.2.2.2", 9001, 9002, "synthetic", 6,
                                     {1}, "d", 100.0, space_fp="fp-other")
    reg.record(wire.unpack_beacon_advert(body_s), b"", now=100.0)
    reg.record(wire.unpack_beacon_advert(body_d), b"", now=100.0)
    by_id = {b["id"]: b for b in reg.snapshot(now=100.0)}
    assert by_id[same.node_id.hex()]["same_space"] is True
    assert by_id[diff.node_id.hex()]["same_space"] is False


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
    print("ok")
