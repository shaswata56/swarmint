"""T5: kademlia-backed rendezvous + peer-exchange.

Confirms the actual fix for the structural bug caught in plan review: since
kademlia's Server.set/get is single-value LWW, discovery must key adverts by
each node's OWN id (never a shared topic key), and grow the peer set via PEX.
"""

import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.network import wire
from swarmint.network.discovery import STALE_AFTER_S, Discovery
from swarmint.network.identity import Identity


async def _start(discovery: Discovery, bootstrap=None, publish=True):
    await discovery.server.listen(0, interface="127.0.0.1")
    dht_port = discovery.server.transport.get_extra_info("sockname")[1]
    await discovery.server.bootstrap(bootstrap or [])
    discovery._my_addr = ("127.0.0.1", 9000 + dht_port % 1000)  # fake gossip addr for the test
    if publish:
        # NOTE: kademlia's set() needs >=1 known DHT neighbor or it silently
        # no-ops (Server.set_digest logs a warning and returns False without
        # storing). The very first node in a network has none yet — this is
        # why the real Discovery._refresh_loop retries every 30s (self-healing
        # in production); in tests we publish explicitly after bootstrapping.
        await discovery._publish_advert()
    return dht_port


async def test_two_node_advert_and_lookup_via_dht():
    """Two separate node IDs publish under DISTINCT keys (the actual fix for
    the topic-key collision bug); B must resolve A's real gossip address."""
    id_a, id_b = Identity.generate(), Identity.generate()
    disc_a, disc_b = Discovery(identity=id_a), Discovery(identity=id_b)

    port_a = await _start(disc_a, publish=False)  # lone node: no neighbors yet
    port_b = await _start(disc_b, bootstrap=[("127.0.0.1", port_a)])
    await asyncio.sleep(0.2)  # let the bootstrap handshake settle
    await disc_a._publish_advert()  # now has disc_b as a neighbor; publish succeeds
    await asyncio.sleep(0.1)

    addr = await disc_b.lookup(id_a.node_id)

    disc_a.server.stop()
    disc_b.server.stop()

    assert addr == disc_a._my_addr
    assert id_a.node_id in disc_b.peer_addrs
    assert disc_b.peer_addrs[id_a.node_id] == disc_a._my_addr


async def test_both_directions_discoverable_distinct_keys():
    """The direct regression test for the topic-key bug: BOTH nodes must be
    independently discoverable, proving neither advert clobbered the other."""
    id_a, id_b = Identity.generate(), Identity.generate()
    disc_a, disc_b = Discovery(identity=id_a), Discovery(identity=id_b)
    port_a = await _start(disc_a, publish=False)
    port_b = await _start(disc_b, bootstrap=[("127.0.0.1", port_a)])
    await asyncio.sleep(0.2)
    await disc_a._publish_advert()  # publish now that disc_b is a neighbor
    await asyncio.sleep(0.1)

    addr_a_from_b = await disc_b.lookup(id_a.node_id)
    addr_b_from_a = await disc_a.lookup(id_b.node_id)

    disc_a.server.stop()
    disc_b.server.stop()

    assert addr_a_from_b == disc_a._my_addr
    assert addr_b_from_a == disc_b._my_addr
    assert addr_a_from_b != addr_b_from_a  # distinct — not one clobbering the other


async def test_forged_advert_rejected():
    victim, attacker = Identity.generate(), Identity.generate()
    disc = Discovery(identity=Identity.generate())

    body = wire.pack_advert(victim.node_id, "10.0.0.1:9999", topics={1}, ts=time.time())
    ts, nonce = time.time(), b"\x00" * 16
    signable = wire.signable_bytes(attacker.pubkey, victim.node_id, ts, nonce, body)
    sig = attacker.signing_key.sign(signable).signature
    forged = wire.pack_envelope(attacker.pubkey, victim.node_id, ts, nonce, sig, body)

    result = disc._ingest_advert(forged)
    assert result is None
    assert victim.node_id not in disc.peer_addrs  # no eclipse: forged entry never lands


def test_pex_growth_and_dedup():
    me = Identity.generate()
    disc = Discovery(identity=me)
    peer_a, peer_b = Identity.generate().node_id, Identity.generate().node_id
    disc.ingest_pex([(peer_a, "127.0.0.1:1000"), (peer_b, "127.0.0.1:1001"), (me.node_id, "1.2.3.4:1")])
    assert set(disc.known_peers()) == {peer_a, peer_b}
    assert me.node_id not in disc.known_peers()  # never learn "self" as a peer

    payload = disc.pex_payload()  # now (node_id, addr, topics) 3-tuples (D2)
    addrs = {(nid, addr) for nid, addr, _topics in payload}
    assert (peer_a, "127.0.0.1:1000") in addrs and (peer_b, "127.0.0.1:1001") in addrs

    # a stale re-advertisement of an already-known peer does not overwrite silently
    disc.ingest_pex([(peer_a, "127.0.0.1:9999")])
    assert disc.peer_addrs[peer_a] == ("127.0.0.1", 1000)


def test_pex_propagates_topics():
    me = Identity.generate()
    disc = Discovery(identity=me)
    peer_a, peer_b = Identity.generate().node_id, Identity.generate().node_id
    disc.ingest_pex([(peer_a, "127.0.0.1:1000", [3, 4]), (peer_b, "127.0.0.1:1001", [4, 5])])
    assert disc.peer_topics[peer_a] == {3, 4}
    assert set(disc.peers_for_topic(3)) == {peer_a}
    assert set(disc.peers_for_topic(4)) == {peer_a, peer_b}
    # topic hints update even for an already-known peer
    disc.ingest_pex([(peer_a, "127.0.0.1:1000", [9])])
    assert disc.peer_topics[peer_a] == {3, 4, 9}


def test_stale_peer_expiry():
    me = Identity.generate()
    disc = Discovery(identity=me)
    fresh, stale = Identity.generate().node_id, Identity.generate().node_id
    now = time.time()
    disc.peer_addrs[fresh] = ("127.0.0.1", 1)
    disc.peer_seen_at[fresh] = now
    disc.peer_addrs[stale] = ("127.0.0.1", 2)
    disc.peer_seen_at[stale] = now - STALE_AFTER_S - 1
    disc._expire_stale()
    assert fresh in disc.peer_addrs
    assert stale not in disc.peer_addrs


async def test_multi_bootstrap_survives_one_dead_contact():
    """Decision #023 D10: a restarting/joining node given SEVERAL bootstrap
    addresses must still join even if one of them is dead (chaos-test SPOF
    fix — a single bootstrap address is a structural weakness)."""
    id_a, id_b = Identity.generate(), Identity.generate()
    disc_a, disc_b = Discovery(identity=id_a), Discovery(identity=id_b)
    port_a = await _start(disc_a, publish=False)
    dead_port = port_a + 1 if port_a < 65000 else port_a - 1  # a port nothing listens on

    port_b = await _start(disc_b, bootstrap=[("127.0.0.1", dead_port), ("127.0.0.1", port_a)])
    await asyncio.sleep(0.2)
    await disc_a._publish_advert()
    await asyncio.sleep(0.1)
    addr = await disc_b.lookup(id_a.node_id)

    disc_a.server.stop()
    disc_b.server.stop()
    assert addr == disc_a._my_addr  # still joined despite the first dead contact


async def _run_all():
    async_tests = [
        test_two_node_advert_and_lookup_via_dht,
        test_both_directions_discoverable_distinct_keys,
        test_forged_advert_rejected,
        test_multi_bootstrap_survives_one_dead_contact,
    ]
    sync_tests = [test_pex_growth_and_dedup, test_pex_propagates_topics, test_stale_peer_expiry]
    for t in async_tests:
        await t()
        print(f"ok  {t.__name__}")
    for t in sync_tests:
        t()
        print(f"ok  {t.__name__}")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
