"""T4: relay fallback — reach a peer with no direct path via a mutual relay.

Loopback can't create an un-punchable NAT, so we SIMULATE that condition the
same way NetNode.mark_relay_only does in the field (after a punch fails): mark
the peer relay-only, which forces every send to it to wrap through the relay
regardless of whether a direct address is known. This exercises the full
transport path — wrap at the origin, forward at the relay, unwrap + verify +
reply-path learning at the destination — which is representation-identical to
the real symmetric-NAT case (the punch just never opens the direct path there).
"""

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.prototype_model import Prototype
from swarmint.network.bus import Message
from swarmint.network.identity import Identity
from swarmint.node.net_node import NetNode


async def _relay_trio():
    """R = rendezvous/relay; A, B = two nodes that (pretend they) can't reach
    each other directly. Returns (R, A, B) all started, with R knowing both."""
    id_r, id_a, id_b = Identity.generate(), Identity.generate(), Identity.generate()
    r = NetNode(identity=id_r, topic=1, rng=random.Random(0), gossip_interval=10.0, enable_relay=True)
    a = NetNode(identity=id_a, topic=1, rng=random.Random(1), gossip_interval=10.0, enable_relay=True)
    b = NetNode(identity=id_b, topic=1, rng=random.Random(2), gossip_interval=10.0, enable_relay=True)
    await r.start("127.0.0.1", 0, 0, [])
    dht_r = r.discovery.server.transport.get_extra_info("sockname")[1]
    await a.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_r)], rendezvous_id=id_r.node_id)
    await b.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_r)], rendezvous_id=id_r.node_id)
    # A and B each contact R (as they do in the field via reflexive discovery),
    # so the relay learns both return addresses and can bridge them.
    await a.nat.discover_reflexive(id_r.node_id, timeout=1.5)
    await b.nat.discover_reflexive(id_r.node_id, timeout=1.5)
    return r, a, b


async def _wait_for(cond, timeout=2.0):
    for _ in range(int(timeout / 0.02)):
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


async def test_relay_delivers_gossip_and_learns_return_path():
    r, a, b = await _relay_trio()
    id_a, id_b, id_r = a.identity.node_id, b.identity.node_id, r.identity.node_id

    # A treats B as unreachable-direct (as if a punch had failed).
    a.mark_relay_only(id_b)
    assert id_b not in a.bus.peer_addrs, "test premise: A must NOT know B directly"

    # Capture what B's node receives.
    got = []
    orig = b.node.receive
    b.node.receive = lambda m: (got.append(m), orig(m))

    proto = Prototype(np.ones(4, dtype=np.float32), label=7, weight=1.0)
    a.bus.send(Message(id_a, id_b, "protos", [proto]))

    delivered = await _wait_for(lambda: len(got) >= 1)

    # Reverse direction: B should now auto-relay its reply back to A through R
    # (it learned the return path when it received the relayed message).
    got_a = []
    origa = a.node.receive
    a.node.receive = lambda m: (got_a.append(m), origa(m))
    b.bus.send(Message(id_b, id_a, "protos", [Prototype(np.zeros(4, dtype=np.float32), 3, 1.0)]))
    reply_delivered = await _wait_for(lambda: len(got_a) >= 1)

    # counters prove the path (origin wrap -> relay forward -> dest unwrap)
    a_tx, r_fwd, b_rx = a.bus.relayed_tx, r.bus.relayed_fwd, b.bus.relayed_rx
    b_learned = id_a in b.bus.relay_only

    await a.stop(); await b.stop(); await r.stop()

    assert delivered, "B never received A's gossip via the relay"
    assert a_tx >= 1, f"A did not wrap a relay message (relayed_tx={a_tx})"
    assert r_fwd >= 1, f"R did not forward the relay (relayed_fwd={r_fwd})"
    assert b_rx >= 1, f"B did not receive a relayed message (relayed_rx={b_rx})"
    assert b_learned, "B did not learn to reply to A via the relay"
    assert reply_delivered, "A never received B's relayed reply"


async def test_multi_relay_fails_over_when_primary_unreachable():
    """T6: with two relays, if the primary becomes unreachable the node fails
    over to the backup and the peer stays reachable — no single relay is a
    point of failure."""
    id_r1, id_r2 = Identity.generate(), Identity.generate()
    id_a, id_b = Identity.generate(), Identity.generate()
    r1 = NetNode(identity=id_r1, topic=1, rng=random.Random(0), gossip_interval=10.0, enable_relay=True)
    r2 = NetNode(identity=id_r2, topic=1, rng=random.Random(1), gossip_interval=10.0, enable_relay=True)
    a = NetNode(identity=id_a, topic=1, rng=random.Random(2), gossip_interval=10.0, enable_relay=True)
    b = NetNode(identity=id_b, topic=1, rng=random.Random(3), gossip_interval=10.0, enable_relay=True)
    await r1.start("127.0.0.1", 0, 0, [])
    await r2.start("127.0.0.1", 0, 0, [])
    dht_r1 = r1.discovery.server.transport.get_extra_info("sockname")[1]
    await a.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_r1)], rendezvous_id=id_r1.node_id)
    await b.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_r1)], rendezvous_id=id_r1.node_id)

    # Make BOTH relays learn both nodes: point A/B at R2's address too and have
    # each contact both relays (as they would each bootstrap node in the field).
    for node in (a, b):
        node.bus.peer_addrs[id_r2.node_id] = ("127.0.0.1", r2.bus.bound_port)
        await node.nat.discover_reflexive(id_r1.node_id, timeout=1.5)
        await node.nat.discover_reflexive(id_r2.node_id, timeout=1.5)
        node.add_relay(id_r2.node_id)          # R1 primary (from bootstrap), R2 backup

    a.mark_relay_only(id_b.node_id)

    got = []
    orig = b.node.receive
    b.node.receive = lambda m: (got.append(m), orig(m))

    # Phase 1: primary (R1) carries it.
    a.bus.send(Message(id_a, id_b.node_id, "protos", [Prototype(np.ones(4, dtype=np.float32), 1, 1.0)]))
    via_primary = await _wait_for(lambda: len(got) >= 1)
    r1_fwd_after = r1.bus.relayed_fwd

    # Phase 2: primary goes dark — drop A's route to R1. Next send must fail over.
    a.bus.peer_addrs.pop(id_r1.node_id, None)
    a.bus.relay_id = None if a.bus.relay_id == id_r1.node_id else a.bus.relay_id
    # (relay_id cleared only reflects our LOCAL view; R2 is still a candidate)
    a.add_relay(id_r2.node_id)
    got.clear()
    a.bus.send(Message(id_a, id_b.node_id, "protos", [Prototype(np.zeros(4, dtype=np.float32), 2, 1.0)]))
    via_backup = await _wait_for(lambda: len(got) >= 1)
    r2_fwd = r2.bus.relayed_fwd

    for n in (a, b, r1, r2):
        await n.stop()

    assert via_primary, "primary relay did not deliver"
    assert r1_fwd_after >= 1, f"primary relay never forwarded (r1 fwd={r1_fwd_after})"
    assert via_backup, "backup relay did not deliver after primary went down"
    assert r2_fwd >= 1, f"backup relay never forwarded (r2 fwd={r2_fwd})"


async def test_relay_off_by_default_no_wrapping():
    """With enable_relay unset (the default), marking a peer relay-only is a
    no-op and nothing is wrapped — the invariant that existing runs are
    byte-for-byte unchanged."""
    id_r, id_a = Identity.generate(), Identity.generate()
    r = NetNode(identity=id_r, topic=1, rng=random.Random(0), gossip_interval=10.0)  # relay OFF
    a = NetNode(identity=id_a, topic=1, rng=random.Random(1), gossip_interval=10.0)  # relay OFF
    await r.start("127.0.0.1", 0, 0, [])
    dht_r = r.discovery.server.transport.get_extra_info("sockname")[1]
    await a.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_r)], rendezvous_id=id_r.node_id)

    a.mark_relay_only(r.identity.node_id)   # should be ignored: relay disabled
    assert not a.bus.enable_relay
    assert a.bus.relay_id is None
    assert r.identity.node_id not in a.bus.relay_only

    await a.stop(); await r.stop()


async def _run_all():
    await test_relay_delivers_gossip_and_learns_return_path()
    print("ok  test_relay_delivers_gossip_and_learns_return_path")
    await test_multi_relay_fails_over_when_primary_unreachable()
    print("ok  test_multi_relay_fails_over_when_primary_unreachable")
    await test_relay_off_by_default_no_wrapping()
    print("ok  test_relay_off_by_default_no_wrapping")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
