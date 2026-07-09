"""D3: NAT traversal hole-punch signaling.

Loopback has no NAT to open, so this validates the full PROTOCOL — reflexive
discovery, rendezvous-relayed connect signaling, and the punch/ack handshake —
and that A and B end up able to talk directly. Opening a real NAT mapping needs
two actual NATs; the handshake exercised here is the same one that does it.
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


async def test_reflexive_discovery():
    """A node learns its own observed (host,port) from the rendezvous."""
    id_r, id_a = Identity.generate(), Identity.generate()
    r = NetNode(identity=id_r, topic=1, rng=random.Random(0), gossip_interval=10.0)
    a = NetNode(identity=id_a, topic=1, rng=random.Random(1), gossip_interval=10.0)
    await r.start("127.0.0.1", 0, 0, [])
    dht_r = r.discovery.server.transport.get_extra_info("sockname")[1]
    await a.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_r)], rendezvous_id=id_r.node_id)

    reflexive = await a.nat.discover_reflexive(id_r.node_id, timeout=1.5)
    await a.stop()
    await r.stop()

    assert reflexive is not None
    assert reflexive[0] == "127.0.0.1"
    assert reflexive[1] == a.bus.bound_port  # loopback: reflexive == local (no NAT remap)


async def test_hole_punch_establishes_direct_path():
    """A and B, each known only to rendezvous R, complete a hole-punch and can
    then exchange a gossip message directly (A never learned B via PEX/DHT)."""
    id_r, id_a, id_b = Identity.generate(), Identity.generate(), Identity.generate()
    r = NetNode(identity=id_r, topic=1, rng=random.Random(0), gossip_interval=10.0)
    a = NetNode(identity=id_a, topic=1, rng=random.Random(1), gossip_interval=10.0)
    b = NetNode(identity=id_b, topic=1, rng=random.Random(2), gossip_interval=10.0)
    await r.start("127.0.0.1", 0, 0, [])
    dht_r = r.discovery.server.transport.get_extra_info("sockname")[1]
    await a.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_r)], rendezvous_id=id_r.node_id)
    await b.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_r)], rendezvous_id=id_r.node_id)

    # Both register their reflexive addr with R (so R can relay). In a live run
    # this also happens naturally as gossip flows through R.
    await a.nat.discover_reflexive(id_r.node_id, timeout=1.5)
    await b.nat.discover_reflexive(id_r.node_id, timeout=1.5)

    assert id_b.node_id not in a.bus.peer_addrs  # A cannot reach B yet

    ok = await a.nat.punch(id_b.node_id, id_r.node_id, timeout=3.0)

    # After the punch, A and B know each other's addresses and are marked reachable.
    a_reaches_b = id_b.node_id in a.nat.reachable
    b_reaches_a = id_a.node_id in b.nat.reachable

    # Prove a real direct message now flows A -> B without going through R.
    b.node.peers = []  # make sure any delivery is from the punched path, not prior state
    received = []
    orig = b.node.receive
    b.node.receive = lambda m: (received.append(m), orig(m))
    a.bus.send(Message(id_a.node_id, id_b.node_id, "protos",
                       [Prototype(np.zeros(4, dtype=np.float32), 0, 1.0)]))
    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.02)

    await a.stop()
    await b.stop()
    await r.stop()

    assert ok, "punch handshake did not complete"
    assert a_reaches_b and b_reaches_a, f"reachability: A->B {a_reaches_b}, B->A {b_reaches_a}"
    assert id_b.node_id in a.bus.peer_addrs, "A never learned B's address"
    assert len(received) == 1, "direct A->B datagram did not arrive after punch"


async def _run_all():
    await test_reflexive_discovery()
    print("ok  test_reflexive_discovery")
    await test_hole_punch_establishes_direct_path()
    print("ok  test_hole_punch_establishes_direct_path")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
