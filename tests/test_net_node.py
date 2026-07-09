"""Sanity check for NetNode (DHT + PEX + UDP + inference wiring) with two
nodes in one process before trusting it inside T7's multiprocess harness."""

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.network.identity import Identity
from swarmint.node.net_node import NetNode


async def test_two_netnodes_rendezvous_and_gossip():
    id_a, id_b = Identity.generate(), Identity.generate()
    net_a = NetNode(identity=id_a, topic=1, rng=random.Random(1), gossip_interval=0.03)
    net_b = NetNode(identity=id_b, topic=1, rng=random.Random(2), gossip_interval=0.03)

    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    net_a.data_feed = lambda: (rng_a.normal(0, 0.3, 8).astype(np.float32), 0)

    def feed_b():
        if rng_b.random() < 0.5:
            return (rng_b.normal(0, 0.3, 8).astype(np.float32), 0)
        return ((rng_b.normal(0, 0.3, 8) + 6).astype(np.float32), 1)
    net_b.data_feed = feed_b

    await net_a.start("127.0.0.1", 0, 0, [])  # node A is the rendezvous origin
    dht_a_port = net_a.discovery.server.transport.get_extra_info("sockname")[1]
    await net_b.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_a_port)], rendezvous_id=id_a.node_id)

    # DHT bootstrap + Discovery's fast-retry-until-published loop + the
    # rendezvous lookup's own retry (net_node.py) all add real wall-clock
    # latency (~1-2s) before gossip even starts flowing — separate from the
    # gossip_interval itself. Budget for that startup cost, not just N rounds.
    for _ in range(400):  # up to ~12s wall clock
        if net_b.node.merges_ok >= 1:
            break
        await asyncio.sleep(0.03)

    await net_a.stop()
    await net_b.stop()

    assert id_a.node_id in net_b.node.peers  # discovered via DHT rendezvous, not hardcoded
    assert net_b.node.merges_ok >= 1


async def _run_all():
    await test_two_netnodes_rendezvous_and_gossip()
    print("ok  test_two_netnodes_rendezvous_and_gossip")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
