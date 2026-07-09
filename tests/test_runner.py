"""T4b: NodeRunner drives gossip on a wall-clock timer over real UDP, while
receive() is invoked by UdpBus's datagram callback — both on the same event
loop, proving no locking is needed (see runner.py docstring)."""

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.network.identity import Identity
from swarmint.network.udp_bus import UdpBus
from swarmint.node.runner import NodeRunner
from swarmint.node.swarm_node import SwarmNode

FAST_INTERVAL = 0.03  # fast enough for a snappy test, still a real timer


def _feeder(rng, mean_offset, label):
    def feed():
        return (rng.normal(0, 0.3, 8).astype(np.float32) + mean_offset, label)
    return feed


async def test_two_runners_converge_over_real_udp():
    id_a, id_b = Identity.generate(), Identity.generate()
    bus_a, bus_b = UdpBus(identity=id_a), UdpBus(identity=id_b)
    port_a = await bus_a.start("127.0.0.1", 0)
    port_b = await bus_b.start("127.0.0.1", 0)
    bus_a.peer_addrs[id_b.node_id] = ("127.0.0.1", port_b)
    bus_b.peer_addrs[id_a.node_id] = ("127.0.0.1", port_a)

    node_a = SwarmNode(node_id=id_a.node_id, topic=1, bus=bus_a, rng=random.Random(1))
    node_b = SwarmNode(node_id=id_b.node_id, topic=1, bus=bus_b, rng=random.Random(2))
    node_a.peers = [id_b.node_id]
    node_b.peers = [id_a.node_id]

    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(2)
    # b already knows class 0 (from a) AND class 1, so a's gossip is
    # immediately verifiable — isolates "does the timer+socket loop work"
    # from the corroboration-latency question already covered by test_core.
    feed_a = _feeder(rng_a, 0.0, 0)

    def feed_b():
        if rng_b.random() < 0.5:
            return (rng_b.normal(0, 0.3, 8).astype(np.float32), 0)
        return ((rng_b.normal(0, 0.3, 8) + 6).astype(np.float32), 1)

    runner_a = NodeRunner(node=node_a, gossip_interval=FAST_INTERVAL, data_feed=feed_a)
    runner_b = NodeRunner(node=node_b, gossip_interval=FAST_INTERVAL, data_feed=feed_b)
    await runner_a.start()
    await runner_b.start()

    for _ in range(200):  # up to ~2s wall clock
        if node_b.merges_ok >= 1:
            break
        await asyncio.sleep(0.02)

    await runner_a.stop()
    await runner_b.stop()
    await bus_a.stop()
    await bus_b.stop()

    assert node_b.merges_ok >= 1
    assert runner_a.round_number() >= 1
    assert runner_a.gossip_ticks >= 1


async def test_stop_cancels_cleanly():
    id_a = Identity.generate()
    bus_a = UdpBus(identity=id_a)
    await bus_a.start("127.0.0.1", 0)
    node_a = SwarmNode(node_id=id_a.node_id, topic=1, bus=bus_a, rng=random.Random(0))
    runner = NodeRunner(node=node_a, gossip_interval=FAST_INTERVAL)
    await runner.start()
    await asyncio.sleep(FAST_INTERVAL * 3)
    await runner.stop()
    ticks_after_stop = runner.gossip_ticks
    await asyncio.sleep(FAST_INTERVAL * 3)
    assert runner.gossip_ticks == ticks_after_stop  # truly stopped, not still ticking
    await bus_a.stop()


async def _run_all():
    for t in [test_two_runners_converge_over_real_udp, test_stop_cancels_cleanly]:
        await t()
        print(f"ok  {t.__name__}")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
