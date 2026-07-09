"""D2: finer topic granularity — topics advertised via DHT + propagated via
PEX, and inference routed to topic-relevant peers over real UDP."""

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.network.identity import Identity
from swarmint.node.net_node import NetNode


async def test_topic_advert_and_topic_scoped_inference():
    """A query node knows two experts: one advertising topic 3 (expert on
    class 3), one advertising topic 7 (class 7). A topic-3 query must be
    answered correctly and routed only to the topic-3 expert."""
    id_q = Identity.generate()
    id_e3 = Identity.generate()
    id_e7 = Identity.generate()

    q = NetNode(identity=id_q, topic=0, topics={0}, rng=random.Random(0), gossip_interval=10.0)
    e3 = NetNode(identity=id_e3, topic=3, topics={3}, rng=random.Random(3), gossip_interval=10.0)
    e7 = NetNode(identity=id_e7, topic=7, topics={7}, rng=random.Random(7), gossip_interval=10.0)

    # q is the rendezvous origin; both experts bootstrap to it
    await q.start("127.0.0.1", 0, 0, [])
    dht_q = q.discovery.server.transport.get_extra_info("sockname")[1]
    await e3.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_q)], rendezvous_id=id_q.node_id)
    await e7.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_q)], rendezvous_id=id_q.node_id)

    # Train the experts on their respective class regions.
    rng = np.random.default_rng(0)
    c3 = np.full(8, 3.0, dtype=np.float32)
    c7 = np.full(8, 30.0, dtype=np.float32)
    for _ in range(100):
        e3.node.model.learn((c3 + rng.normal(0, 0.2, 8)).astype(np.float32), 3, id_e3.node_id)
        e7.node.model.learn((c7 + rng.normal(0, 0.2, 8)).astype(np.float32), 7, id_e7.node_id)

    # q must discover both experts AND their topics. It learns e3/e7 as senders
    # (they bootstrapped to it) but their TOPICS come from the DHT advert, so
    # resolve them explicitly (in a live run this happens via lookup + PEX).
    for eid in (id_e3.node_id, id_e7.node_id):
        for _ in range(15):
            if eid in q.discovery.peer_topics and q.discovery.peer_topics[eid]:
                break
            await q.discovery.lookup(eid)
            await asyncio.sleep(0.2)
        q.bus.peer_addrs[eid] = q.discovery.peer_addrs.get(eid, q.bus.peer_addrs.get(eid))
        if eid not in q.node.peers:
            q.node.peers.append(eid)

    topic3_peers = set(q.discovery.peers_for_topic(3))
    topic7_peers = set(q.discovery.peers_for_topic(7))

    label3, _ = await q.infer_topic(c3, topic=3, timeout=0.3)
    label7, _ = await q.infer_topic(c7, topic=7, timeout=0.3)

    await q.stop()
    await e3.stop()
    await e7.stop()

    assert id_e3.node_id in topic3_peers and id_e7.node_id not in topic3_peers, \
        f"topic-3 routing wrong: {topic3_peers}"
    assert id_e7.node_id in topic7_peers and id_e3.node_id not in topic7_peers, \
        f"topic-7 routing wrong: {topic7_peers}"
    assert label3 == 3, f"topic-3 query got {label3}"
    assert label7 == 7, f"topic-7 query got {label7}"


async def _run_all():
    await test_topic_advert_and_topic_scoped_inference()
    print("ok  test_topic_advert_and_topic_scoped_inference")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
