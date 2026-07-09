"""T3b: proves SwarmNode/PrototypeModel/Version are id-type-agnostic by
re-running representative Phase-1/2 scenarios with real 20-byte
Identity.node_id values instead of small ints — no source change to node
logic was needed for this (grep audit: node_id is never used arithmetically,
only as a dict key / equality / opaque Version field). See decision #023 D5.
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.prototype_model import Prototype
from swarmint.network.bus import Message
from swarmint.network.identity import Identity
from swarmint.network.simbus import SimBus
from swarmint.node.swarm_node import TRUST_FLOOR, TRUST_INIT, SwarmNode


def test_bytes_node_id_learn_and_version():
    me = Identity.generate()
    assert len(me.node_id) == 20 and isinstance(me.node_id, bytes)
    node = SwarmNode(node_id=me.node_id, topic=1,
                     bus=SimBus(rng=random.Random(0)), rng=random.Random(0))
    rng = np.random.default_rng(0)
    for _ in range(50):
        node.model.learn(rng.normal(0, 0.3, 8).astype(np.float32), 0, node.node_id)
    assert node.model.version.node_id == me.node_id
    assert node.model.predict(np.zeros(8, dtype=np.float32)) == 0


def test_bytes_node_id_gossip_merge_and_trust():
    """The exact merge/trust/rollback flow from test_junk_merge_rejected_and_trust_drops,
    but with real cryptographic 20-byte ids as both node_id and gossip sender."""
    honest_id = Identity.generate().node_id
    attacker_id = Identity.generate().node_id
    bus = SimBus(loss_rate=0.0, rng=random.Random(0))
    node = SwarmNode(node_id=honest_id, topic=1, bus=bus, rng=random.Random(1))
    rng = np.random.default_rng(0)
    for _ in range(200):
        node.observe(rng.normal(0, 0.3, 8).astype(np.float32), 0)
        node.observe((rng.normal(0, 0.3, 8) + 6).astype(np.float32), 1)
    junk = [Prototype(rng.normal(0, 0.3, 8).astype(np.float32), 1) for _ in range(30)]
    before = node.model
    for _ in range(10):
        node.receive(Message(sender=attacker_id, recipient=honest_id, kind="protos", payload=junk))
    assert node.model is before
    assert node.trust[attacker_id] < TRUST_FLOOR < TRUST_INIT


def test_bytes_node_id_bus_registration_and_pump():
    a_id, b_id = Identity.generate().node_id, Identity.generate().node_id
    bus = SimBus(loss_rate=0.0, max_latency_ticks=0, rng=random.Random(0))
    a = SwarmNode(node_id=a_id, topic=1, bus=bus, rng=random.Random(1))
    b = SwarmNode(node_id=b_id, topic=1, bus=bus, rng=random.Random(2))
    rng = np.random.default_rng(0)
    for _ in range(100):
        a.observe(rng.normal(0, 0.3, 8).astype(np.float32), 0)
        b.observe(rng.normal(0, 0.3, 8).astype(np.float32), 0)
        b.observe((rng.normal(0, 0.3, 8) + 6).astype(np.float32), 1)
    a.peers = [b_id]
    a.gossip_step()
    bus.pump()
    assert b.merges_ok == 1


def test_bytes_node_id_inference_infer_beats_local():
    q_id, e_id = Identity.generate().node_id, Identity.generate().node_id
    bus = SimBus(rng=random.Random(0))
    query = SwarmNode(node_id=q_id, topic=1, bus=bus, rng=random.Random(0))
    expert = SwarmNode(node_id=e_id, topic=1, bus=bus, rng=random.Random(1))
    rng = np.random.default_rng(0)
    c9 = np.full(8, 20.0, dtype=np.float32)
    for _ in range(100):
        query.model.learn(rng.normal(0, 0.3, 8).astype(np.float32), 0, q_id)
        expert.model.learn((c9 + rng.normal(0, 0.3, 8)).astype(np.float32), 9, e_id)
    query.peers = [e_id]
    assert query.model.predict(c9) == 0
    label, _ = query.infer(c9, {e_id: expert})
    assert label == 9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
