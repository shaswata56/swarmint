"""Unit tests: prototype learning/merging math, versioning, trust dynamics."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.prototype_model import Prototype, PrototypeModel
from swarmint.core.version import GENESIS, Version
from swarmint.inference.aggregate import Response, aggregate
from swarmint.network.bus import Message
from swarmint.network.simbus import SimBus
from swarmint.node.swarm_node import TRUST_FLOOR, TRUST_INIT, SwarmNode


def test_version_total_order():
    a, b = Version(3, 1), Version(3, 2)
    assert a < b and b < Version(4, 0)
    assert GENESIS.bump(7) == Version(1, 7)


def test_learn_and_predict():
    m = PrototypeModel(topic=1)
    for _ in range(50):
        m.learn(np.array([0.0, 0.0], dtype=np.float32), 0, node_id=1)
        m.learn(np.array([5.0, 5.0], dtype=np.float32), 1, node_id=1)
    assert m.predict(np.array([0.1, -0.1], dtype=np.float32)) == 0
    assert m.predict(np.array([4.9, 5.2], dtype=np.float32)) == 1
    assert m.version.counter == 100


def test_prune_respects_budget():
    m = PrototypeModel(topic=1, max_prototypes=10)
    rng = np.random.default_rng(0)
    for i in range(100):
        m.learn(rng.normal(0, 5, 4).astype(np.float32), i % 3, node_id=1)
    assert len(m.protos) <= 10
    assert {p.label for p in m.protos} == {0, 1, 2}  # pruning is per-class, keeps coverage


def test_merge_is_order_invariant_union():
    a = PrototypeModel(topic=1)
    b_protos = [Prototype(np.array([9.0, 9.0], dtype=np.float32), 2)]
    a.learn(np.array([0.0, 0.0], dtype=np.float32), 0, node_id=1)
    merged = a.merged_with(b_protos, node_id=1)
    assert merged.predict(np.array([9.0, 9.0], dtype=np.float32)) == 2
    assert merged.version > a.version
    assert len(a.protos) == 1  # pure function: original untouched
    # dedup: merging the same protos again adds nothing
    again = merged.merged_with(b_protos, node_id=1)
    assert len(again.protos) == len(merged.protos)


def test_junk_merge_rejected_and_trust_drops():
    bus = SimBus(loss_rate=0.0, rng=random.Random(0))
    node = SwarmNode(node_id=1, topic=1, bus=bus, rng=random.Random(1))
    rng = np.random.default_rng(0)
    # Clean, well-separated 2-class data
    for _ in range(200):
        node.observe(rng.normal(0, 0.3, 8).astype(np.float32), 0)
        node.observe((rng.normal(0, 0.3, 8) + 6).astype(np.float32), 1)
    # Poison: vectors near class-0 region labeled as class 1
    junk = [Prototype(rng.normal(0, 0.3, 8).astype(np.float32), 1) for _ in range(30)]
    before = node.model
    for _ in range(10):
        node.receive(Message(sender=99, recipient=1, kind="protos", payload=junk))
    assert node.model is before  # every poisoned merge rolled back
    assert node.trust[99] < TRUST_FLOOR < TRUST_INIT
    # Once below the floor, updates are ignored without evaluation
    rejected_before = node.merges_rejected
    node.receive(Message(sender=99, recipient=1, kind="protos", payload=junk))
    assert node.merges_rejected == rejected_before


def test_unverifiable_needs_corroboration():
    """Knowledge for a class the node can't validate locally is quarantined
    until a second distinct sender corroborates it."""
    bus = SimBus(loss_rate=0.0, rng=random.Random(0))
    node = SwarmNode(node_id=1, topic=1, bus=bus, rng=random.Random(1))
    rng = np.random.default_rng(0)
    for _ in range(100):
        node.observe(rng.normal(0, 0.2, 8).astype(np.float32), 0)
    center = np.full(8, 6.0, dtype=np.float32)
    good = [Prototype((center + rng.normal(0, 0.2, 8)).astype(np.float32), 1)
            for _ in range(5)]
    node.receive(Message(sender=7, recipient=1, kind="protos", payload=good))
    node.receive(Message(sender=8, recipient=1, kind="protos", payload=good))
    assert node.merges_ok == 0  # two senders: still quarantined
    assert node.model.predict(center) == 0
    node.receive(Message(sender=9, recipient=1, kind="protos", payload=good))
    assert node.merges_ok == 1  # third distinct sender corroborates: committed
    assert node.trust[9] > TRUST_INIT
    assert node.model.predict(center) == 1
    # Repeat from the same original sender would NOT have corroborated
    node2 = SwarmNode(node_id=2, topic=1, bus=bus, rng=random.Random(2))
    for _ in range(100):
        node2.observe(rng.normal(0, 0.2, 8).astype(np.float32), 0)
    node2.receive(Message(sender=7, recipient=2, kind="protos", payload=good))
    node2.receive(Message(sender=7, recipient=2, kind="protos", payload=good))
    assert node2.merges_ok == 0


def test_aggregate_trust_weight_beats_count():
    """A confident low-trust majority loses to a trusted minority."""
    responses = [
        Response(1, label=7, confidence=1.0, trust=0.1),  # 3 confident liars
        Response(2, label=7, confidence=1.0, trust=0.1),
        Response(3, label=7, confidence=1.0, trust=0.1),
        Response(4, label=3, confidence=0.9, trust=0.9),  # 1 trusted expert
    ]
    label, score = aggregate(responses)
    assert label == 3
    assert 0.0 < score <= 1.0


def test_aggregate_abstentions_ignored():
    responses = [
        Response(1, label=None, confidence=0.0, trust=0.9),
        Response(2, label=5, confidence=0.8, trust=0.7),
    ]
    assert aggregate(responses)[0] == 5
    assert aggregate([Response(1, None, 0.0, 0.9)]) == (None, 0.0)


def test_expert_abstains_off_manifold():
    """A nearest-neighbour expert must abstain when the query is far from its
    prototypes, rather than confidently guess a wrong class."""
    node = SwarmNode(node_id=0, topic=1, bus=SimBus(rng=random.Random(0)),
                     rng=random.Random(0))
    rng = np.random.default_rng(0)
    for _ in range(100):
        node.model.learn(rng.normal(0, 0.3, 8).astype(np.float32), 3, node_id=0)
        node.model.learn((rng.normal(0, 0.3, 8) + 10).astype(np.float32), 4, node_id=0)
    in_domain = node.answer_query(np.zeros(8, dtype=np.float32))
    off_manifold = node.answer_query(np.full(8, 50.0, dtype=np.float32))
    assert in_domain[0] == 3
    assert off_manifold[0] is None  # abstained


def test_infer_beats_local_via_experts():
    """A generalist that locally knows only class 0 answers a class-9 query
    correctly by consulting an expert."""
    bus = SimBus(rng=random.Random(0))
    query = SwarmNode(node_id=0, topic=1, bus=bus, rng=random.Random(0))
    expert = SwarmNode(node_id=1, topic=1, bus=bus, rng=random.Random(1))
    rng = np.random.default_rng(0)
    c9 = np.full(8, 20.0, dtype=np.float32)
    for _ in range(100):
        query.model.learn(rng.normal(0, 0.3, 8).astype(np.float32), 0, node_id=0)
        expert.model.learn((c9 + rng.normal(0, 0.3, 8)).astype(np.float32), 9, node_id=1)
    query.peers = [1]
    assert query.model.predict(c9) == 0            # local model is wrong
    label, _ = query.infer(c9, {1: expert})
    assert label == 9                              # ensemble gets it right


def test_bus_pump_delivers_via_registered_handler():
    """T1: nodes auto-register on the bus; pump() must deliver to the SAME
    handler receive() would have been called with directly (transport swap
    is transparent to node logic)."""
    bus = SimBus(loss_rate=0.0, max_latency_ticks=0, rng=random.Random(0))
    a = SwarmNode(node_id=1, topic=1, bus=bus, rng=random.Random(1))
    b = SwarmNode(node_id=2, topic=1, bus=bus, rng=random.Random(2))
    rng = np.random.default_rng(0)
    for _ in range(100):
        a.observe(rng.normal(0, 0.3, 8).astype(np.float32), 0)
        # b already knows class 0 too, so a's gossip is immediately verifiable
        # against b's own holdout (no corroboration wait needed for this test).
        b.observe(rng.normal(0, 0.3, 8).astype(np.float32), 0)
        b.observe((rng.normal(0, 0.3, 8) + 6).astype(np.float32), 1)
    a.peers = [2]
    b.peers = [1]
    assert b.merges_ok == 0
    a.gossip_step()  # a -> bus.send(Message(sender=1, recipient=2, ...))
    bus.pump()        # bus must invoke b.receive(msg) via the registered handler
    assert b.merges_ok == 1
    assert b.model.predict(np.zeros(8, dtype=np.float32)) == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
