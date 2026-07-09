"""T6: async inference RPC over real UDP — request-collect-aggregate instead
of direct in-process calls, reusing answer_query()/aggregate() unchanged."""

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.inference.rpc import wire_up
from swarmint.network.identity import Identity
from swarmint.network.udp_bus import UdpBus
from swarmint.node.swarm_node import TRUST_INIT, SwarmNode


async def test_infer_over_udp_beats_local():
    """Network analog of test_infer_beats_local_via_experts: a generalist that
    locally knows only class 0 answers a class-9 query correctly by querying
    a real expert over the wire."""
    id_q, id_e = Identity.generate(), Identity.generate()
    bus_q, bus_e = UdpBus(identity=id_q), UdpBus(identity=id_e)
    port_q = await bus_q.start("127.0.0.1", 0)
    port_e = await bus_e.start("127.0.0.1", 0)
    bus_q.peer_addrs[id_e.node_id] = ("127.0.0.1", port_e)
    bus_e.peer_addrs[id_q.node_id] = ("127.0.0.1", port_q)

    query = SwarmNode(node_id=id_q.node_id, topic=1, bus=bus_q, rng=random.Random(0))
    expert = SwarmNode(node_id=id_e.node_id, topic=1, bus=bus_e, rng=random.Random(1))
    svc_q = wire_up(query, bus_q)
    wire_up(expert, bus_e)

    rng = np.random.default_rng(0)
    c9 = np.full(8, 20.0, dtype=np.float32)
    for _ in range(100):
        query.model.learn(rng.normal(0, 0.3, 8).astype(np.float32), 0, id_q.node_id)
        expert.model.learn((c9 + rng.normal(0, 0.3, 8)).astype(np.float32), 9, id_e.node_id)

    assert query.model.predict(c9) == 0  # local model is wrong

    label, score = await svc_q.infer(c9, [id_e.node_id])
    await bus_q.stop()
    await bus_e.stop()

    assert label == 9  # the networked ensemble gets it right
    assert score > 0.0


async def test_calibrate_trust_over_udp_sinks_liar():
    """A malicious peer that answers confidently wrong must end up with lower
    trust than an honest peer, using the SAME reward formula as the in-process
    SwarmNode.calibrate_trust — but driven by real chunked UDP round-trips."""
    id_q, id_honest, id_liar = Identity.generate(), Identity.generate(), Identity.generate()
    bus_q = UdpBus(identity=id_q)
    bus_h = UdpBus(identity=id_honest)
    bus_l = UdpBus(identity=id_liar)
    port_q = await bus_q.start("127.0.0.1", 0)
    port_h = await bus_h.start("127.0.0.1", 0)
    port_l = await bus_l.start("127.0.0.1", 0)
    for bus, port in [(bus_q, port_q), (bus_h, port_h), (bus_l, port_l)]:
        pass
    bus_q.peer_addrs[id_honest.node_id] = ("127.0.0.1", port_h)
    bus_q.peer_addrs[id_liar.node_id] = ("127.0.0.1", port_l)
    bus_h.peer_addrs[id_q.node_id] = ("127.0.0.1", port_q)
    bus_l.peer_addrs[id_q.node_id] = ("127.0.0.1", port_q)

    query = SwarmNode(node_id=id_q.node_id, topic=1, bus=bus_q, rng=random.Random(0))
    honest = SwarmNode(node_id=id_honest.node_id, topic=1, bus=bus_h, rng=random.Random(1))
    liar = SwarmNode(node_id=id_liar.node_id, topic=1, bus=bus_l, rng=random.Random(2), malicious=True)
    svc_q = wire_up(query, bus_q)
    wire_up(honest, bus_h)
    wire_up(liar, bus_l)

    rng = np.random.default_rng(0)
    # query's own holdout covers class 0 (verifiable evidence for calibration)
    for _ in range(100):
        x = rng.normal(0, 0.3, 8).astype(np.float32)
        query.observe(x, 0)
        honest.model.learn(x + rng.normal(0, 0.05, 8).astype(np.float32), 0, id_honest.node_id)
        liar.model.learn(x + rng.normal(0, 0.05, 8).astype(np.float32), 0, id_liar.node_id)

    await svc_q.calibrate_trust([id_honest.node_id, id_liar.node_id])
    await bus_q.stop()
    await bus_h.stop()
    await bus_l.stop()

    assert query.trust[id_honest.node_id] > TRUST_INIT
    assert query.trust[id_liar.node_id] < TRUST_INIT
    assert query.trust[id_honest.node_id] > query.trust[id_liar.node_id]


async def test_query_timeout_treated_as_abstention():
    """A peer that never responds (e.g. offline) must not hang inference —
    the query times out and is treated exactly like an honest abstention."""
    id_q, id_dead = Identity.generate(), Identity.generate()
    bus_q = UdpBus(identity=id_q)
    await bus_q.start("127.0.0.1", 0)
    bus_q.peer_addrs[id_dead.node_id] = ("127.0.0.1", 1)  # nobody listens on port 1

    query = SwarmNode(node_id=id_q.node_id, topic=1, bus=bus_q, rng=random.Random(0))
    svc_q = wire_up(query, bus_q)
    rng = np.random.default_rng(0)
    for _ in range(50):
        query.model.learn(rng.normal(0, 0.3, 8).astype(np.float32), 0, id_q.node_id)

    label, score = await svc_q.query_peer(id_dead.node_id, np.zeros(8, dtype=np.float32), timeout=0.05)
    await bus_q.stop()
    assert label is None and score == 0.0  # timed out, not hung, not crashed


async def _run_all():
    for t in [test_infer_over_udp_beats_local, test_calibrate_trust_over_udp_sinks_liar,
             test_query_timeout_treated_as_abstention]:
        await t()
        print(f"ok  {t.__name__}")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
