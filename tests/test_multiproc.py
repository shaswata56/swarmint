"""T7: multiprocess integration regression (real OS processes + UDP + DHT).

Heavier and slower than the unit tests (spawns real processes, ~30s), so it
runs a modest scale with lenient thresholds. The full-scale demo with strict
acceptance criteria is `python -m swarmint.sim.run_multiproc`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.sim import run_multiproc as rm
from swarmint.sim.resources import ResourceBudget


def test_multiproc_swarm_converges_and_suppresses_malicious():
    # 8 nodes, 1 malicious (within the corroboration-safe regime for this size),
    # short run. max_nodes pins the count so the test is deterministic across
    # machines regardless of core count.
    budget = ResourceBudget(max_nodes=8)
    result = rm.run(n_nodes=8, n_malicious=1, duration_s=30.0, gossip_interval=0.1,
                    budget=budget, gossip_sample_size=12)

    assert all(v is not None for v in result["per_node"].values()), "some node produced no metrics"
    assert result["mean_honest_acc"] >= 0.60, f"acc too low: {result['mean_honest_acc']}"

    mal_t = result["mean_malicious_trust"]
    hon_t = result["mean_honest_trust"]
    if mal_t is not None and hon_t is not None:
        assert mal_t < hon_t, f"malicious trust {mal_t} not below honest {hon_t}"

    bw = result["mean_bandwidth_bytes_per_s_normalized"]
    assert bw is not None and bw < rm.BANDWIDTH_BUDGET_BYTES_PER_S, f"bandwidth {bw} over budget"


def test_resource_budget_caps_node_count():
    """The 60%-CPU / RAM governor must cap a large request, and honor overrides."""
    import os
    b = ResourceBudget(cpu_fraction=0.60)
    plan = b.resolve(10_000)
    assert plan["nodes"] <= b.cpu_cap()
    assert plan["nodes"] <= int(os.cpu_count() * 0.60) + 1
    assert plan["capped_by"] in ("cpu", "memory")

    # explicit max_nodes override wins and short-circuits the resource math
    b2 = ResourceBudget(max_nodes=4)
    assert b2.resolve(100)["nodes"] == 4

    # env-var configurability
    os.environ["SWARMINT_MAX_NODES"] = "5"
    try:
        assert ResourceBudget().resolve(100)["nodes"] == 5
    finally:
        del os.environ["SWARMINT_MAX_NODES"]


def test_chaos_survives_node_death_restart_and_packet_loss():
    """T8: under 20% injected packet loss, kill 30% of nodes mid-run and
    restart one; the honest survivors must keep converging and the restarted
    node must re-bootstrap (DHT+PEX) and catch up."""
    budget = ResourceBudget(max_nodes=10)
    result = rm.run_chaos(n_nodes=10, n_malicious=1, duration_s=55.0, gossip_interval=0.1,
                          budget=budget, loss_rate=0.20, kill_fraction=0.30, chaos_at_s=18.0)
    c = result["chaos"]
    assert len(c["victims"]) >= 2, "chaos should have killed multiple nodes"
    assert c["mean_survivor_acc"] >= 0.60, f"survivors did not hold convergence: {c['mean_survivor_acc']}"
    assert c["restarted_final_acc"] is not None and c["restarted_final_acc"] >= 0.50, \
        f"restarted node failed to rejoin/catch up: {c['restarted_final_acc']}"
    assert c["mean_dropped_loss"] > 0, "packet loss was supposed to be injected but none recorded"


if __name__ == "__main__":
    # resource-budget test first (instant), then the heavy multiproc + chaos tests
    test_resource_budget_caps_node_count()
    print("ok  test_resource_budget_caps_node_count")
    test_multiproc_swarm_converges_and_suppresses_malicious()
    print("ok  test_multiproc_swarm_converges_and_suppresses_malicious")
    test_chaos_survives_node_death_restart_and_packet_loss()
    print("ok  test_chaos_survives_node_death_restart_and_packet_loss")
    print("all tests passed")
