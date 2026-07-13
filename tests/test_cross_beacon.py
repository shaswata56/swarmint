"""Cross-beacon learning end-to-end over REAL UDP (opt-in, spawns real nodes).

Gated by SWARM_PROC_TESTS=1 like the other real-socket integration tests — it
binds real UDP ports and needs a few seconds of wall-clock convergence, so it is
excluded from the deterministic fast suite CI runs. Asserts the substantive claim:
two beacons whose swarms each cover a DISJOINT half of the labels, once federated
and bridged (same space), end up each holding the OTHER half's classes and scoring
well past the half-coverage ceiling — N beacons as one system.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_two_beacons_learn_as_one_system():
    if os.environ.get("SWARM_PROC_TESTS") != "1":
        print("skip test_two_beacons_learn_as_one_system (set SWARM_PROC_TESTS=1)")
        return
    from swarmint.sim.run_cross_beacon import run
    res = asyncio.run(run(base_port=9631, converge_s=35.0))
    # each beacon learned classes that ONLY existed on the other beacon's swarm
    assert res["A_has_right"], f"beacon A never learned the right side: {res['A_labels']}"
    assert res["B_has_left"], f"beacon B never learned the left side: {res['B_labels']}"
    # and full-set accuracy rose past the ~0.5 half-coverage ceiling
    assert res["A_acc"] > 0.6 and res["B_acc"] > 0.6, (res["A_acc"], res["B_acc"])


if __name__ == "__main__":
    os.environ.setdefault("SWARM_PROC_TESTS", "1")
    test_two_beacons_learn_as_one_system()
    print("ok")
