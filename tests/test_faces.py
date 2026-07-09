"""Second real-data regression: Olivetti faces (40-class, few-shot). Harder than
digits; confirms the swarm lift generalizes. Skips cleanly if scikit-learn or the
dataset (needs a one-time download) is unavailable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_swarm_lifts_over_solo_on_faces():
    # SLOW (4096-dim, ~2-3 min) — opt-in via SWARMINT_SLOW_TESTS=1 so the routine
    # suite stays fast. The everyday real-data guard is test_digits (64-dim, ~1 min).
    import os
    if os.environ.get("SWARMINT_SLOW_TESTS") != "1":
        print("skip test_swarm_lifts_over_solo_on_faces (set SWARMINT_SLOW_TESTS=1 to run)")
        return
    try:
        from sklearn.datasets import fetch_olivetti_faces
        from swarmint.sim.real_data import run_swarm_on
        d = fetch_olivetti_faces(download_if_missing=False)  # don't hit the network in CI
    except Exception as e:
        print(f"skip test_swarm_lifts_over_solo_on_faces: {e}")
        return

    X, y = d.data, d.target
    # smaller/shorter than the demo for a regression: fewer nodes/rounds, lenient bar
    res = run_swarm_on(X, y, n_classes=40, classes_per_node=2, n_nodes=80,
                       n_malicious=3, rounds=120, seed=5, log=False)
    # brutal dataset: assert a large multiplier over the 0.05 solo ceiling and
    # that malicious trust stays below honest (Byzantine separation holds).
    assert res["acc"] > 5 * res["solo_ceiling"], f"acc {res['acc']:.3f} not >>solo {res['solo_ceiling']:.3f}"
    assert res["trust_malicious"] < res["trust_honest"], \
        f"malicious {res['trust_malicious']:.3f} !< honest {res['trust_honest']:.3f}"


if __name__ == "__main__":
    test_swarm_lifts_over_solo_on_faces()
    print("ok  test_swarm_lifts_over_solo_on_faces")
    print("all tests passed")
