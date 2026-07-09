"""Real-data regression: the swarm must beat the solo ceiling on sklearn digits
and separate malicious trust. Guards against regressions in the learning path
on genuinely non-Gaussian data. Skips cleanly if scikit-learn isn't installed."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_swarm_beats_solo_on_real_digits():
    try:
        from swarmint.sim import run_digits as rd
    except Exception as e:  # scikit-learn missing
        print(f"skip test_swarm_beats_solo_on_real_digits: {e}")
        return

    X, y = rd.load_normalized_digits(rd.SEED)
    split = int(0.8 * len(y))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    radii = rd.derive_radii(Xtr, ytr, np.random.default_rng(rd.SEED))

    malicious_ids = {0, 1}
    nodes, assign, by_class, bus = rd.build_swarm(Xtr, ytr, radii, malicious_ids)
    honest = [n for n in nodes if not n.malicious]
    streams = [np.random.default_rng(rd.SEED * 7 + i) for i in range(rd.N_NODES)]

    for _ in range(80):  # modest rounds for a regression test
        for i in range(rd.N_NODES):
            for _ in range(rd.SAMPLES_PER_ROUND):
                c = int(streams[i].choice(assign[i]))
                nodes[i].observe(Xtr[int(streams[i].choice(by_class[c]))], c)
        for n in nodes:
            n.gossip_step()
        bus.pump()

    acc = float(np.mean([n.model.accuracy(Xte, yte) for n in honest]))
    solo_ceiling = rd.CLASSES_PER_NODE / rd.N_CLASSES  # each node saw only 2/10 classes
    t_mal = float(np.mean([n.trust[m] for n in honest for m in malicious_ids if m in n.trust] or [0.5]))
    t_hon = float(np.mean([n.trust[p] for n in honest for p in n.trust if p not in malicious_ids] or [0.5]))

    assert acc > solo_ceiling + 0.30, f"swarm {acc:.3f} not clearly above solo ceiling {solo_ceiling}"
    assert t_mal < t_hon, f"malicious trust {t_mal:.3f} not below honest {t_hon:.3f}"


if __name__ == "__main__":
    test_swarm_beats_solo_on_real_digits()
    print("ok  test_swarm_beats_solo_on_real_digits")
    print("all tests passed")
