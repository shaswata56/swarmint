"""Phase-1 simulator: does gossip make the swarm smarter than solo nodes?

- N nodes, each seeing a non-IID stream (2 of 10 classes).
- Swarm nodes gossip prototypes over a lossy simulated bus; solo nodes don't.
- A few malicious nodes push random junk prototypes.
- Metrics per round: mean GLOBAL test accuracy (all 10 classes) for swarm vs
  solo, and the honest nodes' mean trust in malicious vs honest peers.

Run:  python -m swarmint.sim.run_sim
"""

import csv
import random
from pathlib import Path

import numpy as np

from ..network.simbus import SimBus
from ..node.swarm_node import SwarmNode
from .data import global_test_set, make_world, sample

N_NODES = 100
N_MALICIOUS = 5
N_CLASSES = 10
CLASSES_PER_NODE = 2
DIM = 16
ROUNDS = 100
SAMPLES_PER_ROUND = 4
SEED = 42


def build_nodes(bus, rng, malicious_ids):
    nodes = []
    for i in range(N_NODES):
        nodes.append(
            SwarmNode(
                node_id=i, topic=1, bus=bus,
                rng=random.Random(SEED * 1000 + i),
                malicious=(i in malicious_ids),
            )
        )
    ids = [n.node_id for n in nodes]
    for n in nodes:
        n.peers = [j for j in ids if j != n.node_id]
    return nodes


def main():
    rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)
    centers = make_world(N_CLASSES, DIM, seed=SEED)
    test_x, test_y = global_test_set(centers, n_per_class=30)

    # Per-node class assignment (shared by swarm and solo arms for fairness).
    assignments = [
        sorted(py_rng.sample(range(N_CLASSES), CLASSES_PER_NODE)) for _ in range(N_NODES)
    ]
    malicious_ids = set(py_rng.sample(range(N_NODES), N_MALICIOUS))

    bus = SimBus(loss_rate=0.05, max_latency_ticks=2, rng=random.Random(SEED))
    swarm = build_nodes(bus, py_rng, malicious_ids)
    solo = build_nodes(SimBus(rng=random.Random(SEED)), py_rng, set())  # never gossips

    honest = [n for n in swarm if not n.malicious]
    rows = []
    per_node_streams = [np.random.default_rng(SEED * 7 + i) for i in range(N_NODES)]

    for r in range(1, ROUNDS + 1):
        # 1. everyone observes local data
        for i in range(N_NODES):
            xs, ys = sample(centers, assignments[i], SAMPLES_PER_ROUND, per_node_streams[i])
            for x, y in zip(xs, ys):
                swarm[i].observe(x, int(y))
                solo[i].observe(x, int(y))

        # 2. swarm gossips; bus delivers by calling each recipient's registered
        # receive() handler directly (push model — same interface a real
        # UdpBus uses; see SWARM-P3 T1).
        for n in swarm:
            n.gossip_step()
        bus.pump()

        # 3. metrics every 5 rounds
        if r % 5 == 0 or r == 1:
            acc_swarm = float(np.mean([n.model.accuracy(test_x, test_y) for n in honest]))
            acc_solo = float(np.mean([
                solo[n.node_id].model.accuracy(test_x, test_y) for n in honest
            ]))
            t_mal = float(np.mean([
                n.trust[m] for n in honest for m in malicious_ids if m in n.trust
            ] or [0.5]))
            t_hon = float(np.mean([
                n.trust[p] for n in honest for p in n.trust if p not in malicious_ids
            ] or [0.5]))
            rows.append((r, acc_swarm, acc_solo, t_mal, t_hon))
            print(f"round {r:3d} | swarm acc {acc_swarm:.3f} | solo acc {acc_solo:.3f} "
                  f"| trust(malicious) {t_mal:.3f} | trust(honest) {t_hon:.3f}")

    out = Path(__file__).resolve().parents[2] / "results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "swarm_acc", "solo_acc", "trust_malicious", "trust_honest"])
        w.writerows(rows)

    merges_ok = sum(n.merges_ok for n in honest)
    merges_rej = sum(n.merges_rejected for n in honest)
    print(f"\nhonest merges: {merges_ok} committed, {merges_rej} rejected")
    print(f"results written to {out}")

    final = rows[-1]
    verdict = "PASS" if final[1] > final[2] + 0.15 and final[3] < final[4] else "FAIL"
    print(f"\n{verdict}: swarm {final[1]:.3f} vs solo {final[2]:.3f}; "
          f"malicious trust {final[3]:.3f} vs honest trust {final[4]:.3f}")


if __name__ == "__main__":
    main()
