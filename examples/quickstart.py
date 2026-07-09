"""swarmint quickstart — the smallest end-to-end demonstration.

A tiny swarm where each node sees only 2 of 6 classes, gossips over a simulated
lossy network, and collectively learns the whole task — beating what any single
node could do alone. Needs only numpy (`pip install -e .`).

    python examples/quickstart.py
"""

import random

import numpy as np

from swarmint.network.simbus import SimBus
from swarmint.node.swarm_node import SwarmNode

N_NODES, N_CLASSES, DIM, ROUNDS = 12, 6, 8, 40


def make_data(rng):
    centers = rng.normal(0, 3.0, (N_CLASSES, DIM)).astype(np.float32)

    def sample(cls):
        return (centers[cls] + rng.normal(0, 0.6, DIM)).astype(np.float32)

    test_x = np.stack([sample(c) for c in range(N_CLASSES) for _ in range(30)])
    test_y = np.array([c for c in range(N_CLASSES) for _ in range(30)])
    return sample, test_x, test_y


def main():
    rng = np.random.default_rng(0)
    sample, test_x, test_y = make_data(rng)

    # One shared simulated network; every node registers on it.
    bus = SimBus(loss_rate=0.05, rng=random.Random(0))
    nodes = [SwarmNode(node_id=i, topic=1, bus=bus, rng=random.Random(i), n_classes=N_CLASSES)
             for i in range(N_NODES)]
    for n in nodes:
        n.peers = [j for j in range(N_NODES) if j != n.node_id]

    # Each node is a specialist: it only ever sees 2 of the 6 classes.
    py = random.Random(1)
    assigned = [sorted(py.sample(range(N_CLASSES), 2)) for _ in range(N_NODES)]

    # Honest baseline: a lone node trains on its own 2 classes but never gossips.
    # It can only ever recognize those 2 of 6 classes -> ~0.33 ceiling.
    solo = SwarmNode(node_id=999, topic=1, bus=SimBus(rng=random.Random(0)),
                     rng=random.Random(9), n_classes=N_CLASSES)
    for _ in range(ROUNDS):
        for _ in range(4):
            c = py.choice(assigned[0])
            solo.observe(sample(c), c)
    solo_acc = solo.model.accuracy(test_x, test_y)

    for _ in range(ROUNDS):
        for i, n in enumerate(nodes):
            for _ in range(4):
                c = py.choice(assigned[i])
                n.observe(sample(c), c)
        for n in nodes:
            n.gossip_step()   # push prototypes to a random peer
        bus.pump()            # deliver messages -> receivers validate + merge

    swarm_acc = np.mean([n.model.accuracy(test_x, test_y) for n in nodes])

    print(f"each node sees only 2 of {N_CLASSES} classes  (solo ceiling ~ {2/N_CLASSES:.2f})")
    print(f"solo node (no gossip)  : {solo_acc:.3f}")
    print(f"swarm node (gossip)    : {swarm_acc:.3f}")
    print(f"\n=> gossip pooled the specialists into a full-task model "
          f"({solo_acc:.2f} -> {swarm_acc:.2f}), with no central server.")


if __name__ == "__main__":
    main()
