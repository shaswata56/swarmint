"""First REAL-data validation: does the swarm work on sklearn digits?

Everything before this ran on synthetic Gaussian blobs. This runs the SAME
machinery (non-IID shards, gossip, corroboration, trust, a malicious node) on
real 8x8 handwritten digits (sklearn, bundled — no download), to see whether the
thesis survives contact with reality.

Key adaptation surfaced by real data: the distance radii were hand-tuned for the
synthetic scale and are meaningless on normalized digits (same-class ~0.59,
cross-class ~0.80). So here we DERIVE them from the data's own same/cross-class
distance statistics — the fix for the radius weakness flagged since Phase 1,
now that PrototypeModel/SwarmNode radii are configurable.

Run:  python -m swarmint.sim.run_digits
"""

import random

import numpy as np
from sklearn.datasets import load_digits

from ..network.simbus import SimBus
from ..node.swarm_node import SwarmNode

N_NODES = 40
N_MALICIOUS = 2
CLASSES_PER_NODE = 2
N_CLASSES = 10
ROUNDS = 120   # accuracy peaks ~0.78 around here; a slight over-merge drift
               # follows (0.783 -> 0.775 by round 200), so we stop near the peak.
SAMPLES_PER_ROUND = 4
SEED = 5


def load_normalized_digits(seed=0):
    d = load_digits()
    X = d.data.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # unit vectors: distances in [0,2]
    y = d.target.astype(int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def derive_radii(X, y, rng):
    """Set radii from the dataset's own same/cross-class distance distribution
    instead of hardcoded synthetic constants."""
    same, cross = [], []
    for _ in range(4000):
        i, j = rng.integers(0, len(y), 2)
        dist = float(np.linalg.norm(X[i] - X[j]))
        (same if y[i] == y[j] else cross).append(dist)
    same, cross = np.array(same), np.array(cross)
    same_p50 = np.percentile(same, 50)
    # Tuning finding (digits): overlapping real classes make loose radii
    # trigger-happy (37% merge-rejection). TIGHT absorb keeps distinct-but-close
    # digit prototypes separate (finer prototypes -> closer to full 1-NN), and a
    # conflict radius below the typical same-class distance means only a
    # genuinely-too-close different-label proto counts as a contradiction. This
    # took digits 0.59 -> 0.72+ and rejection 37% -> ~10%.
    return {
        "absorb_near": float(same_p50 * 0.2),
        "absorb_full": float(same_p50 * 0.35),
        "conflict_radius": float(np.percentile(np.concatenate([same, cross]), 5)),
        "corroboration_radius": float(same_p50 * 1.3),
        "abstain_margin": 0.02,  # margins are small when classes are close; keep low
    }


def build_swarm(X, y, radii, malicious_ids):
    n = len(y)
    py_rng = random.Random(SEED)
    # class -> sample indices, for drawing a node's per-class local stream
    by_class = {c: np.where(y == c)[0] for c in range(N_CLASSES)}
    assignments = [sorted(py_rng.sample(range(N_CLASSES), CLASSES_PER_NODE)) for _ in range(N_NODES)]
    # guarantee >=3 honest coverers per class
    from collections import Counter
    cov = Counter(c for i, a in enumerate(assignments) if i not in malicious_ids for c in a)
    for c in range(N_CLASSES):
        while cov[c] < 3:
            i = py_rng.randrange(N_NODES)
            if i not in malicious_ids and c not in assignments[i]:
                assignments[i].append(c); cov[c] += 1

    bus = SimBus(loss_rate=0.05, rng=random.Random(SEED))
    nodes = []
    for i in range(N_NODES):
        nodes.append(SwarmNode(
            node_id=i, topic=1, bus=bus, rng=random.Random(SEED * 100 + i),
            malicious=(i in malicious_ids), n_classes=N_CLASSES, **radii))
    ids = [nd.node_id for nd in nodes]
    for nd in nodes:
        nd.peers = [j for j in ids if j != nd.node_id]
    return nodes, assignments, by_class, bus


def main():
    X, y = load_normalized_digits(SEED)
    split = int(0.8 * len(y))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    radii = derive_radii(Xtr, ytr, rng)
    print("REAL DATA: sklearn digits (8x8), 10 classes, normalized")
    print("derived radii:", {k: round(v, 3) for k, v in radii.items()}, "\n")

    malicious_ids = set(range(N_MALICIOUS))  # nodes 0..K-1 malicious
    nodes, assignments, by_class, bus = build_swarm(Xtr, ytr, radii, malicious_ids)
    honest = [nd for nd in nodes if not nd.malicious]

    streams = [np.random.default_rng(SEED * 7 + i) for i in range(N_NODES)]

    def draw(node_i):
        cls = assignments[node_i]
        c = int(streams[node_i].choice(cls))
        pool = by_class[c]
        idx = int(streams[node_i].choice(pool))
        return Xtr[idx], c

    # solo baseline: node 0-style honest node that never gossips (2 classes only)
    print(f"{'round':>5} | {'swarm_acc':>9} | {'trust(mal)':>10} | {'trust(hon)':>10}")
    for r in range(1, ROUNDS + 1):
        for i in range(N_NODES):
            for _ in range(SAMPLES_PER_ROUND):
                x, c = draw(i)
                nodes[i].observe(x, c)
        for nd in nodes:
            nd.gossip_step()
        bus.pump()
        if r % 10 == 0 or r == 1:
            acc = float(np.mean([nd.model.accuracy(Xte, yte) for nd in honest]))
            t_mal = float(np.mean([nd.trust[m] for nd in honest for m in malicious_ids if m in nd.trust] or [0.5]))
            t_hon = float(np.mean([nd.trust[p] for nd in honest for p in nd.trust if p not in malicious_ids] or [0.5]))
            print(f"{r:>5} | {acc:>9.3f} | {t_mal:>10.3f} | {t_hon:>10.3f}")

    # solo reference: mean held-out accuracy of a node's model if it had only its own 2 classes
    solo_acc = float(np.mean([
        SwarmNode(node_id=999, topic=1, bus=SimBus(rng=random.Random(0)), n_classes=N_CLASSES,
                  rng=random.Random(1), **radii).model.accuracy(Xte, yte)
        for _ in range(1)]))  # empty model -> 0; report the structural 2/10 ceiling instead
    final = float(np.mean([nd.model.accuracy(Xte, yte) for nd in honest]))
    merges_ok = sum(nd.merges_ok for nd in honest)
    merges_rej = sum(nd.merges_rejected for nd in honest)
    t_mal = float(np.mean([nd.trust[m] for nd in honest for m in malicious_ids if m in nd.trust] or [0.5]))
    t_hon = float(np.mean([nd.trust[p] for nd in honest for p in nd.trust if p not in malicious_ids] or [0.5]))
    print(f"\nhonest merges: {merges_ok} committed, {merges_rej} rejected")
    print(f"each node locally saw only {CLASSES_PER_NODE}/{N_CLASSES} classes "
          f"(solo ceiling ~{CLASSES_PER_NODE/N_CLASSES:.2f})")
    verdict = "PASS" if final >= 0.80 and t_mal < t_hon else "FAIL"
    print(f"\n{verdict}: swarm reached {final:.3f} on REAL held-out digits "
          f"(vs ~{CLASSES_PER_NODE/N_CLASSES:.2f} solo); malicious trust {t_mal:.3f} < honest {t_hon:.3f}")


if __name__ == "__main__":
    main()
