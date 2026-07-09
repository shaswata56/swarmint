"""Generic real-data swarm harness — shared by run_digits and run_faces.

Runs the exact swarm machinery (non-IID shards, gossip, corroboration, trust, a
malicious node) on any real (X, y) dataset, with distance radii DERIVED from the
dataset's own same/cross-class distance statistics (not hardcoded — the fix for
the data-scale-dependent-radius weakness). This is what let the thesis be
validated on two very different real datasets:

    digits (10-class, many-shot):  solo ~0.20  ->  swarm ~0.78   (1-NN ceiling 0.93)
    faces  (40-class, few-shot):   solo ~0.05  ->  swarm ~0.51   (1-NN ceiling 0.825)
"""

import random
from collections import Counter

import numpy as np

from ..network.simbus import SimBus
from ..node.swarm_node import SwarmNode


def normalize(X: np.ndarray) -> np.ndarray:
    """Unit-normalize rows: distances land in [0, 2] regardless of raw scale, so
    derived radii are meaningful and comparable across datasets."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def derive_radii(X, y, rng) -> dict:
    """Set radii from the dataset's own same/cross-class distance distribution.
    Tight absorb keeps distinct-but-close prototypes separate; conflict radius
    sits below the typical same-class distance so only a genuinely-too-close
    different-label proto is a contradiction."""
    n = len(y)
    same, alld = [], []
    for _ in range(8000):
        i, j = rng.integers(0, n, 2)
        dist = float(np.linalg.norm(X[i] - X[j]))
        alld.append(dist)
        if y[i] == y[j]:
            same.append(dist)
    sp = float(np.percentile(same, 50)) if same else 0.5
    return {
        "absorb_near": sp * 0.2,
        "absorb_full": sp * 0.35,
        "conflict_radius": float(np.percentile(alld, 5)),
        "corroboration_radius": sp * 1.3,
        "override_trust": 0.6,
        "abstain_margin": 0.02,
    }


def run_swarm_on(X, y, n_classes, classes_per_node=2, n_nodes=40, n_malicious=2,
                 rounds=120, samples_per_round=4, loss_rate=0.05, seed=5, log=True,
                 embedding=None):
    """Train a swarm on real data; return a results dict. Nodes are non-IID
    (each sees `classes_per_node` of `n_classes`), gossip over a lossy SimBus,
    and the first `n_malicious` nodes label-flip. Coverage is guaranteed >=3
    HONEST nodes per class (corroboration needs it).

    `embedding`: a FITTED shared genesis Embedding (see core/embedding.py). When
    given, every node's inputs pass through the SAME frozen projection before the
    prototype model sees them — dim reduction + class separation, applied at the
    data boundary so SwarmNode stays representation-agnostic. When None, raw
    normalized features are used (original behaviour)."""
    y = y.astype(int)
    if embedding is not None:
        X = embedding.transform(X.astype(np.float32))  # shared frozen projection (already normalized)
    else:
        X = normalize(X.astype(np.float32))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    split = int(0.75 * len(y))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    radii = derive_radii(Xtr, ytr, np.random.default_rng(seed))
    malicious = set(range(n_malicious))
    by_class = {c: np.where(ytr == c)[0] for c in range(n_classes)}

    py = random.Random(seed)
    assign = [sorted(py.sample(range(n_classes), classes_per_node)) for _ in range(n_nodes)]
    cov = Counter(c for i, a in enumerate(assign) if i not in malicious for c in a)
    for c in range(n_classes):
        while cov[c] < 3:
            i = py.randrange(n_nodes)
            if i not in malicious and c not in assign[i]:
                assign[i].append(c); cov[c] += 1

    bus = SimBus(loss_rate=loss_rate, rng=random.Random(seed))
    nodes = [SwarmNode(node_id=i, topic=1, bus=bus, rng=random.Random(seed * 100 + i),
                       malicious=(i in malicious), n_classes=n_classes, **radii)
             for i in range(n_nodes)]
    for nd in nodes:
        nd.peers = [j for j in range(n_nodes) if j != nd.node_id]
    honest = [nd for nd in nodes if not nd.malicious]
    streams = [np.random.default_rng(seed * 7 + i) for i in range(n_nodes)]

    if log:
        print(f"{'round':>5} | {'swarm_acc':>9} | {'trust(mal)':>10} | {'trust(hon)':>10}")
    for r in range(1, rounds + 1):
        for i in range(n_nodes):
            for _ in range(samples_per_round):
                c = int(streams[i].choice(assign[i]))
                pool = by_class[c]
                if len(pool):
                    nodes[i].observe(Xtr[int(streams[i].choice(pool))], c)
        for nd in nodes:
            nd.gossip_step()
        bus.pump()
        if log and (r % max(rounds // 6, 1) == 0 or r == 1):
            print(f"{r:>5} | {_acc(honest, Xte, yte):>9.3f} | "
                  f"{_trust(honest, malicious, True):>10.3f} | {_trust(honest, malicious, False):>10.3f}")

    return {
        "acc": _acc(honest, Xte, yte),
        "trust_malicious": _trust(honest, malicious, True),
        "trust_honest": _trust(honest, malicious, False),
        "solo_ceiling": classes_per_node / n_classes,
        "radii": radii,
        "merges_ok": sum(nd.merges_ok for nd in honest),
        "merges_rejected": sum(nd.merges_rejected for nd in honest),
    }


def _acc(honest, Xte, yte):
    return float(np.mean([nd.model.accuracy(Xte, yte) for nd in honest]))


def _trust(honest, malicious, want_malicious):
    vals = [nd.trust[p] for nd in honest for p in nd.trust
            if (p in malicious) == want_malicious]
    return float(np.mean(vals)) if vals else 0.5
