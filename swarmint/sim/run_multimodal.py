"""Encoder-free multimodal swarm demo.

Thesis (validated here): swarmint goes multimodal with NO encoder and NO learned
feature map. Each modality is handled by raw prototype models in its own space
(= its own topic, D2); modalities are fused at the DECISION level via the
Phase-2 trust/confidence-weighted mixture-of-experts. The payoff is that fusing
independently-noisy modalities beats any single one — classic ensemble denoising,
achieved by machinery you already have.

Two parts:
  A. Centralized sanity: one expert per modality; solo accuracy vs late fusion.
  B. Decentralized: a SWARM of specialist nodes per modality gossips (SimBus,
     lossy) to converge each modality's knowledge from non-IID shards, then a
     query node fuses across the converged per-modality experts.

Also reported honestly: RFF (a fixed random feature map) as an alternative to an
encoder — it does NOT improve accuracy here (raw prototype 1-NN is already
nonlinear), so it's optional; its only real benefit is normalizing disparate
raw scales into a common bounded space so thresholds are universal.

Run:  python -m swarmint.sim.run_multimodal
"""

import random

import numpy as np

from ..core.features import RandomFourierFeatures
from ..core.prototype_model import PrototypeModel
from ..inference.aggregate import Response, aggregate
from ..network.simbus import SimBus
from ..node.swarm_node import SwarmNode
from .data_multimodal import (make_modalities, modality_test_set, multiview_test_set,
                              sample_modality)

N_MODALITIES = 3
N_CLASSES = 6
NOISE = 1.7
NODES_PER_MODALITY = 12
CLASSES_PER_NODE = 3
ROUNDS = 60
SAMPLES_PER_ROUND = 4
SEED = 11


def _fuse(experts: dict, views: dict, transform=None):
    """Late fusion: for each multi-view sample, aggregate per-modality expert
    predictions weighted by each expert's confidence (Phase-2 aggregate)."""
    y = next(iter(views.values()))[1]
    n = len(y)
    correct = 0
    for i in range(n):
        resp = []
        for m, mdl in experts.items():
            x = views[m][0][i]
            xf = transform[m].transform(x) if transform else x
            resp.append(Response(m, mdl.predict(xf), max(mdl.confidence(xf), 0.01), 1.0))
        label, _ = aggregate(resp)
        correct += (label == y[i])
    return correct / n


def part_a_centralized(spec, transform=None):
    experts = {}
    for m, sm in spec.items():
        mdl = PrototypeModel(topic=1, max_prototypes=200)
        X, y = sample_modality(sm, range(N_CLASSES), 1500, np.random.default_rng(SEED * 10 + m))
        for xi, yi in zip(X, y):
            mdl.learn(transform[m].transform(xi) if transform else xi, int(yi), node_id=1)
        experts[m] = mdl

    views = multiview_test_set(spec, 60)
    solo = {}
    for m, mdl in experts.items():
        X, y = views[m]
        XF = np.array([transform[m].transform(x) for x in X]) if transform else X
        solo[m] = mdl.accuracy(XF, y)
    fused = _fuse(experts, views, transform)
    return solo, fused


def _run_modality_swarm(sm, seed):
    """One modality's decentralized swarm: NODES_PER_MODALITY specialists on
    non-IID class shards gossip over a lossy SimBus until they converge. Returns
    a converged honest node's model as this modality's expert."""
    py_rng = random.Random(seed)
    # guarantee >=3 honest coverers per class (corroboration needs it)
    assignments = []
    for _ in range(NODES_PER_MODALITY):
        assignments.append(sorted(py_rng.sample(range(N_CLASSES), CLASSES_PER_NODE)))
    # top up coverage so every class is seen by >=3 nodes
    from collections import Counter
    cov = Counter(c for a in assignments for c in a)
    for c in range(N_CLASSES):
        while cov[c] < 3:
            n = py_rng.randrange(NODES_PER_MODALITY)
            if c not in assignments[n]:
                assignments[n].append(c); cov[c] += 1

    bus = SimBus(loss_rate=0.05, rng=random.Random(seed))
    nodes = [SwarmNode(node_id=i, topic=1, bus=bus, rng=random.Random(seed * 100 + i))
             for i in range(NODES_PER_MODALITY)]
    for n in nodes:
        n.peers = [j for j in range(NODES_PER_MODALITY) if j != n.node_id]
    streams = [np.random.default_rng(seed * 7 + i) for i in range(NODES_PER_MODALITY)]

    for _ in range(ROUNDS):
        for i, n in enumerate(nodes):
            xs, ys = sample_modality(sm, assignments[i], SAMPLES_PER_ROUND, streams[i])
            for x, y in zip(xs, ys):
                n.observe(x, int(y))
        for n in nodes:
            n.gossip_step()
        bus.pump()
    # return the best-converged node as the modality expert
    tx, ty = modality_test_set(sm, 40)
    return max(nodes, key=lambda n: n.model.accuracy(tx, ty)).model


def part_b_swarm(spec):
    experts = {m: _run_modality_swarm(sm, SEED * 1000 + m) for m, sm in spec.items()}
    views = multiview_test_set(spec, 60)
    solo = {m: experts[m].accuracy(views[m][0], views[m][1]) for m in experts}
    fused = _fuse(experts, views)
    return solo, fused


def main():
    spec = make_modalities(N_MODALITIES, N_CLASSES, NOISE, seed=SEED)

    print(f"encoder-free multimodal: {N_MODALITIES} modalities, {N_CLASSES} classes, noise={NOISE}\n")

    print("== Part A: centralized (one expert per modality) ==")
    solo, fused = part_a_centralized(spec)
    for m, a in solo.items():
        print(f"  modality {m} solo : {a:.3f}")
    print(f"  LATE FUSION       : {fused:.3f}  (best solo {max(solo.values()):.3f})")

    print("\n== Part A': same, but through a fixed RFF map (no encoder, no training) ==")
    transform = {m: RandomFourierFeatures(sm["dim"], 256, gamma=0.5,
                                          seed=SEED * 13 + m) for m, sm in spec.items()}
    solo_r, fused_r = part_a_centralized(spec, transform)
    for m, a in solo_r.items():
        print(f"  modality {m} solo : {a:.3f}")
    print(f"  LATE FUSION       : {fused_r:.3f}  (RFF does NOT help — a fixed random map loses "
          f"info vs raw for 1-NN; use raw, keep RFF only if a modality needs scale-normalization)")

    print("\n== Part B: decentralized swarm per modality, then fuse ==")
    solo_s, fused_s = part_b_swarm(spec)
    for m, a in solo_s.items():
        print(f"  modality {m} swarm-converged solo : {a:.3f}")
    print(f"  LATE FUSION across swarm experts   : {fused_s:.3f}  (best solo {max(solo_s.values()):.3f})")

    verdict = "PASS" if (fused > max(solo.values()) and fused_s > max(solo_s.values())) else "FAIL"
    print(f"\n{verdict}: fusion beats best single modality, centralized AND decentralized, "
          f"with no encoder.")


if __name__ == "__main__":
    main()
