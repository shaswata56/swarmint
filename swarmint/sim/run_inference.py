"""Phase-2 demo: distributed inference as a mixture of experts.

Isolates the inference benefit from the gossip-convergence benefit (Phase 1):
here nodes do NOT gossip, so each stays a specialist that only knows 2 of 10
classes. A "generalist" query node also knows only 2 classes locally, yet must
answer queries spanning all 10.

Compares three inference strategies for the query node on a global test set:
  - local-only          : the query node's own model (can't know 8 classes)
  - ensemble (naive)     : unweighted majority vote over 5 random peers
  - ensemble (trust)     : trust-weighted vote after calibration

A fraction of experts are malicious and answer confidently WRONG, to show the
trust weighting suppresses them where naive voting does not.

Run:  python -m swarmint.sim.run_inference
"""

import random

import numpy as np

from ..inference.aggregate import Response, aggregate
from ..network.simbus import SimBus
from ..node.swarm_node import TRUST_INIT, SwarmNode
from .data import global_test_set, make_world, sample

N_EXPERTS = 50
N_MALICIOUS = 15
N_CLASSES = 10
CLASSES_PER_NODE = 2
DIM = 16
TRAIN_SAMPLES = 300
SEED = 7


def naive_infer(query: SwarmNode, x, directory, top_n=5):
    """Unweighted majority vote over top_n peers (trust set to 1 for all)."""
    peers = list(directory.keys())[:top_n]
    responses = []
    for pid in peers:
        label, conf = directory[pid].answer_query(x)
        responses.append(Response(pid, label, 1.0, 1.0))  # ignore trust & conf
    label, score = aggregate(responses)
    return label


def main():
    py_rng = random.Random(SEED)
    centers = make_world(N_CLASSES, DIM, seed=SEED)
    test_x, test_y = global_test_set(centers, n_per_class=30)
    streams = [np.random.default_rng(SEED * 3 + i) for i in range(N_EXPERTS + 1)]

    malicious_ids = set(py_rng.sample(range(N_EXPERTS), N_MALICIOUS))
    experts = {}
    for i in range(N_EXPERTS):
        n = SwarmNode(node_id=i, topic=1, bus=SimBus(rng=random.Random(0)),
                      rng=random.Random(SEED * 100 + i), malicious=(i in malicious_ids))
        classes = sorted(py_rng.sample(range(N_CLASSES), CLASSES_PER_NODE))
        xs, ys = sample(centers, classes, TRAIN_SAMPLES, streams[i])
        for x, y in zip(xs, ys):
            n.model.learn(x, int(y), n.node_id)
        experts[i] = n

    # Ensure every class is covered by at least one honest expert (otherwise the
    # task is simply unanswerable and the comparison is meaningless).
    covered = set()
    for i, n in experts.items():
        if i not in malicious_ids:
            covered |= {p.label for p in n.model.protos}
    assert covered == set(range(N_CLASSES)), f"honest experts miss {set(range(N_CLASSES)) - covered}"

    # The query node: a generalist that locally knows only 2 classes.
    query = SwarmNode(node_id=999, topic=1, bus=SimBus(rng=random.Random(0)),
                      rng=random.Random(1), malicious=False)
    q_classes = [0, 1]
    xs, ys = sample(centers, q_classes, TRAIN_SAMPLES, streams[-1])
    for x, y in zip(xs, ys):
        query.model.learn(x, int(y), query.node_id)
        query.observe(x, int(y))  # populate holdout for calibration
    query.peers = list(experts.keys())

    def eval_strategy(fn):
        return float(np.mean([fn(test_x[i]) == test_y[i] for i in range(len(test_y))]))

    acc_local = eval_strategy(lambda x: query.model.predict(x))
    acc_naive = eval_strategy(lambda x: naive_infer(query, x, experts))

    # Trust-weighted: calibrate first, then infer.
    query.calibrate_trust(experts)
    acc_trust = eval_strategy(lambda x: query.infer(x, experts)[0])

    mal_trust = np.mean([query.trust.get(i, TRUST_INIT) for i in malicious_ids])
    hon_trust = np.mean([query.trust.get(i, TRUST_INIT)
                         for i in experts if i not in malicious_ids])

    print(f"experts: {N_EXPERTS} ({N_MALICIOUS} malicious, confidently wrong)")
    print(f"query node knows locally: classes {q_classes} of {N_CLASSES}\n")
    print(f"local-only inference       : {acc_local:.3f}")
    print(f"ensemble, naive vote       : {acc_naive:.3f}")
    print(f"ensemble, trust-weighted   : {acc_trust:.3f}")
    print(f"\ncalibrated trust: malicious {mal_trust:.3f} vs honest {hon_trust:.3f}")

    verdict = ("PASS" if acc_trust > acc_local + 0.3 and acc_trust > acc_naive
               and mal_trust < hon_trust else "FAIL")
    print(f"\n{verdict}: trust-ensemble {acc_trust:.3f} beats local {acc_local:.3f} "
          f"and naive {acc_naive:.3f}; malicious downweighted")
    return {"acc_local": acc_local, "acc_naive": acc_naive, "acc_trust": acc_trust,
            "trust_malicious": float(mal_trust), "trust_honest": float(hon_trust),
            "n_experts": N_EXPERTS, "n_malicious": N_MALICIOUS}


if __name__ == "__main__":
    main()
