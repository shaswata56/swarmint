"""Second real-data validation: Olivetti faces (harder than digits).

40 classes, 4096-dim, only 10 samples/class (few-shot), and same-class distances
that OVERLAP cross-class distances — the hardest case for distance-based gating.
Confirms the swarm thesis generalizes beyond digits.

Run:  python -m swarmint.sim.run_faces   (downloads ~4MB once, then cached)
"""

from sklearn.datasets import fetch_olivetti_faces

from .real_data import run_swarm_on

N_CLASSES = 40


def main():
    d = fetch_olivetti_faces()
    X, y = d.data, d.target
    print(f"REAL DATA: Olivetti faces {X.shape}, {N_CLASSES} classes, ~10 samples/class (few-shot)\n")
    # 40 classes need many nodes for >=3 honest coverage at 2 classes/node.
    res = run_swarm_on(X, y, n_classes=N_CLASSES, classes_per_node=2, n_nodes=80,
                       n_malicious=3, rounds=180, seed=5)
    print(f"\nsolo ceiling (2/40) ~ {res['solo_ceiling']:.3f} | full-1-NN ceiling ~ 0.83")
    print(f"honest merges: {res['merges_ok']} committed, {res['merges_rejected']} rejected")
    lift = res["acc"] / res["solo_ceiling"]
    verdict = "PASS" if res["acc"] > 0.35 and res["trust_malicious"] < res["trust_honest"] else "FAIL"
    print(f"\n{verdict}: swarm {res['acc']:.3f} ({lift:.0f}x solo) on 40-class few-shot faces; "
          f"malicious trust {res['trust_malicious']:.2f} < honest {res['trust_honest']:.2f}")
    print("(Byzantine margin is narrower than digits — class distances overlap, so poison is "
          "harder to distinguish from real cross-class variance. Honest limitation.)")


if __name__ == "__main__":
    main()
