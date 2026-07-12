"""Zone 1 — dataset generality sweep: where does prototype-gossip hold/break?

Runs the SAME frozen-genesis-embedding swarm machinery (see real_data.run_swarm_on
+ core.embedding) across datasets of different MODALITY and shape, so we can map
the approach's operating envelope instead of claiming it from two datasets:

    digits  (image,   10-cls, many-shot)  — established baseline
    text    (20news,   6-cls, many-shot)  — TF-IDF features, sparse->dense
    tabular (covtype,  7-cls, many-shot)  — native low-dim numeric features

Every dataset uses the SAME protocol: carve a small stratified PUBLIC genesis seed
(>=~40/class), fit ONE shared LDA embedding on it, freeze, and let the non-IID
swarm (5% malicious label-flippers) learn on the remainder. We report solo ceiling
vs swarm accuracy vs a full-data 1-NN ceiling (the achievable upper bound in the
SAME embedded space), plus the Byzantine trust margin.

Datasets that need a download are SKIPPED (not failed) when offline — the sweep
reports what it could run. Run:  python -m swarmint.sim.run_generality
"""

import time

import numpy as np

from ..core.embedding import LDAEmbedding, genesis_seed_split
from .real_data import run_swarm_on

GENESIS_PER_CLASS = 40   # LDA genesis needs an adequate public seed (>=~40/class)


# ---- dataset loaders: each returns (X float32, y int, n_classes, title) or raises ----

def load_digits_ds():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data.astype(np.float32), d.target.astype(int), 10, "digits (image 64-d, 10-cls)"


def load_text_ds():
    """20 newsgroups, a 6-category many-shot subset. TF-IDF (headers/footers/quotes
    stripped so the model learns topic language, not metadata artifacts) then dense."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    cats = ["rec.sport.hockey", "sci.space", "talk.politics.mideast",
            "comp.graphics", "rec.autos", "sci.med"]
    ng = fetch_20newsgroups(subset="train", categories=cats,
                            remove=("headers", "footers", "quotes"))
    vec = TfidfVectorizer(max_features=2000, sublinear_tf=True, min_df=3)
    X = vec.fit_transform(ng.data).toarray().astype(np.float32)
    y = ng.target.astype(int)
    return X, y, len(cats), f"text 20news (TF-IDF {X.shape[1]}-d, {len(cats)}-cls)"


def load_tabular_ds(per_class_cap=800):
    """Forest cover-type: 54 native numeric features, 7 classes, many-shot. Subsample
    per class (it is 581k rows) and standardize so LDA/distances are scale-sane."""
    from sklearn.datasets import fetch_covtype
    cv = fetch_covtype()
    X, y = cv.data.astype(np.float32), (cv.target.astype(int) - 1)  # -> 0..6
    n_classes = 7
    rng = np.random.default_rng(0)
    keep = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        keep.extend(rng.choice(idx, size=min(per_class_cap, len(idx)), replace=False).tolist())
    keep = np.array(keep)
    X, y = X[keep], y[keep]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = ((X - mu) / sd).astype(np.float32)
    return X, y, n_classes, f"tabular covtype ({X.shape[1]}-d, {n_classes}-cls)"


def _1nn_ceiling(X, y, embedding, seed=5):
    """Full-data 1-NN accuracy in the SAME embedded space — the achievable upper
    bound this representation allows (what the swarm is judged against)."""
    from sklearn.neighbors import KNeighborsClassifier
    Z = embedding.transform(X.astype(np.float32))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    Z, yy = Z[perm], y[perm]
    split = int(0.75 * len(yy))
    knn = KNeighborsClassifier(n_neighbors=1).fit(Z[:split], yy[:split])
    return float(knn.score(Z[split:], yy[split:]))


def run_one(name, loader, n_nodes=40, n_malicious=2, rounds=100):
    try:
        X, y, n_classes, title = loader()
    except Exception as e:
        return {"title": name, "skipped": f"{type(e).__name__}: {str(e)[:80]}"}
    rng = np.random.default_rng(0)
    (Xs, ys), (Xr, yr) = genesis_seed_split(X, y, n_classes, per_class=GENESIS_PER_CLASS, rng=rng)
    emb = LDAEmbedding().fit(Xs, ys)
    t = time.time()
    res = run_swarm_on(Xr, yr, n_classes=n_classes, classes_per_node=2, n_nodes=n_nodes,
                       n_malicious=n_malicious, rounds=rounds, seed=5, log=False, embedding=emb)
    res["ceiling_1nn"] = _1nn_ceiling(X, y, emb)
    res["title"] = title
    res["dims"] = emb.out_dim
    res["sec"] = time.time() - t
    res["seed_size"] = len(ys)
    return res


def main():
    datasets = [
        ("digits", load_digits_ds),
        ("text", load_text_ds),
        ("tabular", load_tabular_ds),
    ]
    print(f"Zone 1 generality sweep — shared LDA genesis embedding ({GENESIS_PER_CLASS}/class seed), "
          f"5% malicious label-flip\n")
    print(f"{'dataset':<34} | {'dim':>3} | {'solo':>4} | {'swarm':>5} | {'1-NN':>5} | "
          f"{'mal_t':>5} | {'hon_t':>5} | {'sec':>4}")
    print("-" * 92)
    rows = []
    for name, loader in datasets:
        r = run_one(name, loader)
        rows.append((name, r))
        if r.get("skipped"):
            print(f"{r['title']:<34} |  SKIPPED ({r['skipped']})")
            continue
        print(f"{r['title']:<34} | {r['dims']:>3} | {r['solo_ceiling']:>4.2f} | {r['acc']:>5.3f} | "
              f"{r['ceiling_1nn']:>5.3f} | {r['trust_malicious']:>5.2f} | {r['trust_honest']:>5.2f} | "
              f"{r['sec']:>4.0f}")
    print("\nRead: swarm should beat the solo ceiling by a wide margin and approach the 1-NN")
    print("ceiling (the representation's own limit); mal_t < hon_t confirms Byzantine gating held.")
    return rows


if __name__ == "__main__":
    main()
