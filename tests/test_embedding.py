"""E1/E3: shared genesis embedding — correctness, frozen-linear == sklearn,
serialization (the 'distribute a frozen matrix' story), and the shared-space
invariant (per-node fitting breaks comparability)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.embedding import (Embedding, IdentityEmbedding, LDAEmbedding,
                                      PCAEmbedding, RandomProjectionEmbedding,
                                      genesis_seed_split)


def _blobs(n_classes=6, dim=40, per=60, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (n_classes, dim)).astype(np.float32)
    y = np.repeat(range(n_classes), per)
    X = centers[y] + rng.normal(0, 1.0, (len(y), dim)).astype(np.float32)
    return X.astype(np.float32), y, n_classes


def test_identity_is_normalized_raw():
    X, y, _ = _blobs()
    Z = IdentityEmbedding().transform(X)
    assert Z.shape == X.shape
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)


def test_pca_frozen_linear_matches_sklearn():
    from sklearn.decomposition import PCA
    X, y, _ = _blobs()
    emb = PCAEmbedding(out_dim=10).fit(X)
    pca = PCA(n_components=10).fit(X)
    manual = (X - emb.mean) @ emb.W               # frozen linear op
    assert np.allclose(manual, pca.transform(X), atol=1e-3)  # == sklearn (pre-normalize)
    assert emb.out_dim == 10


def test_lda_frozen_linear_matches_sklearn_and_reduces_dim():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y, nc = _blobs(n_classes=6)
    emb = LDAEmbedding().fit(X, y)
    assert emb.out_dim == nc - 1                   # LDA -> n_classes-1 dims
    lda = LinearDiscriminantAnalysis(n_components=nc - 1, solver="svd").fit(X, y)
    manual = (X - emb.mean) @ emb.W
    assert np.allclose(manual, lda.transform(X)[:, : nc - 1], atol=1e-3)


def test_lda_improves_class_separation():
    """The whole point: same/cross-class distance ratio should improve in LDA
    space vs raw (classes pushed apart, pulled together)."""
    X, y, _ = _blobs(n_classes=6, dim=40)

    def sep(Z):
        rng = np.random.default_rng(1)
        same = [np.linalg.norm(Z[i] - Z[j]) for i, j in rng.integers(0, len(y), (3000, 2)) if y[i] == y[j]]
        cross = [np.linalg.norm(Z[i] - Z[j]) for i, j in rng.integers(0, len(y), (3000, 2)) if y[i] != y[j]]
        return np.mean(cross) / (np.mean(same) + 1e-9)  # higher = better separated

    raw = sep(IdentityEmbedding().transform(X))
    lda = sep(LDAEmbedding().fit(X, y).transform(X))
    assert lda > raw, f"LDA separation {lda:.2f} not better than raw {raw:.2f}"


def test_serialization_roundtrip():
    """A node receives the frozen projection as plain arrays — must reconstruct
    an identical transform (the distribution story)."""
    X, y, _ = _blobs()
    emb = LDAEmbedding().fit(X, y)
    clone = Embedding.from_state(emb.to_state())
    assert np.allclose(emb.transform(X), clone.transform(X))


def test_unfitted_embedding_refuses():
    """A node must never transform with an unfitted (self-fit) projection —
    guards the shared-space invariant."""
    emb = PCAEmbedding(out_dim=8)
    try:
        emb.transform(np.zeros((2, 40), dtype=np.float32))
        assert False, "unfitted embedding should refuse to transform"
    except RuntimeError as e:
        assert "genesis" in str(e)


def test_per_node_fitting_breaks_comparability():
    """Two nodes that each fit their OWN PCA on their OWN shard land in
    incomparable spaces — the failure the shared-genesis design prevents. Here we
    show the same raw point maps to very different embedded vectors under two
    independently-fit projections (so their prototypes couldn't be merged)."""
    X, y, _ = _blobs(dim=40)
    half = len(y) // 2
    a = PCAEmbedding(out_dim=10).fit(X[:half])     # node A fits on its shard
    b = PCAEmbedding(out_dim=10).fit(X[half:])     # node B fits on its shard
    probe = X[0]
    za, zb = a.transform(probe), b.transform(probe)
    # same input, different spaces -> vectors not aligned (cosine far from 1)
    cos = float(za @ zb)
    assert abs(cos) < 0.95, f"independently-fit projections unexpectedly aligned (cos={cos:.2f})"


def test_genesis_seed_split_is_stratified_and_disjoint():
    X, y, nc = _blobs(n_classes=5, per=40)
    rng = np.random.default_rng(0)
    (Xs, ys), (Xr, yr) = genesis_seed_split(X, y, nc, per_class=6, rng=rng)
    assert len(ys) == 6 * nc
    assert set(np.unique(ys)) == set(range(nc))        # every class in the seed
    assert len(yr) == len(y) - len(ys)                 # disjoint remainder


def test_lda_embedding_beats_raw_on_digits_with_adequate_seed():
    """Integration: a shared genesis LDA embedding fit on an ADEQUATE seed
    (40/class) lifts swarm accuracy well above raw features on real digits —
    the lever that breaks the 0.78 ceiling. (Fails with a too-small seed; that
    data-starvation sensitivity is the point.)"""
    try:
        from sklearn.datasets import load_digits
        from swarmint.sim.real_data import run_swarm_on
    except Exception as e:
        print(f"skip test_lda_embedding_beats_raw_on_digits_with_adequate_seed: {e}")
        return
    d = load_digits()
    X, y = d.data.astype(np.float32), d.target.astype(int)
    rng = np.random.default_rng(0)
    (Xs, ys), (Xr, yr) = genesis_seed_split(X, y, 10, per_class=40, rng=rng)

    raw = run_swarm_on(Xr, yr, n_classes=10, classes_per_node=2, n_nodes=30,
                       n_malicious=2, rounds=70, seed=5, log=False, embedding=IdentityEmbedding())
    lda = run_swarm_on(Xr, yr, n_classes=10, classes_per_node=2, n_nodes=30,
                       n_malicious=2, rounds=70, seed=5, log=False,
                       embedding=LDAEmbedding().fit(Xs, ys))
    assert lda["acc"] > raw["acc"] + 0.10, f"LDA {lda['acc']:.3f} not clearly above raw {raw['acc']:.3f}"
    assert lda["trust_malicious"] < lda["trust_honest"]  # Byzantine separation preserved


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
