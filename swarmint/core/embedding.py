"""Shared genesis embedding: a projection applied to every input before the
prototype swarm sees it.

WHY a shared embedding (and the invariant that governs it): a prototype is only
comparable to another if both were produced in the SAME space. So the projection
must be identical across every node — a node that fits its own projection on its
own shard drifts into an incomparable space and merge silently breaks (the same
permuted-space failure that killed weight-averaging in Phase 0). The decentralized
way to get a shared LEARNED projection is the spec's own "public benchmark seeded
at genesis": fit ONCE on that public seed, freeze, and distribute as a protocol
parameter (like the RFF seed or topic schema). Every node applies the identical
frozen transform.

WHY it helps (measured, not assumed):
  * dim reduction -> prototype distance is O(protos x dims); 4096-dim faces -> 39
    dims (LDA) is ~100x cheaper per op.
  * separability -> LDA maximizes between/within-class scatter, pulling classes
    apart in the embedded space, which both raises accuracy AND widens the
    Byzantine margin (poison is easier to tell from real cross-class variance).

DESIGN: fit with sklearn (PCA/LDA) but FREEZE to a plain (mean, W) linear op, so
transform needs no sklearn at inference, the frozen projection is two small arrays
(literally distributable), and manual==sklearn is unit-tested. Output is
L2-normalized -> distances in [0, 2] -> derived radii are meaningful.
"""

from dataclasses import dataclass, field

import numpy as np


def _normalize(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float32)
    single = Z.ndim == 1
    if single:
        Z = Z[None, :]
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z = (Z / norms).astype(np.float32)
    return Z[0] if single else Z


@dataclass
class Embedding:
    """Frozen linear projection: transform(X) = normalize((X - mean) @ W).

    An Identity embedding (mean=0, W=I via a passthrough flag) reproduces the raw
    normalized behaviour, so `embedding=None` and `IdentityEmbedding()` are
    equivalent. Subclasses only implement `fit`; transform/serialization are
    shared and framework-free."""
    mean: np.ndarray = None          # (in_dim,)
    W: np.ndarray = None             # (in_dim, out_dim)
    fitted: bool = False
    _identity: bool = False

    @property
    def out_dim(self):
        return None if self.W is None else self.W.shape[1]

    def fit(self, X, y=None) -> "Embedding":
        raise NotImplementedError

    def transform(self, X) -> np.ndarray:
        if self._identity:
            return _normalize(X)
        if not self.fitted:
            raise RuntimeError("embedding used before genesis fit — a node must "
                               "receive the FROZEN shared projection, never fit its own")
        x = np.asarray(X, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        z = (x - self.mean) @ self.W
        z = _normalize(z)
        return z[0] if single else z

    # ---- distribution: the frozen projection is just two small arrays ----
    def to_state(self) -> dict:
        return {"mean": None if self.mean is None else self.mean.tolist(),
                "W": None if self.W is None else self.W.tolist(),
                "identity": self._identity, "kind": type(self).__name__}

    @staticmethod
    def from_state(state: dict) -> "Embedding":
        e = Embedding()
        e._identity = state["identity"]
        e.mean = None if state["mean"] is None else np.array(state["mean"], dtype=np.float32)
        e.W = None if state["W"] is None else np.array(state["W"], dtype=np.float32)
        e.fitted = e.W is not None
        return e

    # ---- compact binary form for network distribution (float32 blobs) ----
    def to_bytes(self) -> bytes:
        import msgpack
        return msgpack.packb({
            "identity": self._identity,
            "mean": None if self.mean is None else self.mean.astype(np.float32).tobytes(),
            "W": None if self.W is None else self.W.astype(np.float32).tobytes(),
            "W_shape": None if self.W is None else list(self.W.shape),
        }, use_bin_type=True)

    @staticmethod
    def from_bytes(b: bytes) -> "Embedding":
        import msgpack
        d = msgpack.unpackb(b, raw=False)
        e = Embedding()
        e._identity = d["identity"]
        e.mean = None if d["mean"] is None else np.frombuffer(d["mean"], dtype=np.float32).copy()
        if d["W"] is None:
            e.W = None
        else:
            e.W = np.frombuffer(d["W"], dtype=np.float32).reshape(d["W_shape"]).copy()
        e.fitted = e.W is not None
        return e


class IdentityEmbedding(Embedding):
    def __init__(self):
        super().__init__(_identity=True, fitted=True)

    def fit(self, X, y=None):
        return self


@dataclass
class RandomProjectionEmbedding(Embedding):
    """Johnson-Lindenstrauss Gaussian random projection — distance-preserving dim
    reduction. Seed-only (no data fit), so it's trivially shareable; helps PERF,
    not separability (distances, hence class overlap, are approximately preserved)."""
    out_dim: int = 128
    seed: int = 0

    def fit(self, X, y=None):
        in_dim = np.asarray(X).shape[1]
        rng = np.random.default_rng(self.seed)
        self.mean = np.zeros(in_dim, dtype=np.float32)
        self.W = (rng.normal(0, 1.0 / np.sqrt(self.out_dim), size=(in_dim, self.out_dim))
                  ).astype(np.float32)
        self.fitted = True
        return self


@dataclass
class PCAEmbedding(Embedding):
    """Unsupervised variance-preserving projection, fit at genesis. Frozen to
    (mean, components^T)."""
    out_dim: int = 64

    def fit(self, X, y=None):
        from sklearn.decomposition import PCA
        X = np.asarray(X, dtype=np.float32)
        k = min(self.out_dim, X.shape[1], X.shape[0])
        pca = PCA(n_components=k).fit(X)
        self.mean = pca.mean_.astype(np.float32)
        self.W = pca.components_.T.astype(np.float32)   # (in, k): (X-mean)@W == pca.transform
        self.fitted = True
        return self


@dataclass
class LDAEmbedding(Embedding):
    """Supervised discriminant projection, fit at genesis — MAXIMIZES class
    separation (between/within scatter). Reduces to <= n_classes-1 dims. Frozen to
    (xbar, scalings[:, :k]) so (X - xbar) @ W == lda.transform(X)[:, :k]."""
    max_dim: int = None   # cap; default = n_classes - 1

    def fit(self, X, y=None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n_classes = len(np.unique(y))
        k = n_classes - 1 if self.max_dim is None else min(self.max_dim, n_classes - 1)
        lda = LinearDiscriminantAnalysis(n_components=k, solver="svd").fit(X, y)
        self.mean = lda.xbar_.astype(np.float32)
        self.W = lda.scalings_[:, :k].astype(np.float32)
        self.fitted = True
        return self


def genesis_seed_split(X, y, n_classes, per_class, rng):
    """Carve a small stratified PUBLIC seed (per_class samples/class) to fit the
    genesis embedding on, leaving the rest for the swarm. Mirrors the spec's
    'public benchmark seeded at genesis' — the embedding is public shared infra;
    the swarm still learns the online distribution decentrally on the remainder."""
    seed_idx = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        take = min(per_class, len(idx))
        seed_idx.extend(rng.choice(idx, size=take, replace=False).tolist())
    seed_mask = np.zeros(len(y), dtype=bool)
    seed_mask[seed_idx] = True
    return (X[seed_mask], y[seed_mask]), (X[~seed_mask], y[~seed_mask])
