"""Task definitions for the LIVE swarm — what a joining node actually learns.

The public swarm needs a task that every node can run with NO data transfer and NO
coordination: the data must be either derivable from a shared seed (synthetic) or
bundled identically on every machine (sklearn digits). Both are here.

  synthetic : Gaussian mixture from a shared world seed. Byte-identical to the
              original net_daemon behaviour (no embedding, default radii) — the net
              tests and the deterministic story are unchanged.
  digits    : REAL sklearn handwritten digits (bundled, zero-download). A shared LDA
              GENESIS embedding is fit ONCE on a deterministic public seed (same seed
              + same bundled data on every node => the identical frozen projection,
              satisfying the shared-space invariant without a distribution round-trip),
              and distance radii are derived from the embedded data. This is what makes
              "join the swarm and it learns to recognize real digits" a real demo, not
              a toy.

A Task exposes exactly what net_daemon / swarm_backbone need: n_classes, dim, the
frozen `embedding` (or None), the `radii` dict for SwarmNode, a `feed(classes, rng)`
that yields one raw (x, y), and an embedded held-out `(test_x, test_y)` for scoring.
"""

from dataclasses import dataclass

import numpy as np

from .data import global_test_set, make_world, sample


@dataclass
class Task:
    name: str
    n_classes: int
    dim: int                 # dimension the SwarmNode sees (post-embedding for digits)
    embedding: object        # frozen genesis Embedding, or None (raw features)
    radii: dict              # SwarmNode radii kwargs, or {} for defaults
    test_x: np.ndarray       # already in the space the model stores (embedded for digits)
    test_y: np.ndarray
    _feed: object            # (classes, np_rng) -> (raw_x, y); net applies the embedding

    def feed(self, classes, np_rng):
        return self._feed(classes, np_rng)


def _synthetic(n_classes: int, dim: int, world_seed: int) -> Task:
    centers = make_world(n_classes=n_classes, dim=dim, seed=world_seed)
    test_x, test_y = global_test_set(centers, n_per_class=20)

    def feed(classes, np_rng):
        xs, ys = sample(centers, classes, 1, np_rng)
        return (xs[0], int(ys[0]))

    # embedding=None, radii={} -> identical to the original net_daemon behaviour.
    return Task("synthetic", n_classes, dim, None, {}, test_x, test_y, feed)


def _digits(n_classes: int, world_seed: int) -> Task:
    """Real sklearn digits with a shared LDA genesis embedding + data-derived radii.

    Determinism => shareable: every node fits the LDA on the SAME public seed carved
    from the SAME bundled dataset, so all nodes land in the identical embedded space
    (the projection is a pure function of (seed, data), both shared). The remaining
    data is split into a per-node sampling POOL and a held-out test set."""
    from sklearn.datasets import load_digits

    from ..core.embedding import LDAEmbedding, genesis_seed_split
    from .real_data import derive_radii, normalize

    d = load_digits()
    X = normalize(d.data.astype(np.float32))
    y = d.target.astype(int)

    # Deterministic public genesis seed (40/class) -> frozen LDA projection.
    (Xs, ys), (Xr, yr) = genesis_seed_split(
        X, y, n_classes, per_class=40, rng=np.random.default_rng(world_seed))
    embedding = LDAEmbedding().fit(Xs, ys)

    # Shuffle the remainder deterministically, split into train-pool / test.
    perm = np.random.default_rng(world_seed + 1).permutation(len(yr))
    Xr, yr = Xr[perm], yr[perm]
    split = int(0.8 * len(yr))
    Xtr, ytr, Xte, yte = Xr[:split], yr[:split], Xr[split:], yr[split:]

    # Radii derived in the EMBEDDED space (merges happen there); test embedded too.
    Ztr = embedding.transform(Xtr)
    radii = derive_radii(Ztr, ytr, np.random.default_rng(world_seed))
    test_x = embedding.transform(Xte)
    test_y = yte

    # Feed yields RAW digits by class; the net applies the shared embedding before
    # the model sees them (so a joiner that only holds raw pixels still ends up in
    # the shared space).
    pool = {c: Xtr[ytr == c] for c in range(n_classes)}

    def feed(classes, np_rng):
        c = int(np_rng.choice(classes))
        p = pool[c]
        if len(p) == 0:
            c = int(np_rng.choice([k for k in range(n_classes) if len(pool[k])]))
            p = pool[c]
        return (p[int(np_rng.integers(0, len(p)))], c)

    return Task("digits", n_classes, embedding.out_dim, embedding, radii, test_x, test_y, feed)


def build_task(name: str = "synthetic", n_classes: int = 10, dim: int = 16,
               world_seed: int = 0) -> Task:
    if name == "synthetic":
        return _synthetic(n_classes, dim, world_seed)
    if name == "digits":
        return _digits(n_classes, world_seed)
    raise ValueError(f"unknown task {name!r} (expected 'synthetic' or 'digits')")
