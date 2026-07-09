"""Encoder-free feature maps: Random Fourier Features (RFF).

The multimodal-without-an-encoder answer. An encoder is a *learned* parametric
map (weights you train and must distribute). RFF (Rahimi & Recht 2007) is a
*fixed* random map that approximates an RBF kernel:

    z(x) = sqrt(2/D) * cos(W x + b),   W ~ N(0, 2*gamma), b ~ U(0, 2*pi)
    <z(x), z(y)> ~= exp(-gamma * ||x - y||^2)

It is fully determined by a SEED, so every node in the swarm computes the
identical map from a single integer — no weights to gossip, no training, and
the "shared metric space" property that merge/corroboration depend on holds by
construction (same seed -> same space -> comparable prototypes).

Why it matters for THIS system specifically: swarmint's merge is centroid
averaging under a tight prototype budget. On non-convex classes (XOR-like), the
centroid of two far-apart same-class blobs lands in the wrong region and raw
prototypes collapse. Lifting into RFF space makes such classes near-linearly
separable, so centroid-merging stays coherent — RFF earns its keep exactly
where the swarm's core operation is weakest.

Output is L2-normalized, so downstream 1-NN in the existing (L2) PrototypeModel
behaves as cosine similarity: on unit vectors ||a-b||^2 = 2 - 2<a,b>, so L2 and
cosine rank neighbours identically. That also bounds all distances to [0, 2]
regardless of raw input scale — which retires the data-scale-dependent radius
weakness flagged since Phase 1, for free and with zero change to core code.

Limitation (honest): a random map gives a good generic per-modality space but
NO learned semantic alignment ACROSS modalities. So modalities are kept in
separate spaces (one RFF each = one topic) and fused at the decision level
(mixture-of-experts, Phase 2 + D2), not in a shared embedding — late fusion,
not early fusion.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RandomFourierFeatures:
    """A fixed, seed-determined RBF feature map. Distributable as its four
    scalar params — no weights."""
    in_dim: int
    out_dim: int
    gamma: float = 0.5
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # W: (in_dim, out_dim); RBF bandwidth folded into the sampling scale.
        self._W = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(self.in_dim, self.out_dim)).astype(np.float32)
        self._b = rng.uniform(0.0, 2.0 * np.pi, size=self.out_dim).astype(np.float32)
        self._scale = np.float32(np.sqrt(2.0 / self.out_dim))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map raw features -> normalized RFF space. Accepts a single vector
        (1-D) or a batch (2-D); returns matching rank."""
        x = np.asarray(X, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        z = self._scale * np.cos(x @ self._W + self._b)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        z = z / norms  # unit vectors -> L2-1NN == cosine-1NN, distances in [0, 2]
        return z[0] if single else z


class ModalityBank:
    """One RFF map per modality (each seeded distinctly, possibly different
    input dims). A single master seed reproduces the whole bank across the
    swarm — the entire multimodal 'front end' is one integer + a schema."""

    def __init__(self, modality_dims: dict, out_dim: int = 256, gamma: float = 0.5,
                 master_seed: int = 0):
        # modality_dims: {modality_id: raw_input_dim}
        self.out_dim = out_dim
        self.maps = {
            m: RandomFourierFeatures(in_dim=dim, out_dim=out_dim, gamma=gamma,
                                     seed=master_seed * 1_000_003 + int(m))
            for m, dim in modality_dims.items()
        }

    def transform(self, modality, X: np.ndarray) -> np.ndarray:
        return self.maps[modality].transform(X)

    def modalities(self) -> list:
        return list(self.maps.keys())
