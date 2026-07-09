"""Synthetic non-IID data: a Gaussian mixture with K classes in d dims.

Each node's stream covers only a small subset of classes, so a solo node
CANNOT learn the full task — any global accuracy gain must come from gossip.
"""

import numpy as np


def make_world(n_classes: int = 10, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, size=(n_classes, dim)).astype(np.float32)
    return centers


def sample(centers: np.ndarray, labels: list[int], n: int, rng: np.random.Generator):
    ys = rng.choice(labels, size=n)
    xs = centers[ys] + rng.normal(0, 0.8, size=(n, centers.shape[1])).astype(np.float32)
    return xs.astype(np.float32), ys


def global_test_set(centers: np.ndarray, n_per_class: int, seed: int = 999):
    rng = np.random.default_rng(seed)
    labels = list(range(centers.shape[0]))
    ys = np.repeat(labels, n_per_class)
    xs = centers[ys] + rng.normal(0, 0.8, size=(len(ys), centers.shape[1])).astype(np.float32)
    return xs.astype(np.float32), ys
