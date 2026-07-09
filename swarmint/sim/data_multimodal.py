"""Multimodal synthetic data for the encoder-free validation.

Design finding (verified, not assumed): raw prototype 1-NN is already nonlinear
and robust — it solves XOR-type tasks at a tiny budget because absorb/prune keep
well-separated same-class blobs as distinct prototypes. So a random feature map
(RFF) does NOT improve classification here, and no encoder is needed for
nonlinearity. The real multimodal question is therefore not "how do we get a
good feature space" but "how do we FUSE modalities" — so this dataset is built
to make fusion the decisive factor.

Each modality is an independent, individually-NOISY view of the same label:
one modality alone classifies only ~moderately (blobs overlap), but the noise
is independent across modalities, so aggregating per-modality expert votes
(Phase-2 mixture-of-experts + D2 topic routing) denoises — late fusion beats
any single modality. Modalities differ in raw dimensionality, so they can't
share one prototype space anyway; each is its own topic. No encoder involved.
"""

import numpy as np


def make_modalities(n_modalities: int = 3, n_classes: int = 6, noise: float = 1.6,
                    seed: int = 0) -> dict:
    """Schema shared by the sampler + the swarm: per modality a raw dim, its own
    class centers, and a noise level chosen so a single modality is weak."""
    rng = np.random.default_rng(seed)
    spec = {}
    for m in range(n_modalities):
        dim = int(rng.integers(3, 7))
        centers = rng.normal(0, 3.0, size=(n_classes, dim)).astype(np.float32)
        spec[m] = {"dim": dim, "centers": centers, "noise": noise, "n_classes": n_classes}
    return spec


def sample_modality(spec_m: dict, labels_wanted, n: int, rng: np.random.Generator):
    centers, noise = spec_m["centers"], spec_m["noise"]
    wanted = list(int(l) for l in labels_wanted)
    ys = rng.choice(wanted, size=n)
    xs = centers[ys] + rng.normal(0, noise, size=(n, centers.shape[1])).astype(np.float32)
    return xs.astype(np.float32), ys


def modality_test_set(spec_m: dict, n_per_class: int, seed: int = 999):
    rng = np.random.default_rng(seed)
    n_classes = spec_m["n_classes"]
    ys = np.repeat(range(n_classes), n_per_class)
    xs = spec_m["centers"][ys] + rng.normal(0, spec_m["noise"],
                                            size=(len(ys), spec_m["centers"].shape[1])).astype(np.float32)
    return xs.astype(np.float32), ys


def multiview_test_set(spec: dict, n_per_class: int, seed: int = 777):
    """Joint test set: each labeled entity gets one INDEPENDENT noisy view per
    modality (same label). Returns {modality: (X, y)} with y aligned across
    modalities — the substrate for measuring late fusion vs single modality."""
    rng = np.random.default_rng(seed)
    any_m = next(iter(spec.values()))
    n_classes = any_m["n_classes"]
    labels = np.repeat(range(n_classes), n_per_class)
    views = {}
    for m, spec_m in spec.items():
        xs = spec_m["centers"][labels] + rng.normal(
            0, spec_m["noise"], size=(len(labels), spec_m["centers"].shape[1])).astype(np.float32)
        views[m] = (xs.astype(np.float32), labels.copy())
    return views
