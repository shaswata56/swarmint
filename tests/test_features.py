"""M1: encoder-free feature map (Random Fourier Features)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.features import ModalityBank, RandomFourierFeatures


def test_seed_reproducible_across_instances():
    """The whole point: two nodes that only share the SEED (+ params) compute
    the identical map — no weights to distribute."""
    a = RandomFourierFeatures(in_dim=5, out_dim=64, gamma=0.5, seed=42)
    b = RandomFourierFeatures(in_dim=5, out_dim=64, gamma=0.5, seed=42)
    x = np.random.default_rng(0).normal(size=5).astype(np.float32)
    assert np.allclose(a.transform(x), b.transform(x))

    c = RandomFourierFeatures(in_dim=5, out_dim=64, gamma=0.5, seed=43)
    assert not np.allclose(a.transform(x), c.transform(x))  # different seed -> different space


def test_output_is_unit_normalized():
    rff = RandomFourierFeatures(in_dim=4, out_dim=128, seed=1)
    X = np.random.default_rng(1).normal(size=(20, 4)).astype(np.float32)
    Z = rff.transform(X)
    assert Z.shape == (20, 128)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)  # -> L2-1NN == cosine-1NN


def test_single_and_batch_consistent():
    rff = RandomFourierFeatures(in_dim=3, out_dim=32, seed=2)
    X = np.random.default_rng(2).normal(size=(5, 3)).astype(np.float32)
    batch = rff.transform(X)
    for i in range(5):
        # float32 matmul accumulates in a different order single-row vs batched,
        # so allow float32-scale tolerance rather than default (float64) rtol.
        assert np.allclose(rff.transform(X[i]), batch[i], atol=1e-5, rtol=1e-4)


def test_similarity_monotonic_in_distance():
    """The property 1-NN relies on: closer raw points -> higher RFF similarity."""
    rff = RandomFourierFeatures(in_dim=4, out_dim=4096, gamma=0.5, seed=3)
    x = np.zeros(4, dtype=np.float32)
    near = np.full(4, 0.1, dtype=np.float32)
    far = np.full(4, 2.0, dtype=np.float32)
    zx, zn, zf = rff.transform(x), rff.transform(near), rff.transform(far)
    sim_near = float(zx @ zn)
    sim_far = float(zx @ zf)
    assert sim_near > sim_far
    assert sim_near > 0.8  # very close points stay very similar


def test_modality_bank_reproducible_and_distinct():
    dims = {0: 3, 1: 5, 2: 2}
    bank_a = ModalityBank(dims, out_dim=64, master_seed=7)
    bank_b = ModalityBank(dims, out_dim=64, master_seed=7)
    x0 = np.random.default_rng(0).normal(size=3).astype(np.float32)
    assert np.allclose(bank_a.transform(0, x0), bank_b.transform(0, x0))  # reproducible
    assert set(bank_a.modalities()) == {0, 1, 2}
    # different modalities use different maps (even at same input dim they'd differ by seed)
    bank_c = ModalityBank(dims, out_dim=64, master_seed=8)
    assert not np.allclose(bank_a.transform(0, x0), bank_c.transform(0, x0))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
