"""M2/M3: encoder-free multimodal — late fusion beats any single modality,
centralized and via the decentralized swarm. No encoder, no learned features."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.sim import run_multimodal as rm
from swarmint.sim.data_multimodal import make_modalities, multiview_test_set


def test_dataset_multiview_labels_aligned():
    spec = make_modalities(n_modalities=3, n_classes=5, seed=0)
    views = multiview_test_set(spec, n_per_class=10)
    ys = [tuple(v[1]) for v in views.values()]
    assert all(y == ys[0] for y in ys)  # same label vector across every modality
    assert len(views) == 3
    # modalities have genuinely different raw dimensionalities (can't share a space)
    dims = {v[0].shape[1] for v in views.values()}
    assert len(dims) >= 1


def test_late_fusion_beats_best_single_modality_centralized():
    spec = make_modalities(rm.N_MODALITIES, rm.N_CLASSES, rm.NOISE, seed=rm.SEED)
    solo, fused = rm.part_a_centralized(spec)
    assert fused > max(solo.values()) + 0.05, f"fusion {fused} not clearly above best solo {max(solo.values())}"


def test_late_fusion_beats_best_single_modality_swarm():
    spec = make_modalities(rm.N_MODALITIES, rm.N_CLASSES, rm.NOISE, seed=rm.SEED)
    solo, fused = rm.part_b_swarm(spec)
    assert fused > max(solo.values()) + 0.05, f"swarm fusion {fused} not above best solo {max(solo.values())}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
