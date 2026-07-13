"""Unit tests for the one-shot inference client's PURE logic (no network):

- backbone_addrs() reconstructs EXACTLY the (node_id, host, port) layout that
  run_backbone assigns, so a query client and the deployed backbone agree with
  zero coordination. This is the invariant the whole `swarmint infer` front door
  rests on — if identities/ports drift from the backbone, every query silently
  hits dead addresses.
- load_query_vectors() always returns query rows already in the shared embedded
  space, whichever input mode (index / eval / raw image) the user picked.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.sim.swarm_backbone import _identity_from_seed, backbone_addrs


def test_backbone_addrs_matches_run_backbone_layout():
    n, base_port, seed_base = 8, 9003, 1000
    addrs = backbone_addrs(n, "203.0.113.7", base_port=base_port, seed_base=seed_base)
    assert len(addrs) == n
    for i, (nid, (host, port)) in enumerate(addrs):
        # identity + port are the SAME pure functions run_backbone uses
        assert nid == _identity_from_seed(seed_base + i).node_id
        assert host == "203.0.113.7"
        assert port == base_port + i
    # deterministic + collision-free
    assert len({nid for nid, _ in addrs}) == n
    assert backbone_addrs(n, "203.0.113.7") == backbone_addrs(n, "203.0.113.7")


def test_load_query_vectors_modes():
    try:
        from swarmint.sim.infer_query import load_query_vectors
        from swarmint.sim.live_task import build_task
    except Exception as e:  # scikit-learn missing
        print(f"skip test_load_query_vectors_modes: {e}")
        return

    task = build_task("digits", n_classes=10, world_seed=0)
    d = task.embedding.out_dim

    # index: one embedded row, true label known
    xs, ys, _ = load_query_vectors(task, index=3)
    assert xs.shape == (1, d) and ys[0] is not None

    # eval: N embedded rows with labels
    xs, ys, _ = load_query_vectors(task, eval_n=5)
    assert xs.shape == (5, d) and all(y is not None for y in ys)

    # image: 64 raw pixels -> normalized + embedded, no label
    p = Path(__file__).parent / "_tmp_digit.txt"
    from sklearn.datasets import load_digits
    dg = load_digits()
    np.savetxt(p, dg.data[0].reshape(1, -1), fmt="%d")
    try:
        xs, ys, _ = load_query_vectors(task, image_path=str(p))
        assert xs.shape == (1, d) and ys == [None]
    finally:
        p.unlink()

    # out-of-range index is rejected, not silently clamped
    try:
        load_query_vectors(task, index=10**9)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_backbone_addrs_matches_run_backbone_layout()
    test_load_query_vectors_modes()
    print("ok")
