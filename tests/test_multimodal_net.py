"""Multimodal late fusion over the REAL network stack (UDP + DHT + topics).

run_multimodal.py's test covers fusion over the in-process SimBus. This covers the
same claim over real sockets + Kademlia DHT + msgpack wire + topic-scoped inference
— the transport a live deployment uses. It spawns ~20 NetNodes (loopback, real
sockets) so, like the multiproc tests, it is opt-in via SWARMINT_PROC_TESTS=1 (a
2-core CI runner starves the timing-sensitive DHT convergence).
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.sim import run_multimodal_net as mm


def _proc_tests_enabled() -> bool:
    return os.environ.get("SWARMINT_PROC_TESTS") == "1"


def test_multimodal_fusion_beats_best_modality_over_real_net():
    if not _proc_tests_enabled():
        print("skip test_multimodal_fusion_beats_best_modality_over_real_net "
              "(set SWARMINT_PROC_TESTS=1; spawns ~20 UDP nodes, needs a multi-core host)")
        return
    res = asyncio.run(mm.run(n_modalities=3, n_classes=6, per_modality=10,
                             noise=1.7, converge_s=45.0, seed=11))
    # Fusion of independently-noisy modalities must beat the best single modality,
    # AND every modality must have contributed a real (non-degenerate) accuracy —
    # a guard that topic routing actually reached each modality's experts.
    assert res["fused"] > res["best_solo"], (
        f"fusion {res['fused']:.3f} did not beat best modality {res['best_solo']:.3f}")
    assert all(a > 0.2 for a in res["solo"].values()), \
        f"a modality was not reached by topic routing: {res['solo']}"
    print(f"ok  multimodal-net fusion {res['fused']:.3f} > best {res['best_solo']:.3f}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("done")
