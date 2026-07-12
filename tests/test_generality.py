"""Zone 1 generality regression: the swarm lift must hold on OTHER modalities
(text, tabular), not just images. Each dataset needs a one-time download, so this
is opt-in via SWARMINT_SLOW_TESTS=1 and skips cleanly when offline / sklearn
absent — CI stays fast and network-free (mirrors test_faces)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Per-dataset guard floors. The Zone-1 sweep found the lift is a GRADIENT, not a
# uniform win: strong on tabular, MARGINAL on text (only ~+0.06 over solo, because
# the shared linear LDA genesis embedding barely separates TF-IDF classes — its
# own 1-NN ceiling is just ~0.55). More rounds don't help (structural, not
# under-trained). So these bars guard against a COLLAPSE TO BASELINE / broken
# Byzantine gating — not a strong-performance claim. Headline numbers live in
# run_generality's table; keep these honest and loose.
_FLOORS = {"load_text_ds": 0.36, "load_tabular_ds": 0.40}


def _check(loader_name):
    from swarmint.sim import run_generality as g
    loader = getattr(g, loader_name)
    r = g.run_one(loader_name, loader)
    if r.get("skipped"):
        print(f"skip {loader_name}: {r['skipped']}")
        return
    floor = _FLOORS[loader_name]
    assert r["acc"] > floor, \
        f"{loader_name}: swarm {r['acc']:.3f} collapsed below floor {floor} (solo {r['solo_ceiling']:.2f})"
    assert r["trust_malicious"] < r["trust_honest"], \
        f"{loader_name}: malicious {r['trust_malicious']:.3f} !< honest {r['trust_honest']:.3f}"


def test_generality_text_and_tabular():
    if os.environ.get("SWARMINT_SLOW_TESTS") != "1":
        print("skip test_generality_text_and_tabular (set SWARMINT_SLOW_TESTS=1 to run)")
        return
    try:
        import sklearn  # noqa: F401
    except Exception as e:
        print(f"skip test_generality_text_and_tabular: {e}")
        return
    _check("load_text_ds")
    _check("load_tabular_ds")


if __name__ == "__main__":
    test_generality_text_and_tabular()
    print("ok  test_generality_text_and_tabular")
    print("all tests passed")
