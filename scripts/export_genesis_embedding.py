"""Export the LIVE digits genesis embedding (world_seed=0) as JSON for the JS
client — the frozen projection every node and now every browser must share
(the shared-space invariant). Deterministic from (seed, bundled sklearn data),
so this is reproducible by anyone, not a secret artifact.

Run:  python scripts/export_genesis_embedding.py > js/swarmint-client/genesis_embedding.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.sim.live_task import build_task


def main():
    task = build_task("digits", n_classes=10, world_seed=0)
    emb = task.embedding
    out = {
        "task": "digits", "world_seed": 0, "n_classes": 10,
        "in_dim": int(emb.mean.shape[0]), "out_dim": int(emb.W.shape[1]),
        "mean": emb.mean.astype(float).tolist(),
        # row-major (in_dim, out_dim), flattened, matching embedding.mjs's indexing
        "W": emb.W.astype(float).flatten().tolist(),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
