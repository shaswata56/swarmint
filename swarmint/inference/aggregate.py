"""Trust-weighted ensemble aggregation for distributed inference.

Pure functions over a list of expert responses — no I/O, no node coupling —
so the voting policy is unit-testable in isolation from the network.
"""

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Response:
    responder: int
    label: int | None      # None = "I have no opinion" (empty/irrelevant model)
    confidence: float      # 0..1 margin-style confidence from the responder
    trust: float           # querier's trust in this responder (weight)


def aggregate(responses: list, min_weight: float = 1e-6):
    """Weighted plurality vote. weight = trust * confidence.

    Returns (label, score) where score is the winning label's share of total
    weight (0..1) — a calibrated confidence for the *ensemble* decision, usable
    against a threshold to decide whether to fall back to local/self inference.
    Returns (None, 0.0) when no expert offered a usable opinion.
    """
    tally: dict[int, float] = defaultdict(float)
    total = 0.0
    for r in responses:
        if r.label is None:
            continue
        w = max(r.trust, 0.0) * max(r.confidence, 0.0)
        if w < min_weight:
            continue
        tally[r.label] += w
        total += w
    if not tally or total <= 0.0:
        return None, 0.0
    label = max(tally, key=tally.get)
    return label, tally[label] / total
