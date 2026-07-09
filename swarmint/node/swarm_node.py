"""SwarmNode: prototype model + local holdout buffer + trust-weighted gossip merge.

Merge policy ("proof of usefulness" on the receiver's own data):
  1. Build a candidate = union(own, peer prototypes), pruned.
  2. Evaluate candidate on the node's local holdout ring buffer.
  3. Commit only if accuracy does not degrade beyond a small epsilon;
     otherwise roll back (discard the candidate).
  4. Trust is an EMA of merge outcomes; low-trust peers are ignored entirely.
"""

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..core.prototype_model import PrototypeModel
from ..network.bus import Bus, Message

TRUST_INIT = 0.5
TRUST_ALPHA = 0.3       # EMA step for merge outcomes
TRUST_FLOOR = 0.15      # below this, peer updates are ignored
DEGRADE_EPS = 0.02      # tolerated holdout-accuracy drop on merge
HOLDOUT_SIZE = 60
QUARANTINE_SIZE = 500   # max buffered unverifiable prototypes
CORROBORATION = 3       # distinct senders required to promote unverifiable knowledge
CORROBORATION_RADIUS = 4.0  # region size for "same claim" — wider than the conflict
                            # radius because within-class scatter is large
ABSTAIN_MARGIN = 0.15   # below this margin-confidence the query is off the model's
                        # manifold; the expert abstains rather than guess. Margin is
                        # a scale-free ratio, so this holds across data scales as long
                        # as classes stay well-separated relative to their spread.


@dataclass
class SwarmNode:
    node_id: object  # opaque hashable: int in the simulator, 20-byte bytes over
                      # the real network (T3b / decision #023 D5) — nothing in
                      # this class does arithmetic on it, only equality/hashing.
    topic: int
    bus: Bus
    rng: random.Random
    max_prototypes: int = 200
    malicious: bool = False
    n_classes: int = 10
    gossip_sample_size: int = 30  # prototypes offered per gossip send. Default 30
                                   # preserves the deterministic simulator exactly;
                                   # the real UDP harness lowers it to fit the
                                   # per-node bandwidth budget (a transport/bandwidth
                                   # knob, not learning logic — smaller just means
                                   # more rounds to converge).
    # Distance radii — defaults match the original synthetic tuning (all existing
    # tests/sims unchanged). Derive these from a dataset's own same/cross-class
    # distance statistics for real data (see sim/run_digits.py).
    absorb_near: float = 0.5
    absorb_full: float = 1.0
    conflict_radius: float = 1.5
    override_trust: float = 0.6
    corroboration_radius: float = CORROBORATION_RADIUS
    abstain_margin: float = ABSTAIN_MARGIN

    model: PrototypeModel = None
    trust: dict = field(default_factory=dict)          # node_id -> float
    holdout_x: deque = field(default_factory=lambda: deque(maxlen=HOLDOUT_SIZE))
    holdout_y: deque = field(default_factory=lambda: deque(maxlen=HOLDOUT_SIZE))
    quarantine: deque = field(default_factory=lambda: deque(maxlen=QUARANTINE_SIZE))
    peers: list = field(default_factory=list)
    merges_ok: int = 0
    merges_rejected: int = 0

    def __post_init__(self):
        self.model = PrototypeModel(
            topic=self.topic, max_prototypes=self.max_prototypes,
            absorb_near=self.absorb_near, absorb_full=self.absorb_full,
            conflict_radius=self.conflict_radius, override_trust=self.override_trust)
        self.bus.register(self.node_id, self.receive)

    # ---- local learning ----

    def observe(self, x: np.ndarray, y: int) -> None:
        """Consume one labeled sample from the local stream."""
        # Roughly 1-in-4 samples are held out for merge validation, never trained on.
        if self.rng.random() < 0.25:
            self.holdout_x.append(x)
            self.holdout_y.append(y)
        else:
            self.model.learn(x, y, self.node_id)

    def _holdout_accuracy(self, model: PrototypeModel) -> float:
        if not self.holdout_x:
            return 0.0
        return model.accuracy(np.stack(list(self.holdout_x)), np.array(self.holdout_y))

    def _corroborate(self, p, sender: int) -> bool:
        """Buffer an unverifiable prototype; promote it once CORROBORATION
        distinct senders have claimed the same label in the same region AND
        that label is the majority claim there (a colluding minority cannot
        out-vote the honest supply)."""
        votes: dict[int, set] = {p.label: {sender}}
        for q, q_sender in self.quarantine:
            if np.linalg.norm(q.vector - p.vector) < self.corroboration_radius:
                votes.setdefault(q.label, set()).add(q_sender)
        self.quarantine.append((p, sender))
        n = len(votes[p.label])
        return n >= CORROBORATION and all(
            n >= len(s) for label, s in votes.items() if label != p.label
        )

    # ---- gossip ----

    def gossip_step(self) -> None:
        """Push prototypes to one random peer (push-style gossip)."""
        if not self.peers or not self.model.protos:
            return
        peer = self.rng.choice(self.peers)
        k = min(self.gossip_sample_size, len(self.model.protos))
        if self.malicious:
            # Targeted poison: the node's REAL prototypes with flipped labels —
            # on-manifold vectors that actively misclassify, unlike random junk.
            sample = self.rng.sample(self.model.protos, k)
            poison = [
                type(p)(vector=p.vector.copy(), label=(p.label + 1) % self.n_classes,
                        weight=p.weight)
                for p in sample
            ]
            self.bus.send(Message(self.node_id, peer, "protos", poison))
        else:
            sample = self.rng.sample(self.model.protos, k)
            self.bus.send(Message(self.node_id, peer, "protos", sample))

    def receive(self, msg: Message) -> None:
        if msg.kind != "protos":
            return
        t = self.trust.get(msg.sender, TRUST_INIT)
        if t < TRUST_FLOOR:
            return  # peer has burned its reputation
        if len(self.holdout_x) < 10:
            return  # not enough local evidence to judge a merge yet

        # Partition: prototypes for classes covered by the local holdout are
        # VERIFIABLE (judged by validation below). Anything else is UNVERIFIABLE
        # and must be corroborated by a second, distinct sender before commit —
        # this stops poison laundering through classes we can't test.
        known = set(self.holdout_y)
        verifiable = [p for p in msg.payload if p.label in known]
        to_merge = verifiable + [
            p for p in msg.payload if p.label not in known
            if self._corroborate(p, msg.sender)
        ]
        if not to_merge:
            return  # everything quarantined; no evidence either way yet

        before = self._holdout_accuracy(self.model)
        candidate = self.model.merged_with(to_merge, self.node_id, sender_trust=t)
        after = self._holdout_accuracy(candidate)

        degraded = after < before - DEGRADE_EPS
        contradictory = candidate.conflicts > candidate.accepted
        if degraded or contradictory:
            self.merges_rejected += 1  # rollback = simply don't commit
            outcome = 0.0
        elif candidate.accepted == 0:
            return  # pure no-op (all duplicates): no commit, no trust signal
        else:
            self.model = candidate  # commit
            self.merges_ok += 1
            outcome = 1.0
        self.trust[msg.sender] = (1 - TRUST_ALPHA) * t + TRUST_ALPHA * outcome

    # ---- distributed inference (mixture of experts) ----

    def answer_query(self, x: np.ndarray):
        """Act as an expert: return (label, confidence) for a peer's query, or
        (None, 0.0) if this node has no relevant knowledge. A malicious node
        lies confidently."""
        pred = self.model.predict(x)
        if pred is None:
            return None, 0.0
        conf = self.model.confidence(x)
        if conf < self.abstain_margin:
            return None, 0.0  # query is off my manifold — abstain, don't guess
        if self.malicious:
            return (pred + 1) % self.n_classes, 1.0  # confident wrong answer
        return pred, conf

    def calibrate_trust(self, directory: dict) -> None:
        """Build peer trust from verifiable evidence: ask each peer to label the
        node's own holdout samples (whose truth we know) and reward/penalise by
        correctness. This is the inference-time analog of merge validation, and
        it is what sinks a confidently-lying malicious expert."""
        if not self.holdout_x:
            return
        xs, ys = list(self.holdout_x), list(self.holdout_y)
        for pid, peer in directory.items():
            if pid == self.node_id:
                continue
            hits = seen = 0
            for x, y in zip(xs, ys):
                label, _ = peer.answer_query(x)
                if label is None:
                    continue  # peer doesn't cover this class; not evidence
                seen += 1
                hits += (label == y)
            if seen:
                t = self.trust.get(pid, TRUST_INIT)
                self.trust[pid] = 0.5 * t + 0.5 * (hits / seen)

    def infer(self, x: np.ndarray, directory: dict, top_n: int = 20,
              fallback_threshold: float = 0.34):
        """Answer a local query by polling trusted topic experts.

        `directory` maps node_id -> SwarmNode (stands in for a DHT lookup of
        peers advertising this topic). Returns (label, ensemble_score).

        Fan-out (top_n) must exceed the topic's expert diversity: with coarse
        topics a single class is served by only a few peers, so a narrow poll
        can miss them entirely. Out-of-domain experts self-suppress via their
        near-zero margin confidence, so over-polling is cheap, not harmful.
        Falls back to the local model when the ensemble is unconfident.
        """
        from ..inference.aggregate import Response, aggregate

        ranked = sorted(
            (p for p in self.peers if p in directory),
            key=lambda p: self.trust.get(p, TRUST_INIT),
            reverse=True,
        )
        responses = []
        for pid in ranked[:top_n]:
            label, conf = directory[pid].answer_query(x)
            responses.append(Response(
                responder=pid, label=label, confidence=conf,
                trust=self.trust.get(pid, TRUST_INIT),
            ))
        label, score = aggregate(responses)
        if label is None or score < fallback_threshold:
            local = self.model.predict(x)  # trust our own model when peers disagree
            if local is not None:
                return local, score
        return label, score
