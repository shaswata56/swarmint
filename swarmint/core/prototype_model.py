"""Prototype-based micro-model: a bounded set of labeled exemplars + 1-NN classify.

Merging is order-invariant (set union + prune), so no gradient alignment or
shared init is needed. The gossip "delta" is just prototypes to add.
"""

from dataclasses import dataclass, field

import numpy as np

from .version import GENESIS, Version


@dataclass
class Prototype:
    vector: np.ndarray  # feature vector, float32
    label: int
    weight: float = 1.0  # how many samples this prototype absorbed

    def key(self) -> tuple:
        # Quantized identity for dedup across the network.
        return (self.label, tuple(np.round(self.vector, 4).tolist()))


@dataclass
class PrototypeModel:
    topic: int
    max_prototypes: int = 200
    protos: list = field(default_factory=list)
    version: Version = GENESIS
    accepted: int = 0   # set by merged_with: incoming protos taken
    conflicts: int = 0  # set by merged_with: incoming protos contradicting local ones
    # Distance radii — defaults match the original synthetic-Gaussian tuning so
    # every existing test/sim is unchanged. They are configurable (and should be
    # DERIVED from a dataset's own same/cross-class distance statistics) because
    # the right value is data-scale-dependent — the residual weakness flagged
    # since Phase 1, which real data (normalized digits) exposed immediately.
    absorb_near: float = 0.5    # absorb into nearest same-class proto when closer than this (low fill)
    absorb_full: float = 1.0    # ...and when the budget is >=half full
    conflict_radius: float = 1.5   # different-label proto this close == a contradiction
    override_trust: float = 0.6    # sender trust at/above which a conflict evicts local protos
    # NOTE: per-class ADAPTIVE conflict radii were tried and REVERTED (decision #028):
    # they lowered digit accuracy 0.78->0.75 AND weakened Byzantine defense (tight
    # per-class scales shrink the poison guard). Both fixed and adaptive geometry
    # cap ~0.78, so the remaining gap to full-1-NN is structural, not a radius knob.

    def _cfg(self) -> dict:
        return {"absorb_near": self.absorb_near, "absorb_full": self.absorb_full,
                "conflict_radius": self.conflict_radius, "override_trust": self.override_trust}

    # ---- inference ----

    def predict(self, x: np.ndarray) -> int | None:
        if not self.protos:
            return None
        mat = np.stack([p.vector for p in self.protos])
        d = np.linalg.norm(mat - x, axis=1)
        return self.protos[int(np.argmin(d))].label

    def confidence(self, x: np.ndarray) -> float:
        """Margin-style confidence: gap between nearest other-class and nearest same-class."""
        if len(self.protos) < 2:
            return 0.0
        mat = np.stack([p.vector for p in self.protos])
        d = np.linalg.norm(mat - x, axis=1)
        order = np.argsort(d)
        best = self.protos[int(order[0])].label
        d_best = d[order[0]]
        for i in order[1:]:
            if self.protos[int(i)].label != best:
                return float((d[i] - d_best) / (d[i] + d_best + 1e-9))
        return 1.0

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.protos or len(X) == 0:
            return 0.0
        mat = np.stack([p.vector for p in self.protos])
        labels = np.array([p.label for p in self.protos])
        # (n_samples, n_protos) distances
        d = np.linalg.norm(X[:, None, :] - mat[None, :, :], axis=2)
        pred = labels[np.argmin(d, axis=1)]
        return float(np.mean(pred == y))

    # ---- learning ----

    def learn(self, x: np.ndarray, label: int, node_id) -> None:
        """Online insert: absorb into nearest same-class prototype if very close,
        else add a new prototype. Prune when over budget."""
        x = x.astype(np.float32)
        same = [p for p in self.protos if p.label == label]
        if same:
            mat = np.stack([p.vector for p in same])
            d = np.linalg.norm(mat - x, axis=1)
            i = int(np.argmin(d))
            if d[i] < self._absorb_radius():
                p = same[i]
                p.vector = (p.vector * p.weight + x) / (p.weight + 1)
                p.weight += 1
                self.version = self.version.bump(node_id)
                return
        self.protos.append(Prototype(vector=x, label=label))
        self._prune()
        self.version = self.version.bump(node_id)

    def _absorb_radius(self) -> float:
        # Heuristic: tighter absorption as the budget fills up.
        fill = len(self.protos) / max(self.max_prototypes, 1)
        return self.absorb_near if fill < 0.5 else self.absorb_full

    def _prune(self) -> None:
        """Online k-means-style consolidation: while over budget, merge the two
        nearest same-class prototypes into their weighted centroid."""
        while len(self.protos) > self.max_prototypes:
            by_label: dict[int, list[int]] = {}
            for i, p in enumerate(self.protos):
                by_label.setdefault(p.label, []).append(i)
            best = None  # (dist, i, j)
            for idxs in by_label.values():
                if len(idxs) < 2:
                    continue
                mat = np.stack([self.protos[i].vector for i in idxs])
                d = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                a, b = np.unravel_index(np.argmin(d), d.shape)
                if best is None or d[a, b] < best[0]:
                    best = (d[a, b], idxs[int(a)], idxs[int(b)])
            if best is None:  # every class has one prototype; drop lightest overall
                self.protos.remove(min(self.protos, key=lambda p: p.weight))
                continue
            _, i, j = best
            pi, pj = self.protos[i], self.protos[j]
            w = pi.weight + pj.weight
            pi.vector = (pi.vector * pi.weight + pj.vector * pj.weight) / w
            pi.weight = w
            del self.protos[j]

    # ---- merging ----

    def merged_with(self, incoming: list, node_id,
                    sender_trust: float = 0.0) -> "PrototypeModel":
        """Return a NEW model = union of prototypes, pruned. Caller validates
        before committing (merge is a pure function; rollback = discard).

        Conflict rule: an incoming prototype within conflict_radius of an
        existing one but with a different label is a contradiction. From a
        low-trust sender it is dropped (blocks label-flip poisoning of classes
        the node cannot validate). From a high-trust sender it EVICTS the
        conflicting local prototypes — so truth from reputable peers can
        repair poison that happened to arrive first."""
        m = PrototypeModel(topic=self.topic, max_prototypes=self.max_prototypes, **self._cfg())
        m.protos = [Prototype(p.vector.copy(), p.label, p.weight) for p in self.protos]
        seen = {p.key() for p in m.protos}
        for p in incoming:
            if p.key() in seen:
                continue
            if m.protos:
                mat = np.stack([q.vector for q in m.protos])
                labels = np.array([q.label for q in m.protos])
                near = np.linalg.norm(mat - p.vector, axis=1) < self.conflict_radius
                clash = near & (labels != p.label)
                if np.any(clash):
                    if sender_trust < self.override_trust:
                        m.conflicts += 1
                        continue
                    m.protos = [q for q, c in zip(m.protos, clash) if not c]
            m.protos.append(Prototype(p.vector.copy(), p.label, p.weight))
            m.accepted += 1
        m._prune()
        m.version = self.version.bump(node_id)
        return m
