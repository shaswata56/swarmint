"""Scalar logical-clock versioning: (counter, node_id), lexicographic total order.

node_id is opaque (int in the simulator, 20-byte bytes for real identities —
see T3b / decision #023 D5): nothing here does arithmetic on it, only equality
and tuple comparison. Comparison only inspects node_id when two versions have
an EQUAL counter, so id types must stay consistent within one deployment
(never compare an int-id version against a bytes-id version) — enforced by
convention, not at runtime, since the simulator and the real network never
share a Version instance.
"""

from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True, order=True)
class Version:
    counter: int
    node_id: Hashable

    def bump(self, node_id: Hashable) -> "Version":
        return Version(self.counter + 1, node_id)


GENESIS = Version(0, 0)  # sentinel node_id=0; real bumps overwrite it immediately
