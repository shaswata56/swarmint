"""Simulated network: an in-process message bus with packet loss and latency.

Stands in for UDP/Kademlia in Phase 1-2. Implements the Bus protocol: nodes
register a receive handler, and pump() delivers ready messages by invoking it
directly — the same push model a real UdpBus uses, so SwarmNode's wiring is
identical under both.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .bus import Message


@dataclass
class SimBus:
    loss_rate: float = 0.05
    max_latency_ticks: int = 2
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    _queues: dict = field(default_factory=lambda: defaultdict(list))  # deliver_tick -> [Message]
    _handlers: dict = field(default_factory=dict)  # node_id -> Callable[[Message], None]
    _tick: int = 0

    def register(self, node_id, handler: Callable[[Message], None]) -> None:
        self._handlers[node_id] = handler

    def send(self, msg: Message) -> None:
        if self.rng.random() < self.loss_rate:
            return  # dropped
        delay = self.rng.randint(0, self.max_latency_ticks)
        self._queues[self._tick + delay].append(msg)

    def pump(self) -> None:
        """Advance one tick, delivering every message due now by calling the
        recipient's registered handler directly."""
        for msg in self._queues.pop(self._tick, []):
            handler = self._handlers.get(msg.recipient)
            if handler is not None:
                handler(msg)
        self._tick += 1
