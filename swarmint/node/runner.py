"""NodeRunner: the ONLY caller into a SwarmNode once real networking is
involved (decision #023 D7 / plan review gap #3).

Why no locks are needed: SwarmNode's methods (observe, gossip_step, receive)
are fully synchronous — none of them ever `await`. asyncio is single-threaded
and cooperative: a callback or task only yields control at an `await` point.
So as long as every call into a given node's methods happens from asyncio
code on ONE event loop (never from a thread pool), each call runs to
completion before anything else touches that node — no interleaving is
possible, by construction, without any explicit locking. NodeRunner's actual
job is narrower than "prevent races": it's to be the single scheduled task
that drives gossip_step()/observe() on a wall-clock timer, while UdpBus's
datagram callback (also always running on the same loop) drives receive().
Both paths already share one loop; NodeRunner just gives the timer path a
home and a "round" definition for T7's metrics.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .swarm_node import SwarmNode

GOSSIP_INTERVAL_S = 2.0  # decision #023 D7: -> ~0.5 gossip msg/s/node


@dataclass
class NodeRunner:
    node: SwarmNode
    gossip_interval: float = GOSSIP_INTERVAL_S
    data_feed: Optional[Callable[[], object]] = None  # () -> (x, y) | None
    started_at: Optional[float] = field(default=None, init=False)
    gossip_ticks: int = field(default=0, init=False)
    _task: Optional[asyncio.Task] = field(default=None, init=False)
    _stopping: bool = field(default=False, init=False)

    async def start(self) -> None:
        self.started_at = time.time()
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def round_number(self) -> int:
        """Wall-clock round count: one per gossip_interval elapsed since
        start(). This is the "round" T7's convergence/bandwidth windows use —
        there is no global tick under real networking, only elapsed time."""
        if self.started_at is None:
            return 0
        return int((time.time() - self.started_at) // self.gossip_interval)

    async def _loop(self) -> None:
        try:
            while not self._stopping:
                if self.data_feed is not None:
                    sample = self.data_feed()
                    if sample is not None:
                        x, y = sample
                        self.node.observe(x, y)
                self.node.gossip_step()
                self.gossip_ticks += 1
                await asyncio.sleep(self.gossip_interval)
        except asyncio.CancelledError:
            raise
