"""Transport abstraction: every bus (simulated or real) implements this.

Keeping the interface this narrow is what lets SwarmNode stay transport-agnostic:
a node calls bus.send(msg) and registers a handler for inbound messages; it never
knows whether delivery happens via an in-process queue (SimBus) or a real UDP
socket (UdpBus, T4).
"""

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class Message:
    sender: object       # node id (int today; 20-byte identity from T3b on)
    recipient: object
    kind: str             # "protos" | "query" | "response" | "advert" | "pex" | ...
    payload: object


class Bus(Protocol):
    def register(self, node_id: object, handler: Callable[[Message], None]) -> None:
        """Bind a node id to the callback invoked for messages addressed to it."""
        ...

    def send(self, msg: Message) -> None:
        """Enqueue/transmit a message. Never raises on drop — loss is silent,
        matching real UDP semantics."""
        ...
