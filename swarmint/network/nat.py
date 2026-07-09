"""NAT traversal via rendezvous-coordinated UDP hole-punching (deferred item D3).

Two peers A and B behind their own NATs can't dial each other directly: neither
NAT has an inbound mapping for the other. The standard fix (used by WebRTC/ICE,
libp2p, BitTorrent) is *simultaneous open* coordinated by a mutually-reachable
rendezvous node:

  1. Reflexive discovery (STUN-lite): each peer asks the rendezvous "what
     source address did you see me from?" — that reflexive (host,port) is the
     peer's NAT-mapped public endpoint, which differs from its local bind addr.
  2. Signaling: A asks the rendezvous to connect it to B. The rendezvous relays
     a punch-request to B carrying A's reflexive addr, and replies to A with
     B's reflexive addr.
  3. Punch: A and B both fire UDP packets at each other's reflexive addr. The
     outbound packet opens each NAT's mapping, so the (near-simultaneous)
     inbound packet finds an open hole and gets through. First packet through
     → the mapping is established and normal traffic flows.

This rides the existing signed UdpBus: punch packets are ordinary signed
envelopes, so the id<->pubkey binding still holds and a rendezvous can't inject
a forged peer (it can only *suggest* an address to dial).

TESTABILITY NOTE: on loopback there is no NAT to open, so packets flow directly
regardless — the loopback test therefore validates the full SIGNALING PROTOCOL
(reflexive discovery, connect relay, punch/ack handshake, and that both sides
end up able to talk directly) but NOT the mapping-opening behaviour itself,
which by definition requires two real NATs (or a simulated NAT layer). The
protocol implemented here is the same one that opens real NATs.
"""

import asyncio

from . import wire
from .bus import Message


class HolePuncher:
    """Attaches NAT-traversal signaling to a node's UdpBus."""

    def __init__(self, identity, bus, discovery=None):
        self.identity = identity
        self.bus = bus
        self.discovery = discovery          # rendezvous uses it to resolve targets' reflexive addrs
        self.reflexive = None               # our own observed (host, port), filled by discover_reflexive
        self.reachable = set()              # peer ids we've completed a punch with
        self._whoami_fut = None
        self._punch_futs = {}               # peer_id -> Future set when a punch/ack arrives
        bus.register_kind("nat", wire.pack_nat, wire.unpack_nat)

    # ---- message dispatch (call from the node's bus handler for kind == "nat") ----

    def handle(self, msg: Message) -> None:
        op = msg.payload.get("op")
        if op == "whoami":
            # We are acting as rendezvous: echo the sender's observed source
            # addr (captured by UdpBus into peer_addrs on receipt).
            src = self.bus.peer_addrs.get(msg.sender)
            if src is not None:
                self.bus.send(Message(self.identity.node_id, msg.sender, "nat",
                                      {"op": "whoami_resp", "addr": f"{src[0]}:{src[1]}"}))
        elif op == "whoami_resp":
            if self._whoami_fut is not None and not self._whoami_fut.done():
                self._whoami_fut.set_result(_parse_addr(msg.payload["addr"]))
        elif op == "connect_req":
            self._relay_connect(msg)
        elif op == "connect_resp":
            self._begin_punch(msg.payload["target_id"], _parse_addr(msg.payload["target_addr"]))
        elif op == "punch_req":
            self._begin_punch(msg.payload["initiator_id"], _parse_addr(msg.payload["initiator_addr"]))
        elif op == "punch":
            # A hole-punch packet arrived → the mapping is open. Ack it and
            # mark the peer reachable.
            self.bus.peer_addrs.setdefault(msg.sender, self.bus.peer_addrs.get(msg.sender))
            self._mark_reachable(msg.sender)
            self.bus.send(Message(self.identity.node_id, msg.sender, "nat", {"op": "punch_ack"}))
        elif op == "punch_ack":
            self._mark_reachable(msg.sender)

    # ---- rendezvous role ----

    def _relay_connect(self, msg: Message) -> None:
        """Rendezvous relays A's connect request toward target B: tell B to
        punch back to A (with A's reflexive addr), and tell A B's addr."""
        target_id = msg.payload["target_id"]
        a_addr = self.bus.peer_addrs.get(msg.sender)          # A's reflexive addr (we saw it)
        b_addr = None
        if self.discovery is not None:
            b_addr = self.discovery.peer_addrs.get(target_id)
        b_addr = b_addr or self.bus.peer_addrs.get(target_id)
        if a_addr is None or b_addr is None:
            return  # can't coordinate without both reflexive addresses
        # tell B to punch to A
        self.bus.peer_addrs[target_id] = b_addr
        self.bus.send(Message(self.identity.node_id, target_id, "nat",
                              {"op": "punch_req", "initiator_id": msg.sender,
                               "initiator_addr": f"{a_addr[0]}:{a_addr[1]}"}))
        # tell A where B is
        self.bus.send(Message(self.identity.node_id, msg.sender, "nat",
                              {"op": "connect_resp", "target_id": target_id,
                               "target_addr": f"{b_addr[0]}:{b_addr[1]}"}))

    # ---- initiator / responder punch ----

    def _begin_punch(self, peer_id, peer_addr) -> None:
        self.bus.peer_addrs[peer_id] = peer_addr  # now we can dial the reflexive addr
        # Fire several punch packets: the first to traverse opens the mapping;
        # extras cover the case where our packet arrives before theirs.
        for _ in range(5):
            self.bus.send(Message(self.identity.node_id, peer_id, "nat", {"op": "punch"}))

    def _mark_reachable(self, peer_id) -> None:
        self.reachable.add(peer_id)
        fut = self._punch_futs.get(peer_id)
        if fut is not None and not fut.done():
            fut.set_result(True)

    # ---- public API ----

    async def discover_reflexive(self, rendezvous_id, timeout: float = 1.0):
        """STUN-lite: learn our own NAT-mapped (host,port) from the rendezvous."""
        self._whoami_fut = asyncio.get_running_loop().create_future()
        self.bus.send(Message(self.identity.node_id, rendezvous_id, "nat", {"op": "whoami"}))
        try:
            self.reflexive = await asyncio.wait_for(self._whoami_fut, timeout)
        except asyncio.TimeoutError:
            self.reflexive = None
        finally:
            self._whoami_fut = None
        return self.reflexive

    async def punch(self, peer_id, rendezvous_id, timeout: float = 2.0) -> bool:
        """Establish a direct path to peer_id via hole-punching coordinated by
        rendezvous_id. Returns True once the punch handshake completes."""
        if peer_id in self.reachable:
            return True
        fut = asyncio.get_running_loop().create_future()
        self._punch_futs[peer_id] = fut
        self.bus.send(Message(self.identity.node_id, rendezvous_id, "nat",
                              {"op": "connect_req", "target_id": peer_id}))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return peer_id in self.reachable
        finally:
            self._punch_futs.pop(peer_id, None)


def _parse_addr(s: str):
    host, port = s.rsplit(":", 1)
    return (host, int(port))
