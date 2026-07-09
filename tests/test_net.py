"""T4: real asyncio UDP transport — loopback gossip over actual sockets,
oversized-message rejection, and Windows transport-loss resilience.

Async tests use plain asyncio.run() (no pytest/pytest-asyncio in the pinned
venv — see requirements.txt; keeping this dependency-free matches the rest of
the suite's plain-script convention).
"""

import asyncio
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.prototype_model import Prototype
from swarmint.network import wire
from swarmint.network.bus import Message
from swarmint.network.identity import Identity, ReplayGuard
from swarmint.network.udp_bus import UdpBus, _Protocol


async def test_two_node_loopback_gossip():
    """The decisive T4 test: two real UDP sockets on 127.0.0.1, signed
    envelopes, end-to-end through UdpBus into a plain callback."""
    id_a, id_b = Identity.generate(), Identity.generate()
    bus_a, bus_b = UdpBus(identity=id_a), UdpBus(identity=id_b)
    port_a = await bus_a.start("127.0.0.1", 0)
    port_b = await bus_b.start("127.0.0.1", 0)
    bus_a.peer_addrs[id_b.node_id] = ("127.0.0.1", port_b)
    bus_b.peer_addrs[id_a.node_id] = ("127.0.0.1", port_a)

    received = []
    bus_b.register(id_b.node_id, lambda msg: received.append(msg))
    bus_a.register(id_a.node_id, lambda msg: None)

    protos = [Prototype(np.array([1.0, 2.0, 3.0], dtype=np.float32), label=5, weight=1.0)]
    bus_a.send(Message(sender=id_a.node_id, recipient=id_b.node_id, kind="protos", payload=protos))

    for _ in range(50):  # poll up to ~500ms for the datagram to arrive
        if received:
            break
        await asyncio.sleep(0.01)

    await bus_a.stop()
    await bus_b.stop()

    assert len(received) == 1
    msg = received[0]
    assert msg.sender == id_a.node_id
    assert msg.kind == "protos"
    assert msg.payload[0].label == 5
    assert np.allclose(msg.payload[0].vector, [1.0, 2.0, 3.0])
    assert bus_a.tx_bytes > 0 and bus_b.rx_bytes > 0


async def test_forged_sender_dropped_before_handler():
    """An envelope with a valid signature but a forged sender-id claim must
    never reach the registered handler (anti-impersonation, decision #023 D4)."""
    victim, attacker = Identity.generate(), Identity.generate()
    bus = UdpBus(identity=victim)
    await bus.start("127.0.0.1", 0)
    received = []
    bus.register(victim.node_id, lambda msg: received.append(msg))

    body = wire.pack_gossip_protos([Prototype(np.zeros(4, dtype=np.float32), 0, 1.0)])
    ts, nonce = time.time(), b"\x00" * 16
    signable = wire.signable_bytes(attacker.pubkey, victim.node_id, ts, nonce, body)
    sig = attacker.signing_key.sign(signable).signature
    forged = wire.pack_envelope(attacker.pubkey, victim.node_id, ts, nonce, sig, body)

    bus._on_datagram(forged, ("127.0.0.1", 1))  # deliver directly, bypassing the socket
    await bus.stop()
    assert received == []
    assert bus.dropped_unverified == 1


async def test_oversized_gossip_batch_chunked_not_dropped():
    """SwarmNode.gossip_step() samples up to 30 prototypes per call (untouched
    node logic — decision #023 D9's 12-proto cap is a WIRE decision, not a
    node-logic one). A 200-proto batch must be split into <=12-proto chunks
    and all delivered, not silently dropped whole (the original behavior here
    was 100% packet loss for any node whose model grew past ~12 prototypes —
    found via the T7 NetNode integration test)."""
    id_a, id_b = Identity.generate(), Identity.generate()
    bus_a, bus_b = UdpBus(identity=id_a), UdpBus(identity=id_b)
    port_a = await bus_a.start("127.0.0.1", 0)
    port_b = await bus_b.start("127.0.0.1", 0)
    bus_a.peer_addrs[id_b.node_id] = ("127.0.0.1", port_b)

    received_chunks = []
    bus_b.register(id_b.node_id, lambda msg: received_chunks.append(msg.payload))

    huge = [Prototype(np.zeros(16, dtype=np.float32), i % 10, 1.0) for i in range(200)]
    bus_a.send(Message(sender=id_a.node_id, recipient=id_b.node_id, kind="protos", payload=huge))

    for _ in range(100):
        if sum(len(c) for c in received_chunks) >= 200:
            break
        await asyncio.sleep(0.02)

    await bus_a.stop()
    await bus_b.stop()

    assert bus_a.dropped_oversized == 0
    assert len(received_chunks) == 17  # ceil(200/12)
    assert all(len(c) <= 12 for c in received_chunks)
    assert sum(len(c) for c in received_chunks) == 200


async def test_non_list_oversized_payload_still_refused():
    """A payload that ISN'T chunkable (not a list) and still exceeds the
    budget must be refused, not silently truncated or crash the send path."""
    id_a = Identity.generate()
    bus = UdpBus(identity=id_a)
    bus._encoders["oversized_test"] = lambda payload: b"\x00" * 2000
    bus.peer_addrs[b"\xff" * 20] = ("127.0.0.1", 1)
    await bus.start("127.0.0.1", 0)
    bus.send(Message(sender=id_a.node_id, recipient=b"\xff" * 20,
                     kind="oversized_test", payload={"not": "a list"}))
    await bus.stop()
    assert bus.dropped_oversized == 1
    assert bus.tx_bytes == 0


async def test_transport_survives_connection_lost_and_rebinds():
    """Simulates the Windows WinError-10054 scenario deterministically: force
    connection_lost() and confirm UdpBus rebinds at the same port and keeps
    working, rather than requiring a real ICMP round-trip (which would be
    flaky to rely on on this machine)."""
    id_a = Identity.generate()
    bus = UdpBus(identity=id_a)
    port = await bus.start("127.0.0.1", 0)

    # Faithfully reproduce the OS-level event: the socket is actually gone
    # (freeing the port) before the reset notification fires. Proactor's
    # get_extra_info("socket") is a restricted TransportSocket with no
    # close(), so free the port via a graceful close (exc=None -> no rebind
    # triggered by that call) and then manually deliver the reset
    # notification our real handler would get from Windows.
    bus._transport.close()
    await asyncio.sleep(0.05)
    bus._protocol.connection_lost(OSError("simulated WinError 10054"))
    for _ in range(50):
        if bus._transport is not None and not bus._transport.is_closing():
            try:
                bus._transport.get_extra_info("sockname")
                break
            except Exception:
                pass
        await asyncio.sleep(0.01)

    assert bus._transport.get_extra_info("sockname")[1] == port  # rebound to same port

    # bus must still be able to send/receive after the simulated reset
    id_b = Identity.generate()
    bus_b = UdpBus(identity=id_b)
    port_b = await bus_b.start("127.0.0.1", 0)
    received = []
    bus_b.register(id_b.node_id, lambda msg: received.append(msg))
    bus.peer_addrs[id_b.node_id] = ("127.0.0.1", port_b)
    bus.send(Message(id_a.node_id, id_b.node_id, "protos",
                     [Prototype(np.zeros(4, dtype=np.float32), 0, 1.0)]))
    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.01)

    await bus.stop()
    await bus_b.stop()
    assert len(received) == 1


async def test_send_to_closed_port_does_not_crash():
    """Real-world companion to the deterministic test above: sending to a
    port nobody is listening on (the actual trigger for a Windows ICMP
    port-unreachable) must not raise or wedge the bus."""
    id_a = Identity.generate()
    bus = UdpBus(identity=id_a, peer_addrs={b"\xee" * 20: ("127.0.0.1", 1)})  # port 1: nothing listens
    await bus.start("127.0.0.1", 0)
    bus.send(Message(id_a.node_id, b"\xee" * 20, "protos",
                     [Prototype(np.zeros(4, dtype=np.float32), 0, 1.0)]))
    await asyncio.sleep(0.2)  # give any ICMP response time to arrive/be handled
    # bus must still be usable afterward
    id_b = Identity.generate()
    bus_b = UdpBus(identity=id_b)
    port_b = await bus_b.start("127.0.0.1", 0)
    received = []
    bus_b.register(id_b.node_id, lambda msg: received.append(msg))
    bus.peer_addrs[id_b.node_id] = ("127.0.0.1", port_b)
    bus.send(Message(id_a.node_id, id_b.node_id, "protos",
                     [Prototype(np.zeros(4, dtype=np.float32), 0, 1.0)]))
    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.01)
    await bus.stop()
    await bus_b.stop()
    assert len(received) == 1


async def _run_all():
    tests = [
        test_two_node_loopback_gossip,
        test_forged_sender_dropped_before_handler,
        test_oversized_gossip_batch_chunked_not_dropped,
        test_non_list_oversized_payload_still_refused,
        test_transport_survives_connection_lost_and_rebinds,
        test_send_to_closed_port_does_not_crash,
    ]
    for t in tests:
        await t()
        print(f"ok  {t.__name__}")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
