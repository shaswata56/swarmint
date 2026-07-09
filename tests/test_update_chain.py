"""D1: per-node tamper-evident hash-chain — unit tests for the chain itself
plus an end-to-end fetch+verify over real UDP, and tamper detection."""

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.prototype_model import Prototype, PrototypeModel
from swarmint.core.update_chain import UpdateChain, model_digest, verify_chain
from swarmint.core.version import Version
from swarmint.network.identity import Identity
from swarmint.node.net_node import NetNode


def _model_with(labels):
    m = PrototypeModel(topic=1)
    for i, lbl in enumerate(labels):
        m.protos.append(Prototype(np.full(4, float(i), dtype=np.float32), lbl, 1.0))
    return m


def test_chain_builds_and_verifies():
    idn = Identity.generate()
    chain = UpdateChain(idn)
    m = _model_with([0])
    for c in range(1, 6):
        m.protos.append(Prototype(np.full(4, float(c), dtype=np.float32), c, 1.0))
        chain.checkpoint(Version(c, idn.node_id), m.protos)
    ok, reason = verify_chain(chain.entries, expected_node_id=idn.node_id)
    assert ok, reason
    assert len(chain.entries) == 5
    assert chain.entries[0].prev_hash == b"\x00" * 32
    assert chain.entries[3].prev_hash == chain.entries[2].entry_hash  # contiguous


def test_model_digest_is_order_independent():
    a = [Prototype(np.array([1.0, 2.0], dtype=np.float32), 0, 1.0),
         Prototype(np.array([3.0, 4.0], dtype=np.float32), 1, 1.0)]
    b = list(reversed(a))
    assert model_digest(a) == model_digest(b)  # set semantics, matches order-invariant merge


def test_tamper_reordering_detected():
    idn = Identity.generate()
    chain = UpdateChain(idn)
    m = _model_with([0])
    for c in range(1, 5):
        m.protos.append(Prototype(np.full(4, float(c), dtype=np.float32), c, 1.0))
        chain.checkpoint(Version(c, idn.node_id), m.protos)
    entries = chain.entries[:]
    entries[1], entries[2] = entries[2], entries[1]  # reorder
    ok, reason = verify_chain(entries, expected_node_id=idn.node_id)
    assert not ok and ("link" in reason or "seq" in reason)


def test_tamper_content_detected():
    idn = Identity.generate()
    chain = UpdateChain(idn)
    m = _model_with([0, 1])
    chain.checkpoint(Version(1, idn.node_id), m.protos)
    chain.checkpoint(Version(2, idn.node_id), m.protos)
    entries = chain.to_wire()
    entries[0]["m"] = b"\xde" * 32  # tamper the recorded model digest
    ok, reason = verify_chain(entries, expected_node_id=idn.node_id)
    assert not ok and ("hash" in reason or "tamper" in reason or "signature" in reason)


def test_forged_entry_by_other_key_detected():
    victim, attacker = Identity.generate(), Identity.generate()
    chain = UpdateChain(attacker)          # attacker builds a real chain...
    m = _model_with([0])
    chain.checkpoint(Version(1, attacker.node_id), m.protos)
    entries = chain.to_wire()
    entries[0]["id"] = victim.node_id      # ...then claims it's the victim's
    ok, reason = verify_chain(entries, expected_node_id=victim.node_id)
    assert not ok  # id no longer matches H(pubkey), or belongs-to check fails


async def test_fetch_chain_over_udp_multi_chunk():
    """End-to-end: build a chain long enough to span multiple datagrams, fetch
    it over real UDP, and verify it reassembles and validates."""
    id_a, id_b = Identity.generate(), Identity.generate()
    net_a = NetNode(identity=id_a, topic=1, rng=random.Random(1), gossip_interval=10.0)
    net_b = NetNode(identity=id_b, topic=1, rng=random.Random(2), gossip_interval=10.0)
    await net_a.start("127.0.0.1", 0, 0, [])
    dht_a = net_a.discovery.server.transport.get_extra_info("sockname")[1]
    await net_b.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_a)], rendezvous_id=id_a.node_id)

    # build node A a chain of 8 entries (> CHAIN_ENTRIES_PER_CHUNK=3, so 3 datagrams)
    m = _model_with([0])
    for c in range(1, 9):
        m.protos.append(Prototype(np.full(4, float(c), dtype=np.float32), c % 10, 1.0))
        net_a.chain.checkpoint(Version(c, id_a.node_id), m.protos)

    # B must know A's gossip address to send the request
    net_b.bus.peer_addrs[id_a.node_id] = net_b.discovery.peer_addrs.get(
        id_a.node_id, ("127.0.0.1", net_a.bus.bound_port))
    ok, reason, entries = await net_b.fetch_chain(id_a.node_id, timeout=2.0)

    await net_a.stop()
    await net_b.stop()

    assert ok, reason
    assert entries is not None and len(entries) == 8


async def _run_all():
    sync_tests = [test_chain_builds_and_verifies, test_model_digest_is_order_independent,
                  test_tamper_reordering_detected, test_tamper_content_detected,
                  test_forged_entry_by_other_key_detected]
    for t in sync_tests:
        t()
        print(f"ok  {t.__name__}")
    await test_fetch_chain_over_udp_multi_chunk()
    print("ok  test_fetch_chain_over_udp_multi_chunk")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
