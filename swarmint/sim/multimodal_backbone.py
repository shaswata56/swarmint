"""A persistent, LIVE multimodal backbone — the multimodal analogue of
swarm_backbone.py, attached to a real remote rendezvous (the public beacon)
instead of a local one-shot process.

run_multimodal_net.py proved cross-modal fusion over real sockets, but it spins up
its OWN rendezvous, converges, queries once, and tears everything down — useful for
a regression test, useless for a deployment. This module instead attaches N
per-modality expert nodes (each modality = its own topic, D2) to an EXISTING
rendezvous (e.g. the live beacon) and gossips forever, so a remote client can join
later and fuse across modalities against a standing swarm — the same shape as the
digits backbone (swarm_backbone.py), generalized to multiple topics.

Pairs with sim/multimodal_query.py, which is the client side: a one-shot join that
discovers each modality's experts via the DHT, queries them, and fuses.
"""

import asyncio
import random
import socket
import time

import numpy as np
from nacl.signing import SigningKey

from ..network.identity import Identity
from ..node.net_node import NetNode
from .data_multimodal import make_modalities, modality_test_set, sample_modality
from .run_multiproc import _assign_classes_with_min_coverage, _required_classes_per_node

# Modality topics start here, NOT at 1: topic=1 is the universal hardcoded topic
# for the single-task deployments (net_daemon.py, swarm_backbone.py — digits and
# synthetic). Reusing topic=1 collided a modality-0 expert with the live digits
# beacon/backbone via PEX: same topic -> mutual gossip peers -> a merge between
# incompatible raw dimensionalities crashed with a numpy broadcast ValueError.
# Gossip fan-out (SwarmNode.gossip_step) is NOT topic-scoped by design (topic
# routing lives in the network layer, per invariant #1) — it trusts that same-topic
# peers share a representation. This offset keeps that assumption true in practice.
MODALITY_TOPIC_BASE = 100


def _ident(seed: int) -> Identity:
    return Identity(SigningKey(seed.to_bytes(4, "big").rjust(32, b"\0")))


def expert_node_ids(n_modalities: int, per_modality: int, seed: int = 11) -> dict:
    """Reconstruct every expert's (deterministic) node_id, keyed by modality index.

    Identities are deterministic from `seed` (the same "known bootstrap identity"
    pattern used for the rendezvous itself)."""
    out = {m: [] for m in range(n_modalities)}
    sk = seed * 1000
    for m in range(n_modalities):
        for _ in range(per_modality):
            sk += 1
            out[m].append(_ident(sk).node_id)
    return out


def expert_addrs(n_modalities: int, per_modality: int, public_host: str,
                 base_port: int = 9401, seed: int = 11) -> dict:
    """Reconstruct every expert's (deterministic) node_id AND gossip (host, port),
    keyed by modality index, with NO DHT lookup at all.

    A query client should depend on neither PEX nor the DHT to discover this
    backbone: PEX only propagates when a peer's periodic gossip tick happens to
    pick us as its random target, and — the harder finding, from testing this live
    — the DHT ADVERT-PUBLISH path itself doesn't reliably scale to ~24 co-located
    nodes bootstrapping through one rendezvous contact on a small VM: most experts'
    own `set()` publish calls kept failing ("no known neighbors") for minutes,
    because Kademlia routing tables only grow from ongoing two-way DHT traffic, and
    a cold cohort this size mostly never accumulates enough. Only 1 of 15 experts
    ever got its advert published in testing. Since every expert's (node_id, port)
    is a pure function of (seed, base_port, running index) — the SAME
    "seed the address directly, don't depend on a flaky lookup" principle used to
    fix the digits joiner's DHT-bootstrap fragility on cellular (decision #043) —
    the client can construct every address without any DHT round-trip."""
    out = {m: [] for m in range(n_modalities)}
    sk = seed * 1000
    port = base_port
    for m in range(n_modalities):
        for _ in range(per_modality):
            sk += 1
            out[m].append((_ident(sk).node_id, (public_host, port)))
            port += 1
    return out


def _log(event, **kv):
    print(f"EVENT {event} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


async def run_multimodal_backbone(*, beacon_host: str, beacon_id_hex: str,
                                  n_modalities: int = 3, per_modality: int = 5,
                                  n_classes: int = 6, noise: float = 1.7,
                                  base_port: int = 9101, dht_base_port: int = 9201,
                                  beacon_gossip_port: int = 9001, beacon_dht_port: int = 9002,
                                  advertise: str = "0.0.0.0", public_host: str = None,
                                  seed: int = 11, gossip_interval: float = 2.0,
                                  report_every: float = 30.0, duration_s: float = 0.0,
                                  enable_relay: bool = True) -> int:
    """Attach n_modalities x per_modality honest expert nodes to an existing
    rendezvous, each modality its own topic, and gossip forever (duration_s=0)."""
    beacon_id = bytes.fromhex(beacon_id_hex)
    r_ip = socket.gethostbyname(beacon_host)
    spec = make_modalities(n_modalities, n_classes, noise=noise, seed=seed)

    def _quiet(loop, ctx):
        if isinstance(ctx.get("exception"), asyncio.InvalidStateError):
            return
        loop.default_exception_handler(ctx)
    asyncio.get_running_loop().set_exception_handler(_quiet)

    experts = []
    port = base_port
    dht_port = dht_base_port
    sk = seed * 1000
    for m in range(n_modalities):
        topic = MODALITY_TOPIC_BASE + m + 1
        cpn = _required_classes_per_node(per_modality, n_classes, 2, min_coverage=3)
        assign = _assign_classes_with_min_coverage(per_modality, n_classes, cpn,
                                                   random.Random(seed * 31 + m))
        for j in range(per_modality):
            sk += 1
            ident = _ident(sk)
            net = NetNode(identity=ident, topic=topic, rng=random.Random(sk),
                          n_classes=n_classes, gossip_interval=gossip_interval,
                          gossip_sample_size=12, enable_relay=enable_relay)
            net.topics = {topic}
            classes = assign[j]
            np_rng = np.random.default_rng(sk)

            def feed(_m=m, _cls=classes, _rng=np_rng):
                xs, ys = sample_modality(spec[_m], _cls, 1, _rng)
                return (xs[0], int(ys[0]))
            net.data_feed = feed

            await net.start(advertise, port, dht_port, [(r_ip, beacon_dht_port)],
                            rendezvous_id=beacon_id, rendezvous_addr=(r_ip, beacon_gossip_port),
                            advertise_host=public_host)
            port += 1
            dht_port += 1
            experts.append((m, topic, net))
    _log("multimodal_backbone_up", n=len(experts), modalities=n_modalities,
         per_modality=per_modality, beacon=f"{beacon_host}", resolved=r_ip,
         public_host=public_host or advertise, ports=f"{base_port}..{port - 1}")

    test_sets = {m: modality_test_set(spec[m], n_per_class=20) for m in range(n_modalities)}
    start = time.time()
    last = 0.0
    while duration_s == 0 or (time.time() - start) < duration_s:
        await asyncio.sleep(1.0)
        now = time.time() - start
        if now - last >= report_every:
            last = now
            for m in range(n_modalities):
                topic = MODALITY_TOPIC_BASE + m + 1
                accs = [nd.node.model.accuracy(*test_sets[m])
                        for mm, tt, nd in experts if mm == m]
                _log("modality_metrics", t=round(now, 1), modality=m, topic=topic,
                     mean_acc=round(float(np.mean(accs)), 3),
                     min_acc=round(float(np.min(accs)), 3))

    for _, _, net in experts:
        await net.stop()
    return 0
