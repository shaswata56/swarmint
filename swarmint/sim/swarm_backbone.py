"""An always-on honest BACKBONE for a live swarm.

Why this exists (the missing piece that makes the public swarm actually useful):
a single joiner reaching only the rendezvous learns nothing new — the Byzantine
defense requires >=3 distinct honest senders to corroborate any class the joiner
doesn't already own (README lesson 4 / decision #023). An empty network can never
supply that quorum, so a lone joiner sits at its solo ceiling forever. This module
runs N honest NetNodes IN ONE PROCESS (asyncio tasks — far lighter than the
multiproc test's separate interpreters, right for a tiny always-on VM), covering
ALL classes with >=3 honest coverage, all bootstrapped to the same rendezvous.
With the backbone online, any joiner's missing classes are immediately corroborated
and its accuracy climbs toward the centralized ceiling.

The backbone nodes advertise a PUBLIC host (for a 1:1-NAT cloud VM) on fixed gossip
ports, so external joiners — which discover them via PEX from the rendezvous — can
reach them directly (the firewall must open that port range). Among themselves the
nodes talk over the loopback the co-located sockets already share.

Reuses the exact, test-covered coverage-assignment logic from run_multiproc so the
"every class reaches >=3 honest senders" guarantee is identical to the integration
test that proves the swarm hits >=0.90.
"""

import asyncio
import random
import socket
import time

import numpy as np
from nacl.signing import SigningKey

from ..network.identity import Identity
from ..node.net_node import NetNode
from .live_task import build_task
from .run_multiproc import (_assign_classes_with_min_coverage,
                            _required_classes_per_node)


def _identity_from_seed(seed: int) -> Identity:
    return Identity(SigningKey(seed.to_bytes(4, "big").rjust(32, b"\0")))


def _log(event: str, **kv) -> None:
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"EVENT {event} {parts}".rstrip(), flush=True)


def backbone_addrs(n: int, public_host: str, base_port: int = 9003,
                   seed_base: int = 1000) -> list:
    """Reconstruct every backbone node's deterministic (node_id, (host, port)),
    with NO DHT lookup — the single-task analogue of multimodal's expert_addrs().

    Both identity AND port are pure functions of (seed_base, base_port, index),
    exactly as run_backbone lays them out: node i uses identity seed
    ``seed_base + i`` and gossip port ``base_port + i``. A one-shot query client
    can therefore build the whole address book locally and hand it straight to
    the bus, depending on neither PEX (random-tick luck) nor the DHT (which does
    not reliably scale to a cold co-located cohort on a small VM — decision
    #050) to find a backbone whose layout is already deterministic. This is the
    same "seed the address directly, don't wait on a flaky lookup" principle that
    fixed the joiner's DHT-bootstrap fragility on cellular (#043)."""
    return [(_identity_from_seed(seed_base + i).node_id, (public_host, base_port + i))
            for i in range(n)]


async def run_backbone(*, beacon_host: str, beacon_id_hex: str, n: int = 8,
                       base_port: int = 9003, dht_base_port: int = 9103,
                       beacon_gossip_port: int = 9001, beacon_dht_port: int = 9002,
                       advertise: str = "0.0.0.0", public_host: str = None,
                       n_classes: int = 10, dim: int = 16, world_seed: int = 0,
                       gossip_interval: float = 2.0, seed_base: int = 1000,
                       report_every: float = 15.0, duration_s: float = 0.0,
                       enable_relay: bool = True, task_name: str = "synthetic") -> int:
    """Launch N honest backbone nodes and keep them gossiping forever (duration_s=0)."""
    beacon_id = bytes.fromhex(beacon_id_hex)
    r_ip = socket.gethostbyname(beacon_host)
    task = build_task(task_name, n_classes=n_classes, dim=dim, world_seed=world_seed)

    # Same honest-coverage guarantee the multiproc test uses: every class reaches
    # >=3 distinct backbone senders, or corroboration could never promote it.
    cpn = _required_classes_per_node(n, n_classes, 2, min_coverage=3)
    assignments = _assign_classes_with_min_coverage(n, n_classes, cpn, random.Random(world_seed))
    _log("backbone_plan", n=n, task=task_name, classes_per_node=cpn,
         beacon=f"{beacon_host}:{beacon_dht_port}",
         resolved=r_ip, public_host=public_host or advertise)

    def _quiet(loop, context):
        if isinstance(context.get("exception"), asyncio.InvalidStateError):
            return
        loop.default_exception_handler(context)
    asyncio.get_running_loop().set_exception_handler(_quiet)

    nodes = []
    for i in range(n):
        ident = _identity_from_seed(seed_base + i)
        rng = random.Random(seed_base * 10 + i)
        np_rng = np.random.default_rng(seed_base * 7 + i)
        my_classes = assignments[i]
        net = NetNode(identity=ident, topic=1, rng=rng, n_classes=n_classes,
                      gossip_interval=gossip_interval, gossip_sample_size=12,
                      enable_relay=enable_relay, embedding=task.embedding,
                      radii=(task.radii or None))

        def feed(_task=task, _cls=my_classes, _rng=np_rng):
            return _task.feed(_cls, _rng)
        net.data_feed = feed

        await net.start(advertise, base_port + i, dht_base_port + i,
                        [(r_ip, beacon_dht_port)], rendezvous_id=beacon_id,
                        rendezvous_addr=(r_ip, beacon_gossip_port),
                        advertise_host=public_host)
        nodes.append((i, net, my_classes))
        _log("backbone_node_up", i=i, node_id=ident.node_id.hex(),
             classes=",".join(map(str, my_classes)),
             gossip=f"{public_host or advertise}:{base_port + i}")

    test_x, test_y = task.test_x, task.test_y
    start = time.time()
    last = 0.0
    while duration_s == 0 or (time.time() - start) < duration_s:
        await asyncio.sleep(1.0)
        now = time.time() - start
        if now - last >= report_every:
            last = now
            accs = [float(net.node.model.accuracy(test_x, test_y)) for _, net, _ in nodes]
            peers = [len(net.node.peers) for _, net, _ in nodes]
            _log("backbone_metrics", t=round(now, 1),
                 mean_acc=round(float(np.mean(accs)), 3),
                 min_acc=round(float(np.min(accs)), 3),
                 mean_peers=round(float(np.mean(peers)), 1),
                 protos=sum(len(net.node.model.protos) for _, net, _ in nodes))

    for _, net, _ in nodes:
        await net.stop()
    return 0
