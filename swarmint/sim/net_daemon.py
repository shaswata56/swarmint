"""Single real node, one process, configured entirely from the environment.

Unlike run_multiproc (which forks many workers inside one host and talks over
loopback, where there is no NAT to open), this launches ONE NetNode meant to
run alone inside its own container / host / network namespace. It is the unit
the Docker NAT harness (docker/nat-test/) places behind a real Linux netfilter
NAT so the hole-punching path is exercised across genuine NAT mappings — the
one thing loopback and in-process tests structurally cannot cover
(see network/nat.py's TESTABILITY NOTE).

Roles
-----
  rendezvous : bind a reachable address, act as DHT bootstrap + hole-punch
               coordinator + genesis (shared world). Also learns, so gossip
               has content. This is the only node others look up by address.
  node       : join via the rendezvous, discover our own reflexive (NAT-mapped)
               address, optionally hole-punch to named peers, then gossip.

Config (all via env; SWARM_ prefix)
  SWARM_ROLE                rendezvous | node          (required)
  SWARM_SEED                int -> deterministic identity + rng   (required)
  SWARM_ADVERTISE_HOST      IP to bind + advertise      (default 0.0.0.0)
  SWARM_GOSSIP_PORT         UDP gossip port             (default 9001)
  SWARM_DHT_PORT            Kademlia DHT port           (default 9002)
  SWARM_RENDEZVOUS_HOST     rendezvous reachable IP     (node role)
  SWARM_RENDEZVOUS_DHT_PORT rendezvous DHT port         (node role)
  SWARM_RENDEZVOUS_ID       rendezvous node_id, hex     (node role)
  SWARM_PUNCH_PEERS         comma-sep peer node_ids (hex) to hole-punch
  SWARM_CLASSES             comma-sep class ids this node learns (default auto)
  SWARM_N_CLASSES           total classes in the world  (default 10)
  SWARM_DIM                 feature dim                  (default 16)
  SWARM_WORLD_SEED          shared world seed (MUST match across nodes; default 0)
  SWARM_GOSSIP_INTERVAL     seconds between gossip rounds (default 0.5)
  SWARM_DURATION_S          run length, seconds; 0 = forever (default 90)
  SWARM_REPORT_EVERY_S      metrics log cadence, seconds (default 3)

Every line is printed to stdout as `EVENT key=val ...` so `docker logs` /
the orchestrator can grep the run without a metrics volume. The headline proof
line is `EVENT reflexive ... nat_detected=True|False`.

Run:  SWARM_ROLE=rendezvous SWARM_SEED=0 python -m swarmint.sim.net_daemon
"""

import asyncio
import os
import random
import socket
import sys
import time

import numpy as np
from nacl.signing import SigningKey

from ..network.identity import Identity
from ..node.net_node import NetNode
from .live_task import build_task


def _env(name, default=None):
    return os.environ.get(name, default)


def _identity_from_seed(seed: int) -> Identity:
    # 32-byte Ed25519 seed derived deterministically from the int seed, so the
    # orchestrator can precompute every node_id and wire up PUNCH_PEERS ahead of
    # time without a discovery round-trip.
    seed_bytes = seed.to_bytes(4, "big").rjust(32, b"\0")
    return Identity(SigningKey(seed_bytes))


def _log(event: str, **kv) -> None:
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"EVENT {event} {parts}".rstrip(), flush=True)


async def main() -> int:
    role = _env("SWARM_ROLE")
    if role not in ("rendezvous", "node"):
        _log("fatal", reason="SWARM_ROLE must be rendezvous|node", got=role)
        return 2

    seed = int(_env("SWARM_SEED", "0"))
    advertise = _env("SWARM_ADVERTISE_HOST", "0.0.0.0")
    gossip_port = int(_env("SWARM_GOSSIP_PORT", "9001"))
    dht_port = int(_env("SWARM_DHT_PORT", "9002"))
    n_classes = int(_env("SWARM_N_CLASSES", "10"))
    dim = int(_env("SWARM_DIM", "16"))
    world_seed = int(_env("SWARM_WORLD_SEED", "0"))
    gossip_interval = float(_env("SWARM_GOSSIP_INTERVAL", "0.5"))
    duration_s = float(_env("SWARM_DURATION_S", "90"))
    report_every = float(_env("SWARM_REPORT_EVERY_S", "3"))
    enable_relay = _env("SWARM_ENABLE_RELAY", "0") == "1"
    # Public (dial-able) address to ADVERTISE, distinct from the bind host, for
    # 1:1-NAT cloud hosts (GCP/AWS) where the public IP isn't on the VM's NIC.
    public_host = _env("SWARM_PUBLIC_HOST") or None
    http_port = int(_env("SWARM_HTTP_PORT", "0"))  # rendezvous status page; 0 = off
    http_bind = _env("SWARM_HTTP_BIND", "0.0.0.0")  # bind 127.0.0.1 when a TLS proxy fronts it

    identity = _identity_from_seed(seed)
    rng = random.Random(1000 + seed)
    np_rng = np.random.default_rng(7 * seed + 1)

    # The learning task. `synthetic` (default) keeps the original behaviour exactly;
    # `digits` runs the swarm on real bundled sklearn digits with a shared LDA genesis
    # embedding + data-derived radii (see sim/live_task.py).
    task_name = _env("SWARM_TASK", "synthetic")
    task = build_task(task_name, n_classes=n_classes, dim=dim, world_seed=world_seed)

    classes_env = _env("SWARM_CLASSES")
    if classes_env:
        my_classes = [int(c) for c in classes_env.split(",") if c != ""]
    else:  # deterministic 2-class slice per node when unset
        my_classes = sorted({(seed * 2) % n_classes, (seed * 2 + 1) % n_classes})

    net = NetNode(identity=identity, topic=1, rng=rng, n_classes=n_classes,
                  gossip_interval=gossip_interval, gossip_sample_size=12,
                  enable_relay=enable_relay, embedding=task.embedding,
                  radii=(task.radii or None))

    def feed():
        return task.feed(my_classes, np_rng)
    net.data_feed = feed

    _log("identity", role=role, seed=seed, node_id=identity.node_id.hex(),
         classes=",".join(map(str, my_classes)), advertise=advertise,
         gossip_port=gossip_port, dht_port=dht_port)

    # rpcudp benign duplicate-response race (see run_multiproc) — swallow only that.
    def _quiet(loop, context):
        if isinstance(context.get("exception"), asyncio.InvalidStateError):
            return
        loop.default_exception_handler(context)
    asyncio.get_running_loop().set_exception_handler(_quiet)

    status_server = None
    if role == "rendezvous":
        await net.start(advertise, gossip_port, dht_port, [], advertise_host=public_host)
        _log("rendezvous_up", node_id=identity.node_id.hex(),
             gossip=f"{public_host or advertise}:{net.bus.bound_port}",
             dht=f"{public_host or advertise}:{dht_port}")
        if http_port:
            from ..network import beacon_status
            status_server = await beacon_status.serve(net, http_bind, http_port, time.time())
            _log("status_http_up", bind=f"{http_bind}:{http_port}")
        rendezvous_id = None
        punch_peers = []
    else:
        r_host = _env("SWARM_RENDEZVOUS_HOST")
        r_dht = int(_env("SWARM_RENDEZVOUS_DHT_PORT", "9002"))
        rendezvous_id = bytes.fromhex(_env("SWARM_RENDEZVOUS_ID", ""))
        punch_peers = [bytes.fromhex(p) for p in (_env("SWARM_PUNCH_PEERS", "") or "").split(",") if p]
        # Resolve the rendezvous by NAME, not a pinned IP: the DNS record
        # (beacon.swarmint.org) is the durable, relocatable anchor — re-hosting
        # the beacon is then a DNS change only, never a config/code edit in every
        # peer. Kademlia's UDP bootstrap needs a numeric address, so we resolve
        # the name locally at join time (picking up the current IP each start).
        r_ip = socket.gethostbyname(r_host)
        r_gossip = int(_env("SWARM_RENDEZVOUS_GOSSIP_PORT", "9001"))
        _log("joining", rendezvous=f"{r_host}:{r_dht}", resolved=r_ip,
             rendezvous_id=rendezvous_id.hex())
        # Seed the rendezvous gossip addr directly (r_ip:r_gossip) so connectivity
        # survives a flaky DHT lookup on lossy/high-latency links (cellular).
        await net.start(advertise, gossip_port, dht_port,
                        [(r_ip, r_dht)], rendezvous_id=rendezvous_id,
                        rendezvous_addr=(r_ip, r_gossip))
        reachable = rendezvous_id in net.bus.peer_addrs
        _log("joined", rendezvous_resolved=reachable,
             rendezvous_addr=str(net.bus.peer_addrs.get(rendezvous_id)))

        # --- the NAT proof: our reflexive (public-mapped) addr vs our local bind ---
        reflexive = await net.nat.discover_reflexive(rendezvous_id, timeout=3.0)
        local = f"{advertise}:{net.bus.bound_port}"
        if reflexive is not None:
            refl = f"{reflexive[0]}:{reflexive[1]}"
            nat_detected = (reflexive[0] != advertise) or (reflexive[1] != net.bus.bound_port)
            _log("reflexive", local=local, reflexive=refl, nat_detected=nat_detected)
        else:
            _log("reflexive", local=local, reflexive="none",
                 nat_detected="unknown", note="rendezvous whoami timed out")

        # --- hole-punch to each named peer across its own NAT ---
        # Brief settle so every peer has run whoami and registered its reflexive
        # addr with the rendezvous before anyone asks it to relay a connect —
        # otherwise the first puncher races ahead of the others' registration.
        if punch_peers:
            await asyncio.sleep(float(_env("SWARM_PUNCH_DELAY_S", "4")))
        for peer in punch_peers:
            ok = False
            for attempt in range(3):  # tolerate a slow peer / dropped punch packet
                ok = await net.nat.punch(peer, rendezvous_id, timeout=5.0)
                if ok:
                    break
                await asyncio.sleep(1.0)
            _log("punch", peer=peer.hex(), ok=ok, attempts=attempt + 1,
                 peer_addr=str(net.bus.peer_addrs.get(peer)),
                 in_reachable=(peer in net.nat.reachable))
            # If the punch didn't open a direct path, fall back to relaying via
            # the rendezvous (T4) so the swarm still converges. No-op unless
            # SWARM_ENABLE_RELAY=1.
            if not ok and enable_relay:
                net.mark_relay_only(peer)
                _log("relay_fallback", peer=peer.hex(), via=rendezvous_id.hex())

    # --- run loop: gossip proceeds via NodeRunner; we just report ---
    test_x, test_y = task.test_x, task.test_y
    start = time.time()
    last = 0.0
    while duration_s == 0 or (time.time() - start) < duration_s:
        await asyncio.sleep(0.5)
        now = time.time() - start
        if now - last >= report_every:
            last = now
            reachable = sorted(p.hex()[:8] for p in getattr(net.nat, "reachable", set())) if net.nat else []
            _log("metrics", t=round(now, 1),
                 acc=round(float(net.node.model.accuracy(test_x, test_y)), 3),
                 n_peers=len(net.node.peers), n_protos=len(net.node.model.protos),
                 reachable=",".join(reachable) or "-",
                 tx=net.bus.tx_bytes, rx=net.bus.rx_bytes,
                 merges_ok=net.node.merges_ok, merges_rej=net.node.merges_rejected,
                 relay_tx=net.bus.relayed_tx, relay_fwd=net.bus.relayed_fwd,
                 relay_rx=net.bus.relayed_rx)

    if status_server is not None:
        status_server.close()
    await net.stop()
    _log("stopped", role=role, seed=seed)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
