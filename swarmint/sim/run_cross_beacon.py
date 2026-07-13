"""Cross-beacon learning over the REAL network stack: N beacons + their swarms
behaving as ONE system.

Two beacons, each hosting its own local swarm that covers a DISJOINT half of the
label space (beacon A's workers know classes {0,1,2}; beacon B's know {3,4,5}),
each with the >=3-honest-sender coverage the Byzantine merge requires. The beacons
join the federation; because they share the SAME embedding space (same fingerprint)
they BRIDGE — each becomes a gossip peer of the other and introduces its local
swarm via a directed PEX. Prototypes then flow across the boundary over the exact
signed gossip/merge path used everywhere (trust, corroboration, conflict rejection
all intact — no new learning code).

The proof measured: after convergence each beacon holds prototypes for the OTHER
half's classes (knowledge it could never have learned alone), and its accuracy on
the FULL label set rises well past the ~0.5 half-coverage ceiling. Biologically:
two specialized regions, wired by a tract, become one integrated model.

Run:  python -m swarmint.sim.run_cross_beacon
"""

import asyncio
import random

import numpy as np
from nacl.signing import SigningKey

from ..network.federation import space_fingerprint
from ..network.identity import Identity
from ..node.net_node import NetNode
from .live_task import build_task


def _ident(seed: int) -> Identity:
    return Identity(SigningKey(seed.to_bytes(4, "big").rjust(32, b"\0")))


def _log(event, **kv):
    print(f"EVENT {event} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


async def _beacon(host, gport, dport, *, seed, fp, n_classes, task, classes, genesis=None):
    """A federating beacon (rendezvous) that ALSO learns its side's classes — like
    the live deployment ("the rendezvous learns too, so gossip has content"). It
    needs its own data to build a holdout; without one a node cannot validate (and
    so rejects) any incoming merge."""
    ident = _ident(seed)
    np_rng = np.random.default_rng(seed)
    net = NetNode(identity=ident, topic=1, rng=random.Random(seed), n_classes=n_classes,
                  gossip_interval=0.3, gossip_sample_size=16, enable_relay=True,
                  embedding=task.embedding, radii=(task.radii or None),
                  enable_federation=True, beacon_task="synthetic",
                  beacon_name=("A" if genesis is None else "B"), beacon_space_fp=fp,
                  federation_genesis_id=(genesis[0] if genesis else None),
                  federation_genesis_addr=(genesis[1] if genesis else None))

    def feed(_cls=classes, _rng=np_rng):
        return task.feed(_cls, _rng)
    net.data_feed = feed
    await net.start(host, gport, dport, [])
    return ident, net


async def _worker(host, gport, dport, *, seed, beacon_id, beacon_gport, beacon_dport,
                  classes, task, n_classes):
    """A swarm node bootstrapped to its side's beacon, feeding a class subset."""
    ident = _ident(seed)
    np_rng = np.random.default_rng(seed)
    net = NetNode(identity=ident, topic=1, rng=random.Random(seed), n_classes=n_classes,
                  gossip_interval=0.3, gossip_sample_size=16, enable_relay=True,
                  embedding=task.embedding, radii=(task.radii or None))

    def feed(_cls=classes, _rng=np_rng):
        return task.feed(_cls, _rng)
    net.data_feed = feed
    await net.start(host, gport, dport, [(host, beacon_dport)],
                    rendezvous_id=beacon_id, rendezvous_addr=(host, beacon_gport))
    return net


def _labels_in(net) -> set:
    return {int(p.label) for p in net.node.model.protos}


async def run(*, host="127.0.0.1", base_port=9601, n_classes=6, per_side=3,
              converge_s=40.0) -> dict:
    task = build_task("synthetic", n_classes=n_classes, dim=16, world_seed=0)
    fp = space_fingerprint("synthetic", n_classes, embedding=None, dim=16, world_seed=0)
    left = list(range(0, n_classes // 2))          # {0,1,2}
    right = list(range(n_classes // 2, n_classes))  # {3,4,5}

    def _quiet(loop, ctx):
        if isinstance(ctx.get("exception"), asyncio.InvalidStateError):
            return
        loop.default_exception_handler(ctx)
    asyncio.get_running_loop().set_exception_handler(_quiet)

    # --- two peer beacons; B bootstraps from A (genesis), then they're equals ---
    a_id, beacon_a = await _beacon(host, base_port, base_port + 1, seed=1, fp=fp,
                                   n_classes=n_classes, task=task, classes=left)
    b_id, beacon_b = await _beacon(host, base_port + 2, base_port + 3, seed=2, fp=fp,
                                   n_classes=n_classes, task=task, classes=right,
                                   genesis=(a_id.node_id, (host, base_port)))
    _log("beacons_up", A=a_id.node_id.hex()[:8], B=b_id.node_id.hex()[:8], fp=fp[:8])

    # --- each side's swarm: per_side workers covering that side's classes fully ---
    workers = []
    port = base_port + 4
    for k in range(per_side):
        workers.append(await _worker(host, port, port + 1, seed=100 + k, beacon_id=a_id.node_id,
                                     beacon_gport=base_port, beacon_dport=base_port + 1,
                                     classes=left, task=task, n_classes=n_classes))
        port += 2
    for k in range(per_side):
        workers.append(await _worker(host, port, port + 1, seed=200 + k, beacon_id=b_id.node_id,
                                     beacon_gport=base_port + 2, beacon_dport=base_port + 3,
                                     classes=right, task=task, n_classes=n_classes))
        port += 2
    _log("workers_up", per_side=per_side, left=left, right=right)

    tx, ty = task.test_x, task.test_y
    for t in range(int(converge_s)):
        await asyncio.sleep(1.0)
        if (t + 1) % 10 == 0:
            _log("converging", t=t + 1,
                 A_labels=sorted(_labels_in(beacon_a)), B_labels=sorted(_labels_in(beacon_b)),
                 A_acc=round(float(beacon_a.node.model.accuracy(tx, ty)), 3),
                 B_acc=round(float(beacon_b.node.model.accuracy(tx, ty)), 3),
                 bridged_A=len(beacon_a.federation._bridged),
                 bridged_B=len(beacon_b.federation._bridged))

    a_labels, b_labels = _labels_in(beacon_a), _labels_in(beacon_b)
    a_acc = float(beacon_a.node.model.accuracy(tx, ty))
    b_acc = float(beacon_b.node.model.accuracy(tx, ty))

    for w in workers:
        await w.stop()
    await beacon_b.stop()
    await beacon_a.stop()

    return {
        "A_labels": sorted(a_labels), "B_labels": sorted(b_labels),
        "A_has_right": bool(set(right) & a_labels), "B_has_left": bool(set(left) & b_labels),
        "A_acc": a_acc, "B_acc": b_acc, "left": left, "right": right,
    }


def main():
    res = asyncio.run(run())
    print("\n== cross-beacon learning over REAL UDP (two beacons, one system) ==")
    print(f"  beacon A learned labels {res['A_labels']}  (its own side was {res['left']})")
    print(f"  beacon B learned labels {res['B_labels']}  (its own side was {res['right']})")
    print(f"  full-set accuracy   A={res['A_acc']:.3f}  B={res['B_acc']:.3f}  "
          f"(half-coverage ceiling ~0.5)")
    crossed = res["A_has_right"] and res["B_has_left"]
    lifted = res["A_acc"] > 0.6 and res["B_acc"] > 0.6
    ok = crossed and lifted
    print(f"\n{'PASS' if ok else 'FAIL'}: prototypes {'CROSSED' if crossed else 'did NOT cross'} "
          f"the beacon boundary and full-set accuracy {'rose past' if lifted else 'stayed at'} "
          f"the half-coverage ceiling — the two beacons learned as one system.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
