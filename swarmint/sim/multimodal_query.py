"""Client side of the live multimodal swarm: join a real remote rendezvous, find
each modality's experts over the DHT, and fuse per-modality answers — the live,
over-the-network counterpart of run_multimodal_net's in-loopback query node.

Run once (join, resolve, query, fuse, report, exit) rather than staying resident —
this is a MEASUREMENT of the standing multimodal_backbone.py swarm, not a peer that
contributes data.
"""

import asyncio
import random
import socket

import numpy as np
from nacl.signing import SigningKey

from ..inference.aggregate import Response, aggregate
from ..network.identity import Identity
from ..node.net_node import NetNode
from .data_multimodal import make_modalities, multiview_test_set
from .multimodal_backbone import MODALITY_TOPIC_BASE, expert_addrs


def _ident(seed: int) -> Identity:
    return Identity(SigningKey(seed.to_bytes(4, "big").rjust(32, b"\0")))


def _log(event, **kv):
    print(f"EVENT {event} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


async def run_multimodal_query(*, beacon_host: str, beacon_id_hex: str,
                               n_modalities: int = 3, per_modality: int = 5,
                               n_classes: int = 6, noise: float = 1.7, seed: int = 11,
                               advertise: str = "0.0.0.0", gossip_port: int = 9301,
                               dht_port: int = 9302, beacon_gossip_port: int = 9001,
                               beacon_dht_port: int = 9002, settle_s: float = 5.0,
                               n_per_class: int = 10, timeout: float = 1.0,
                               backbone_public_host: str = None,
                               backbone_base_port: int = 9401) -> dict:
    beacon_id = bytes.fromhex(beacon_id_hex)
    r_ip = socket.gethostbyname(beacon_host)
    backbone_public_host = backbone_public_host or r_ip
    # SAME (seed, n_classes, noise) as the deployed backbone -> identical modality
    # centers, so this client's test samples are answerable by the live experts.
    spec = make_modalities(n_modalities, n_classes, noise=noise, seed=seed)

    def _quiet(loop, ctx):
        if isinstance(ctx.get("exception"), asyncio.InvalidStateError):
            return
        loop.default_exception_handler(ctx)
    asyncio.get_running_loop().set_exception_handler(_quiet)

    q_ident = _ident(random.randint(1, 2**31 - 1))
    query = NetNode(identity=q_ident, topic=0, rng=random.Random(1),
                    n_classes=n_classes, gossip_interval=1.0, gossip_sample_size=12,
                    enable_relay=True)
    query.topics = {MODALITY_TOPIC_BASE + m + 1 for m in range(n_modalities)}
    await query.start(advertise, gossip_port, dht_port, [(r_ip, beacon_dht_port)],
                      rendezvous_id=beacon_id, rendezvous_addr=(r_ip, beacon_gossip_port))
    _log("joined", rendezvous=f"{beacon_host}", node_id=q_ident.node_id.hex()[:8])

    # Discovery is NEITHER PEX-based NOR DHT-based: PEX only propagates when a
    # peer's periodic gossip tick happens to randomly pick us as its target, and
    # — the harder finding, from testing this live — Kademlia advert-PUBLISH
    # itself doesn't reliably scale to ~24 nodes cold-bootstrapping through one
    # rendezvous contact on a small VM (routing tables only grow from ongoing
    # two-way DHT traffic; observed only 1 of 15 experts ever get its advert
    # published, even after 6+ minutes). Every expert's (node_id, host:port) is
    # instead a pure function of (seed, backbone_public_host, backbone_base_port,
    # running index) — the same "seed the address directly, don't depend on a
    # flaky lookup" fix already used for the digits joiner's DHT-bootstrap
    # fragility (decision #043) — so we construct every address with zero
    # network round-trips and hand them straight to the bus.
    addrs = expert_addrs(n_modalities, per_modality, backbone_public_host,
                        base_port=backbone_base_port, seed=seed)
    reachable = {m: [] for m in range(n_modalities)}
    for m, entries in addrs.items():
        for nid, addr in entries:
            query.bus.peer_addrs[nid] = addr
            if nid not in query.node.peers:
                query.node.peers.append(nid)
            reachable[m].append(nid)
    _log("addrs_seeded", reachable={m: len(ids) for m, ids in reachable.items()},
         expected_per_modality=per_modality)

    # Brief settle: give in-flight replies a moment and let our own presence
    # propagate (helps trust/PEX bookkeeping), though correctness no longer
    # depends on it.
    for t in range(int(settle_s)):
        await asyncio.sleep(1.0)

    views = multiview_test_set(spec, n_per_class=n_per_class)
    y = next(iter(views.values()))[1]
    n = len(y)

    async def per_modality_label(m, i):
        # Query ONLY the modality's own resolved experts (never
        # discovery.peers_for_topic — its "unknown topic matches every topic"
        # fallback can include wrong-modality/wrong-task peers, which would send
        # a cross-modal query with an incompatible vector shape to an unrelated
        # node; core/prototype_model.py now rejects that defensively too, but
        # there is no reason to send it in the first place).
        cands = reachable[m]
        if not cands:
            return (None, 0.0)
        return await query.inference.infer(views[m][0][i], cands, top_n=20, timeout=timeout)

    solo_correct = {m: 0 for m in range(n_modalities)}
    fused_correct = 0
    for i in range(n):
        per_m = await asyncio.gather(*(per_modality_label(m, i) for m in range(n_modalities)))
        resp = []
        for m, (label, conf) in enumerate(per_m):
            if label is not None:
                resp.append(Response(responder=m, label=label, confidence=max(conf, 0.01), trust=1.0))
                solo_correct[m] += (label == y[i])
        fused_label, _ = aggregate(resp) if resp else (None, 0.0)
        fused_correct += (fused_label == y[i])

    await query.stop()
    solo = {m: solo_correct[m] / n for m in range(n_modalities)}
    fused = fused_correct / n
    return {"solo": solo, "fused": fused, "best_solo": max(solo.values()), "n": n,
           "reachable": {m: len(ids) for m, ids in reachable.items()}}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Query + fuse the live multimodal swarm")
    ap.add_argument("--beacon", default="beacon.swarmint.org")
    ap.add_argument("--beacon-id", default="8c30c97e7fd5460ce3b962db4cd75879eecd8abd")
    ap.add_argument("--n-modalities", type=int, default=3)
    ap.add_argument("--per-modality", type=int, default=5)
    ap.add_argument("--settle", type=float, default=5.0)
    ap.add_argument("--backbone-public-host", default=None,
                    help="backbone's advertised host (default: same as --beacon)")
    ap.add_argument("--backbone-base-port", type=int, default=9401)
    args = ap.parse_args()
    res = asyncio.run(run_multimodal_query(beacon_host=args.beacon, beacon_id_hex=args.beacon_id,
                                           n_modalities=args.n_modalities,
                                           per_modality=args.per_modality, settle_s=args.settle,
                                           backbone_public_host=args.backbone_public_host,
                                           backbone_base_port=args.backbone_base_port))
    print("\n== live multimodal fusion (query against the deployed backbone) ==")
    for m, a in res["solo"].items():
        print(f"  modality {m} distributed solo : {a:.3f}")
    print(f"  LATE FUSION (live, real net)      : {res['fused']:.3f}  "
         f"(best single {res['best_solo']:.3f}, n={res['n']})")
    ok = res["fused"] > res["best_solo"]
    print(f"\n{'PASS' if ok else 'FAIL'}: live fusion {'beat' if ok else 'did NOT beat'} "
         f"the best single modality against the deployed swarm.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
