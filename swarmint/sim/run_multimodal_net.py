"""Encoder-free multimodal late fusion over the REAL network stack.

run_multimodal.py proved fusion-beats-best-modality in-process over a SimBus. This
proves the SAME thing over real UDP sockets + Kademlia DHT discovery + msgpack wire
+ topic routing — the machinery a live deployment actually uses. It is the network
analogue of that demo, and the first time cross-modal fusion is exercised end-to-end
on the transport that carries the public swarm.

Topology (one asyncio process, real UdpBus per node on loopback):
  * 1 rendezvous (DHT bootstrap + relay), no data.
  * n_modalities x per_modality EXPERT nodes. Each modality m is its own TOPIC (D2):
    a node on topic m learns only modality m's noisy view, and gossips only within
    that topic. Honest coverage >=3 nodes/class WITHIN each modality (corroboration).
  * 1 QUERY node. It holds a joint multi-view test set and, per sample, runs
    topic-scoped distributed inference against each modality's experts, then fuses
    the per-modality answers with the SAME Phase-2 aggregate() used everywhere.

The point measured: fused accuracy across modalities > the best single modality —
independently-noisy views denoise when fused, and this now holds over real sockets.

Run:  python -m swarmint.sim.run_multimodal_net
"""

import asyncio
import random

import numpy as np
from nacl.signing import SigningKey

from ..inference.aggregate import Response, aggregate
from ..network.identity import Identity
from ..node.net_node import NetNode
from .data_multimodal import make_modalities, multiview_test_set, sample_modality
from .run_multiproc import _assign_classes_with_min_coverage, _required_classes_per_node


def _ident(seed: int) -> Identity:
    return Identity(SigningKey(seed.to_bytes(4, "big").rjust(32, b"\0")))


def _log(event, **kv):
    print(f"EVENT {event} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


async def run(n_modalities: int = 3, n_classes: int = 6, per_modality: int = 10,
              noise: float = 1.7, converge_s: float = 45.0, seed: int = 11,
              host: str = "127.0.0.1", base_port: int = 9401) -> dict:
    spec = make_modalities(n_modalities, n_classes, noise=noise, seed=seed)

    def _quiet(loop, ctx):
        if isinstance(ctx.get("exception"), asyncio.InvalidStateError):
            return
        loop.default_exception_handler(ctx)
    asyncio.get_running_loop().set_exception_handler(_quiet)

    # --- rendezvous (bootstrap + relay), no data of its own ---
    r_ident = _ident(seed)
    rendezvous = NetNode(identity=r_ident, topic=0, rng=random.Random(seed),
                         n_classes=n_classes, gossip_interval=0.4, gossip_sample_size=12,
                         enable_relay=True)
    await rendezvous.start(host, base_port, base_port + 1, [])
    r_id = r_ident.node_id
    _log("rendezvous_up", node_id=r_id.hex()[:8], gossip=f"{host}:{base_port}")

    # --- expert nodes, partitioned by modality (topic = m + 1) ---
    experts = []
    port = base_port + 2
    sk = seed * 1000
    for m in range(n_modalities):
        topic = m + 1
        cpn = _required_classes_per_node(per_modality, n_classes, 2, min_coverage=3)
        assign = _assign_classes_with_min_coverage(per_modality, n_classes, cpn,
                                                   random.Random(seed * 31 + m))
        for j in range(per_modality):
            sk += 1
            ident = _ident(sk)
            net = NetNode(identity=ident, topic=topic, rng=random.Random(sk),
                          n_classes=n_classes, gossip_interval=0.4, gossip_sample_size=12,
                          enable_relay=True)
            net.topics = {topic}
            classes = assign[j]
            np_rng = np.random.default_rng(sk)

            def feed(_m=m, _cls=classes, _rng=np_rng):
                xs, ys = sample_modality(spec[_m], _cls, 1, _rng)
                return (xs[0], int(ys[0]))
            net.data_feed = feed

            await net.start(host, port, port + 1, [(host, base_port + 1)],
                            rendezvous_id=r_id, rendezvous_addr=(host, base_port))
            port += 2
            experts.append((m, topic, net))
    _log("experts_up", n=len(experts), per_modality=per_modality, modalities=n_modalities)

    # --- query node: knows every topic via PEX, holds the joint test set ---
    q_ident = _ident(seed + 77)
    query = NetNode(identity=q_ident, topic=0, rng=random.Random(seed + 77),
                    n_classes=n_classes, gossip_interval=0.4, gossip_sample_size=12,
                    enable_relay=True)
    # advertise interest in every modality topic so PEX/topic routing surface experts
    query.topics = {m + 1 for m in range(n_modalities)}
    await query.start(host, port, port + 1, [(host, base_port + 1)],
                      rendezvous_id=r_id, rendezvous_addr=(host, base_port))

    # --- let the per-modality swarms converge + peers propagate via PEX ---
    for t in range(int(converge_s)):
        await asyncio.sleep(1.0)
        if (t + 1) % 10 == 0:
            _log("converging", t=t + 1, peers=len(query.node.peers))

    # --- resolve each peer's TOPIC authoritatively via its signed DHT advert.
    # PEX only spreads topic HINTS relayed by others; the advert lookup is the
    # trust boundary (discovery.ingest_pex docstring), and passively-heard peers
    # have no topic recorded, so peers_for_topic would otherwise fall back to
    # "unknown => matches every topic". Looking each peer up populates
    # peer_topics with its OWN signed claim, so topic routing filters correctly.
    await asyncio.gather(*(query.discovery.lookup(p) for p in list(query.node.peers)),
                         return_exceptions=True)
    seen = {m + 1: len(query.discovery.peers_for_topic(m + 1)) for m in range(n_modalities)}
    _log("topics_resolved", peers_by_topic=seen)

    # --- cross-modal inference: per sample, query each modality, fuse ---
    views = multiview_test_set(spec, n_per_class=20)
    y = next(iter(views.values()))[1]
    n = len(y)

    async def per_modality_label(m, i):
        topic = m + 1
        cands = query.discovery.peers_for_topic(topic) or list(query.node.peers)
        return await query.inference.infer(views[m][0][i], cands, top_n=20, timeout=0.4)

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

    solo = {m: solo_correct[m] / n for m in range(n_modalities)}
    fused = fused_correct / n

    for _, _, net in experts:
        await net.stop()
    await query.stop()
    await rendezvous.stop()

    return {"solo": solo, "fused": fused, "best_solo": max(solo.values()), "n": n}


def main():
    res = asyncio.run(run())
    print("\n== multimodal late fusion over REAL UDP + DHT + topics ==")
    for m, a in res["solo"].items():
        print(f"  modality {m} (topic {m + 1}) distributed solo : {a:.3f}")
    print(f"  LATE FUSION across modalities (real net)       : {res['fused']:.3f}  "
          f"(best single {res['best_solo']:.3f}, n={res['n']})")
    ok = res["fused"] > res["best_solo"]
    print(f"\n{'PASS' if ok else 'FAIL'}: fusion {'beats' if ok else 'did NOT beat'} the best single "
          f"modality over the real network stack, with no encoder.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
