"""One-shot INFERENCE client — the front door to the live swarm's knowledge.

`swarmint join` lets a stranger CONTRIBUTE (spin up a learning node and watch it
converge). This is the other half: let a stranger ASK the swarm a question without
becoming a long-running peer. It joins the beacon, reconstructs the digits
backbone's deterministic addresses (no DHT/PEX wait — decision #043/#050), embeds a
query digit locally through the SAME frozen LDA genesis every backbone node uses,
fans the query out to the backbone with the existing trust-ranked InferenceService,
and prints the swarm's answer plus every expert's vote — then exits.

Trust calibration (honest decision): a transient querier has no local holdout to
run calibrate_trust against, so it uses UNIFORM trust. aggregate() then reduces to
confidence-weighted plurality with abstention — which is exactly right when the
target is the known-honest public backbone. The Byzantine-inference result (0.79
with 30% liars, PAPER §3.2) assumed CALIBRATED trust and is a separate claim; this
front door does not inherit it and does not pretend to. Against liars, calibrate
first (needs a holdout) or rank by a reputation you already hold.
"""

import asyncio
import random
import socket

import numpy as np
from nacl.signing import SigningKey

from ..inference.aggregate import Response, aggregate
from ..network.identity import Identity
from ..node.net_node import NetNode
from ..node.swarm_node import TRUST_INIT
from .live_task import build_task
from .swarm_backbone import backbone_addrs


def _ident(seed: int) -> Identity:
    return Identity(SigningKey(seed.to_bytes(4, "big").rjust(32, b"\0")))


def _log(event, **kv):
    print(f"EVENT {event} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def load_query_vectors(task, *, index=None, image_path=None, eval_n=None):
    """Return (embedded_xs, true_ys, source_desc) for whatever the user asked to
    classify, ALWAYS in the shared embedded space the backbone models live in.

    - image_path: a file of 64 whitespace/comma-separated raw pixels (sklearn
      digits scale, 0..16). Normalized + embedded through the genesis LDA. No
      true label known -> y is None.
    - eval_n: the first N held-out test digits (true labels known -> reports live
      accuracy against the swarm).
    - index (default 0): a single held-out test digit by index.

    task.test_x is ALREADY embedded (live_task._digits transforms it), so indexed
    samples pass straight through; a raw image is normalized + transformed here."""
    if image_path is not None:
        from .real_data import normalize
        with open(image_path) as fh:
            text = fh.read()
        raw = np.fromstring(text.replace(",", " "), sep=" ")
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.size != 64:
            raise ValueError(f"expected 64 pixels (8x8 digit), got {raw.size} from {image_path}")
        x = normalize(raw.reshape(1, -1))
        z = task.embedding.transform(x)  # into the shared space
        return z, [None], f"image {image_path}"
    if eval_n is not None:
        n = min(int(eval_n), len(task.test_y))
        return task.test_x[:n], list(task.test_y[:n]), f"{n} held-out test digits"
    i = 0 if index is None else int(index)
    if not (0 <= i < len(task.test_y)):
        raise ValueError(f"--index {i} out of range (0..{len(task.test_y) - 1})")
    return task.test_x[i:i + 1], [int(task.test_y[i])], f"held-out test digit #{i}"


async def run_infer(*, beacon_host: str, beacon_id_hex: str, n: int = 8,
                    backbone_public_host: str = None, backbone_base_port: int = 9003,
                    seed_base: int = 1000, world_seed: int = 0, n_classes: int = 10,
                    beacon_gossip_port: int = 9001, beacon_dht_port: int = 9002,
                    advertise: str = "0.0.0.0", gossip_port: int = 9201, dht_port: int = 9202,
                    settle_s: float = 3.0, timeout: float = 1.0,
                    index=None, image_path=None, eval_n=None) -> dict:
    beacon_id = bytes.fromhex(beacon_id_hex)
    r_ip = socket.gethostbyname(beacon_host)
    backbone_public_host = backbone_public_host or r_ip

    # Build the digits task locally: SAME (world_seed, bundled data) as the deployed
    # backbone => the identical frozen LDA genesis embedding and comparable held-out
    # digits, with zero download and zero coordination (shared-space invariant).
    task = build_task("digits", n_classes=n_classes, world_seed=world_seed)
    xs, ys, source = load_query_vectors(task, index=index, image_path=image_path, eval_n=eval_n)

    def _quiet(loop, ctx):
        if isinstance(ctx.get("exception"), asyncio.InvalidStateError):
            return
        loop.default_exception_handler(ctx)
    asyncio.get_running_loop().set_exception_handler(_quiet)

    q_ident = _ident(random.randint(1, 2**31 - 1))
    query = NetNode(identity=q_ident, topic=1, rng=random.Random(1), n_classes=n_classes,
                    gossip_interval=1.0, gossip_sample_size=12, enable_relay=True,
                    embedding=task.embedding, radii=(task.radii or None))
    await query.start(advertise, gossip_port, dht_port, [(r_ip, beacon_dht_port)],
                      rendezvous_id=beacon_id, rendezvous_addr=(r_ip, beacon_gossip_port))
    _log("joined", rendezvous=beacon_host, node_id=q_ident.node_id.hex()[:8])

    # Deterministic backbone address book — no DHT/PEX wait (see backbone_addrs).
    addrs = backbone_addrs(n, backbone_public_host, base_port=backbone_base_port,
                           seed_base=seed_base)
    peer_ids = []
    for nid, addr in addrs:
        query.bus.peer_addrs[nid] = addr
        if nid not in query.node.peers:
            query.node.peers.append(nid)
        peer_ids.append(nid)
    _log("backbone_seeded", n=len(peer_ids), host=backbone_public_host,
         ports=f"{backbone_base_port}..{backbone_base_port + n - 1}")

    for _ in range(int(settle_s)):
        await asyncio.sleep(1.0)

    async def classify(xq):
        """Query every backbone node once, keep per-expert votes, then run the
        SAME pure aggregate() InferenceService.infer() uses (uniform trust)."""
        results = await asyncio.gather(*(query.inference.query_peer(pid, xq, timeout=timeout)
                                         for pid in peer_ids))
        responses, votes = [], []
        for pid, (label, conf) in zip(peer_ids, results):
            votes.append({"node": pid.hex()[:8], "label": label, "confidence": round(float(conf), 3)})
            if label is not None:
                responses.append(Response(responder=pid, label=label, confidence=conf,
                                          trust=TRUST_INIT))
        label, score = aggregate(responses) if responses else (None, 0.0)
        return label, score, votes

    predictions = []
    correct = 0
    for x, y in zip(xs, ys):
        label, score, votes = await classify(x)
        predictions.append({"true": y, "label": label, "score": round(float(score), 3),
                            "votes": votes})
        if y is not None and label == y:
            correct += 1

    await query.stop()
    labeled = sum(1 for y in ys if y is not None)
    return {"source": source, "predictions": predictions, "n": len(ys),
            "labeled": labeled, "correct": correct,
            "accuracy": (correct / labeled) if labeled else None,
            "n_backbone": len(peer_ids)}


def _print_report(res: dict) -> int:
    print(f"\n== swarmint inference over the live swarm ({res['source']}) ==")
    print(f"   fanned out to {res['n_backbone']} backbone experts\n")
    single = res["n"] == 1
    for k, p in enumerate(res["predictions"]):
        heard = [v for v in p["votes"] if v["label"] is not None]
        head = f"  prediction: {p['label']}  (agg score {p['score']}, {len(heard)}/{len(p['votes'])} experts answered)"
        if p["true"] is not None:
            head += f"  [true {p['true']} -> {'CORRECT' if p['label'] == p['true'] else 'wrong'}]"
        print(head)
        if single:
            for v in p["votes"]:
                vote = "abstain" if v["label"] is None else f"digit {v['label']} @ {v['confidence']}"
                print(f"      expert {v['node']}: {vote}")
    if res["accuracy"] is not None and res["n"] > 1:
        print(f"\n  live swarm accuracy: {res['correct']}/{res['labeled']} = {res['accuracy']:.3f}")
    ok = res["accuracy"] is None or res["accuracy"] > 0.0
    return 0 if ok else 1


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ask the live swarm to classify a digit (one-shot)")
    ap.add_argument("--beacon", default="beacon.swarmint.org")
    ap.add_argument("--beacon-id", default="8c30c97e7fd5460ce3b962db4cd75879eecd8abd")
    ap.add_argument("--n", type=int, default=8, help="backbone size (must match the deployment)")
    ap.add_argument("--index", type=int, default=None, help="held-out test digit index (default 0)")
    ap.add_argument("--image", default=None, help="file of 64 raw pixels (8x8 sklearn digit)")
    ap.add_argument("--eval", type=int, default=None, dest="eval_n",
                    help="classify the first N held-out digits and report live accuracy")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument("--backbone-public-host", default=None,
                    help="backbone advertised host (default: same as --beacon)")
    ap.add_argument("--backbone-base-port", type=int, default=9003)
    args = ap.parse_args()
    res = asyncio.run(run_infer(
        beacon_host=args.beacon, beacon_id_hex=args.beacon_id, n=args.n,
        backbone_public_host=args.backbone_public_host,
        backbone_base_port=args.backbone_base_port, settle_s=args.settle, timeout=args.timeout,
        index=args.index, image_path=args.image, eval_n=args.eval_n))
    return _print_report(res)


if __name__ == "__main__":
    raise SystemExit(main())
