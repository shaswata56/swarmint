"""swarmint command-line interface — the front door to the live swarm.

The point of this module is to erase the gap between "swarmint is a concept" and
"anyone can join the running swarm in one command". Under the hood a joining node
already works (see sim/net_daemon.py), but joining used to require ~8 SWARM_* env
vars AND knowing the beacon's hex node_id — friction that made the live network
effectively unreachable to a stranger. This CLI bakes in the public beacon's
address and identity so that, after `pip install swarmint[net,learn]`:

    swarmint join      # join the live global swarm at beacon.swarmint.org
    swarmint sim       # run the deterministic 100-node sim (no network)
    swarmint beacon    # host your own rendezvous/beacon
    swarmint status    # print the live beacon status

`join` and `beacon` delegate to the exact, test-covered net_daemon code path
(NAT discovery, hole-punch, relay fallback, gossip, metrics) — the CLI only sets
sensible defaults into the environment first, so there is one implementation of
the networking, not two.
"""

import argparse
import os
import random
import sys

# --- the public swarm's durable coordinates ------------------------------------
# The beacon is reached by NAME (a DNS A record), never a pinned IP, so re-hosting
# it is a DNS change only. Its node_id is DETERMINISTIC from SWARM_SEED=0, so it is
# a stable public constant every joiner can trust to identify the rendezvous.
BEACON_HOST = "beacon.swarmint.org"
BEACON_NODE_ID = "8c30c97e7fd5460ce3b962db4cd75879eecd8abd"
BEACON_DHT_PORT = 9002
BEACON_GOSSIP_PORT = 9001

# The live beacon hosts a shared "world" (identical class centers derived from a
# shared seed). A joiner MUST use the same world parameters to be learning the
# same task — these mirror net_daemon's defaults, which the beacon runs with.
WORLD_N_CLASSES = 10
WORLD_DIM = 16
WORLD_SEED = 0


def _setdefault_env(**kv):
    """Set an env var only if the user hasn't already — CLI flags/explicit env win."""
    for k, v in kv.items():
        if v is not None and os.environ.get(k) in (None, ""):
            os.environ[k] = str(v)


def _run_net_daemon() -> int:
    import asyncio
    from .sim import net_daemon
    return asyncio.run(net_daemon.main())


def cmd_join(args) -> int:
    # A joiner needs its OWN identity: SWARM_SEED=0 would collide with the beacon.
    # Pick a random non-zero seed (also selects a random 2-class non-IID slice, so
    # independent joiners naturally spread coverage across the 10 classes).
    seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
    _setdefault_env(
        SWARM_ROLE="node",
        SWARM_SEED=seed,
        SWARM_RENDEZVOUS_HOST=args.beacon,
        SWARM_RENDEZVOUS_ID=args.beacon_id,
        SWARM_RENDEZVOUS_DHT_PORT=args.dht_port,
        SWARM_RENDEZVOUS_GOSSIP_PORT=args.gossip_port,
        SWARM_N_CLASSES=WORLD_N_CLASSES,
        SWARM_DIM=WORLD_DIM,
        SWARM_WORLD_SEED=WORLD_SEED,
        SWARM_DURATION_S=args.duration,   # 0 = run until Ctrl-C
        SWARM_REPORT_EVERY_S=args.report_every,
        SWARM_ENABLE_RELAY=("1" if args.relay else "0"),
        SWARM_TASK=args.task,
    )
    print(f"swarmint: joining the live swarm at {args.beacon} "
          f"(rendezvous {args.beacon_id[:8]}…, seed {seed})")
    print("swarmint: each `EVENT metrics` line below shows this node's held-out "
          "accuracy (acc=) climbing as the swarm covers more of the 10 classes.")
    print("swarmint: Ctrl-C to leave.\n")
    try:
        return _run_net_daemon()
    except KeyboardInterrupt:
        print("\nswarmint: left the swarm.")
        return 0


def cmd_beacon(args) -> int:
    # A hosted beacon needs its OWN identity. Default to a random non-zero seed so
    # `swarmint beacon` (one command, no flags) is a UNIQUE peer that bootstraps
    # from the genesis — NOT seed 0, which is the genesis's own id (that would
    # collide, and net_daemon would think this box IS the genesis). Only the
    # genesis itself is deployed with an explicit seed 0 (via its env, not this CLI).
    seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
    _setdefault_env(
        SWARM_ROLE="rendezvous",
        SWARM_SEED=seed,
        SWARM_ADVERTISE_HOST=args.bind,
        SWARM_PUBLIC_HOST=args.public_host,
        SWARM_GOSSIP_PORT=args.gossip_port,
        SWARM_DHT_PORT=args.dht_port,
        SWARM_N_CLASSES=WORLD_N_CLASSES,
        SWARM_DIM=WORLD_DIM,
        SWARM_WORLD_SEED=WORLD_SEED,
        SWARM_DURATION_S=args.duration,
        SWARM_HTTP_PORT=(args.http_port or None),
        SWARM_ENABLE_RELAY="1",
        SWARM_TASK=args.task,
        # Beacon federation: by default a beacon joins the global mesh by bootstrapping
        # from the genesis beacon (beacon.swarmint.org — the well-known entry point, not
        # an authority). It then holds the full directory and gossips peer-to-peer like
        # every other beacon. --no-federate opts out; --genesis points at any beacon.
        SWARM_FEDERATION=("0" if args.no_federate else "1"),
        SWARM_GENESIS_HOST=(None if args.no_federate else args.genesis),
        SWARM_GENESIS_ID=(None if args.no_federate else args.genesis_id),
        SWARM_GENESIS_GOSSIP_PORT=(None if args.no_federate else args.genesis_gossip_port),
        SWARM_BEACON_NAME=(args.name or None),
    )
    print(f"swarmint: hosting a beacon (seed {seed}); node_id is deterministic "
          f"from the seed. Others join with:\n"
          f"  swarmint join --beacon <this-host> --beacon-id <node_id>\n")
    if not args.no_federate:
        print(f"swarmint: joining the federation via genesis {args.genesis} — run with --http-port "
              f"to see the beacon directory (every beacon holds the full map), or check the genesis "
              f"page to confirm this beacon is reachable.\n")
    return _run_net_daemon()


def cmd_beacons(args) -> int:
    """List the beacon federation as seen by a beacon's /federation.json (default:
    the genesis beacon over HTTPS; --url for a local/plain-HTTP beacon). Every beacon
    holds the full directory, so any beacon's view is the whole mesh."""
    import json
    import urllib.request
    url = args.url or f"https://{args.beacon}/federation.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        print(f"swarmint: could not reach {url}: {e}", file=sys.stderr)
        return 1
    fed = data.get("federation", [])
    print(f"beacon {data.get('beacon_id', '?')[:16]}.. knows {len(fed)} beacon(s), "
          f"{data.get('n_beacons_reachable', 0)} reachable:\n")
    if not fed:
        print("  (no other beacons known yet)")
        return 0
    print(f"  {'reach':<7} {'name':<16} {'endpoint':<24} {'task':<10} beacon-id")
    for b in fed:
        reach = "up" if b["reachable"] else "-"
        endpoint = f"{b['host']}:{b['gossip_port']}"
        print(f"  {reach:<7} {(b['name'] or '-'):<16} {endpoint:<24} "
              f"{(b['task'] or '-'):<10} {b['id'][:16]}..")
    return 0


def cmd_backbone(args) -> int:
    import asyncio
    from .sim.swarm_backbone import run_backbone
    print(f"swarmint: starting an always-on honest backbone of {args.n} nodes against "
          f"{args.beacon} — gives joiners the >=3-sender quorum they need to learn.")
    return asyncio.run(run_backbone(
        beacon_host=args.beacon, beacon_id_hex=args.beacon_id, n=args.n,
        base_port=args.base_port, dht_base_port=args.dht_base_port,
        beacon_gossip_port=args.gossip_port, beacon_dht_port=args.dht_port,
        advertise=args.bind, public_host=args.public_host,
        duration_s=args.duration, task_name=args.task))


def cmd_sim(args) -> int:
    from .sim import run_sim
    run_sim.main()
    return 0


def cmd_multimodal_demo(args) -> int:
    """Spin up a whole multimodal swarm locally over the REAL UDP/DHT stack
    (rendezvous + per-modality expert topics + a fusing query node) and show that
    late fusion across modalities beats the best single modality — no encoder."""
    from .sim import run_multimodal_net
    return run_multimodal_net.main()


def cmd_multimodal_backbone(args) -> int:
    """Attach an always-on multimodal honest backbone (N experts/modality, each
    modality its own topic) to a real rendezvous (default: the public beacon) so
    a later `swarmint multimodal-query` can fuse across modalities live."""
    import asyncio
    from .sim.multimodal_backbone import run_multimodal_backbone
    print(f"swarmint: starting a multimodal backbone ({args.n_modalities} modalities x "
          f"{args.per_modality} experts) against {args.beacon}.")
    return asyncio.run(run_multimodal_backbone(
        beacon_host=args.beacon, beacon_id_hex=args.beacon_id,
        n_modalities=args.n_modalities, per_modality=args.per_modality,
        base_port=args.base_port, dht_base_port=args.dht_base_port,
        beacon_gossip_port=args.gossip_port, beacon_dht_port=args.dht_port,
        advertise=args.bind, public_host=args.public_host, duration_s=args.duration))


def cmd_multimodal_query(args) -> int:
    """One-shot: join a real rendezvous, construct each modality expert's
    deterministic address, and fuse per-modality answers — the live measurement
    of a standing `swarmint multimodal-backbone` swarm."""
    import asyncio
    from .sim.multimodal_query import run_multimodal_query
    res = asyncio.run(run_multimodal_query(
        beacon_host=args.beacon, beacon_id_hex=args.beacon_id,
        n_modalities=args.n_modalities, per_modality=args.per_modality, settle_s=args.settle,
        beacon_gossip_port=args.gossip_port, beacon_dht_port=args.dht_port,
        backbone_public_host=args.backbone_public_host, backbone_base_port=args.backbone_base_port))
    print("\n== live multimodal fusion (query against the deployed backbone) ==")
    for m, a in res["solo"].items():
        print(f"  modality {m} distributed solo : {a:.3f}")
    print(f"  LATE FUSION (live, real net)      : {res['fused']:.3f}  "
         f"(best single {res['best_solo']:.3f}, n={res['n']})")
    ok = res["fused"] > res["best_solo"]
    print(f"\n{'PASS' if ok else 'FAIL'}: live fusion {'beat' if ok else 'did NOT beat'} "
         f"the best single modality.")
    return 0 if ok else 1


def cmd_infer(args) -> int:
    """One-shot: join the beacon, reconstruct the digits backbone's deterministic
    addresses, embed a query digit through the shared genesis, fan out with the
    trust-ranked InferenceService, and print the swarm's answer + every expert's
    vote — then exit (no long-running learning node)."""
    import asyncio
    from .sim.infer_query import _print_report, run_infer
    res = asyncio.run(run_infer(
        beacon_host=args.beacon, beacon_id_hex=args.beacon_id, n=args.n,
        backbone_public_host=args.backbone_public_host,
        backbone_base_port=args.backbone_base_port, settle_s=args.settle, timeout=args.timeout,
        index=args.index, image_path=args.image, eval_n=args.eval_n))
    return _print_report(res)


def cmd_status(args) -> int:
    import urllib.request
    url = f"https://{args.beacon}/"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 (fixed https host)
            body = r.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"swarmint: could not reach {url}: {e}", file=sys.stderr)
        return 1
    # The status page is HTML; print a terse text view without pulling in a parser.
    import re
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"\s+", " ", text).strip()
    print(text[:1200])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swarmint",
        description="Decentralized, Byzantine-robust swarm learning over a real P2P network.")
    sub = p.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("join", help="join the live global swarm at beacon.swarmint.org")
    j.add_argument("--beacon", default=BEACON_HOST, help=f"rendezvous host (default {BEACON_HOST})")
    j.add_argument("--beacon-id", default=BEACON_NODE_ID, help="rendezvous node_id (hex)")
    j.add_argument("--dht-port", type=int, default=BEACON_DHT_PORT)
    j.add_argument("--gossip-port", type=int, default=BEACON_GOSSIP_PORT)
    j.add_argument("--seed", type=int, default=None,
                   help="node identity seed (default: random; also picks the class slice)")
    j.add_argument("--duration", type=int, default=0, help="seconds to run (0 = until Ctrl-C)")
    j.add_argument("--report-every", type=int, default=5, help="metrics cadence, seconds")
    j.add_argument("--relay", action="store_true",
                   help="fall back to relaying via the beacon if a direct punch fails")
    j.add_argument("--task", default="digits", choices=["digits", "synthetic"],
                   help="what the swarm learns (default: real sklearn digits)")
    j.set_defaults(func=cmd_join)

    b = sub.add_parser("beacon", help="host your own beacon (one command; joins the federation)")
    b.add_argument("--seed", type=int, default=None,
                   help="identity seed (default: random unique peer; do not use 0 — that's the genesis)")
    b.add_argument("--bind", default="0.0.0.0", help="bind host")
    b.add_argument("--public-host", default=None, help="public IP to advertise (1:1-NAT cloud hosts)")
    b.add_argument("--gossip-port", type=int, default=BEACON_GOSSIP_PORT)
    b.add_argument("--dht-port", type=int, default=BEACON_DHT_PORT)
    b.add_argument("--http-port", type=int, default=0, help="status page port (0 = off)")
    b.add_argument("--duration", type=int, default=0, help="seconds to run (0 = forever)")
    b.add_argument("--task", default="digits", choices=["digits", "synthetic"],
                   help="what the swarm learns (default: real sklearn digits)")
    b.add_argument("--name", default=None, help="human label for this beacon in the directory")
    b.add_argument("--genesis", default=BEACON_HOST,
                   help=f"beacon to bootstrap the federation from (default: the genesis, {BEACON_HOST})")
    b.add_argument("--genesis-id", default=BEACON_NODE_ID, help="bootstrap beacon node_id (hex)")
    b.add_argument("--genesis-gossip-port", type=int, default=BEACON_GOSSIP_PORT)
    b.add_argument("--no-federate", action="store_true",
                   help="run standalone; do not join the beacon federation")
    b.set_defaults(func=cmd_beacon)

    bcs = sub.add_parser("beacons", help="list the beacon federation (directory of beacons)")
    bcs.add_argument("--beacon", default=BEACON_HOST, help="beacon whose directory to read (HTTPS)")
    bcs.add_argument("--url", default=None,
                     help="explicit /federation.json URL (e.g. a local http://host:port beacon)")
    bcs.set_defaults(func=cmd_beacons)

    bb = sub.add_parser("backbone",
                        help="run an always-on honest backbone so joiners have a learning quorum")
    bb.add_argument("--beacon", default=BEACON_HOST, help="rendezvous host to attach to")
    bb.add_argument("--beacon-id", default=BEACON_NODE_ID, help="rendezvous node_id (hex)")
    bb.add_argument("--n", type=int, default=8, help="number of honest backbone nodes")
    bb.add_argument("--base-port", type=int, default=9003, help="first gossip port (n consecutive)")
    bb.add_argument("--dht-base-port", type=int, default=9103, help="first DHT port (n consecutive)")
    bb.add_argument("--gossip-port", type=int, default=BEACON_GOSSIP_PORT, help="beacon gossip port")
    bb.add_argument("--dht-port", type=int, default=BEACON_DHT_PORT, help="beacon DHT port")
    bb.add_argument("--bind", default="0.0.0.0", help="bind host")
    bb.add_argument("--public-host", default=None, help="public IP to advertise (1:1-NAT cloud)")
    bb.add_argument("--duration", type=int, default=0, help="seconds to run (0 = forever)")
    bb.add_argument("--task", default="digits", choices=["digits", "synthetic"],
                    help="what the swarm learns (default: real sklearn digits)")
    bb.set_defaults(func=cmd_backbone)

    s = sub.add_parser("sim", help="run the deterministic 100-node simulation (no network)")
    s.set_defaults(func=cmd_sim)

    mm = sub.add_parser("multimodal-demo",
                        help="multimodal late fusion over the real UDP/DHT stack (no encoder)")
    mm.set_defaults(func=cmd_multimodal_demo)

    mb = sub.add_parser("multimodal-backbone",
                        help="attach an always-on multimodal honest backbone to a rendezvous")
    mb.add_argument("--beacon", default=BEACON_HOST)
    mb.add_argument("--beacon-id", default=BEACON_NODE_ID)
    mb.add_argument("--n-modalities", type=int, default=3)
    mb.add_argument("--per-modality", type=int, default=5)
    mb.add_argument("--base-port", type=int, default=9101, help="first gossip port (n consecutive)")
    mb.add_argument("--dht-base-port", type=int, default=9201, help="first DHT port (n consecutive)")
    mb.add_argument("--gossip-port", type=int, default=BEACON_GOSSIP_PORT, help="beacon gossip port")
    mb.add_argument("--dht-port", type=int, default=BEACON_DHT_PORT, help="beacon DHT port")
    mb.add_argument("--bind", default="0.0.0.0")
    mb.add_argument("--public-host", default=None, help="public IP to advertise (1:1-NAT cloud)")
    mb.add_argument("--duration", type=int, default=0, help="seconds to run (0 = forever)")
    mb.set_defaults(func=cmd_multimodal_backbone)

    mq = sub.add_parser("multimodal-query",
                        help="join + fuse across modalities against a live multimodal-backbone")
    mq.add_argument("--beacon", default=BEACON_HOST)
    mq.add_argument("--beacon-id", default=BEACON_NODE_ID)
    mq.add_argument("--n-modalities", type=int, default=3)
    mq.add_argument("--per-modality", type=int, default=5, help="experts/modality (must match the backbone)")
    mq.add_argument("--gossip-port", type=int, default=BEACON_GOSSIP_PORT, help="beacon gossip port")
    mq.add_argument("--dht-port", type=int, default=BEACON_DHT_PORT, help="beacon DHT port")
    mq.add_argument("--settle", type=float, default=5.0, help="seconds to let replies/PEX settle")
    mq.add_argument("--backbone-public-host", default=None,
                    help="backbone's advertised host (default: same as --beacon)")
    mq.add_argument("--backbone-base-port", type=int, default=9401)
    mq.set_defaults(func=cmd_multimodal_query)

    inf = sub.add_parser("infer",
                         help="ask the live swarm to classify a digit (one-shot, no learning node)")
    inf.add_argument("--beacon", default=BEACON_HOST)
    inf.add_argument("--beacon-id", default=BEACON_NODE_ID)
    inf.add_argument("--n", type=int, default=8, help="backbone size (must match the deployment)")
    inf.add_argument("--index", type=int, default=None, help="held-out test digit index (default 0)")
    inf.add_argument("--image", default=None, help="file of 64 raw pixels (8x8 sklearn digit)")
    inf.add_argument("--eval", type=int, default=None, dest="eval_n",
                     help="classify the first N held-out digits and report live accuracy")
    inf.add_argument("--settle", type=float, default=3.0, help="seconds to let the join settle")
    inf.add_argument("--timeout", type=float, default=1.0, help="per-expert query timeout, seconds")
    inf.add_argument("--backbone-public-host", default=None,
                     help="backbone advertised host (default: same as --beacon)")
    inf.add_argument("--backbone-base-port", type=int, default=9003)
    inf.set_defaults(func=cmd_infer)

    st = sub.add_parser("status", help="print the live beacon status")
    st.add_argument("--beacon", default=BEACON_HOST)
    st.set_defaults(func=cmd_status)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
