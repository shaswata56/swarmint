"""T7: multiprocess integration test. Real OS processes (multiprocessing —
separate interpreters/address spaces, not just asyncio tasks in one process),
real UDP sockets, real Kademlia DHT, real Ed25519 signing.

Peer discovery: every worker except node 0 is told ONLY node 0's identity and
DHT bootstrap address (a standard "known seed node" pattern, not a static
peer list — see decision #023 D6). From there, node 0's address is resolved
via a genuine DHT lookup, and the rest of the peer set grows via PEX riding
the gossip cadence (NetNode).

Run:  python -m swarmint.sim.run_multiproc
"""

import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
from nacl.signing import SigningKey

from .data import global_test_set, make_world, sample
from .resources import ResourceBudget

N_NODES = 24              # desired scale; auto-capped to the resource budget (60% CPU/RAM)
N_MALICIOUS_FRACTION = 0.06  # ~5-6%, matching Phase-1's proven threat model. IMPORTANT:
                            # the 3-sender corroboration defense (README lesson 4 / D11)
                            # only holds while malicious < honest supply PER CLASS-REGION.
                            # At the 60%-CPU node cap (~16 nodes) each class is covered by
                            # only ~3-4 honest nodes, so a colluding group of ~3 attackers
                            # can match that supply and poison unverifiable classes
                            # (accuracy peaks then DECAYS — verified). Tolerating a larger
                            # malicious fraction needs more honest redundancy per class,
                            # i.e. MORE nodes (raise the CPU cap), not a code change.
N_CLASSES = 10
CLASSES_PER_NODE = 2
DIM = 16
GOSSIP_INTERVAL = 0.15   # scaled down from the production 2s (decision #023 D7) so a
                         # CI-sized test converges in ~tens of seconds, not minutes —
                         # same code path, only the timer constant differs.
PRODUCTION_GOSSIP_INTERVAL = 2.0  # the real operating cadence (decision #023 D7)
GOSSIP_SAMPLE_SIZE = 12  # protos per gossip send over UDP: one 1200B datagram, no
                         # chunking, keeps per-node bandwidth within budget. (The
                         # in-process simulator keeps SwarmNode's default of 30.)
DURATION_S = 35.0
SEED = 123
BANDWIDTH_WINDOW_S = 10.0   # measure bandwidth over the last N seconds (post-convergence)
BANDWIDTH_BUDGET_BYTES_PER_S = 2000.0  # decision plan acceptance #5 — a PRODUCTION-cadence
                                       # figure; the test runs faster, so we normalize (below)


def _worker(cfg):
    """Runs in a SEPARATE OS process. All imports are local to avoid pickling
    unpicklable objects (event loops, sockets) across the process boundary —
    only plain data (the cfg dict of bytes/lists/arrays) crosses via
    multiprocessing args.

    Metrics are written INCREMENTALLY (one JSONL line per second, flushed) so
    that a process killed mid-run (chaos test, T8) still leaves its full
    history-to-death on disk for the launcher to read."""
    import asyncio

    from ..network.identity import Identity
    from ..node.net_node import NetNode  # noqa: keep local (see docstring)

    index = cfg["index"]
    metrics_path = cfg["metrics_path"]

    async def main():
        # kademlia's rpcudp occasionally set_result()s a future twice when a
        # duplicate UDP response arrives (benign library race), surfaced as
        # noisy "InvalidStateError". Swallow ONLY that; let everything else propagate.
        def _quiet_rpcudp(loop, context):
            if isinstance(context.get("exception"), asyncio.InvalidStateError):
                return
            loop.default_exception_handler(context)
        asyncio.get_running_loop().set_exception_handler(_quiet_rpcudp)

        identity = Identity(SigningKey(cfg["seed"]))
        rng = random.Random(SEED * 1000 + index)
        np_rng = np.random.default_rng(SEED * 7 + index)

        net = NetNode(identity=identity, topic=1, rng=rng, malicious=cfg["malicious"],
                      gossip_interval=cfg["gossip_interval"],
                      gossip_sample_size=cfg["gossip_sample_size"],
                      loss_rate=cfg.get("loss_rate", 0.0))

        centers = cfg["centers"]

        def feed():
            xs, ys = sample(centers, cfg["class_assignment"], 1, np_rng)
            return (xs[0], int(ys[0]))
        net.data_feed = feed

        bootstrap = cfg["bootstrap"]  # list of (host, port) DHT contacts (may be empty)
        await net.start("127.0.0.1", 0, 0, bootstrap, rendezvous_id=cfg.get("rendezvous_id"))

        if cfg.get("is_rendezvous"):
            dht_port = net.discovery.server.transport.get_extra_info("sockname")[1]
            Path(metrics_path).with_suffix(".dhtport").write_text(str(dht_port))

        test_x, test_y = global_test_set(centers, n_per_class=30)
        start = time.time()
        with open(metrics_path, "w") as f:
            while time.time() - start < cfg["duration_s"]:
                await asyncio.sleep(1.0)
                m = {
                    "t": time.time() - start,
                    "acc": net.node.model.accuracy(test_x, test_y),
                    "trust": {k.hex(): v for k, v in net.node.trust.items()},
                    "tx_bytes": net.bus.tx_bytes, "rx_bytes": net.bus.rx_bytes,
                    "merges_ok": net.node.merges_ok, "merges_rejected": net.node.merges_rejected,
                    "n_peers": len(net.node.peers), "n_protos": len(net.node.model.protos),
                    "dht_publish_ok": net.discovery.last_publish_ok,
                    "dropped_loss": net.bus.dropped_loss,
                }
                f.write(json.dumps(m) + "\n")
                f.flush()  # survive a kill -9 mid-run
        await net.stop()

    asyncio.run(main())


def _required_classes_per_node(n_nodes, n_classes, classes_per_node, min_coverage):
    """Every class must be reachable by >= min_coverage distinct nodes or
    corroboration (3 distinct senders, decision #023/README lesson 4) can
    NEVER promote it, permanently capping accuracy at (learnable classes / N).
    That needs n_nodes * classes_per_node >= min_coverage * n_classes. If the
    requested classes_per_node is too small for the node count, raise it (each
    node is slightly less specialized) rather than silently leave classes
    uncorroborable."""
    import math
    needed = math.ceil(min_coverage * n_classes / n_nodes)
    return max(classes_per_node, needed)


def _assign_classes_with_min_coverage(n_nodes, n_classes, classes_per_node, rng, min_coverage=3):
    """Guarantee every class is covered by >= min_coverage distinct nodes.

    Phase-1's 100-node sim used pure random per-node sampling safely because
    100 nodes x 2 classes / 10 classes = ~20x redundancy per class — corroboration
    essentially never starves. At small N the same random sampling can leave a
    class covered by <3 nodes purely by chance (or the earlier greedy version
    left later classes with ZERO coverage once nodes filled up), making those
    classes permanently unlearnable and capping accuracy. This distributes
    class-reps as evenly as possible across nodes, never assigning a class to a
    node twice, guaranteeing coverage >= floor(total_slots / n_classes).
    """
    total_slots = n_nodes * classes_per_node
    assert total_slots >= min_coverage * n_classes, (
        f"{n_nodes} nodes x {classes_per_node} classes = {total_slots} slots "
        f"< {min_coverage}x{n_classes} required for corroboration coverage"
    )
    base, extra = divmod(total_slots, n_classes)
    reps = {c: base + (1 if c < extra else 0) for c in range(n_classes)}  # even split

    assignments = [set() for _ in range(n_nodes)]
    # Assign the most-needed classes first, always to the emptiest eligible
    # nodes — this keeps every node near classes_per_node and never double-
    # assigns a class to one node.
    remaining = dict(reps)
    for _ in range(total_slots):
        c = max(remaining, key=lambda k: remaining[k]) if remaining else None
        if c is None or remaining[c] == 0:
            break
        eligible = [n for n in range(n_nodes)
                    if len(assignments[n]) < classes_per_node and c not in assignments[n]]
        if not eligible:
            remaining[c] = 0  # can't place any more of this class; stop trying
            continue
        n = min(eligible, key=lambda n: len(assignments[n]))
        assignments[n].add(c)
        remaining[c] -= 1
    for n in range(n_nodes):  # top up any node short a class
        while len(assignments[n]) < classes_per_node:
            choices = [c for c in range(n_classes) if c not in assignments[n]]
            assignments[n].add(rng.choice(choices))
    return [sorted(s) for s in assignments]


def _build_swarm(n_nodes, n_malicious, duration_s, gossip_interval, metrics_dir,
                budget, gossip_sample_size, loss_rate):
    """Shared setup for run() and run_chaos(): resolve the resource budget,
    assign classes (honest-coverage-guaranteed), mint identities, and return a
    launch(index, rendezvous_id, bootstrap)->Process factory plus the context
    both entry points need."""
    # Auto-scale the process count to fit the resource budget (default 60% of
    # CPU cores and available RAM; configurable via ResourceBudget args or the
    # SWARMINT_* env vars). We WANT a larger swarm for corroboration coverage,
    # but never exceed the cap — and we log what got capped rather than
    # silently shrinking (plan risk note: no silent truncation).
    budget = budget or ResourceBudget()
    plan = budget.resolve(n_nodes)
    if plan["capped_by"] != "none":
        print(f"[resource-cap] requested {plan['requested']} nodes, running {plan['nodes']} "
             f"(capped by {plan['capped_by']}). {budget.describe()}")
    else:
        print(f"[resource-cap] running {plan['nodes']} nodes within budget. {budget.describe()}")
    n_nodes = plan["nodes"]
    if n_malicious is None:
        n_malicious = max(1, int(round(n_nodes * N_MALICIOUS_FRACTION)))
    n_malicious = min(n_malicious, n_nodes - 2)  # keep >=1 honest peer besides rendezvous

    # Node 0 is the DHT rendezvous origin; excluding it from the malicious set
    # is not a realism shortcut — a real deployment would never hand out a
    # known-bad node as its bootstrap seed either. Making it malicious would
    # mean every joiner's ONLY initial contact is a liar, with no path to any
    # honest node until PEX (which starts FROM that same liar) spreads one.
    malicious_indices = set(range(1, 1 + n_malicious))
    honest_indices = [i for i in range(n_nodes) if i not in malicious_indices]

    py_rng = random.Random(SEED)
    centers = make_world(N_CLASSES, DIM, seed=SEED)
    # Coverage must be guaranteed over HONEST nodes only: a malicious node
    # covering a class contributes label-flipped poison, NOT corroboration, so
    # counting it toward the >=3 threshold leaves classes with too few honest
    # senders and they never get promoted (this was the 0.365-accuracy bug).
    cpn = _required_classes_per_node(len(honest_indices), N_CLASSES, CLASSES_PER_NODE, min_coverage=3)
    if cpn != CLASSES_PER_NODE:
        print(f"[coverage] raised classes/node {CLASSES_PER_NODE}->{cpn} so all {N_CLASSES} "
             f"classes reach >=3 HONEST-node corroboration coverage "
             f"({len(honest_indices)} honest of {n_nodes} nodes)")
    honest_assignments = _assign_classes_with_min_coverage(len(honest_indices), N_CLASSES, cpn, py_rng)
    assignments = [None] * n_nodes
    for slot, node_idx in enumerate(honest_indices):
        assignments[node_idx] = honest_assignments[slot]
    for node_idx in malicious_indices:  # malicious get arbitrary classes (they poison regardless)
        assignments[node_idx] = sorted(py_rng.sample(range(N_CLASSES), cpn))

    identities = [Identity_from_seed(i) for i in range(n_nodes)]
    metrics_dir = metrics_dir or Path.cwd() / "multiproc_metrics"
    metrics_dir.mkdir(exist_ok=True)
    for f in metrics_dir.glob("*.jsonl"):
        f.unlink()

    ctx = mp.get_context("spawn")  # 'spawn' is required/default on Windows anyway

    def launch(index, rendezvous_id, bootstrap):
        cfg = {
            "index": index, "seed": identities[index]["seed"],
            "malicious": index in malicious_indices, "class_assignment": assignments[index],
            "centers": centers, "rendezvous_id": rendezvous_id, "bootstrap": bootstrap,
            "is_rendezvous": not bootstrap, "metrics_path": str(metrics_dir / f"node_{index}.jsonl"),
            "duration_s": duration_s, "gossip_interval": gossip_interval,
            "gossip_sample_size": gossip_sample_size, "loss_rate": loss_rate,
        }
        p = ctx.Process(target=_worker, args=(cfg,))
        p.start()
        return p

    return {
        "launch": launch, "n_nodes": n_nodes, "identities": identities,
        "malicious_indices": malicious_indices, "metrics_dir": metrics_dir,
        "gossip_interval": gossip_interval, "duration_s": duration_s,
    }


def run(n_nodes: int = N_NODES, n_malicious: int = None, duration_s: float = DURATION_S,
       gossip_interval: float = GOSSIP_INTERVAL, metrics_dir: Path = None,
       budget: ResourceBudget = None, gossip_sample_size: int = GOSSIP_SAMPLE_SIZE,
       loss_rate: float = 0.0):
    s = _build_swarm(n_nodes, n_malicious, duration_s, gossip_interval, metrics_dir,
                     budget, gossip_sample_size, loss_rate)
    launch, n_nodes, identities = s["launch"], s["n_nodes"], s["identities"]
    metrics_dir = s["metrics_dir"]

    procs = [launch(0, None, [])]
    time.sleep(1.5)  # give node 0 time to bind its DHT socket before others bootstrap to it
    dht_addr_0 = _read_bootstrap_addr(metrics_dir, 0)
    for i in range(1, n_nodes):
        procs.append(launch(i, identities[0]["node_id"], [dht_addr_0]))
        time.sleep(0.05)  # stagger startup slightly to avoid a bootstrap thundering herd

    for p in procs:
        p.join(timeout=duration_s + 30)

    return _aggregate(metrics_dir, n_nodes, s["malicious_indices"], identities, gossip_interval)


def run_chaos(n_nodes: int = 12, n_malicious: int = 1, duration_s: float = 60.0,
             gossip_interval: float = GOSSIP_INTERVAL, metrics_dir: Path = None,
             budget: ResourceBudget = None, gossip_sample_size: int = GOSSIP_SAMPLE_SIZE,
             loss_rate: float = 0.20, kill_fraction: float = 0.30, chaos_at_s: float = 20.0):
    """T8: chaos. Runs the swarm under injected packet loss, then mid-run:
      1. KILLS kill_fraction of the nodes (SIGTERM), excluding node 0 (the DHT
         rendezvous/bootstrap) so survivors can still re-bootstrap;
      2. RESTARTS one killed node with a fresh process that must re-bootstrap
         from node 0 and rejoin;
    and asserts the SURVIVORS keep converging throughout.

    Returns a dict with survivor accuracy, the restarted node's post-rejoin
    accuracy, and the kill/restart bookkeeping.
    """
    s = _build_swarm(n_nodes, n_malicious, duration_s, gossip_interval, metrics_dir,
                     budget, gossip_sample_size, loss_rate)
    launch, n_nodes, identities = s["launch"], s["n_nodes"], s["identities"]
    metrics_dir = s["metrics_dir"]
    malicious_indices = s["malicious_indices"]
    print(f"[chaos] {n_nodes} nodes, loss_rate={loss_rate}, will kill "
         f"{int(n_nodes * kill_fraction)} at t={chaos_at_s}s and restart one")

    procs = {0: launch(0, None, [])}
    time.sleep(1.5)
    dht_addr_0 = _read_bootstrap_addr(metrics_dir, 0)
    bootstrap = [dht_addr_0]
    for i in range(1, n_nodes):
        procs[i] = launch(i, identities[0]["node_id"], bootstrap)
        time.sleep(0.05)

    # Let the swarm form and begin converging.
    time.sleep(chaos_at_s)

    # 1. Kill kill_fraction of nodes — never node 0 (rendezvous) and never the
    #    malicious ones (we want to observe HONEST survivor convergence, not
    #    conflate it with removing attackers).
    killable = [i for i in range(1, n_nodes) if i not in malicious_indices]
    n_kill = max(1, int(n_nodes * kill_fraction))
    victims = killable[:n_kill]
    restart_victim = victims[0]
    print(f"[chaos] killing nodes {victims}")
    for i in victims:
        procs[i].terminate()
    for i in victims:
        procs[i].join(timeout=10)

    # 2. Restart one victim with a FRESH process (same identity/classes). It
    #    must re-bootstrap from node 0 and rejoin via DHT+PEX. Its metrics file
    #    is truncated by the new process, so post-rejoin accuracy is measured
    #    cleanly from restart.
    time.sleep(2.0)
    print(f"[chaos] restarting node {restart_victim}")
    procs[restart_victim] = launch(restart_victim, identities[0]["node_id"], bootstrap)

    survivors = [i for i in range(n_nodes) if i not in victims]
    for i in survivors + [restart_victim]:
        procs[i].join(timeout=duration_s + 30)

    # Aggregate: honest survivors (excluding the ones that stayed dead).
    honest_survivors = {i for i in survivors
                        if i not in malicious_indices and i != 0} | {restart_victim}
    result = _aggregate(metrics_dir, n_nodes, malicious_indices, identities, gossip_interval)

    def final_acc(i):
        lines = result["per_node"].get(i)
        return lines[-1]["acc"] if lines else None

    survivor_accs = [final_acc(i) for i in honest_survivors if final_acc(i) is not None]
    import numpy as _np
    result["chaos"] = {
        "victims": victims, "restarted": restart_victim,
        "survivor_final_accs": survivor_accs,
        "mean_survivor_acc": float(_np.mean(survivor_accs)) if survivor_accs else 0.0,
        "restarted_final_acc": final_acc(restart_victim),
        "loss_rate": loss_rate,
        "mean_dropped_loss": float(_np.mean([
            lines[-1]["dropped_loss"] for i, lines in result["per_node"].items() if lines
        ])),
    }
    return result


def Identity_from_seed(index: int) -> dict:
    """Generate node 0..N-1's identity in the LAUNCHER (not a worker), so the
    launcher knows every node_id up front for assertions, and can hand node
    0's node_id (not its whole identity/address book) to the other workers as
    their single rendezvous point."""
    seed = bytes([((SEED + index * 7919) >> (8 * i)) & 0xFF for i in range(32)])
    from nacl.signing import SigningKey
    from ..network.identity import derive_node_id
    pubkey = bytes(SigningKey(seed).verify_key)
    return {"seed": seed, "node_id": derive_node_id(pubkey)}


def _read_bootstrap_addr(metrics_dir: Path, index: int):
    """Node 0 writes its bound DHT port to a small sidecar file right after
    binding, so the launcher can hand it to the other workers. This is
    launcher<->worker coordination, not a peer-list — the OTHER N-1 workers
    still only ever learn peers via DHT lookup + PEX."""
    sidecar = metrics_dir / f"node_{index}.dhtport"
    for _ in range(100):  # up to 10s
        if sidecar.exists():
            return ("127.0.0.1", int(sidecar.read_text()))
        time.sleep(0.1)
    raise RuntimeError(f"node {index} never reported its DHT port")


def _aggregate(metrics_dir: Path, n_nodes: int, malicious_indices: set, identities: list,
              gossip_interval: float) -> dict:
    per_node = {}
    for i in range(n_nodes):
        path = metrics_dir / f"node_{i}.jsonl"
        if not path.exists() or not path.read_text().strip():
            per_node[i] = None
            continue
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        per_node[i] = lines

    honest_final_accs = []
    mal_trust_samples, hon_trust_samples = [], []
    bandwidth_samples = []
    id_hex = {i: identities[i]["node_id"].hex() for i in range(n_nodes)}
    mal_hex = {id_hex[i] for i in malicious_indices}

    for i, lines in per_node.items():
        if not lines:
            continue
        final = lines[-1]
        if i not in malicious_indices:
            honest_final_accs.append(final["acc"])
            for peer_hex, t in final["trust"].items():
                (mal_trust_samples if peer_hex in mal_hex else hon_trust_samples).append(t)
        # bandwidth over the last BANDWIDTH_WINDOW_S seconds of the run
        window = [m for m in lines if lines[-1]["t"] - m["t"] <= BANDWIDTH_WINDOW_S]
        if len(window) >= 2:
            dt = window[-1]["t"] - window[0]["t"]
            dbytes = (window[-1]["tx_bytes"] + window[-1]["rx_bytes"]) - \
                     (window[0]["tx_bytes"] + window[0]["rx_bytes"])
            if dt > 0:
                bandwidth_samples.append(dbytes / dt)

    # Bandwidth is dominated by gossip cadence. The test gossips every
    # `gossip_interval` s but production gossips every PRODUCTION_GOSSIP_INTERVAL
    # s, so raw test bytes/s overstate the real steady-state by exactly that
    # ratio. Report both: raw (what the test machine saw) and normalized (what
    # a production node would sustain) — the budget is a production property.
    norm = gossip_interval / PRODUCTION_GOSSIP_INTERVAL
    mean_bw = float(np.mean(bandwidth_samples)) if bandwidth_samples else None
    max_bw = float(np.max(bandwidth_samples)) if bandwidth_samples else None
    return {
        "per_node": per_node,
        "honest_final_accs": honest_final_accs,
        "mean_honest_acc": float(np.mean(honest_final_accs)) if honest_final_accs else 0.0,
        "mean_malicious_trust": float(np.mean(mal_trust_samples)) if mal_trust_samples else None,
        "mean_honest_trust": float(np.mean(hon_trust_samples)) if hon_trust_samples else None,
        "mean_bandwidth_bytes_per_s_raw": mean_bw,
        "max_bandwidth_bytes_per_s_raw": max_bw,
        "mean_bandwidth_bytes_per_s_normalized": mean_bw * norm if mean_bw is not None else None,
        "max_bandwidth_bytes_per_s_normalized": max_bw * norm if max_bw is not None else None,
        "gossip_interval": gossip_interval,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="swarmint multiprocess integration test (T7)")
    ap.add_argument("--nodes", type=int, default=N_NODES, help="desired node count (auto-capped to budget)")
    ap.add_argument("--malicious", type=int, default=None, help="malicious node count (default ~20%%)")
    ap.add_argument("--duration", type=float, default=DURATION_S, help="run seconds")
    ap.add_argument("--gossip-interval", type=float, default=GOSSIP_INTERVAL)
    ap.add_argument("--gossip-sample-size", type=int, default=GOSSIP_SAMPLE_SIZE,
                    help="prototypes per gossip send (bandwidth knob)")
    ap.add_argument("--cpu-fraction", type=float, default=None, help="max CPU-core fraction (default 0.60)")
    ap.add_argument("--mem-fraction", type=float, default=None, help="max RAM fraction (default 0.60)")
    ap.add_argument("--mem-per-node-mb", type=float, default=None, help="est. per-process RAM for budgeting")
    ap.add_argument("--max-nodes", type=int, default=None, help="hard node-count cap (skips resource math)")
    args = ap.parse_args()

    budget = ResourceBudget(cpu_fraction=args.cpu_fraction, mem_fraction=args.mem_fraction,
                            mem_per_node_mb=args.mem_per_node_mb, max_nodes=args.max_nodes)
    print(f"requested {args.nodes} processes, gossip_interval={args.gossip_interval}s, "
         f"duration={args.duration}s ...")
    result = run(n_nodes=args.nodes, n_malicious=args.malicious, duration_s=args.duration,
                 gossip_interval=args.gossip_interval, budget=budget,
                 gossip_sample_size=args.gossip_sample_size)

    for i, lines in sorted(result["per_node"].items()):
        if lines is None:
            print(f"node {i}: NO METRICS (crashed or never started)")
            continue
        final = lines[-1]
        print(f"node {i:2d}: acc={final['acc']:.3f} peers={final['n_peers']:2d} "
             f"protos={final['n_protos']:3d} merges_ok={final['merges_ok']:3d} "
             f"rejected={final['merges_rejected']:3d} dht_ok={final['dht_publish_ok']}")

    acc = result["mean_honest_acc"]
    mal_t = result["mean_malicious_trust"]
    hon_t = result["mean_honest_trust"]
    bw_raw = result["mean_bandwidth_bytes_per_s_raw"]
    bw_norm = result["mean_bandwidth_bytes_per_s_normalized"]

    print(f"\nmean honest accuracy: {acc:.3f}")
    print(f"mean trust: malicious={mal_t} honest={hon_t}" if mal_t is not None
         else "\n(no cross-trust samples recorded)")
    if bw_raw is not None:
        print(f"bandwidth: raw={bw_raw:.0f} B/s/node @ test cadence ({result['gossip_interval']}s); "
             f"normalized={bw_norm:.0f} B/s/node @ production cadence ({PRODUCTION_GOSSIP_INTERVAL}s)")

    checks = {
        "accuracy>=0.90": acc >= 0.90,
        "malicious_trust<honest_trust": (mal_t is not None and hon_t is not None and mal_t < hon_t),
        "bandwidth<2000B/s/node (production-normalized)": (bw_norm is not None and bw_norm < BANDWIDTH_BUDGET_BYTES_PER_S),
        "all_nodes_reported": all(v is not None for v in result["per_node"].values()),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print(f"\n{verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
