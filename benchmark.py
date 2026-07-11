"""Reproduce swarmint's headline claims in one command.

    python benchmark.py

Runs the deterministic in-process scenarios that back the README's numbers and
prints a single results table with a machine-checkable PASS/FAIL per claim
(exit code 0 iff all pass). This is both the "reproduce our claims" entry point
for anyone evaluating the project and a regression guard for the headline
behaviour — if a change quietly degrades the swarm, this goes red.

What it checks:
  1. Federation      — a 100-node swarm reaches near-centralized accuracy while
                       each node sees only 2 of 10 classes; solo nodes can't.
  2. Byzantine       — honest nodes drive malicious peers' trust below honest.
  3. Inference (MoE) — a node answers about classes it never learned by a
                       trust-weighted vote across peers, beating local + naive.
  4. Cross-NAT relay — two peers with no direct path still exchange gossip
                       through a relay (T4), the basis of real-world P2P.

Scenarios 1-3 are fully deterministic (fixed seeds). Scenario 4 uses real UDP on
loopback; its outcome (delivery) is deterministic even though keys are random.
"""

import asyncio
import io
import random
import sys
from contextlib import redirect_stdout


def _quiet(fn, *args, **kwargs):
    """Run a noisy sim, swallow its stdout, return its result dict."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        return fn(*args, **kwargs)


async def _relay_scenario():
    """Two nodes that (are told they) can't reach each other directly still
    exchange a prototype through the rendezvous relay. Returns (delivered,
    forwarded) — mirrors tests/test_relay.py, condensed."""
    import numpy as np

    from swarmint.core.prototype_model import Prototype
    from swarmint.network.bus import Message
    from swarmint.network.identity import Identity
    from swarmint.node.net_node import NetNode

    id_r, id_a, id_b = Identity.generate(), Identity.generate(), Identity.generate()
    mk = lambda i, s: NetNode(identity=i, topic=1, rng=random.Random(s),
                              gossip_interval=10.0, enable_relay=True)
    r, a, b = mk(id_r, 0), mk(id_a, 1), mk(id_b, 2)
    await r.start("127.0.0.1", 0, 0, [])
    dht = r.discovery.server.transport.get_extra_info("sockname")[1]
    await a.start("127.0.0.1", 0, 0, [("127.0.0.1", dht)], rendezvous_id=id_r.node_id)
    await b.start("127.0.0.1", 0, 0, [("127.0.0.1", dht)], rendezvous_id=id_r.node_id)
    await a.nat.discover_reflexive(id_r.node_id, timeout=1.5)
    await b.nat.discover_reflexive(id_r.node_id, timeout=1.5)

    a.mark_relay_only(id_b.node_id)
    got = []
    orig = b.node.receive
    b.node.receive = lambda m: (got.append(m), orig(m))
    a.bus.send(Message(id_a.node_id, id_b.node_id, "protos",
                       [Prototype(np.ones(4, dtype=np.float32), 1, 1.0)]))
    for _ in range(100):
        if got:
            break
        await asyncio.sleep(0.02)
    forwarded = r.bus.relayed_fwd
    for n in (a, b, r):
        await n.stop()
    return (len(got) >= 1, forwarded)


def main() -> int:
    from swarmint.sim import run_inference, run_sim

    # A reduced config so the whole benchmark runs in ~40s. The qualitative
    # claims (swarm >> solo, Byzantine suppression) hold at any scale; the exact
    # 0.959 headline is the canonical full run: python -m swarmint.sim.run_sim
    fed = _quiet(run_sim.main, n_nodes=40, rounds=50)
    inf = _quiet(run_inference.main)
    delivered, forwarded = asyncio.run(_relay_scenario())

    # (claim, detail, value_str, passed)
    checks = [
        ("Federation (swarm >> solo)",
         f"{fed['n_nodes']} nodes, 2/10 classes each",
         f"swarm {fed['swarm_acc']:.3f} vs solo {fed['solo_acc']:.3f}",
         fed["swarm_acc"] > fed["solo_acc"] + 0.30 and fed["swarm_acc"] >= 0.60),
        ("Byzantine suppression",
         f"{fed['n_malicious']} label-flip attackers",
         f"trust: malicious {fed['trust_malicious']:.3f} < honest {fed['trust_honest']:.3f}",
         fed["trust_malicious"] < fed["trust_honest"]),
        ("Distributed inference (MoE)",
         f"{inf['n_experts']} experts, {inf['n_malicious']} liars",
         f"trust-vote {inf['acc_trust']:.3f} vs local {inf['acc_local']:.3f} / naive {inf['acc_naive']:.3f}",
         inf["acc_trust"] > inf["acc_local"] + 0.30 and inf["acc_trust"] > inf["acc_naive"]),
        ("Byzantine suppression (inference)",
         "confidently-wrong experts downweighted",
         f"trust: malicious {inf['trust_malicious']:.3f} < honest {inf['trust_honest']:.3f}",
         inf["trust_malicious"] < inf["trust_honest"]),
        ("Cross-NAT relay (T4)",
         "no direct path A<->B",
         f"delivered={delivered}, relay forwards={forwarded}",
         delivered and forwarded >= 1),
    ]

    name_w = max(len(c[0]) for c in checks)
    detail_w = max(len(c[1]) for c in checks)
    val_w = max(len(c[2]) for c in checks)
    print("\n" + "=" * 4 + " swarmint benchmark " + "=" * 4)
    print(f"{'claim'.ljust(name_w)}  {'scenario'.ljust(detail_w)}  {'result'.ljust(val_w)}  ok")
    print("-" * (name_w + detail_w + val_w + 10))
    for name, detail, val, ok in checks:
        print(f"{name.ljust(name_w)}  {detail.ljust(detail_w)}  {val.ljust(val_w)}  {'PASS' if ok else 'FAIL'}")

    passed = sum(1 for c in checks if c[3])
    total = len(checks)
    print("-" * (name_w + detail_w + val_w + 10))
    ok_all = passed == total
    print(f"{passed}/{total} claims reproduced -- {'ALL PASS' if ok_all else 'SOME FAILED'}")
    print("(reduced config for speed; exact 0.959 headline: python -m swarmint.sim.run_sim)\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
