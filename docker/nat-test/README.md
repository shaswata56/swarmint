# Real-NAT test harness (Docker)

Puts swarmint nodes behind **real Linux netfilter NAT** — not a simulation — to
exercise the parts of the P2P stack that loopback and in-process tests
structurally cannot reach (see [`swarmint/network/nat.py`](../../swarmint/network/nat.py)'s
TESTABILITY NOTE).

## Topology

Three separate Docker networks = three separate address realms. Two nodes sit
on **different** LANs, each behind its own NAT gateway; the rendezvous is the
only directly-reachable host.

```
 lanA 10.10.1.0/24         pubnet 172.30.0.0/24         lanB 10.10.2.0/24
 +--------+   +------+     +--------------+     +------+   +--------+
 | nodeA  |---| gwA  |=====|  rendezvous  |=====| gwB  |---| nodeB  |
 | .10    |   |.2/.2 |     |    .10       |     |.3/.2 |   | .10    |
 +--------+   +------+     +--------------+     +------+   +--------+
              MASQUERADE                        MASQUERADE
```

- Each gateway runs `iptables -t nat MASQUERADE` for its LAN (real conntrack NAT).
- Each node's default route is forced through its gateway, so **the only path
  off-LAN is through the NAT**. Neither node can address the other directly.
- node_ids are deterministic from seeds (rendezvous=0, nodeA=1, nodeB=2) so the
  orchestrator can wire up `PUNCH_PEERS` without a discovery round-trip.

## Run

```bash
# from repo root; needs Docker with Linux containers
python docker/nat-test/gen_compose.py > docker/nat-test/docker-compose.yml
cd docker/nat-test
docker compose build
docker compose up -d
docker compose wait nodeA nodeB
docker compose logs nodeA nodeB | grep EVENT
docker compose down -v
```

`NAT_SYMMETRIC=1 python gen_compose.py > docker-compose.yml` makes the gateways
allocate ports per-destination (`MASQUERADE --random-fully`) to model a
symmetric NAT.

## Results (2026-07-10, cone / default MASQUERADE)

| Claim | Result | Evidence |
|-------|--------|----------|
| Node observes a **real NAT mapping** | ✅ confirmed | `reflexive=172.30.0.2:9001` vs `local=10.10.1.10:9001`, `nat_detected=True` on both nodes |
| Rendezvous **signaling across NAT** (relays each peer's public endpoint) | ✅ confirmed | nodeA learns `peer_addr=('172.30.0.3', 9001)` for nodeB via the rendezvous, and vice-versa |
| NAT is **endpoint-independent** (cone-type) | ✅ confirmed | same source port maps to the same public `172.30.0.2:8001` toward two different destinations |
| **Hole-punch opens a direct path** | ❌ did **not** complete | `EVENT punch ok=False in_reachable=False` (3 retries) on both nodes |

### The punch failure is environmental, not a swarmint bug

A **raw-UDP control** — two plain sockets doing simultaneous-open to each
other's reflexive address, no swarmint code involved — **also fails** in this
topology, both directions. So the blocker is the stacked dual-`MASQUERADE`
Docker environment (host `bridge-nf-call-iptables` + Docker's own pubnet
MASQUERADE/isolation interfering with the two gateways' conntrack), **not** the
hole-punch protocol in `nat.py`. The mapping is cone-type and the signaling is
healthy; a punch would open on a NAT with clean endpoint-independent
*filtering* (real home routers, or a veth/netns router with explicit nftables).

Follow-up runs showed the punch is also **flaky/asymmetric** here: sometimes one
direction opens (nodeA→nodeB) while the reverse doesn't. That reinforces the
same conclusion — a direct punch cannot be depended on.

## Relay fallback (T4) — verified working over real NAT

Set `SWARM_ENABLE_RELAY=1` (the compose generator's default now). When a punch
fails, the node marks the peer relay-only and routes to it through the
rendezvous, which forwards the (still origin-signed) datagram. Observed in a
real-NAT run where the punch did not open both directions:

| Signal | Result |
|--------|--------|
| rendezvous forwards relayed traffic (`relay_fwd`) | ✅ climbs 10 → 23 → 108 |
| both nodes receive relayed gossip (`relay_rx`) | ✅ 59 / 80 |
| both models keep growing (`n_protos`) | ✅ 12→92 and 39→125 |

So two nodes that cannot punch to each other still exchange knowledge across
their NATs, via the rendezvous. (Accuracy stays ~0.2 in this 3-node setup
because the 3-sender corroboration rule can't be satisfied with so few nodes —
that's expected and unrelated to relay; the counters and proto growth are the
proof. `test_relay.py` asserts the same path deterministically on loopback.)

### Why this matters for the roadmap

The run empirically confirms the premise of the P2P-hardening phase:
**hole-punching cannot be the only path to a peer** — and relay fallback (**T4**,
now implemented) makes the swarm converge even when the punch can't. Remaining:
(a) reproduce a *successful, reliable* punch on a filtering-clean NAT (netns
router) to validate the optimization path, (b) keepalive (**T3**) to hold direct
mappings open, (c) NAT-type detection (**T2**) to choose punch-vs-relay.

All findings are printed as `EVENT ...` lines to stdout (see
[`swarmint/sim/net_daemon.py`](../../swarmint/sim/net_daemon.py)).
