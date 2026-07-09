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

### Why this matters for the roadmap

Even though the failure is environmental, the run empirically confirms the
premise of the P2P-hardening phase: **hole-punching cannot be the only path to a
peer.** A relay fallback (phase task **T4**, TURN-lite through the rendezvous)
and keepalive (**T3**) are mandatory, not optional. The next milestone is to
(a) reproduce a *successful* punch on a filtering-clean NAT (netns router), and
(b) implement relay fallback so the swarm converges even when the punch can't.

All findings are printed as `EVENT ...` lines to stdout (see
[`swarmint/sim/net_daemon.py`](../../swarmint/sim/net_daemon.py)).
