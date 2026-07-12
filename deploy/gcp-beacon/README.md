# beacon.swarmint.org — real-hardware NAT hole-punch test

A public UDP **beacon** (hole-punch rendezvous + relay) on a free-tier GCP
e2-micro, so two real NATed peers (a home PC + a phone on cellular) can attempt a
hole-punch across genuine router NATs — the one thing the netns/Docker harness
can't prove (emulated netfilter NAT breaks the data-plane punch even for raw UDP).

## Why a real VM (not Vercel / a proxy / ngrok)
The beacon must hold a long-lived **UDP** socket on a **real public IP** and see
each peer's *true* source `IP:port`. Serverless (HTTP-only, no UDP) and anything
that proxies (Fly edge, ngrok, SOCKS/HTTP) rewrite the source address, destroying
the one measurement the rendezvous exists to make. GCP `e2-micro` free tier gives
a dedicated public IPv4 with no proxy in front — correct and $0.

## Steps
**You (once):** create a GCP account, `gcloud auth login`, select a project.

**Provision (scripted):**
```bash
cd deploy/gcp-beacon
bash provision.sh                     # creates VM + UDP firewall, prints PUBLIC_IP
```
**You:** add DNS `A` record `beacon.swarmint.org -> PUBLIC_IP`.

**Bootstrap the daemon (scripted):**
```bash
gcloud compute scp bootstrap.sh swarmint-beacon:~ --zone=us-central1-a
gcloud compute ssh swarmint-beacon --zone=us-central1-a --command='bash ~/bootstrap.sh <PUBLIC_IP>'
```
Copy the `rendezvous_up ... node_id=<HEX>` value it prints — peers need it.

## Run the two peers
On the **PC** (NAT A) and the **phone/2nd network** (NAT B), each:
```bash
SWARM_ROLE=node SWARM_SEED=1 \
SWARM_RENDEZVOUS_HOST=beacon.swarmint.org SWARM_RENDEZVOUS_DHT_PORT=9002 \
SWARM_RENDEZVOUS_ID=<HEX from beacon> \
SWARM_PUNCH_PEERS=<other peer's node_id> \
SWARM_ENABLE_RELAY=1 \
python -m swarmint.sim.net_daemon
```
(Use `SWARM_SEED=1` and `2` for the two peers; each peer's `node_id` is printed in
its own `identity ... node_id=` line — cross-wire them into `SWARM_PUNCH_PEERS`.)

**Read the result:**
- `reflexive ... nat_detected=True` — each peer's NAT mapping was observed (signaling works).
- `punch ... ok=True in_reachable=True` — **direct hole-punch succeeded** (the green punch we're after).
- `punch ... ok=False` then `relay_fallback` — punch failed on this router pair; relay carried the swarm anyway (still a valid, honest outcome — records the router's NAT behavior).

## Live status page
The beacon serves a read-only HTML status page (port 80) — visit
`http://beacon.swarmint.org/` (or the raw IP) to see active/inactive peers, the
public post-NAT endpoint the beacon observes for each, reachability, and freshness.
It auto-refreshes every 5s and stores nothing beyond the live session. Requires the
DNS `A` record above (or use the raw IP); the HTTP firewall rule is
`swarmint-beacon-http` (tcp:80).

Note: WebFetch/https won't work — the page is plain HTTP on :80 by design (a STUN
rendezvous can't sit behind an HTTPS-terminating proxy without losing the true UDP
source address). Browse it directly or `curl http://<ip>/`.

## Cost / lifecycle
The service runs 24/7 (`SWARM_DURATION_S=0`) on an always-free `e2-micro` +
`pd-standard` disk in a free-tier region — $0 within the allowance. `sudo
systemctl stop swarmint-beacon` pauses it; teardown below removes it entirely.

## Teardown
```bash
gcloud compute instances delete swarmint-beacon --zone=us-central1-a
gcloud compute firewall-rules delete swarmint-beacon-udp
```
