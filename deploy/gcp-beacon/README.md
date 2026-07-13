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

`DOMAIN` is **opt-in** — only set it if this VM owns a real domain (an A record
already points here) and you want free Caddy-managed HTTPS on it:
```bash
DOMAIN=beacon.swarmint.org gcloud compute ssh swarmint-beacon --zone=us-central1-a \
  --command='DOMAIN=beacon.swarmint.org bash ~/bootstrap.sh <PUBLIC_IP>'
```
Leave `DOMAIN` unset for a **peer beacon** (federating with an existing genesis,
not owning a domain) — the status page then serves plain HTTP directly on `:80`,
no TLS, no Caddy. ⚠️ Never hardcode a domain default here again: an earlier
version of this script defaulted `DOMAIN` to the genesis's own domain, so a peer
bootstrapped from it wrongly installed Caddy fronting a domain it didn't own —
HTTPS on that peer's bare IP then failed with a TLS/SNI mismatch (`ERR_SSL_PROTOCOL_ERROR`).

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
The beacon serves a read-only status page — a cached HTML shell that updates
live via `/status.json` (no reload, no flicker; see the front page's client-side
JS). Genesis (`DOMAIN` set): `https://beacon.swarmint.org/` via Caddy-managed TLS,
auto-redirected from `:80`. Peer (`DOMAIN` unset): plain `http://<PUBLIC_IP>/` on
`:80`, no TLS — a bare IP can never get a browser-trusted cert, so don't try HTTPS
directly against a peer's IP. `swarmint beacons` reads any beacon's
`/status.json`/`/federation.json` and clicking a beacon's id in the Federated
Beacons table opens that beacon's own status page (its advertised `status_url`).

## Cost / lifecycle
The service runs 24/7 (`SWARM_DURATION_S=0`) on an always-free `e2-micro` +
`pd-standard` disk in a free-tier region — $0 within the allowance. `sudo
systemctl stop swarmint-beacon` pauses it; teardown below removes it entirely.

## Teardown
```bash
gcloud compute instances delete swarmint-beacon --zone=us-central1-a
gcloud compute firewall-rules delete swarmint-beacon-udp
```
