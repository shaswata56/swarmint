#!/usr/bin/env bash
# Runs ON the GCP VM. Installs swarmint and launches net_daemon in the `rendezvous`
# role as a systemd service that survives reboots. Arg 1 = the VM's PUBLIC IP.
#
#   bash bootstrap.sh <PUBLIC_IP>
#
# CLOUD-NAT NOTE: GCP gives the VM a private NIC address and 1:1-NATs the public
# IP, so the daemon BINDS 0.0.0.0 (SWARM_ADVERTISE_HOST=0.0.0.0) and peers reach
# it via the public IP passed as SWARM_RENDEZVOUS_HOST on their side. The public
# IP here is only recorded for the operator; the daemon must not try to bind it.
set -euo pipefail

PUBLIC_IP="${1:?usage: bootstrap.sh <PUBLIC_IP>}"
GOSSIP_PORT="${GOSSIP_PORT:-9001}"
DHT_PORT="${DHT_PORT:-9002}"
REPO="${REPO:-https://github.com/shaswata56/swarmint.git}"
DEST="${DEST:-/opt/swarmint}"

echo ">> installing deps ..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3 python3-venv python3-dev build-essential

echo ">> cloning $REPO -> $DEST ..."
sudo rm -rf "$DEST"
sudo git clone --depth 1 "$REPO" "$DEST"
sudo python3 -m venv "$DEST/.venv"
sudo "$DEST/.venv/bin/pip" install -q --upgrade pip
sudo "$DEST/.venv/bin/pip" install -q -e "$DEST[net,learn]"
# [learn] adds scikit-learn — needed for the real-digits task (SWARM_TASK=digits).

echo ">> writing env + systemd unit ..."
sudo tee /etc/swarmint-beacon.env >/dev/null <<EOF
SWARM_ROLE=rendezvous
SWARM_SEED=0
SWARM_ADVERTISE_HOST=0.0.0.0
SWARM_PUBLIC_HOST=${PUBLIC_IP}
SWARM_GOSSIP_PORT=${GOSSIP_PORT}
SWARM_DHT_PORT=${DHT_PORT}
SWARM_HTTP_PORT=8080
SWARM_HTTP_BIND=127.0.0.1
SWARM_ENABLE_RELAY=1
SWARM_DURATION_S=0
SWARM_TASK=${SWARM_TASK:-digits}
EOF
# SWARM_PUBLIC_HOST: bind 0.0.0.0 but ADVERTISE the public IP (1:1 cloud NAT).
# Status page binds 127.0.0.1:8080 — Caddy (below) fronts it with HTTPS on 443.

sudo tee /etc/systemd/system/swarmint-beacon.service >/dev/null <<EOF
[Unit]
Description=swarmint beacon (hole-punch rendezvous + relay)
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/etc/swarmint-beacon.env
ExecStart=${DEST}/.venv/bin/python -m swarmint.sim.net_daemon
WorkingDirectory=${DEST}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now swarmint-beacon.service
sleep 3
echo ">> service status + rendezvous node_id (needed by peers):"
sudo systemctl --no-pager --lines=0 status swarmint-beacon.service || true
sudo journalctl -u swarmint-beacon.service --no-pager | grep -m1 rendezvous_up || \
  sudo journalctl -u swarmint-beacon.service --no-pager | tail -5
echo
echo ">> beacon live on ${PUBLIC_IP}:${GOSSIP_PORT}(gossip)/${DHT_PORT}(dht)"
echo "   peers set: SWARM_RENDEZVOUS_HOST=beacon.swarmint.org SWARM_RENDEZVOUS_ID=<node_id above>"

# ---- HTTPS status page via Caddy (auto Let's Encrypt) ----
# DOMAIN must have an A record -> this VM's public IP before running (ACME needs it).
DOMAIN="${DOMAIN:-beacon.swarmint.org}"
echo ">> installing Caddy for HTTPS on ${DOMAIN} ..."
if ! command -v caddy >/dev/null 2>&1; then
  sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl gnupg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq caddy
fi
sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
${DOMAIN} {
	reverse_proxy 127.0.0.1:8080
	header -Server
}
CADDY
sudo systemctl restart caddy
echo "   status page: https://${DOMAIN}/ (HTTP auto-redirects to HTTPS)"
