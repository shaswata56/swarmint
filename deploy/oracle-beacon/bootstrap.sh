#!/usr/bin/env bash
# Runs ON an Oracle Cloud Always-Free VM to stand up a SECOND swarmint beacon that
# joins the federation hosted by the master (beacon.swarmint.org) and CROSS-LEARNS
# with it — two independent beacons on two different clouds behaving as one system.
#
#   bash bootstrap.sh <PUBLIC_IP> [NAME] [BACKBONE_N]
#
#     PUBLIC_IP    this VM's public IPv4 (Oracle shows it on the instance page).
#     NAME         label shown in the federation directory (default: oracle-1).
#     BACKBONE_N   if >0, also run a local honest backbone of N nodes so this
#                  beacon CONTRIBUTES a quorum (default 0 = beacon learns FROM the
#                  master's swarm via cross-beacon bridging, contributes its own slice).
#
# This is a federation MEMBER (not the hub): SWARM_MASTER_* point at the public
# master, and SWARM_TASK=digits + the same world seed => the SAME space fingerprint,
# so the master bridges to it and prototypes flow both ways over the signed gossip
# path. The VM must be publicly reachable on the UDP ports below — see README for
# the Oracle VCN ingress rule (console step you must do; the host firewall is opened
# here).
set -euo pipefail

PUBLIC_IP="${1:?usage: bootstrap.sh <PUBLIC_IP> [NAME] [BACKBONE_N]}"
NAME="${2:-oracle-1}"
BACKBONE_N="${3:-0}"

REPO="${REPO:-https://github.com/shaswata56/swarmint.git}"
DEST="${DEST:-/opt/swarmint}"
MASTER_HOST="${MASTER_HOST:-beacon.swarmint.org}"
MASTER_ID="${MASTER_ID:-8c30c97e7fd5460ce3b962db4cd75879eecd8abd}"  # seed-0 hub id
GOSSIP_PORT="${GOSSIP_PORT:-9001}"
DHT_PORT="${DHT_PORT:-9002}"
BB_BASE_PORT="${BB_BASE_PORT:-9003}"
BB_DHT_BASE_PORT="${BB_DHT_BASE_PORT:-9103}"

echo ">> installing packages ..."
if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y -q git python3 python3-pip
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq git python3 python3-venv python3-dev build-essential
fi

echo ">> opening the HOST firewall for UDP $GOSSIP_PORT-$((BB_BASE_PORT + 20)) ..."
# Oracle images ship restrictive defaults. Cover all three common stacks; each is
# best-effort (|| true) so the wrong one for this image doesn't abort the run.
HI=$((BB_BASE_PORT + 20))
if command -v firewall-cmd >/dev/null 2>&1; then
  sudo firewall-cmd --permanent --add-port=${GOSSIP_PORT}-${HI}/udp || true
  sudo firewall-cmd --reload || true
fi
if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow ${GOSSIP_PORT}:${HI}/udp || true
fi
# Oracle's Ubuntu image uses a legacy iptables INPUT chain with a REJECT catch-all;
# insert an ACCEPT for our UDP range ABOVE it and persist.
if command -v iptables >/dev/null 2>&1; then
  sudo iptables -I INPUT -p udp --dport ${GOSSIP_PORT}:${HI} -j ACCEPT || true
  (command -v netfilter-persistent >/dev/null 2>&1 && sudo netfilter-persistent save) || \
    (sudo mkdir -p /etc/iptables && sudo sh -c 'iptables-save > /etc/iptables/rules.v4') || true
fi

echo ">> cloning + installing swarmint ..."
if [ ! -d "$DEST/.git" ]; then
  sudo rm -rf "$DEST"
  sudo git clone --depth 1 "$REPO" "$DEST"
fi
sudo python3 -m venv "$DEST/.venv" 2>/dev/null || true
sudo "$DEST/.venv/bin/pip" install -q --upgrade pip
sudo "$DEST/.venv/bin/pip" install -q -e "$DEST[net,learn]"

echo ">> writing member-beacon env + systemd unit ..."
sudo tee /etc/swarmint-beacon.env >/dev/null <<EOF
SWARM_ROLE=rendezvous
SWARM_SEED=0
SWARM_ADVERTISE_HOST=0.0.0.0
SWARM_PUBLIC_HOST=${PUBLIC_IP}
SWARM_GOSSIP_PORT=${GOSSIP_PORT}
SWARM_DHT_PORT=${DHT_PORT}
SWARM_HTTP_PORT=8080
SWARM_HTTP_BIND=0.0.0.0
SWARM_ENABLE_RELAY=1
SWARM_DURATION_S=0
SWARM_TASK=digits
SWARM_FEDERATION=1
SWARM_BEACON_NAME=${NAME}
SWARM_MASTER_HOST=${MASTER_HOST}
SWARM_MASTER_ID=${MASTER_ID}
SWARM_MASTER_GOSSIP_PORT=${GOSSIP_PORT}
EOF
# NOTE: this beacon uses SEED=0 too, so its own node_id equals the master id. That
# is exactly how net_daemon detects "I am NOT the hub only if master_id != my id" —
# here they'd be equal, so it would wrongly think it's the hub. Use a DISTINCT seed.
sudo sed -i 's/^SWARM_SEED=0$/SWARM_SEED=7/' /etc/swarmint-beacon.env

sudo tee /etc/systemd/system/swarmint-beacon.service >/dev/null <<EOF
[Unit]
Description=swarmint beacon (federation member, cross-learns with the master)
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

if [ "$BACKBONE_N" -gt 0 ]; then
  echo ">> starting a local backbone of ${BACKBONE_N} nodes (own quorum) ..."
  sudo tee /etc/systemd/system/swarmint-backbone.service >/dev/null <<EOF
[Unit]
Description=swarmint honest backbone (local quorum for this beacon)
After=network-online.target swarmint-beacon.service
Requires=swarmint-beacon.service

[Service]
ExecStart=${DEST}/.venv/bin/python -m swarmint.cli backbone \\
  --beacon 127.0.0.1 --beacon-id 8c30c97e7fd5460ce3b962db4cd75879eecd8abd \\
  --n ${BACKBONE_N} --task digits --base-port ${BB_BASE_PORT} \\
  --dht-base-port ${BB_DHT_BASE_PORT} --public-host ${PUBLIC_IP} --bind 0.0.0.0 \\
  --gossip-port ${GOSSIP_PORT} --dht-port ${DHT_PORT}
WorkingDirectory=${DEST}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now swarmint-backbone.service
fi

sleep 6
echo ">> beacon status:"
sudo journalctl -u swarmint-beacon --no-pager -n 40 | grep -E "rendezvous_up|federation|status_http_up" | tail -5 || true
echo
echo ">> this beacon '${NAME}' should now appear (reachable) at:"
echo "     https://beacon.swarmint.org/federation.json"
echo "   and its own directory at:  http://${PUBLIC_IP}:8080/federation.json"
