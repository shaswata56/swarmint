#!/usr/bin/env bash
# Runs ON the GCP beacon VM (after bootstrap.sh). Adds an always-on honest
# BACKBONE: N swarmint nodes that cover all classes with >=3-sender corroboration
# so that ANY external `swarmint join` immediately has the quorum it needs to
# learn (an empty rendezvous can never supply that — a lone joiner would sit at
# its solo ceiling forever). The backbone attaches to the rendezvous already
# running on this VM (127.0.0.1:9001/9002) and ADVERTISES the public IP so
# external joiners, which discover backbone nodes via PEX, can reach them.
#
#   bash backbone.sh <PUBLIC_IP> [N]
#
# The gossip ports base_port .. base_port+N-1 (default 9003..9010) MUST be open
# in the firewall for external joiners to reach the backbone directly — see
# provision.sh (swarmint-beacon-backbone rule). DHT ports stay VM-local.
set -euo pipefail

PUBLIC_IP="${1:?usage: backbone.sh <PUBLIC_IP> [N] [TASK]}"
N="${2:-8}"
TASK="${3:-digits}"                 # what the backbone (and thus the swarm) learns
DEST="${DEST:-/opt/swarmint}"
BASE_PORT="${BASE_PORT:-9003}"
DHT_BASE_PORT="${DHT_BASE_PORT:-9103}"
BEACON_ID="${BEACON_ID:-8c30c97e7fd5460ce3b962db4cd75879eecd8abd}"  # seed-0 rendezvous id

# The digits task needs scikit-learn (the [net] extra doesn't include it).
if [ "$TASK" = "digits" ]; then
  echo ">> ensuring scikit-learn is installed (digits task) ..."
  sudo "$DEST/.venv/bin/pip" install -q scikit-learn
fi

echo ">> writing swarmint-backbone systemd unit (N=$N) ..."
sudo tee /etc/systemd/system/swarmint-backbone.service >/dev/null <<EOF
[Unit]
Description=swarmint honest backbone (learning quorum for joiners)
After=network-online.target swarmint-beacon.service
Wants=network-online.target
Requires=swarmint-beacon.service

[Service]
ExecStart=${DEST}/.venv/bin/python -m swarmint.cli backbone \\
  --beacon 127.0.0.1 --beacon-id ${BEACON_ID} --n ${N} --task ${TASK} \\
  --base-port ${BASE_PORT} --dht-base-port ${DHT_BASE_PORT} \\
  --public-host ${PUBLIC_IP} --bind 0.0.0.0
WorkingDirectory=${DEST}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now swarmint-backbone.service
sleep 4
echo ">> backbone status:"
sudo systemctl --no-pager --lines=0 status swarmint-backbone.service || true
sudo journalctl -u swarmint-backbone.service --no-pager --lines=20 | grep -E "backbone_plan|backbone_node_up|backbone_metrics" | tail -12 || \
  sudo journalctl -u swarmint-backbone.service --no-pager --lines=20
echo
echo ">> backbone of ${N} nodes attached to the local rendezvous, advertising ${PUBLIC_IP}:${BASE_PORT}..$((BASE_PORT+N-1))"
echo "   any peer can now:  swarmint join   (accuracy will climb toward the ceiling)"
