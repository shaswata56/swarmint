#!/usr/bin/env bash
# Provision a free-tier GCP e2-micro as the swarmint BEACON (hole-punch rendezvous
# + relay). Run AFTER `gcloud auth login` and selecting a project.
#
#   ./provision.sh              # create VM + firewall, print the public IP
#
# Free-tier note: e2-micro is always-free ONLY in us-west1 / us-central1 / us-east1.
# The beacon speaks UDP on 9001 (gossip) and 9002 (DHT); we open both.
set -euo pipefail

NAME="${NAME:-swarmint-beacon}"
ZONE="${ZONE:-us-central1-a}"            # free-tier region
MACHINE="${MACHINE:-e2-micro}"           # always-free tier
IMAGE_FAMILY="${IMAGE_FAMILY:-debian-12}"
GOSSIP_PORT="${GOSSIP_PORT:-9001}"
DHT_PORT="${DHT_PORT:-9002}"

echo ">> creating e2-micro '$NAME' in $ZONE ..."
gcloud compute instances create "$NAME" \
  --zone="$ZONE" --machine-type="$MACHINE" \
  --image-family="$IMAGE_FAMILY" --image-project=debian-cloud \
  --boot-disk-type=pd-standard --boot-disk-size=30GB \
  --tags=swarmint-beacon
# pd-standard/30GB stays inside the always-free allowance; the default balanced
# disk is NOT free-tier covered.

echo ">> opening inbound UDP $GOSSIP_PORT,$DHT_PORT ..."
gcloud compute firewall-rules create swarmint-beacon-udp \
  --allow="udp:${GOSSIP_PORT},udp:${DHT_PORT}" \
  --target-tags=swarmint-beacon \
  --description="swarmint beacon gossip+DHT (hole-punch rendezvous/relay)" \
  2>/dev/null || echo "   (udp rule already exists, skipping)"

echo ">> opening inbound TCP 80,443 (status page + Let's Encrypt) ..."
gcloud compute firewall-rules create swarmint-beacon-web \
  --allow="tcp:80,tcp:443" \
  --target-tags=swarmint-beacon \
  --description="swarmint beacon status page (Caddy TLS + ACME)" \
  2>/dev/null || echo "   (web rule already exists, skipping)"

IP=$(gcloud compute instances describe "$NAME" --zone="$ZONE" \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo
echo "============================================================"
echo " BEACON public IP: $IP"
echo " 1) DNS: add an A record  beacon.swarmint.org -> $IP"
echo " 2) bootstrap the daemon:"
echo "      gcloud compute scp bootstrap.sh $NAME:~ --zone=$ZONE"
echo "      gcloud compute ssh $NAME --zone=$ZONE --command='bash ~/bootstrap.sh $IP'"
echo "============================================================"
