#!/usr/bin/env bash
# Clean-NAT hole-punch lab using raw Linux network namespaces (NOT Docker
# bridges). The Docker harness proved real NAT mapping + signaling but the
# punch was blocked/flaky because of the host's bridge-nf-call-iptables +
# Docker's own stacked MASQUERADE. Here we build the whole topology by hand
# inside ONE privileged container, so each side is an ordinary Linux
# port-restricted-cone NAT with nothing else in the path:
#
#   nsA (10.1.0.2) --lan-- natA[10.1.0.1 | 172.31.0.2] --.
#                                                         br-inet 172.31.0.1  (rendezvous, root ns)
#   nsB (10.2.0.2) --lan-- natB[10.2.0.1 | 172.31.0.3] --'
#
# If the punch opens here, nat.py's protocol is validated end-to-end; if it
# still fails, the bug is deeper than the environment.
#
# Run:  docker run --rm --privileged --entrypoint bash swarmint-nat:test -s < punch-lab.sh
set -euo pipefail

PY="python -m swarmint.sim.net_daemon"
DUR="${SWARM_DURATION_S:-30}"

nid() { python -c "from nacl.signing import SigningKey; from swarmint.network.identity import Identity; print(Identity(SigningKey(($1).to_bytes(4,'big').rjust(32,b'\0'))).node_id.hex())"; }
R=$(nid 0); A=$(nid 1); B=$(nid 2)
echo "LAB node_ids: R=$R A=$A B=$B"

# --- backbone ("the internet") in the root namespace ---
ip link add br-inet type bridge
ip addr add 172.31.0.1/24 dev br-inet
ip link set br-inet up

setup_side() {  # $1=idx  $2=wan_ip  $3=lan_subnet  $4=lan_gw  $5=node_ip
  local nat="nat$1" node="node$1"
  ip netns add "$nat"; ip netns add "$node"
  # WAN: root(br-inet) <-> nat ns
  ip link add "wan$1" type veth peer name "wan${1}p"
  ip link set "wan$1" master br-inet up
  ip link set "wan${1}p" netns "$nat"
  ip netns exec "$nat" ip addr add "$2/24" dev "wan${1}p"
  ip netns exec "$nat" ip link set "wan${1}p" up
  ip netns exec "$nat" ip link set lo up
  ip netns exec "$nat" ip route add default via 172.31.0.1
  # LAN: nat ns <-> node ns
  ip link add "lan$1" type veth peer name "lan${1}p"
  ip link set "lan$1" netns "$nat"
  ip link set "lan${1}p" netns "$node"
  ip netns exec "$nat" ip addr add "$4/24" dev "lan$1"
  ip netns exec "$nat" ip link set "lan$1" up
  ip netns exec "$node" ip addr add "$5/24" dev "lan${1}p"
  ip netns exec "$node" ip link set "lan${1}p" up
  ip netns exec "$node" ip link set lo up
  ip netns exec "$node" ip route add default via "$4"
  # NAT: forward + masquerade (echo into per-netns /proc/sys; the slim image
  # has no `sysctl` binary)
  ip netns exec "$nat" sh -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'
  ip netns exec "$nat" iptables -t nat -A POSTROUTING -s "$3" ! -d "$3" -j MASQUERADE
}

setup_side 1 172.31.0.2 10.1.0.0/24 10.1.0.1 10.1.0.2   # side A
setup_side 2 172.31.0.3 10.2.0.0/24 10.2.0.1 10.2.0.2   # side B

echo "=== connectivity check: nsA -> rendezvous through natA ==="
ip netns exec node1 ping -c1 -W2 172.31.0.1 | grep -E "packets" || true

# --- launch: rendezvous in root ns, nodes in their netns ---
SWARM_ROLE=rendezvous SWARM_SEED=0 SWARM_ADVERTISE_HOST=172.31.0.1 \
  SWARM_GOSSIP_PORT=9001 SWARM_DHT_PORT=9002 SWARM_DURATION_S="$DUR" $PY >/tmp/r.log 2>&1 &
sleep 3
ip netns exec node1 env SWARM_ROLE=node SWARM_SEED=1 SWARM_ADVERTISE_HOST=10.1.0.2 \
  SWARM_GOSSIP_PORT=9001 SWARM_DHT_PORT=9002 SWARM_RENDEZVOUS_HOST=172.31.0.1 \
  SWARM_RENDEZVOUS_DHT_PORT=9002 SWARM_RENDEZVOUS_ID="$R" SWARM_PUNCH_PEERS="$B" \
  SWARM_DURATION_S="$DUR" $PY >/tmp/a.log 2>&1 &
ip netns exec node2 env SWARM_ROLE=node SWARM_SEED=2 SWARM_ADVERTISE_HOST=10.2.0.2 \
  SWARM_GOSSIP_PORT=9001 SWARM_DHT_PORT=9002 SWARM_RENDEZVOUS_HOST=172.31.0.1 \
  SWARM_RENDEZVOUS_DHT_PORT=9002 SWARM_RENDEZVOUS_ID="$R" SWARM_PUNCH_PEERS="$A" \
  SWARM_DURATION_S="$DUR" $PY >/tmp/b.log 2>&1 &
wait

echo "################ NODE A ################"; grep -E "EVENT (reflexive|punch)" /tmp/a.log || true
echo "################ NODE B ################"; grep -E "EVENT (reflexive|punch)" /tmp/b.log || true
echo "=================== VERDICT ==================="
PUNCH_OK=$(grep -hc "EVENT punch .*ok=True" /tmp/a.log /tmp/b.log | paste -sd+ | bc 2>/dev/null || echo 0)
NAT_OK=$(grep -hc "nat_detected=True" /tmp/a.log /tmp/b.log | paste -sd+ | bc 2>/dev/null || echo 0)
echo "-> nodes behind a real NAT (reflexive != local): expected 2"
echo "-> successful hole-punches: $(grep -hE 'EVENT punch .*ok=True' /tmp/a.log /tmp/b.log | wc -l)/2"
