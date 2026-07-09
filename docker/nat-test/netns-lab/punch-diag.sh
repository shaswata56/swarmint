#!/usr/bin/env bash
# Diagnostic: same netns topology as punch-lab.sh, but instead of the daemon it
# runs raw-UDP probes to localize WHY the punch fails:
#   (1) mapping type  — same src port to two dests: same public port => cone.
#   (2) simultaneous open — two sockets punch to each other's reflexive addr.
set -uo pipefail

ip link add br-inet type bridge; ip addr add 172.31.0.1/24 dev br-inet; ip link set br-inet up

setup_side() {  # idx wan_ip lan_subnet lan_gw node_ip
  local nat="nat$1" node="node$1"
  ip netns add "$nat"; ip netns add "$node"
  ip link add "wan$1" type veth peer name "wan${1}p"
  ip link set "wan$1" master br-inet up; ip link set "wan${1}p" netns "$nat"
  ip netns exec "$nat" ip addr add "$2/24" dev "wan${1}p"; ip netns exec "$nat" ip link set "wan${1}p" up
  ip netns exec "$nat" ip link set lo up; ip netns exec "$nat" ip route add default via 172.31.0.1
  ip link add "lan$1" type veth peer name "lan${1}p"
  ip link set "lan$1" netns "$nat"; ip link set "lan${1}p" netns "$node"
  ip netns exec "$nat" ip addr add "$4/24" dev "lan$1"; ip netns exec "$nat" ip link set "lan$1" up
  ip netns exec "$node" ip addr add "$5/24" dev "lan${1}p"; ip netns exec "$node" ip link set "lan${1}p" up
  ip netns exec "$node" ip link set lo up; ip netns exec "$node" ip route add default via "$4"
  ip netns exec "$nat" sh -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'
  ip netns exec "$nat" iptables -t nat -A POSTROUTING -s "$3" ! -d "$3" -j MASQUERADE
}
setup_side 1 172.31.0.2 10.1.0.0/24 10.1.0.1 10.1.0.2
setup_side 2 172.31.0.3 10.2.0.0/24 10.2.0.1 10.2.0.2

echo "=== (1) MAPPING TYPE: node1:5000 -> rendezvous AND -> natB, compare observed src ==="
# listener on root (rendezvous IP) and on node2's WAN-facing nat (172.31.0.3 is natB)
python -c "
import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(('172.31.0.1',6001)); s.settimeout(4)
try: d,a=s.recvfrom(9); print('  rendezvous saw src',a)
except: print('  rendezvous TIMEOUT')
" &
ip netns exec nat2 python -c "
import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(('172.31.0.3',6002)); s.settimeout(4)
try: d,a=s.recvfrom(9); print('  natB saw src',a)
except: print('  natB TIMEOUT')
" &
sleep 1
ip netns exec node1 python -c "
import socket,time
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(('10.1.0.2',5000))
s.sendto(b'x',('172.31.0.1',6001)); time.sleep(0.3); s.sendto(b'x',('172.31.0.3',6002))
"
sleep 4

echo "=== (2) SIMULTANEOUS OPEN: node1:8002 <-> node2:8002 via reflexive (172.31.0.2/.3) ==="
ip netns exec node1 python -c "
import socket,threading,time
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(('10.1.0.2',8002)); s.settimeout(8)
threading.Thread(target=lambda:[ (s.sendto(b'A',('172.31.0.3',8002)),time.sleep(0.2)) for _ in range(30)],daemon=True).start()
try: d,a=s.recvfrom(9); print('  A RECEIVED from',a)
except: print('  A TIMEOUT (punch failed)')
" &
ip netns exec node2 python -c "
import socket,threading,time
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(('10.2.0.2',8002)); s.settimeout(8)
threading.Thread(target=lambda:[ (s.sendto(b'B',('172.31.0.2',8002)),time.sleep(0.2)) for _ in range(30)],daemon=True).start()
try: d,a=s.recvfrom(9); print('  B RECEIVED from',a)
except: print('  B TIMEOUT (punch failed)')
" &
wait
echo "=== done ==="
