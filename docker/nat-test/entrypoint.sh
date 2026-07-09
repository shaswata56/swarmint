#!/usr/bin/env bash
# Dual-role entrypoint. SWARM_KIND selects behaviour:
#   gateway : turn this container into a NAT router for its LAN
#   swarm   : (default) route via a gateway, then run the swarmint node daemon
set -euo pipefail

log() { echo "ENTRY $*"; }

if [[ "${SWARM_KIND:-swarm}" == "gateway" ]]; then
    # ---- NAT gateway role -------------------------------------------------
    # Masquerade everything originating on the LAN subnet as it leaves toward
    # the "public" net. This is a REAL Linux netfilter NAT (conntrack), not a
    # simulation — the whole point of the harness.
    : "${LAN_SUBNET:?gateway needs LAN_SUBNET (e.g. 10.10.1.0/24)}"
    # net.ipv4.ip_forward is enabled via the compose `sysctls:` key (the slim
    # image has no `sysctl` binary), so we only install the NAT rule here.
    log "gateway: MASQUERADE for ${LAN_SUBNET}"
    iptables -t nat -A POSTROUTING -s "${LAN_SUBNET}" ! -d "${LAN_SUBNET}" -j MASQUERADE
    iptables -P FORWARD ACCEPT
    # Optional: emulate a SYMMETRIC NAT (per-destination port allocation) so the
    # hole-punch path can be proven to FAIL where relay fallback is required.
    if [[ "${NAT_SYMMETRIC:-0}" == "1" ]]; then
        log "gateway: SYMMETRIC mode (--random-fully port allocation)"
        iptables -t nat -A POSTROUTING -s "${LAN_SUBNET}" ! -d "${LAN_SUBNET}" \
            -p udp -j MASQUERADE --random-fully
    fi
    log "gateway: up; iptables nat table:"
    iptables -t nat -L POSTROUTING -n -v || true
    exec sleep infinity
fi

# ---- swarm node role ------------------------------------------------------
# Point the default route at our LAN's NAT gateway so all off-LAN traffic is
# forced through the real NAT (Docker's own bridge would otherwise offer a
# direct, un-NATed path out).
if [[ -n "${SWARM_GATEWAY:-}" ]]; then
    log "node: default route via ${SWARM_GATEWAY}"
    ip route replace default via "${SWARM_GATEWAY}"
fi
log "node: routes:"; ip route || true

exec python -m swarmint.sim.net_daemon
