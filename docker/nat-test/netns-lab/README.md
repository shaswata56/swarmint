# netns hole-punch lab

A clean-room NAT lab built from raw Linux network namespaces (`ip netns` + veth +
one `iptables MASQUERADE` per side), inside a single `--privileged` container.
Its purpose was to reproduce a **successful** hole-punch on a pristine
port-restricted-cone NAT — sidestepping the Docker bridge-netfilter +
stacked-MASQUERADE interference that the `../` Docker harness ran into.

```
 node1 10.1.0.2 ──lan── nat1[10.1.0.1 | 172.31.0.2]──┐
                                                       br-inet 172.31.0.1 (rendezvous)
 node2 10.2.0.2 ──lan── nat2[10.2.0.1 | 172.31.0.3]──┘
```

## Run

```bash
# needs the image built by ../: docker compose -f ../docker-compose.yml build
tr -d '\r' < punch-lab.sh  | docker run --rm -i --privileged --entrypoint bash swarmint-nat:test -s   # swarmint daemon
tr -d '\r' < punch-diag.sh | docker run --rm -i --privileged --entrypoint bash swarmint-nat:test -s   # raw-UDP probes
```

(`-i` is required — without it `bash -s` reads a closed stdin and does nothing.
`tr -d '\r'` guards against CRLF checkouts breaking the shebang.)

## Findings

| Probe | Result |
|-------|--------|
| Nodes sit behind a **real NAT** (reflexive `172.31.0.2/.3` ≠ local `10.x.0.2`) | ✅ both |
| Mapping is **endpoint-independent** (cone): same src port → same public port to two dests | ✅ confirmed |
| swarmint signaling: rendezvous relays each peer's reflexive addr | ✅ both learn each other |
| **Hole-punch opens a bidirectional path** | ❌ fails — simultaneous *and* sequenced |

### The punch fails even here — and it's not swarmint

`punch-diag.sh` runs the punch with **raw UDP sockets, no swarmint code**, both
as a simultaneous open and as a sequenced open (one side establishes its mapping
first). Both time out. So the failure is a property of the **Linux netfilter NAT
data-plane** (present in both Docker and hand-built netns), not of `nat.py` —
whose signaling (reflexive discovery + rendezvous address relay) is correct.

Conntrack shows each side's outbound mapping is created and port-preserved, and
by the textbook restricted-cone model the peer's punch packet *should* match the
reply tuple and be delivered — but in practice it isn't, across two independent
netfilter-MASQUERADE setups. Emulated netfilter NAT is evidently not a faithful
stand-in for the consumer-router NATs where WebRTC/ICE-style punching works in
the wild.

### Conclusion for the roadmap

- **Reliable cross-NAT connectivity rests on relay fallback (T4)** — which *is*
  verified working over real NAT (see `../README.md`). Hole-punch is a
  best-effort optimization, and this lab shows it cannot be depended upon in a
  Linux-NAT environment.
- A **green punch needs real router hardware** (home/mobile NATs), not a
  netfilter emulation. That is the remaining way to validate the optimization
  path; it requires physical/multi-site testing, not a container.
- `nat.py` is not implicated: its protocol exchanges the right addresses; the
  data-plane simply doesn't open under emulated netfilter NAT.
