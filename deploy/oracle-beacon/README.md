# Second beacon on Oracle Cloud (Always Free, forever)

A second swarmint beacon on Oracle's **Always Free** tier — a genuinely separate
host on a different cloud/network — that joins the federation via the genesis beacon
(`beacon.swarmint.org`) and **cross-learns** with the mesh. There's no master: every
beacon holds the full directory. Two independent beacons, one system, $0 forever.

Why Oracle (not a 2nd GCP VM): GCP Always Free is exactly **one** e2-micro per
billing account; a second would be billed. Oracle Always Free includes Arm Ampere
**A1 (up to 4 OCPU / 24 GB)** plus 2 AMD micro VMs — forever, with a public IPv4
and openable UDP.

## Steps you do once (console — needs your account)

1. **Account:** sign up at <https://cloud.oracle.com/free> (card is for identity
   only; Always Free is never charged). Pick an Always-Free-eligible home region.
2. **Instance:** Compute → Instances → **Create**.
   - Shape → **Ampere** → `VM.Standard.A1.Flex`, **1 OCPU / 6 GB** (well within free).
   - Image: **Ubuntu 22.04** (user `ubuntu`) or Oracle Linux (user `opc`).
   - **SSH keys:** upload your public key (or download the generated private key).
   - Keep the default **public IPv4** in a public subnet. Note the IP.
3. **Open UDP ingress (VCN):** Networking → your VCN → **Security Lists** →
   default → **Add Ingress Rules**:
   - Source `0.0.0.0/0`, IP Protocol **UDP**, Destination port range **`9001-9023`**
     (gossip + DHT + room for a local backbone). Leave "stateless" unchecked.
   - This is the one step the host-side script can't do; without it the genesis's
     reachability probe can't reach the beacon and it shows "unverified".

## One command on the box

```bash
curl -fsSL https://raw.githubusercontent.com/shaswata56/swarmint/main/deploy/oracle-beacon/bootstrap.sh -o bootstrap.sh
bash bootstrap.sh <PUBLIC_IP> oracle-1
```

That installs swarmint, opens the host firewall (firewalld/ufw/iptables — Oracle
images vary), and runs a **federation-peer** beacon (task=digits, distinct seed)
that bootstraps from the genesis. It learns the full digit model from the genesis's
swarm across the internet via cross-beacon bridging. Add a third arg (`bash
bootstrap.sh <IP> oracle-1 4`) to also run a local 4-node backbone so this beacon
contributes its own quorum.

## Verify (from anywhere)

```bash
swarmint beacons                                   # the directory (any beacon) — oracle-1 should be "up"
curl -s http://<PUBLIC_IP>:8080/federation.json    # this beacon's own view (holds the full mesh)
```

On the box, `sudo journalctl -u swarmint-beacon -f | grep -E "federation|metrics"` shows
`federation enabled=True role=peer` and the held-out `acc=` climbing toward the
ceiling as prototypes cross from the genesis's swarm.

## Cost guard (keep it forever-free)
- Use **only** Always-Free shapes: A1.Flex ≤ 4 OCPU / 24 GB total, or VM.Standard.E2.1.Micro.
- One A1 instance at 1 OCPU/6 GB is comfortably inside the free allowance.
- Boot volume: keep the default (Always Free includes up to 200 GB block storage).
- Don't add a public load balancer / extra reserved IPs (those can bill).
