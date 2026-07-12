# swarmint: Intelligence with No Center to Fail

**Decentralized, Byzantine-robust swarm learning by gossiping prototypes, not gradients.**

Shaswata Das · AGPL-3.0-or-later · [github.com/shaswata56/swarmint](https://github.com/shaswata56/swarmint) · live swarm: [beacon.swarmint.org](https://beacon.swarmint.org)

---

## Abstract

Federated learning that averages neural-network weights is fragile — independently
trained networks do not average well — and it almost always depends on a central
coordinator to synchronize rounds. **swarmint** removes both. Each participant runs
a tiny **prototype micro-model** (a bounded set of labeled exemplars with
nearest-neighbor prediction) and shares knowledge by gossiping *prototypes* rather
than gradients. Merging two models is **order-invariant** (set union followed by a
prune), so there is no averaging step to go wrong and no need to coordinate rounds.
Peers find each other over a Kademlia DHT, exchange signed messages over UDP, and
traverse NATs by hole-punching with a relay fallback. A layered Byzantine defense —
merge-validation, multi-sender corroboration, and per-peer reputation — lets the
swarm keep converging while roughly 5% of nodes actively try to poison it.

We validate the approach on synthetic mixtures (100 nodes, exact 0.959 accuracy vs.
0.20 solo), on three real modalities (image, tabular, text), across 16 real OS
processes over real UDP/DHT, and — the headline evidence — on **real hardware over
the open internet**: direct NAT hole-punches between a corporate-NAT PC and phones
on home Wi-Fi, Cloudflare WARP, and plain carrier CGNAT, plus a **live, public swarm
anyone can join in one command** and watch their node's accuracy climb from the solo
ceiling to the task ceiling. We also report negative results as first-class:
adaptive per-class radii and a learned shareable distance metric both fail to move a
structural accuracy ceiling that is the price of Byzantine robustness under bounded
memory.

---

## 1. Problem

Three properties are usually in tension in collaborative learning:

1. **No raw-data sharing.** Edge/private settings can't centralize data.
2. **No central coordinator.** A single coordinator is a single point of failure,
   a trust bottleneck, and an operational cost.
3. **Robustness to malicious participants.** Open networks contain liars.

Weight-averaging federated learning (FedAvg and kin) struggles with all three at
once. Averaging the parameters of independently trained neural networks is unstable
because loss landscapes are non-convex and permutation-symmetric — two good models
can average to a bad one. Byzantine-robust aggregation rules exist but add
coordinator-side complexity, and the coordinator itself remains central.

swarmint targets the intersection: **share knowledge, not data; discover peers, not
coordinate them; and make poisoning fail by construction.**

## 2. Design

### 2.1 Prototype micro-models

A node's entire model is a bounded set of **prototypes** — labeled exemplars in a
shared feature space — with nearest-neighbor prediction. Learning is online:
observe `(x, y)`, and either absorb the point into a nearby same-label prototype or
add it as a new prototype, pruning to a memory bound. There are no weights and no
gradients.

### 2.2 Order-invariant merging

Two models merge by **set union + prune**: take both prototype sets, resolve
conflicts, and prune back to the bound. Because union is commutative and
associative, **merge order does not matter** — the failure mode that makes
weight-averaging fragile simply does not exist here. This is what lets gossip be
asynchronous and coordinator-free: any node can merge anything it hears in any order.

### 2.3 Gossip federation

Each node sees only a **non-IID slice** of the classes, so no node can learn the
full task alone. Periodically each node gossips a random sample of its prototypes to
peers. Knowledge diffuses; every node approaches full-task accuracy while its raw
data never leaves the device.

### 2.4 Byzantine defense (layered)

- **Merge-validation.** An incoming prototype that contradicts local structure (a
  different label too close to an established prototype) is rejected.
- **Corroboration.** A class a node cannot itself verify is only promoted when
  **≥3 distinct senders** independently corroborate it. Two senders is insufficient
  — a colluding pair could manufacture agreement; holdout-only validation lets
  poison launder through. Three distinct senders is the smallest quorum that
  resists a single colluding pair.
- **Reputation (trust).** Each peer carries an EMA trust score; label-flip
  attackers lose trust and are down-weighted or ignored.

This holds while malicious supply stays below honest supply *per class-region*
(≈5% globally in our threat model). It is deliberately **not** a global consensus
ledger — per-node tamper-evidence only (a signed hash-chain of checkpoints), by
design.

### 2.5 Shared genesis embedding

Prototypes are only comparable if every node lives in the **same** feature space. A
node that fits its own projection drifts into an incomparable space and merging
silently breaks. swarmint resolves this with a **shared genesis embedding**: a
projection (Identity / random / PCA / **LDA**) fit **once** on a small public seed
at genesis, frozen to a plain `(mean, W)` linear operator, and distributed as a
protocol parameter (two small arrays) — every node applies the identical frozen
transform. On many-shot data this lifts accuracy toward the centralized ceiling
while keeping the decentralized, poison-resistant properties.

### 2.6 The P2P stack

UDP transport with msgpack framing; **Ed25519**-signed messages (`node_id =
SHA1(pubkey)[:20]`, bound to the sender) with replay protection; **Kademlia DHT**
rendezvous + peer-exchange for discovery; **NAT hole-punching** with a **relay
fallback** (and multi-relay failover) when a direct path can't be opened. Transport
and representation are strictly separated from the learning core.

### 2.7 Design invariants

1. The learning node is **synchronous and representation-agnostic**; all async,
   transport, and relay logic lives below it; all feature-space logic lives in the
   embedding layer.
2. The learning core is **frozen and byte-stable**: the deterministic 100-node sim
   reproduces **exactly 0.959**. New behavior ships behind opt-in flags with
   defaults unchanged.
3. **Shared-space invariant**: any embedding is identical across all nodes (fit
   once at genesis, frozen).
4. **No silent truncation** — anything capped is logged.

## 3. Results

### 3.1 Collective learning (synthetic)

100 nodes, 10 classes, 2 classes/node, 5% label-flip malicious, lossy network:

| metric | solo | swarm |
|---|---|---|
| held-out accuracy | 0.20 | **0.959** |

Malicious peers' trust is suppressed below honest peers'; the swarm converges and
stays converged. This exact figure is the project's regression anchor.

### 3.2 Distributed inference (mixture-of-experts)

A node answers queries about classes it never learned by consulting the swarm,
trust-weighting each expert and letting experts **abstain** off-manifold. With 30%
of experts as confident liars:

| strategy | accuracy |
|---|---|
| local-only | 0.20 |
| naive vote | 0.30 |
| trust-weighted + abstain | **0.79** |

### 3.3 Generality across modalities (real data)

Each dataset uses a shared LDA genesis embedding; solo → swarm, 5% malicious, with
the full-knowledge 1-NN ceiling for reference:

| modality | dataset | solo | swarm | 1-NN ceiling | gap closed |
|---|---|---|---|---|---|
| image | sklearn digits | 0.20 | **0.882** | 0.951 | 91% |
| tabular | covtype | 0.29 | **0.450** | 0.718 | 37% |
| text | 20-newsgroups (TF-IDF) | 0.33 | **0.389** | 0.552 | 27% |

The lift is a **gradient, not a uniform win**. It is strong on image data, modest on
tabular, and near-break on text — because a linear LDA genesis embedding cannot
separate high-dimensional sparse TF-IDF (its own 1-NN ceiling is only 0.55). A
rounds-sensitivity sweep (100 vs. 250 rounds) does not help, confirming the ceiling
is **structural**, not under-training. Byzantine gating held on every modality.

Earlier single-dataset runs: digits without the embedding reach 0.78 (data-derived
radii); Olivetti faces (40-class, few-shot, 10/class) reach **0.51** vs. 0.05 solo —
10× solo on a brutal dataset, though few-shot data can't spare an adequate genesis
seed, so the embedding helps perf but not accuracy there.

### 3.4 Multimodal late fusion over the real network

swarmint goes multimodal with **no encoder**: each modality is handled by raw
prototype experts in its own space (its own **topic**, D2), and modalities are fused
at the *decision* level by the same trust/confidence-weighted mixture-of-experts. In
process this beats the best single modality (0.65 → 0.84). We now confirm the same
over the **real stack** — UDP + Kademlia DHT + msgpack wire + topic-scoped inference:
30 expert nodes partitioned across 3 modalities (10 experts/topic, ≥3-sender coverage
per class), then a query node fuses per-modality answers over real sockets. Reproduced
across independent runs:

| run | best single modality | late fusion (real net) |
|---|---|---|
| 1 | 0.617 | **0.700** |
| 2 | 0.558 | **0.642** |

Fusion beats the best single modality over real sockets on every run — independently-
noisy views denoise when fused, with no encoder and no shared feature space across
modalities. (One finding surfaced: topic routing requires an authoritative DHT advert
lookup to populate a passively-learned peer's topic; PEX only spreads topic *hints*.
A thin-margin under-provisioned swarm can occasionally show fusion not beating the
best modality on a given real-timing run; 10 experts/modality with a larger held-out
set removes that flakiness.)

### 3.5 Real networking (16 processes)

16 real OS processes, real UDP + Kademlia DHT + Ed25519 signing:

| metric | value |
|---|---|
| honest accuracy | 0.95 |
| malicious trust vs. honest | 0.67 < 0.99 |
| bandwidth (production-normalized) | ~1354 B/s/node |
| chaos (30% killed + restart, 20% loss) | survivors keep converging; restarted node rejoins |

### 3.6 Real-hardware NAT traversal (the open internet)

A public **beacon** (hole-punch rendezvous + relay) runs on a free-tier GCP
e2-micro at [beacon.swarmint.org](https://beacon.swarmint.org). Over genuine router
NATs, direct hole-punches (relay unused, `relay_fwd=0`) succeeded on **three**
independent network types:

| peer A | peer B | result |
|---|---|---|
| corporate-NAT PC | home-Wi-Fi Android (port-remapped) | direct punch, ok |
| corporate-NAT PC | Cloudflare-WARP cellular | direct punch, ok |
| corporate-NAT PC | **plain carrier CGNAT** (port-remapped) | direct punch, ok |

No proxy is required on any path. This is something an emulated netfilter NAT
(netns/Docker) never achieved even for raw UDP — the punch is a property of the real
data-plane. Along the way we corrected an earlier wrong claim ("carrier blocks UDP")
with an actual raw-UDP echo test: the real cause was a fragile one-shot Kademlia DHT
bootstrap on lossy cellular, fixed by seeding the rendezvous gossip address directly
from the known bootstrap address.

### 3.7 A live, joinable swarm

The concept becomes a **living network** that learns a **real** task. The public
beacon hosts an always-on honest **backbone** (8 nodes covering all classes with
≥3-sender corroboration) that learns **sklearn handwritten digits** (bundled on
every machine — no data transfer) under a shared LDA genesis embedding. Anyone runs
one command and their node learns to classify digits it mostly never saw locally:

```bash
pip install "swarmint[net,learn]"
swarmint join            # → beacon.swarmint.org, learns real digits
```

Measured from a corporate-NAT PC (`nat_detected=True`) joining **over the open
internet** (direct P2P, `relay_fwd=0`), held-out accuracy on the **real digit test
set** climbed as the honest quorum corroborated each previously-unseen class,
reaching the same ~0.92 ceiling the offline digits experiments hit:

| t (s) | 20 | 30 | 40 | 50 | 60 | 80 | 90 | 120 |
|---|---|---|---|---|---|---|---|---|
| accuracy | 0.15 | 0.33 | 0.42 | 0.66 | 0.75 | 0.85 | 0.89 | **0.92** |

Nothing but signed prototypes crosses the wire — no raw data, no gradients, no
central model. A lone joiner reaching an *empty* network correctly stays at its solo
ceiling: the ≥3-sender rule refuses to trust an unbacked source. The backbone exists
precisely to supply that honest quorum.

## 4. Honest limits (negative results as first-class)

- **Structural accuracy ceiling.** On digits, the swarm reaches ~0.88 against a 1-NN
  ceiling of ~0.95. This gap is **structural** — the cost of Byzantine gating +
  bounded memory on overlapping classes — not a tuning problem. Two independent
  attempts to close it failed and were documented:
  - **Adaptive per-class radii** made accuracy *worse* and weakened the Byzantine
    margin; reverted.
  - **A learned shareable metric.** Since a Mahalanobis metric is Euclidean distance
    after a linear map, a learned metric drops into the same frozen-genesis slot as
    the embedding. We tried **NCA** (which optimizes exactly the 1-NN objective the
    model uses) and whitened Mahalanobis. NCA — even initialized at the LDA solution
    and optimized for 200 iterations — reached **0.870**, *below* LDA's **0.882**;
    whitened Mahalanobis collapsed to 0.482. The metric is not the lever; a
    supervised linear LDA genesis embedding is already at the useful frontier.
- **Few-shot genesis.** The shared LDA embedding needs an adequate public seed
  (≈40/class). It fits many-shot data (digits, covtype, 20news) but not few-shot
  (faces, 10/class).
- **Text near-break.** Linear genesis embeddings cannot separate sparse TF-IDF; a
  nonlinear-but-still-shareable genesis is open work.
- **Scope.** No global consensus ledger (per-node tamper-evidence only, by design).
  Cross-modal *semantic* alignment is out of scope (late fusion only). Sybil
  resistance beyond the ≈5% threat model needs more honest redundancy per class
  (more nodes), not a code change.

## 5. Reproduce

```bash
git clone https://github.com/shaswata56/swarmint && cd swarmint
pip install -e ".[dev]"

python benchmark.py                  # headline claims as a PASS/FAIL table (exit 0 iff all pass)
swarmint sim                         # 100-node synthetic sim, exact 0.959
python -m swarmint.sim.run_generality   # image/tabular/text sweep (SWARMINT_SLOW_TESTS=1)
python -m swarmint.sim.run_multiproc    # 16 real processes over UDP + DHT
python run_tests.py                  # full test suite (no pytest needed)

swarmint join                        # join the LIVE public swarm
```

The beacon + backbone are one-command deployable to a free-tier GCP e2-micro; see
[`deploy/gcp-beacon/`](deploy/gcp-beacon/). Continuous integration runs the test
matrix (Python 3.10–3.13) and the benchmark job on every push.

## 6. Conclusion

Order-invariant prototype merging removes weight-averaging fragility; gossip over a
DHT removes the coordinator; a layered trust/corroboration defense removes the
assumption that participants are honest. The result is a learning system with **no
center to fail** — validated not only in simulation but as a live network that
tolerates real NATs, real packet loss, and real adversaries, and that anyone can
join today. Its accuracy ceiling is the honest price of that robustness, and we
report the attempts to lower it that did not work as plainly as the ones that did.
