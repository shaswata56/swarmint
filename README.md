# swarmint

[![tests](https://github.com/shaswata56/swarmint/actions/workflows/tests.yml/badge.svg)](https://github.com/shaswata56/swarmint/actions/workflows/tests.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Decentralized, Byzantine-robust swarm learning.** Thousands of tiny models
each learn from their own local data and gossip knowledge peer-to-peer over a
real P2P network — no central server, no gradient sharing, no raw data leaving
the device. The swarm converges to near-centralized accuracy while tolerating
malicious nodes trying to poison it.

License: **AGPL-3.0-or-later** · Status: **research prototype (v0.1, alpha)** · Python 3.10+

---

## What it is

Each node runs a **prototype micro-model** (a bounded set of labeled exemplars +
nearest-neighbor) and shares knowledge by gossiping prototypes, not gradients.
Merging is order-invariant, so there's no weight-averaging fragility. On top of
that core:

- **Gossip federation** — each node sees only a slice of the classes; gossip
  pools them so every node approaches full-task accuracy.
- **Byzantine robustness** — merge-validation, corroboration, and reputation
  make label-flip poisoning fail; malicious nodes lose trust and are ignored.
- **Real P2P stack** — UDP transport, Kademlia DHT discovery, peer-exchange,
  Ed25519-signed messages with replay protection, NAT hole-punching.
- **Distributed inference** — trust-weighted mixture-of-experts across peers;
  a node answers queries about classes it never learned by consulting the swarm.
- **Extras** — tamper-evident per-node update chains, topic routing, encoder-free
  multimodal late fusion, and an optional shared genesis embedding that lifts
  accuracy toward centralized on many-shot data.

## Why

Federated learning that averages neural-net weights is fragile (independently
trained nets don't average well) and usually needs a central coordinator.
swarmint replaces both: order-invariant prototype merging removes the averaging
problem, and gossip + a DHT remove the coordinator. It targets edge/private-data
settings where data can't be centralized but many parties want a shared model.

## Install

```bash
pip install -e ".[net,learn]"   # P2P networking + scikit-learn (needed to join the live swarm)
pip install -e ".[dev]"         # everything (adds the cluster harness)
pip install -e .                # core only (numpy) — runs the in-process simulator
```

## Join the live swarm

There is a **live, public swarm** running right now. One command connects you to
it over the real internet (NAT hole-punching + relay fallback are automatic), and
your node's accuracy climbs as it corroborates knowledge from the swarm:

```bash
swarmint join            # join beacon.swarmint.org and learn real digits; Ctrl-C to leave
```

Your node learns to classify **real handwritten digits** (sklearn digits, bundled —
no download) it mostly never saw locally: each `EVENT metrics` line shows its
held-out `acc=` rising toward the ceiling as the honest quorum corroborates the
missing classes. Nothing but signed prototypes crosses the wire — no raw data, no
gradients, no central model. Live status: **https://beacon.swarmint.org**

> The public beacon runs an always-on honest **backbone** (`swarmint backbone`)
> so a joiner always has the ≥3-sender quorum the Byzantine defense requires;
> without a quorum a lone node correctly stays at its solo ceiling.

## Ask the live swarm a question

You don't have to run a long-lived learning node to use the swarm's knowledge —
one command classifies a digit against the live backbone and exits:

```bash
swarmint infer                       # classify a held-out digit; print the swarm's answer
swarmint infer --eval 40             # accuracy over 40 held-out digits (verified live: 0.925)
swarmint infer --image mydigit.txt   # classify your own 8x8 digit (64 raw pixels, sklearn scale)
```

It joins the beacon, reconstructs the backbone's deterministic addresses (no DHT
wait), embeds the query through the **same frozen genesis** every node shares, fans
out to all experts with the trust-ranked inference service, and prints the label,
an aggregate confidence, and every expert's vote. Verified over the real internet
from a corporate NAT: **37/40 = 0.925**, i.e. the centralized ceiling.

> A transient querier has no local holdout to calibrate peer trust against, so
> `infer` uses **uniform trust** — aggregation is confidence-weighted plurality
> with abstention, which is exactly right against the known-honest public backbone.
> The Byzantine-inference result (0.79 with 30% liars) assumes *calibrated* trust
> and is a separate claim this front door does not inherit.

## Quickstart (offline)

```bash
swarmint sim                            # 100-node learning sim (synthetic, exact 0.959)
swarmint multimodal-demo                # multimodal late fusion over the REAL UDP/DHT stack
python benchmark.py                     # reproduce the headline claims (~40s, one table)
python examples/quickstart.py           # smallest end-to-end: gossip beats solo
python -m swarmint.sim.run_digits       # real data (sklearn digits): swarm vs solo
python -m swarmint.sim.run_multiproc    # real UDP + DHT across OS processes
python run_tests.py                     # the test suite (plain scripts, no pytest needed)
```

## Host your own swarm

```bash
swarmint beacon --public-host <IP> --http-port 8080 --name my-beacon   # a rendezvous/relay
swarmint backbone --beacon <IP> --public-host <IP>                      # an always-on honest quorum
```
See [`deploy/gcp-beacon/`](deploy/gcp-beacon/) for a one-command free-tier GCP
deployment (systemd + HTTPS status page).

### Beacon federation — one singular system

Your beacon doesn't stand alone, and there is **no master**. By default `swarmint
beacon` **joins the global federation**: it bootstraps from the **genesis beacon**
(`beacon.swarmint.org` — the well-known DNS entry point, *not* an authority), which
verifies your beacon is genuinely reachable by probing its advertised address, then
gossips it into a decentralized, signed directory. **Every beacon holds the full
directory** — links to all beacons and their swarms — and gossips it peer-to-peer.
Think of each beacon as a specialized brain region and the directory as the
white-matter tracts connecting them: kill the genesis and the map survives (every
beacon holds it; a restarted genesis re-learns it from the others), and a newcomer
can bootstrap off *any* beacon — the genesis is just the default door.

```bash
swarmint beacon                  # host a beacon in ONE command: random unique id, auto friendly
                                 # name (beacon-<adj>-<noun>-<hex>), joins the federation
swarmint beacon --domain my.example.com   # ...advertise your own domain (you front the TLS)
swarmint beacons                 # list the whole mesh from any beacon (they all hold it)
```

Each beacon's status page shows the **whole swarm's peers**, not just its own —
every node is aggregated across all beacons over the same **signed P2P census
gossip** (no central server, no server-to-server HTTP), with a "seen via" column
naming which beacons vouch for each peer. Click any beacon's id in the directory
to open that beacon's own status page.

You confirm your setup works by seeing your beacon appear **reachable** in the
directory — the same view from any beacon. Beacon adverts are signed with each
beacon's Ed25519 key, so no beacon can forge or alter another it relays, and only
genuinely-reachable beacons are listed (anti-Sybil). Point at a different bootstrap
beacon with `--genesis`, or run standalone with `--no-federate`.

#### Cross-beacon learning — N beacons, one swarm

Federated beacons don't just *see* each other — if they share the same embedding
space (the advert carries a space fingerprint), they **learn as one system**. Two
compatible, reachable beacons bridge: each becomes a gossip peer of the other and
introduces its local swarm via a directed peer-exchange, so the two node
populations interconnect and prototypes flow across the boundary — over the exact
same signed-gossip + order-invariant-merge path used within one swarm, so trust,
the ≥3-sender corroboration quorum, and conflict rejection all stay intact. A
beacon whose swarm only ever saw half the classes ends up covering all of them,
learned entirely from its peer beacon's swarm. Different-space beacons still coexist
in the directory as separate regions (the path to cross-modal fusion), they just
don't cross-learn.

```bash
python -m swarmint.sim.run_cross_beacon   # two beacons, disjoint class halves -> one full model
```

Biologically: each beacon is a specialized region; the federation is the tract map;
bridging is inter-regional projection. Wire the regions and the halves become one
integrated model — no coordinator, no central model, just prototypes crossing signed
links. Verified over real UDP: two beacons each covering 3 of 6 classes converge to
all 6 (0.5 → 1.0) within seconds of bridging.

`python benchmark.py` is the fastest way to verify the project does what it
claims: it runs the federation, Byzantine-suppression, distributed-inference,
and cross-NAT-relay scenarios and prints a PASS/FAIL table (exit code 0 iff all
pass), so it doubles as a regression guard for the headline behaviour.

The full experimental record — including negative results and design reasoning —
is in the **Experimental record** section below.

## Headline results

| setting | solo baseline | swarm | notes |
|---|---|---|---|
| synthetic, 100 nodes, 5% malicious | 0.20 | **0.96** | stable, Byzantine-robust |
| real digits (10-class) | 0.20 | 0.78 → **0.88** | 0.88 with a shared genesis embedding |
| real faces (40-class, few-shot) | 0.05 | **0.51** | 10× solo on a brutal dataset |
| distributed inference (30% liars) | 0.20 | **0.79** | mixture-of-experts across peers |

Full-1-NN centralized ceilings are 0.93 (digits) / 0.83 (faces) — the swarm
reaches these while staying decentralized and poison-resistant.

## Status & scope

Research prototype. The learning core, P2P stack, and Byzantine defenses are
validated on synthetic and two real datasets. Deliberately **out of scope**: a
global consensus ledger (per-node tamper-evidence only, by design), production
hardening, and cross-modal *semantic* alignment (late fusion only). See
[CONTRIBUTING.md](CONTRIBUTING.md) for where help is most useful.

---

# Experimental record

Everything below is the detailed, honest research log — including negative
results and the reasoning behind each design choice.

## Results

**Phase 1 — collective learning** (100 nodes, 10 classes, 2 classes/node, 5% malicious, lossy net):

| metric | solo (no gossip) | swarm |
|---|---|---|
| global test accuracy @ round 100 | 0.20 | **0.96, stable** |

**Phase 2 — distributed inference / mixture of experts** (50 specialist experts,
30% malicious answering confidently wrong; a generalist query node that locally
knows only 2 of 10 classes):

| inference strategy | global accuracy |
|---|---|
| local-only (query node's own model) | 0.20 |
| ensemble, naive majority vote | 0.30 |
| ensemble, trust-weighted + abstention | **0.79** |

**Phase 3 — real networking** (16 real OS processes, real UDP + Kademlia DHT +
Ed25519 signing, DHT+PEX peer discovery, 1 malicious, gossip chunked to fit a
1200 B datagram):

| criterion | result |
|---|---|
| mean honest accuracy | **0.95** (≥0.90 target) |
| malicious vs honest trust | 0.67 vs 0.99 (suppressed) |
| bandwidth/node (production-normalized) | 1354 B/s (<2000 target) |
| chaos: kill 30% + restart + 20% packet loss | survivors **0.975**, restarted node rejoins to **0.967** |

Run: `python -m swarmint.sim.run_sim`, `python -m swarmint.sim.run_inference`
(numpy only), and `python -m swarmint.sim.run_multiproc` (needs the pinned deps
in requirements.txt: kademlia, pynacl, msgpack, psutil). Tests: run each
`tests/test_*.py` file directly (plain-script runners, no pytest needed).

## Design (what survived contact with the simulator)

- **Micro-model** ([prototype_model.py](swarmint/core/prototype_model.py)): bounded set of ≤200
  labeled exemplars + 1-NN. Merge = order-invariant union + online-k-means-style
  pruning. Versioning is a scalar `(counter, node_id)` clock.
- **Merge validation** ([swarm_node.py](swarmint/node/swarm_node.py)): candidate merge is a pure
  function; the receiver tests it on its own holdout ring buffer and commits
  only if accuracy doesn't degrade. Trust = EMA of merge outcomes; peers below
  a floor are ignored.
- **Conflict rule**: an incoming prototype on top of an existing one with a
  different label is a contradiction — dropped (and counted against the sender)
  unless the sender's trust clears an override threshold.
- **Quarantine + majority corroboration** — the key mechanism, discovered
  through three failed iterations (see below): prototypes for classes the node
  *cannot validate locally* are never committed directly. They sit in a
  quarantine buffer until **3 distinct senders** claim the same label in the
  same region AND that label is the majority claim there.

## Lessons from the failed iterations (do not regress these)

1. **Random-junk poisoning is a strawman.** Off-manifold junk is never a
   nearest neighbor; it passes validation harmlessly. Adversarial tests must
   use on-manifold, label-flipped data.
2. **Holdout validation alone allows poison laundering.** A node can only
   validate its own classes; label-flipped poison for other classes passes
   validation, commits, and is then re-gossiped by honest nodes. Accuracy
   peaked 0.94 then decayed to 0.57.
3. **First-writer-wins conflict rules protect the poison** that arrives before
   the truth.
4. **2-sender corroboration falls to collusion**: with 5 attackers, pairs
   sharing a class corroborate each other's flips (decay returned). 3-sender
   + majority-vote corroboration held.

## Phase 2 design (distributed inference)

- **Aggregation** ([aggregate.py](swarmint/inference/aggregate.py)): pure trust-weighted plurality
  vote, weight = `trust × confidence`. Returns the winner's share of total
  weight as a calibrated ensemble confidence for the fallback decision.
- **Abstention** ([swarm_node.py](swarmint/node/swarm_node.py) `answer_query`): an expert returns
  `None` when the query's margin-confidence is below `ABSTAIN_MARGIN` — i.e. the
  query is off its manifold. This is what makes coarse-topic routing work: only
  relevant experts vote, so fan-out can be wide without polluting the result.
- **Trust calibration** (`calibrate_trust`): the querier tests each peer on its
  *own* holdout (whose labels it knows) and rewards correctness. This is the
  inference-time analog of merge validation and is what sinks a confidently-lying
  expert — but *only for classes the querier can verify*.

### Phase 2 lessons (do not regress)

5. **A nearest-neighbour expert that never abstains poisons every ensemble.**
   Without abstention, off-domain experts guess a wrong class and calibration
   punishes honest experts for it (trust collapsed to ~0.37; ensemble 0.27).
   Margin-confidence is a scale-free OOD detector (0.01 off-manifold vs 0.9
   in-domain) — use it to abstain.
6. **Trust calibration only protects verifiable classes.** The querier can't
   label classes it's never seen, so for those it falls back to honest-majority-
   by-confidence — which holds only while honest experts outnumber colluding
   liars (same limit as Phase 1 corroboration). This is why Phase-2 accuracy is
   0.79, not 1.0: the 8 unverifiable classes carry residual malicious noise.
7. **Fan-out must exceed topic expert-diversity.** With coarse topics a class is
   served by few peers; a narrow top-5 poll by near-uniform trust misses them.
   Abstention makes over-polling free, so widen fan-out (default 20).

Residual known weaknesses (fine for now, revisit later): attacker
trust is neutralized structurally but only mildly punished (~0.64 vs ~0.69);
corroboration is Sybil-vulnerable without identity cost; corroboration radius
(4.0) is data-scale dependent and should become adaptive.

## Phase 3 design (real networking)

The governing invariant: **the proven learning/inference logic is untouched.**
`SwarmNode` stays synchronous; all async lives behind a `Bus` interface. Both
`SimBus` (deterministic sim) and `UdpBus` (real sockets) implement it, and
`test_core.py` reproduces the exact Phase-1/2 numbers as proof.

- **Transport** ([bus.py](swarmint/network/bus.py) / [udp_bus.py](swarmint/network/udp_bus.py)): asyncio
  `DatagramProtocol`; a Windows WinError-10054 restart wrapper; oversized
  gossip batches auto-chunked to ≤12 protos/1200 B datagram.
- **Wire** ([wire.py](swarmint/network/wire.py)): msgpack (not protobuf — no codegen, no
  gencode/runtime version-lock in a mixed-version mesh); float32 vectors as raw
  blobs; a measured 1200 B budget test.
- **Identity** ([identity.py](swarmint/network/identity.py)): Ed25519; `node_id = SHA1(pubkey)[:20]`;
  every envelope verified `sig` **and** `sender == H(pubkey)` before the trust
  table is touched (blocks trust-impersonation); FIFO nonce replay guard.
- **Discovery** ([discovery.py](swarmint/network/discovery.py)): Kademlia DHT for rendezvous
  (per-node-id keys, since the lib is single-value last-writer-wins) + peer
  exchange (PEX) for peer-set growth. Multi-bootstrap for chaos survival.
- **NodeRunner** ([runner.py](swarmint/node/runner.py)): the one asyncio task that drives a
  node's sync methods on a wall-clock gossip timer — no locks needed.
- **Inference RPC** ([rpc.py](swarmint/inference/rpc.py)): `infer`/`calibrate_trust`
  redesigned to request-collect-aggregate over the bus, reusing the pure
  `answer_query`/`aggregate` byte-for-byte.
- **Resource governor** ([resources.py](swarmint/sim/resources.py)): caps the multiprocess
  harness at 60% of CPU cores and RAM by default; configurable via
  `ResourceBudget(...)` args, `--cpu-fraction`/`--mem-fraction`/`--max-nodes`
  CLI flags, or `SWARMINT_*` env vars.

### Phase 3 lessons (do not regress)

8.  **A nearest-neighbour node that never abstains poisons the network too** —
    same as lesson 5, but here `gossip_step`'s 30-proto sample × real UDP made
    it a 100%-packet-loss bug: any batch >12 protos silently exceeded the
    datagram budget until chunking was added.
9.  **Coverage must be guaranteed over HONEST nodes only.** Counting malicious
    nodes toward per-class corroboration coverage leaves classes with <3 honest
    senders → permanently unlearnable → accuracy pinned at (learnable/N). This
    was a 0.36-accuracy bug at 16 nodes.
10. **Corroboration tolerates only malicious < honest-supply-per-class.** At the
    60%-CPU node cap (~16 nodes) each class has ~3–4 honest coverers, so ~3
    colluding attackers overwhelm it and accuracy *peaks then decays* (the
    Phase-1 poison-laundering curve, live). Robustness to a higher malicious
    fraction needs more nodes, not a code change — the documented Sybil/scale
    limit. Default threat model kept at Phase-1's ~5%.
11. **Two peer-address stores must stay in sync.** The rendezvous node never
    calls DHT `lookup()`, so its PEX payload (read from `discovery.peer_addrs`)
    stayed empty and it could never introduce peers — every verified sender now
    updates both the bus and discovery address maps.

## Deferred items — now implemented

- **D1 tamper-evident hash-chain** ([update_chain.py](swarmint/core/update_chain.py)): each node
  keeps a Git-like, Ed25519-signed, append-only chain of its model-version
  checkpoints; any peer can fetch and verify it (contiguity + signatures +
  id↔pubkey binding), making a node's update history auditable and
  non-repudiable without any consensus. Opt-in — the deterministic sim is
  untouched. Fetch is chunked over UDP; tamper/reorder/forgery all detected.
- **D2 finer topic granularity** ([discovery.py](swarmint/network/discovery.py) topics,
  [net_node.py](swarmint/node/net_node.py) `infer_topic`): topics are a first-class per-node
  *set*, advertised via the DHT and propagated via PEX, so gossip and inference
  route to topic-relevant peers instead of the whole swarm. Kademlia can't
  enumerate members, so membership rides PEX (the same channel peer addresses do).
- **D3 NAT traversal** ([nat.py](swarmint/network/nat.py)): rendezvous-coordinated UDP
  hole-punching — reflexive-address discovery (STUN-lite), connect signaling
  relayed through the rendezvous, and simultaneous punch — riding the signed
  bus so a rendezvous can't forge a peer. The loopback test validates the full
  handshake and a resulting direct A→B datagram; opening a *real* NAT mapping
  needs two actual NATs (the protocol implemented is the one that does it).

Still genuinely out of scope: a real multi-NAT test environment; a global
tamper-evident consensus (D1 is per-node only, by design).

## Real-data validation (sklearn digits)

Everything above ran on synthetic Gaussians. This is the first test on real data
— sklearn's 8×8 handwritten digits (10 classes, bundled, offline) — running the
*same* machinery (non-IID shards, gossip, corroboration, trust, a malicious
node). Each node sees only 2 of 10 classes locally.

| approach | held-out accuracy |
|---|---|
| solo node (2/10 classes) | ~0.20 |
| **swarm (gossip), Byzantine-robust** | **~0.78** |
| full-knowledge 1-NN (all points, no budget, no defense) | 0.93 |

Malicious trust is crushed to **0.18** vs honest **0.80** — Byzantine defense
works on real data, not just Gaussians. Run: `python -m swarmint.sim.run_digits`.

### Second dataset — Olivetti faces (validate-first, harder)

To check digits wasn't a lucky case, the same harness ([real_data.py](swarmint/sim/real_data.py),
shared by `run_digits`/`run_faces`) was run on Olivetti faces: **40 classes,
4096-dim, only 10 samples/class (few-shot)**, and same-class distances that
*overlap* cross-class — the hardest case for distance gating.

| dataset | solo ceiling | swarm | full-1-NN ceiling | malicious/honest trust |
|---|---|---|---|---|
| digits (10-class, many-shot) | 0.20 | 0.78 | 0.93 | 0.18 / 0.80 |
| faces (40-class, few-shot) | 0.05 | **0.51** (10× solo) | 0.83 | 0.40 / 0.58 |

The thesis generalizes across two very different real regimes. Two honest
caveats faces surfaced: (1) the **Byzantine margin narrows** when class
distances overlap (0.40 vs 0.58) — poison is harder to tell from real
cross-class variance; (2) **raw high-dim input is slow** — prototype distance is
O(prototypes × dims), so 4096-dim faces cost ~64× per op vs digits. Both point
the same way: a dimensionality-reducing embedding in front of the swarm would
help *performance and separability at once* (the concrete case for the encoder
question, beyond cross-modal semantics). `run_faces` is the showcase;
`test_faces` is opt-in (`SWARMINT_SLOW_TESTS=1`) since 4096-dim is slow.

**Two honest findings the digits run surfaced (that synthetic never would):**

1. **The radii had to become data-derived.** The hand-tuned constants were
   calibrated for the synthetic scale and misfired on normalized digits
   (same-class 0.59 vs cross-class 0.80 — only 0.21 apart → 37% merge-rejection).
   All radii (`absorb_near/full`, `conflict_radius`, `override_trust`,
   `corroboration_radius`, `abstain_margin`) are now **configurable** (defaults
   unchanged → the 0.959 sim and every test are byte-identical) and
   [run_digits.py](swarmint/sim/run_digits.py) *derives* them from the dataset's own same/cross-class
   distance statistics. This retires the data-scale-dependent-radius weakness
   flagged since Phase 1. Deriving them took digits 0.59 → 0.78 and rejection
   37% → 10%.
2. **The 0.78-vs-0.93 gap is structural, not a tuning miss** (verified: more
   budget + tighter absorb plateaus, doesn't climb). It's the price of
   decentralization + Byzantine robustness + bounded memory on *overlapping*
   real classes: the conservative merge-validation and corroboration gating that
   make the swarm poison-resistant also reject/quarantine some legitimate
   knowledge. Full 1-NN reaches 0.93 precisely because it keeps every point and
   trusts everything — no budget, no defense. Closing the gap is a real research
   direction (adaptive per-class region estimation / a learned metric), not a
   config tweak.

## Shared genesis embedding — breaking the 0.78 ceiling (E1–E3)

Radius geometry capped at ~0.78 on real digits (structural gap). The lever that
breaks it: run the swarm in a **learned lower-dim space** instead of raw
features. The invariant makes this subtle — a prototype is only comparable to
another if both came from the *same* space, so the projection must be identical
across all nodes. A node can't fit its own ([test_embedding.py](tests/test_embedding.py) proves
per-node fitting diverges). The decentralized fix is the spec's own **"public
benchmark seeded at genesis"**: fit the projection ONCE on a small public seed,
freeze it to a plain `(mean, W)` linear op, and distribute it as a protocol
parameter — a shared frozen encoder, just linear ([embedding.py](swarmint/core/embedding.py)).

A/B on digits (`run_embedding_ab`), each embedding fit on a public genesis seed,
swarm trained on the remainder:

| embedding | dims | swarm accuracy |
|---|---|---|
| identity (raw) | 64 | 0.66 |
| PCA (unsupervised) | 32 | 0.69 |
| **LDA (supervised), 40+/class seed** | **9** | **0.88** |

**LDA nearly closes the gap to the 0.93 full-1-NN ceiling — 0.66 → 0.88 — on 9
dims.** Three findings, all measured:

1. **It's the *supervised* separation that matters**, not dim reduction. PCA
   barely helps; LDA (maximizes between/within-class scatter) is the win.
2. **Seed size is decisive.** LDA on 8 samples/class *overfits and hurts* (0.64);
   40/class (400 total — a modest public benchmark) gives 0.88. My first A/B used
   8/class and got a misleading negative — corrected here.
3. **The lever needs public labeled data, so it fits many-shot but not few-shot.**
   Digits (180/class) → LDA soars. Faces (only 10/class) can't spare a good
   genesis seed → embedding helps *perf* (39 dims, ~100× cheaper distance ops)
   but not accuracy. This is the honest boundary: a shared learned embedding and
   the swarm's private-data premise are in tension, resolved only when a modest
   public benchmark exists.

Net: with a small public seed, the swarm reaches near-centralized accuracy (0.88
vs 0.93) *while staying decentralized and Byzantine-robust* — the encoder-lite
path, validated. `SwarmNode` never changed; the embedding is shared preprocessing
at the data boundary.

## Multimodal — encoder-free (M1–M3)

**Finding: swarmint goes multimodal with no encoder and no learned feature map.**
The question was "transformer vs JEPA for the model"; the honest answer is the
transformer was never the model — the model is the prototype set — so the real
question is only how to (a) get each modality into a comparable space and (b)
fuse across modalities.

- **(a) No encoder needed for nonlinearity.** Raw prototype 1-NN is already
  nonlinear: it solves XOR-type tasks at a tiny budget because absorb/prune
  keep well-separated same-class blobs as distinct prototypes (verified). A
  fixed Random Fourier Feature map ([features.py](swarmint/core/features.py)) is available —
  seed-distributable, no weights, no training — but it does **not** improve
  accuracy (a random map loses info vs raw for 1-NN; it scored *lower* in the
  demo). Keep it only if a modality needs scale-normalization into a common
  bounded space (a bonus: normalized features also retire the data-scale-
  dependent radius weakness). A *learned* encoder (JEPA/ImageBind) would only be
  worth it for **cross-modal semantic alignment** (image of a dog ≈ the word
  "dog"), which late fusion doesn't need.
- **(b) Late fusion via the machinery you already have.** Each modality is its
  own topic (D2) with its own raw prototype experts; a multi-view query
  aggregates per-modality expert votes with the Phase-2 trust/confidence
  mixture-of-experts. Independently-noisy modalities denoise each other.

Result ([run_multimodal.py](swarmint/sim/run_multimodal.py), 3 modalities, 6 classes, noisy):

| setting | best single modality | late fusion |
|---|---|---|
| centralized (one expert/modality) | 0.55 | **0.71** |
| decentralized (swarm gossip/modality, then fuse) | 0.65 | **0.84** |

Fusion beats the best single modality in both cases, and the swarm+fusion
compound (gossip pools non-IID shards → better per-modality experts → better
fusion). Zero new architecture: it's D2 topics + Phase-2 aggregation on raw
per-modality prototypes.

## Layout

```
swarmint/
├── core/       # PrototypeModel, Version, UpdateChain (tamper-evident, D1)
├── network/    # Bus interface; SimBus (in-process) + UdpBus (real UDP);
│               #   wire (msgpack), identity (Ed25519), discovery (Kademlia+PEX+topics),
│               #   nat (hole-punching, D3)
├── node/       # SwarmNode (sync learning/inference), NodeRunner, NetNode (composed P2P node)
├── inference/  # pure aggregation + async RPC (request-collect-aggregate)
├── sim/        # synthetic data; learning sim; inference demo; multiprocess
│               #   integration + chaos harness; resource governor
└── tests/      # unit + integration (merge, identity, UDP, discovery, chaos,
                #   update-chain, topics, nat)
```

Deferred to a later phase (unchanged from the plan): NAT traversal (LAN-only
for now); a per-node hash-chain of updates for tamper evidence, only if local
trust proves insufficient at scale; finer topic granularity than the current
single coarse topic.

Next (per revised roadmap, Phase 3): real networking — Kademlia/libp2p peer
discovery and message transport replacing SimBus; then the deferred lifecycle
parts (per-node hash-chain of updates for tamper evidence) only if local trust
proves insufficient at scale.
