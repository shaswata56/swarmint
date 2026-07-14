# swarmint browser client — peer-first P2P inference

Draw a digit; your **browser** signs the query, opens a **WebRTC data channel
directly to a live swarm node**, and the node answers with its own
locally-learned model. The beacon only exchanges the SDP/ICE handshake — the
same signaling role it already plays for UDP hole-punching — and is never in
the data path once the channel opens.

This is deliberately **not** a gateway: the beacon does not relay your query or
see your drawing. It introduces two peers; they talk directly.

## Files

- `infer.html` — the page. No build step, no bundler — plain ES modules.
- `swarmint-client/` — browser copies of the JS client (`identity.mjs`,
  `wire.mjs`, `embedding.mjs`), with imports rewritten to `../vendor/...` (the
  canonical, npm-resolvable source lives in `js/swarmint-client/src/` and is
  parity-tested against Python in `tests/test_js_parity.py`).
- `vendor/` — the browser-ready ESM builds of `@noble/ed25519`, `@noble/hashes`
  (just the sha1 chain), and `@msgpack/msgpack`, committed directly so the page
  works with zero install. Regenerate by copying fresh dist files from
  `js/swarmint-client/node_modules/` after `npm install` there (strip `.d.ts`
  and `.map` files; fix the one self-referencing bare import in
  `@noble/hashes/utils.js` to `./crypto.js`).
- `genesis_embedding.json` — the frozen digits LDA projection, exported by
  `scripts/export_genesis_embedding.py`. Must match whatever the live swarm's
  genesis actually uses (task=digits, world_seed=0) — the shared-space
  invariant, now spanning languages.

## Serving it

**Must be served over real HTTP** (`file://` breaks ES module `fetch()` of
`genesis_embedding.json` and same-origin module resolution in most browsers).
Any static file server works, e.g. from the repo root:

```bash
python -m http.server 8000 --directory web
# then open http://localhost:8000/infer.html
```

The beacon field in the page defaults to `https://beacon.swarmint.org` — the
live genesis, once it runs the `web` extra (`pip install -e .[web]`, needs
`aiortc`) and its `/signal` endpoint is reachable.

## The one thing NOT YET live-browser-verified

Everything up through a real WebRTC handshake **signaled over the beacon's
actual HTTP endpoint**, answered by a **real SwarmNode holding real digits
knowledge**, is verified — see `swarmint/sim/run_webrtc_infer.py` (Phase 2) and
`swarmint/sim/run_webrtc_signaling.py` (Phase 3), both real `aiortc`
executions, not mocks. What hasn't been driven yet is *this exact page* from
inside an actual browser engine (Chrome/Firefox) — jsdom has no WebRTC support,
so that step needs a real browser session. Everything it depends on (crypto,
wire, embedding) is proven byte/value-parity with Python first.
