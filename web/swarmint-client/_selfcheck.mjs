// Not shipped to the browser — a Node-side regression check that THESE web
// copies (vendor-relative imports, distinct from the npm-resolvable copies in
// js/swarmint-client/src/ that tests/test_js_parity.py exercises) still load
// and produce sane output. Run after touching web/vendor/ or web/swarmint-client/:
//   node web/swarmint-client/_selfcheck.mjs
import { readFileSync } from "fs";
import { Identity } from "./identity.mjs";
import { GenesisEmbedding } from "./embedding.mjs";

const id = await Identity.generate();
const body = new Uint8Array([1, 2, 3]);
const env = await id.signEnvelope(body, Date.now() / 1000);
if (env.length < 50) throw new Error("envelope suspiciously short: " + env.length);
console.log("signed envelope bytes:", env.length, "nodeId:", Buffer.from(id.nodeId).toString("hex").slice(0, 8));

const genesis = JSON.parse(readFileSync(new URL("../genesis_embedding.json", import.meta.url), "utf-8"));
const emb = GenesisEmbedding.fromJSON(genesis);
if (emb.inDim !== 64) throw new Error("expected in_dim=64, got " + emb.inDim);
const z = emb.embedRawDigit(new Float32Array(64).fill(1));
if (z.length !== emb.outDim) throw new Error("embed output dim mismatch");
console.log("embedded dim:", z.length, "sample:", Array.from(z.slice(0, 3)));
console.log("SELFCHECK_OK");
