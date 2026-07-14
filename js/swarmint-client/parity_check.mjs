// Invoked by tests/test_js_parity.py with a JSON "job" on argv[1] (path to a
// job file) and writes a JSON "result" to argv[2]. Kept as a thin driver so the
// actual library code (identity.mjs/wire.mjs/embedding.mjs) is exactly what a
// browser would import — no test-only branching in the client itself.
import { readFileSync, writeFileSync } from "fs";
import { Identity, deriveNodeId, verifyEnvelope } from "./src/identity.mjs";
import { packInferenceQuery, unpackEnvelope, unpackInferenceResponse,
        packCorrectionClaim, unpackCorrectionAck } from "./src/wire.mjs";
import { GenesisEmbedding, normalizeRawPixels } from "./src/embedding.mjs";

function hex(u8) { return Buffer.from(u8).toString("hex"); }
function fromHex(h) { return new Uint8Array(Buffer.from(h, "hex")); }

async function main() {
  const job = JSON.parse(readFileSync(process.argv[2], "utf-8"));
  const out = {};

  if (job.sign) {
    const priv = fromHex(job.sign.privateKeyHex);
    const id = await Identity.fromPrivateKey(priv);
    const body = fromHex(job.sign.bodyHex);
    const nonce = fromHex(job.sign.nonceHex);
    const envelope = await id.signEnvelope(body, job.sign.ts, nonce);
    out.sign = {
      pubkeyHex: hex(id.pubkey), nodeIdHex: hex(id.nodeId), envelopeHex: hex(envelope),
    };
  }

  if (job.verify) {
    const envelope = fromHex(job.verify.envelopeHex);
    const res = await verifyEnvelope(envelope, unpackEnvelope);
    out.verify = { ok: res.ok, reason: res.reason || null,
                  senderHex: res.ok ? hex(res.sender) : null };
  }

  if (job.query) {
    const xBytes = fromHex(job.query.xFloat32Hex);
    const x = new Float32Array(xBytes.buffer, xBytes.byteOffset, xBytes.byteLength / 4);
    const qid = fromHex(job.query.qidHex);
    const packed = packInferenceQuery(qid, x, job.query.confidenceThreshold);
    out.query = { packedHex: hex(packed) };
  }

  if (job.unpackResponse) {
    const data = fromHex(job.unpackResponse.dataHex);
    const r = unpackInferenceResponse(data);
    out.unpackResponse = { idHex: hex(r.id), label: r.label, confidence: r.confidence };
  }

  if (job.verifyThenUnpackResponse) {
    // Mirrors infer.html's channel.onmessage EXACTLY: verify the OUTER signed
    // envelope, then unpack the response from the VERIFIED INNER body (res.body)
    // -- not from the raw envelope bytes. A prior bug passed the raw envelope to
    // unpackInferenceResponse instead of res.body, which has no "k" field (only
    // the inner body does) -> "expected 'response', got 'undefined'" at runtime.
    // This job guards that exact pattern end-to-end.
    const envelope = fromHex(job.verifyThenUnpackResponse.envelopeHex);
    const res = await verifyEnvelope(envelope, unpackEnvelope);
    if (!res.ok) {
      out.verifyThenUnpackResponse = { ok: false, reason: res.reason };
    } else {
      const r = unpackInferenceResponse(res.body);
      out.verifyThenUnpackResponse = { ok: true, idHex: hex(r.id), label: r.label, confidence: r.confidence };
    }
  }

  if (job.correction) {
    const xBytes = fromHex(job.correction.xFloat32Hex);
    const x = new Float32Array(xBytes.buffer, xBytes.byteOffset, xBytes.byteLength / 4);
    const qid = fromHex(job.correction.qidHex);
    const packed = packCorrectionClaim(qid, x, job.correction.label);
    out.correction = { packedHex: hex(packed) };
  }

  if (job.unpackCorrectionAck) {
    const data = fromHex(job.unpackCorrectionAck.dataHex);
    const a = unpackCorrectionAck(data);
    out.unpackCorrectionAck = { idHex: hex(a.id), promoted: a.promoted, corroborated: a.corroborated };
  }

  if (job.embed) {
    const emb = GenesisEmbedding.fromJSON(job.embed.embeddingJson);
    const raw = Float32Array.from(job.embed.rawPixels);
    const z = emb.embedRawDigit(raw);
    out.embed = { z: Array.from(z) };
  }

  if (job.deriveNodeId) {
    const pub = fromHex(job.deriveNodeId.pubkeyHex);
    out.deriveNodeId = { nodeIdHex: hex(deriveNodeId(pub)) };
  }

  writeFileSync(process.argv[3], JSON.stringify(out));
}

main().catch((e) => { console.error(e.stack || e); process.exit(1); });
