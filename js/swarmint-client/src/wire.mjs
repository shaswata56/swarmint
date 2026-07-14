// Mirrors swarmint/network/wire.py field-for-field (same short keys, same types)
// so a JS-built envelope/query is byte-parity with what Python would produce for
// the same inputs, and verifies under Python's verify_envelope() unmodified.
import { encode, decode } from "@msgpack/msgpack";

export const WIRE_VERSION = 1;

function pack(body) {
  return encode(body, { useBigInt64: false });
}
function unpack(data) {
  return decode(data);
}

// ---- envelope ----

export function signableBytes(pubkey, sender, ts, nonce, body) {
  return pack({ pk: pubkey, s: sender, ts, n: nonce, b: body });
}

export function packEnvelope(pubkey, sender, ts, nonce, sig, body) {
  return pack({ v: WIRE_VERSION, pk: pubkey, s: sender, ts, n: nonce, sig, b: body });
}

export function unpackEnvelope(data) {
  const d = unpack(data);
  if (d.v !== WIRE_VERSION) throw new Error(`unsupported wire version ${d.v}`);
  return d;
}

// ---- inference RPC (the two message kinds a browser client needs) ----

export function packInferenceQuery(qid, xFloat32, confidenceThreshold) {
  return pack({
    v: WIRE_VERSION, k: "query", id: qid,
    x: new Uint8Array(xFloat32.buffer, xFloat32.byteOffset, xFloat32.byteLength),
    t: confidenceThreshold,
  });
}

export function unpackInferenceResponse(data) {
  const body = unpack(data);
  if (body.k !== "response") throw new Error(`expected 'response', got '${body.k}'`);
  return { id: body.id, label: body.l, confidence: body.c };
}

export function bytesToFloat32(u8) {
  // u8 comes from msgpack as Uint8Array; must be 4-byte aligned to view in place.
  const buf = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
  return new Float32Array(buf);
}
