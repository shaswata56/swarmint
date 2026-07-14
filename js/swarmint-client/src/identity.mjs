// Mirrors swarmint/network/identity.py: node_id = sha1(pubkey)[:20], and
// sign_envelope's exact signable-bytes construction, so a browser-signed
// envelope verifies under Python's verify_envelope() unmodified.
import * as ed from "@noble/ed25519";
import { sha1 } from "@noble/hashes/sha1";
import { signableBytes, packEnvelope } from "./wire.mjs";

export function deriveNodeId(pubkey) {
  return sha1(pubkey).slice(0, 20);
}

export function newNonce() {
  const n = new Uint8Array(16);
  crypto.getRandomValues(n);
  return n;
}

export class Identity {
  constructor(privateKey, publicKey, nodeId) {
    this.privateKey = privateKey;   // 32 bytes
    this.pubkey = publicKey;        // 32 bytes
    this.nodeId = nodeId;           // 20 bytes
  }

  static async generate() {
    const priv = ed.utils.randomPrivateKey();
    return Identity.fromPrivateKey(priv);
  }

  static async fromPrivateKey(priv) {
    const pub = await ed.getPublicKeyAsync(priv);
    return new Identity(priv, pub, deriveNodeId(pub));
  }

  async signEnvelope(body, ts, nonce = null) {
    nonce = nonce ?? newNonce();
    const signable = signableBytes(this.pubkey, this.nodeId, ts, nonce, body);
    const sig = await ed.signAsync(signable, this.privateKey);
    return packEnvelope(this.pubkey, this.nodeId, ts, nonce, sig, body);
  }
}

export async function verifyEnvelope(data, unpackEnvelopeFn) {
  const env = unpackEnvelopeFn(data);
  const { pk, s, ts, n, sig, b } = env;
  const expectedId = deriveNodeId(pk);
  if (!bytesEqual(expectedId, s)) {
    return { ok: false, reason: "sender id does not match H(pubkey) (impersonation)" };
  }
  const signable = signableBytes(pk, s, ts, n, b);
  const ok = await ed.verifyAsync(sig, signable, pk);
  if (!ok) return { ok: false, reason: "bad signature" };
  return { ok: true, sender: s, body: b };
}

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
