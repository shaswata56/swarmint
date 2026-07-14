// Mirrors swarmint/core/embedding.py's Embedding.transform + the raw-pixel
// normalize in swarmint/sim/real_data.py, so a browser-embedded digit vector is
// numerically identical (within float32 tolerance) to what Python produces for
// the same 64-pixel input. Loads the FROZEN genesis projection exported by
// scripts/export_genesis_embedding.py — never fits its own (shared-space invariant).

function l2NormalizeRow(v, eps) {
  let ss = 0;
  for (let i = 0; i < v.length; i++) ss += v[i] * v[i];
  const norm = Math.sqrt(ss);
  const denom = eps !== undefined ? norm + eps : (norm === 0 ? 1 : norm);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / denom;
  return out;
}

// swarmint.sim.real_data.normalize: x / (||x|| + 1e-9) — the RAW pixel normalize.
export function normalizeRawPixels(x) {
  return l2NormalizeRow(x, 1e-9);
}

export class GenesisEmbedding {
  constructor(mean, W, inDim, outDim) {
    this.mean = mean;   // Float32Array(inDim)
    this.W = W;         // Float32Array(inDim * outDim), row-major (in, out)
    this.inDim = inDim;
    this.outDim = outDim;
  }

  static fromJSON(obj) {
    const mean = Float32Array.from(obj.mean);
    const W = Float32Array.from(obj.W);
    return new GenesisEmbedding(mean, W, obj.in_dim, obj.out_dim);
  }

  static async fetch(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch genesis embedding failed: ${resp.status}`);
    return GenesisEmbedding.fromJSON(await resp.json());
  }

  // z = normalize((x - mean) @ W) — exactly Embedding.transform for a single row.
  transform(x) {
    if (x.length !== this.inDim) {
      throw new Error(`expected ${this.inDim}-dim input, got ${x.length}`);
    }
    const centered = new Float32Array(this.inDim);
    for (let i = 0; i < this.inDim; i++) centered[i] = x[i] - this.mean[i];
    const z = new Float32Array(this.outDim);
    for (let j = 0; j < this.outDim; j++) {
      let s = 0;
      for (let i = 0; i < this.inDim; i++) s += centered[i] * this.W[i * this.outDim + j];
      z[j] = s;
    }
    return l2NormalizeRow(z); // embedding.Embedding._normalize semantics (zero-guard, no epsilon)
  }

  // The full pipeline infer_query.py's --image path uses: raw pixels ->
  // normalizeRawPixels -> transform. One call for the browser client.
  embedRawDigit(rawPixels64) {
    return this.transform(normalizeRawPixels(rawPixels64));
  }
}
