#!/usr/bin/env python3
"""CKKS with response packing: matrix-product instead of per-record dots.

AIRS-generated experiment. Packs the database into a plaintext matrix and
computes one encrypted matrix-vector product, returning a single packed
ciphertext of scores instead of one ciphertext per record.
"""
import json
import time
from pathlib import Path

import numpy as np
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

def make_data(n=5_000, dim=128, nq=20, seed=42, clusters=200):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xb = centers[rng.integers(0, clusters, n)] + 0.15 * rng.normal(size=(n, dim))
    xq = centers[rng.integers(0, clusters, nq)] + 0.15 * rng.normal(size=(nq, dim))
    xb = xb.astype(np.float32); xq = xq.astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    return xb, xq

def main():
    n, dim, nq = 5_000, 128, 20
    xb, xq = make_data(n=n, dim=dim, nq=nq)
    print(f"data: {n} records, {nq} queries, dim={dim}")

    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    # server pre-encrypts the packed database once (reused across queries)
    enc_db = ts.ckks_vector(context, xb)

    lat, resp_bytes, recalls = [], [], []
    for q in xq:
        t0 = time.perf_counter()
        # one packed matrix-vector product: D (Nxd) @ q (d) -> N scores in one ciphertext
        scores_enc = enc_db.enc_matmul_plain(q.tolist(), row_size=dim)
        lat.append(time.perf_counter() - t0)
        resp_bytes.append(len(scores_enc.serialize()))
        scores = np.array(scores_enc.decrypt())
        topk = np.argpartition(-scores, 10)[:10]
        gt = np.argpartition(-(xb @ q), 10)[:10]
        recalls.append(len(set(topk) & set(gt)) / 10)

    result = {
        "route": "ckks-packed-response",
        "n_base": n, "n_queries": nq, "dim": dim,
        "latency_s_mean": round(float(np.mean(lat)), 3),
        "latency_s_p95": round(float(np.percentile(lat, 95)), 3),
        "qps": round(float(nq / sum(lat)), 4),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "response_ciphertext_kb": round(float(np.mean(resp_bytes)) / 1024, 2),
        "response_ciphertext_mb": round(float(np.mean(resp_bytes)) / 1e6, 4),
        "client_storage": "CKKS context + keys",
    }
    (R / "ckks_packed.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
