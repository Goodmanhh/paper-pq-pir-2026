#!/usr/bin/env python3
"""CKKS route: encrypted inner-product similarity for vector retrieval.

AIRS-generated experiment. The query embedding is encrypted with CKKS;
the server computes inner products against plaintext database vectors
without seeing the query. Measures per-query latency, communication,
and Recall@10 against the plaintext HNSW baseline.
"""
import json
import time
from pathlib import Path

import numpy as np
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def make_data(n=10_000, dim=128, nq=50, seed=42, clusters=200):
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
    print(f"data: {n} base vectors, {nq} queries, dim={dim}")

    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    # encrypt query once per query; server computes inner products
    lat = []
    query_ct_bytes = []
    response_ct_bytes = []
    recalls = []

    for q in xq:
        t0 = time.perf_counter()
        enc_q = ts.ckks_vector(context, q)
        query_ct_bytes.append(len(enc_q.serialize()))
        # server side: encrypted inner product with every database vector
        scores_enc = [enc_q.dot(xb[j]) for j in range(n)]
        # response: encrypted scores (one scalar per record)
        response_ct_bytes.append(sum(len(s.serialize()) for s in scores_enc))
        scores = np.array([s.decrypt()[0] for s in scores_enc])
        lat.append(time.perf_counter() - t0)

        topk = np.argpartition(-scores, 10)[:10]
        gt = np.argpartition(-(xb @ q), 10)[:10]
        recalls.append(len(set(topk) & set(gt)) / 10)

    result = {
        "route": "ckks-inner-product",
        "n_base": n, "n_queries": nq, "dim": dim,
        "latency_s_mean": round(float(np.mean(lat)), 3),
        "latency_s_p95": round(float(np.percentile(lat, 95)), 3),
        "qps": round(float(nq / sum(lat)), 4),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "query_ciphertext_kb": round(float(np.mean(query_ct_bytes)) / 1024, 2),
        "response_ciphertext_mb": round(float(np.mean(response_ct_bytes)) / 1e6, 2),
        "client_storage": "CKKS context + keys",
    }
    (RESULTS / "ckks_inner_product.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
