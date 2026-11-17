#!/usr/bin/env python3
"""Expanded experiments: latency scaling with database size (AIRS)."""
import json
import time
from pathlib import Path

import numpy as np
import hnswlib
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
R.mkdir(exist_ok=True)

def make_data(n, dim, nq, seed=42, clusters=200):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xb = centers[rng.integers(0, clusters, n)] + 0.15 * rng.normal(size=(n, dim))
    xq = centers[rng.integers(0, clusters, nq)] + 0.15 * rng.normal(size=(nq, dim))
    xb = xb.astype(np.float32); xq = xq.astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    return xb, xq

def hnsw_bench(xb, xq):
    n, dim = xb.shape
    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.add_items(xb)
    index.set_ef(200)
    lat = []
    for q in xq:
        t0 = time.perf_counter()
        index.knn_query(q, k=10)
        lat.append(time.perf_counter() - t0)
    return float(np.mean(lat) * 1000)

def ckks_bench(xb, xq, context):
    n = xb.shape[0]
    lat = []
    for q in xq:
        t0 = time.perf_counter()
        enc_q = ts.ckks_vector(context, q)
        scores_enc = [enc_q.dot(xb[j]) for j in range(n)]
        lat.append(time.perf_counter() - t0)
    return float(np.mean(lat))

def main():
    rows = []
    # HNSW at multiple sizes and dims
    for n, dim in [(10_000, 128), (50_000, 128), (100_000, 128), (100_000, 256)]:
        xb, xq = make_data(n, dim, 200)
        ms = hnsw_bench(xb, xq)
        rows.append({"route": "hnsw", "n": n, "dim": dim, "latency_ms": round(ms, 4)})
        print(rows[-1])

    # CKKS at small sizes
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    for n in [1_000, 2_500, 5_000]:
        xb, xq = make_data(n, 128, 5)
        s = ckks_bench(xb, xq, context)
        rows.append({"route": "ckks", "n": n, "dim": 128, "latency_s": round(s, 3)})
        print(rows[-1])

    (R / "scaling.json").write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
