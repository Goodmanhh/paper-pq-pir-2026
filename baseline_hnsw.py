#!/usr/bin/env python3
"""Baseline: plaintext HNSW vector retrieval benchmark.

AIRS-generated experiment. Measures build time, query latency, QPS,
Recall@10 against brute-force ground truth, and index size.
"""
import json
import time
from pathlib import Path

import numpy as np
import hnswlib

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---------- data: synthetic embeddings (seeded, reproducible) ----------
def make_data(n=100_000, dim=128, nq=1_000, seed=42, clusters=200):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xb = centers[rng.integers(0, clusters, n)] + 0.15 * rng.normal(size=(n, dim))
    xq = centers[rng.integers(0, clusters, nq)] + 0.15 * rng.normal(size=(nq, dim))
    xb = xb.astype(np.float32); xq = xq.astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    return xb, xq

def brute_force_recall(xb, xq, k=10):
    """Ground truth by exact nearest neighbors (inner product)."""
    gt = []
    for q in xq:
        sims = xb @ q
        gt.append(np.argpartition(-sims, k)[:k])
    return np.array(gt)

def main():
    n, dim, nq = 100_000, 128, 1_000
    xb, xq = make_data(n=n, dim=dim, nq=nq)
    print(f"data: {n} base vectors, {nq} queries, dim={dim}")

    # ---------- HNSW ----------
    index = hnswlib.Index(space="ip", dim=dim)
    t0 = time.perf_counter()
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.add_items(xb)
    build_s = time.perf_counter() - t0
    index.set_ef(200)

    # single-query latency
    lat = []
    for q in xq:
        t0 = time.perf_counter()
        index.knn_query(q, k=10)
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat)

    # batch QPS
    t0 = time.perf_counter()
    index.knn_query(xq, k=10)
    batch_s = time.perf_counter() - t0

    # recall
    labels, _ = index.knn_query(xq, k=10)
    gt = brute_force_recall(xb, xq, k=10)
    recall = np.mean([len(set(labels[i]) & set(gt[i])) / 10 for i in range(nq)])

    index_size_mb = (n * dim * 4 + n * 16 * 2 * 4) / 1e6  # vectors + links approx

    result = {
        "route": "plaintext-hnsw",
        "n_base": n, "n_queries": nq, "dim": dim,
        "build_seconds": round(build_s, 3),
        "latency_ms_mean": round(float(lat.mean() * 1000), 4),
        "latency_ms_p95": round(float(np.percentile(lat, 95) * 1000), 4),
        "batch_qps": round(nq / batch_s, 1),
        "recall_at_10": round(float(recall), 4),
        "index_size_mb_approx": round(index_size_mb, 1),
        "online_communication": "query vector + k records (plaintext)",
        "client_storage": "none",
    }
    (RESULTS / "baseline_hnsw.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
