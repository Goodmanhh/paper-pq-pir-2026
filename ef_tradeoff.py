#!/usr/bin/env python3
"""HNSW ef-recall tradeoff and QPS comparison (AIRS)."""
import json
import time
from pathlib import Path

import numpy as np
import hnswlib

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

def make_data(n=50_000, dim=128, nq=500, seed=42, clusters=200):
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
    gt = []
    for q in xq:
        sims = xb @ q
        gt.append(np.argpartition(-sims, k)[:k])
    return np.array(gt)

def main():
    xb, xq = make_data()
    nq = xq.shape[0]
    gt = brute_force_recall(xb, xq)
    rows = []
    for ef in [50, 100, 200, 400]:
        index = hnswlib.Index(space="ip", dim=xb.shape[1])
        index.init_index(max_elements=xb.shape[0], ef_construction=200, M=16)
        index.add_items(xb)
        index.set_ef(ef)
        lat = []
        for q in xq:
            t0 = time.perf_counter()
            index.knn_query(q, k=10)
            lat.append(time.perf_counter() - t0)
        labels, _ = index.knn_query(xq, k=10)
        recall = np.mean([len(set(labels[i]) & set(gt[i])) / 10 for i in range(nq)])
        rows.append({"ef": ef, "latency_ms": round(float(np.mean(lat) * 1000), 4), "recall_at_10": round(float(recall), 4)})
        print(rows[-1])
    (R / "ef_tradeoff.json").write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
