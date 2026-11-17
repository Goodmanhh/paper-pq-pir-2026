#!/usr/bin/env python3
"""CKKS parameter study: poly_modulus_degree vs latency and response size."""
import json
import time
from pathlib import Path

import numpy as np
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

def make_data(n=1_000, dim=128, nq=10, seed=42, clusters=200):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xb = centers[rng.integers(0, clusters, n)] + 0.15 * rng.normal(size=(n, dim))
    xq = centers[rng.integers(0, clusters, nq)] + 0.15 * rng.normal(size=(nq, dim))
    xb = xb.astype(np.float32); xq = xq.astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    return xb, xq

def bench(xb, xq, degree):
    bits = {4096: [30, 20, 20, 30], 8192: [60, 40, 40, 60], 16384: [60, 40, 40, 60, 60]}[degree]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=degree,
                         coeff_mod_bit_sizes=bits)
    context.generate_galois_keys()
    context.global_scale = 2 ** (20 if degree == 4096 else 40)
    lat, resp = [], []
    for q in xq:
        t0 = time.perf_counter()
        enc_q = ts.ckks_vector(context, q)
        scores_enc = [enc_q.dot(xb[j]) for j in range(xb.shape[0])]
        lat.append(time.perf_counter() - t0)
        resp.append(sum(len(s.serialize()) for s in scores_enc))
    return float(np.mean(lat)), float(np.mean(resp)) / 1e6

def main():
    xb, xq = make_data()
    rows = []
    for degree in [4096, 8192, 16384]:
        s, mb = bench(xb, xq, degree)
        rows.append({"poly_modulus_degree": degree, "latency_s": round(s, 3), "response_mb": round(mb, 3)})
        print(rows[-1])
    (R / "ckks_parameters.json").write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
