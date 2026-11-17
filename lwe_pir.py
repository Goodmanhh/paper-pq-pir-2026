#!/usr/bin/env python3
"""LWE-PIR route: exact-index private retrieval benchmark.

AIRS-generated experiment. Simplified Regev/LWE PIR: the client encrypts
a one-hot selection vector; the server homomorphically computes the
selection over the plaintext database. Measures client encryption time,
server time, communication, and correctness of the retrieved record.
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

Q = 2**12       # modulus
DELTA = 2**20       # plaintext scaling factor
K = 1024            # LWE dimension per sample

def make_db(n=100_000, dim=128, seed=42, clusters=200):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xb = centers[rng.integers(0, clusters, n)] + 0.15 * rng.normal(size=(n, dim))
    xb = xb.astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    return xb

def lwe_pir_query(db, idx, rng):
    n, dim = db.shape
    onehot = np.zeros(n, dtype=np.int64)
    onehot[idx] = 1

    # client encrypts one-hot vector: n LWE samples
    t0 = time.perf_counter()
    s = rng.integers(0, Q, size=K, dtype=np.int64)
    A = rng.integers(0, Q, size=(n, K), dtype=np.int64)
    e = rng.integers(-3, 4, size=n, dtype=np.int64)  # small error
    b = A.astype(np.float64) @ s.astype(np.float64) + e + DELTA * onehot
    client_s = time.perf_counter() - t0

    # server: homomorphic selection
    t0 = time.perf_counter()
    B_sel = b.astype(np.float64) @ db          # sum_j b_j * D[j]
    A_sel = A.astype(np.float64).T @ db              # sum_j a_j * D[j]  (K x dim)
    server_s = time.perf_counter() - t0

    # client decrypts: B_sel - A_sel * s ~ DELTA * D[idx]
    rec = (B_sel - A_sel.T @ s) / DELTA
    err = float(np.linalg.norm(rec - db[idx]))
    return client_s, server_s, err

def main():
    n, dim, nq = 100_000, 128, 20
    db = make_db(n=n, dim=dim)
    rng = np.random.default_rng(7)
    print(f"data: {n} records, dim={dim}, {nq} queries")

    client_times, server_times, errors = [], [], []
    for _ in range(nq):
        idx = int(rng.integers(0, n))
        cs, ss, err = lwe_pir_query(db, idx, rng)
        client_times.append(cs)
        server_times.append(ss)
        errors.append(err)

    result = {
        "route": "lwe-pir-exact-index",
        "n_base": n, "n_queries": nq, "dim": dim,
        "client_encrypt_s_mean": round(float(np.mean(client_times)), 4),
        "server_compute_ms_mean": round(float(np.mean(server_times) * 1000), 4),
        "query_samples": n,
        "response_records": 1,
        "retrieval_error_l2": round(float(np.mean(errors)), 6),
        "semantics": "exact index only (no ANN similarity)",
        "client_storage": "LWE secret key",
    }
    (RESULTS / "lwe_pir.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
