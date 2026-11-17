#!/usr/bin/env python3
"""Real-data experiment with TF-IDF embeddings (no model download needed)."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

def main():
    data = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    texts = data.data[:2000]
    queries = data.data[2000:2020]

    vec = TfidfVectorizer(max_features=512, stop_words="english")
    xb = vec.fit_transform(texts).toarray().astype(np.float32)
    xq = vec.transform(queries).toarray().astype(np.float32)
    xb /= (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-9)
    xq /= (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-9)
    print("TF-IDF embeddings:", xb.shape, xq.shape)

    # HNSW
    import hnswlib
    index = hnswlib.Index(space="ip", dim=xb.shape[1])
    index.init_index(max_elements=xb.shape[0], ef_construction=200, M=16)
    index.add_items(xb)
    index.set_ef(200)
    lat = []
    for q in xq:
        t0 = time.perf_counter()
        index.knn_query(q, k=10)
        lat.append(time.perf_counter() - t0)
    labels, _ = index.knn_query(xq, k=10)
    gt = [np.argpartition(-(xb @ q), 10)[:10] for q in xq]
    recall = np.mean([len(set(labels[i]) & set(gt[i])) / 10 for i in range(len(xq))])

    result_hnsw = {
        "route": "hnsw-real-20ng-tfidf",
        "n_base": int(xb.shape[0]), "n_queries": int(xq.shape[0]),
        "dim": int(xb.shape[1]),
        "latency_ms_mean": round(float(np.mean(lat) * 1000), 4),
        "recall_at_10": round(float(recall), 4),
    }
    (R / "real_hnsw_tfidf.json").write_text(json.dumps(result_hnsw, indent=2))
    print("HNSW:", result_hnsw)

    # CKKS
    import tenseal as ts
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    xb_small = xb[:500]
    lat2, resp, recalls = [], [], []
    for q in xq[:5]:
        t0 = time.perf_counter()
        enc_q = ts.ckks_vector(context, q)
        scores_enc = [enc_q.dot(xb_small[j]) for j in range(xb_small.shape[0])]
        lat2.append(time.perf_counter() - t0)
        resp.append(sum(len(s.serialize()) for s in scores_enc))
        scores = np.array([s.decrypt()[0] for s in scores_enc])
        topk = np.argpartition(-scores, 10)[:10]
        gt2 = np.argpartition(-(xb_small @ q), 10)[:10]
        recalls.append(len(set(topk) & set(gt2)) / 10)

    result_ckks = {
        "route": "ckks-real-20ng-tfidf",
        "n_base": int(xb_small.shape[0]), "n_queries": 5,
        "dim": int(xb.shape[1]),
        "latency_s_mean": round(float(np.mean(lat2)), 3),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "response_mb_mean": round(float(np.mean(resp)) / 1e6, 3),
    }
    (R / "real_ckks_tfidf.json").write_text(json.dumps(result_ckks, indent=2))
    print("CKKS:", result_ckks)

if __name__ == "__main__":
    main()
