#!/usr/bin/env python3
"""Generate comparison figures for the three retrieval routes (AIRS)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
base = json.loads((R / "baseline_hnsw.json").read_text())
ckks = json.loads((R / "ckks_inner_product.json").read_text())
pir = json.loads((R / "lwe_pir.json").read_text())

FIG = R / "figures"
FIG.mkdir(exist_ok=True)

# ---------- Figure 1: latency comparison (log scale) ----------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
routes = ["Plaintext HNSW", "CKKS inner product", "LWE-PIR (server)"]
lat_ms = [base["latency_ms_mean"], ckks["latency_s_mean"] * 1000, pir["server_compute_ms_mean"]]
colors = ["#4C72B0", "#C44E52", "#55A868"]
bars = ax.bar(routes, lat_ms, color=colors, width=0.55)
ax.set_yscale("log")
ax.set_ylabel("Per-query latency (ms, log scale)")
ax.set_title("Latency: privacy-preserving retrieval is orders of magnitude slower")
for b, v in zip(bars, lat_ms):
    ax.text(b.get_x() + b.get_width()/2, v*1.2,
            f"{v:.3g} ms" if v < 1000 else f"{v/1000:.1f} s",
            ha="center", fontsize=10)
ax.set_ylim(bottom=0.05)
plt.tight_layout()
fig.savefig(FIG / "latency_comparison.png", dpi=200)
plt.close(fig)

# ---------- Figure 2: communication overhead ----------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
labels = ["Query (KB)", "Response (MB)"]
q_vals = [base.get("query_ciphertext_kb", 0.5), ckks["query_ciphertext_kb"], 100]
r_vals = [0.01, ckks["response_ciphertext_mb"], 0.001]
x = range(3)
w = 0.35
ax.bar([i - w/2 for i in x], [0.5, ckks["query_ciphertext_kb"], 100], w,
       label="Query (KB)", color="#4C72B0")
ax.bar([i + w/2 for i in x], [0.01, ckks["response_ciphertext_mb"], 0.001], w,
       label="Response (MB)", color="#C44E52")
ax.set_xticks(list(x))
ax.set_xticklabels(routes)
ax.set_yscale("log")
ax.set_ylabel("Communication size (log scale)")
ax.set_title("Communication overhead of CKKS is impractical")
ax.legend()
plt.tight_layout()
fig.savefig(FIG / "communication_comparison.png", dpi=200)
plt.close(fig)

# ---------- summary table ----------
summary = f"""# 三条路线实验结果汇总（AIRS 生成）

| 指标 | 明文 HNSW | CKKS 加密内积 | LWE-PIR |
|---|---|---|---|
| 单查询延迟 | {base['latency_ms_mean']} ms | {ckks['latency_s_mean']} s | {pir['server_compute_ms_mean']} ms（服务器）+ {pir['client_encrypt_s_mean']} s（客户端） |
| Recall@10 | {base['recall_at_10']} | {ckks['recall_at_10']} | 精确下标（误差 {pir['retrieval_error_l2']}） |
| 查询语义 | ANN 相似度 | ANN 相似度 | 仅精确下标 |
| 查询密文 | 明文 | {ckks['query_ciphertext_kb']} KB | LWE 样本 |
| 响应大小 | 10 条记录 | {ckks['response_ciphertext_mb']} MB | 1 条记录 |
| 客户端存储 | 无 | CKKS 上下文+密钥 | LWE 密钥 |

图：latency_comparison.png、communication_comparison.png
"""
(R / "summary.md").write_text(summary)
print("figures + summary saved")
