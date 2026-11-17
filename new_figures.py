#!/usr/bin/env python3
"""New-dimension figures: QPS comparison and ef-recall tradeoff (AIRS)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures"
FIG.mkdir(exist_ok=True)

# ---------- QPS comparison ----------
base = json.loads((R / "baseline_hnsw.json").read_text())
ckks = json.loads((R / "ckks_inner_product.json").read_text())
pir = json.loads((R / "lwe_pir.json").read_text())

fig, ax = plt.subplots(figsize=(7.2, 4.4))
routes = ["Plaintext HNSW", "CKKS inner product", "LWE-PIR (server)"]
qps = [base["batch_qps"], ckks["qps"], 1000 / pir["server_compute_ms_mean"]]
colors = ["#4C72B0", "#C44E52", "#55A868"]
bars = ax.bar(routes, qps, color=colors, width=0.55)
ax.set_yscale("log")
ax.set_ylabel("Queries per second (log scale)")
ax.set_title("Throughput: plaintext is six orders of magnitude faster than CKKS")
for b, v in zip(bars, qps):
    ax.text(b.get_x() + b.get_width()/2, v*1.2, f"{v:.3g}", ha="center", fontsize=10)
plt.tight_layout()
fig.savefig(FIG / "qps_comparison.png", dpi=200)
plt.close(fig)

# ---------- ef-recall tradeoff ----------
rows = json.loads((R / "ef_tradeoff.json").read_text())
ef = [r["ef"] for r in rows]
lat = [r["latency_ms"] for r in rows]
rec = [r["recall_at_10"] for r in rows]

fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
ax1.plot(ef, rec, "o-", color="#4C72B0", label="Recall@10")
ax1.set_xlabel("ef (search breadth)")
ax1.set_ylabel("Recall@10", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax1.set_ylim(0.94, 1.0)

ax2 = ax1.twinx()
ax2.plot(ef, lat, "s--", color="#C44E52", label="Latency (ms)")
ax2.set_ylabel("Latency (ms)", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")

ax1.set_title("HNSW: higher ef improves recall at the cost of latency")
fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
plt.tight_layout()
fig.savefig(FIG / "ef_tradeoff.png", dpi=200)
plt.close(fig)

print("new figures saved")
