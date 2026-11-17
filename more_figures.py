#!/usr/bin/env python3
"""Additional figures: scaling curve and parameter study (AIRS)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures"
FIG.mkdir(exist_ok=True)

# ---------- scaling curve ----------
fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
hnsw_n = [10000, 50000, 100000]
hnsw_ms = [0.159, 0.1881, 0.1906]
ckks_n = [1000, 2500, 5000]
ckks_s = [4.383, 10.896, 22.057]

ax1.plot(hnsw_n, hnsw_ms, "o-", color="#4C72B0", label="Plaintext HNSW (ms)")
ax1.set_xlabel("Number of records")
ax1.set_ylabel("HNSW latency (ms)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax1.set_ylim(0, 0.6)

ax2 = ax1.twinx()
ax2.plot(ckks_n, ckks_s, "s--", color="#C44E52", label="CKKS (s)")
ax2.set_ylabel("CKKS latency (s)", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")
ax2.set_ylim(0, 26)

ax1.set_title("Latency scaling: HNSW is nearly flat, CKKS grows linearly")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.92))
plt.tight_layout()
fig.savefig(FIG / "scaling_curve.png", dpi=200)
plt.close(fig)

# ---------- parameter study ----------
deg = [4096, 8192, 16384]
lat = [2.036, 4.329, 15.234]
resp = [67.836, 235.072, 657.77]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = range(len(deg))
w = 0.35
ax.bar([i - w/2 for i in x], lat, w, label="Latency (s)", color="#4C72B0")
ax.set_ylabel("Latency (s)")
ax.set_xlabel("Polynomial modulus degree")
ax.set_xticks(list(x))
ax.set_xticklabels([str(d) for d in deg])
ax2 = ax.twinx()
ax2.bar([i + w/2 for i in x], resp, w, label="Response (MB)", color="#C44E52")
ax2.set_ylabel("Response size (MB)")
ax.set_title("Parameter study: smaller degree is faster but less secure")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.92))
plt.tight_layout()
fig.savefig(FIG / "parameter_study.png", dpi=200)
plt.close(fig)

print("additional figures saved")
