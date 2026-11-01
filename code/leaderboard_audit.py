"""Section 9.3: public vs private audit of all scored submissions.
Usage: python leaderboard_audit.py results/all_submissions_public_private.csv figs/fig_public_private.png"""
import sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = pd.read_csv(sys.argv[1]); L = L.rename(columns={"public_rmse": "public", "private_rmse": "private"})
print("n", len(L), "corr", L[["public", "private"]].corr().iloc[0, 1].round(4), "median |diff|", (L.private - L.public).abs().median().round(5))
fig, ax = plt.subplots(figsize=(5.2, 5)); lo, hi = 0.063, 0.082
ax.scatter(L.public, L.private, s=14, c="#2c3e50"); ax.plot([lo, hi], [lo, hi], "k--", lw=0.7); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Public RMSE (≈30% of scored tracts)"); ax.set_ylabel("Private RMSE (≈70%)"); ax.set_title("All scored submissions: public vs private", fontsize=10)
b = L.loc[L.private.idxmin()]; ax.annotate("t24 (final)", (b.public, b.private), xytext=(0.066, 0.0645), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
fig.savefig(sys.argv[2], dpi=130, bbox_inches="tight")
