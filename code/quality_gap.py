"""Part I, Finding 1 — "Counted but not known".
Computes per-tract building provenance/attribute measures and POI provenance measures from the
challenge bundle only, stratifies them by the scorecard strata, and writes results/quality_gap_table.csv
plus three figures. Usage:  python quality_gap.py <bundle_dir> <out_dir>
"""
import sys, numpy as np, pandas as pd
from math import erf, sqrt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B, OUT = sys.argv[1], sys.argv[2]
R = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx"]
TR = pd.concat([pd.read_parquet(f"{B}/{r}-tracts.parquet").assign(region=r) for r in R]).drop_duplicates("GEOID").set_index("GEOID")
ST = pd.concat([pd.read_parquet(f"{B}/{r}-strata.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID").reindex(TR.index)
AC = pd.concat([pd.read_parquet(f"{B}/{r}-acs.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID").reindex(TR.index)
b = pd.concat([pd.read_parquet(f"{B}/{r}-bldg-tract.parquet") for r in R])
g = b.groupby("GEOID")
D = pd.DataFrame({"n": g.n.sum(),
                  "ml": b[b.src == "Microsoft ML Buildings"].groupby("GEOID").n.sum(),
                  "osm": b[b.src == "OpenStreetMap"].groupby("GEOID").n.sum(),
                  "attr": b[b.subtype.notna()].groupby("GEOID").n.sum(),
                  "multi": g.n_multisrc.sum()}).reindex(TR.index).fillna(0)
den = D.n.replace(0, np.nan)
D["ml_share"], D["attr_share"], D["multi_share"], D["osm_share"] = D.ml/den, D.attr/den, D.multi/den, D.osm/den
D["count_cov"] = D.n / ST.usfs_BuildingCount_sum.replace(0, np.nan)
D["pop"] = TR.pop_total; D["region"] = TR.region
D["rural"] = ST.ur_class == "Rural"; D["tribal"] = ST.tribal_statistical > 0
D["wf"] = ST.usfs_Exposure_mean >= ST.usfs_Exposure_mean.quantile(.9)
D["burned"] = ST.usgs_wildfire_ever == 1; D["svi"] = ST.svi_overall >= 0.75
D["cvi"] = ST.cvi_overall >= ST.cvi_overall.quantile(.9)
D["heat"] = ST.epht_heat_days_summer >= ST.epht_heat_days_summer.quantile(.9)
P = pd.concat([pd.read_parquet(f"{B}/{r}-pois-tract.parquet") for r in R]); gp = P.groupby("GEOID")
D["npoi"] = gp.size().reindex(D.index).fillna(0)
D["conf"] = gp.confidence.mean().reindex(D.index)
D["lowconf"] = gp.confidence.apply(lambda s: (s < 0.5).mean()).reindex(D.index)
D["meta_share"] = gp.src.apply(lambda s: (s == "meta").mean()).reindex(D.index)

def tt(x, y):
    x, y = x.dropna().astype(float), y.dropna().astype(float)
    mx, my, vx, vy = x.mean(), y.mean(), x.var(ddof=1), y.var(ddof=1)
    t = (mx-my)/np.sqrt(vx/len(x)+vy/len(y)); p = 2*(1-0.5*(1+erf(abs(t)/sqrt(2))))
    return mx, my, t, p, (mx-my)/np.sqrt((vx+vy)/2)
rows = []
for col in ["ml_share", "attr_share", "multi_share", "conf", "lowconf", "meta_share"]:
    for grp in ["rural", "tribal", "wf", "burned", "svi", "cvi", "heat"]:
        mx, my, t, p, d = tt(D.loc[D[grp] == True, col], D.loc[D[grp] == False, col])
        rows.append(dict(metric=col, group=grp, n_in=int((D[grp] == True).sum()), mean_in=mx, mean_out=my, ratio=mx/my, t=t, p=p, cohen_d=d))
pd.DataFrame(rows).to_csv(f"{OUT}/quality_gap_table.csv", index=False)
m = D.ml_share > 0.9
print(f"n tracts {len(D)}; pop {D['pop'].sum():,.0f}")
print(f"ML>90%: {m.sum()} tracts, pop {D['pop'][m].sum():,.0f}, median count_cov {D.count_cov[m].median():.3f} vs {D.count_cov[~m].median():.3f}, median attr {D.attr_share[m].median():.3f}")
nc = D.osm_share < 0.05
print(f"<5% OSM: {nc.sum()} tracts ({100*nc.mean():.1f}%), pop {D['pop'][nc].sum():,.0f}")
# figures
cx = np.array([(x["xmin"]+x["xmax"])/2 for x in TR.bbox]); cy = np.array([(x["ymin"]+x["ymax"])/2 for x in TR.bbox])
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
NAMES = {"maricopa-az": "Maricopa, AZ", "northern-ca": "Northern California", "eastern-ok": "Eastern Oklahoma", "south-central-tx": "South-Central Texas"}
for ax, r in zip(axs.ravel(), R):
    mm = (D.region == r).values
    sc = ax.scatter(cx[mm], cy[mm], c=D.ml_share.values[mm], cmap="RdYlBu_r", vmin=0, vmax=1, s=6, linewidths=0)
    ax.set_title(f"{NAMES[r]} — mean ML-only share {np.nanmean(D.ml_share.values[mm]):.2f}", fontsize=10)
    ax.set_aspect(1/np.cos(np.deg2rad(np.nanmean(cy[mm])))); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=axs, shrink=0.6, label="Share of Overture buildings that are Microsoft ML footprints (no attributes)")
fig.suptitle("Where open building coverage is machine-only (tract bbox centres)", fontsize=12)
fig.savefig(f"{OUT}/fig_ml_share_map.png", dpi=130, bbox_inches="tight"); plt.close()
grp = ["rural", "tribal", "wf", "burned", "cvi", "svi"]
fig, ax = plt.subplots(figsize=(9, 4.2)); x = np.arange(len(grp)); w = 0.38
ax.bar(x-w/2, [D.loc[D[k], "ml_share"].mean() for k in grp], w, label="in group", color="#c0392b")
ax.bar(x+w/2, [D.loc[~D[k], "ml_share"].mean() for k in grp], w, label="rest", color="#7f8c8d")
lab = ["Rural", "Tribal stat. area", "Wildfire exposure top 10%", "Ever burned (USGS)", "CVI top 10%", "SVI ≥ 0.75"]
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=15, ha="right", fontsize=9); ax.set_ylim(0, 1); ax.legend(); ax.set_ylabel("Mean share of ML-only buildings")
for i, k in enumerate(grp):
    ax.text(i-w/2, D.loc[D[k], "ml_share"].mean()+0.02, f"{D.loc[D[k], 'ml_share'].mean():.2f}", ha="center", fontsize=8)
    ax.text(i+w/2, D.loc[~D[k], "ml_share"].mean()+0.02, f"{D.loc[~D[k], 'ml_share'].mean():.2f}", ha="center", fontsize=8)
ax.set_title(f"Machine-only building coverage by scorecard stratum (4 scored regions, n = {len(D):,} tracts)", fontsize=10)
fig.savefig(f"{OUT}/fig_ml_share_groups.png", dpi=130, bbox_inches="tight"); plt.close()
ok = D.n > 50
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(D.attr_share[ok], D.count_cov.clip(0, 2)[ok], s=4, alpha=0.3, c=np.where(D.rural[ok], "#c0392b", "#2c3e50"), linewidths=0)
ax.axhline(1, color="k", lw=0.6, ls="--"); ax.set_xlabel("Share of buildings with any attribute (type/class)"); ax.set_ylabel("Count coverage: Overture / USFS building count"); ax.set_ylim(0, 2)
ax.set_title("Count-complete but content-empty (red = rural tracts)", fontsize=10)
fig.savefig(f"{OUT}/fig_count_vs_content.png", dpi=130, bbox_inches="tight"); plt.close()
D.to_parquet(f"{OUT}/quality_gap_tracts.parquet")
