"""Part I, Finding 1 ("Counted but not known") and its companion numbers.
Reads only the challenge bundle (+ the shipped final submission for the calibrated-gap table) and writes
results/quality_gap_table.csv, results/quality_gap_regions.csv, results/quality_gap_numbers.json and three figures.
Usage: python quality_gap.py <bundle_dir> <out_dir> [<t24_csv>]
Definitions: ML-only share = Microsoft ML footprints / all buildings; attribute share = buildings with
non-null subtype / all; multi-source share = n_multisrc / n; Cohen's d uses the pooled SD
sqrt(((n1-1)s1^2+(n2-1)s2^2)/(n1+n2-2)); Welch t for unequal variances; tract means unless stated.
"""
import sys, json, numpy as np, pandas as pd
from math import erf, sqrt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B, OUT = sys.argv[1], sys.argv[2]; T24 = sys.argv[3] if len(sys.argv) > 3 else None
R = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx"]
NAMES = {"maricopa-az": "Maricopa, AZ", "northern-ca": "Northern California", "eastern-ok": "Eastern Oklahoma", "south-central-tx": "South-Central Texas"}
TR = pd.concat([pd.read_parquet(f"{B}/{r}-tracts.parquet").assign(region=r) for r in R]).drop_duplicates("GEOID").set_index("GEOID")
ST = pd.concat([pd.read_parquet(f"{B}/{r}-strata.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID").reindex(TR.index)
AC = pd.concat([pd.read_parquet(f"{B}/{r}-acs.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID").reindex(TR.index)
b = pd.concat([pd.read_parquet(f"{B}/{r}-bldg-tract.parquet") for r in R]); g = b.groupby("GEOID")
D = pd.DataFrame({"n": g.n.sum(), "ml": b[b.src == "Microsoft ML Buildings"].groupby("GEOID").n.sum(),
                  "osm": b[b.src == "OpenStreetMap"].groupby("GEOID").n.sum(),
                  "attr": b[b.subtype.notna()].groupby("GEOID").n.sum(), "multi": g.n_multisrc.sum()}).reindex(TR.index).fillna(0)
den = D.n.replace(0, np.nan)
D["ml_share"], D["attr_share"], D["multi_share"], D["osm_share"] = D.ml / den, D.attr / den, D.multi / den, D.osm / den
D["count_cov"] = D.n / ST.usfs_BuildingCount_sum.replace(0, np.nan)
D["pop"] = TR.pop_total; D["region"] = TR.region; D["hu"] = AC.housing_units
D["rural"] = ST.ur_class == "Rural"; D["tribal"] = ST.tribal_statistical > 0
D["wf"] = ST.usfs_Exposure_mean >= ST.usfs_Exposure_mean.quantile(.9); D["burned"] = ST.usgs_wildfire_ever == 1
D["svi"] = ST.svi_overall >= 0.75; D["cvi"] = ST.cvi_overall >= ST.cvi_overall.quantile(.9)
D["heat"] = ST.epht_heat_days_summer >= ST.epht_heat_days_summer.quantile(.9)
P = pd.concat([pd.read_parquet(f"{B}/{r}-pois-tract.parquet") for r in R]); gp = P.groupby("GEOID")
D["npoi"] = gp.size().reindex(D.index).fillna(0); D["conf"] = gp.confidence.mean().reindex(D.index)
D["lowconf"] = gp.confidence.apply(lambda s: (s < 0.5).mean()).reindex(D.index)
D["meta_share"] = gp.src.apply(lambda s: (s == "meta").mean()).reindex(D.index)
FIRE = ["fire_department", "fire_protection_service"]
D["has_fire"] = P[P.cat.isin(FIRE)].groupby("GEOID").size().reindex(D.index).fillna(0) > 0

def tt(x, y):
    x, y = x.dropna().astype(float), y.dropna().astype(float); n1, n2 = len(x), len(y)
    mx, my, vx, vy = x.mean(), y.mean(), x.var(ddof=1), y.var(ddof=1)
    t = (mx - my) / np.sqrt(vx / n1 + vy / n2); p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    sp = np.sqrt(((n1 - 1) * vx + (n2 - 1) * vy) / (n1 + n2 - 2)); return mx, my, t, p, (mx - my) / sp
GROUPS = ["rural", "tribal", "wf", "burned", "cvi", "heat", "svi"]
rows = []
for col in ["ml_share", "attr_share", "multi_share", "conf", "lowconf", "meta_share"]:
    for grp in GROUPS:
        mx, my, t, p, d = tt(D.loc[D[grp] == True, col], D.loc[D[grp] == False, col])
        rows.append(dict(metric=col, group=grp, n_in=int((D[grp] == True).sum()), mean_in=mx, mean_out=my, ratio=mx / my, t=t, p=p, cohen_d=d))
TAB = pd.DataFrame(rows); TAB.to_csv(f"{OUT}/quality_gap_table.csv", index=False)
# region x stratum table (tract means + population)
rr = []
for r in R:
    for grp in ["rural", "tribal", "wf"]:
        m = (D.region == r) & D[grp]
        if m.sum() > 20: rr.append(dict(region=NAMES[r], stratum=grp, tracts=int(m.sum()), ml_share=D.ml_share[m].mean(), attr_share=D.attr_share[m].mean(), multi_share=D.multi_share[m].mean(), population=int(D["pop"][m].sum())))
for name, m in [("rural AND wildfire top 10%", D.rural & D.wf), ("tribal AND rural", D.tribal & D.rural)]:
    rr.append(dict(region="all four", stratum=name, tracts=int(m.sum()), ml_share=D.ml_share[m].mean(), attr_share=D.attr_share[m].mean(), multi_share=D.multi_share[m].mean(), population=int(D["pop"][m].sum())))
pd.DataFrame(rr).to_csv(f"{OUT}/quality_gap_regions.csv", index=False)
N = {}
N["tracts"] = int(len(D)); N["population"] = int(D["pop"].sum())
N["region_ml_share_mean"] = D.groupby("region").ml_share.mean().round(3).to_dict()
N["ms_null_subtype_share"] = float(1 - b[b.src == "Microsoft ML Buildings"].subtype.notna().mul(b.n[b.src == "Microsoft ML Buildings"]).sum() / b.n[b.src == "Microsoft ML Buildings"].sum())
for src in ["Microsoft ML Buildings", "OpenStreetMap"]:
    N[f"multisrc_share_{src}"] = float(b.n_multisrc[b.src == src].sum() / b.n[b.src == src].sum())
osm = b[b.src == "OpenStreetMap"]; N["osm_subtype_rate"] = float(osm.n[osm.subtype.notna()].sum() / osm.n.sum())
N["src_counts_buildings"] = b.groupby("src").n.sum().astype(int).to_dict()
m = D.ml_share > 0.9
N["ml90"] = dict(tracts=int(m.sum()), population=int(D["pop"][m].sum()), median_count_cov=float(D.count_cov[m].median()), median_count_cov_rest=float(D.count_cov[~m].median()),
                 median_attr=float(D.attr_share[m].median()), tract_share_rural=float(D.rural[m].mean()), tract_share_tribal=float(D.tribal[m].mean()), tract_share_wf=float(D.wf[m].mean()),
                 pop_share_rural=float(D["pop"][m & D.rural].sum() / D["pop"][m].sum()), pop_share_tribal=float(D["pop"][m & D.tribal].sum() / D["pop"][m].sum()), pop_share_wf=float(D["pop"][m & D.wf].sum() / D["pop"][m].sum()))
nc = D.osm_share < 0.05
N["osm_lt5"] = dict(tracts=int(nc.sum()), share=float(nc.mean()), population=int(D["pop"][nc].sum()), **{f"share_in_{g_}": float(nc[D[g_]].mean()) for g_ in ["rural", "tribal", "wf"]}, **{f"share_out_{g_}": float(nc[~D[g_]].mean()) for g_ in ["rural", "tribal", "wf"]})
a = D.n - D.attr
N["buildings"] = dict(rural=int(D.n[D.rural].sum()), rural_attrless=int(a[D.rural].sum()), nonrural=int(D.n[~D.rural].sum()), nonrural_attrless=int(a[~D.rural].sum()),
                      ml_share_bw_rural=float(D.ml[D.rural].sum() / D.n[D.rural].sum()), ml_share_bw_nonrural=float(D.ml[~D.rural].sum() / D.n[~D.rural].sum()),
                      attrless_bw_rural_wf=float(a[D.rural & D.wf].sum() / D.n[D.rural & D.wf].sum()), attrless_bw_tribal_rural=float(a[D.tribal & D.rural].sum() / D.n[D.tribal & D.rural].sum()))
N["pop_affected"] = {g_: dict(population=int(D["pop"][D[g_]].sum()), in_attr_lt10=int(D["pop"][D[g_] & (D.attr_share < 0.1)].sum())) for g_ in ["rural", "tribal", "wf"]}
N["poi"] = dict(per1k_pooled_rural=float(D.npoi[D.rural].sum() / D["pop"][D.rural].sum() * 1000), per1k_pooled_nonrural=float(D.npoi[~D.rural].sum() / D["pop"][~D.rural].sum() * 1000),
                per_hu_pooled_rural=float(D.npoi[D.rural].sum() / D.hu[D.rural].sum()), per_hu_pooled_nonrural=float(D.npoi[~D.rural].sum() / D.hu[~D.rural].sum()),
                has_fire_rural=float(D.has_fire[D.rural].mean()), has_fire_nonrural=float(D.has_fire[~D.rural].mean()))
# named counties (first five digits of GEOID), building-weighted
cty = {"06007": "Butte County, CA", "06063": "Plumas County, CA", "48265": "Kerr County, TX", "48171": "Gillespie County, TX", "40021": "Cherokee County, OK", "04013": "Maricopa County, AZ"}
N["counties"] = {}
for f, nm in cty.items():
    mm = D.index.str.startswith(f)
    N["counties"][nm] = dict(tracts=int(mm.sum()), population=int(D["pop"][mm].sum()), buildings=int(D.n[mm].sum()), ml_share_bw=float(D.ml[mm].sum() / D.n[mm].sum()), attrless_bw=float(a[mm].sum() / D.n[mm].sum()))
# calibrated composite gap by ML-share bin (needs the final submission)
if T24:
    s = pd.read_csv(T24, dtype={"GEOID": str}).set_index("GEOID").coverage_gap_score.reindex(D.index)
    bins = pd.cut(D.ml_share, [-0.01, 0.1, 0.5, 0.9, 1.0], labels=["<10%", "10-50%", "50-90%", ">90%"])
    N["t24_by_mlbin"] = {str(k): dict(tracts=int(v.size), all=float(v.mean()), rural=float(s[(bins == k) & D.rural].mean()), urban=float(s[(bins == k) & ~D.rural].mean())) for k, v in s.groupby(bins, observed=True)}
    N["t24_disparity"] = dict(rural_ratio=float(s[D.rural].mean() / s[~D.rural].mean()), ml90_ratio=float(s[m].mean() / s[~m].mean()), ml90_vs_lt10=float(s[m].mean() / s[D.ml_share < 0.1].mean()), rural_pooled_d=float(tt(s[D.rural], s[~D.rural])[4]))
json.dump(N, open(f"{OUT}/quality_gap_numbers.json", "w"), indent=1)
print(json.dumps(N, indent=1)[:6000])
# figures
cx = np.array([(x["xmin"] + x["xmax"]) / 2 for x in TR.bbox]); cy = np.array([(x["ymin"] + x["ymax"]) / 2 for x in TR.bbox])
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for ax, r in zip(axs.ravel(), R):
    mm = (D.region == r).values
    sc = ax.scatter(cx[mm], cy[mm], c=D.ml_share.values[mm], cmap="RdYlBu_r", vmin=0, vmax=1, s=6, linewidths=0)
    ax.set_title(f"{NAMES[r]} — mean ML-only share {np.nanmean(D.ml_share.values[mm]):.2f}", fontsize=10)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(cy[mm])))); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=axs, shrink=0.6, label="Share of Overture buildings that are Microsoft ML footprints (no attributes)")
fig.suptitle("Where open building coverage is machine-only (tract bounding-box centres)", fontsize=12)
fig.savefig(f"{OUT}/fig_ml_share_map.png", dpi=130, bbox_inches="tight"); plt.close()
lab = {"rural": "Rural", "tribal": "Tribal stat. area", "wf": "Wildfire exposure top 10%", "burned": "Ever burned (USGS)", "cvi": "CVI top 10%", "heat": "Heat days top 10%", "svi": "SVI ≥ 0.75"}
fig, ax = plt.subplots(figsize=(9.5, 4.2)); x = np.arange(len(GROUPS)); w = 0.38
mi = [D.loc[D[k], "ml_share"].mean() for k in GROUPS]; mo = [D.loc[~D[k], "ml_share"].mean() for k in GROUPS]
ax.bar(x - w / 2, mi, w, label="in stratum", color="#c0392b"); ax.bar(x + w / 2, mo, w, label="rest", color="#7f8c8d")
ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in GROUPS], rotation=15, ha="right", fontsize=9); ax.set_ylim(0, 1); ax.legend(); ax.set_ylabel("Mean share of ML-only buildings")
for i in range(len(GROUPS)):
    ax.text(i - w / 2, mi[i] + 0.02, f"{mi[i]:.2f}", ha="center", fontsize=8); ax.text(i + w / 2, mo[i] + 0.02, f"{mo[i]:.2f}", ha="center", fontsize=8)
ax.set_title(f"Machine-only building coverage by scorecard stratum (4 scored regions, n = {len(D):,} tracts)", fontsize=10)
fig.savefig(f"{OUT}/fig_ml_share_groups.png", dpi=130, bbox_inches="tight"); plt.close()
ok = D.n > 50
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(D.attr_share[ok], D.count_cov.clip(0, 2)[ok], s=4, alpha=0.3, c=np.where(D.rural[ok], "#c0392b", "#2c3e50"), linewidths=0)
ax.axhline(1, color="k", lw=0.6, ls="--"); ax.set_xlabel("Share of buildings with any attribute (type/class)"); ax.set_ylabel("Count coverage: Overture / USFS building count"); ax.set_ylim(0, 2)
ax.set_title(f"Count-complete but content-empty (tracts with > 50 buildings, n = {int(ok.sum()):,}; red = rural)", fontsize=9)
fig.savefig(f"{OUT}/fig_count_vs_content.png", dpi=130, bbox_inches="tight"); plt.close()
D.to_parquet(f"{OUT}/quality_gap_tracts.parquet")
