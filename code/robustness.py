"""Part I robustness: region fixed effects, county-clustered standard errors, density control,
population weighting, tagging decomposition, exposure by urban/rural, artifact clustering.
Usage: python robustness.py <bundle_dir> <rb3.parquet> <out_dir>
Writes results/robustness.json and results/regressions.csv.
OLS with county-clustered (CR1) SEs; county = first five GEOID digits. Density = log10(Overture buildings per km^2 of land)."""
import sys, os, json, numpy as np, pandas as pd
B, RB, OUT = sys.argv[1:4]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx"]
TR = pd.concat([pd.read_parquet(f"{B}/{r}-tracts.parquet").assign(region=r) for r in R]).drop_duplicates("GEOID").set_index("GEOID")
ST = pd.concat([pd.read_parquet(f"{B}/{r}-strata.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID").reindex(TR.index)
b = pd.concat([pd.read_parquet((f"{B}/{r}-bldg-tract.parquet" if os.path.exists(f"{B}/{r}-bldg-tract.parquet") else f"{_ROOT}/data/derived/{r}-bldg-tract.parquet")) for r in R]); g = b.groupby("GEOID")
D = pd.DataFrame({"n": g.n.sum(), "ml": b[b.src == "Microsoft ML Buildings"].groupby("GEOID").n.sum(),
                  "osm": b[b.src == "OpenStreetMap"].groupby("GEOID").n.sum(),
                  "osm_typed": b[(b.src == "OpenStreetMap") & b.subtype.notna()].groupby("GEOID").n.sum(),
                  "attr": b[b.subtype.notna()].groupby("GEOID").n.sum(), "multi": g.n_multisrc.sum()}).reindex(TR.index).fillna(0)
D = D[D.n > 0]
D["ml_share"], D["attr_share"], D["multi_share"] = D.ml / D.n, D.attr / D.n, D.multi / D.n
T = TR.reindex(D.index); S = ST.reindex(D.index)
D["region"] = T.region; D["county"] = D.index.str[:5]; D["pop"] = T.pop_total.fillna(0)
D["rural"] = (S.ur_class == "Rural").astype(float); D["tribal"] = (S.tribal_statistical > 0).astype(float)
D["wf"] = (S.usfs_Exposure_mean >= ST.usfs_Exposure_mean.quantile(.9)).astype(float)
D["svi"] = (S.svi_overall >= 0.75).astype(float)
D["ldens"] = np.log10(D.n / (T.ALAND / 1e6).clip(lower=0.01))
D = D[np.isfinite(D.ldens)]
def ols(y, X, cl, w=None):
    X = np.column_stack(X); y = np.asarray(y, float); w = np.ones(len(y)) if w is None else np.asarray(w, float)
    sw = np.sqrt(w); Xw, yw = X * sw[:, None], y * sw
    XtX = np.linalg.inv(Xw.T @ Xw); beta = XtX @ Xw.T @ yw; e = yw - Xw @ beta
    meat = np.zeros((X.shape[1], X.shape[1])); cls = pd.Series(cl).values
    for c in np.unique(cls):
        idx = cls == c; s = Xw[idx].T @ e[idx]; meat += np.outer(s, s)
    G, n, k = len(np.unique(cls)), len(y), X.shape[1]
    V = XtX @ meat @ XtX * G / (G - 1) * (n - 1) / (n - k)
    return beta, np.sqrt(np.diag(V))
FE = pd.get_dummies(D.region, drop_first=True).astype(float).values.T
rows = []
for yname in ["ml_share", "attr_share", "multi_share"]:
    for spec, cols, use_fe, weighted in [("marginal rural", ["rural"], False, False), ("rural + region FE", ["rural"], True, False),
                                         ("rural + region FE + density", ["rural", "ldens"], True, False),
                                         ("full", ["rural", "tribal", "wf", "svi", "ldens"], True, False),
                                         ("full, population-weighted", ["rural", "tribal", "wf", "svi", "ldens"], True, True)]:
        X = [np.ones(len(D))] + [D[c].values for c in cols] + (list(FE) if use_fe else [])
        beta, se = ols(D[yname], X, D.county, D["pop"] if weighted else None)
        for i, c in enumerate(cols):
            rows.append(dict(outcome=yname, spec=spec, variable=c, coef=beta[i + 1], se_cluster=se[i + 1], t_cluster=beta[i + 1] / se[i + 1]))
REG = pd.DataFrame(rows); REG.to_csv(f"{OUT}/regressions.csv", index=False)
N = {"n_tracts": int(len(D)), "n_counties": int(D.county.nunique())}
# tagging decomposition of the attribute gap (building-weighted, rural vs urban)
for lab, m in [("rural", D.rural == 1), ("urban", D.rural == 0)]:
    N[f"bw_{lab}"] = dict(osm_share=float(D.osm[m].sum() / D.n[m].sum()), osm_subtype_rate=float(D.osm_typed[m].sum() / D.osm[m].sum()),
                         attr_share=float(D.attr[m].sum() / D.n[m].sum()), ml_share=float(D.ml[m].sum() / D.n[m].sum()))
ru, ur = N["bw_rural"], N["bw_urban"]
N["attr_if_rural_had_urban_osm_share"] = ru["osm_subtype_rate"] * ur["osm_share"]
N["attr_if_rural_had_urban_tagging"] = ur["osm_subtype_rate"] * ru["osm_share"]
tot = np.log(ur["attr_share"] / ru["attr_share"]); N["log_decomp_share_osm_coverage"] = float(np.log(ur["osm_share"] / ru["osm_share"]) / tot)
# population-weighted shares
for grp in ["rural", "tribal", "wf"]:
    m = D[grp] == 1
    N[f"popw_ml_{grp}"] = [float(np.average(D.ml_share[m], weights=D["pop"][m].clip(lower=1e-9))), float(np.average(D.ml_share[~m], weights=D["pop"][~m].clip(lower=1e-9)))]
# exposure: tracts > 90% machine-only by urban/rural and region
m90 = D.ml_share > 0.9
N["ml90_exposure"] = dict(tracts=int(m90.sum()), pop=int(D["pop"][m90].sum()), pop_urban=int(D["pop"][m90 & (D.rural == 0)].sum()),
                          tracts_urban=int((m90 & (D.rural == 0)).sum()), tracts_rural=int((m90 & (D.rural == 1)).sum()),
                          urban_tracts_by_region=D[m90 & (D.rural == 0)].region.value_counts().to_dict())
# wildfire within rural tracts by region
N["wf_within_rural_by_region"] = {r: [float(D.ml_share[(D.region == r) & (D.rural == 1) & (D.wf == 1)].mean()), float(D.ml_share[(D.region == r) & (D.rural == 1) & (D.wf == 0)].mean())] for r in R}
N["tribal_tracts_by_region"] = D[D.tribal == 1].region.value_counts().to_dict()
# count coverage dispersion by ML-share bin
cc = D.n / S.usfs_BuildingCount_sum.reindex(D.index).replace(0, np.nan)
bins = pd.cut(D.ml_share, [-.01, .1, .5, .9, 1.0], labels=["<10%", "10-50%", "50-90%", ">90%"])
N["count_cov_by_bin"] = {str(k): dict(median=float(v.median()), iqr=float(v.quantile(.75) - v.quantile(.25))) for k, v in cc.groupby(bins, observed=True)}
# Finding 2 artifacts with region FE + county clusters
rb = pd.read_parquet(RB); rb = rb[rb.region.isin(R)]
c4 = rb["class"].isin(["motorway", "trunk", "primary", "secondary"]); lk = rb["sub"] == "link"; rf = (rb.core | rb.fm) & ~lk
gg = lambda m, col: rb[m].groupby("GEOID")[col].sum()
A = pd.DataFrame({"r8": gg(rf, "L8"), "r0": gg(rf, "L0"), "ov": gg(c4, "L0"), "ovn": gg(c4 & ~lk, "L0")}).fillna(0); A = A[A.r0 > 100]
gap = lambda o, r: (1 - o / r.clip(lower=1)).clip(0, 1)
A["bnd"] = gap(A.ov, A.r8) - gap(A.ov, A.r0); A["ramp"] = gap(A.ovn, A.r0) - gap(A.ov, A.r0); A["live"] = gap(A.ov, A.r0) > 0
SA = ST.reindex(A.index); A["region"] = TR.region.reindex(A.index); A["county"] = A.index.str[:5]
FEa = pd.get_dummies(A.region, drop_first=True).astype(float).values.T
art = []
for y in ["bnd", "ramp"]:
    for gname, gv in [("rural", (SA.ur_class == "Rural")), ("tribal", (SA.tribal_statistical > 0)), ("svi_minority", (SA.svi_minority >= 0.75))]:
        x = gv.fillna(False).astype(float).values
        for spec, X in [("raw, clustered", [np.ones(len(A)), x]), ("region FE, clustered", [np.ones(len(A)), x] + list(FEa))]:
            beta, se = ols(A[y], X, A.county); art.append(dict(artifact=y, group=gname, spec=spec, coef=beta[1], t_cluster=beta[1] / se[1]))
N["artifact_regressions"] = art
ruA = (SA.ur_class == "Rural").values
N["artifact_live_vs_zero"] = dict(live_rural=float(A.bnd[A.live & ruA].mean()), live_urban=float(A.bnd[A.live & ~ruA].mean()),
                                  zero_rural=float(A.bnd[~A.live & ruA].mean()), zero_urban=float(A.bnd[~A.live & ~ruA].mean()),
                                  share_rural_artifact_from_zero_gap=float(A.bnd[~A.live & ruA].sum() / A.bnd[ruA].sum()))
json.dump(N, open(f"{OUT}/robustness.json", "w"), indent=1)
pd.set_option("display.width", 200); print(REG.round(3).to_string()); print(json.dumps(N, indent=1, default=str))
