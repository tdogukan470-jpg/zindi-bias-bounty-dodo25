"""Section 1.4: how much of the leaderboard evidence for Finding 1 survives region, rurality and density.
The OSM-share probe u = t4b - t3 has a known inner product g_u with the hidden residual (quadratic identity).
Region, rural, rural x region and density indicators z are not scored directly, but each can be expressed in
the span of the 75 scored difference vectors d_k = s_k - t3 (least squares); its inner product with the
hidden residual is then estimated as beta_z . g. We partial z out of u and recompute the attainable gain.
Usage: python probe_partial.py <bundle_dir> <out_json>   (reads data/scored_vectors.parquet, data/scores.csv)"""
import sys, os, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); B, OUT = sys.argv[1:3]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx"]
V = pd.read_parquet(f"{ROOT}/data/scored_vectors.parquet").set_index("GEOID"); V = V[V.region != "eastern-wa"]
SC = pd.read_csv(f"{ROOT}/data/scores.csv"); SC = SC[SC.has_vector].drop_duplicates("name"); pub = dict(zip(SC.name, SC.public))
TR = pd.concat([pd.read_parquet(f"{B}/{r}-tracts.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID").reindex(V.index)
ST = pd.concat([pd.read_parquet(f"{B}/{r}-strata.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID").reindex(V.index)
bl = pd.concat([pd.read_parquet((f"{B}/{r}-bldg-tract.parquet" if os.path.exists(f"{B}/{r}-bldg-tract.parquet") else f"{_ROOT}/data/derived/{r}-bldg-tract.parquet")) for r in R]).groupby("GEOID").n.sum().reindex(V.index).fillna(0)
names = [n for n in pub if n in V.columns and n not in ("f4_scale_wa", "t24_global")]; base = "t3_transport_8_25"
A = V[names].values; r = names.index(base); Dm = A - A[:, [r]]; q = (Dm ** 2).mean(0)
M = np.array([pub[k] ** 2 for k in names]); g = (M - M[r] - q) / 2
keep = [i for i in range(len(names)) if i != r]; Dk, gk = Dm[:, keep], g[keep]
rural = (ST.ur_class == "Rural").astype(float).values; reg = pd.get_dummies(V.region).astype(float)
ld = np.log10(bl / (TR.ALAND / 1e6).clip(lower=0.01)).replace([np.inf, -np.inf], np.nan); ld = ld.fillna(ld.median()).values
Z = {c: reg[c].values for c in reg}; Z["rural"] = rural; Z["ldens"] = ld - ld.mean()
for c in reg: Z["rural_x_" + c] = reg[c].values * rural
G = Dk.T @ Dk / len(Dk); est = {}
for k, v in Z.items():
    beta = np.linalg.solve(G + 1e-8 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), Dk.T @ v / len(v))
    est[k] = dict(g=float(beta @ gk), resid_frac=float(((v - Dk @ beta) ** 2).mean() / (v ** 2).mean()))
u = V["t4b_bldg_osmshare_flip"].values - V[base].values; iu = names.index("t4b_bldg_osmshare_flip"); gu, qu = g[iu], (u ** 2).mean()
N = {"t4b": dict(a_star=float(-gu / qu), mse_gain=float(gu ** 2 / qu), rmse_after=float(np.sqrt(M[r] - gu ** 2 / qu))), "indicator_projection": est}
for lab, keys in [("region+rural", list(reg.columns) + ["rural"]), ("region+rural+rural_x_region+density", list(reg.columns) + ["rural", "ldens"] + ["rural_x_" + c for c in reg])]:
    Zm = np.column_stack([Z[k] for k in keys]); c, *_ = np.linalg.lstsq(Zm, u, rcond=None)
    up = u - Zm @ c; gp = gu - sum(c[j] * est[k]["g"] for j, k in enumerate(keys)); qp = (up ** 2).mean()
    N[lab] = dict(var_share_left=float(qp / qu), a_star=float(-gp / qp), mse_gain=float(gp ** 2 / qp), share_of_gain=float((gp ** 2 / qp) / (gu ** 2 / qu)), rmse_after=float(np.sqrt(M[r] - gp ** 2 / qp)))
json.dump(N, open(OUT, "w"), indent=1); print(json.dumps({k: v for k, v in N.items() if k != "indicator_projection"}, indent=1)); print({k: round(v["resid_frac"], 3) for k, v in est.items()})
