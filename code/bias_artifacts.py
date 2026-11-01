"""Part I, Findings 2-3: boundary double-count and ramp artifacts in the transport component.
Reads data/rb3.parquet (per tract x class x route-flags x subclass: L0 = length strictly inside the tract,
L8/L25 = inside the tract buffered by 8/25 m, Lbd8 = within 8 m of the boundary) and the strata tables.
Writes results/bias_artifacts_table.csv.   Usage: python bias_artifacts.py <rb3.parquet> <bundle_dir> <out_csv>
Definitions (match the writeup):
  reference  = route-tagged roads (core | fm), subclass != link
  numerator  = Overture motorway/trunk/primary/secondary, all subclasses (as in the leaderboard model)
  defined    = strict-inside reference > 100 m
  boundary_added_gap = gap(ov, ref_L8) - gap(ov, ref_L0)          (what double counting adds)
  ramp_hidden_gap    = gap(ov_without_links, ref_L0) - gap(ov, ref_L0)   (what counting ramps hides)
"""
import sys, numpy as np, pandas as pd
from math import erf, sqrt
RB, B, OUT = sys.argv[1:4]
R = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx"]
rb = pd.read_parquet(RB); rb = rb[rb.region.isin(R)]
ST = pd.concat([pd.read_parquet(f"{B}/{r}-strata.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID")
c4 = rb["class"].isin(["motorway", "trunk", "primary", "secondary"]); lk = rb["sub"] == "link"; rf = (rb.core | rb.fm) & ~lk
g = lambda m, col: rb[m].groupby("GEOID")[col].sum()
T = pd.DataFrame({"ref_L8": g(rf, "L8"), "ref_L0": g(rf, "L0"), "ov": g(c4, "L0"), "ov_nolink": g(c4 & ~lk, "L0")}).fillna(0)
T = T[T.ref_L0 > 100]
gap = lambda o, r: (1 - o / r.clip(lower=1)).clip(0, 1)
T["boundary_added_gap"] = gap(T.ov, T.ref_L8) - gap(T.ov, T.ref_L0)
T["ramp_hidden_gap"] = gap(T.ov_nolink, T.ref_L0) - gap(T.ov, T.ref_L0)
S = ST.reindex(T.index)
groups = {"High SVI minority (>=0.75)": S.svi_minority >= 0.75, "High SVI overall (>=0.75)": S.svi_overall >= 0.75,
          "High CVI (top 10%)": S.cvi_overall >= S.cvi_overall.quantile(.9), "Rural": S.ur_class == "Rural",
          "Tribal statistical area": S.tribal_statistical > 0}
def tt(x, y):
    mx, my, vx, vy = x.mean(), y.mean(), x.var(ddof=1), y.var(ddof=1)
    t = (mx - my) / np.sqrt(vx / len(x) + vy / len(y))
    return mx, my, t, 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))), (mx - my) / np.sqrt((vx + vy) / 2)
rows = []
for art in ["ramp_hidden_gap", "boundary_added_gap"]:
    for name, msk in groups.items():
        msk = msk.fillna(False).values; mx, my, t, p, d = tt(T[art][msk], T[art][~msk])
        rows.append(dict(artifact=art, group=name, n_in=int(msk.sum()), n_out=int((~msk).sum()), mean_in=mx, mean_out=my, t=t, p=p, cohen_d=d))
out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
print(f"defined tracts n={len(T)}; mean boundary artifact {T.boundary_added_gap.mean():.4f}; share >0.05: {(T.boundary_added_gap>0.05).mean():.3f}; mean ramp artifact {T.ramp_hidden_gap.mean():.4f}")
print(out.round(4).to_string())
