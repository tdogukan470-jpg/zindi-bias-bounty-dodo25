"""Part I, Findings 2-3: boundary double-count and ramp artifacts in the transport component.
Reads data/rb3.parquet (per tract x class x route-flags x subclass: L0 = road length strictly inside the
tract, L8/L25 = inside the tract buffered by 8/25 m, Lbd8 = within 8 m of the tract boundary), the strata
tables, and optionally the final submission and the two leaderboard probes. Writes
results/bias_artifacts_table.csv and results/bias_artifacts_numbers.json.
Usage: python bias_artifacts.py <rb3.parquet> <bundle_dir> <out_dir>
(the final submission, probes t14/c4 and their scores are read from data/scored_vectors.parquet and data/scores.csv;
 the model definedness flag from data/roadfeat2.parquet)
Definitions (artifact analysis): reference proxy = route-tagged roads (core: Interstate/US/State; fm: Texas
FM/RM), subclass != link; numerator = Overture motorway/trunk/primary/secondary, all subclasses; defined =
strict-inside reference > 100 m. boundary_added_gap = gap(ov, ref_L8) - gap(ov, ref_L0); ramp_hidden_gap =
gap(ov_without_links, ref_L0) - gap(ov, ref_L0); gap(o, r) = clip(1 - o/r, 0, 1). Cohen's d uses the pooled SD.
"""
import sys, json, numpy as np, pandas as pd
from math import erf, sqrt
import os
RB, B, OUT = sys.argv[1:4]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC = pd.read_parquet(f"{ROOT}/data/scored_vectors.parquet").set_index("GEOID")
SCS = pd.read_csv(f"{ROOT}/data/scores.csv").drop_duplicates("name").set_index("name")
R = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx"]
rb = pd.read_parquet(RB); rb = rb[rb.region.isin(R)]
ST = pd.concat([pd.read_parquet(f"{B}/{r}-strata.parquet") for r in R]).drop_duplicates("GEOID").set_index("GEOID")
c4 = rb["class"].isin(["motorway", "trunk", "primary", "secondary"]); lk = rb["sub"] == "link"; rf = (rb.core | rb.fm) & ~lk
g = lambda m, col: rb[m].groupby("GEOID")[col].sum()
T = pd.DataFrame({"ref_L8": g(rf, "L8"), "ref_L0": g(rf, "L0"), "ref_bd": g(rf, "Lbd8"), "ov": g(c4, "L0"), "ov_nolink": g(c4 & ~lk, "L0"), "ov_link": g(c4 & lk, "L0")}).fillna(0)
ALL = T.copy(); T = T[T.ref_L0 > 100]
gap = lambda o, r: (1 - o / r.clip(lower=1)).clip(0, 1)
T["boundary_added_gap"] = gap(T.ov, T.ref_L8) - gap(T.ov, T.ref_L0)
T["ramp_hidden_gap"] = gap(T.ov_nolink, T.ref_L0) - gap(T.ov, T.ref_L0)
S = ST.reindex(T.index)
groups = {"Rural": S.ur_class == "Rural", "Tribal statistical area": S.tribal_statistical > 0, "High SVI minority (>=0.75)": S.svi_minority >= 0.75,
          "High SVI overall (>=0.75)": S.svi_overall >= 0.75, "High CVI (top 10%)": S.cvi_overall >= ST.cvi_overall.quantile(.9)}
def tt(x, y):
    n1, n2 = len(x), len(y); mx, my, vx, vy = x.mean(), y.mean(), x.var(ddof=1), y.var(ddof=1)
    t = (mx - my) / np.sqrt(vx / n1 + vy / n2); sp = np.sqrt(((n1 - 1) * vx + (n2 - 1) * vy) / (n1 + n2 - 2))
    return mx, my, t, 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))), (mx - my) / sp
rows = []
for art in ["boundary_added_gap", "ramp_hidden_gap"]:
    for name, msk in groups.items():
        msk = msk.fillna(False).values; mx, my, t, p, d = tt(T[art][msk], T[art][~msk])
        rows.append(dict(artifact=art, group=name, n_in=int(msk.sum()), n_out=int((~msk).sum()), mean_in=mx, mean_out=my, ratio=mx / my, t=t, p=p, cohen_d=d))
out = pd.DataFrame(rows); out.to_csv(f"{OUT}/bias_artifacts_table.csv", index=False); print(out.round(4).to_string())
N = {"defined_tracts": int(len(T)), "mean_boundary_artifact": float(T.boundary_added_gap.mean()), "share_gt_0_05_defined": float((T.boundary_added_gap > 0.05).mean()),
     "share_gt_0_05_all": float((T.boundary_added_gap > 0.05).sum() / len(ALL)), "mean_ramp_artifact": float(T.ramp_hidden_gap.mean())}
# exposure vs sensitivity (why the artifact concentrates in rural tracts)
ru = (S.ur_class == "Rural").fillna(False).values
N["boundary_share_of_ref_length"] = dict(all_regions=float(rb.Lbd8[rf].sum() / rb.L8[rf].sum()), motorway_trunk=float(rb.Lbd8[rf & rb["class"].isin(["motorway", "trunk"])].sum() / rb.L8[rf & rb["class"].isin(["motorway", "trunk"])].sum()),
                                         tract_mean_rural=float((T.ref_bd / T.ref_L8)[ru].mean()), tract_mean_nonrural=float((T.ref_bd / T.ref_L8)[~ru].mean()))
N["buffer_adds_share_of_ref"] = dict(median_rural=float(((T.ref_L8 - T.ref_L0) / T.ref_L0)[ru].median()), median_nonrural=float(((T.ref_L8 - T.ref_L0) / T.ref_L0)[~ru].median()))
N["ov_over_ref_median"] = dict(rural=float((T.ov / T.ref_L0)[ru].median()), nonrural=float((T.ov / T.ref_L0)[~ru].median()))
N["gap_zero_share"] = dict(rural=float((gap(T.ov, T.ref_L0) == 0)[ru].mean()), nonrural=float((gap(T.ov, T.ref_L0) == 0)[~ru].mean()))
# link share of all four-class Overture length (tract mean, all tracts with roads)
ls = (ALL.ov_link / ALL.ov.replace(0, np.nan)); SA = ST.reindex(ALL.index)
N["link_share_of_4class_length"] = dict(high_svi_minority=float(ls[(SA.svi_minority >= 0.75).fillna(False).values].mean()), rest=float(ls[~(SA.svi_minority >= 0.75).fillna(False).values].mean()),
                                        rural=float(ls[(SA.ur_class == "Rural").values].mean()), nonrural=float(ls[(SA.ur_class != "Rural").values].mean()))
N["link_length_by_class"] = (rb[c4 & lk].groupby("class").L0.sum() / rb[c4 & lk].L0.sum()).round(3).to_dict()
if True:  # propagate the boundary artifact into the final composite (model output, not the reference)
    s = VEC["t24_global"]
    s = s[s.index.isin(ST.index)]; a = T.boundary_added_gap.reindex(s.index).fillna(0); s2 = (s - a / 3).clip(0, 1); SS = ST.reindex(s.index)
    for nm, msk in [("rural", (SS.ur_class == "Rural").values), ("tribal", (SS.tribal_statistical > 0).values)]:
        r1, r2 = s[msk].mean() / s[~msk].mean(), s2[msk].mean() / s2[~msk].mean(); d1, d2 = s[msk].mean() - s[~msk].mean(), s2[msk].mean() - s2[~msk].mean()
        N[f"composite_{nm}"] = dict(ratio_with=float(r1), ratio_without=float(r2), diff_with=float(d1), diff_without=float(d2), artifact_share_of_diff=float((d1 - d2) / d1))
if True:  # probe calculus for c4 (no buffer) relative to t14; public board only (four regions)
    base = VEC["t14_joint_b1234"]; probe = VEC["c4_ref_noBuffer"]
    u = (probe - base).reindex(base.index); mk = base.index.isin(ST.index); u = u[mk].fillna(0).values
    M0, M1 = SCS.loc["t14_joint_b1234", "public"], SCS.loc["c4_ref_noBuffer", "public"]
    q = (u ** 2).mean(); c = (M1 ** 2 - M0 ** 2 - q) / 2; N["c4_astar_public"] = float(-c / q)
    N["c4_scores"] = dict(t14_public=float(M0), c4_public=float(M1), t14_private=float(SCS.loc["t14_joint_b1234", "private"]), c4_private=float(SCS.loc["c4_ref_noBuffer", "private"]))
if True:  # artifact under the leaderboard model's definedness flag (core-route reference within buffers > 100 m)
    f = pd.read_parquet(f"{ROOT}/data/roadfeat2.parquet").set_index("GEOID"); mflag = (f.coreB.reindex(ALL.index).fillna(0) > 100)
    TT = ALL[mflag]; art = gap(TT.ov, TT.ref_L8) - gap(TT.ov, TT.ref_L0); S2 = ST.reindex(TT.index); ru2 = (S2.ur_class == "Rural").fillna(False).values
    N["boundary_artifact_under_model_flag"] = dict(defined=int(len(TT)), mean=float(art.mean()), rural=float(art[ru2].mean()), nonrural=float(art[~ru2].mean()))
json.dump(N, open(f"{OUT}/bias_artifacts_numbers.json", "w"), indent=1); print(json.dumps(N, indent=1))
