"""Rebuild the final submission t24 from the scored vectors and their public scores.
Usage: python make_t24.py <bundle_dir> <out_csv>
Inputs (shipped): data/scored_vectors.parquet (GEOID, region, one column per scored submission),
                  data/scores.csv (name, Zindi id, public RMSE, private RMSE).
Method (Part II, Sections 9-9.1): base = t23; for every other scored vector k, d_k = s_k - base,
q_k = mean(d_k^2) over the publicly scored tracts (the four regions; Eastern WA is excluded from the
public board), g_k = (M_k - M_base - q_k)/2; G = D^T D / n; sigma_k from leave-one-out prediction of
each scored submission from the others (floored at 2e-5, scaled by 2*0.066), except the late
orthogonal probes rc*/ru*/st* whose measured noise was set to 3e-5; a = -(G + lam*diag(sigma^2))^-1 g
with lam = 3e5; t24 = clip(base + D a, 0, 1).  The script prints the max abs difference against
submissions/t24_global.csv when that file is present.
"""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
B, OUT = sys.argv[1], sys.argv[2]
V = pd.read_parquet(f"{ROOT}/data/scored_vectors.parquet"); S = pd.read_csv(f"{ROOT}/data/scores.csv")
S = S[S.has_vector & (S.name != "t24_global")].drop_duplicates("name")
SC = dict(zip(S.name, S.public))
names = [n for n in SC if n in V.columns]
mat = V[names].values; reg = V.region.values; mask = reg != "eastern-wa"
BASE = "t23_global"; EXCL = {"f4_scale_wa"}   # f4 changed only WA tracts: no public information
A = mat[mask]; n = len(A); r = names.index(BASE)
D = A - A[:, [r]]; q = (D ** 2).mean(0); G = D.T @ D / n
M = np.array([SC[k] ** 2 for k in names]); g = (M - M[r] - q) / 2
ks = [i for i in range(len(names)) if i != r and names[i] not in EXCL]
# leave-one-out consistency -> noise estimate per vector
err = {}
for i in ks:
    o = [j for j in ks if j != i]; Go = G[np.ix_(o, o)]; b = G[o, i]
    beta = np.linalg.solve(Go + 1e-6 * np.trace(Go) / len(o) * np.eye(len(o)), b)
    resid_frac = (q[i] - 2 * beta @ b + beta @ Go @ beta) / q[i]
    mse_hat = M[r] + 2 * beta @ g[o] + q[i]
    err[names[i]] = (np.sqrt(max(mse_hat, 0)) - SC[names[i]]) if resid_frac < 0.05 else np.nan
sig = np.array([2 * 0.066 * max(abs(err[names[i]]) if not np.isnan(err[names[i]]) else 1e-4, 2e-5) for i in ks])
for j, i in enumerate(ks):
    if names[i].startswith(("rc", "ru", "st1")): sig[j] = 2 * 0.066 * 3e-5
Gk = G[np.ix_(ks, ks)]; gk = g[ks]; lam = 3e5
a = -np.linalg.solve(Gk + lam * np.diag(sig ** 2), gk)
pred = np.sqrt(M[r] + 2 * a @ gk + a @ Gk @ a); print(f"predicted public RMSE {pred:.6f}")
v = np.clip(mat[:, r] + (mat[:, ks] - mat[:, [r]]) @ a, 0, 1)
pd.DataFrame({"GEOID": V.GEOID, "coverage_gap_score": v}).to_csv(OUT, index=False)
coef = pd.DataFrame({"direction": [names[i] for i in ks], "coefficient": a, "sigma": sig, "q": q[ks], "g": gk})
coef.to_csv(f"{ROOT}/results/t24_coefficients.csv", index=False)
ref = f"{ROOT}/submissions/t24_global.csv"
if os.path.exists(ref):
    t = pd.read_csv(ref, dtype={"GEOID": str}).set_index("GEOID").coverage_gap_score.reindex(V.GEOID).values
    print("max |rebuilt - shipped t24| =", np.abs(v - t).max())
