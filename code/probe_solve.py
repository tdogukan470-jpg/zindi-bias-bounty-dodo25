"""Ridge-regularized multi-probe solver.

Verilenler:
  base: baseline tahmin vektörü (n,)
  U:    {name: delta_vektör (n,)}  # her probe'da base'e eklenen vektör (α=1 taban)
  scores: {name: measured_rmse}    # probe'un LB skoru
  M0:    base'in LB RMSE'si

Çıktı:
  a_dict: {name: a*}  optimum ağırlıklar
  pred:   RMSE tahmini
"""
import numpy as np

def ridge_solve(U_mat, c_vec, lam):
    """Solve (Q + lam*diag(Q))*a = -c"""
    Q = U_mat.T @ U_mat / U_mat.shape[0]
    reg = lam * np.diag(np.diag(Q))
    a = -np.linalg.solve(Q + reg, c_vec)
    pred_sq = np.maximum(0, -c_vec @ a)   # a=-Q^-1 c, min = M0^2 - c^T Q^-1 c
    return a, pred_sq, Q

def multi_probe(base, U_dict, scores, M0_rmse, lam=1e-2):
    """Fit optimal weights with Tikhonov on diag(Q)."""
    M0 = M0_rmse**2
    names = list(U_dict)
    U_mat = np.stack([U_dict[k] for k in names], axis=1)  # (n, p)
    c = np.array([
        (scores[k]**2 - M0 - (U_dict[k]**2).mean()) / 2
        for k in names
    ])
    a, pred_sq, Q = ridge_solve(U_mat, c, lam)
    pred = np.sqrt(max(0, M0 - c @ a))
    return dict(zip(names, a)), pred, Q, c

def apply_combo(base, U_dict, a_dict, clip_lo=0.01, clip_hi=0.45):
    d = sum(a_dict[k] * U_dict[k] for k in a_dict)
    return np.clip(base + d, clip_lo, clip_hi)
