"""
HNO3-CDR & SNO3-CDR — Do & Lauw (2025)  [FAST version]
Run AFTER matrix_vg (source) and matrix_ty (target) are loaded.
"""

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Speed knobs (badhao agar accuracy chahiye) ────────────────────────────────
FAST = True          # True = tez run, False = paper-like (slow)
HP = dict(
    F=20,
    ETA=0.01 if FAST else 0.005,
    EPOCHS=30 if FAST else 100,
    l_uv=0.01, l_bi=0.01,
    gamma=0.5,
    warmup_epochs=3 if FAST else 5,
    sinkhorn_reg=0.1,
    sinkhorn_every=5 if FAST else 1,   # SNO3: har N epoch par Sinkhorn
    patience=4,
)


# ── MF (vectorised inner loop) ────────────────────────────────────────────────

def mf_sgd(R, f, epochs, eta, l_uv, l_bi, verbose=False, tag=""):
    t0 = time.time()
    m, n = R.shape
    U = np.random.rand(m, f) * 0.01
    V = np.random.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = float(R[R > 0].mean()) if (R > 0).any() else 0.0
    us, is_ = R.nonzero()
    rs = R[us, is_].astype(float)
    n_ratings = len(rs)
    if n_ratings == 0:
        return None
    if verbose:
        print(f"    [{tag}] MF train: {n_ratings} ratings, {epochs} epochs...", flush=True)
    for ep in range(epochs):
        idx = np.random.permutation(n_ratings)
        for j in idx:
            u, i, r = us[j], is_[j], rs[j]
            pred = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - pred
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"    [{tag}] epoch {ep+1}/{epochs} ({time.time()-t0:.0f}s)", flush=True)
    if not np.isfinite(U).all():
        return None
    return U, V, bu, bi, mu


def mf_sgd_es(R, R_val, f, epochs, eta, l_uv, l_bi, patience=4, min_delta=1e-4, verbose=False, tag=""):
    t0 = time.time()
    m, n = R.shape
    U = np.random.rand(m, f) * 0.01
    V = np.random.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = float(R[R > 0].mean()) if (R > 0).any() else 0.0
    us, is_ = R.nonzero()
    rs = R[us, is_].astype(float)
    if len(rs) == 0:
        return None
    best_rmse, wait = np.inf, 0
    bU = bV = bbu = bbi = None
    if verbose:
        print(f"    [{tag}] MF+ES: {len(rs)} ratings...", flush=True)
    for ep in range(epochs):
        idx = np.random.permutation(len(rs))
        for j in idx:
            u, i, r = us[j], is_[j], rs[j]
            pred = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - pred
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
        _, rmse = eval_pred(U @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best_rmse - min_delta:
            best_rmse, wait = rmse, 0
            bU, bV, bbu, bbi = U.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    [{tag}] early stop @ epoch {ep+1} ({time.time()-t0:.0f}s)", flush=True)
                break
    if bU is None:
        bU, bV, bbu, bbi = U, V, bu, bi
    if not np.isfinite(bU).all():
        return None
    return bU, bV, bbu, bbi, mu


def predict(U, V, bu, bi, mu):
    return U @ V.T + mu + bu[:, None] + bi[None, :]


def eval_pred(R_hat, R_true):
    us, is_ = R_true.nonzero()
    if len(us) == 0:
        return np.inf, np.inf
    t, p = R_true[us, is_], R_hat[us, is_]
    e = t - p
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


# ── NO3 helpers (vectorised) ──────────────────────────────────────────────────

def build_no3_matrix(R1, R2):
    n1, m1 = R1.shape
    n2, m2 = R2.shape
    R = np.zeros((n1 + n2, m1 + m2))
    R[:n1, :m1] = R1
    R[n1:, m1:] = R2
    return R, n1, n2, m1, m2


def hungarian_user_match(U1, U2):
    sim = cosine_similarity(U1, U2)
    _, tgt_idx = linear_sum_assignment(1.0 - sim)
    return tgt_idx  # mapping[k] = matched target user for source user k


def build_substituted_matrix(R1, R2, mapping, m1, m2):
    n1 = R1.shape[0]
    R_sub = np.zeros((n1, m1 + m2))
    R_sub[:, :m1] = R1
    R_sub[np.arange(n1), :]  # touch
    for k in range(n1):
        R_sub[k, m1:] = R2[mapping[k]]
    return R_sub


def _remap_target_val(R2_val, mapping, m1, m2, n1):
    inv = np.empty(R2_val.shape[0], dtype=int)
    inv[mapping] = np.arange(n1)
    R_out = np.zeros((n1, m1 + m2))
    rows = inv[np.arange(R2_val.shape[0])]
    for u in range(R2_val.shape[0]):
        R_out[rows[u], m1:] = R2_val[u]
    return R_out


# ── Sinkhorn (vectorised) ─────────────────────────────────────────────────────

def _sinkhorn_ot(C, reg=0.1, n_iter=30):
    n, m = C.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    K = np.exp(-C / reg)
    u = np.full(n, 1.0 / n)
    for _ in range(n_iter):
        v = b / (K.T @ u + 1e-12)
        u = a / (K @ v + 1e-12)
    P = u[:, None] * K * v[None, :]
    return float(np.sum(P * C)), P


def _sq_cost(A, B):
    # ||a_i - b_j||^2 via expansion (vectorised)
    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    return aa + bb - 2.0 * (A @ B.T)


def sinkhorn_symmetric(U1, U2, reg=0.1, n_iter=30):
    C12 = _sq_cost(U1, U2)
    C21 = _sq_cost(U2, U1)
    d12, P12 = _sinkhorn_ot(C12, reg, n_iter)
    d21, P21 = _sinkhorn_ot(C21, reg, n_iter)
    return d12 + d21, P12, P21


def sinkhorn_grad(U1, U2, P):
    # grad of sum_ij P_ij ||u1_i - u2_j||^2  w.r.t U1
    return 2.0 * (P @ U2 - (P.sum(1, keepdims=True)) * U1) / max(U1.shape[0], 1)


# ── Models ────────────────────────────────────────────────────────────────────

def hno3_cdr(R1_full, R2_train, R2_val, hp, verbose=False):
    R_comb, n1, n2, m1, m2 = build_no3_matrix(R1_full, R2_train)
    out1 = mf_sgd(R_comb, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"],
                  verbose=verbose, tag="HNO3-p1")
    if out1 is None:
        return None, None
    U, _, _, _, _ = out1
    mapping = hungarian_user_match(U[:n1], U[n1:n1 + n2])
    R_sub = build_substituted_matrix(R1_full, R2_train, mapping, m1, m2)
    R_val_hat = _remap_target_val(R2_val, mapping, m1, m2, n1)
    out2 = mf_sgd_es(R_sub, R_val_hat, hp["F"], hp["EPOCHS"], hp["ETA"],
                     hp["l_uv"], hp["l_bi"], patience=hp["patience"],
                     verbose=verbose, tag="HNO3-p2")
    if out2 is None:
        return None, None
    Uf, Vf, bu, bi, mu = out2
    return eval_pred(predict(Uf, Vf, bu, bi, mu), R_val_hat)


def sno3_cdr(R1_full, R2_train, R2_val, hp, verbose=False):
    R_comb, n1, n2, m1, m2 = build_no3_matrix(R1_full, R2_train)
    f, m, n = hp["F"], *R_comb.shape
    U = np.random.rand(m, f) * 0.01
    V = np.random.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = float(R_comb[R_comb > 0].mean()) if (R_comb > 0).any() else 0.0
    us, is_ = R_comb.nonzero()
    rs = R_comb[us, is_].astype(float)
    w = hp["warmup_epochs"]
    gamma = hp["gamma"]
    reg = hp["sinkhorn_reg"]
    severy = hp.get("sinkhorn_every", 5)
    if verbose:
        print(f"    [SNO3] train {len(rs)} ratings, {hp['EPOCHS']} epochs...", flush=True)
    for ep in range(hp["EPOCHS"]):
        idx = np.random.permutation(len(rs))
        for j in idx:
            u, i, r = us[j], is_[j], rs[j]
            pred = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - pred
            bu[u] += hp["ETA"] * (e - hp["l_bi"] * bu[u])
            bi[i] += hp["ETA"] * (e - hp["l_bi"] * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += hp["ETA"] * (e * vo - hp["l_uv"] * uo)
            V[i] += hp["ETA"] * (e * uo - hp["l_uv"] * vo)
        if ep >= w and (ep - w) % severy == 0:
            U1, U2 = U[:n1], U[n1:n1 + n2]
            _, P12, P21 = sinkhorn_symmetric(U1, U2, reg=reg)
            lr_s = hp["ETA"] * gamma
            U[:n1]  -= lr_s * sinkhorn_grad(U1, U2, P12)
            U[n1:n1 + n2] -= lr_s * sinkhorn_grad(U2, U1, P21)
        if verbose and ep % 10 == 0:
            print(f"    [SNO3] epoch {ep+1}/{hp['EPOCHS']}", flush=True)
    if not np.isfinite(U).all():
        return None, None
    mapping = hungarian_user_match(U[:n1], U[n1:n1 + n2])
    R_val_hat = _remap_target_val(R2_val, mapping, m1, m2, n1)
    return eval_pred(predict(U, V, bu, bi, mu), R_val_hat)


def mf_baseline(R1_full, R2_train, R2_val, hp, verbose=False):
    R_comb, n1, n2, m1, m2 = build_no3_matrix(R1_full, R2_train)
    R_val_comb, _, _, _, _ = build_no3_matrix(np.zeros_like(R1_full), R2_val)
    out = mf_sgd_es(R_comb, R_val_comb, hp["F"], hp["EPOCHS"], hp["ETA"],
                    hp["l_uv"], hp["l_bi"], patience=hp["patience"],
                    verbose=verbose, tag="MF-base")
    if out is None:
        return None, None
    U, V, bu, bi, mu = out
    R_hat = predict(U, V, bu, bi, mu)
    R_val_tgt = np.zeros_like(R_comb)
    R_val_tgt[n1:, m1:] = R2_val
    return eval_pred(R_hat, R_val_tgt)


METHODS = {
    "MF-HNO3": hno3_cdr,
    "MF-SNO3": sno3_cdr,
}


def ci95(values):
    arr = np.asarray(values, float)
    if len(arr) < 2:
        return float(arr.mean()), 0.0
    h = stats.sem(arr) * stats.t.ppf(0.975, len(arr) - 1)
    return float(arr.mean()), float(h)


def _print_cv_summary(fold_maes, fold_rmses):
    avg_mae = float(np.mean(fold_maes))
    avg_rmse = float(np.mean(fold_rmses))
    mae_ci, mae_h = ci95(fold_maes)
    rmse_ci, rmse_h = ci95(fold_rmses)
    print(f" -> Average Cross-Validation MAE:  {avg_mae:.4f}")
    print(f" -> Average Cross-Validation RMSE: {avg_rmse:.4f}")
    print(f" -> MAE with 95% CI: {mae_ci:.4f} ± {mae_h:.4f}")
    print(f" -> RMSE with 95% CI: {rmse_ci:.4f} ± {rmse_h:.4f}")
    return dict(mae_mean=avg_mae, mae_ci=mae_h, rmse_mean=avg_rmse, rmse_ci=rmse_h)


def run_experiment(R_source, R_target, train_sizes=(0.25, 0.50, 0.75),
                   n_splits=5, hp=None, methods=None, verbose=False):
    hp = hp or HP
    methods = methods or METHODS
    all_users, all_items = R_target.nonzero()
    all_ratings = R_target[all_users, all_items]
    n_r = len(all_ratings)
    results = {}

    for size in train_sizes:
        sample_idx = np.random.choice(n_r, size=int(size * n_r), replace=False)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_splits = list(kf.split(sample_idx))

        for name, fn in methods.items():
            print(f"\n==================== Training Size: {int(size * 100)}% ====================")
            print(name)
            fold_maes, fold_rmses = [], []

            for fold, (tr_idx, va_idx) in enumerate(fold_splits):
                train_ix, val_ix = sample_idx[tr_idx], sample_idx[va_idx]
                R2_train = np.zeros_like(R_target)
                R2_val = np.zeros_like(R_target)
                R2_train[all_users[train_ix], all_items[train_ix]] = all_ratings[train_ix]
                R2_val[all_users[val_ix], all_items[val_ix]] = all_ratings[val_ix]

                mae, rmse = fn(R_source, R2_train, R2_val, hp, verbose=verbose)
                if mae is not None and np.isfinite(mae):
                    fold_maes.append(mae)
                    fold_rmses.append(rmse)
                    print(f"  Fold {fold + 1}/{n_splits}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
                else:
                    print(f"  Fold {fold + 1}/{n_splits}: FAILED")

            if fold_maes:
                if size not in results:
                    results[size] = {}
                results[size][name] = _print_cv_summary(fold_maes, fold_rmses)
            print()

    return results


def results_to_dataframe(results):
    rows = []
    for size, methods in results.items():
        for name, m in methods.items():
            rows.append(dict(
                train_pct=int(size * 100), method=name,
                MAE=f"{m['mae_mean']:.4f} ± {m['mae_ci']:.4f}",
                RMSE=f"{m['rmse_mean']:.4f} ± {m['rmse_ci']:.4f}",
            ))
    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        _vg, _ty = matrix_vg, matrix_ty  # noqa: F821
    except NameError:
        raise RuntimeError("Pehle matrix_vg aur matrix_ty load karo!")

    for name, mat in [("Video_Games", _vg), ("Toys", _ty)]:
        nr = (mat.values != 0).sum()
        sp = (1 - nr / (mat.shape[0] * mat.shape[1])) * 100
        print(f"{name}: users={mat.shape[0]}, items={mat.shape[1]}, "
              f"ratings={nr}, sparsity={sp:.1f}%")

    R_source = _vg.fillna(0).values.astype(float)
    R_target = _ty.fillna(0).values.astype(float)
    results = run_experiment(R_source, R_target, train_sizes=(0.25, 0.50, 0.75))
    df = results_to_dataframe(results)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
    df.to_csv("HNO3_CDR_results.csv", index=False)
