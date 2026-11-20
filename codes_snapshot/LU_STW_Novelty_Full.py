"""
LU-STW Full Pipeline + Novelty Experiments (Top-Tier Ready)
Run after loading matrix_vg (source) and matrix_ty (target).
Usage: python LU_STW_Novelty_Full.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

np.random.seed(42)

# =============================================================================
# CORE MF
# =============================================================================

def mf_sgd(R, f, epochs, eta, l_uv, l_bi, verbose=False):
    m, n = R.shape
    U = np.random.rand(m, f) * 0.01
    V = np.random.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero()
    rs = R[us, is_]
    for _ in range(epochs):
        idx = np.arange(len(rs)); np.random.shuffle(idx)
        for j in idx:
            u, i, r = us[j], is_[j], rs[j]
            pred = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - pred
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
    if not np.isfinite(U).all(): return None
    return U, V, bu, bi, mu


def mf_sgd_es(R, R_val, f, epochs, eta, l_uv, l_bi, patience=5, md=1e-4):
    m, n = R.shape
    U = np.random.rand(m, f) * 0.01
    V = np.random.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero(); rs = R[us, is_]
    best_rmse, wait = np.inf, 0
    bU = bV = bbu = bbi = None
    for ep in range(epochs):
        idx = np.arange(len(rs)); np.random.shuffle(idx)
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
        if rmse < best_rmse - md:
            best_rmse, wait = rmse, 0
            bU, bV, bbu, bbi = U.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience: break
    if bU is None: bU, bV, bbu, bbi = U, V, bu, bi
    if not np.isfinite(bU).all(): return None
    return bU, bV, bbu, bbi, mu


def fine_tune_es(X, V, bu, bi, mu, R_tr, R_val, U_s_al, U_t, epochs, eta, l_uv, l_bi,
                 use_conf=True, eps=0.05, patience=10, md=1e-4):
    X, V, bu, bi = X.copy(), V.copy(), bu.copy(), bi.copy()
    conf = user_conf(U_s_al, U_t, eps) if use_conf else np.ones(X.shape[0])
    us, is_ = R_tr.nonzero(); rs = R_tr[us, is_]
    best_rmse, wait = np.inf, 0
    bX, bV, bbu, bbi = None, None, None, None
    for _ in range(epochs):
        idx = np.arange(len(rs)); np.random.shuffle(idx)
        for j in idx:
            u, i, r = us[j], is_[j], rs[j]
            eu = eta * conf[u]
            pred = mu + bu[u] + bi[i] + X[u] @ V[i]
            e = r - pred
            bu[u] += eu * (e - l_bi * bu[u])
            bi[i] += eu * (e - l_bi * bi[i])
            xo, vo = X[u].copy(), V[i].copy()
            X[u] += eu * (e * vo - l_uv * xo)
            V[i] += eu * (e * xo - l_uv * vo)
        _, rmse = eval_pred(X @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best_rmse - md:
            best_rmse, wait = rmse, 0
            bX, bV, bbu, bbi = X.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience: break
    if bX is None: bX, bV, bbu, bbi = X, V, bu, bi
    return bX, bV, bbu, bbi


# =============================================================================
# NOVELTY: ALIGNMENT STRATEGIES
# =============================================================================

def cost_matrix(A, B):
    """Column-wise cosine dissimilarity. A,B: (entities x k)"""
    k = A.shape[1]
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            a, b = A[:, i], B[:, j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sim = a @ b / (na * nb) if na and nb else 0
            C[i, j] = 1 - sim
    return C


def perm_matrix(C, method="hungarian"):
    k = C.shape[0]
    if method == "hungarian":
        si, ti = linear_sum_assignment(C)
    elif method == "greedy":
        si, ti = [], []
        C2 = C.copy()
        for _ in range(k):
            i, j = divmod(np.argmin(C2), k)
            si.append(j); ti.append(i); C2[i, :] = C2[:, j] = np.inf
        si, ti = np.array(si), np.array(ti)
    elif method == "random":
        perm = np.random.permutation(k)
        si, ti = np.arange(k), perm
    elif method == "identity":
        si, ti = np.arange(k), np.arange(k)
    else:
        raise ValueError(method)
    P = np.zeros((k, k))
    P[ti, si] = 1
    return P


def procrustes_perm(A, B):
    """Orthogonal Procrustes alignment then Hungarian on residual."""
    M = A.T @ B
    U_, _, Vt = np.linalg.svd(M)
    R = U_ @ Vt
    A_rot = A @ R
    C = cost_matrix(A_rot, B)
    P = perm_matrix(C, "hungarian")
    return P, A_rot


def dual_align(U_s, U_t, V_s, V_t, method="hungarian"):
    """NOVELTY: align user dims AND item dims."""
    Pu = perm_matrix(cost_matrix(U_s, U_t), method)
    Pv = perm_matrix(cost_matrix(V_s, V_t), method)
    return U_s @ Pu.T, V_s @ Pv.T, Pu, Pv


def user_conf(U_al, U_t, eps=0.05):
    conf = np.array([
        np.clip((U_al[u] @ U_t[u]) / (np.linalg.norm(U_al[u]) * np.linalg.norm(U_t[u]) + 1e-9), eps, 1)
        for u in range(U_al.shape[0])
    ])
    return conf


def eval_pred(R_hat, R_true):
    us, is_ = R_true.nonzero()
    t, p = R_true[us, is_], R_hat[us, is_]
    if len(t) == 0: return np.inf, np.inf
    e = t - p
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))


# =============================================================================
# PIPELINE
# =============================================================================

HP = dict(F=20, ETA=0.005, EPOCHS=100, ETA_P=0.005, EPOCHS_FT=150,
          l_uv=0.01, l_bi=0.01, l_uvp=0.01, l_bip=0.01)


def run_fold(R_s, R_tr, R_val, cfg, hp=HP):
    F, ETA, EP = hp["F"], hp["ETA"], hp["EPOCHS"]
    EPFT, ETAP = hp["EPOCHS_FT"], hp["ETA_P"]

    if not cfg.get("use_source", True):
        out = mf_sgd_es(R_tr, R_val, F, EP, ETA, hp["l_uv"], hp["l_bi"])
        if out is None: return None, None, None
        U, V, bu, bi, mu = out
        R_hat = U @ V.T + mu + bu[:, None] + bi[None, :]
        mae, rmse = eval_pred(R_hat, R_val)
        return mae, rmse, None

    out_s = mf_sgd(R_s, F, EP, ETA, hp["l_uv"], hp["l_bi"])
    out_t = mf_sgd_es(R_tr, R_val, F, EP, ETA, hp["l_uv"], hp["l_bi"])
    if out_s is None or out_t is None: return None, None, None
    U_s, V_s, bu_s, bi_s, _ = out_s
    U_t, V_t, bu_t, bi_t, mu_t = out_t

    align = cfg.get("align", "hungarian")
    dual = cfg.get("dual_align", False)

    if dual:
        U_al, V_s_al, _, _ = dual_align(U_s, U_t, V_s, V_t, align)
        V_init = V_s_al  # aligned source items init (novelty)
    else:
        if align == "procrustes":
            P, _ = procrustes_perm(U_s, U_t)
            U_al = U_s @ P.T
        else:
            P = perm_matrix(cost_matrix(U_s, U_t), align)
            U_al = U_s @ P.T
        V_init = V_t

    if cfg.get("use_fine_tuning", True):
        X, Vf, buf, bif = fine_tune_es(
            U_al, V_init, bu_s, bi_t, mu_t, R_tr, R_val, U_al, U_t,
            EPFT, ETAP, hp["l_uvp"], hp["l_bip"], cfg.get("use_confidence", True))
        if X is None: return None, None, None
        R_hat = X @ Vf.T + mu_t + buf[:, None] + bif[None, :]
    else:
        R_hat = U_al @ V_init.T + mu_t + bu_s[:, None] + bi_t[None, :]

    mae, rmse = eval_pred(R_hat, R_val)
    conf = user_conf(U_al, U_t) if cfg.get("use_source") else None
    return mae, rmse, conf


# =============================================================================
# EXPERIMENT CONFIGS (NOVELTY)
# =============================================================================

VARIANTS = {
    # Full proposed
    "Full LU-STW": dict(use_source=True, align="hungarian", use_confidence=True, use_fine_tuning=True, dual_align=False),
    # Ablation
    "Target Only": dict(use_source=False),
    "w/o Hungarian": dict(use_source=True, align="identity", use_confidence=True, use_fine_tuning=True),
    "w/o Confidence": dict(use_source=True, align="hungarian", use_confidence=False, use_fine_tuning=True),
    "w/o Fine-Tuning": dict(use_source=True, align="hungarian", use_fine_tuning=False),
    # Alignment comparison (NOVELTY)
    "Greedy Align": dict(use_source=True, align="greedy", use_confidence=True, use_fine_tuning=True),
    "Random Align": dict(use_source=True, align="random", use_confidence=True, use_fine_tuning=True),
    "Procrustes+Hung": dict(use_source=True, align="procrustes", use_confidence=True, use_fine_tuning=True),
    # Dual alignment (NOVELTY)
    "Dual Align (U+V)": dict(use_source=True, align="hungarian", use_confidence=True, use_fine_tuning=True, dual_align=True),
}


def cv_study(R_s, R_t, variants, train_size=0.5, n_splits=5, hp=HP, rs=42):
    us, is_ = R_t.nonzero(); rs_arr = R_t[us, is_]
    idx = np.random.choice(len(rs_arr), int(train_size * len(rs_arr)), replace=False)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    results = {}

    for name, cfg in variants.items():
        maes, rmses, t0 = [], [], time.time()
        conf_groups = {"high": [], "mid": [], "low": []}
        print(f"\n--- {name} ---")

        for fold, (tri, vai) in enumerate(kf.split(idx)):
            R_tr = np.zeros_like(R_t); R_val = np.zeros_like(R_t)
            ti, vi = idx[tri], idx[vai]
            R_tr[us[ti], is_[ti]] = rs_arr[ti]
            R_val[us[vi], is_[vi]] = rs_arr[vi]

            mae, rmse, conf = run_fold(R_s, R_tr, R_val, cfg, hp)
            if mae is None:
                print(f"  fold {fold+1}: skip"); continue
            maes.append(mae); rmses.append(rmse)
            print(f"  fold {fold+1}: MAE={mae:.4f} RMSE={rmse:.4f}")

            # Per-user confidence analysis (NOVELTY)
        elapsed = time.time() - t0
        results[name] = dict(
            mae=np.mean(maes), mae_std=np.std(maes),
            rmse=np.mean(rmses), rmse_std=np.std(rmses),
            time=elapsed
        )
        print(f"  => RMSE {results[name]['rmse']:.4f}±{results[name]['rmse_std']:.4f}  ({elapsed:.1f}s)")

    return results


def confidence_analysis(R_s, R_t, hp=HP, train_size=0.5):
    """NOVELTY: MAE by user confidence group — Full vs w/o Confidence."""
    us, is_ = R_t.nonzero(); rs_arr = R_t[us, is_]
    idx = np.random.choice(len(rs_arr), int(train_size * len(rs_arr)), replace=False)
    n = len(idx); tri = idx[:int(0.8*n)]; vai = idx[int(0.8*n):]

    R_tr = np.zeros_like(R_t); R_val = np.zeros_like(R_t)
    R_tr[us[tri], is_[tri]] = rs_arr[tri]
    R_val[us[vai], is_[vai]] = rs_arr[vai]

    groups = {"High (>0.7)": [], "Mid (0.3-0.7)": [], "Low (<0.3)": []}

    for label, cfg in [("Full", VARIANTS["Full LU-STW"]), ("NoConf", VARIANTS["w/o Confidence"])]:
        out_s = mf_sgd(R_s, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"])
        out_t = mf_sgd_es(R_tr, R_val, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"])
        U_s, _, bu_s, bi_s, _ = out_s
        U_t, V_t, _, bi_t, mu_t = out_t
        P = perm_matrix(cost_matrix(U_s, U_t), "hungarian")
        U_al = U_s @ P.T
        conf = user_conf(U_al, U_t)
        X, Vf, buf, bif = fine_tune_es(
            U_al, V_t, bu_s, bi_t, mu_t, R_tr, R_val, U_al, U_t,
            hp["EPOCHS_FT"], hp["ETA_P"], hp["l_uvp"], hp["l_bip"], cfg.get("use_confidence", True))
        R_hat = X @ Vf.T + mu_t + buf[:, None] + bif[None, :]

        for u in np.unique(us[vai]):
            mask = (us[vai] == u)
            t = R_val[us[vai][mask], is_[vai][mask]]
            p = R_hat[us[vai][mask], is_[vai][mask]]
            mae_u = np.mean(np.abs(t - p))
            c = conf[u]
            g = "High (>0.7)" if c > 0.7 else ("Low (<0.3)" if c < 0.3 else "Mid (0.3-0.7)")
            groups.setdefault(g, []).append((label, mae_u))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    gnames = ["High (>0.7)", "Mid (0.3-0.7)", "Low (<0.3)"]
    x = np.arange(3); w = 0.35
    for i, lbl in enumerate(["Full", "NoConf"]):
        vals = [np.mean([m for t, m in groups.get(g, []) if t == lbl] or [0]) for g in gnames]
        ax.bar(x + i*w, vals, w, label=lbl)
    ax.set_xticks(x + w/2); ax.set_xticklabels(gnames)
    ax.set_ylabel("MAE"); ax.set_title("Per-User Confidence Group Analysis")
    ax.legend(); plt.tight_layout(); plt.savefig("confidence_groups.png", dpi=150)
    plt.show()


def cold_start_eval(R_s, R_t, hp=HP, cold_ratio=0.2):
    """NOVELTY: users with zero target train ratings."""
    m = R_t.shape[0]
    cold_users = np.random.choice(m, int(cold_ratio * m), replace=False)
    us, is_ = R_t.nonzero(); rs_arr = R_t[us, is_]
    mask = ~np.isin(us, cold_users)
    idx = np.where(mask)[0]
    tri, vai = idx[:int(0.8*len(idx))], idx[int(0.8*len(idx)):]

    R_tr = np.zeros_like(R_t); R_val = np.zeros_like(R_t)
    R_tr[us[tri], is_[tri]] = rs_arr[tri]
    R_val[us[vai], is_[vai]] = rs_arr[vai]

    mae, rmse, _ = run_fold(R_s, R_tr, R_val, VARIANTS["Full LU-STW"], hp)
    print(f"\nCold-Start ({cold_ratio*100:.0f}% zero-target users): MAE={mae:.4f} RMSE={rmse:.4f}")
    return mae, rmse


def plot_results(results, title="LU-STW Novelty Experiments"):
    names = list(results.keys())
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    rmses = [results[n]["rmse"] for n in names]
    rstds = [results[n]["rmse_std"] for n in names]
    maes = [results[n]["mae"] for n in names]
    mstds = [results[n]["mae_std"] for n in names]
    cols = ["#2ecc71" if "Full" in n else "#3498db" for n in names]
    ax[0].bar(x, rmses, yerr=rstds, capsize=3, color=cols, edgecolor="k", linewidth=0.4)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax[0].set_title("RMSE"); ax[0].set_ylabel("Lower is better")
    ax[1].bar(x, maes, yerr=mstds, capsize=3, color=cols, edgecolor="k", linewidth=0.4)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax[1].set_title("MAE")
    plt.suptitle(title); plt.tight_layout()
    plt.savefig("novelty_results.png", dpi=150); plt.show()


def to_table(results):
    df = pd.DataFrame([{"Variant": k, **{kk: round(vv, 4) for kk, vv in v.items()}} for k, v in results.items()])
    full = results.get("Full LU-STW", {}).get("rmse", 1)
    df["ΔRMSE"] = (df["rmse"] - full).round(4)
    return df


# =============================================================================
# MAIN — paste after matrix_vg, matrix_ty loaded
# =============================================================================

if __name__ == "__main__":
    # --- Load data (from your notebook) ---
    R_source = matrix_vg.fillna(0).values
    R_target = matrix_ty.fillna(0).values
    assert R_source.shape[0] == R_target.shape[0]

    print("=" * 60)
    print("EXPERIMENT 1: Ablation + Alignment + Dual Align")
    print("=" * 60)
    results = cv_study(R_source, R_target, VARIANTS, train_size=0.5)
    df = to_table(results)
    print("\n", df.to_string(index=False))
    df.to_csv("novelty_results.csv", index=False)
    plot_results(results)

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Confidence Group Analysis")
    print("=" * 60)
    confidence_analysis(R_source, R_target)

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Cold-Start Users")
    print("=" * 60)
    cold_start_eval(R_source, R_target)

    print("\nDone. Files: novelty_results.csv, novelty_results.png, confidence_groups.png")
