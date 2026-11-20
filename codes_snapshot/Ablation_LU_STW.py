"""LU-STW Ablation v5 + 5-Fold CV — RMSE only."""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

FAST = True
N_FOLDS = 5
F = 20 if FAST else 30
EPOCHS_S = 50 if FAST else 100
EPOCHS_T = 50 if FAST else 100
EPOCHS_FT = 60 if FAST else 150

HP = dict(F=F, ETA=0.01 if FAST else 0.005, ETA_P=0.005, EPOCHS_S=EPOCHS_S,
          EPOCHS_T=EPOCHS_T, EPOCHS_FT=EPOCHS_FT,
          l_uv=0.01, l_bi=0.01, l_uvp=0.006, l_bip=0.01, patience=5, ft_patience=8)

PAPER_ABLATION = [
    dict(name="Full LU-STW",    hungarian=True,  confidence=True,  finetune=True),
    dict(name="w/o Hungarian",  hungarian=False, confidence=True,  finetune=True),
    dict(name="w/o Confidence", hungarian=True,  confidence=False, finetune=True),
    dict(name="w/o Fine-Tune",  hungarian=True,  confidence=True,  finetune=False),
    dict(name="Target Only",    target_only=True),
]


def log(msg):
    print(msg, flush=True)


def eval_rmse(R_hat, R_true):
    us, is_ = R_true.nonzero()
    if len(us) == 0:
        return np.inf
    e = R_true[us, is_] - R_hat[us, is_]
    return float(np.sqrt(np.mean(e ** 2)))


def mf_sgd(R, f, epochs, eta, l_uv, l_bi):
    m, n = R.shape
    U = np.random.rand(m, f) * 0.01
    V = np.random.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = float(R[R > 0].mean()) if (R > 0).any() else 0.0
    us, is_ = R.nonzero(); rs = R[us, is_].astype(float)
    for _ in range(epochs):
        for j in np.random.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            pred = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - pred
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
    return None if not np.isfinite(U).all() else (U, V, bu, bi, mu)


def mf_es(R, R_val, f, epochs, eta, l_uv, l_bi, patience=5, md=1e-4):
    m, n = R.shape
    U = np.random.rand(m, f) * 0.01
    V = np.random.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = float(R[R > 0].mean()) if (R > 0).any() else 0.0
    us, is_ = R.nonzero(); rs = R[us, is_].astype(float)
    best_rmse, wait, bU, bV, bbu, bbi = np.inf, 0, None, None, None, None
    for _ in range(epochs):
        for j in np.random.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            pred = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - pred
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
        rmse = eval_rmse(U @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best_rmse - md:
            best_rmse, wait = rmse, 0
            bU, bV, bbu, bbi = U.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience:
                break
    if bU is None:
        bU, bV, bbu, bbi = U, V, bu, bi
    return None if not np.isfinite(bU).all() else (bU, bV, bbu, bbi, mu)


def fine_tune_es(R_tr, R_val, X0, V, bu, bi, mu, conf_ft, hp, seed,
                 R_target=None, use_ensemble=False):
    X, V, bu, bi = X0.copy(), V.copy(), bu.copy(), bi.copy()
    eta, l_uv, l_bi = hp["ETA_P"], hp["l_uvp"], hp["l_bip"]
    us, is_ = R_tr.nonzero(); rs = R_tr[us, is_].astype(float)
    best_rmse, wait, bX, bV, bbu, bbi = np.inf, 0, None, None, None, None
    for ep in range(hp["EPOCHS_FT"]):
        for j in np.random.RandomState(seed + ep).permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            eu = eta * conf_ft[u]
            pred = mu + bu[u] + bi[i] + X[u] @ V[i]
            e = r - pred
            bu[u] += eu * (e - l_bi * bu[u])
            bi[i] += eu * (e - l_bi * bi[i])
            xo, vo = X[u].copy(), V[i].copy()
            X[u] += eu * (e * vo - l_uv * xo)
            V[i] += eu * (e * xo - l_uv * vo)
        R_ft = X @ V.T + mu + bu[:, None] + bi[None, :]
        if use_ensemble and R_target is not None:
            R_hat = conf_ft[:, None] * R_ft + (1 - conf_ft[:, None]) * R_target
        else:
            R_hat = R_ft
        rmse = eval_rmse(R_hat, R_val)
        if rmse < best_rmse - 1e-4:
            best_rmse, wait = rmse, 0
            bX, bV, bbu, bbi = X.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= hp["ft_patience"]:
                break
    if bX is None:
        bX, bV, bbu, bbi = X, V, bu, bi
    return bX, bV, bbu, bbi


def align_users(U_s, U_t, hungarian=True):
    Un = U_s / (np.linalg.norm(U_s, axis=0, keepdims=True) + 1e-9)
    Vn = U_t / (np.linalg.norm(U_t, axis=0, keepdims=True) + 1e-9)
    C = 1.0 - Un.T @ Vn
    f = C.shape[0]
    si, ti = linear_sum_assignment(C if hungarian else -C)
    P = np.zeros((f, f)); P[ti, si] = 1
    return U_s @ P.T


def user_conf(U_al, U_t):
    num = np.sum(U_al * U_t, axis=1)
    den = np.linalg.norm(U_al, axis=1) * np.linalg.norm(U_t, axis=1) + 1e-9
    sim = np.clip(num / den, 0, 1)
    return np.clip(0.05 + 0.95 * (sim ** 2), 0.05, 1.0)


def predict(U, V, bu, bi, mu):
    return U @ V.T + mu + bu[:, None] + bi[None, :]


def apply_variant(out_s, out_t, R_tr, R_val, hp, cfg, seed):
    if out_s is None or out_t is None:
        return None
    U_t, V_t, bu_t, bi_t, mu_t = out_t
    R_target = predict(U_t, V_t, bu_t, bi_t, mu_t)
    if cfg.get("target_only"):
        return eval_rmse(R_target, R_val)

    U_s, _, _, _, _ = out_s
    U_al = align_users(U_s, U_t, cfg.get("hungarian", True))
    conf = user_conf(U_al, U_t)
    use_conf = cfg.get("confidence", True)

    if use_conf:
        X0 = conf[:, None] * U_al + (1 - conf[:, None]) * U_t
        conf_ft = conf
    else:
        X0 = U_al.copy()
        conf_ft = np.ones(len(conf))

    if not cfg.get("finetune", True):
        return eval_rmse(predict(X0, V_t, bu_t, bi_t, mu_t), R_val)

    X, Vf, bu, bi = fine_tune_es(
        R_tr, R_val, X0, V_t, bu_t.copy(), bi_t.copy(), mu_t, conf_ft, hp, seed + 3,
        R_target=R_target, use_ensemble=use_conf)
    R_ft = predict(X, Vf, bu, bi, mu_t)
    R_hat = conf[:, None] * R_ft + (1 - conf[:, None]) * R_target if use_conf else R_ft
    return eval_rmse(R_hat, R_val)


def run_ablation_cv(R_s, R_t, configs, hp, n_folds=N_FOLDS):
    us, is_ = R_t.nonzero(); rs = R_t[us, is_]
    n_r = len(rs)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    agg = {c["name"]: [] for c in configs}
    t0 = time.time()
    log(f"\n{'='*20} 5-Fold CV (k={n_folds}) — RMSE only {'='*20}")

    for fold, (tri, vai) in enumerate(kf.split(np.arange(n_r))):
        tf = time.time()
        R_tr = np.zeros_like(R_t); R_val = np.zeros_like(R_t)
        R_tr[us[tri], is_[tri]] = rs[tri]
        R_val[us[vai], is_[vai]] = rs[vai]
        seed = 42 + fold * 77
        log(f"  fold {fold+1}/{n_folds} | train={len(tri)} val={len(vai)}")

        np.random.seed(seed)
        out_s = mf_sgd(R_s, hp["F"], hp["EPOCHS_S"], hp["ETA"], hp["l_uv"], hp["l_bi"])
        np.random.seed(seed + 1)
        out_t = mf_es(R_tr, R_val, hp["F"], hp["EPOCHS_T"], hp["ETA"],
                      hp["l_uv"], hp["l_bi"], patience=hp["patience"])

        for cfg in configs:
            rmse = apply_variant(out_s, out_t, R_tr, R_val, hp, cfg, seed)
            if rmse is None:
                continue
            agg[cfg["name"]].append(rmse)
            log(f"    [{cfg['name']}] RMSE={rmse:.4f}")
        log(f"  fold {fold+1} done ({time.time()-tf:.0f}s)")

    rows = []
    for name, rmses in agg.items():
        if rmses:
            rows.append(dict(
                variant=name,
                rmse_mean=round(np.mean(rmses), 4),
                rmse_std=round(np.std(rmses), 4),
            ))
            log(f" -> {name}: RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}")

    log(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    return pd.DataFrame(rows)


def plot_rmse(df, out="ablation_rmse.png"):
    order = [c["name"] for c in PAPER_ABLATION]
    variants = [v for v in order if v in df["variant"].values]
    x = np.arange(len(variants))
    sub = df.set_index("variant").reindex(variants)
    colors = ["#27ae60" if v == "Full LU-STW" else "#3498db" for v in variants]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, sub["rmse_mean"], yerr=sub["rmse_std"], capsize=4, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMSE"); ax.set_title(f"LU-STW Ablation — {N_FOLDS}-Fold CV")
    ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.show()


if __name__ == "__main__":
    log(f"Ablation v5+CV | k={N_FOLDS} | F={F} ep S/T/FT={EPOCHS_S}/{EPOCHS_T}/{EPOCHS_FT}")
    try:
        R_s = matrix_vg.fillna(0).values  # noqa: F821
        R_t = matrix_ty.fillna(0).values  # noqa: F821
    except NameError:
        raise RuntimeError("Pehle matrix_vg aur matrix_ty load karo!")

    np.random.seed(42)
    df = run_ablation_cv(R_s, R_t, PAPER_ABLATION, HP)
    log("\n=== RMSE (5-Fold CV) ===\n" + df.to_string(index=False))
    sub = df.sort_values("rmse_mean")
    full = sub.loc[sub["variant"] == "Full LU-STW", "rmse_mean"]
    tag = "WIN" if len(full) and sub.iloc[0]["variant"] == "Full LU-STW" else "---"
    log(f"[{tag}] Best={sub.iloc[0]['variant']} RMSE={sub.iloc[0]['rmse_mean']:.4f} "
        f"Full={full.values[0]:.4f}" if len(full) else "")
    df.to_csv("ablation_rmse.csv", index=False)
    plot_rmse(df)
    log("Done → ablation_rmse.csv")
