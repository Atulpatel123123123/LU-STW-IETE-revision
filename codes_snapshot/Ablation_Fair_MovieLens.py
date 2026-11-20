# LU-STW Ablation — MovieLens (50% train, MAE + dMAE; Full = fixed baseline)
# Prerequisite: romance_matrix (source), drama_matrix (target) loaded
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

FOLDS = 5
TRAIN_FRAC = 0.50
SEED = 42
# MovieLens @ 50% — reported Full LU-STW MAE (fixed baseline for dMAE)
FULL_MAE_BASELINE = 0.7046

HP = dict(F=20, ETA=0.005, ETA_P=0.005, EPOCHS=100, EPOCHS_FT=150,
          l_uv=0.005, l_bi=0.01, l_uvp=0.005, l_bip=0.01)
CONF_MIN = 0.10

VARIANTS = [
    "Target Only",
    "w/o Hungarian Alignment",
    "w/o Confidence Weighting",
    "w/o Fine-Tuning",
]


def mf_sgd(R, f, epochs, eta, l_uv, l_bi, seed=0):
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U = rng.rand(m, f) * 0.01
    V = rng.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero(); rs = R[us, is_]
    for _ in range(epochs):
        for j in rng.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            p = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - p
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
    if not np.isfinite(U).all():
        return None
    return U, V, bu, bi, mu


def mf_sgd_es(R, R_val, f, epochs, eta, l_uv, l_bi, seed=0, patience=5, md=1e-4):
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U = rng.rand(m, f) * 0.01
    V = rng.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero(); rs = R[us, is_]
    best_rmse, wait = np.inf, 0
    bU = bV = bbu = bbi = None
    for _ in range(epochs):
        for j in rng.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            p = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - p
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
        _, rmse = _eval_mae_rmse(U @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best_rmse - md:
            best_rmse, wait = rmse, 0
            bU, bV, bbu, bbi = U.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience:
                break
    if bU is None:
        bU, bV, bbu, bbi = U, V, bu, bi
    if not np.isfinite(bU).all():
        return None
    return bU, bV, bbu, bbi, mu


def fine_tune(X0, V0, bu0, bi0, mu, R_tr, R_val, U_ref, U_t,
              epochs, eta, l_uv, l_bi, use_conf=True, seed=0, patience=10, md=1e-4):
    rng = np.random.RandomState(seed)
    X, V = X0.copy(), V0.copy()
    bu, bi = bu0.copy(), bi0.copy()
    conf = user_conf(U_ref, U_t, CONF_MIN) if use_conf else np.ones(X.shape[0])
    us, is_ = R_tr.nonzero(); rs = R_tr[us, is_]
    best_rmse, wait = np.inf, 0
    bX = bV = bbu = bbi = None
    for _ in range(epochs):
        for j in rng.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            eu = eta * conf[u]
            p = mu + bu[u] + bi[i] + X[u] @ V[i]
            e = r - p
            bu[u] += eu * (e - l_bi * bu[u])
            bi[i] += eu * (e - l_bi * bi[i])
            xo, vo = X[u].copy(), V[i].copy()
            X[u] += eu * (e * vo - l_uv * xo)
            V[i] += eu * (e * xo - l_uv * vo)
        _, rmse = _eval_mae_rmse(X @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best_rmse - md:
            best_rmse, wait = rmse, 0
            bX, bV, bbu, bbi = X.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience:
                break
    if bX is None:
        bX, bV, bbu, bbi = X, V, bu, bi
    return bX, bV, bbu, bbi


def col_norm(U):
    n = np.linalg.norm(U, axis=0, keepdims=True)
    n[n == 0] = 1
    return U / n


def cost_mat(U_s, U_t):
    Us, Ut = col_norm(U_s), col_norm(U_t)
    k = Us.shape[1]
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            C[i, j] = 1 - Us[:, i] @ Ut[:, j]
    return C


def hungarian_P(U_s, U_t):
    si, ti = linear_sum_assignment(cost_mat(U_s, U_t))
    k = U_s.shape[1]
    P = np.zeros((k, k))
    P[ti, si] = 1
    return P


def user_conf(U_al, U_t, eps):
    m = U_al.shape[0]
    c = np.zeros(m)
    for u in range(m):
        a, b = U_al[u], U_t[u]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        c[u] = (a @ b) / (na * nb) if na and nb else 0
    return np.clip(c, eps, 1.0)


def _eval_mae_rmse(R_hat, R_true):
    us, is_ = R_true.nonzero()
    t, p = R_true[us, is_], R_hat[us, is_]
    if len(t) == 0:
        return np.inf, np.inf
    e = t - p
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))


def eval_mae(R_hat, R_true):
    mae, _ = _eval_mae_rmse(R_hat, R_true)
    return mae


def run_variant(base, name, R_tr, R_val, hp):
    U_s = base["U_s"]
    U_t, V_t, bu_t, bi_t, mu_t = base["U_t"], base["V_t"], base["bu_t"], base["bi_t"], base["mu_t"]
    seed = base["seed"]

    if name == "Target Only":
        return eval_mae(U_t @ V_t.T + mu_t + bu_t[:, None] + bi_t[None, :], R_val)

    if name == "w/o Hungarian Alignment":
        U_al = U_s.copy()
        U_ref = U_s.copy()
    else:
        P = hungarian_P(U_s, U_t)
        U_al = U_s @ P.T
        U_ref = U_al

    X0 = 0.6 * U_al + 0.4 * U_t

    if name == "w/o Fine-Tuning":
        return eval_mae(X0 @ V_t.T + mu_t + bu_t[:, None] + bi_t[None, :], R_val)

    use_conf = name != "w/o Confidence Weighting"
    X, Vf, bu, bi = fine_tune(
        X0, V_t, bu_t.copy(), bi_t.copy(), mu_t, R_tr, R_val,
        U_ref, U_t, hp["EPOCHS_FT"], hp["ETA_P"], hp["l_uvp"], hp["l_bip"],
        use_conf=use_conf, seed=seed + 7)
    return eval_mae(X @ Vf.T + mu_t + bu[:, None] + bi[None, :], R_val)


def train_base(R_s, R_tr, R_val, hp, seed):
    out_s = mf_sgd(R_s, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"], seed)
    out_t = mf_sgd_es(R_tr, R_val, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"], seed + 1)
    if out_s is None or out_t is None:
        return None
    U_s, V_s, bu_s, bi_s, _ = out_s
    U_t, V_t, bu_t, bi_t, mu_t = out_t
    return dict(U_s=U_s, V_s=V_s, bu_s=bu_s, bi_s=bi_s,
                U_t=U_t, V_t=V_t, bu_t=bu_t, bi_t=bi_t, mu_t=mu_t, seed=seed)


def ablation_cv(R_s, R_t, hp, train_frac):
    us, is_ = R_t.nonzero(); rs = R_t[us, is_]
    idx = np.random.RandomState(SEED).choice(len(rs), int(train_frac * len(rs)), replace=False)
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    results = {v: [] for v in VARIANTS}

    for fold, (tri, vai) in enumerate(kf.split(idx)):
        R_tr = np.zeros_like(R_t); R_va = np.zeros_like(R_t)
        R_tr[us[idx[tri]], is_[idx[tri]]] = rs[idx[tri]]
        R_va[us[idx[vai]], is_[idx[vai]]] = rs[idx[vai]]
        base = train_base(R_s, R_tr, R_va, hp, seed=SEED + fold * 100)
        if base is None:
            print(f"  fold {fold + 1}: skip (instability)")
            continue
        for v in VARIANTS:
            results[v].append(run_variant(base, v, R_tr, R_va, hp))
        print(f"  fold {fold + 1}: "
              + "  ".join(f"{v}={results[v][-1]:.4f}" for v in VARIANTS))

    return {v: dict(
        mae_mean=np.mean(d), mae_std=np.std(d, ddof=1) if len(d) > 1 else 0.0,
    ) for v, d in results.items() if d}


if __name__ == "__main__":
    try:
        R_source = romance_matrix.fillna(0).values  # noqa: F821
        R_target = drama_matrix.fillna(0).values  # noqa: F821
    except NameError:
        raise RuntimeError("Pehle romance_matrix aur drama_matrix load karo!")

    assert R_source.shape[0] == R_target.shape[0]
    print(f"MovieLens | Source Romance {R_source.shape} | Target Drama {R_target.shape}\n",
          flush=True)

    res = ablation_cv(R_source, R_target, HP, TRAIN_FRAC)
    rows = []
    for v in VARIANTS:
        r = res[v]
        dmae = r["mae_mean"] - FULL_MAE_BASELINE
        rows.append(dict(
            Variant=v,
            MAE=f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}",
            dMAE=round(dmae, 4),
        ))
        print(f"  {v}: MAE={r['mae_mean']:.4f}±{r['mae_std']:.4f}  "
              f"dMAE={dmae:+.4f} (vs Full={FULL_MAE_BASELINE})",
              flush=True)

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False), flush=True)
    df.to_csv("ablation_fair_movielens.csv", index=False)

    x = np.arange(len(VARIANTS))
    fig, ax = plt.subplots(figsize=(8, 5))
    dmae = [res[v]["mae_mean"] - FULL_MAE_BASELINE for v in VARIANTS]
    ax.bar(x, dmae, color="#e74c3c")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"dMAE vs Full LU-STW ({FULL_MAE_BASELINE})")
    ax.set_title("MovieLens Ablation — dMAE")
    plt.tight_layout()
    plt.savefig("ablation_fair_movielens.png", dpi=150)
    plt.show()
    print("Saved: ablation_fair_movielens.csv, ablation_fair_movielens.png", flush=True)
