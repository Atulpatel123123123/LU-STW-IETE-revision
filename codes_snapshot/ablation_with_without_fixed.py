# =============================================================================
# LU-STW — With/Without ablation (FIXED)
# Prerequisite: matrix_vg, matrix_ty loaded
# =============================================================================
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

FOLDS, SEED = 5, 42
TRAIN_SIZES = [0.25, 0.50, 0.75]
HP = dict(F=20, ETA=0.005, ETA_P=0.008, EPOCHS=100, EPOCHS_S=150, EPOCHS_FT=200,
          l_uv=0.01, l_bi=0.01, l_uvp=0.008, l_bip=0.01)
CONF_MIN = 0.05

VARIANTS = {
    "1. Target Only":              dict(h=False, c=False, ft=False),
    "2. + Hungarian only":         dict(h=True,  c=False, ft=False),
    "3. + H + Fine-Tuning":        dict(h=True,  c=False, ft=True),
    "4. Full (H + Conf + FT)":     dict(h=True,  c=True,  ft=True),
    "5. w/o Hungarian (C + FT)":   dict(h=False, c=True,  ft=True),
}

PAIRWISE = [
    ("Hungarian",  "3. + H + Fine-Tuning",      "5. w/o Hungarian (C + FT)"),
    ("Confidence", "4. Full (H + Conf + FT)",   "3. + H + Fine-Tuning"),
    ("Fine-Tuning","4. Full (H + Conf + FT)",   "2. + Hungarian only"),
]


def mf_sgd(R, f, epochs, eta, l_uv, l_bi, seed=0):
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U, V = rng.rand(m, f) * 0.01, rng.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero(); rs = R[us, is_]
    for _ in range(epochs):
        for j in rng.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            e = r - (mu + bu[u] + bi[i] + U[u] @ V[i])
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
    return None if not np.isfinite(U).all() else (U, V, bu, bi, mu)


def mf_sgd_es(R, R_val, f, epochs, eta, l_uv, l_bi, seed=0, patience=5, md=1e-4):
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U, V = rng.rand(m, f) * 0.01, rng.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero(); rs = R[us, is_]
    best, wait, st = np.inf, 0, None
    for _ in range(epochs):
        for j in rng.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            e = r - (mu + bu[u] + bi[i] + U[u] @ V[i])
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
        _, rmse = eval_pred(U @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best - md:
            best, wait, st = rmse, 0, (U.copy(), V.copy(), bu.copy(), bi.copy())
        else:
            wait += 1
            if wait >= patience:
                break
    if st is None:
        st = (U, V, bu, bi)
    return None if not np.isfinite(st[0]).all() else (*st, mu)


def fine_tune(X0, V0, bu0, bi0, mu, R_tr, R_val, epochs, eta, l_uv, l_bi,
              seed=0, patience=12, md=1e-4):
    """Uniform LR — confidence only in init (not LR)."""
    rng = np.random.RandomState(seed)
    X, V = X0.copy(), V0.copy()
    bu, bi = bu0.copy(), bi0.copy()
    us, is_ = R_tr.nonzero(); rs = R_tr[us, is_]
    best, wait, st = np.inf, 0, None
    for _ in range(epochs):
        for j in rng.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            e = r - (mu + bu[u] + bi[i] + X[u] @ V[i])
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            xo, vo = X[u].copy(), V[i].copy()
            X[u] += eta * (e * vo - l_uv * xo)
            V[i] += eta * (e * xo - l_uv * vo)
        _, rmse = eval_pred(X @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best - md:
            best, wait, st = rmse, 0, (X.copy(), V.copy(), bu.copy(), bi.copy())
        else:
            wait += 1
            if wait >= patience:
                break
    return st if st else (X, V, bu, bi)


def cost_mat(U_s, U_t):
    k = U_s.shape[1]
    Us = U_s / (np.linalg.norm(U_s, axis=0, keepdims=True) + 1e-9)
    Ut = U_t / (np.linalg.norm(U_t, axis=0, keepdims=True) + 1e-9)
    return np.array([[1 - Us[:, i] @ Ut[:, j] for j in range(k)] for i in range(k)])


def hungarian_P(U_s, U_t):
    si, ti = linear_sum_assignment(cost_mat(U_s, U_t))
    P = np.zeros((U_s.shape[1], U_s.shape[1]))
    P[ti, si] = 1
    return P


def user_conf(U_a, U_t, eps):
    c = np.array([
        (U_a[u] @ U_t[u]) / (np.linalg.norm(U_a[u]) * np.linalg.norm(U_t[u]) + 1e-9)
        for u in range(U_a.shape[0])
    ])
    return np.clip(c, eps, 1.0)


def make_init(U_s, U_t, use_h, use_c):
    """Confidence → init blend; no confidence → aligned or target init."""
    if use_h:
        U_al = U_s @ hungarian_P(U_s, U_t).T
        if use_c:
            conf = user_conf(U_al, U_t, CONF_MIN)
            return np.array([conf[u] * U_al[u] + (1 - conf[u]) * U_t[u] for u in range(len(conf))])
        return U_al.copy()
    if use_c:
        conf = user_conf(U_s, U_t, CONF_MIN)
        return np.array([conf[u] * U_s[u] + (1 - conf[u]) * U_t[u] for u in range(len(conf))])
    return U_t.copy()


def eval_pred(R_hat, R_true):
    us, is_ = R_true.nonzero()
    t, p = R_true[us, is_], R_hat[us, is_]
    if len(t) == 0:
        return np.inf, np.inf
    e = t - p
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))


def run_variant(base, cfg, R_tr, R_val, hp):
    U_s, bu_s = base["U_s"], base["bu_s"]
    U_t, V_t = base["U_t"], base["V_t"]
    bu_t, bi_t, mu_t = base["bu_t"], base["bi_t"], base["mu_t"]
    h, c, ft = cfg["h"], cfg["c"], cfg["ft"]

    if not h and not ft:
        return eval_pred(U_t @ V_t.T + mu_t + bu_t[:, None] + bi_t[None, :], R_val)

    if not ft:
        U_al = U_s @ hungarian_P(U_s, U_t).T if h else U_s
        return eval_pred(U_al @ V_t.T + mu_t + bu_s[:, None] + bi_t[None, :], R_val)

    X0 = make_init(U_s, U_t, h, c)
    X, Vf, bu, bi = fine_tune(
        X0, V_t, bu_s.copy(), bi_t.copy(), mu_t, R_tr, R_val,
        hp["EPOCHS_FT"], hp["ETA_P"], hp["l_uvp"], hp["l_bip"], seed=base["seed"] + 7)
    return eval_pred(X @ Vf.T + mu_t + bu[:, None] + bi[None, :], R_val)


def train_base(R_s, R_tr, R_val, hp, seed):
    np.random.seed(seed)
    out_s = mf_sgd(R_s, hp["F"], hp["EPOCHS_S"], hp["ETA"], hp["l_uv"], hp["l_bi"], seed)
    out_t = mf_sgd_es(R_tr, R_val, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"], seed + 1)
    if out_s is None or out_t is None:
        return None
    U_s, _, bu_s, bi_s, _ = out_s
    U_t, V_t, bu_t, bi_t, mu_t = out_t
    return dict(U_s=U_s, bu_s=bu_s, U_t=U_t, V_t=V_t, bu_t=bu_t, bi_t=bi_t, mu_t=mu_t, seed=seed)


def ablation_cv(R_s, R_t, hp, train_size):
    us, is_ = R_t.nonzero(); rs = R_t[us, is_]
    idx = np.random.RandomState(SEED).choice(len(rs), int(train_size * len(rs)), replace=False)
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    res = {v: {"mae": [], "rmse": []} for v in VARIANTS}

    for fold, (tri, vai) in enumerate(kf.split(idx)):
        R_tr, R_va = np.zeros_like(R_t), np.zeros_like(R_t)
        R_tr[us[idx[tri]], is_[idx[tri]]] = rs[idx[tri]]
        R_va[us[idx[vai]], is_[idx[vai]]] = rs[idx[vai]]
        base = train_base(R_s, R_tr, R_va, hp, SEED + fold * 100)
        if base is None:
            print(f"  fold {fold+1}: skip")
            continue
        for name, cfg in VARIANTS.items():
            mae, rmse = run_variant(base, cfg, R_tr, R_va, hp)
            res[name]["mae"].append(mae)
            res[name]["rmse"].append(rmse)
        print(
            f"  fold {fold+1}: Full={res['4. Full (H + Conf + FT)']['rmse'][-1]:.4f} "
            f"H+FT={res['3. + H + Fine-Tuning']['rmse'][-1]:.4f} "
            f"w/oH={res['5. w/o Hungarian (C + FT)']['rmse'][-1]:.4f} "
            f"Target={res['1. Target Only']['rmse'][-1]:.4f}"
        )
    return {
        v: dict(rmse_m=np.mean(d["rmse"]), rmse_s=np.std(d["rmse"]),
                mae_m=np.mean(d["mae"]), mae_s=np.std(d["mae"]))
        for v, d in res.items() if d["rmse"]
    }


def pairwise_rows(all_res, ts):
    r = all_res[ts]
    rows = []
    for comp, with_n, without_n in PAIRWISE:
        rw, rwo = r[with_n]["rmse_m"], r[without_n]["rmse_m"]
        rows.append(dict(
            Train=f"{int(ts*100)}%", Component=comp,
            With=with_n, Without=without_n,
            With_RMSE=round(rw, 4), Without_RMSE=round(rwo, 4),
            Gain=round(rwo - rw, 4),
        ))
    full = r["4. Full (H + Conf + FT)"]["rmse_m"]
    tgt = r["1. Target Only"]["rmse_m"]
    rows.append(dict(
        Train=f"{int(ts*100)}%", Component="Full vs Target",
        With="4. Full", Without="1. Target Only",
        With_RMSE=round(full, 4), Without_RMSE=round(tgt, 4),
        Gain=round(tgt - full, 4),
    ))
    return rows


if __name__ == "__main__":
    R_source = matrix_vg.fillna(0).values
    R_target = matrix_ty.fillna(0).values
    print(f"Source {R_source.shape} | Target {R_target.shape}\n")

    all_res, detail_rows, pair_rows = {}, [], []
    for ts in TRAIN_SIZES:
        print(f"=== {int(ts*100)}% ===")
        all_res[ts] = ablation_cv(R_source, R_target, HP, ts)
        pair_rows.extend(pairwise_rows(all_res, ts))
        for name, cfg in VARIANTS.items():
            r = all_res[ts][name]
            detail_rows.append(dict(
                Train=int(ts * 100), Variant=name,
                Hungarian=cfg["h"], Confidence=cfg["c"], FineTuning=cfg["ft"],
                RMSE=f"{r['rmse_m']:.4f}±{r['rmse_s']:.4f}",
                MAE=f"{r['mae_m']:.4f}±{r['mae_s']:.4f}",
            ))
            print(f"  {name}: RMSE={r['rmse_m']:.4f}")

    df_pair = pd.DataFrame(pair_rows)
    df_detail = pd.DataFrame(detail_rows)
    print("\n=== WITH vs WITHOUT ===")
    print(df_pair.to_string(index=False))
    df_pair.to_csv("ablation_with_without_fixed.csv", index=False)
    df_detail.to_csv("ablation_all_variants_fixed.csv", index=False)

    sub = df_pair[df_pair["Train"] == "50%"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w / 2, sub["With_RMSE"], w, label="With", color="#2ecc71")
    ax.bar(x + w / 2, sub["Without_RMSE"], w, label="Without", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Component"], rotation=15, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("With vs Without @ 50% (fixed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("with_without_fixed.png", dpi=150)
    plt.show()
