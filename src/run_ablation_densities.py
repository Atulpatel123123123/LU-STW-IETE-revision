"""
MovieLens ablation at 25% and 75% (same protocol as Codes/Ablation_Fair_MovieLens.py).
Paper Table 6 already has 50%.
"""
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

FOLDS = 5
SEED = 42
HP = dict(F=20, ETA=0.005, ETA_P=0.005, EPOCHS=100, EPOCHS_FT=150,
          l_uv=0.005, l_bi=0.01, l_uvp=0.005, l_bip=0.01)
CONF_MIN = 0.10
VARIANTS = [
    "Target Only",
    "w/o Hungarian Alignment",
    "w/o Confidence Weighting",
    "w/o Fine-Tuning",
    "LU-STW",
]


def mf_sgd(R, f, epochs, eta, l_uv, l_bi, seed=0):
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U = rng.rand(m, f) * 0.01
    V = rng.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero()
    rs = R[us, is_]
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
    return U, V, bu, bi, mu


def eval_mae_rmse(R_hat, R_true):
    us, is_ = R_true.nonzero()
    e = R_true[us, is_] - R_hat[us, is_]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def mf_sgd_es(R, R_val, f, epochs, eta, l_uv, l_bi, seed=0, patience=5, md=1e-4):
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U = rng.rand(m, f) * 0.01
    V = rng.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero()
    rs = R[us, is_]
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
        _, rmse = eval_mae_rmse(U @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best_rmse - md:
            best_rmse, wait = rmse, 0
            bU, bV, bbu, bbi = U.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience:
                break
    if bU is None:
        bU, bV, bbu, bbi = U, V, bu, bi
    return bU, bV, bbu, bbi, mu


def user_conf(U_al, U_t, eps):
    c = np.zeros(U_al.shape[0])
    for u in range(U_al.shape[0]):
        a, b = U_al[u], U_t[u]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        c[u] = (a @ b) / (na * nb) if na and nb else 0.0
    return np.clip(c, eps, 1.0)


def hungarian_P(U_s, U_t):
    Us = U_s / np.maximum(np.linalg.norm(U_s, axis=0, keepdims=True), 1e-12)
    Ut = U_t / np.maximum(np.linalg.norm(U_t, axis=0, keepdims=True), 1e-12)
    C = 1.0 - Us.T @ Ut
    si, ti = linear_sum_assignment(C)
    P = np.zeros((U_s.shape[1], U_s.shape[1]))
    P[ti, si] = 1.0
    return P


def fine_tune(X0, V0, bu0, bi0, mu, R_tr, R_val, U_ref, U_t,
              epochs, eta, l_uv, l_bi, use_conf=True, seed=0, patience=10, md=1e-4):
    rng = np.random.RandomState(seed)
    X, V = X0.copy(), V0.copy()
    bu, bi = bu0.copy(), bi0.copy()
    conf = user_conf(U_ref, U_t, CONF_MIN) if use_conf else np.ones(X.shape[0])
    us, is_ = R_tr.nonzero()
    rs = R_tr[us, is_]
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
        _, rmse = eval_mae_rmse(X @ V.T + mu + bu[:, None] + bi[None, :], R_val)
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


def run_variant(base, name, R_tr, R_val, hp):
    U_s = base["U_s"]
    U_t, V_t, bu_t, bi_t, mu_t = base["U_t"], base["V_t"], base["bu_t"], base["bi_t"], base["mu_t"]
    seed = base["seed"]
    if name == "Target Only":
        mae, _ = eval_mae_rmse(U_t @ V_t.T + mu_t + bu_t[:, None] + bi_t[None, :], R_val)
        return mae
    if name == "w/o Hungarian Alignment":
        U_al = U_s.copy()
        U_ref = U_s.copy()
    else:
        P = hungarian_P(U_s, U_t)
        U_al = U_s @ P.T
        U_ref = U_al
    X0 = 0.6 * U_al + 0.4 * U_t
    if name == "w/o Fine-Tuning":
        mae, _ = eval_mae_rmse(X0 @ V_t.T + mu_t + bu_t[:, None] + bi_t[None, :], R_val)
        return mae
    use_conf = name != "w/o Confidence Weighting"
    X, Vf, bu, bi = fine_tune(
        X0, V_t, bu_t.copy(), bi_t.copy(), mu_t, R_tr, R_val, U_ref, U_t,
        hp["EPOCHS_FT"], hp["ETA_P"], hp["l_uvp"], hp["l_bip"],
        use_conf=use_conf, seed=seed + 7)
    mae, _ = eval_mae_rmse(X @ Vf.T + mu_t + bu[:, None] + bi[None, :], R_val)
    return mae


def ablation_cv(R_s, R_t, train_frac):
    us, is_ = R_t.nonzero()
    rs = R_t[us, is_]
    idx = np.random.RandomState(SEED).choice(len(rs), int(train_frac * len(rs)), replace=False)
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    results = {v: [] for v in VARIANTS}
    for fold, (tri, vai) in enumerate(kf.split(idx)):
        R_tr = np.zeros_like(R_t)
        R_va = np.zeros_like(R_t)
        R_tr[us[idx[tri]], is_[idx[tri]]] = rs[idx[tri]]
        R_va[us[idx[vai]], is_[idx[vai]]] = rs[idx[vai]]
        out_s = mf_sgd(R_s, HP["F"], HP["EPOCHS"], HP["ETA"], HP["l_uv"], HP["l_bi"], SEED + fold * 100)
        out_t = mf_sgd_es(R_tr, R_va, HP["F"], HP["EPOCHS"], HP["ETA"], HP["l_uv"], HP["l_bi"], SEED + fold * 100 + 1)
        base = dict(U_s=out_s[0], U_t=out_t[0], V_t=out_t[1], bu_t=out_t[2], bi_t=out_t[3], mu_t=out_t[4], seed=SEED + fold * 100)
        for v in VARIANTS:
            results[v].append(run_variant(base, v, R_tr, R_va, HP))
        print(f"  fold {fold + 1}: " + "  ".join(f"{v}={results[v][-1]:.4f}" for v in VARIANTS), flush=True)
    return {v: (float(np.mean(d)), float(np.std(d, ddof=1) if len(d) > 1 else 0.0)) for v, d in results.items()}


def load_movielens(root="/content"):
    data = os.path.join(root, "ml-100k")
    zip_path = os.path.join(root, "ml-100k.zip")
    if not os.path.isfile(os.path.join(data, "u.data")):
        urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
    ratings = pd.read_csv(os.path.join(data, "u.data"), sep="\t", names=["user_id", "movie_id", "rating", "ts"])
    cols = ["movie_id", "title", "release", "video", "url"] + [f"g{i}" for i in range(19)]
    movies = pd.read_csv(os.path.join(data, "u.item"), sep="|", names=cols, encoding="latin-1")
    names = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
             "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
             "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    for i, n in enumerate(names):
        movies[n] = movies[f"g{i}"]
    rom = set(movies.loc[movies["Romance"] == 1, "movie_id"])
    dra = set(movies.loc[movies["Drama"] == 1, "movie_id"])
    rs = ratings[ratings.movie_id.isin(rom)]
    rt = ratings[ratings.movie_id.isin(dra)]
    users = sorted(set(rs.user_id) & set(rt.user_id))
    R_s = rs[rs.user_id.isin(users)].pivot(index="user_id", columns="movie_id", values="rating").reindex(users).fillna(0).to_numpy(float)
    R_t = rt[rt.user_id.isin(users)].pivot(index="user_id", columns="movie_id", values="rating").reindex(users).fillna(0).to_numpy(float)
    print("MovieLens", R_s.shape, int((R_s > 0).sum()), R_t.shape, int((R_t > 0).sum()))
    return R_s, R_t


def main():
    R_s, R_t = load_movielens()
    rows = []
    for dens in (0.25, 0.75):
        print(f"\n==== MovieLens density {dens} ====", flush=True)
        res = ablation_cv(R_s, R_t, dens)
        full = res["LU-STW"][0]
        for v, (m, s) in res.items():
            rows.append(dict(density=dens, variant=v, mae=m, mae_std=s, dmae=m - full))
            print(f"  {v}: {m:.4f}±{s:.4f}  dMAE={m - full:+.4f}", flush=True)
    df = pd.DataFrame(rows)
    out = "/content/ablation_ml_25_75.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
