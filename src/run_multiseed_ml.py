"""
Comment 7: MovieLens 50% multi-seed CV (S=5 seeds x 5 folds = 25 scores).
Compares LU-STW to fair MF peers (same f=20) + Target Only.
"""
import os
import sys
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

SEEDS = (0, 1, 2, 3, 4)
FOLDS = 5
DENS = 0.5
HP = dict(F=20, ETA=0.005, EPOCHS=100, ETA_P=0.005, EPOCHS_FT=150,
          l_uv=0.005, l_bi=0.01, l_uvp=0.005, l_bip=0.01)
EPS = 0.05


def mf_sgd(R, f, epochs, eta, l_uv, l_bi, seed):
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U, V = rng.rand(m, f) * 0.01, rng.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero()
    rs = R[us, is_]
    for _ in range(epochs):
        for j in rng.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            e = r - (mu + bu[u] + bi[i] + U[u] @ V[i])
            bu[u] += eta * (e - l_bi * bu[u])
            bi[i] += eta * (e - l_bi * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - l_uv * uo)
            V[i] += eta * (e * uo - l_uv * vo)
    return U, V, bu, bi, mu


def eval_pred(R_hat, R_true):
    us, is_ = R_true.nonzero()
    e = R_true[us, is_] - R_hat[us, is_]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def mf_sgd_es(R, R_val, f, epochs, eta, l_uv, l_bi, seed, patience=5, md=1e-4):
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U, V = rng.rand(m, f) * 0.01, rng.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = R[R > 0].mean()
    us, is_ = R.nonzero()
    rs = R[us, is_]
    best, wait = np.inf, 0
    bU = bV = bbu = bbi = None
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
            best, wait = rmse, 0
            bU, bV, bbu, bbi = U.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience:
                break
    if bU is None:
        bU, bV, bbu, bbi = U, V, bu, bi
    return bU, bV, bbu, bbi, mu


def hungarian_align(U_s, U_t):
    Us = U_s / np.maximum(np.linalg.norm(U_s, axis=0, keepdims=True), 1e-12)
    Ut = U_t / np.maximum(np.linalg.norm(U_t, axis=0, keepdims=True), 1e-12)
    C = 1.0 - Us.T @ Ut
    r, c = linear_sum_assignment(C)
    P = np.zeros((U_s.shape[1], U_s.shape[1]))
    P[c, r] = 1.0
    return U_s @ P.T


def user_conf(U_al, U_t, eps=EPS):
    c = np.zeros(U_al.shape[0])
    for u in range(U_al.shape[0]):
        a, b = U_al[u], U_t[u]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        c[u] = (a @ b) / (na * nb) if na and nb else 0.0
    return np.maximum(c, eps)


def fine_tune(X0, V0, bu0, bi0, mu, R_tr, R_val, conf, epochs, eta, l_uv, l_bi, seed, patience=10):
    rng = np.random.RandomState(seed)
    X, V, bu, bi = X0.copy(), V0.copy(), bu0.copy(), bi0.copy()
    us, is_ = R_tr.nonzero()
    rs = R_tr[us, is_]
    best, wait = np.inf, 0
    bX = bV = bbu = bbi = None
    for _ in range(epochs):
        for j in rng.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            eu = eta * conf[u]
            e = r - (mu + bu[u] + bi[i] + X[u] @ V[i])
            bu[u] += eu * (e - l_bi * bu[u])
            bi[i] += eu * (e - l_bi * bi[i])
            xo, vo = X[u].copy(), V[i].copy()
            X[u] += eu * (e * vo - l_uv * xo)
            V[i] += eu * (e * xo - l_uv * vo)
        _, rmse = eval_pred(X @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best - 1e-4:
            best, wait = rmse, 0
            bX, bV, bbu, bbi = X.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience:
                break
    if bX is None:
        bX, bV, bbu, bbi = X, V, bu, bi
    return bX, bV, bbu, bbi


def predict_target_only(R_s, R_tr, R_val, seed):
    U, V, bu, bi, mu = mf_sgd_es(R_tr, R_val, HP["F"], HP["EPOCHS"], HP["ETA"], HP["l_uv"], HP["l_bi"], seed)
    return eval_pred(U @ V.T + mu + bu[:, None] + bi[None, :], R_val)


def predict_mf_combine(R_s, R_tr, R_val, seed):
    R_comb = np.hstack([R_s, R_tr])
    R_val_comb = np.zeros_like(R_comb)
    R_val_comb[:, R_s.shape[1]:] = R_val
    U, V, bu, bi, mu = mf_sgd_es(R_comb, R_val_comb, HP["F"], 30, HP["ETA"], 0.01, 0.01, seed)
    ns = R_s.shape[1]
    return eval_pred(U @ V[ns:].T + mu + bu[:, None] + bi[ns:][None, :], R_val)


def predict_lustw(R_s, R_tr, R_val, seed):
    U_s, _, bu_s, _, _ = mf_sgd(R_s, HP["F"], HP["EPOCHS"], HP["ETA"], HP["l_uv"], HP["l_bi"], seed)
    U_t, V_t, _, bi_t, mu_t = mf_sgd_es(R_tr, R_val, HP["F"], HP["EPOCHS"], HP["ETA"], HP["l_uv"], HP["l_bi"], seed + 1)
    U_al = hungarian_align(U_s, U_t)
    conf = user_conf(U_al, U_t)
    X, Vf, buf, bif = fine_tune(U_al, V_t, bu_s, bi_t, mu_t, R_tr, R_val, conf,
                                HP["EPOCHS_FT"], HP["ETA_P"], HP["l_uvp"], HP["l_bip"], seed + 7)
    return eval_pred(X @ Vf.T + mu_t + buf[:, None] + bif[None, :], R_val)


MODELS = {
    "Target Only": predict_target_only,
    "MF-combine": predict_mf_combine,
    "LU-STW": predict_lustw,
}


def load_ml(root):
    data = os.path.join(root, "ml-100k")
    zpath = os.path.join(root, "ml-100k.zip")
    if not os.path.isfile(os.path.join(data, "u.data")):
        urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
    ratings = pd.read_csv(os.path.join(data, "u.data"), sep="\t", names=["user_id", "movie_id", "rating", "ts"])
    cols = ["movie_id", "title", "release", "video", "url"] + [f"g{i}" for i in range(19)]
    movies = pd.read_csv(os.path.join(data, "u.item"), sep="|", names=cols, encoding="latin-1")
    gnames = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
              "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
              "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    for i, n in enumerate(gnames):
        movies[n] = movies[f"g{i}"]
    rom = set(movies.loc[movies["Romance"] == 1, "movie_id"])
    dra = set(movies.loc[movies["Drama"] == 1, "movie_id"])
    rs = ratings[ratings.movie_id.isin(rom)]
    rt = ratings[ratings.movie_id.isin(dra)]
    users = sorted(set(rs.user_id) & set(rt.user_id))
    R_s = rs[rs.user_id.isin(users)].pivot(index="user_id", columns="movie_id", values="rating").reindex(users).fillna(0).to_numpy(float)
    R_t = rt[rt.user_id.isin(users)].pivot(index="user_id", columns="movie_id", values="rating").reindex(users).fillna(0).to_numpy(float)
    return R_s, R_t


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    out = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        adj = max(adj, prev)
        out[idx] = adj
        prev = adj
    return out


def main():
    root = "/content" if os.path.isdir("/content") else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(root, exist_ok=True)
    R_s, R_t = load_ml(root)
    print("ML", R_s.shape, R_t.shape, flush=True)
    scores = {name: {"mae": [], "rmse": []} for name in MODELS}
    for seed in SEEDS:
        us, is_ = R_t.nonzero()
        rs = R_t[us, is_]
        idx = np.random.RandomState(seed).choice(len(rs), int(DENS * len(rs)), replace=False)
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fi, (tri, vai) in enumerate(kf.split(idx)):
            R_tr = np.zeros_like(R_t)
            R_val = np.zeros_like(R_t)
            R_tr[us[idx[tri]], is_[idx[tri]]] = rs[idx[tri]]
            R_val[us[idx[vai]], is_[idx[vai]]] = rs[idx[vai]]
            for name, fn in MODELS.items():
                mae, rmse = fn(R_s, R_tr, R_val, seed * 100 + fi)
                scores[name]["mae"].append(mae)
                scores[name]["rmse"].append(rmse)
            print(f"seed {seed} fold {fi+1}: LU-STW MAE={scores['LU-STW']['mae'][-1]:.4f}", flush=True)

    rows = []
    for name, d in scores.items():
        rows.append(dict(
            model=name,
            mae=np.mean(d["mae"]), mae_std=np.std(d["mae"], ddof=1),
            rmse=np.mean(d["rmse"]), rmse_std=np.std(d["rmse"], ddof=1),
            n=len(d["mae"]),
        ))
        print(f"{name}: MAE {np.mean(d['mae']):.4f}±{np.std(d['mae'], ddof=1):.4f}  "
              f"RMSE {np.mean(d['rmse']):.4f}±{np.std(d['rmse'], ddof=1):.4f}  n={len(d['mae'])}", flush=True)

    lustw_mae = np.array(scores["LU-STW"]["mae"])
    lustw_rmse = np.array(scores["LU-STW"]["rmse"])
    p_mae, p_rmse = [], []
    names = [n for n in MODELS if n != "LU-STW"]
    for name in names:
        d_mae = np.array(scores[name]["mae"]) - lustw_mae
        d_rmse = np.array(scores[name]["rmse"]) - lustw_rmse
        # one-sided: LU-STW lower error => positive diff
        t_m, p_m = stats.ttest_1samp(d_mae, 0.0, alternative="greater")
        t_r, p_r = stats.ttest_1samp(d_rmse, 0.0, alternative="greater")
        p_mae.append(p_m)
        p_rmse.append(p_r)
        print(f"vs {name}: MAE p={p_m:.4g}  RMSE p={p_r:.4g}  "
              f"meanΔMAE={d_mae.mean():+.4f}", flush=True)
    h_mae = holm(np.array(p_mae))
    h_rmse = holm(np.array(p_rmse))
    for i, name in enumerate(names):
        print(f"Holm vs {name}: MAE {h_mae[i]:.4g}  RMSE {h_rmse[i]:.4g}", flush=True)

    out = "/content/multiseed_ml50.csv" if os.path.isdir("/content") else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "multiseed_ml50.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
