"""
IETE revision experiments (TITR-2026-0877).
Run in Colab after loading R_s, R_t (overlap users, same rows).

Covers reviewer asks:
  R2  Pure orthogonal Procrustes vs Hungarian
  R3  Confidence tercile vs target-only gain
  R4  User-only eta_u ablation
  R5  Genre-disjoint MovieLens helper
  R6  Precision/Recall/NDCG/MRR @K
  R7  Multi-seed CV
  R8  Ablation at 25/50/75% + train/infer timing
"""

import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

# Reuse core from LU_STW_Novelty_Full when available
try:
    from lu_stw_core import (
        mf_sgd, mf_sgd_es, eval_pred, cost_matrix, perm_matrix, user_conf, HP
    )
except ImportError:
    try:
        from src.lu_stw_core import (
            mf_sgd, mf_sgd_es, eval_pred, cost_matrix, perm_matrix, user_conf, HP
        )
    except ImportError:
        raise SystemExit("Import lu_stw_core from src/ (pip install -e . or add repo root to PYTHONPATH).")


def orthogonal_procrustes(U_s, U_t):
    """Pure Procrustes: U_al = U_s @ Q, Q = argmin ||U_t - U_s Q||_F, Q orthogonal."""
    M = U_s.T @ U_t
    U_, _, Vt = np.linalg.svd(M)
    Q = U_ @ Vt
    return U_s @ Q, Q


def fine_tune_es_rev(X, V, bu, bi, mu, R_tr, R_val, U_s_al, U_t, epochs, eta, l_uv, l_bi,
                     use_conf=True, user_only=False, eps=0.05, patience=10, md=1e-4,
                     rank_aware=False, bpr_weight=0.5, rel_thresh=4.0, n_bpr=1):
    """If rank_aware: upweight high ratings + BPR pairs (pos r>=thresh vs unobserved)."""
    X, V, bu, bi = X.copy(), V.copy(), bu.copy(), bi.copy()
    conf = user_conf(U_s_al, U_t, eps) if use_conf else np.ones(X.shape[0])
    us, is_ = R_tr.nonzero()
    rs = R_tr[us, is_]
    m, n = R_tr.shape
    # per-user positives for BPR
    pos = {u: is_[us == u][rs[us == u] >= rel_thresh] for u in np.unique(us)}
    best_rmse, wait = np.inf, 0
    bX = bV = bbu = bbi = None
    rng = np.random.RandomState(0)
    for _ in range(epochs):
        idx = np.arange(len(rs))
        rng.shuffle(idx)
        for j in idx:
            u, i, r = us[j], is_[j], rs[j]
            eu = eta * conf[u]
            ei = eu if not user_only else eta
            w = (1.0 + max(0.0, r - 3.0)) if rank_aware else 1.0
            pred = mu + bu[u] + bi[i] + X[u] @ V[i]
            e = w * (r - pred)
            bu[u] += eu * (e - l_bi * bu[u])
            bi[i] += ei * (e - l_bi * bi[i])
            xo, vo = X[u].copy(), V[i].copy()
            X[u] += eu * (e * vo - l_uv * xo)
            V[i] += ei * (e * xo - l_uv * vo)

            if rank_aware and len(pos.get(u, [])) > 0:
                for _b in range(n_bpr):
                    i_pos = int(rng.choice(pos[u]))
                    i_neg = int(rng.randint(0, n))
                    if R_tr[u, i_neg] > 0:
                        continue
                    s_pos = mu + bu[u] + bi[i_pos] + X[u] @ V[i_pos]
                    s_neg = mu + bu[u] + bi[i_neg] + X[u] @ V[i_neg]
                    # logistic BPR grad
                    x = s_pos - s_neg
                    sig = 1.0 / (1.0 + np.exp(min(20, max(-20, x))))
                    g = bpr_weight * eu * sig
                    # update user + both items
                    X[u] += g * (V[i_pos] - V[i_neg]) - eu * l_uv * X[u]
                    V[i_pos] += g * X[u] - ei * l_uv * V[i_pos]
                    V[i_neg] -= g * X[u] + ei * l_uv * V[i_neg]
                    bu[u] += g
                    bi[i_pos] += g
                    bi[i_neg] -= g

        _, rmse = eval_pred(X @ V.T + mu + bu[:, None] + bi[None, :], R_val)
        if rmse < best_rmse - md:
            best_rmse, wait = rmse, 0
            bX, bV, bbu, bbi = X.copy(), V.copy(), bu.copy(), bi.copy()
        else:
            wait += 1
            if wait >= patience:
                break
    if bX is None:
        bX, bV, bbu, bbi = X, V, bu, bi
    return bX, bV, bbu, bbi, conf


def run_variant(R_s, R_tr, R_val, align="hungarian", use_conf=True, user_only=False,
                use_source=True, use_ft=True, hp=HP, rank_aware=False, bpr_weight=0.5):
    t0 = time.time()
    if not use_source:
        out = mf_sgd_es(R_tr, R_val, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"])
        U, V, bu, bi, mu = out
        R_hat = U @ V.T + mu + bu[:, None] + bi[None, :]
        mae, rmse = eval_pred(R_hat, R_val)
        return dict(mae=mae, rmse=rmse, conf=None, R_hat=R_hat, train_s=time.time() - t0)

    out_s = mf_sgd(R_s, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"])
    out_t = mf_sgd_es(R_tr, R_val, hp["F"], hp["EPOCHS"], hp["ETA"], hp["l_uv"], hp["l_bi"])
    U_s, _, bu_s, _, _ = out_s
    U_t, V_t, _, bi_t, mu_t = out_t

    if align == "procrustes":
        U_al, _ = orthogonal_procrustes(U_s, U_t)
    elif align == "identity":
        U_al = U_s.copy()
    else:
        P = perm_matrix(cost_matrix(U_s, U_t), "hungarian")
        U_al = U_s @ P.T

    if use_ft:
        X, Vf, buf, bif, conf = fine_tune_es_rev(
            U_al, V_t, bu_s, bi_t, mu_t, R_tr, R_val, U_al, U_t,
            hp["EPOCHS_FT"], hp["ETA_P"], hp["l_uvp"], hp["l_bip"],
            use_conf=use_conf, user_only=user_only,
            rank_aware=rank_aware, bpr_weight=bpr_weight)
        R_hat = X @ Vf.T + mu_t + buf[:, None] + bif[None, :]
    else:
        conf = user_conf(U_al, U_t)
        R_hat = U_al @ V_t.T + mu_t + bu_s[:, None] + bi_t[None, :]

    mae, rmse = eval_pred(R_hat, R_val)
    return dict(mae=mae, rmse=rmse, conf=conf, R_hat=R_hat, train_s=time.time() - t0)


def ranking_at_k(R_hat, R_train, R_test, K=10, thresh=4.0, n_neg=100, seed=0):
    rng = np.random.RandomState(seed)
    us, is_ = R_test.nonzero()
    precs, recalls, ndcgs, mrrs = [], [], [], []
    for u in np.unique(us):
        mask = us == u
        items = is_[mask]
        rel = items[R_test[u, items] >= thresh]
        if len(rel) == 0:
            continue
        seen = set(R_train[u].nonzero()[0])
        cand = [i for i in range(R_hat.shape[1]) if i not in seen]
        if len(cand) > n_neg:
            neg = rng.choice(cand, n_neg, replace=False).tolist()
        else:
            neg = cand
        pool = list(set(rel.tolist() + neg))
        scores = R_hat[u, pool]
        order = np.argsort(-scores)
        ranked = [pool[i] for i in order]
        top = ranked[:K]
        hits = sum(1 for i in top if i in set(rel))
        precs.append(hits / K)
        recalls.append(hits / len(rel))
        dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(top) if i in set(rel))
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(K, len(rel))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        rr = 0.0
        for r, i in enumerate(ranked):
            if i in set(rel):
                rr = 1.0 / (r + 1)
                break
        mrrs.append(rr)
    if not precs:
        return {f"P@{K}": 0, f"R@{K}": 0, f"NDCG@{K}": 0, "MRR": 0}
    return {
        f"P@{K}": float(np.mean(precs)),
        f"R@{K}": float(np.mean(recalls)),
        f"NDCG@{K}": float(np.mean(ndcgs)),
        "MRR": float(np.mean(mrrs)),
    }


def make_folds(R_t, train_size=0.5, n_splits=5, seed=42):
    us, is_ = R_t.nonzero()
    rs = R_t[us, is_]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(rs), int(train_size * len(rs)), replace=False)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tri, vai in kf.split(idx):
        R_tr = np.zeros_like(R_t)
        R_val = np.zeros_like(R_t)
        ti, vi = idx[tri], idx[vai]
        R_tr[us[ti], is_[ti]] = rs[ti]
        R_val[us[vi], is_[vi]] = rs[vi]
        folds.append((R_tr, R_val))
    return folds


def cv_report(R_s, R_t, variants, densities=(0.25, 0.5, 0.75), seeds=(0, 1, 2, 3, 4), n_splits=5):
    """R7+R8: multi-seed × densities × variants."""
    all_out = {}
    for dens in densities:
        for name, kwargs in variants.items():
            key = (name, dens)
            maes, rmses, times = [], [], []
            for seed in seeds:
                for R_tr, R_val in make_folds(R_t, dens, n_splits, seed):
                    out = run_variant(R_s, R_tr, R_val, **kwargs)
                    maes.append(out["mae"])
                    rmses.append(out["rmse"])
                    times.append(out["train_s"])
            all_out[key] = dict(
                mae=np.mean(maes), mae_std=np.std(maes),
                rmse=np.mean(rmses), rmse_std=np.std(rmses),
                train_s=np.mean(times), n=len(maes),
            )
            print(f"{name} dens={dens}: MAE {all_out[key]['mae']:.4f}±{all_out[key]['mae_std']:.4f}  "
                  f"RMSE {all_out[key]['rmse']:.4f}±{all_out[key]['rmse_std']:.4f}  "
                  f"train {all_out[key]['train_s']:.1f}s")
    return all_out


def confidence_tercile_gain(R_s, R_t, train_size=0.5, seed=42, hp=HP):
    """R3: high vs low omega tercile — LU-STW gain over target-only."""
    folds = make_folds(R_t, train_size, 5, seed)
    R_tr, R_val = folds[0]
    base = run_variant(R_s, R_tr, R_val, use_source=False, hp=hp)
    full = run_variant(R_s, R_tr, R_val, align="hungarian", use_conf=True, hp=hp)
    conf = full["conf"]
    us, is_ = R_val.nonzero()
    q1, q2 = np.quantile(conf, [1 / 3, 2 / 3])
    buckets = {"low": [], "mid": [], "high": []}
    for u in np.unique(us):
        mask = us == u
        t = R_val[us[mask], is_[mask]]
        mae_b = np.mean(np.abs(t - base["R_hat"][us[mask], is_[mask]]))
        mae_f = np.mean(np.abs(t - full["R_hat"][us[mask], is_[mask]]))
        gain = mae_b - mae_f
        c = conf[u]
        b = "high" if c >= q2 else ("low" if c <= q1 else "mid")
        buckets[b].append(gain)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in buckets.items()}


def build_movielens_disjoint(ratings_df, movies_df, src_genre="Romance", tgt_genre="Drama"):
    """
    R5 helper. ratings_df: user_id, movie_id, rating
    movies_df: movie_id, genres (pipe-separated)
    Drops movies labeled BOTH genres from both domains.
    """
    g = movies_df.set_index("movie_id")["genres"].str.split("|")
    src_ids = set(g[g.apply(lambda x: src_genre in x and tgt_genre not in x)].index)
    tgt_ids = set(g[g.apply(lambda x: tgt_genre in x and src_genre not in x)].index)
    both = set(g[g.apply(lambda x: src_genre in x and tgt_genre in x)].index)
    print(f"Disjoint: |src|={len(src_ids)} |tgt|={len(tgt_ids)} dropped_both={len(both)}")
    rs = ratings_df[ratings_df.movie_id.isin(src_ids)]
    rt = ratings_df[ratings_df.movie_id.isin(tgt_ids)]
    users = sorted(set(rs.user_id) & set(rt.user_id))
    # pivot to dense matrices aligned on users
    R_s = rs[rs.user_id.isin(users)].pivot(index="user_id", columns="movie_id", values="rating")
    R_t = rt[rt.user_id.isin(users)].pivot(index="user_id", columns="movie_id", values="rating")
    R_s = R_s.reindex(users).fillna(0).values
    R_t = R_t.reindex(users).fillna(0).values
    return R_s, R_t, dict(n_both=len(both), n_src=len(src_ids), n_tgt=len(tgt_ids), n_users=len(users))


REVISION_VARIANTS = {
    "LU-STW": dict(align="hungarian", use_conf=True, user_only=False),
    "Procrustes": dict(align="procrustes", use_conf=True, user_only=False),
    "User-only eta": dict(align="hungarian", use_conf=True, user_only=True),
    "w/o Confidence": dict(align="hungarian", use_conf=False, user_only=False),
    "w/o Hungarian": dict(align="identity", use_conf=True, user_only=False),
    "Target Only": dict(use_source=False),
}


if __name__ == "__main__":
    # After loading:
    #   R_s, R_t = ...
    raise SystemExit(
        "Import this module in Colab, set R_s/R_t, then call:\n"
        "  cv_report(R_s, R_t, REVISION_VARIANTS)\n"
        "  confidence_tercile_gain(R_s, R_t)\n"
        "  ranking_at_k(...)\n"
        "  build_movielens_disjoint(...)\n"
    )
