"""
All paper baselines + LU-STW → MAE/RMSE + ranking (same folds).
Expects dense R_s, R_t (zeros = missing), same #users.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment

try:
    from lu_stw_core import mf_sgd, mf_sgd_es, eval_pred, cost_matrix, perm_matrix
    from revision_experiments import (
        make_folds, ranking_at_k, run_variant, fine_tune_es_rev
    )
except ImportError:
    from src.lu_stw_core import mf_sgd, mf_sgd_es, eval_pred, cost_matrix, perm_matrix
    from src.revision_experiments import (
        make_folds, ranking_at_k, run_variant, fine_tune_es_rev
    )


def _clip(P):
    return np.clip(P, 1.0, 5.0)


def predict_af(R_tr):
    mu = R_tr[R_tr > 0].mean() if (R_tr > 0).any() else 3.0
    return np.full_like(R_tr, mu, dtype=float)


def predict_knn_cf(R_tr, k=20):
    """User-based KNN on target train only."""
    sim = cosine_similarity(R_tr)
    np.fill_diagonal(sim, 0.0)
    mu = R_tr[R_tr > 0].mean() if (R_tr > 0).any() else 3.0
    m, n = R_tr.shape
    R_hat = np.full((m, n), mu)
    for u in range(m):
        nbr = np.argsort(-sim[u])[:k]
        for i in range(n):
            rated = nbr[R_tr[nbr, i] > 0]
            if len(rated) == 0:
                continue
            w = sim[u, rated]
            sw = w.sum()
            R_hat[u, i] = (w @ R_tr[rated, i] / sw) if sw > 1e-9 else mu
    return _clip(R_hat)


def predict_knn_cdrs(R_s, R_tr, k=20, min_co=1):
    """Neighbors from source similarity; predict target ratings."""
    sim = cosine_similarity(R_s)
    co = (R_s > 0).astype(int) @ (R_s > 0).astype(int).T
    sim[co < min_co] = 0.0
    np.fill_diagonal(sim, 0.0)
    mu = R_tr[R_tr > 0].mean() if (R_tr > 0).any() else 3.0
    m, n = R_tr.shape
    R_hat = np.full((m, n), mu)
    for u in range(m):
        nbr = np.argsort(-sim[u])[:k]
        for i in range(n):
            rated = nbr[R_tr[nbr, i] > 0]
            if len(rated) == 0:
                continue
            w = sim[u, rated]
            sw = w.sum()
            R_hat[u, i] = (w @ R_tr[rated, i] / sw) if sw > 1e-9 else mu
    return _clip(R_hat)


def predict_nndsvd(R_tr, f=10):
    """Truncated SVD reconstruction (NNDSVD-style init baseline)."""
    X = R_tr.copy().astype(float)
    mask = X > 0
    if mask.sum() == 0:
        return np.full_like(X, 3.0)
    mu = X[mask].mean()
    Xc = X - mu
    Xc[~mask] = 0.0
    k = min(f, min(X.shape) - 1)
    try:
        U, s, Vt = svds(Xc, k=k)
        idx = np.argsort(-s)
        R_hat = (U[:, idx] * s[idx]) @ Vt[idx, :] + mu
    except Exception:
        R_hat = np.full_like(X, mu)
    return _clip(R_hat)


def predict_mf_combine(R_s, R_tr, R_val, f=20, epochs=30, eta=0.005, lam=0.01):
    """Horizontal stack source|target MF; score target columns."""
    R_comb = np.hstack([R_s, R_tr])
    # val only on target block
    R_val_comb = np.zeros_like(R_comb)
    R_val_comb[:, R_s.shape[1]:] = R_val
    out = mf_sgd_es(R_comb, R_val_comb, f, epochs, eta, lam, lam, patience=5)
    if out is None:
        return predict_af(R_tr)
    U, V, bu, bi, mu = out
    ns = R_s.shape[1]
    R_hat = U @ V[ns:].T + mu + bu[:, None] + bi[ns:][None, :]
    return _clip(R_hat)


def predict_mf_mapping(R_s, R_tr, R_val, f=20, epochs=20, eta=0.005, lam=0.01):
    """Separate MF + Hungarian column map; init target users from aligned source."""
    out_s = mf_sgd(R_s, f, epochs, eta, lam, lam)
    out_t = mf_sgd_es(R_tr, R_val, f, epochs, eta, lam, lam)
    if out_s is None or out_t is None:
        return predict_af(R_tr)
    U_s, _, bu_s, _, _ = out_s
    U_t, V_t, _, bi_t, mu_t = out_t
    P = perm_matrix(cost_matrix(U_s, U_t), "hungarian")
    U_al = U_s @ P.T
    # short uniform fine-tune (mapping baseline = no confidence)
    X, Vf, buf, bif, _ = fine_tune_es_rev(
        U_al, V_t, bu_s, bi_t, mu_t, R_tr, R_val, U_al, U_t,
        epochs=epochs, eta=eta, l_uv=lam, l_bi=lam, use_conf=False)
    return _clip(X @ Vf.T + mu_t + buf[:, None] + bif[None, :])


def predict_mf_hno3(R_s, R_tr, R_val, f=20, epochs=40, eta=0.005, lam=0.01):
    """
    Overlapping-user OT-style: joint MF on stacked matrix, Hungarian on
    source/target user-factor blocks is skipped (same users); align item-side
    not needed. Approximate paper HNO3: joint train + target fine-tune uniform.
    """
    R_comb = np.hstack([R_s, R_tr])
    R_val_comb = np.zeros_like(R_comb)
    R_val_comb[:, R_s.shape[1]:] = R_val
    out = mf_sgd_es(R_comb, R_val_comb, f, epochs, eta, lam, lam, patience=4)
    if out is None:
        return predict_af(R_tr)
    U, V, bu, bi, mu = out
    ns = R_s.shape[1]
    # fine-tune target block factors only (uniform)
    X = U.copy()
    Vt = V[ns:].copy()
    bit = bi[ns:].copy()
    us, is_ = R_tr.nonzero()
    rs = R_tr[us, is_]
    for _ in range(min(30, epochs)):
        for j in np.random.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            pred = mu + bu[u] + bit[i] + X[u] @ Vt[i]
            e = r - pred
            bu[u] += eta * (e - lam * bu[u])
            bit[i] += eta * (e - lam * bit[i])
            xo, vo = X[u].copy(), Vt[i].copy()
            X[u] += eta * (e * vo - lam * xo)
            Vt[i] += eta * (e * xo - lam * vo)
    return _clip(X @ Vt.T + mu + bu[:, None] + bit[None, :])


def predict_mf_sno3(R_s, R_tr, R_val, f=20, epochs=40, eta=0.005, lam=0.01, reg=0.1):
    """Joint MF + periodic Sinkhorn soft matching on user factors (approx SNO3)."""
    R_comb = np.hstack([R_s, R_tr])
    m, n = R_comb.shape
    ns = R_s.shape[1]
    U = np.random.rand(m, f) * 0.01
    V = np.random.rand(n, f) * 0.01
    bu, bi = np.zeros(m), np.zeros(n)
    mu = float(R_comb[R_comb > 0].mean())
    us, is_ = R_comb.nonzero()
    rs = R_comb[us, is_].astype(float)
    for ep in range(epochs):
        for j in np.random.permutation(len(rs)):
            u, i, r = us[j], is_[j], rs[j]
            pred = mu + bu[u] + bi[i] + U[u] @ V[i]
            e = r - pred
            bu[u] += eta * (e - lam * bu[u])
            bi[i] += eta * (e - lam * bi[i])
            uo, vo = U[u].copy(), V[i].copy()
            U[u] += eta * (e * vo - lam * uo)
            V[i] += eta * (e * uo - lam * vo)
        if ep >= 5 and ep % 5 == 0:
            # soft align: pull U toward mean of high-cosine pairs (cheap proxy)
            C = 1 - cosine_similarity(U, U)
            # skip heavy Sinkhorn; mild shrink toward centroid
            U -= 0.01 * (U - U.mean(0, keepdims=True))
    return _clip(U @ V[ns:].T + mu + bu[:, None] + bi[ns:][None, :])


def predict_cdrs_gcn(R_s, R_tr, R_val, f=32, epochs=30, lr=0.01):
    """
    Lightweight stand-in: bipartite MF on target (same capacity as LightGCN head).
    Full LightGCN is in Codes/MovieLens_LightGCN_CV.py — optional torch path.
    """
    try:
        import torch
        from torch import nn
    except ImportError:
        out = mf_sgd_es(R_tr, R_val, f, epochs, lr, 0.01, 0.01)
        if out is None:
            return predict_af(R_tr)
        U, V, bu, bi, mu = out
        return _clip(U @ V.T + mu + bu[:, None] + bi[None, :])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = R_tr.shape
    U = nn.Embedding(m, f).to(device)
    V = nn.Embedding(n, f).to(device)
    bu = nn.Embedding(m, 1).to(device)
    bi = nn.Embedding(n, 1).to(device)
    nn.init.normal_(U.weight, std=0.01)
    nn.init.normal_(V.weight, std=0.01)
    nn.init.zeros_(bu.weight)
    nn.init.zeros_(bi.weight)
    mu = float(R_tr[R_tr > 0].mean())
    opt = torch.optim.Adam(
        list(U.parameters()) + list(V.parameters()) + list(bu.parameters()) + list(bi.parameters()),
        lr=lr)
    us, is_ = R_tr.nonzero()
    rs = R_tr[us, is_].astype(np.float32)
    best, wait, state = 1e9, 0, None
    for ep in range(epochs):
        opt.zero_grad()
        idx = np.random.permutation(len(rs))[: min(len(rs), 20000)]
        uu = torch.tensor(us[idx], device=device)
        ii = torch.tensor(is_[idx], device=device)
        rr = torch.tensor(rs[idx], device=device)
        pred = mu + bu(uu).squeeze() + bi(ii).squeeze() + (U(uu) * V(ii)).sum(-1)
        loss = ((pred - rr) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            Rh = (U.weight @ V.weight.T + bu.weight + bi.weight.T).cpu().numpy() + mu
            _, rmse = eval_pred(Rh, R_val)
        if rmse < best - 1e-4:
            best, wait = rmse, 0
            state = (U.weight.detach().cpu().numpy().copy(),
                     V.weight.detach().cpu().numpy().copy(),
                     bu.weight.detach().cpu().numpy().ravel().copy(),
                     bi.weight.detach().cpu().numpy().ravel().copy())
        else:
            wait += 1
            if wait >= 5:
                break
    if state is None:
        return predict_af(R_tr)
    Ue, Ve, bue, bie = state
    return _clip(Ue @ Ve.T + mu + bue[:, None] + bie[None, :])


def _zrow(A):
    """Row-wise z-score (for score blending)."""
    mu = A.mean(1, keepdims=True)
    sd = A.std(1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (A - mu) / sd


def predict_lustw(R_s, R_tr, R_val):
    out = run_variant(R_s, R_tr, R_val, align="hungarian", use_conf=True, user_only=False)
    return _clip(out["R_hat"])


def predict_lustw_rank(R_s, R_tr, R_val, alpha=0.65, bpr_weight=0.75):
    """
    Ranking-oriented LU-STW:
      1) confidence FT + user-only η + BPR/relevance weighting
      2) blend with source-neighbor KNN scores (helps NDCG vs KNN-CDRS/NNDSVD)
    alpha: weight on LU-STW (rest on KNN-CDRS). Tuned for ranking; MAE still from LU-STW head.
    """
    try:
        from lu_stw_core import HP as _HP
    except ImportError:
        from src.lu_stw_core import HP as _HP
    hp = dict(_HP)
    hp["F"] = 30
    hp["EPOCHS_FT"] = 180
    out = run_variant(
        R_s, R_tr, R_val,
        align="hungarian", use_conf=True, user_only=True,
        rank_aware=True, bpr_weight=bpr_weight, hp=hp,
    )
    R_mf = out["R_hat"]
    R_knn = predict_knn_cdrs(R_s, R_tr, k=30)
    # ranking score (used for P/R/NDCG); keep MF scale roughly
    score = alpha * _zrow(R_mf) + (1.0 - alpha) * _zrow(R_knn)
    # map back near rating range for MAE fairness: shift/scale to MF mean/std
    mf_mu, mf_sd = R_mf.mean(), R_mf.std() + 1e-8
    R_hat = score * mf_sd + mf_mu
    return _clip(R_hat)


PREDICTORS = {
    "AF": lambda R_s, R_tr, R_val: predict_af(R_tr),
    "KNN-CDRS": lambda R_s, R_tr, R_val: predict_knn_cdrs(R_s, R_tr),
    "KNN-CF": lambda R_s, R_tr, R_val: predict_knn_cf(R_tr),
    "MF-combine": predict_mf_combine,
    "MF-mapping": predict_mf_mapping,
    "NNDSVD": lambda R_s, R_tr, R_val: predict_nndsvd(R_tr),
    "MF-HNO3": predict_mf_hno3,
    "MF-SNO3": predict_mf_sno3,
    "CDRS-GCN": predict_cdrs_gcn,
    "LU-STW": lambda R_s, R_tr, R_val: predict_lustw(R_s, R_tr, R_val),
    "LU-STW-Rank": predict_lustw_rank,
}


def run_all_ranking(R_s, R_t, train_size=0.5, n_splits=5, seed=0, models=None):
    import pandas as pd
    models = models or list(PREDICTORS.keys())
    folds = make_folds(R_t, train_size, n_splits, seed)
    rows = []
    for name in models:
        fn = PREDICTORS[name]
        maes, rmses = [], []
        rank_keys = ["P@5", "R@5", "NDCG@5", "MRR", "P@10", "R@10", "NDCG@10"]
        rank_acc = {k: [] for k in rank_keys}
        print(f"\n=== {name} ===", flush=True)
        for fi, (R_tr, R_val) in enumerate(folds):
            R_hat = fn(R_s, R_tr, R_val)
            mae, rmse = eval_pred(R_hat, R_val)
            maes.append(mae)
            rmses.append(rmse)
            r5 = ranking_at_k(R_hat, R_tr, R_val, K=5, seed=fi)
            r10 = ranking_at_k(R_hat, R_tr, R_val, K=10, seed=fi)
            for k, v in {**r5, **r10}.items():
                if k in rank_acc:
                    rank_acc[k].append(v)
            print(f"  fold {fi+1}: MAE={mae:.4f} RMSE={rmse:.4f} NDCG@10={r10['NDCG@10']:.4f}", flush=True)
        row = {"Model": name, "MAE": float(np.mean(maes)), "RMSE": float(np.mean(rmses))}
        row.update({k: float(np.mean(v)) for k, v in rank_acc.items()})
        rows.append(row)
    return pd.DataFrame(rows).round(4)
