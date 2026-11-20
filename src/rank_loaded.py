"""
Ranking on the matrices already built in the Colab runtime.
Does not read or rewrite any dataset file.

Yelp notebook:   matrix_veterinarians, matrix_pet_services
Amazon notebook: matrix_vg, matrix_ty
"""

import os

MODELS = [
    "AF", "KNN-CDRS", "KNN-CF", "MF-combine", "MF-mapping",
    "NNDSVD", "MF-HNO3", "MF-SNO3", "CDRS-GCN", "LU-STW",
]


def frames_from_namespace(ns):
    if "matrix_veterinarians" in ns and "matrix_pet_services" in ns:
        return "yelp", ns["matrix_veterinarians"], ns["matrix_pet_services"]
    if "matrix_vg" in ns and "matrix_ty" in ns:
        return "amazon", ns["matrix_vg"], ns["matrix_ty"]
    seen = sorted(k for k in ns if "matrix" in str(k).lower())
    raise RuntimeError(
        "Matrices are not in this Colab runtime. "
        "Open Yelp(LU_STW).ipynb or Amazon(LU_STW).ipynb, run the cells that "
        "build the matrices, then run this cell in that same notebook. "
        f"matrix* names found: {seen}"
    )


def rank_current(ns, train_size=0.5, n_splits=5, seed=42, out_dir="/content"):
    from src.baselines_ranking import run_all_ranking

    pair, src, tgt = frames_from_namespace(ns)
    users = src.index.intersection(tgt.index)
    R_s = src.loc[users].fillna(0).to_numpy(dtype=float)
    R_t = tgt.loc[users].fillna(0).to_numpy(dtype=float)
    print(
        pair, "users", len(users),
        "R_s", R_s.shape, "ratings", int((R_s > 0).sum()),
        "R_t", R_t.shape, "ratings", int((R_t > 0).sum()),
    )
    df = run_all_ranking(R_s, R_t, train_size=train_size, n_splits=n_splits, seed=seed, models=MODELS)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ranking_{pair}_50.csv")
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print("BEST NDCG@10:", df.loc[df["NDCG@10"].idxmax(), "Model"])
    print("saved", path)
    return df
