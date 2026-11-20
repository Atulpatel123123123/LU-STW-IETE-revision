# Point-by-point response — TITR-2026-0877

**Manuscript:** Confidence-Aware Learning of User-Specific Transfer Weights in Cross-Domain Recommendation Systems  
**Codes analysed:** `Codes/` (Ablation_Fair_*.py, LU_STW_Novelty_Full.py, HNO3_CDR.py, Movielens/Yelp/Amazon notebooks, Statistical_Significance_Test_Hungarian(LU_STW).ipynb, Revision_Experiments.py)  
**Revised text** in Overleaf `Paper.tex` is marked in blue (`\rev{...}`).

---

## Comment 1 — Novelty (Hungarian + cosine LR)

**Reviewer ask:** Explain how LU-STW differs from existing alignment / personalized transfer; combining two techniques may not suffice.

**Code evidence**
| File | What it shows |
|------|----------------|
| `HNO3_CDR.py` | MF-HNO3/SNO3: stacked matrix + OT match + **uniform** fine-tune |
| `LU_STW_Novelty_Full.py` (`VARIANTS`) | Separate MF → Hungarian / greedy / random / Procrustes → **confidence** FT |
| `Ablation_Fair_*.py` | Removing Hungarian or confidence each raises MAE (Yelp ΔMAE +0.0669 / +0.0713) |

**Revision**
- Related Work + Table 1: separate vs stacked MF; map personalization vs **step-size** personalization; 2024–25 selective-transfer rows.
- Claim: novelty = ordered combination under overlapping-user MF, not either piece alone.

---

## Comment 2 — Hungarian vs Procrustes / rotation

**Reviewer ask:** Independently trained MF can differ by rotation; why is permutation enough? Compare orthogonal Procrustes.

**Code evidence**
| File | What it shows |
|------|----------------|
| `LU_STW_Novelty_Full.py` → `procrustes_perm`, variant `Procrustes+Hung` | Orthogonal rotate then Hungarian |
| `src/revision_experiments.py` → `orthogonal_procrustes` | Pure Procrustes ablation (no permutation) |
| Ablation `w/o Hungarian` (`align="identity"`) | No alignment baseline |

**Revision**
- Phase 2: defense of permutation (column semantics for ωᵤ; special case of O(k); O(k³)).
- Ablation list extended with Procrustes; Kuhn–Munkres citations added (`kuhn1955hungarian`, `munkres1957algorithms`).

**Pending numbers:** run `Procrustes` / `Procrustes+Hung` on Colab → fill ablation table.

---

## Comment 3 — Does ωᵤ indicate usefulness? Why frozen?

**Reviewer ask:** Show high-confidence users benefit more; explain why ωᵤ fixed during FT.

**Code evidence**
| File | What it shows |
|------|----------------|
| `LU_STW_Novelty_Full.py` → `confidence_analysis`, `user_conf` | MAE by High/Mid/Low cosine groups |
| `Ablation_Fair_*.py` | w/o Confidence vs Full (Yelp +0.0713 MAE) |
| Phase-3 code | `conf` computed once before SGD loop |

**Revision**
- Method text: source transfer = initialization; ωᵤ = target adaptation strength; frozen prior from Phase-1 Uᵀ.
- Stratified tercile analysis hooked in `confidence_tercile_gain`.

**Pending numbers:** write tercile ΔMAE table after Colab run.

---

## Comment 4 — ηᵤ on item factors / biases

**Reviewer ask:** Item updated by users with different ω; justify; compare user-only weighting.

**Code evidence**
| File | What it shows |
|------|----------------|
| All `fine_tune*` | Same `eu = eta * conf[u]` updates X, bu, **V, bi** |
| `revision_experiments.py` → `user_only=True` | Scales only user params; global η for items |

**Revision**
- Method paragraph: residual correction should not let low-ω users push shared Vᵢ at full rate.
- New ablation: “User-only ηᵤ”.

**Pending numbers:** Colab `User-only eta` vs `LU-STW`.

---

## Comment 5 — MovieLens data leakage

**Reviewer ask:** Romance∩Drama movies; source used every fold → leakage. Report overlap; re-run with no common movies.

**Code evidence**
| File | What it shows |
|------|----------------|
| `Movielens100K(LU_STW).ipynb` | Romance→Drama matrices; dual-genre titles kept in both |
| ML-100K analysis (this revision) | **99** dual-genre movies; **8,017** ratings = **20.1%** of Drama ratings; disjoint → 148/626 movies, 926 users |

**Revision (manuscript)**
- Exact overlap stats inserted in §4.1.2.
- Genre-disjoint protocol defined; primary ML results to use disjoint split.

**Pending numbers:** re-train LU-STW + baselines on disjoint matrices (`build_movielens_disjoint`).

---

## Comment 6 — Ranking metrics + domain-pair rationale

**Reviewer ask:** Add P@K, R@K, NDCG@K, MRR; explain why these source–target pairs.

**Code evidence**
| File | What it shows |
|------|----------------|
| `revision_experiments.py` → `ranking_at_k` | P/R/NDCG/MRR with 100 negatives |
| MovieLens notebook cell “cross_domain_pairs” | Pair justifications (affinity / density) |
| Paper Table 2 | Romance–Drama, Vet–Pet, VG–Toys |

**Revision**
- §Evaluation Metrics: ranking definitions.
- Domain-pair rationale paragraph (semantic relatedness + catalog disjointness for Yelp/Amazon).

**Pending numbers:** ranking tables for three datasets.

---

## Comment 7 — Fair baselines + statistics

**Reviewer ask:** NNDSVD / CDRS-GCN caveats; 5 CV folds not independent; multi-seed + better tests.

**Code evidence**
| File | What it shows |
|------|----------------|
| `Statistical_Significance_Test_Hungarian(LU_STW).ipynb` | Paired t / Cohen d / Holm |
| `Amazon_LightGCN_CV.py`, `MovieLens_LightGCN_CV.py` | GCN tuning grids |
| Paper §4.1.3 | Already notes NNDSVD f=10, CDRS-GCN task mismatch |

**Revision**
- New Table `tab:hyperparams` (from code HP blocks).
- Multi-seed protocol: S=5 seeds × 5 folds = 25 paired scores.

**Pending numbers:** multi-seed re-test JSON.

---

## Comment 8 — Ablation all densities + train/infer cost

**Reviewer ask:** Ablation only at 50%; Amazon gaps small; report train vs infer cost separately.

**Code evidence**
| File | What it shows |
|------|----------------|
| `Ablation_Fair.py`, `_Yelp.py`, `_MovieLens.py` | 50% TRAIN_FRAC; MAE table already in paper |
| `LU_STW_Novelty_Full.py` → `cv_study` timing | Wall-clock per variant |
| `revision_experiments.py` → `cv_report` | densities 25/50/75 + `train_s` |

**Revision**
- Complexity: training O(e(rₛ+rₜ)k+k²m+k³) vs inference O(k) per pair.
- Ablation text: extend to all densities; Amazon significance caveat kept.

**Pending numbers:** 25%/75% ablation + timing rows.

---

## Comment 9 — Literature 2023–2026

**Reviewer ask:** Add recent CDR on adaptation, interest alignment, selective / negative transfer.

**Revision done**
- FedGCDR (Yang et al., NeurIPS’24), MetaRec-style selective transfer, AutoCDSR, DisCo; Table 1 rows; positioning vs LU-STW MF pipeline.
- Classical Hungarian citations corrected (Kuhn 1955 / Munkres 1957).

---

## Ablation numbers already in paper (from `Ablation_Fair_*.py`, 50%)

| Dataset | Target Only Δ | w/o Hung Δ | w/o Conf Δ | w/o FT Δ | Full MAE |
|---------|---------------|------------|------------|----------|----------|
| Amazon | +0.0073 | +0.0059 | +0.0057 | +0.0051 | 0.7151 |
| Yelp | +0.0718 | +0.0669 | +0.0713 | +0.0718 | 0.7391 |
| MovieLens | +0.0385 | +0.0133 | +0.0052 | +0.0382 | 0.7046 |

---

## Next (Colab)

1. Open `notebooks/colab_run.ipynb` from this repo.  
2. Load disjoint ML + Yelp + Amazon matrices.  
3. Run `cv_report` / `confidence_tercile_gain` / `ranking_at_k`.  
4. Paste JSON into `results/` and update blue placeholders in `Paper.tex`.
