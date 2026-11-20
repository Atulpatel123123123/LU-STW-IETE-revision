# Response to Reviewer — TITR-2026-0877

Manuscript: *Confidence-Aware Learning of User-Specific Transfer Weights in Cross-Domain Recommendation Systems*

Revised text in `Paper.tex` is highlighted in **blue** (`\rev{...}`).

---

## Comment 1 — Novelty of Hungarian + cosine LR adaptation

**Response.** We clarify that novelty is the *ordered combination* under overlapping-user MF, not either technique alone:
1. Exact $k\times k$ Hungarian on **separately** trained factors (vs heuristic MF-mapping / EMCDR).
2. Vs MF-HNO3/SNO3: separate MF + per-user $\eta_u$, not joint stacked MF + uniform $\eta$.
3. Vs PTUPCDR/CATN: personalize the **step size**, not a learned bridge network.

Revised: Related Work (after Table 1) + comparative table rows for 2024–2025 selective-transfer methods.

---

## Comment 2 — Hungarian vs orthogonal Procrustes

**Response.** We add justification that permutation is a discrete special case of orthogonal maps, preserves column semantics for $\omega_u$, and is $O(k^3)$. We add a **Procrustes ablation** (Phase 2 replaced by orthogonal Procrustes; Phases 1 & 3 unchanged).

**Status:** Text done. Numbers → run `Codes/Revision_Experiments.py` (Procrustes variant).

---

## Comment 3 — Does confidence indicate usefulness? Why frozen?

**Response.**
- Stratified analysis: high vs low $\omega_u$ terciles → ΔMAE vs target-only.
- Frozen $\omega_u$: avoids coupling step size to evolving $X$; stable prior from Phase-1 $U^T$.

**Status:** Text done. Stratified table → experiment script.

---

## Comment 4 — Confidence LR on item parameters

**Response.** Same $\eta_u$ on $(X_u,b_u,V_i,b_i)$ so low-confidence users do not push shared item factors at full rate. Added **user-only $\eta_u$** control ablation.

**Status:** Text done. Numbers → experiment script.

---

## Comment 5 — MovieLens data leakage

**Response.** Acknowledge dual-genre overlap; adopt **genre-disjoint** protocol (drop Romance∩Drama movies from both domains). Report overlap extent; primary ML results on disjoint split.

**Status:** Protocol text done. Re-run MovieLens required.

---

## Comment 6 — Ranking metrics + domain-pair rationale

**Response.** Added Precision@K, Recall@K, NDCG@K, MRR ($K\in\{5,10\}$). Added rationale for Romance→Drama, Vet→Pet Services, VG→Toys.

**Status:** Metric definitions + rationale done. Ranking tables → experiment script.

---

## Comment 7 — Fair baselines + multi-seed stats

**Response.** Keep NNDSVD/CDRS-GCN caveats. Add $S=5$ seeds × 5 folds = 25 paired scores; Holm on those. Original 5-fold tests kept as secondary.

**Status:** Protocol text done. Re-run stats required.

---

## Comment 8 — Ablation at all densities + train/infer cost

**Response.** Ablations at 25/50/75%; complexity split into training $O(e(r_s+r_t)k+k^2m+k^3)$ vs inference $O(k)$ per pair; wall-clock table with ablation.

**Status:** Text done. Full-density ablation + timing → experiment script.

---

## Comment 9 — Literature 2023–2026

**Response.** Added FedGCDR, MetaRec-style selective transfer, AutoCDSR, DisCo; Table 1 rows; positioning vs LU-STW.

**Status:** Done in Related Work + `cas-refs.bib`.

---

## Experiment checklist (Colab)

- [ ] Genre-disjoint MovieLens MAE/RMSE (+ baselines)
- [ ] Ranking metrics all datasets
- [ ] Procrustes vs Hungarian
- [ ] Confidence tercile analysis
- [ ] User-only $\eta_u$ ablation
- [ ] Ablation @ 25% and 75%
- [ ] Multi-seed significance
- [ ] Train / infer wall-clock
