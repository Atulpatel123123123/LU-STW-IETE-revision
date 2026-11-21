# Response to Reviewers

**Manuscript ID:** TITR-2026-0877  
**Title:** Confidence-Aware Learning of User-Specific Transfer Weights in Cross-Domain Recommendation Systems  
**Journal:** IETE Technical Review  

Dear Editor,

Thank you for the reviews. Every comment is answered in the table below. The location of each change in the revised manuscript is given in brackets. Revised text is marked in blue.

Sincerely,  
Vibhor Kant  
Corresponding Author  

**Reviewer:** 1  

| Comment | Response |
|---|---|
| 1. The novelty of combining Hungarian-based latent-factor alignment with cosine-similarity-based learning-rate adaptation is not clearly established. Explain how the method differs from existing cross-domain alignment and personalized transfer methods. | Thank you for this comment. The novelty is rewritten as an ordered three-phase pipeline, and LU-STW is clearly compared with existing alignment and personalized-transfer methods. (Section 2 — Related Work, novelty paragraphs) (Table 1 — Novelty positioning) |
| 2. Independently trained MF may differ by rotation. Explain why permutation-based alignment is sufficient and compare it with orthogonal Procrustes. | Thank you for raising this point. Why permutation is sufficient is explained, and an orthogonal Procrustes comparison is added. (Section 3.4 — Phase 2, Permutation alignment versus orthogonal Procrustes) (Table 3 — Phase 2 alignment variants) |
| 3. Show whether the confidence score indicates how useful source information is for a user, and explain why the score remains unchanged during fine-tuning. | We appreciate this request. A tercile analysis is added, and the fixed-ωu choice is explained. (Section 3.5 — Phase 3, Relation between ωu and transfer benefit) (Table 15 — ωu tercile analysis) (Section 6 — Limitations and Future Directions) |
| 4. Explain why the confidence-based learning rate is applied to item factors and biases, and compare with user-only weighting. | Thank you for this suggestion. The choice is explained, and a user-only weighting comparison is added. (Section 3.5 — Phase 3, Confidence weighting of item factors and item biases) (Table 4 — Full versus user-only confidence weighting) |
| 5. Report the Romance–Drama movie overlap and repeat the experiments with no common movies. | Thank you for this careful reading. The overlap is reported, and a no-common-movies re-run is added. (Section 4.1 — Experimental Setup, Item-overlap quantification) (Table 6 — Overlapping-genre vs genre-disjoint) (Section 6 — Limitations and Future Directions) |
| 6. Report Precision@K, Recall@K, NDCG@K, and MRR, and explain the choice of source–target pairs. | Thank you for this suggestion. Ranking metrics are added, and the domain-pair choice is explained. (Section 4.2 — Evaluation Metrics) (Section 4.1 — Experimental Setup, Rationale for source–target pairs) (Table 10 — Ranking metrics) |
| 7. Ensure fair baseline comparisons and conduct multi-seed experiments with a more suitable statistical analysis. | Thank you for this comment. Fair baseline settings are clarified, and a multi-seed statistical check is added. (Section 4.1 — Experimental Setup, Baseline fairness and multi-seed testing) (Table 7 — Hyperparameter and reproducibility settings) (Table 12 — Multi-seed evaluation) |
| 8. Repeat the ablation at all three training densities, check the small Amazon improvements, and report training and inference costs separately. | Thank you for these points. Ablation is repeated at all three densities, the Amazon gaps are checked against fold variation, and training and inference costs are reported separately. (Section 3.6 — Computational Complexity Analysis, Training versus inference) (Table 13 — Ablation at 50%) (Table 14 — Ablation at 25% and 75%) |
| 9. Include more recent work (2023–2026) on domain adaptation, interest alignment, selective knowledge transfer, and negative transfer, and compare LU-STW with them. | Thank you for this suggestion. Recent work is discussed and cited, and LU-STW is compared with these approaches. (Section 2 — Related Work, Personalized Cross-Domain Transfer) (Table 2 — Summary of previous models) |
