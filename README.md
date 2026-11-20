# LU-STW IETE Revision (TITR-2026-0877)

Revision package for *Confidence-Aware Learning of User-Specific Transfer Weights in Cross-Domain Recommendation Systems* (IETE Technical Review).

**Deadline:** 16-Dec-2026

## Repo layout

| Path | Purpose |
|------|---------|
| `src/` | LU-STW core + revision experiments (Procrustes, ranking, multi-seed, disjoint ML) |
| `notebooks/` | Colab / Jupyter runners |
| `docs/` | Point-by-point response to reviewers |
| `results/` | JSON/CSV outputs (gitignored large files; small summaries ok) |
| `data/` | Placeholders + download notes (raw data not committed) |

## Manuscript

Overleaf: `https://git.overleaf.com/6a64630014a23ec940288d02`  
Local clone may live next to this folder as `../6a64630014a23ec940288d02/`.

## Colab (recommended)

1. Open `notebooks/colab_run.ipynb` from this repo in Colab.
2. Run all (clones repo, installs deps, runs revision suite).
3. Download `results/*.json` and paste numbers into the manuscript.

Or:

```bash
git clone https://github.com/Atulpatel123123123/LU-STW-IETE-revision.git
cd LU-STW-IETE-revision
pip install -r requirements.txt
# then load R_s, R_t and call src.revision_experiments.cv_report(...)
```

## Reviewer experiment checklist

- [ ] R2 Procrustes vs Hungarian
- [ ] R3 Confidence tercile analysis
- [ ] R4 User-only ηᵤ ablation
- [ ] R5 Genre-disjoint MovieLens
- [ ] R6 Precision/Recall/NDCG/MRR @K
- [ ] R7 Multi-seed significance
- [ ] R8 Ablation @ 25/50/75% + train/infer time
- [ ] R9 Related work (manuscript already updated)

## Citation

Manuscript ID: **TITR-2026-0877**
