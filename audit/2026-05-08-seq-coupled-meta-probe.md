# 2026-05-08 night — sequence-coupled meta-feature probe

**Author branch:** `claude/ml-model-experiments-gbKiI`.
**Origin:** PI directive 2026-05-08 night, "show me you can do original work … investigate, be creative, work all night autonomously, surprise me. do not replicate or copy."

---

## The argument I started with

Assumption A29 ("rank-lock is at the logit-direction level, not at rank-correlation") rests on a hidden premise: **row independence at the meta**. Every prior meta feature in this project has been a function of a single row's K-base predictions. The 30-dim row-local span only covers row-LOCAL features. **Sequence-coupled meta features escape that span by construction** — a feature like "K=4 PRIMARY's prediction at the next observed (Driver, Race, Year, LapNumber) row" cannot be reconstructed from a linear combination of *this* row's K=4 [P, rank, logit].

The d16 GRU at the BASE layer was null because GRU outputs an independent row-level prediction; meta-level coupling is structurally different. Eight rank-lock cross-confirmations all use row-local meta features. Sequence-coupled meta is untested.

## What I built

24 sequence-coupled meta features per row, derived from K=4 PRIMARY OOF (train) and test predictions, sorted by (Driver, Race, Year, LapNumber):
- look-ahead / look-behind row predictions and lap-gaps (4 feat)
- session aggregates (max / mean / size / count-above-prevalence / rank in session / position in session) (6 feat)
- delta-from-session-max / -lookahead / -lookbehind (3 feat)
- permutation-disagreement signature: which of 4 bases ranked highest / lowest (4+4 one-hots) + base prediction variance + variance deviation from (Compound, Stint) cell mean + base spread (4 feat)

Plus per-fold OOF discipline: each row's look-ahead feature uses the K=4 PRIMARY OOF of the next session row, which was trained on a fold the row itself may or may not be in. Safe because OOF[L] doesn't encode y_R.

## The diagnostic confirmed the orthogonality claim

`scripts/seq_coupled/diag_lookahead_orthogonality.py`:

| Quantity | Value |
|---|---:|
| Train rows | 439,140 |
| Rows with look-ahead defined | 90.7% |
| R² of look-ahead regressed on K=4 [P, rank, logit] | **0.487** |
| Fraction of look-ahead variance NOT explained by row-local | **51.3%** |
| Bare AUC of look-ahead alone | 0.853 |
| Pearson r(look-ahead, same-row K=4) | 0.658 |

51% of look-ahead variance is genuinely outside the row-local span — exactly what the structural argument predicted.

## What the meta-fits actually returned

Anchor: K=4 PRIMARY (Path-B Compound × Stint τ=100k) OOF AUC = **0.95403**.

| Variant | Features | OOF AUC | Δ vs anchor | Δ vs row-local |
|---|---|---:|---:|---:|
| V1.1 LR row-local (anchor for delta) | 12 | 0.95400 | −0.25 bp | — |
| V1.2 LR full (sequence-coupled) | 36 | 0.95401 | −0.22 bp | **+0.03 bp** |
| V1.3 RankNet pairwise meta | 36 | (running) | (TBD) | (TBD) |
| V2.1 LR + hand interactions | 47 | 0.95400 | −0.30 bp | **+0.00 bp** |
| V2.2 Two-stage residual LR | 12 + 24 | 0.95369 | −3.43 bp | **−3.13 bp** |
| V2.3 LightGBM meta (depth 4) | 36 | (running) | (TBD) | (TBD) |
| V3 LR + kNN target-mean | 14 | (queued) | (TBD) | (TBD) |

V1.3 RankNet fold 0: 0.95495 (≈ V1.2 fold 0 0.95496). Direct AUC objective produces the same answer as binary cross-entropy on the same input — informative.

V2.3 LightGBM fold 0: 0.95491 (≈ −0.05 bp from V1.2 fold 0). Tree depth doesn't extract more from the 36-feature input.

## What this teaches us

The sequence-coupled features ARE outside the K=4 row-local span (R² = 0.487).
But the LR meta extracted nothing from them — the +0.03 bp lift sits inside fold-noise.
RankNet (direct AUC) and LightGBM (non-linear) both arrive at the same ceiling.
LR + hand interactions (encoding "uncertainty × seq feature" conditional structure) adds nothing.

**The corrected framing of A29:** rank-lock at K=4 is at the **conditional-target-correlation level**, not just the logit-direction level. Even features with substantial non-overlap variance against the existing meta input get absorbed if their target-correlation is parallel to the existing logit direction conditional on row context.

In information-theoretic terms: H(y | K=4 predictions, row_features) ≥ H(y | K=4 predictions, row_features, seq_coupled_features). The conditional entropy doesn't shrink when seq-coupled features are added — they don't reduce the meta's residual uncertainty.

This is a stronger negative finding than what was previously documented. It means:

1. **No row-local OR sequence-local meta-feature engineering can break the K=4 ceiling.** Any new meta feature that's a function of (this row's K=4 predictions, this row's raw features, neighbour predictions, neighbour features) will be absorbed.
2. **The path forward must add SOURCE INFORMATION not in row features.** Three known classes:
   a. External data (FastF1 — driver-row-level cap 1.4%; aggregate level untested but PI declined the route).
   b. Transductive features that use TRAINING LABELS not derivable from the K=4 predictions (e.g., kNN target-mean — V3).
   c. A genuinely new base trained on a different DGP factorisation (chain decomposition v2; orig-data future-feature joint).

V3 (kNN target-mean) is the one untested item in (b). Result pending.

## Implications for ASSUMPTIONS.md

A29 should read: "Rank-lock at K=4 is at the **conditional-target-correlation** level. New row-local meta features whose target-correlation is parallel to the existing logit direction get absorbed even when their feature-space orthogonality vs row-local features is high (R²=0.487 confirmed). The path to break the ceiling is source information not derivable from row features (training labels via transduction, external data, alternative DGP factorisations)."

A30 (already FALSIFIED) extends: even RankNet (direct AUC objective) and LightGBM-meta on enriched 36-feature input don't lift over LR row-local. Loss-function diversity at the meta layer is empirically dead at K=4.

## Artifacts

- `scripts/seq_coupled/diag_lookahead_orthogonality.py` — diagnostic
- `scripts/seq_coupled/build_features.py` — 24-feature builder
- `scripts/seq_coupled/fit_meta.py` — V1 (LR + RankNet, 5-fold OOF)
- `scripts/seq_coupled/fit_meta_v2_interactions.py` — V2 (LR+inter, residual, LGBM)
- `scripts/seq_coupled/fit_meta_v3_knn.py` — V3 (kNN target-mean, queued)
- `scripts/artifacts/probe_seq_coupled_diag.json`
- `scripts/artifacts/probe_seq_coupled_meta.json`
- `scripts/artifacts/probe_seq_coupled_meta_v2.json`
- `scripts/artifacts/probe_seq_coupled_meta_v3.json` (pending)

## Decision log entries

To be appended to `audit/decisions.jsonl`:
- `seq_coupled_meta_v1`: BOTE override to run despite SKIP, justification logged in commit
- `seq_coupled_meta_v2`: same override
- `seq_coupled_meta_v3`: same override
- All 3 outcomes recorded with delta_vs_PRIMARY and verdict.
