# Night session 2026-05-08 — sequence-coupled meta probe (in flight)

**Author:** `claude/ml-model-experiments-gbKiI`.
**PI directive:** "show me original work … investigate, be creative, work all night autonomously, surprise me. do not replicate or copy."

This file is a self-contained progress report at the time of writing. Update it again at session-end.

---

## What was different this session

Eight days of internal-mechanism research had nulled progressively, with all five new prongs (FastF1 aggregates, per-segment LightGBM head, sequence transformer, 2-layer stack replication, blend-with-public) returning SKIP from the BOTE harness. PI declined the public-blend route on principle and challenged the agent to work autonomously on something genuinely original.

The agent paused, re-read the rank-lock argument (assumption A29: "any base whose row-level prediction lies in the K=10 [P, rank, logit] = 30-feature expansion gets absorbed"), and noticed a hidden premise: **row independence at the meta**. Every prior meta feature in the project has been a function of a single row's K-base predictions. The 30-dim row-local span only covers row-LOCAL features.

**Sequence-coupled meta features escape that span by construction.** A feature like "K=4 PRIMARY's prediction at the next observed (Driver, Race, Year, LapNumber) row" cannot be reconstructed from a linear combination of *this* row's K=4 [P, rank, logit]. The d16 GRU at the BASE layer was null because GRU outputs an independent row-level prediction; meta-level coupling is structurally different.

## What the diagnostic confirmed

`scripts/seq_coupled/diag_lookahead_orthogonality.py`:

| Quantity | Value |
|---|---:|
| Train rows | 439,140 |
| Rows with look-ahead defined | 90.7% |
| **R² of look-ahead regressed on K=4 [P, rank, logit]** | **0.487** |
| Fraction of look-ahead variance NOT explained by row-local | **51.3%** |
| Bare AUC of look-ahead alone | 0.853 |
| Pearson r(look-ahead, same-row K=4) | 0.658 |

51% of look-ahead variance is genuinely outside the row-local span — exactly what the structural argument predicted.

## What the meta-fits delivered

Anchor: K=4 PRIMARY (Path-B Compound × Stint τ=100k) OOF AUC = **0.95403**.

| Variant | Features | OOF AUC | Δ vs anchor | Δ vs row-local |
|---|---|---:|---:|---:|
| V1.1 LR row-local — anchor for delta | 12 | 0.95400 | −0.25 bp | — |
| **V1.2 LR full (sequence-coupled)** | **36** | **0.95401** | **−0.22 bp** | **+0.03 bp** |
| V1.3 RankNet pairwise (direct AUC) | 36 | (running) | — | TIE on f0/f1/f2 |
| V2.1 LR + hand interactions | 47 | 0.95400 | −0.30 bp | +0.00 bp |
| V2.2 Two-stage residual LR | 12 + 24 | 0.95369 | −3.43 bp | −3.13 bp |
| V2.3 LightGBM meta (depth 4) | 36 | 0.95397 | −0.59 bp | −0.30 bp |
| V3 LR + kNN target-mean | 14 | (running) | — | — |
| V4 kNN-augmented BASE (LightGBM) | 13 | (queued) | — | — |

V1.3 fold 0 AUC 0.95495 ≈ V1.2 fold 0 AUC 0.95496 — direct AUC objective produces the same answer as binary cross-entropy. RankNet not extracting anything LR misses on the 36-feature input.

V2.3 LightGBM meta fold 0/1: 0.95491 / 0.95299 vs LR's 0.95494 / 0.95312 — tree depth doesn't extract more.

## What this teaches us

The sequence-coupled features ARE outside the K=4 row-local span (R² = 0.487). The LR meta extracted +0.03 bp from them (within fold-noise). RankNet (direct AUC) and LightGBM (non-linear) both arrived at the same ceiling. LR + hand interactions added nothing. Residual-LR regressed.

**The corrected framing of A29:** rank-lock at K=4 is at the **conditional-target-correlation level**, not just the logit-direction level. Even features with substantial non-overlap variance against the existing meta input get absorbed if their target-correlation is parallel to the existing logit direction conditional on row context.

In information-theoretic terms: H(y | K=4 predictions, row_features) ≥ H(y | K=4 predictions, row_features, seq_coupled_features). The conditional entropy doesn't shrink when seq-coupled features are added — they don't reduce the meta's residual uncertainty.

This is a stronger negative finding than what was previously documented:

1. **No row-local OR sequence-local meta-feature engineering can break the K=4 ceiling.** Any new meta feature that's a function of (this row's K=4 predictions, this row's raw features, neighbour predictions, neighbour features) will be absorbed.
2. **The path forward must add SOURCE INFORMATION not in row features.** Three known classes:
   a. External data (FastF1 — driver-row-level cap 1.4%; aggregate level untested but PI declined the route).
   b. Transductive features that use TRAINING LABELS not derivable from K=4 predictions (V3 kNN target-mean — running; V4 kNN-augmented BASE — ready).
   c. A genuinely new base trained on a different DGP factorisation (chain decomposition v2; orig-data future-feature joint; etc).

V3 (kNN target-mean at meta) is testing class (b) at the meta layer. V4 (kNN-augmented BASE) tests class (b) at the base layer with the same transductive label info but ingested non-linearly through tree splits.

## Implications for ASSUMPTIONS.md (proposed)

A29 should read: "Rank-lock at K=4 is at the **conditional-target-correlation** level. New row-local meta features whose target-correlation is parallel to the existing logit direction get absorbed even when their feature-space orthogonality vs row-local features is high (R²=0.487 confirmed). Loss-function variation at the meta (RankNet pairwise direct-AUC) doesn't escape the ceiling — the limitation is information, not optimisation. The path to break the ceiling is source information not derivable from row features (training labels via transduction, external data, alternative DGP factorisations)."

A30 (already FALSIFIED) extends with: even RankNet (direct AUC objective) and LightGBM-meta on enriched 36-feature input produce no lift over LR row-local. Loss-function diversity at the meta layer is empirically dead at K=4.

## Code artefacts (committed to `claude/ml-model-experiments-gbKiI`)

- `scripts/seq_coupled/diag_lookahead_orthogonality.py` — diagnostic
- `scripts/seq_coupled/build_features.py` — 24-feature builder
- `scripts/seq_coupled/fit_meta.py` — V1 (LR + RankNet, 5-fold OOF)
- `scripts/seq_coupled/fit_meta_v2_interactions.py` — V2 (LR+inter, residual, LGBM)
- `scripts/seq_coupled/fit_meta_v3_knn.py` — V3 (kNN target-mean at meta)
- `scripts/seq_coupled/build_knn_base.py` — V4 (kNN-augmented base)

Decision-log entries (in `audit/decisions.jsonl`): seq_coupled_v1_lr_full, seq_coupled_v2.

## Final results — full table

Anchor: K=4 PRIMARY (Path-B Compound × Stint τ=100k) OOF AUC = **0.95403**.

| Variant | Features | OOF AUC | Δ vs PRIMARY | Δ vs LR-row-local | Verdict |
|---|---|---:|---:|---:|---|
| V1.1 LR row-local | 12 | 0.95400 | −0.25 bp | — (anchor) | — |
| V1.2 LR full | 36 | 0.95401 | −0.22 bp | +0.03 bp | WEAK |
| V1.3 RankNet pairwise | 36 | 0.95399 | −0.37 bp | −0.12 bp | FAIL |
| V2.1 LR + interactions | 47 | 0.95400 | −0.30 bp | +0.00 bp | TIE |
| V2.2 Two-stage residual LR | 12 + 24 | 0.95369 | −3.43 bp | −3.13 bp | REGRESS |
| V2.3 LightGBM meta | 36 | 0.95397 | −0.59 bp | −0.30 bp | REGRESS |
| V3 LR + kNN target-mean | 14 | 0.95401 | −0.17 bp | +0.01 bp | WEAK |
| **V4 K=4+1 LR meta with V4 base** | **15** | **0.95402** | **−0.01 bp** | **+0.24 bp** | **WEAK** |
| **V4 K=5 + Path-B C×S τ=100k** | (production) | **0.95405** | **+0.20 bp** | — | **WEAK** |

Bare V4 standalone OOF AUC: **0.94163**. ρ_spearman vs K=4 bases 0.85–0.94 (most diverse vs d16_orig at 0.85).

## What V4 actually demonstrated

V4 (kNN target-mean ingested through tree splits inside a new LightGBM base) is the **first thing in 8 days that consistently beats LR row-local at K=4+1**, with all 5 folds showing a positive lift (range +0.12 to +0.44 bp; mean +0.24 bp).

The V3 vs V4 contrast is structurally important:
- V3 used the SAME kNN-target-mean as a META feature → +0.01 bp NULL.
- V4 used the SAME kNN-target-mean as a BASE INPUT FEATURE, ingested through LightGBM tree splits → +0.24 bp consistently.

This refines the structural finding: **tree non-linearity inside a base CAN retrieve signal that linear meta absorption blocks**. But the magnitude is small.

## Why no submission

K=5 + Path-B production OOF lift is +0.20 bp vs K=4 PRIMARY. Pre-submit prediction diff: **ρ_test_K5_vs_K4 = 0.999891** — exceeds Rule 27's 0.999 abort threshold. Per Rule 27, LB will tie. **Submission file built but held.**

## Refined ASSUMPTIONS.md candidate updates

A29 should now read: "Rank-lock at K=4 is at the **conditional-target-correlation** level. New row-local meta features whose target-correlation is parallel to the existing logit direction get absorbed even when their feature-space orthogonality vs row-local features is high (R²=0.487 confirmed). RankNet (direct AUC objective) does not escape it. **Tree non-linearity at the BASE layer (not meta) extracts a small +0.20 bp from transductive label-derived features, but the OOF→LB transfer at K=4+1 sits inside Rule 27's abort zone (ρ_test 0.99989).** The path to a leader-bridging lift is source information not derivable from row features."

## Code artefacts (committed to `claude/ml-model-experiments-gbKiI`)

- `scripts/seq_coupled/diag_lookahead_orthogonality.py` — diagnostic
- `scripts/seq_coupled/build_features.py` — 24-feature builder
- `scripts/seq_coupled/fit_meta.py` — V1 (LR + RankNet, 5-fold OOF)
- `scripts/seq_coupled/fit_meta_v2_interactions.py` — V2 (LR+inter, residual, LGBM)
- `scripts/seq_coupled/fit_meta_v3_knn.py` — V3 (kNN target-mean at meta)
- `scripts/seq_coupled/build_knn_base.py` — V4 (kNN-augmented base)
- `scripts/seq_coupled/gate_K4_plus_1.py` — K=4+1 plain LR-meta gate
- `scripts/seq_coupled/path_b_K5.py` — production Path-B C×S K=5 candidate
- `submissions/submission_K5_kNNaugbase_pathb.csv` — held, NOT submitted

## Path forward (no internal-mechanism lift left)

The empirical conclusion of the night is sharper than what was in the handover. The K=4 conditional-target-correlation ceiling holds even against:
- Sequence-coupled meta features (V1.1, V1.2)
- Direct AUC pairwise loss (V1.3)
- Hand-crafted interactions (V2.1)
- Two-stage residual LR (V2.2)
- LightGBM meta (V2.3)
- Transductive label-derived meta features (V3)

It is breached only by:
- Transductive label info ingested non-linearly inside a NEW BASE (V4) — but the lift is +0.20 bp at Path-B, inside Rule 27's tie-zone.

Three avenues remain:
1. **External aggregate data** at (Race, Year, Lap) granularity — explicitly UN-tested. PI declined the route earlier in the session, but the V4 finding suggests a tree-base that ingests external data via tree splits could break the ceiling more substantially than V4 did.
2. **Wrap-up posture** — final R5 hedge-ladder preparation and the two final-submit slots. The K=5 file is now a hedge candidate.
3. **Fundamentally new base on different DGP factorisation** — e.g., chain decomposition v2 conditioning on different orderings; orig-data future-feature joint at aggregate level.
