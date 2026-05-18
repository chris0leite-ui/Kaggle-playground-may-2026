# 2026-05-18 — Round-6 execution audit

## Result headline

**PRIMARY unchanged at LB 0.95387** (R5.2 K=13 + Path-B τ=100k).
R6.1 fold-fit bag tied LB; R6 Phase A retest 5/5 NULL under Path-B
operator (operator-axis hypothesis falsified for these candidates).

Top-5% gap unchanged: 1.8 bp (boundary 0.95405).

## Phase results

### Phase A — Operator-axis retest (5 candidates, 45 min CPU)

`scripts/probe_r6_operator_axis_retest.py`. Re-gated 5 prior LR-meta
nulls at K=13+Path-B τ=100k vs R5.2 baseline OOF 0.95446.

| candidate | OOF | Δ vs R5.2 | verdict |
|---|---|---|---|
| conformal_widths | 0.95446 | -0.028 bp | NULL |
| rrf_k60          | 0.95446 | -0.026 bp | NULL |
| meta_lgbm_rank   | 0.95446 | -0.032 bp | NULL |
| trimmed_rank     | 0.95445 | -0.069 bp | NULL |
| seg_fe_v2        | 0.95445 | -0.090 bp | NULL |

**5 of 5 NULL under Path-B operator.** The operator-axis hypothesis
(that LR-meta nulls might survive under Path-B) is FALSIFIED for
these specific candidates. The +5 bp LB swing in R5 (LR-meta vs
Path-B at K=13) was **pool-composition-specific** (seg + HMM
mechanism-orthogonality interacting with Path-B's per-segment
shrinkage), not a general "Path-B preserves more signal" pattern.

### Phase B — Fold-fit bagging (~7 min CPU)

`scripts/build_K13_seghmm_pathb_foldbag.py`. Implements proper
multi-seed bagging by replacing `run_pathb`'s seed-invariant
full-train fit with **per-fold per-seed test-prediction averaging**.

| seed | OOF |
|---|---|
| 42 | 0.95446 |
| 43 | 0.95447 |
| 44 | 0.95448 |
| 45 | 0.95449 |
| 46 | 0.95448 |
| **bag (5-seed)** | **0.95448** (+0.212 bp vs single-seed R5.2) |

**Bag predictions DIFFER from R5.2** (ρ=0.999988 vs ρ=1.0 from R5's
broken full-train bag). The fix worked: this is genuine variance
reduction, not a no-op.

LB submission: **R6.1 LB 0.95387 — ties R5.2 within 5-decimal
quantization.** TIE_ZONE prediction confirmed (ρ 0.999988 ≥ 0.9999).
The +0.212 bp OOF improvement did not translate to LB lift at the
5-decimal precision level; it may register on private LB
(lower-noise estimate).

### Phase C — Transformer v2 on Kaggle T4 (~3 hr T4)

`kernels/r6-transformer-v2-gpu/r6_transformer_v2.py`. Fixes vs v1:
- D_MODEL=256 (was 128), N_LAYERS=6 (was 4), EPOCHS=15 (was 5).
- **GroupKFold by (Year, Race, Driver) sequence** (was Stratified
  per row — leaky for sequence learning).

| metric | v1 (R5) | v2 (R6) |
|---|---|---|
| Standalone OOF | 0.91974 | **0.93330** (+13.5 bp) |
| Fold split | Stratified per row | GroupKFold by sequence |
| Architecture | 4×128 | 6×256 |

Standalone OOF improved by +13.5 bp despite the structurally
harder GroupKFold split (no intra-sequence leakage from train to
val). Confirms larger arch + proper split was the right v2 design.

**At K=14+Path-B (R5.2 pool + TRFv2): Δ -0.014 bp** — absorbed at
meta. Standalone OOF 0.93330 is still 21 bp below K=11 baseline
(0.95443); transformer doesn't reach the meta-utility threshold.

### Phase D — Multi-class super-stack combos

Tried K=14 fold-fit bag (R5.2 pool + TRFv2 + bagged):
- OOF 0.95448 (same as Phase B alone)
- vs Phase B: Δ -0.033 bp (transformer absorbs)
- vs R5.2: Δ +0.179 bp
- ρ vs Phase B: 0.999992 (TIE_ZONE)

**Conclusion**: Phase B's fold-bag captures the full Phase B+D
value; transformer adds nothing on top under Path-B.

## Submissions this iteration

| ref | LB | description |
|---|---|---|
| **R6.1 52776849** | **0.95387** | K=13+Path-B 5-seed fold-fit bag (ties R5.2) |

Total daily: 5/10; total comp: 47/270.

## Strategic verdict

Round 6 completes the operator-axis exploration: the +5 bp LB
swing from R5 is **pool-composition-specific**, not generalizable
across mechanisms. Row-feature ceiling holds under BOTH operator
classes for the 5 retested candidates.

Fold-fit bagging is technically validated as a true variance-
reduction mechanism (bag predictions ≠ single-seed; OOF lifts by
+0.212 bp). At 5-decimal LB resolution this doesn't move PRIMARY,
but it's a structurally-distinct hedge candidate for the final-window
R7d submission.

**Top-5% (1.8 bp gap) remains out of reach** via:
- Mechanism-class retest at K=11 (all 5 candidates null today)
- Operator-class swap (validated already in R5)
- Standard variance reduction (LB-quantized away)

Realistic remaining paths to top-5% (P estimates):
1. **C2 swap-noise DAE on Kaggle T4** — embedding-class mechanism
   class not yet tested. P≈25% at +0.3-0.5 bp.
2. **Multi-segmentation Path-B** — Year×Compound, Driver-cluster×Stint.
   P≈20% at +0.1-0.3 bp.
3. **OpenF1 C1 per-Race external data** — 1.4% match cap.
   P≈15% at +0.1-0.2 bp.
4. **Multi-pool rank-blend (R5.2 + R6.1 fold-bag + new mechanism)** —
   exploits the structurally-distinct fold-bag for private LB.

## Files touched

New:
- `scripts/probe_r6_operator_axis_retest.py` — Phase A
- `scripts/build_K13_seghmm_pathb_foldbag.py` — Phase B fold-fit bagging
- `kernels/r6-transformer-v2-gpu/r6_transformer_v2.py` — Phase C
- `kernels/r6-transformer-v2-gpu/kernel-metadata.json` — Phase C
- `audit/2026-05-18-round-6-execution.md` — this file
- `audit/2026-05-18-round-6-phase-a.json` — Phase A results
- `audit/2026-05-18-round-6-phase-b.json` — Phase B results
- `submissions/submission_K13_seghmm_pathb_foldbag_tau100000.csv` — R6.1 (LB 0.95387)
- `submissions/submission_K14_seghmm_trf_pathb_tau100000.csv` — held (Phase C marginal)
- `submissions/submission_K14_seghmm_trf_pathb_foldbag.csv` — held (Phase D combo)

Artifacts:
- `scripts/artifacts/oof_K14_r5plus_{conformal_widths,rrf_k60,meta_lgbm_rank,trimmed_rank,seg_fe_v2}_pathb_strat.npy` — Phase A retest OOFs (all null)
- `scripts/artifacts/oof_K13_seghmm_pathb_foldbag_strat.npy` + test — Phase B bag
- `scripts/artifacts/oof_K14_seghmm_trf_pathb_foldbag_strat.npy` + test — Phase D combo
- `scripts/artifacts/oof_r6_transformer_v2_strat.npy` + test — Phase C v2
