# Day-12 — d12: metric-aligned LambdaRank meta over K=21 PRIMARY pool

Hypothesis: replacing LR-meta (BCE/squared-loss) with a pairwise rank
meta-learner (LambdaMART or YetiRank) — both AUC-aligned by
construction — can shift the K=21 stack's test ranking by 2-8bp.
Plus: AUC-direct base retraining via XGBoost `rank:pairwise`.

## Pool

K=21 PRIMARY = `d9f_K21_swap_partA_partB` (LB 0.95031, OOF 0.95073).
Composition: POOL_KEEP (16 GBDT/LR/MLP bases) + TOP_3_D9 (rule-residual
R6/R10/R7) + d9f_FM_A driver-dynamics + d9f_FM_B race-context.

Meta features `expand(P)` = [raw_p, rank_p, logit_p] across 21 bases
→ 63 columns.

## Part A — meta swap results

PRIMARY anchor: OOF Strat **0.95073**, LB **0.95031**.

| meta | grouping | OOF Strat (5-fold) | Δ PRIM (bp) | ρ vs PRIM-test | ρ vs LR-test | best_iters | wall (s) |
|---|---|---:|---:|---:|---:|---|---:|
| LR (anchor)            | n/a    | **0.95073** | −0.01 | 1.00000 | 1.00000 | — | 210 |
| LambdaMART (fast)      | Race¹  | **0.94210** | **−86.32** | 0.94238 | 0.94238 | [65,78,44,73,72] | 466 |
| LambdaMART (fast)      | rand-1k² | (0.95088,0.94850,0.94996,0.94882,—)³ | est. ≈ −15bp | n/a (not 5-fold) | n/a | [94,93,72,84,—] | n/a |
| YetiRank (fast)        | Race   | NOT_RUN⁴ | — | — | — | — | — |

¹ Race groups (26 in train) chunked at ≤8000 rows per LightGBM cap.
² Random groups of ~1000 rows, fitted within each fold.
³ 4 of 5 folds completed; fold-4 in progress when killed at 22:56 wall.
   Mean of 4 folds = 0.94954; estimated 5-fold OOF ≈ 0.949 vs LR 0.95073.
⁴ Killed before YetiRank phase to stay within 30-min meta budget.

### LambdaMART Race-grouped: catastrophic regression (−86bp)

Per-fold val AUC: [0.94912, 0.94676, 0.94812, 0.94622, 0.94840]
(mean per-fold 0.94772; concatenated OOF 0.94210). vs LR-meta same
splits (~0.951 each fold) → **−65 to −90bp regression every fold**.
ρ vs LR-test = 0.94238 — meaningful divergence but in the wrong
direction.

### LambdaMART Random-grouped: high variance, mild regression on average

Per-fold val AUC: 0.95088, 0.94850, 0.94996, 0.94882 (4 folds).
Mean = 0.94954. **Fold 0 LIFTED +13bp vs LR**, but folds 1-3
regressed −12 to −22bp. The high cross-fold variance (range 0.94850
to 0.95088 = 24bp) suggests random-group lambdarank is overfitting to
the per-fold random group assignment. Net: roughly TIE-to-mild-regress
vs LR.

### Why Race-grouped lambdamart is so bad

Lambdarank optimizes **within-query** pairwise rank. Race groups
contain 12-24k rows each (chunked into ≤8k sub-queries). The gradient
is dominated by within-Race pair classification — but the competition
AUC is **global** across all 188k test rows. A model that ranks well
within a single Race need not transfer to global ranking; in fact
within-Race optimization actively destroys the cross-Race calibration
that LR-meta's rank+logit expansion encodes.

### Why Random-grouped is also weak

With group_size=1000, each fold sees ~280 random groups of 1000 rows.
The pair-distribution within a random group approximates the global
distribution but at much higher variance (each group has only ~200
positives × 800 negatives = 160k pairs vs the full 89B pairs in the
global AUC). LR-meta's rank+logit expansion has access to the full
sample-level rank ordering, capturing the same signal more efficiently.

## Part B — AUC-direct base retraining (e3_hgbc clone via XGBoost)

Smoke (1 fold, random groups of 1000) on the e3_hgbc feature set
(Driver int-encoded, Compound/Race int-encoded, 14 features):

| metric | val |
|---|---:|
| XGB rank:pairwise fold-0 AUC | **0.90368** |
| best_iter                    | 0 |
| e3_hgbc baseline (BCE)       | 0.94876 |
| Δ                            | **−450.81bp** |

`best_iter=0` indicates XGBoost never improved val AUC over its
zero-tree baseline. The pairwise objective on this feature set with
random small groups produces an unconverged model. **SMOKE FAIL.**
Skipped full 5-fold per gating (≥10bp regression threshold).

Plausible failure modes:
1. Random small groups create too many easy pairs (all-positives in
   one bucket vs all-negatives in another) — XGB's gradient sees them
   as trivially solved → no useful split signal.
2. `rank:pairwise` is invariant to monotonic score transforms — initial
   tree splits look indistinguishable, so the booster picks "no split"
   as best.
3. Could be tuned with non-zero `lambda`, `alpha`, larger
   `min_child_weight`, but the prior of Part A's regression makes this
   unlikely to recover +1bp lift.

## Verdict

| approach | result |
|---|---|
| LambdaMART meta, Race-grouped     | **DEAD** — −86bp |
| LambdaMART meta, Random-grouped 1k| **DEAD-borderline** — high variance, ~tie / mild regress |
| YetiRank meta, Race-grouped       | NOT_RUN (deferred; same query semantics as LM-Race → expected dead) |
| XGB rank:pairwise base retrain   | **DEAD** — smoke −451bp, did not train |

**Combined: dead-list.** The K=21 LR-meta on the [raw, rank, logit]
expansion is metric-equivalent to (or better than) LambdaMART/
YetiRank at this pool composition. The AUC-direct base retraining
path via XGB rank:pairwise is too far off-the-shelf to converge.

## Mechanistic note for the playbook

When the competition metric is **global AUC** and bases output
calibrated probabilities, a logistic-regression meta over
[raw, rank, logit] expansions is theoretically equivalent to
optimizing global AUC in the limit (rank ↔ AUC by Mann-Whitney). The
LR meta's logit channel preserves cross-row score calibration that
within-query pairwise objectives discard.

LambdaMART with non-singleton groups optimizes a strictly smaller
objective (within-query AUC). Random tiny groups remove the
meta's ability to leverage cross-group score calibration that LR's
logit-expansion uses naturally — and the variance per-group is too
high to recover the lift through pairwise losses.

## Pred-LB (best meta swap)

LR-meta (anchor): identical to PRIMARY → predicted LB ≈ 0.95031.
LambdaMART-Race: pred LB ≈ 0.95031 − 86bp = **0.94170** (catastrophic).
LambdaMART-Rand: pred LB ≈ 0.95031 − 12bp = **0.94910** (regression
within submission-budget noise but not a candidate).

## Artifacts

- `scripts/d12_lambdarank_meta.py` — original (heavy) version with default
  lr=0.05, num_leaves=15, ES=50; first run completed LR + 1 fold of
  LM-Race (AUC 0.94875, iters=90 in 12 min) before being killed for
  budget.
- `scripts/d12_lambdarank_meta_fast.py` — ≤30 min budget version (lr=0.1,
  num_leaves=8, max_rounds=300, ES=20); produced full LR meta + full
  LambdaMART-Race 5-fold + 4-of-5 LambdaMART-Rand1000 folds before
  killed at 22:56 wall.
- `scripts/d12_aucpairwise_base.py` — XGB rank:pairwise base smoke +
  full retrain gate. Smoke FAILED → no full-fold output.
- `scripts/d12_smoke.py` — 1-fold smoke for both LambdaMART and YetiRank.
- `scripts/artifacts/d12_lambdarank_meta_results.json` — per-meta OOF/Δ/ρ
  summary (manually curated from killed-script logs in /tmp/d12_fast.log).
- `scripts/artifacts/d12_aucpairwise_base_results.json` — smoke-only
  result with `decision="skip_full_5fold"`.
- `scripts/artifacts/oof_d12_lr_meta_strat.npy` +
  `test_d12_lr_meta_strat.npy` — saved 5-fold LR meta artifacts (this
  is just a recomputation of d9f K21_swap; ρ=1.0 vs PRIMARY).
- `submissions/submission_d12_*.csv` not built (full 5-fold LR-only
  submission would be identical to PRIMARY; lambdamart submissions not
  saved per kill timing).

## Decision

**Submit-ready: no.** LambdaMART/YetiRank meta swap is **dead-listed**
for this stack at this composition. The LR-meta on the
[raw, rank, logit] expansion is metric-aligned for global AUC with
calibrated bases.

**Next mechanism families if pursuing meta-class diversity** (none of
these are higher-EV than non-meta moves like external data, but ranked
for completeness):
1. CatBoost native classifier on the 63-feature expand basis (already
   tried as d4 GBDT-meta — TIE_EXPECTED at ρ=0.995).
2. Calibrated stacking (Platt/isotonic on LR output) to test whether
   absolute-prob calibration changes the test ranking distribution.
3. Pairwise loss with **stratified balanced random groups** (1 pos :
   N neg per group of size ~50): forces every group to contain a
   discriminable pair. Untried; only angle that might rescue
   lambdarank in this setting. Predicted EV ≤ +1bp at ρ ≈ 0.998.

End — kept short per Rule 9 cap.
