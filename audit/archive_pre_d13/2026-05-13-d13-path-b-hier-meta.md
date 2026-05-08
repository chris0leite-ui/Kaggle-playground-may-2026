# Day-13 Path B — empirical-Bayes hierarchical LR meta

> Synthesis of d10b/c/d (FM-class lift amplifies under leak-blocking,
> but full leak-correction over-smooths rare-class extremes).
> Per HANDOVER Path B: per-segment partial-pooling weights, EV +1-3bp.
> Result: marginal — best variant clears OOF gate (+0.88bp) but fails
> G3 (flip ratio 0.22 < 0.5).

## Architecture

For row i in segment s:
  ŷ_i = sigmoid(w_s · F_aug_i)
  w_s = (n_s · w_s_local + τ · w_global) / (n_s + τ)

τ controls shrinkage (τ → ∞ recovers PRIMARY, τ → 0 recovers
per-segment LR). Each segment fits its own LR on the K=21 expanded
feature set (raw + rank + logit, 63 features).

K=21 = PRIMARY pool (POOL_KEEP 16 + TOP_3_D9 + FM_A + FM_B).

`scripts/d13_path_b_hier_meta.py` (full sweep, killed mid-run on
Compound×Stint), `scripts/d13b_path_b_stint_only.py` (focused
finalizer, ~5 min wall).

## Sweep results — Stint dominates

| Segments | n | τ | OOF | Δ vs global | ρ vs PRIMARY |
|---|---:|---:|---:|---:|---:|
| Compound | 5 | 100000 | 0.95076 | **+0.30bp** | **0.99901** |
| Compound | 5 | 20000 | 0.95075 | +0.23bp | 0.99744 |
| Stint | 5/6 | 20000 | **0.95082** | **+0.88bp** | 0.99609 |
| Stint | 5/6 | 100000 | **0.95082** | **+0.86bp** | 0.99837 |
| Compound × Stint | 21/30 | — | killed | — | — |
| Year × Compound | — | — | not run | — | — |

The Stint segmentation captures the strongest pit-behavior dynamic:
LR weights for stint-1 (long fresh tyre) differ structurally from
stint-3+ (degraded, near-pit-window). FM and rule_residual bases
naturally encode stint-stage but are still treated globally by the
PRIMARY meta. Per-stint LR partial-pooling lets each stint pick its
own FM-vs-GBDT mixture.

## G3 rare-class flip diagnosis

| Variant | flips: PRIMARY-pos → new-neg | flips: new-pos | ratio | G3 ≥ 0.5 |
|---|---:|---:|---:|:---:|
| Stint τ=20000 | 328 | 72 | 0.220 | **FAIL** |
| Stint τ=100000 | 209 | 44 | 0.211 | **FAIL** |
| (d10d for ref) | 1751 | 2 | 0.001 | FAIL ×500 |
| (PRIMARY) | — | — | 1.000 | — |

The Path B Stint variants fail G3 but with a flip pattern **4-5×
better balanced** than d10d. The asymmetry — 4.6× more demotions
than promotions — means the per-stint meta systematically reduces
top-1% predictions in some segment, likely later stints where FM
weight rises and GBDT row-extremes lose ground.

## R7 sign-off threshold

| Variant | total rare-class movements | R7 threshold | Sign-off needed |
|---|---:|---|:---:|
| Stint τ=20000 | 400 (328+72) | < 200 → HEDGE-only | **YES** |
| Stint τ=100000 | 253 (209+44) | < 200 → HEDGE-only | **YES** |

Both variants exceed the R7 < 200 flip threshold for "HEDGE-only"
behavior. Per Rule 16's 5-question check:
  (1) Mechanism in families? **No** — empirical-Bayes hierarchical
      meta is genuinely new, never tried.
  (2) In rank-lock pool? **No** — different routing structure, not
      meta-only nor base-rebuild.
  (3) Predict OOF? +0.5-1.0bp from analogous per-segment thresholding.
      Actual: +0.88bp. Within range.
  (4) Predict ρ? Closest analog is "different meta architecture",
      ρ ~ 0.995-0.999. Actual 0.996-0.998. Within range.
  (5) Closest gate-PASS/FAIL precedent at this ρ? d10d at ρ 0.987
      was −3bp LB; m5_meta_lgbm_shallow at ρ 0.995 was −4bp LB.
      **At ρ 0.996-0.998, expected LB is −1 to −2bp.**

5-question check ⇒ **downgrade EV midpoint**. Path B at +0.88bp OOF
is too borderline to clear the LB-transfer gates. The OOF lift is
real but doesn't justify a PRIMARY swap.

## Why this works at all (mechanism)

GBDT bases capture row-extreme signals globally — but their feature
importance differs by stint. Stint-1 pit decisions depend heavily
on TyreLife and Compound (both well-handled by GBDTs). Stint-3+
pit decisions depend more on RaceProgress and Position (the
Y-axis FM_B partition). The global LR meta picks a single weight
ratio across all stints; the per-stint shrunk meta picks a slightly
different ratio per stint. The OOF lift comes from the fact that
later-stint folds genuinely benefit from higher FM weight.

But the test set's row-extreme rare-pit predictions come from a
mix of GBDT confidence + FM stint-context. The per-stint meta
re-weighting demotes some of those rows, and unlike d10d (which
fully GKF-blinded the GBDTs), here the demotion is mild and
distributed across stints rather than concentrated.

## Decision

| Option | Pro | Con |
|---|---|---|
| Submit Stint τ=100000 | best OOF lift, ρ closest to TIE | G3 fail; ~−1bp expected LB |
| Submit Compound τ=100000 | only ρ ≥ 0.999 PASS variant | only +0.30bp OOF; calibration probe |
| Hold all variants | preserve slot budget | no LB calibration on this mechanism |

Recommended: **hold Stint τ=100000** as R5 final-window OOF-best
candidate. It clears the OOF gate but the public-LB transfer is
uncertain (similar ρ-band to d10d but better flip balance). Submit
**Compound τ=100000** as a calibration probe: it's the only ρ ≥
0.999 variant, costs 1 slot, expected LB +0.0 to +0.3bp.

## What's not exhausted

1. **Compound × Stint segmentation** — full d13 was killed at fold 2.
   Restart with stricter min_rows (1000 instead of 200) and
   max_iter=500 to avoid LR convergence stalls. The interaction
   could be the right partition.
2. **Year × Compound** — never run. 12 segments, ~15-25k rows each;
   smallest size. EV uncertain.
3. **PyMC full Bayesian model** — replace empirical-Bayes shrinkage
   with proper hierarchical Gaussian. ~30-60 min PyMC. Should give
   similar OOF but better-calibrated uncertainty for ρ predictions.

## Pointers

- `scripts/d13_path_b_hier_meta.py` — full sweep (killed).
- `scripts/d13b_path_b_stint_only.py` — Stint finalizer.
- `scripts/artifacts/d13b_path_b_stint_results.json` — full results.
- `scripts/artifacts/{oof,test}_d13_path_b_stint_tau{20000,100000}_strat.npy`.
- `submissions/submission_d13_path_b_stint_tau{20000,100000}.csv` — HELD.
