# Day-9 — 10 simple math/heuristic rule_residual probes

> Pragmatic-mathematician brainstorm: 10 closed-form / simple-ML rule
> bases mirroring the F1.2 multi-rule template (single rule lookup +
> HGBC residual on raw features). Builders
> `scripts/d9_math_heuristics.py` (per-base) and
> `scripts/d9_kn_stack.py` (K=N stack experiments). Strat-only (R1),
> 5-fold SEED=42.

## Approach catalogue

| ID | Name | Mechanism | Heuristic / model |
|---|---|---|---|
| R5 | weibull_compound | per-Compound Weibull(k, λ) hazard via method-of-moments on pit-event TyreLife | closed-form parametric |
| R6 | next_compound | smoothed lookup `p(pit \| curr × next × stint_q)` | Bayesian-smoothed (α=50) |
| R7 | prev_compound | smoothed lookup `p(pit \| prev × curr × laps_in_stint_q)` | Bayesian-smoothed (α=50) |
| R8 | position_progress | Position-decile × RaceProgress-decile lookup | Bayesian-smoothed (α=50) |
| R9 | laptime_delta_z | per-Compound z-score sigmoid σ((Δ−μ_c)/σ_c − 1) | closed-form parametric |
| R10 | driver_eb | per-Driver Beta-Binomial empirical-Bayes shrinkage (α=20) | closed-form Bayes |
| R11 | stint_overdue | (Compound, Stint) median-stint λ; row score = σ((tyre−λ)/scale) | closed-form parametric |
| R12 | cumdeg_knee | per-Compound 2-segment knee on Cumulative_Degradation | piecewise-linear + AUC search |
| R13 | race_lapbin | Race × within-race RaceProgress-decile lookup | Bayesian-smoothed (α=50) |
| R14 | hash_lr_3way | sparse one-hot LR over Driver × Compound × Stint (no residual) | simple ML (LR) |

R5–R13 fit an HGBC residual on the 14 raw features (matching the F1.2
rule_residual recipe in `scripts/d6_multi_rule.py`); R14 is a pure
simple-ML base (no residual).

## Coverages

- `next_compound` test coverage (within (Year, Race, Driver), unioned
  with train neighbours): **93.3%** (vs the 68% Day-8 P5 estimate
  which counted only test-internal successors).
- `prev_compound` test coverage: **93.2%**.

## Standalone-OOF + ρ + minimal-meta gate

> Minimal-meta gate vs PRIMARY (`d6_k18_multi_rule`, Strat OOF 0.95065).
> ρ is Spearman vs PRIMARY test predictions. Builder wall = 943s.

| ID | Std OOF | ρ M5q | ρ PRIMARY | min-meta | Δ vs PRIMARY (bp) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| R5 weibull_compound | 0.94600 | 0.93521 | 0.94277 | 0.95064 | −0.09 | FAIL |
| R6 next_compound | 0.94443 | 0.90086 | **0.90778** | 0.95064 | −0.12 | FAIL |
| R7 prev_compound | 0.94481 | 0.90588 | 0.91397 | 0.95064 | −0.10 | FAIL |
| R8 position_progress | 0.94554 | 0.92292 | 0.93060 | 0.95064 | −0.11 | FAIL |
| R9 laptime_delta_z | 0.94558 | 0.93425 | 0.94174 | 0.95064 | −0.09 | FAIL |
| R10 driver_eb | 0.94463 | 0.90353 | 0.91200 | 0.95064 | −0.10 | FAIL |
| R11 stint_overdue | 0.94557 | 0.91744 | 0.92486 | 0.95064 | −0.09 | FAIL |
| R12 cumdeg_knee | 0.94535 | 0.92705 | 0.93447 | 0.95064 | −0.09 | FAIL |
| R13 race_lapbin | 0.94539 | 0.91790 | 0.92541 | 0.95064 | −0.12 | FAIL |
| **R14 hash_lr_3way** | **0.79377** | 0.43577 | **0.44358** | 0.95063 | **−0.02** | FAIL (closest) |

## Headline finding — `rule_residual` mechanism is saturated in PRIMARY

Every rule_residual variant (R5–R13) lands in the **min-meta band
0.95064 ± 0bp**, FAILing the +0bp gate vs PRIMARY by 0.09–0.12bp.
This is independent of:
- Lookup-key choice (Compound × TyreLife, Compound × Stint, Year ×
  Race, Driver, Position × RaceProgress, Race × LapBin, Stint, …).
- Rule kind (Bayesian-smoothed lookup, closed-form Weibull,
  empirical-Bayes Beta-Binomial, piecewise-linear knee, sigmoid).
- Standalone strength (rule_AUC ranges 0.42–0.78; OOF after residual
  is monolithically ~0.94–0.945).

This is the **5th independent confirmation of the Day-8 P10 finding**
(`audit/2026-05-08-data-probe-results.md`):

> "Lift requires NEW SIGNALS or NEW MODEL CLASS, not better
> extraction." — HANDOVER §Updated priors

After T1.5 Deotte (Day-8), T1.3 Q12 (Day-8), T1.2 Poisson (Day-8),
F5 aux-meta (Day-6), and Move-B 2-base (Day-6), the d9 cohort closes
the loop on the `rule_residual` family: **adding *any* GBDT-residual
with raw features cannot lift PRIMARY's K=2 minimal-meta**.

## R14 hash_lr_3way is the structural outlier

Pure LR over hashed Driver × Compound × Stint interactions — no GBDT
residual, just the lookup as a probability:

- Std OOF only 0.794 (much weaker than rule_residual's 0.94+).
- **ρ vs PRIMARY = 0.444** — the most-diverse single base seen since
  M5q's RealMLP entry (which was 0.972).
- Min-meta Δ = **−0.02bp** — *just 0.02bp short of the gate*.

R14 is qualitatively different. It says: even when we change the
**model class** (LR ≠ GBDT) and the prediction *itself* is much
worse, the K=2 minimal-meta only loses 0.02bp. The information R14
adds is **mostly already in PRIMARY**, but the residual signal that
isn't is structurally orthogonal.

## K=N stack experiments (`scripts/d9_kn_stack.py`)

Tested four pool configurations, predicted-LB heuristic from
`scripts/d6_multi_rule.py:predicted_lb` (ρ-aware downweighting):

| Stack | K | Δ PRIMARY OOF | ρ vs PRIMARY | pred-LB Δ | Verdict |
|---|---:|---:|---:|---:|---|
| S1 K20 PRIMARY + 2 most-diverse d9 (R14, R6) | 20 | +0.09bp | 0.99991 | +0.09bp | TIE-bordering |
| S2 K28 PRIMARY + ALL 10 d9 | 28 | +0.07bp | 0.99973 | +0.07bp | dilution |
| S3 K18 swap (drop 2 redundant rules, add 2 d9) | 18 | +0.02bp | 0.99981 | +0.02bp | flat |
| **S4 K20 swap+add (drop 2 rules, add 4 d9)** | **20** | **+0.13bp** | **0.99971** | **+0.13bp** | **best, ρ-PASS** |

- **Existing rules ranked by ρ (most-redundant first)**: rule_compound_tyre
  0.938, rule_compound_stint 0.937, rule_year_race 0.933,
  rule_driver_compound 0.902. Driver_compound is *least* redundant
  (highest standalone diversity in the existing 4-rule cohort).
- **d9 bases ranked by ρ (most-diverse first)**: R14 0.444, R6 0.908,
  R10 0.912, R7 0.914, R11 0.925, R13 0.925, R8 0.931, R12 0.934,
  R9 0.942, R5 0.943.
- S4's L1 ranking puts R14, R6, R7 in the top-15 with weights 0.54,
  0.50, 0.46 — comparable to RealMLP (0.60) and a_horizon (0.55).
  These bases are **earning their slots** by the LR meta's metric.

## Triage decision

1. **Do NOT submit S4 as a single-shot**: predicted Δ LB is +0.13bp,
   below the +0.5bp slot-worthy threshold from F1.2's predicted_lb
   heuristic. Per CLAUDE.md Rule 1, every submit needs PI approval —
   and the EV here doesn't justify a slot.
2. **Do NOT submit S1 / S2 / S3 either**: all hover at ±0.02–0.09bp.
3. The **swap-and-add mechanism is real but small** — confirms the
   `rule_residual` pool absorbs only 4 effective rule-degrees-of-
   freedom; replacement reshuffles rather than expands the
   information capacity.
4. **R14 hash_lr_3way is the most interesting d9 base**: a different
   *model class* with extreme diversity. Its 0.02bp min-meta gap
   suggests that with a stronger sister base (e.g., L2-LR with
   higher-order interactions, or a TabM/EmbMLP from the GPU queue),
   the model-class axis could plausibly lift the gate.

## Rule-16 (5-Q) pre-flight retro

For each d9 approach, the 5 questions retrospectively:

1. **Is the underlying mechanism in `mechanism_families_explored`?**
   - R5–R13: YES, all are `rule_residual_l1_base` (Day-6 entry).
   - R14: NO — sparse-LR over categorical interactions is a new
     mechanism family (closest: `lr_meta_stacker_3view`, but that's
     a meta over predictions, not a single-base LR over raw cats).
2. **Mechanism-vulnerability classification?**
   - R5–R13: rank-lock-vulnerable (clones of an existing-pool family).
   - R14: not rank-lock-vulnerable (different model class).
3. **Predicted standalone OOF?**
   - R5–R13 forecast: 0.944–0.946 (matches existing rule_residual
     band). Actual: 0.944–0.946. ✓ ladder calibration tight.
   - R14 forecast: 0.93–0.95 (LR with categoricals). Actual: 0.794.
     **Forecast was high by ~10bp** — the LR class is genuinely
     weaker on raw counts than even the cheapest GBDT residual.
4. **Predicted ρ vs PRIMARY?**
   - R5–R13 forecast: 0.92–0.94 (rule_residual class precedent).
     Actual: 0.91–0.94. ✓ tight.
   - R14 forecast: 0.95–0.98 (single LR base usually high-ρ).
     Actual: 0.444. **Major underestimate of diversity** — the model
     class shift produces orders-of-magnitude more independent
     signal than predicted.
5. **Closest gate-PASS/FAIL precedent?**
   - R5–R13 closest: F1.2 rules (4× PASS at min-meta, but vs M5q,
     not vs PRIMARY). When the gate moved to PRIMARY (which already
     has F1.2's 4 rules), the precedent flipped to FAIL — exactly
     the menu-overcrediting trap from Rule 16's Day-8 origin.
   - R14 closest: m5_meta_lgbm_shallow (Day-4, also a different meta
     family) — TIE_EXPECTED at ρ=0.995 with OOF −1bp; **R14's actual
     ρ is much LOWER (0.444 vs 0.995), but min-meta is similar (-0.02
     vs -0.06). New territory.**

EV midpoint for the d9 cohort should have been downweighted **0.3×**
on Q2/Q5 for R5–R13 (rank-lock-vulnerable + Day-8 same-mode
precedent). The observed 9-of-9 FAILs match.

## Pointers

- `scripts/d9_math_heuristics.py` — 10-approach builder (Strat-only
  5-fold; 943s wall on local CPU).
- `scripts/d9_smoke.py` — 50k 1-fold smoke harness.
- `scripts/d9_kn_stack.py` — K=N stack experiments (4 stack configs).
- `scripts/artifacts/d9_math_heuristics_results.json` — per-approach
  metrics.
- `scripts/artifacts/d9_kn_stack_results.json` — stack experiments.
- `scripts/artifacts/oof_d9_R*.npy`, `test_d9_R*.npy` — saved OOF +
  test predictions for each base.
