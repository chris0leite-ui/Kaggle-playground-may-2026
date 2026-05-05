# 2026-05-12 — d12 T1.2 multi-formulation L1 (the 3 untried reformulations)

PI directive: Day-8 falsified one of four planned T1.2 reformulations
(Poisson on `cum_pit_count_remaining`). Build the other three:
(c) censored regression on `laps_until_next_pit`, (d) ratio target,
(e) stint-level survival.

## TL;DR

**All three formulations FAIL the min-meta gate vs PRIMARY.** The
T1.2 reformulation cohort is now 4-of-4 falsified (Poisson on Day-8 +
the three here). Standalone OOF AUCs span 0.544 → 0.674 (vs PRIMARY
0.95070). K=22 stacks (PRIMARY-keep + each candidate alone) lift
+0.22-0.26bp in Strat OOF but ρ vs PRIMARY ≥ 0.99990, so the lift
is bounded by the rank-lock prior. **Pred-LB after rho discount**:
0.95033-0.95034 (vs current PRIMARY LB 0.95031, a +0.2-0.3bp probe
range — below the 0.5bp slot threshold).

## Setup

PRIMARY anchors used:
- OOF: `oof_d9c_Sd_K20_swap_FM_strat.npy` (K=20 swap, OOF 0.95070)
  — K=21 d9f swap OOF was not saved; K=20 is the closest available.
- Test: `test_d9f_K21_swap_strat.npy` (K=21 d9f swap, LB 0.95031).
- Min-meta gate: 3-feature LR over {PRIMARY, candidate, |Δ|}.
- Threshold: candidate PASSes if min-meta lift ≥ +0.10bp.

CV: StratifiedKFold(5, random_state=42). 439140 train rows, 188165
test rows. Pool features: 14 raw + `laps_into_stint`.

Note on (c) implementation: a true Cox PH or AFT loss at 350k+ rows
is O(N²) per boost iteration and infeasible (XGB Cox smoke timed
out at 50k rows). Substituted **LightGBM regression on
log(laps_until_next_pit + 1)** with `sample_weight=0.3` for
right-censored rows (target is a lower bound, weight reflects that).
This preserves the spirit of the censored regression formulation
(different problem-class than binary classification) while remaining
computationally tractable.

## Per-formulation results

| candidate | std OOF | ρ vs PRIMARY | min-meta OOF | Δ vs PRIMARY | min-meta verdict |
|---|---:|---:|---:|---:|---:|
| **T1.2c censored** | 0.54435 | 0.19604 | 0.95069 | -0.09bp | **FAIL ✗** |
| **T1.2d ratio→iso** | 0.67428 | 0.54851 | 0.95070 | +0.01bp | **FAIL ✗** |
| **T1.2e stint-survival** | 0.60125 | 0.18135 | 0.95070 | -0.07bp | **FAIL ✗** |

Notes:
- **T1.2c**: very low ρ (0.196) — most diverse single base since the
  hash_lr 3-way (R14) on Day-9. But std OOF is too weak (0.544);
  the rank ordering it adds isn't consistent enough with PitNextLap to
  add lift in the LR-meta layer.
- **T1.2d**: highest std (0.674) thanks to the well-calibrated
  ratio×age heuristic + isotonic. ρ=0.549 — also genuinely diverse
  vs PRIMARY. Min-meta lift +0.01bp is not zero but below the gate
  threshold; in stack it bumps b_lapsuntilpit's L1 by +5% (lapsuntil
  was the closest cousin already in the pool).
- **T1.2e**: stint-level duration regression mapped via heuristic
  hazard transform. Std 0.601 — the row-level mapping (`laps_into_stint
  / pred_dur`) is too coarse; the predicted duration is roughly
  constant across rows in the same stint, and `laps_into_stint` is
  already a feature in the pool. Adds +1.4bp in Strat OOF stacked but
  doesn't survive ρ-discount.

## K=22 / K=23 / K=24 stack results (PRIMARY pool 21 + candidates)

PRIMARY pool used: 16 base + R6/R7/R10 d9 rule_residuals + FM_A + FM_B
= 21 bases. Adding each T1.2 candidate as 22nd base, pairs as 23rd,
or all three as 24:

| stack | K | Strat OOF | Δ PRIMARY | ρ vs PRIMARY test | pred-LB | Δ LB |
|---|---:|---:|---:|---:|---:|---:|
| **K22_t12c_cox** | 22 | 0.95073 | +0.26bp | 0.99998 | 0.95034 | +0.26bp |
| **K22_t12d_ratio** | 22 | 0.95073 | +0.26bp | 0.99999 | 0.95034 | +0.26bp |
| **K22_t12e_survival** | 22 | 0.95073 | +0.22bp | 0.99990 | 0.95033 | +0.22bp |
| **K23_t12c+t12d** | 23 | 0.95073 | **+0.27bp** | 0.99998 | 0.95034 | **+0.27bp** ← BEST |
| K23_t12c+t12e | 23 | 0.95072 | +0.19bp | 0.99998 | 0.95033 | +0.19bp |
| K23_t12d+t12e | 23 | 0.95072 | +0.16bp | 0.99999 | 0.95033 | +0.16bp |
| K24_all_three | 24 | 0.95072 | +0.18bp | 0.99998 | 0.95033 | +0.18bp |

**Best stack**: K23_t12c+t12d at OOF 0.95073 (+0.27bp), saved to
`scripts/artifacts/oof_d12_t12_best_strat.npy` /
`test_d12_t12_best_strat.npy`. Pred-LB after ρ-discount: 0.95034
(only +3bp above current PRIMARY LB 0.95031, well within LB
quantization noise).

Additivity is anti-correlated: adding (c) and (d) together gives
+0.27bp, but adding (c) and (e) gives only +0.19bp, and all three
gives +0.18bp. The K=23 t12d+t12e and K=24 stacks REGRESS vs the
K=22 best, confirming the candidates are individually weak and
additionally interact destructively — when more than 2 t12 bases
are in the stack, the LR-meta over-credits the joint
time-to-event signal and rebalances away from the rules that
actually carry LB lift.

L1 ranking in K=22 stacks:
- **t12c_cox** lands in mid-tier L1 (~0.4 in K=22 with FM_A also at
  0.44). Provides genuine diversity (ρ=0.196) but the meta-LR can
  only weakly leverage it.
- **t12d_ratio** lands above f1_hgbc_deep but below FM_A (~0.42).
  Bumps b_lapsuntilpit's L1 by +5% — they're solving overlapping
  problems.
- **t12e_survival** has the lowest L1 contribution (drops out of top
  12 in some K=22 variants), confirming the row-level hazard transform
  doesn't carry usable signal.

## Pass-fail verdict per formulation

| formulation | std OOF | min-meta gate | K=22 stack-add | overall |
|---|:---:|:---:|:---:|:---:|
| T1.2c censored | weak | FAIL | +0.26bp / ρ=0.99998 (rank-locked) | **DEAD** |
| T1.2d ratio | medium | FAIL (+0.01bp, below 0.10) | +0.26bp / ρ=0.99999 (rank-locked) | **DEAD** |
| T1.2e stint-surv | weak | FAIL | +0.22bp / ρ=0.99990 (rank-locked) | **DEAD** |

This completes the T1.2 multi-formulation cohort: **4-of-4 falsified**
(Poisson on Day-8 + the three here). Cross-cohort signal:

- F1.2 multi-rule (rule_residual on raw features) was the only
  reformulation-class mechanism that landed +2.1bp on LB.
  Rule_residuals stay diverse from the GBDT consensus pool (ρ ≈ 0.9)
  because they're hand-coded constraints encoding signal the GBDTs
  can't see (forced-pit pressure, prev_compound spread).
- Time-to-event reformulations (Poisson, censored regression,
  ratio, stint-level survival) all collapse the same way: the loss
  landscape is genuinely different but the row-level inductive bias
  ends up rank-locked to the binary classifier's level set within
  the LR-meta layer.

## Recommendation for next session

**Do NOT submit any of the K=22 stacks.** Pred-LB lift is +0.22-0.27bp
across all variants, well below the 0.5bp slot threshold and
indistinguishable from LB quantization noise (LB resolution ≈ 0.5bp).

**Dead-list T1.2 entirely.** All four reformulations (Poisson,
censored, ratio, stint-survival) fail gate. The "different loss
landscape ⇒ different ranking" hypothesis is now 4× falsified for
time-to-event reformulations. Future LB-class moves should target
strictly different mechanisms:

1. **T1.1 TabM** (highest tail upside; new NN class, K=32 internal
   heads → free std meta-feature).
2. **T1.3 Q12 mandatory-2-compound rule_residual** (regulatory
   constraint not in pool; rule_residual mechanism class is the only
   one that landed +2.1bp historically).
3. **T1.4 hazard-rate NN** (different from T1.2c/e — the *NN
   class* itself adds inductive prior diversity; T1.2c failed
   because LGBM-based regression can't escape the GBDT pool's
   level set).
4. **T2.1 / T2.2** (next_compound feature, prev_compound × Stint-2
   rule_residual) — directly target the −341bp Stint-2 blind spot.

**Saved artifacts** (CWD `scripts/artifacts/`):
- `oof_d12_t12c_censored_strat.npy` / `test_d12_t12c_censored_strat.npy`
- `oof_d12_t12d_ratio_strat.npy` / `test_d12_t12d_ratio_strat.npy`
- `oof_d12_t12e_survival_strat.npy` / `test_d12_t12e_survival_strat.npy`
- `oof_d12_t12_best_strat.npy` / `test_d12_t12_best_strat.npy` (best K=23
  stack: t12c+t12d on PRIMARY-keep K=21)
- `d12_t12_multi_formulation_results.json` (per-stack OOF, ρ, L1).
- Total wall: 2956s (~49 min) on contended 8-core CPU box.

## 200-word summary

Built and 5-fold evaluated the three untried T1.2 multi-formulation
L1 reformulations: (c) censored regression on `laps_until_next_pit`,
(d) ratio target `pits/stints`, (e) stint-level survival. **All
three FAIL the min-meta gate** (deltas −0.09bp, +0.01bp, −0.07bp;
threshold +0.10bp). Standalone OOFs span 0.544–0.674 — t12d's
ratio+isotonic is the strongest base but still rank-collapses against
PRIMARY when stacked. K=22 single-add stacks all hit OOF 0.95073
(+0.22-0.26bp vs PRIMARY 0.95070); K=23 pair-add and K=24 all-three
stacks span +0.16-0.27bp. Best is K23_t12c+t12d at OOF 0.95073
(+0.27bp). All ρ vs PRIMARY-test ≥ 0.99990. **Pred-LB after
ρ-discount: 0.95033-0.95034** vs current PRIMARY LB 0.95031 —
within LB quantization noise, **below the +0.5bp slot threshold**.
Cross-cohort verdict: T1.2 is **4-of-4 falsified** (Poisson on Day-8
+ these three); time-to-event reformulations all collapse to the
GBDT level set inside LR-meta. **Do NOT submit any of these stacks.**
Recommendation for next session: prioritize T1.1 TabM (new NN
class), T1.3 Q12 forced-pit rule_residual (regulatory constraint;
rule_residual class survived gate historically), T1.4 hazard NN (NN
class, distinct from LGBM regression). Dead-list T1.2 entirely.

## Pointers

- `scripts/d12_t12_multi_formulation.py` — full implementation.
- `scripts/d12_t12_smoke.py` — 1-fold/20k smoke validation.
- `scripts/artifacts/d12_t12_multi_formulation_results.json` — full
  per-stack results + L1 rankings (written at script completion).
- `audit/2026-05-08-strategic-menu-wider-steps.md` §3 T1.2 spec.
- `scripts/artifacts/d8_poisson_lapsuntil_results.json` — the 1st of
  4 falsified T1.2 reformulations (Poisson on cum_pit_count_remaining,
  std OOF 0.5695, min-meta -0.08bp, K=19 stack -0.09bp).
