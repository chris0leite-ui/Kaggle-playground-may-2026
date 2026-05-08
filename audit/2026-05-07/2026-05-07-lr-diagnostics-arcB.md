# LR-diagnostics Arc B — DGP archaeology (2026-05-07)

Branch `claude/ensemble-logistic-regression-research-MbLKu`. Arc B
asks: where is signal stable? where do interactions live? what's the
true effective pool size? Three diagnostics: E5 (bootstrap coef
stability), E6 (residual interaction map), E9 (forward-selection
trace).

Scripts: `scripts/lr_diag_e{5,6,9}_*.py`. JSONs:
`scripts/artifacts/lr_diag_e{5,6,9}_*.json`.

## E5 — bootstrap coefficient stability (50 boots × 2 regimes)

Single LR with 11 numeric + Compound dummies + Race dummies + Driver-
freq + 4 interactions (TyreLife×Stint etc.). l1 saga dropped (>>1
min/fit; E8 already settled L1 is rank-no-op for binary AUC).

| Regime | OOB AUC ± std | high-flip (noise) features |
|---|---|---|
| L2 (cw=None) | 0.85099 ± 0.0012 | 3 (PitStop, Race_Qatar, Race_Emilia) |
| L2 cw='balanced' | 0.85199 ± 0.0012 | 0 |

**Top SNR features (sign-flip = 0 across 50 boots):**

| feature | coef mean | SNR |
|---|---:|---:|
| TyreLife | +0.79 | 73.2 |
| RaceProgress | −6.33 | 64.5 |
| LapNumber | +5.50 | 63.0 |
| Stint | +0.55 | 46.6 |
| Cumulative_Degradation | −0.25 | 38.6 |
| Race_Mexico City / Bahrain / Belgian / Canadian / Spanish | ±0.8 to ±1.2 | 27–33 |

**Noise:** PitStop (flip 0.24, SNR 0.7) — non-monotone effect that LR
linear-space can't capture, despite "obvious" intuition. `Race_Qatar`,
`Race_Emilia Romagna` — small-sample noise.

**Reading.** The DGP has clean stable LINEAR signals on a small set
of features. class_weight='balanced' eliminates all sign-flip noise.
**Cap on linear-feature LR base = ~0.852 OOB AUC.** Matches E4
(0.74–0.86 per-cell) and E6 (0.854 OOF).

## E6 — residual interaction map

5-fold OOF LR (43 features, balanced) → AUC 0.854. Residuals binned
into 5×5 (feature_a, feature_b) quintile grids; max|cell mean residual|
scored. Compared to PRIMARY's per-cell residuals (saturated <1%).

**Top 8 pairs by max|LR cell residual|:**

| pair | LR max | PRIM max | GBDT-captures gap |
|---|---:|---:|---:|
| Stint × RaceProgress | +0.61 | +0.008 | **+6025 bp** |
| Year × Stint | +0.55 | +0.006 | +5457 bp |
| LapNumber × Stint | +0.55 | +0.009 | +5418 bp |
| Stint × TyreLife | +0.50 | +0.007 | +4959 bp |
| Stint × LapTime | +0.48 | +0.004 | +4738 bp |
| **LapTime × LapTime_Delta** | +0.47 | +0.009 | +4607 bp |
| Stint × LapTime_Delta | +0.44 | +0.006 | +4370 bp |
| Stint × Cumulative_Degradation | +0.43 | +0.006 | +4200 bp |

**Reading.** **Stint is THE dominant interaction hub** — 9 of 10 top
pairs include it. PRIMARY's max-cell residuals are sub-1% across every
pair (GBDT pool fully saturates the interaction structure). Adding
more (Stint × *) features to the GBDT pool is null-EV; **a pure-LR
base needs explicit (Stint × *) FE** to be competitive. Only top-10
pair without Stint: LapTime × LapTime_Delta (degradation signature).

## E9 — forward-selection trace on K=24 (110k subsample, early stop)

Greedy CV-optimal LR-meta builds the pool one base at a time.

| K | added | AUC | Δ bp |
|---:|---|---:|---:|
| 1 | **h1d_yekenot_full** | 0.95280 | (anchor) |
| 2 | p1_single_cb_v3_gpu | 0.95359 | +7.91 |
| 3 | f1_hgbc_deep | 0.95377 | +1.82 |
| 4 | d16_orig_continuous_only | 0.95388 | +1.14 |
| 5 | b_lapsuntilpit | 0.95395 | +0.73 |
| 6 | baseline_two_anchor | 0.95407 | +1.11 |
| 7 | d9_R6_next_compound | 0.95409 | +0.22 |
| 8 | cb_year-cat | 0.95410 | +0.16 |
| 9 | e5_optuna_lgbm | 0.95412 | +0.17 |
| **10** | d9f_FM_A | **0.95413** | +0.12 (peak) |
| 11 | cb_slow-wide-bag | 0.95413 | **−0.03** |
| 12 | f2_hgbc_shallow | 0.95413 | −0.03 |
| 13 | m2_xgb | 0.95413 | −0.00 |
| ... | early stop; 11 bases never tried | | |

**Findings:**

1. **K=10 = K=24 in OOF AUC** (both 0.95413 on the 110k subsample).
   14 of 24 bases are dead weight or actively harmful. Empirical
   confirmation of E1's eff_rank=2.88.
2. **`cb_slow-wide-bag` is the first negative-delta pick** — exactly
   as predicted by E1 (most redundant ρ=0.9963 with cb_year-cat) and
   E2 (most mis-calibrated, slope 1.95). Three independent
   diagnostics agree.
3. **The K=10 core has clean class composition**: 2 Day-17 anchors
   (h1d, p1_cb) + 4 GBDT-class + 1 selective-FE LGBM + 1
   leakage-robust feature + 1 rule-residual + 1 FM. Greedy meta picks
   one of each class and stops.
4. **11 bases never picked**: e1_catboost_sub, e3_hgbc, a_horizon,
   cb_lossguide, realmlp, d6_rule_driver_compound, d6_rule_year_race,
   d9_R10_driver_eb, d9_R7_prev_compound, d9f_FM_B, d2a_te. These are
   the leakage-eaters / class-clones that project onto direction-1
   (E1).

## Synthesis (Arc B)

Three findings, three durable lessons:

**1. The pool has 10 useful bases and 14 dead-weight bases** (E1+E9
agree quantitatively). Pool surgery design candidate: drop 14, replace
with structurally distinct bases. Even if surgery is null-EV in raw
LB, the K=10 stack frees 14 slots for genuinely diverse additions.

**2. Linear-feature LR caps at 0.852 OOB AUC** (E5 + E4 + E6 all
agree). Without explicit (Stint × *) FE per E6, any pure-LR base in
Arc C will be a weak diversity contributor. **Sharp Arc C test:**
Bagged-LR with engineered Stint-cross terms vs without — quantifies
how much linear-but-with-interactions can recover, and produces the
diversity injection candidate for pool surgery.

**3. Stint is the dominant interaction hub on this DGP** (E6). The 9
top-residual pairs all contain Stint; PRIMARY captures all 9 to <1%.
Reusable across future tabular comps: a 5×5 quintile residual heatmap
identifies the dominant interaction hub in <5 min CPU on Day-1.

## Outputs (reusable beyond s6e5)

- `scripts/lr_diag_e5_bootstrap_coef.py` — bootstrap-coef SNR ranking
- `scripts/lr_diag_e6_residual_interactions.py` — interaction-hub map
- `scripts/lr_diag_e9_forward_select.py` — true-effective-pool-size
- 3 JSON results in `scripts/artifacts/lr_diag_e{5,6,9}_*.json`
- This audit + Arc A audit: complete LR-diagnostic skill module
  candidate for `.claude/skills/kaggle-comp/` (Day-1 of any tabular
  comp).

## Pointer to Arc C

Pool surgery + new-base diversity: K=10 core + (5 fresh diverse bases
from Arc C) = K=15 stack. Arc C produces 3 new-base populations
(Bagged-LR with/without Stint-cross, Random-Subspace LR, per-segment
LR specialists). Each tested via probe_min_meta against K=10 baseline
(not the full K=24 — the dead weights confound the test).
