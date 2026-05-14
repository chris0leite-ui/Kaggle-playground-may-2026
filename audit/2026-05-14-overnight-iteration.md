# 2026-05-14 — Overnight iteration: K=11 ceiling confirmed

**Branch:** `claude/bootstrap-ml-problem-solving-6gK6W`
**Session window:** 2026-05-13 evening → 2026-05-14 ~01:30 UTC
**Submissions used today:** 4 of 5-10 daily slots
**Outcome:** PRIMARY unchanged at LB 0.95386. Five distinct creative
mechanisms tested; all NULL or REGRESSION. Empirical evidence for noise
ceiling.

## Submission ladder this session

| Submission | Mechanism | OOF lift vs K=11 | ρ_test vs K=11 | LB | Delta vs PRIMARY |
|---|---|---:|---:|---:|---:|
| K=8 = K=4 + qAT/qAV/qAO + K=27 + Path-B | Rebuilt 3-of-6 slim-nearest-neighbour | n/a | 0.999901 | 0.95382 | -0.4 bp |
| Blend 70/10/20 K=11/K=10/K=27 (harness top-1) | Tight-ρ blend | +0.065 bp | 0.999955 | 0.95386 | tie |
| K=12 = K=11 + control_logloss (rich-feature LGBM) | Wide-ρ base add | +18.194 bp | 0.928 | **0.95232** | **-15.4 bp REGRESSION** |
| Blend 60/15/25 K=11/K=10/K=27 | Deeper-in-OK-zone push | +0.059 bp | 0.99992 | 0.95386 | tie |

## Key findings (all NULL but informative)

### 1. ρ_test transfer-trap: <0.999 = regression risk
K=12 submission was the cleanest demonstration. Rich-feature LightGBM
on 68 engineered features lifted cross-validation by +18.194 bp at the
K=11+1 LR-meta gate. Split-stability check (seed=42 vs seed=43)
confirmed the lift was not a fold artifact (0.118 bp drift across seeds).
But ρ_test versus K=11 = 0.928 — well below the empirical transfer
floor. The leaderboard score regressed by 15.4 basis points.

**Implication:** the cross-validation gate is **systematically misleading**
for low-rank-correlation additions at this saturation level. Only
ρ_test in the [0.999, 0.9999] band transfers cleanly. Above 0.9999 ties
the proxy at 5-decimal precision; below 0.999 risks the K=12-class
regression.

Files: `scripts/build_K12_K11_plus_control.py`,
`scripts/artifacts/K12_K11plus_control_pathb_tau100000.json`.

### 2. Observable lead-feature is DEAD (synth label noise)
12.4% of test rows have lap L+1 in test for the same (Driver, Race,
Year). The natural read on the data would be PitNextLap[L] =
PitStop[L+1]. Train sanity check shows only **80.95% agreement** between
PitStop_next and PitNextLap. Per-group sums correlate at Spearman 0.60
— well below identity. Even when both PitStop_next=1 AND
Stint_next>Stint, agreement with PitNextLap=1 is 20.7% (essentially the
global base rate).

**Implication:** the synthetic data generator decouples PitStop and
PitNextLap with a stochastic component that no observable lookup or
group constraint can recover. K=11's 0.95443 cross-validation likely
sits at or near the **Bayes-optimal ceiling** of the noise floor.

Files: `scripts/submit_K11_with_observable_leak.py`.

### 3. Tree recalibration on K=11 — regression risk
LightGBM-meta with K=11 prediction + 13 context features (lag, per-(Race,
Year) relative, raw extras) trained on 5-fold CV. Standalone OOF 0.95420
(-2.3 bp vs K=11). K=11+1 LR-meta gate: -0.314 bp regression. ρ_test
0.9975 — REGRESSION_RISK by tonight's updated threshold.

**Implication:** K=11's LR-meta combiner is **already extracting** what's
extractable from the feature space. Adding a tree non-linear layer on
top doesn't unlock new signal.

Files: `scripts/probe_K11_recalibration.py`,
`scripts/artifacts/K11_recal.json`.

### 4. Per-row adaptive blend driven by base disagreement — zero lift
For each row, computed standard deviation across K=11's 11 underlying
base predictions. Tested 72 (tau, k_slope, max_alpha) combinations of
the function alpha[i] = clip((std[i] - tau) * k, 0, max_alpha), with
the blend new_pred = (1 - alpha) * K=11 + alpha * K=10.

**Max OOF lift across all 72 settings: +0.019 bp.** No candidate lifted
even one tenth of a basis point. All candidates either tie K=11 at ρ ≥
0.9999 or regress at ρ < 0.999. Base-disagreement carries no
incremental information not already in K=11.

Files: `scripts/probe_adaptive_blend.py`,
`scripts/artifacts/adaptive_blend_sweep.json`.

### 5. τ-trio blend harness — same top-1 as without τ variants
Rebuilt K=11 with Path-B τ=20k and τ=5k variants. Re-ran blend harness
with K=11 τ=100k + τ=20k + τ=5k + K=8 + K=10 + K=27 (6 ingredients,
~11k weight candidates). Top-1 blend: identical to the previous run
without τ variants — 0.7 × K=11_τ=100k + 0.1 × K=10 + 0.2 × K=27 at
+0.065 bp OOF. The τ=20k and τ=5k variants never made the top weights
because ρ to τ=100k is 0.9992-0.9998 — same logit subspace.

**Implication:** Path-B τ axis is saturated for blending. Different
shrinkage strengths on the same 11 bases all live in essentially the
same predictive subspace.

Files: `scripts/probe_blend_harness.py` (updated INGREDIENTS list).

## Architectural conclusion

K=11 + LR-meta + Path-B at LB 0.95386 is at or very near the predictable-
signal ceiling for row-level prediction on this synthetic dataset. The
label-decoupling between PitStop and PitNextLap reveals a stochastic
component in the synth generator that no row-feature model can recover.
This is consistent with the audit's prior conclusion (2026-05-08
night-session-summary) that "source information not derivable from
row features" is the gap.

## Directions worth attempting next session

Ranked by expected value, all multi-hour:

1. **Cross-domain training boost.** Retrain ONE K=4 anchor base (e.g.,
   d16_orig — already uses the real F1 reference) on combined synth +
   real F1 data with TRUE labels from the real-domain. V4 precedent:
   +0.8 bp at the leaderboard. ~30-60 min compute.
2. **Multi-seed bagging of the FULL K=11 pipeline.** Rebuild every
   underlying base + LR-meta + Path-B under random_state in {42, 43,
   44}, average submission-level. The K=11 pipeline took ~3-4 hours
   for the seed=42 build tonight; full 3-seed average = 9-12 hours.
3. **Bayesian group-constraint with smoothing.** Use observed
   per-(Driver,Race,Year) PitStop sum as a soft prior on per-group
   PitNextLap sum, adjust K=11 predictions by partial Bayesian
   shrinkage (NOT hard replacement). Even at Spearman 0.60 correlation,
   a 0.3-weighted prior might shift inter-group rankings usefully.

Hedge candidates locked in for final-window R7 selection:
- K=8 (LB 0.95382) — different stack structure (3 slim-kNN + K=27,
  no K=10 leg) — Tier-1 hedge.
- K=11 τ=20k or τ=5k (untested at LB but predicted ~0.95380-0.95385)
  — Tier-2 hedge.

## Files added today

```
scripts/
  build_K8_qAT_qAV_qAO_K27_pathb.py         (early-evening rebuild)
  build_K10_slim_pathb.py                   (slim-kNN-only stack)
  build_K11_full_pathb.py                   (K=11 reproduction; --tau arg)
  build_K12_K11_plus_control.py             (K=11 + rich-feat LGBM stacker)
  probe_loss_diversity.py                   (5-variant loss probe; killed pivot)
  probe_K11_recalibration.py                (tree meta-recalibrator; null)
  probe_adaptive_blend.py                   (per-row blend sweep; null)
  probe_blend_harness.py                    (updated ingredient list + verdict bands)
  submit_K11_with_observable_leak.py        (observable-lead trick; DEAD)

scripts/artifacts/
  dgp_v3_qAT_K1_*.npy                       (slim nearest-neighbour bases)
  dgp_v3_qAV_K1_7feat_*.npy
  dgp_v3_qAO_knn_multi_*.npy
  dgp_v3_qAA_stint_imputed_*.npy
  dgp_v3_qAF_d16plus_*.npy
  dgp_v3_qAK_knn3_*.npy
  K8_qAT_qAV_qAO_K27_pathb_tau100000_*.npy  + .json + submission CSV
  K10_slim_pathb_tau100000_*.npy            + .json + submission CSV
  K11_full_pathb_tau100000_*.npy            + .json + submission CSV
  K11_full_pathb_tau20000_*.npy             + .json + submission CSV
  K11_full_pathb_tau5000_*.npy              + .json + submission CSV
  K12_K11plus_control_pathb_tau100000_*.npy + .json + submission CSV
  loss_div_control_logloss_*.npy            (the rich-feature LGBM control)
  loss_div_scale_pos_weight_*.npy           (variant 2 from killed iteration)
  K11_recal_*.npy                           + .json
  adaptive_blend_test.npy                   + sweep .json
```

## Friction surfaced today

See `audit/friction.md` 2026-05-14 entries:
- `tag: cross-val-gate-misleading` — K=12 cross-validation +18 bp → LB
  -15 bp regression.
- `tag: synth-label-decoupling` — PitStop and PitNextLap correlate
  Spearman 0.60 per group, not the expected ~1.0; structural
  observation about the synth generator.
- `tag: rule-27-band-recalibration` — empirical threshold updated:
  TIE_ZONE at ρ_test >= 0.9999, REGRESSION_RISK at ρ_test < 0.999,
  OK transfer in between.
- `tag: rho-099-transfer-trap` — wide-ρ additions destroy LB even
  when cross-validation is split-stable.
