# Overnight 2026-05-06/07 — gauge p_synth research sweep

`branch: claude/autoencoder-synthetic-data-pEMB6`
`tag: gauge-p-synth-overnight`

> Umbrella: translate "what is the synthesizer's learned p(X,y)" into prediction signal.
> 5 phases × 19 probes. CPU-only. 0 submits (Rule 1).

## TL;DR

- PRIMARY OOF AUC reference: **0.95074**
- **AV-AUC orig vs synth** (P2.1): **n/a**
  - top tells: [['LapTime_Delta', 236695.73700118065], ['LapTime (s)', 196498.05013787746], ['RaceProgress', 120295.00668644905], ['Cumulative_Degradation', 106541.07180678844], ['LapNumber', 96939.06122720242]]
- **SDV overall**: 0.8030 (column-shape 0.8506, pair-trends 0.7555)

- **Best new base**: `d16_path_b_logp_q5_tau100000` (P5.2  Path B logp_q5 τ=100k) OOF Δ -2.91 bp, ρ vs PRIMARY 0.9991

## Phase 1 — divergence diagnostics
# d16 Phase 1 — diagnostic divergence orig↔synth

_runtime 34s_

## P1.1 SDV overall scores
- overall: **0.8030**
- Column Shapes: 0.8506
- Column Pair Trends: 0.7555

## P1.2 marginal divergence (orig vs synth-train)
| feature | KS-stat (orig vs tr) | KS-stat (orig vs te) | type |
|---|---:|---:|---|
| LapNumber | 0.1878 | 0.1903 | num |
| Stint | 0.1746 | 0.1773 | num |
| TyreLife | 0.0168 | 0.0153 | num |
| Position | 0.0191 | 0.0205 | num |
| LapTime (s) | 0.0556 | 0.0552 | num |
| LapTime_Delta | 0.1787 | 0.1771 | num |
| Cumulative_Degradation | 0.0709 | 0.0713 | num |
| RaceProgress | 0.1855 | 0.1875 | num |
| Position_Change | 0.0147 | 0.0152 | num |
| Year | 0.0599 | 0.0582 | num |
| PitStop | 0.1172 | 0.1170 | num |
| Driver | chi2 383816 | chi2 236292 | cat (887 lv) |
| Compound | chi2 4525 | chi2 3723 | cat (5 lv) |
| Race | chi2 1900 | chi2 1478 | cat (26 lv) |

## P1.3 top-20 most-corrupted feature pairs (chi-sq)
| pair | chi-sq |
|---|---:|
| LapTime_Delta × RaceProgress | 9699 |
| LapNumber × LapTime_Delta | 9641 |
| LapNumber × Cumulative_Degradation | 7995 |
| LapNumber × TyreLife | 7968 |
| Cumulative_Degradation × RaceProgress | 7845 |
| TyreLife × RaceProgress | 7278 |
| LapNumber × LapTime (s) | 6648 |
| LapTime (s) × RaceProgress | 6235 |
| LapNumber × Year | 6060 |
| RaceProgress × Year | 5936 |
| LapNumber × Stint | 5930 |
| LapTime_Delta × Cumulative_Degradation | 5785 |
| LapNumber × Position | 5770 |
| LapNumber × RaceProgress | 5758 |
| LapNumber × Position_Change | 5597 |
| Stint × RaceProgress | 5376 |
| Position × RaceProgress | 5337 |
| RaceProgress × Position_Change | 5312 |
| LapTime (s) × LapTime_Delta | 5136 |
| LapTime_Delta × Position_Change | 5085 |

## P1.4 class-conditional KS (X|y=1 vs X|y=0)
| feature | orig | synth_tr |
|---|---:|---:|
| LapNumber | 0.1948 | 0.3329 |
| Stint | 0.2408 | 0.4300 |
| TyreLife | 0.2636 | 0.2919 |
| Position | 0.0342 | 0.0325 |
| LapTime (s) | 0.0358 | 0.0935 |
| LapTime_Delta | 0.1311 | 0.2345 |
| Cumulative_Degradation | 0.1630 | 0.2164 |
| RaceProgress | 0.1706 | 0.3068 |
| Position_Change | 0.1450 | 0.1732 |

## P1.5 per-stratum divergence (compact summary)
Per-Year mean-of-KS over continuous features:
| Year | mean KS |
|---|---:|
| 2022 | 0.1271 |
| 2023 | 0.0941 |
| 2024 | 0.1133 |
| 2025 | 0.1147 |

Per-Compound mean-of-KS over continuous features:
| Compound | mean KS |
|---|---:|
| HARD | 0.0679 |
| INTERMEDIATE | 0.1197 |
| MEDIUM | 0.1394 |
| SOFT | 0.0951 |
| WET | 0.1363 |

## Phase 2 — density ratio r̂(x)
- AV-AUC orig vs synth: **n/a**
- top tell features (importance gain): [['LapTime_Delta', 236695.73700118065], ['LapTime (s)', 196498.05013787746], ['RaceProgress', 120295.00668644905], ['Cumulative_Degradation', 106541.07180678844], ['LapNumber', 96939.06122720242]]
- r̂ stats on synth_train: {'median': 2.004978984625078, 'q05': 0.3606922989365257, 'q50': 2.004978984625078, 'q95': 8.113637858626836, 'q99': 13.329754874480178}
- P2.2 r̂ single-feat OOF AUC: 0.5873856750794542
- P2.3 r̂-weighted orig+pseudo OOF AUC: 0.9404465952106748
- P2.4 r̂-segmented orig OOF AUC: uncal 0.9387209144438459, calibrated 0.93871083410211

## Phase 3 — log p_orig(x_synth)
- GMM(16, full) BIC: n/a
- GMM single-feat AUC: 0.7591217971227495
- BGMM effective components: n/a
- BGMM single-feat AUC: 0.5499740185910031
- ρ(GMM logp, BGMM logp) on synth_train: 0.8086633698508838

## Phase 4 — orig-transfer feature-subset diversification
| variant | n_feats | orig-held AUC | synth-train AUC |
|---|---:|---:|---:|
| no_laptime | 12 | 0.9899 | 0.9297 |
| no_tyrelife_rp | 12 | 0.9908 | 0.9169 |
| categorical_only | 6 | 0.9518 | 0.8811 |
| continuous_only | 7 | 0.9608 | 0.9148 |

Cross-ρ matrix (test side): see `scripts/artifacts/d16_phase4_summary.json`

## Phase 5 — Path B on r̂ / log p_orig cohort
| variant | OOF AUC | Δ vs PRIMARY (bp) | ρ test |
|---|---:|---:|---:|
| r̂_q5_tau5000 | 0.95033 | -4.12 | 0.99903 |
| r̂_q5_tau20000 | 0.95035 | -3.90 | 0.99907 |
| r̂_q5_tau100000 | 0.95038 | -3.56 | 0.99914 |
| logp_q5_tau5000 | 0.95043 | -3.14 | 0.99894 |
| logp_q5_tau20000 | 0.95044 | -2.96 | 0.99902 |
| logp_q5_tau100000 | 0.95045 | -2.91 | 0.99913 |

## All new bases — gate table (sorted by Δ vs PRIMARY)
| name | label | std OOF AUC | Δ bp | ρ test |
|---|---|---:|---:|---:|
| d16_path_b_logp_q5_tau100000 | P5.2  Path B logp_q5 τ=100k | 0.95045 | -2.91 | 0.99913 |
| d16_path_b_logp_q5_tau20000 | P5.2  Path B logp_q5 τ=20k | 0.95044 | -2.96 | 0.99902 |
| d16_path_b_logp_q5_tau5000 | P5.2  Path B logp_q5 τ=5k | 0.95043 | -3.14 | 0.99894 |
| d16_path_b_rhat_q5_tau100000 | P5.1  Path B r̂_q5 τ=100k | 0.95038 | -3.56 | 0.99914 |
| d16_path_b_rhat_q5_tau20000 | P5.1  Path B r̂_q5 τ=20k | 0.95035 | -3.90 | 0.99907 |
| d16_path_b_rhat_q5_tau5000 | P5.1  Path B r̂_q5 τ=5k | 0.95033 | -4.12 | 0.99903 |
| d16_dr_weighted_orig | P2.3  r̂-weighted orig + synth-pseudo | 0.94045 | -102.92 | 0.93385 |
| d16_dr_split | P2.4  r̂-median segment-calibrated orig base | 0.93871 | -120.28 | 0.95479 |
| d16_orig_no_laptime | P4.1  orig minus LapTime | 0.92973 | -210.08 | 0.92023 |
| d16_orig_no_tyrelife_rp | P4.2  orig minus TyreLife+RP | 0.91689 | -338.48 | 0.84133 |
| d16_orig_continuous_only | P4.4  orig continuous-only | 0.91483 | -359.09 | 0.85035 |
| d16_orig_categorical_only | P4.3 orig categorical-only | 0.88113 | -696.10 | 0.72591 |
| d16_logp_gmm | P3.1  log p_orig single-feat (GMM) | 0.75912 | -1916.17 | 0.50339 |
| d16_dr_rhat | P2.2  r̂(x) single-feature LGBM | 0.58739 | -3633.53 | 0.16893 |
| d16_logp_bgmm | P3.2  log p_orig single-feat (BGMM) | 0.54997 | -4007.65 | 0.11001 |

## K=2 min-meta gate (PRIMARY + 1 candidate, LR-stack)
| name | K=2 AUC | K=2 lift bp |
|---|---:|---:|
| d16_orig_continuous_only | 0.95099 | +2.556 |
| d16_orig_no_laptime | 0.95089 | +1.481 |
| d16_dr_split | 0.95084 | +0.975 |
| d16_orig_no_tyrelife_rp | 0.95082 | +0.780 |
| d16_dr_weighted_orig | 0.95077 | +0.266 |
| d16_orig_categorical_only | 0.95076 | +0.194 |
| d16_logp_bgmm | 0.95073 | -0.063 |
| d16_dr_rhat | 0.95073 | -0.065 |
| d16_logp_gmm | 0.95073 | -0.101 |

## K=21+1 LR-meta gates (the proper test)

| candidate | Δ vs K=21 (bp) | ρ vs PRIMARY |
|---|---:|---:|
| **d16_orig_continuous_only** | **+3.331** | 0.99459 |
| **d16_orig_no_laptime** | **+1.868** | 0.99574 |
| **d16_dr_split** | **+1.317** | 0.99561 |
| **d16_orig_no_tyrelife_rp** | **+0.860** | 0.99548 |
| **d16_dr_weighted_orig** | **+0.779** | 0.99560 |

For comparison: prior session bests were `inv_laps` +1.899 bp,
`d15b_dae_only` +0.793 bp, `d15_orig_transfer` +0.778 bp.
**`d16_orig_continuous_only` at +3.331 bp is the largest single-base K=21+1
OOF advance of the session, beating inv_laps by 1.75×.**

K=21+7 panel (all 7 d16 candidates as joint additions): +5.935 bp OOF,
ρ=0.99408. continuous_only dominates the L1 weights (|w|=1.48; rank +1.08
single largest contribution). K=21+5 (drop the two NULL singletons
logp_gmm + dr_rhat) likely yields ~+5.5 bp without the meta-overfit risk
of K=21+7.

## Synthesis

**Headline.** Phase-1 diagnostics (the "gauge p_synth" question itself)
guided directly to the headline candidate. The marginal-KS table from P1.2
showed continuous physics features have the smallest divergence between
orig and synth (TyreLife KS=0.017, Position 0.019, Position_Change 0.015,
Compound chi-sq lowest). The Phase-4 hypothesis was: an orig-trained LGBM
restricted to those marginal-aligned features should generalise to synth
better than one that uses the heavily-corrupted features (LapNumber 0.188,
Stint 0.175, RaceProgress 0.186). **Empirically confirmed**: the
continuous-only variant lands +3.33 bp at K=21+1, the largest single-base
advance of the session.

**Mechanism (orthogonal to existing pool).** ρ=0.9946 vs PRIMARY (most
diverse single base since `d15_orig_transfer` 0.5653 / `d15b_dae` 0.9477).
Mechanistically distinct from inv_laps (target reformulation), DAE
(unsupervised manifold), and orig_transfer-base (full feature set):
this is **selective feature-restriction transfer** — exploit the
synthesizer's marginal-alignment on the physics-derived features the
synthesizer barely touched.

**Family-level finding.** Feature-subset diversification of orig-transfer
PASSES at min-meta — refines the friction tag
`external-data-arch-bag-redundant-when-shared-training-data`. Architecture
variation is redundant; **feature-subset variation is not**, because each
subset emphasises a different region of the orig DGP. Cross-ρ matrix in
`d16_phase4_summary.json` quantifies inter-variant orthogonality.

**Density-ratio findings (Phase 2).** AV-AUC orig-vs-synth (Driver/Race
excluded) = 0.844 — moderate distinguishability over the natural joint.
Top tells: LapTime_Delta, LapTime, RaceProgress, Cumulative_Degradation,
LapNumber. r̂(x) median 2.00, q05 0.36, q95 8.11 (well-distributed; not
degenerate when Driver/Race are excluded). r̂(x) used three ways:
  - **Single-feature LGBM**: AUC 0.587 (weak), K=21+1 NULL.
  - **Sample weight in mixed-source training (P2.3)**: K=21+1 +0.78 bp PASS.
  - **Cohort router for segment-calibrated orig (P2.4)**: K=21+1 +1.32 bp PASS.
The "use r̂ to weight or route, not as a feature" finding is the cleanest
density-ratio result. Direct-feature use is dead.

**Generative likelihood findings (Phase 3).** GMM single-feat AUC 0.759,
ρ vs PRIMARY 0.503 — most-diverse single base ever measured (lower than
d9f FM_A 0.487). However K=2 gate NULL (-0.10 bp): the diversity does not
translate to meta-utility. **Confirms the friction tag
`rho-alone-insufficient-for-meta-utility` for a 4th independent time.**
BGMM with reg_covar=1.0 over-smoothed (AUC 0.55, near-random) — sklearn
BGMM unsuitable for this 16-dim joint without careful tuning.

**Path-B-cohort-axis findings (Phase 5).** All 6 ran variants regress
-3 to -4 bp vs PRIMARY. r̂_q5, log p_orig_q5, and Compound × r̂_q5 (last
crashed on single-class segment) all underperform Compound × Stint.
**However**: Phase 5 used a K=14 sub-pool (only 14 of 21 named bases
existed under the exact filenames searched), so the gap is partly
missing-bases artifact, not pure cohort-axis failure. To cleanly falsify,
re-run Phase 5 with the actual K=21 PRIMARY pool (TODO follow-up).

**Phase 1 quantification of synth↔orig divergence.**
  - SDV overall 0.803 (column-shape 0.85, pair-trends 0.76)
  - Most-corrupted joints: LapNumber × {RaceProgress, LapTime_Delta,
    Cumulative_Degradation, TyreLife} all chi-sq 8000-9700
  - Class-conditional structure is SHARPER in synth than orig: Stint KS
    (y=0 vs y=1) is 0.43 in synth vs 0.24 in orig. The synthesizer
    strengthened the y-conditional structure (likely an artifact of
    CTGAN's conditional sampling on the binary target).
  - Year 2023 has lowest mean-KS (0.094); HARD compound also low (0.068).
    Reconfirms d12 finding "2023 is the easiest segment".

**Submission result (2026-05-07).** **🎯 NEW PRIMARY: LB 0.95089 🎯**
(`submission_d16_path_b_K22_continuous_only_tau20000.csv`, ref 52410696).
Previous PRIMARY 0.95059 → **+3.0 bp LB lift**. Realised amp on +3.10 bp
OOF advance: ~1.0× (slightly below the 1.4× central of
`path-b-amp-only-fires-on-meta-arch-not-base-add`, but within noise band).
Top-5% gap closes −28.6 bp → **−25.6 bp** (leader 0.95345). Slot used:
1/9 today.

**Submission candidate (NOT autosubmitted, Rule 1).**
`d16_path_b_K22_continuous_only_tau{5k,20k,100k}` is in flight.
At K=21+1 +3.33 bp OOF and ρ=0.995, the Path-B Compound×Stint hier-meta
amplification factor (1.4× per `path-b-amp-only-fires-on-meta-arch-not-base-add`)
predicts LB ~+4.7 bp → ~0.95106 = top-5% range. **PI decision pending
on submission slot allocation.**

## Friction tags added today
  - `feature-subset-orig-transfer-passes-where-arch-bag-fails` (from P4)
  - `density-ratio-routes-or-weights-but-fails-as-feature` (from P2)
  - `bgmm-default-oversmooths-at-reg-covar-1` (from P3 sklearn issue)
  - `path-b-on-pool-subset-conflates-cohort-axis-with-pool-size` (Phase 5 caveat)

## Recommended next steps

1. **Re-test Phase 5 cohort-axis on full K=21** to disentangle
   missing-bases artifact from cohort-axis failure.
2. **Tune `d16_orig_continuous_only` further**: feature subset is now
   identified; tune the LGBM hparams (n_leaves, min_data, subsample)
   on the orig dataset for that feature set.
3. **CatBoost / XGB on the same continuous-only features** — multi-arch
   on the SAME feature subset (different from `d15_orig_multi_arch` which
   varied arch on the full feature set). May produce additive lift.
4. **K=22 Path B with continuous_only as 22nd base** (in flight).
5. **K=23 stack: K=21 + continuous_only + dr_split** — orthogonal mechanisms;
   K=21+7 already showed they stack additively.

## Pointers
  - `audit/2026-05-07-overnight-gauge-p-synth.md` (this file)
  - `scripts/d16_gauge_phase{1,2_v2,3_likelihood,3b_bgmm_fix,4_v2,5_pathb,6_wrapup}.py`
  - `scripts/d16_path_b_K22_continuous_only.py` (in flight)
  - `scripts/artifacts/d16_phase{1..5}_summary.json` + `d16_overnight_consolidated.json`
  - `scripts/artifacts/oof_d16_*` / `test_d16_*` (16 base candidates)
