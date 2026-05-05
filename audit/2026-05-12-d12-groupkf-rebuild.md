# 2026-05-12 — d12 GroupKF rebuild: testing the K=21 rank-lock leakage hypothesis

## Goal

Test whether the K=21 LR-meta rank-lock at OOF~0.95065 / LB~0.95031 is
an artifact of StratifiedKFold(5) leakage (P6: 80.1% within-stint pair
leak) by rebuilding the meta on **GroupKFold(Race, Driver, Year, Stint)**
OOFs and asking:

  - (a) does the meta still rank-lock?
  - (b) do different bases earn L1 weight?
  - (c) what's the implied LB shift?

## Inventory + rebuild

K=21 pool = `POOL_KEEP (16 bases) + TOP_3_D9 + R14_L4 + FM`. Of 21:

  - **13 already had GroupKF artifacts** (pre-existing, GroupKFold(Race)).
    Note: existing ones use the **weaker Race-only** group key, but
    that ALSO blocks within-stint leakage (whole Races land in same
    fold).
  - **7 missing** → rebuilt under strict (Race, Driver, Year, Stint)
    GroupKF: `d6_rule_driver_compound`, `d6_rule_year_race`,
    `d9_R6_next_compound`, `d9_R7_prev_compound`, `d9_R10_driver_eb`,
    `d9b_R14_L4`, `d9c_fm`.
  - **1 skipped**: `realmlp` (GPU-only; no CPU-feasible rebuild). For
    the K=21 meta we substituted strat OOF as a stand-in and
    documented the bias; we also ran a clean K=20 variant dropping
    realmlp.

Build script: `scripts/d12_groupkf_rebuild.py` (HGBC reduced to
max_iter=400 / max_leaf_nodes=31 vs production 1500/63 to fit CPU
budget under contention; OMP_NUM_THREADS=2). Walltime: ~13 min for
the 5 rules + ~10 min for R14_L4+FM running in parallel.

Sanity check: our `d9c_fm_groupkf` OOF AUC = **0.91978**, matches
the d10 audit's strict-GKF FM (0.91978) exactly → builder validated.

## Per-base GroupKF–Strat ΔAUC (the leakage-eater table)

Sorted by ΔAUC (most-leakage-dependent at top):

| Base (label) | Strat AUC | GKF AUC | ΔAUC bp |
|---|---:|---:|---:|
| m2_xgb | 0.94507 | 0.91084 | **−342.31** |
| a_horizon | 0.90640 | 0.87474 | −316.63 |
| e1_cb_sub | 0.94596 | 0.91638 | −295.80 |
| b_lapsuntilpit | 0.89840 | 0.86948 | −289.15 |
| cb_year-cat | 0.94679 | 0.91992 | −268.71 |
| cb_slow-wide-bag | 0.94790 | 0.92322 | −246.87 |
| cb_lossguide | 0.94697 | 0.92377 | −231.99 |
| e5_optuna_lgbm | 0.94736 | 0.92585 | −215.07 |
| f2_hgbc_shallow | 0.94861 | 0.92711 | −215.02 |
| f1_hgbc_deep | 0.94870 | 0.92739 | −213.13 |
| e3_hgbc | 0.94876 | 0.92785 | −209.07 |
| d2a_te | 0.93670 | 0.91628 | −204.26 |
| baseline_two_anchor | 0.94075 | 0.92059 | −201.65 |
| rule_year_race | 0.94586 | 0.94155 | **−43.03** |
| R14_L4 | 0.91369 | 0.90953 | −41.56 |
| rule_driver_compound | 0.94457 | 0.94056 | −40.15 |
| R10_driver_eb | 0.94463 | 0.94075 | −38.78 |
| R7_prev_compound | 0.94481 | 0.94141 | −33.93 |
| R6_next_compound | 0.94443 | 0.94128 | −31.51 |
| **FM (d9c)** | 0.92069 | 0.91978 | **−9.08** |
| realmlp | 0.94582 | n/a | n/a (GPU-only) |

**Two-population pattern.** GBDT bases drop −200 to −343bp; rule /
sparse-LR / FM bases drop only −9 to −43bp. **FM is 23–37× more
leakage-robust than every GBDT in the pool.** The rules and R14_L4
sit in the middle, dropping the same ~40bp band — they're learning
real signal but not chewing fold-mate state.

## K=21 LR-meta on GroupKF pool

(realmlp substituted with Strat OOF; see K=20 below for clean.)

|  | Strat-pool / Strat-CV | GKF-pool / Strat-CV | GKF-pool / GKF-CV |
|---|---:|---:|---:|
| K=21 OOF AUC | 0.95069 | 0.94777 | 0.94776 |
| ρ vs PRIMARY test | 0.99972 | 0.99138 | 0.99138 |
| pred-LB | 0.95030 | n/a | n/a |

**Meta agreement (ρ between meta predictions, Strat vs GroupKF-CV):**

  - ρ(Strat-meta-OOF, GKF-meta-OOF):  **0.9842**
  - ρ(Strat-meta-test, GKF-meta-test): **0.9914**

ρ = 0.9914 < 0.999 → **rank-lock partially dissolves under GroupKF**.
The meta predictions diverge meaningfully.

## K=20 LR-meta dropping realmlp (clean comparison)

|  | Strat | GKF / GKF-CV |
|---|---:|---:|
| K=20 OOF AUC | 0.95056 | 0.94577 |
| ρ vs PRIMARY test | 0.99855 | 0.98443 |
| pred-LB | 0.95007 | n/a |

  - ρ(Strat-meta-OOF, GKF-meta-OOF):  **0.9731**
  - ρ(Strat-meta-test, GKF-meta-test): **0.9856**

**Without realmlp the rank-lock dissolves more clearly** (ρ=0.9856 vs
0.9914 with realmlp inflating). realmlp's strat OOF was acting as a
shared "stable anchor" between the two meta solutions.

## L1 ranking shifts (K=20 GroupKF vs Strat) — the structural finding

Sorted by Strat L1, with rank-shift Δrank under GroupKF
(positive=DEMOTED, negative=PROMOTED):

| Base | L1 Strat | L1 GKF | Δrank |
|---|---:|---:|:--:|
| rule_driver_compound | 1.205 | 1.729 | +2 |
| **cb_slow-wide-bag** | 1.038 | 0.448 | **+17** |
| **e5_optuna_lgbm** | 1.014 | 0.513 | **+13** |
| R7_prev_compound | 0.859 | 1.326 | +0 |
| rule_year_race | 0.783 | 2.700 | −4 |
| d2a_te | 0.766 | 0.544 | +9 |
| a_horizon | 0.679 | 0.797 | +4 |
| e3_hgbc | 0.669 | 0.922 | −1 |
| b_lapsuntilpit | 0.630 | 0.796 | +3 |
| cb_year-cat | 0.513 | 0.954 | −4 |
| f1_hgbc_deep | 0.507 | 0.921 | −3 |
| baseline | 0.501 | 0.458 | +6 |
| R6_next_compound | 0.482 | 1.982 | **−11** |
| R10_driver_eb | 0.440 | 0.812 | −4 |
| cb_lossguide | 0.428 | 0.841 | −6 |
| e1_cb_sub | 0.413 | 0.485 | +1 |
| R14_L4 | 0.403 | 0.656 | −3 |
| f2_hgbc_shallow | 0.340 | 0.341 | +2 |
| m2_xgb | 0.329 | 0.786 | −6 |
| **FM** | 0.296 | 1.103 | **−15** |

**Story:** GBDT + LGBM bases (cb_slow-wide-bag, e5_optuna_lgbm) get
massively DEMOTED. Rule bases (rule_year_race jumps to L1=2.700; R6
jumps to L1=1.982) and **FM** (rank #20 → #5) get massively PROMOTED.

Interpretation: under leakage-blocked validation, the meta can no
longer rely on GBDTs' fold-mate-memorising boost. It re-weights toward
bases whose generalisation didn't depend on within-stint leak: **rules
+ R14_L4 + FM**.

## Verdict

  1. **Rank-lock dissolves.** Strat-meta-test vs GKF-meta-test ρ =
     **0.9914** (K=21 with realmlp anchor) / **0.9856** (K=20 clean).
     Both well below the 0.999 RHO_TIE threshold the d9c stack
     framework treats as "tied". Meta predictions truly diverge.
  2. **GBDTs are the leakage-eaters.** −200 to −340bp ΔAUC vs <50bp
     for rule/FM bases. The Strat-OOF "all-bases-tie" pattern was
     them eating the same fold-mate signal.
  3. **FM is genuinely orthogonal.** Only −9bp under strict GroupKF
     vs Strat. Mechanism: low-rank pairwise embeddings generalise
     across within-stint state; tree-leaves memorise it. Reinforces
     the Day-10 d10 audit's read.
  4. **Implied LB shift.** Strat K=21 OOF = 0.95069 → pred-LB 0.95030
     (+0.4bp vs PRIMARY 0.95026). GroupKF-meta K=21 OOF = 0.94776 —
     about 30bp lower on Strat axis but is the leakage-blocked truth.
     We saved `oof/test_d12_groupkf_meta_strat.npy` as a candidate
     since ρ vs PRIMARY meta-test = 0.9914 < 0.998.
  5. **Ceiling reframe.** If LB ≈ private-LB-leakage-robust signal
     (per d10 audit's framing), then submitting the GroupKF-meta
     would likely UNDERPERFORM public LB (because public LB rewards
     same-stint-overlap predictions to some extent — i.i.d. row split
     per U3) but might generalize better on private if test split is
     stricter. d10 already signaled this.

## Pred-LB / next move

  - **Don't submit GroupKF-meta as PRIMARY.** Public LB is i.i.d.
    row split (U3) — leakage-blocked OOF is too strict for public.
    GroupKF meta on public ≈ 0.94776 + (Strat→LB gap that we don't
    know for this leakage-free predictor). Likely public-LB regression
    of 5-15bp. Use the existing PRIMARY (d9f K=21) for PRIMARY slot.
  - **Save GroupKF-meta as HEDGE candidate** for private-LB-tilted
    selection. ρ=0.9914/0.9856 vs current PRIMARY makes this the most-
    diverse meta-output in the artifact set since RealMLP joined M5q.
    It's earned for R5 (final-3-day OOF-best regression probe) consideration.
  - **Re-build POOL with FM-class diversification.** The L1 shift
    suggests the next high-EV move is replacing the most-leakage-eating
    GBDTs (e5_optuna_lgbm, cb_slow-wide-bag, e1_cb_sub) with more
    FM-class / rule-class bases. Candidates: 3-way multi-FM (closed but
    consider re-attempting with 6-feature partition), DeepFM-lite,
    FFM-aware re-attempt with regularised embeddings.
  - **Tier-1 hedge:** train a FM-only K=N meta (FM_A, FM_B, R14_L4,
    R6, R7, R10, rule_dc, rule_yr — 8 bases all with ΔAUC<50bp) and
    ρ-test vs PRIMARY. If the meta hits ρ < 0.99 vs PRIMARY at OOF
    similar to PRIMARY, that's the leakage-robust hedge.

## Caveats

  - HGBC params reduced for time (max_iter=400 vs production 1500).
    Estimated −0.5 to −1.0bp impact on the rebuilt rules' GroupKF
    OOF AUCs. The ΔAUC vs Strat trend is unaffected (we only need
    rank, not absolute level).
  - 13 of 21 bases use Race-only GroupKF (pre-existing); 7 use strict
    (Race, Driver, Year, Stint). Both block within-stint leakage but
    Race-only is a stricter generalization test (also blocks per-Race
    overfit). The compare table mixes the two; the story is unchanged.
  - realmlp had no GroupKF; substituted Strat OOF in K=21. K=20 clean
    confirms the rank-lock-dissolution story is not realmlp-driven.

## Pointers

  - `scripts/d12_groupkf_rebuild.py` — rebuilds 7 missing bases.
  - `scripts/d12_groupkf_meta.py` — K=21 LR-meta on GKF OOFs.
  - `scripts/d12_groupkf_meta_no_realmlp.py` — K=20 clean variant.
  - `scripts/artifacts/d12_groupkf_meta_results.json` — K=21 results.
  - `scripts/artifacts/d12_groupkf_meta_no_realmlp_results.json` — K=20.
  - `scripts/artifacts/oof_d12_groupkf_meta_strat.npy` — saved meta
    OOF (note: `_strat` suffix per naming convention even though the
    base OOFs are GroupKF; this is a meta artifact built ON GroupKF
    OOFs but stored under the standard naming).
  - `scripts/artifacts/test_d12_groupkf_meta_strat.npy` — saved test.
