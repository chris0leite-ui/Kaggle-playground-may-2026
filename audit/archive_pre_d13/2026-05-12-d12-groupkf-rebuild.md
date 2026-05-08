# 2026-05-12 — d12 GroupKF rebuild: K=21 rank-lock leakage probe

## Goal

Test whether the K=21 LR-meta rank-lock at OOF~0.95065 / LB~0.95031 is
an artifact of StratifiedKFold(5) leakage (P6: 80.1% within-stint pair
leak). Rebuild the meta on GroupKFold(Race, Driver, Year, Stint) OOFs.

## Inventory + rebuild

K=21 pool = `POOL_KEEP (16) + TOP_3_D9 + R14_L4 + FM`. 13 already had
GroupKF artifacts (Race-only key); **7 missing rebuilt** under strict
(Race, Driver, Year, Stint): `d6_rule_driver_compound`,
`d6_rule_year_race`, `d9_R{6,7,10}_*`, `d9b_R14_L4`, `d9c_fm`.
**realmlp skipped** (GPU-only). HGBC reduced to max_iter=400 /
max_leaf_nodes=31; OMP_NUM_THREADS=2. Wall ~13 min for 5 rules + 10 min
R14_L4+FM in parallel. Script: `scripts/d12_groupkf_rebuild.py`. Sanity
check: our `d9c_fm_groupkf` AUC = 0.91978 = d10 audit's strict-GKF FM
AUC exactly → builder validated.

## Per-base GroupKF–Strat ΔAUC (the leakage-eater table)

| Base | Strat | GKF | ΔAUC bp |
|---|---:|---:|---:|
| m2_xgb | 0.94507 | 0.91084 | **−342** |
| a_horizon | 0.90640 | 0.87474 | −317 |
| e1_cb_sub | 0.94596 | 0.91638 | −296 |
| b_lapsuntilpit | 0.89840 | 0.86948 | −289 |
| cb_year-cat | 0.94679 | 0.91992 | −269 |
| cb_slow-wide-bag | 0.94790 | 0.92322 | −247 |
| cb_lossguide | 0.94697 | 0.92377 | −232 |
| e5_optuna_lgbm | 0.94736 | 0.92585 | −215 |
| f2_hgbc_shallow | 0.94861 | 0.92711 | −215 |
| f1_hgbc_deep | 0.94870 | 0.92739 | −213 |
| e3_hgbc | 0.94876 | 0.92785 | −209 |
| d2a_te | 0.93670 | 0.91628 | −204 |
| baseline | 0.94075 | 0.92059 | −202 |
| rule_year_race | 0.94586 | 0.94155 | **−43** |
| R14_L4 | 0.91369 | 0.90953 | −42 |
| rule_driver_compound | 0.94457 | 0.94056 | −40 |
| R10_driver_eb | 0.94463 | 0.94075 | −39 |
| R7_prev_compound | 0.94481 | 0.94141 | −34 |
| R6_next_compound | 0.94443 | 0.94128 | −32 |
| **FM (d9c)** | 0.92069 | 0.91978 | **−9** |
| realmlp | 0.94582 | n/a | n/a |

**Two-population pattern.** GBDT bases drop −200 to −343bp; rule /
sparse-LR / FM bases drop only −9 to −43bp. **FM is 23–37× more
leakage-robust than every GBDT.** Rules + R14_L4 sit middle (~40bp).

## K=21 LR-meta on GroupKF pool

(realmlp substituted with Strat OOF; clean K=20 below.)

|  | Strat / Strat-CV | GKF / GKF-CV |
|---|---:|---:|
| K=21 OOF AUC | 0.95069 | 0.94776 |
| ρ vs PRIMARY test | 0.99972 | 0.99138 |
| pred-LB Strat | 0.95030 | n/a |

**Meta agreement** (ρ between Strat-meta vs GKF-meta predictions):

  - ρ(meta-OOF):  **0.9842**
  - ρ(meta-test): **0.9914**

ρ = 0.9914 < 0.999 → **rank-lock partially dissolves**.

## K=20 clean (drop realmlp)

K=20 OOF AUC: Strat 0.95056, GKF/GKF-CV **0.94577**.
ρ(meta-test Strat vs GKF-CV): **0.9856** (cleaner divergence; realmlp
strat anchor was inflating ρ). ρ(GKF-meta-test, PRIMARY-test): 0.9844.

## L1 ranking shifts (K=20 GKF vs Strat)

Strat→GKF rank changes (sorted by Strat L1):

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
| ... (12 more) | ... | ... | ... |
| R6_next_compound | 0.482 | 1.982 | **−11** |
| **FM** | 0.296 | 1.103 | **−15** |

**Story:** GBDT/LGBM bases (cb_slow-wide-bag, e5_optuna_lgbm)
massively DEMOTED. Rules + FM (rule_year_race L1 0.78→2.70; FM rank
#20→#5) massively PROMOTED. Under leakage-blocked validation the
meta cannot rely on GBDTs' fold-mate-memorising boost; it re-weights
to bases whose generalisation didn't depend on within-stint leak.

## Verdict

  1. **Rank-lock dissolves.** ρ(meta-test Strat vs GKF) = 0.9914
     (K=21) / 0.9856 (K=20 clean). Both well below the 0.999
     RHO_TIE threshold treating bases as "tied".
  2. **GBDTs eat the leakage.** −200 to −340bp ΔAUC vs <50bp for
     rule/FM bases. The Strat-OOF "all-bases-tie" pattern was them
     eating shared fold-mate signal.
  3. **FM is genuinely orthogonal.** Only −9bp under strict GroupKF.
     Mechanism: low-rank embeddings generalise across within-stint
     state; tree-leaves memorise it. Reinforces d10 read.
  4. **Implied LB shift.** Strat K=21 pred-LB 0.95030 (+0.4bp).
     GroupKF-meta K=21 OOF 0.94776 — ~30bp lower on Strat axis but
     the leakage-blocked truth. Saved as
     `oof/test_d12_groupkf_meta_strat.npy` (ρ vs PRIMARY = 0.9914).

## Next move

  - **Do NOT submit GroupKF-meta as PRIMARY.** Public LB is i.i.d.
    row split (U3) — leakage-blocked OOF is too strict for public.
    Estimated public regression −5 to −15bp. Keep d9f K=21 PRIMARY.
  - **Save GroupKF-meta as HEDGE candidate.** ρ=0.9914/0.9856 makes
    it the most-diverse meta-output since RealMLP joined M5q.
    Earns R5 final-3-day OOF-best regression probe consideration.
  - **Pool refactor.** Replace most-leakage-eating GBDTs
    (e5_optuna_lgbm, cb_slow-wide-bag, e1_cb_sub) with more FM-class
    bases. Candidates: 6-feature multi-FM partition (3-way DEAD,
    2-way d9f is sweet spot — try 5/3 or 4/4 splits), DeepFM-lite,
    regularised FFM re-attempt.
  - **FM-only K=8 hedge meta.** Train a meta on the 8 ΔAUC<50bp
    bases (FM, rule_dc, rule_yr, R6, R7, R10, R14_L4, baseline).
    If ρ < 0.99 vs PRIMARY at OOF~PRIMARY, that's the leakage-robust
    hedge for private LB.

## Caveats

  - HGBC params reduced (max_iter 400 vs production 1500). Estimated
    −0.5 to −1bp on rebuilt rules' GKF OOF AUCs. Trend unaffected.
  - 13 of 21 bases use Race-only GroupKF (pre-existing); 7 use strict.
    Both block within-stint leakage. Story unchanged.
  - realmlp no GroupKF; K=20 clean confirms story is not
    realmlp-driven.

## Pointers

  - `scripts/d12_groupkf_rebuild.py` — rebuilds 7 missing bases
  - `scripts/d12_groupkf_meta.py` — K=21 LR-meta on GKF OOFs
  - `scripts/d12_groupkf_meta_no_realmlp.py` — K=20 clean variant
  - `scripts/artifacts/d12_groupkf_meta_results.json` — K=21 results
  - `scripts/artifacts/d12_groupkf_meta_no_realmlp_results.json` — K=20
  - `scripts/artifacts/oof_d12_groupkf_meta_strat.npy` — saved meta OOF
  - `scripts/artifacts/test_d12_groupkf_meta_strat.npy` — saved meta test
