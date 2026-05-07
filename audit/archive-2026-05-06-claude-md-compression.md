# Archive — 2026-05-06 CLAUDE.md compression

> Day-15 PM compression pass per `audit/2026-05-06-agentic-kaggle-research.md`
> tip #6 / item 6. Moved Day-1→Day-12 calibration-ladder rows + Day-9→Day-12
> hypothesis-board DONE entries + pre-Day-12 audit-note pointers OUT of
> CLAUDE.md to here. Nothing deleted; everything is grep-able.
>
> Intent: keep CLAUDE.md scannable for non-coding PI (≤330 lines). Active
> 6-Q precedents stay in CLAUDE.md; historical anchors live here.

## Calibration ladder — pre-Day-13 rows

| Mechanism | Strat OOF | GroupKF OOF | LB | Notes |
|---|---:|---:|---:|---|
| baseline_two_anchor | 0.94075 | 0.92059 | 0.94113 | LB-proxy ✓ gap +3.8bp |
| m3_catboost | 0.94612 | 0.91645 | n/a | depth=6, default CTR; Race-overfit |
| m4_relstate | 0.94244 | 0.92195 | n/a | only B1 lifting both anchors |
| e3_hgbc | 0.94876 | 0.92785 | 0.94870 | BEST single pre-CB; gap −0.6bp |
| e5_optuna_lgbm | 0.94736 | 0.92585 | n/a | LGBM Optuna 30 trials |
| f1_hgbc_deep / f2_hgbc_shallow | 0.94870 / 0.94861 | 0.92739 / 0.92711 | n/a | β E3 clones (~99% corr) |
| cb_year-cat | 0.94679 | 0.91992 | n/a | Year ∈ CAT_COLS; +60bp/+34bp vs M3 |
| cb_lossguide | 0.94697 | 0.92377 | n/a | Lossguide; BEST CB GroupKF (+31.8bp) |
| cb_slow-wide-bag | 0.94790 | 0.92322 | n/a | GPU 3-seed bag; BEST CB Strat (+71.5bp) |
| m5b_lr_meta_expanded | 0.94926 | 0.92871 | 0.94891 | gap −3.5bp (anchor) |
| m5d_lr_meta_expanded | 0.95023 | 0.92994 | 0.94963 | D2 PRIMARY; gap −6.0bp (widened) |
| m5e (CB-only 13-base) | 0.95027 | 0.93084 | n/a | held; M5c + 3 CB winners |
| m5f (combined 15-base) | 0.95042 | 0.93105 | n/a | held; M5d + M5e new bases |
| m5g (corr ρ≥0.97 prune) | 0.94961 | 0.92915 | n/a | TOO aggressive; 5/15 surv |
| m5h (L1coef top-13) | 0.95043 | 0.93087 | 0.94991 | OLD PRIMARY; gap −5.2bp; drop m3+m4 |
| d3a_te_unified | 0.93692 | 0.91284 | n/a | +2.2bp Strat vs d2a; std-alone redundant |
| m5i (M5h+d3a, 14) | 0.95043 | 0.93096 | n/a | d3a L1=0.079 last; tie M5h Strat |
| m5j (d3a swaps d2a, 13) | 0.95044 | 0.93092 | 0.94991 | TIED M5h LB; TE-key swap LB-neutral |
| d3b_seqfe | 0.94254 | 0.92136 | n/a | +18bp Strat over baseline; FAIL gate |
| m5k (M5h+d3a+d3b, 15) | 0.95045 | 0.93102 | n/a | d3b L1=0.316; +0.2bp Strat tie |
| m5h2_v1 (drop a_horizon, 12) | 0.95044 | n/a | 0.94991 | TIED M5h LB; pool-size gap-falsified |
| m5p (minimal+LR-FE+EBM, 6) | 0.94839 | n/a | 0.94754 | -237bp; orthogonal-mech FAILED |
| m5n_3b (minimal-basis, 4) | 0.94808 | n/a | 0.94700 | -291bp; minimal-basis FAILED |
| m5q (M5h + RealMLP, 14) | 0.95057 | n/a | 0.95005 | OLD PRIMARY; +14bp LB; 10× amp |
| d4_cb_yetirank | 0.90508 | n/a | n/a | std weak; ρ=0.666 most-diverse; +0.0bp TIE |
| d4_nb (Gaussian + TE) | 0.87984 | n/a | n/a | std weak; ρ=0.853; +0.24bp TIE |
| m5x (M5q + yetirank, K=15) | 0.95057 | n/a | n/a | held; ρ=0.99966 TIE_EXPECTED |
| m5z (M5q + yetirank + nb, K=16) | 0.95060 | n/a | n/a | held; ρ=0.99957 TIE_EXPECTED |
| m5_meta_lgbm_shallow (LGBM d=3) | 0.95048 | n/a | 0.95001 | slot 2; -4bp LB; meta-switch bounded |
| m5_meta_lgbm_medium (LGBM d=5) | 0.95047 | n/a | n/a | held; ρ=0.99436; OOF -1bp |
| m5_meta_hgbc | 0.95042 | n/a | n/a | held; ρ=0.99490; OOF -1.5bp |
| d5_recursive_m5q | 0.94994 | n/a | n/a | std-alone +92bp; ρ=0.99159 |
| d5_M5_K15a (M5q + recursive, LR) | 0.95056 | n/a | n/a | NULL (-0.06bp); ρ=0.99991 |
| d5_meta_k15_lgbm_shallow | 0.95038 | n/a | n/a | NULL (-1.0bp); GBDT-meta ceiling |
| d5_partial_pseudo_m5q (K=14) | 0.95082 | n/a | 0.94963 | slot-1 −4.2bp LB; pseudo over-amp falsified |
| d6_aux_meta_with_aux | 0.95049 | n/a | n/a | F5 falsified; +0.12bp; held |
| d6_2base_v1_lr_expand | 0.95055 | n/a | n/a | Move B falsified; ρ=0.99996 (tie) |
| d6_rule_residual (standalone) | 0.94593 | n/a | n/a | Δe3 −28bp; ρ vs M5q 0.92887 most diverse |
| d6_k15_rule_residual | 0.95062 | n/a | n/a | held; +0.51bp; ρ=0.99971; superseded by K=18 |
| d6_rule_compound_stint (std) | 0.94604 | n/a | n/a | F1.2 R2; min-meta +0.30bp PASS |
| d6_rule_driver_compound (std) | 0.94457 | n/a | n/a | F1.2 R3; ρ=0.89144; min-meta +0.45bp PASS |
| d6_rule_year_race (std) | 0.94586 | n/a | n/a | F1.2 R4; min-meta +0.37bp PASS |
| d6_k18_multi_rule | 0.95065 | n/a | 0.95026 | OLD PRIMARY (Day-7); +2.1bp LB; gap −3.9bp |
| d9_R5_weibull_compound | 0.94600 | n/a | n/a | ρ 0.943; min-meta -0.09bp FAIL |
| d9_R6_next_compound | 0.94443 | n/a | n/a | ρ 0.908; min-meta -0.12bp FAIL |
| d9_R7_prev_compound | 0.94481 | n/a | n/a | ρ 0.914; min-meta -0.10bp FAIL |
| d9_R10_driver_eb | 0.94463 | n/a | n/a | ρ 0.912 Beta-Binom; min-meta -0.10bp FAIL |
| d9_R14_hash_lr_3way (L0) | 0.79377 | n/a | n/a | ρ=0.444 most diverse; min-meta -0.02bp FAIL |
| d9b_R14_L2 | 0.91449 | n/a | n/a | ρ 0.874; min-meta +0.01bp PASS; sweet spot |
| d9b_R14_L3 | 0.91626 | n/a | n/a | ρ 0.875; min-meta +0.01bp PASS; best rung |
| d9b_R14_L4 | 0.91369 | n/a | n/a | ρ 0.869; min-meta +0.01bp PASS; K=20 chosen |
| d9b_k20_swap_l4 | 0.95067 | n/a | 0.95025 | SUBMITTED; pred +0.19bp actual −0.01bp TIE |
| d9c_FM | 0.92069 | n/a | n/a | ρ 0.899; min-meta +0.18bp PASS; new model class |
| d9c_Sd_K20_swap_FM | 0.95070 | n/a | 0.95029 | hedge; +3bp LB (5.7× upside); demoted by d9f |
| d10b_K13_baseline | 0.95043 | 0.92744 | n/a | 13 GBDT/baseline; gap −229.92bp (leak signature) |
| d10b_K15_+FMA+FMB | 0.95052 | 0.92764 | n/a | FM-class +0.87bp Strat → +2.01bp GKF (2.3× amp) |
| d10d_leak_corrected_meta | n/a | 0.92764 | n/a | held; G3 FAIL (flip 0.001); pred-LB 0.95001 |
| d12_groupkf_meta (K=21 GKF) | 0.95069 | GKF 0.94776 | n/a | STRUCTURAL: ρ(Strat-vs-GKF)=0.9914; rank-lock partial dissolves |
| d12_groupkf_meta_no_realmlp K=20 | 0.95056 | GKF 0.94577 | n/a | clean K=20; ρ vs Strat-meta 0.9856; R5 HEDGE |
| d12 single bags (e3 5seed / cb 3seed) | 0.94876 / 0.94790 | n/a | n/a | -19/-28bp every segment; K=21 complexity JUSTIFIED |

## Hypothesis board — Day-9 to Day-12 DONE history

```
- DONE: d9 simple-math rule_residual cohort — 9 of 10 FALSIFIED at min-meta vs PRIMARY (-0.09 to -0.12bp).
- DONE: d9 R14 hash_lr_3way — std 0.794 ρ=0.444 most-diverse; min-meta -0.02bp FAIL by hair.
- DONE: d9b R14 strength ladder L0-L5 — L2/L3/L4 PASS at +0.01bp; K=20 swap+L4 LB 0.95025 TIE.
- DONE: d9c FM — std 0.92069, ρ 0.899, min-meta +0.18bp PASS (18× R14 lift).
- DONE: d9c Sd K=20 swap+FM SUBMITTED → LB 0.95029 (+3bp NEW PRIMARY at the time). 5.7× upside.
- DONE: d9d FM hparam sweep + 3-seed bag — FLAT. k=8 sweet spot; bagging HURTS K=20 stack.
- DONE: d9e FFM — STRICTLY WORSE than FM (4× more params overfits 351k rows; 8 fields too few).
- DONE: d9f multi-FM partition (FM_A driver-dyn + FM_B race-ctx, ρ=0.406 ≈ orth) → K=21 swap LB 0.95031 (+2bp). 6.25× upside.
- DONE: d9g 3-way multi-FM (3+2+3) — REGRESSED at all 3 stack configs; per-FM too weak.
- DONE: d9h FM_aug12 K=22 add LB 0.95034 (+3bp NEW PRIMARY tied). 300× upside on +0.01bp OOF.
- DONE: d9i FM_aug 2-way K=21 swap LB 0.95034 (+3bp tied). OOF predicted -0.19bp REGRESSION; LB amplified positive.
- INSIGHT: FM-class OOF on Strat underestimates LB lift (P6 80% within-group leakage inflates GBDT-pool OOF). 4 consecutive FM-class submits confirm. **OOF Δ is a LOWER BOUND on LB Δ for FM-class at ρ ≈ 0.9997.**
- DONE: d10 GroupKF audit — strict (Race,Driver,Year,Stint) GKF: FM bases drop only 2.5–54bp vs GBDTs 209–247bp.
- DONE: d10b/c GroupKF stack rebuild — FM-class lift +0.87bp Strat → +2.01bp GKF (2.3× AMPLIFIED). FM_B is #1 L1 under GKF (L1=6.96, 2× next).
- DONE: d10d leak-corrected LR meta — G3 flip 0.001 (1751 rows drop top-1%). FM dominance over-credits, smooths GBDT row-extremes. HELD; pred-LB 0.95001. Bayesian hierarchical (Path B) is correct synthesis.
- DONE: d13/d13b Path B empirical-Bayes hier-meta. Compound (5seg) τ=100k +0.30bp OOF; Stint (5/6seg) τ=100k +0.86bp OOF G3 fail (4-5× better than d10d). Compound×Stint segmentation killed at fold 2.
- DONE: d13c Path B Compound τ=100k SUBMITTED → LB 0.95033 (+2bp). 6.7× LB upside on +0.30bp OOF — FM-class amp pattern TRANSFERS to hier-meta arch.
- DONE: d13 Path B Stint τ=100k SUBMITTED → LB 0.95041 (+7bp NEW PRIMARY). 11.6× LB upside on +0.86bp OOF.
- DONE: d13d GroupKF probe of hier-meta. Strat lift +0.90bp → GKF lift +2.59bp = 2.9× AMPLIFIED. **Public-LB +7bp lift is mechanism-driven, not sample variance.**
- DONE Day-12: 6 wider-step options run as parallel subagents overnight. 5 FALSIFIED. Option 1 produced load-bearing finding.
- DONE: Option 1 GroupKF full rebuild — K=21 LR-meta on GKF OOFs ρ=0.9914 vs Strat-meta. Rank-lock partially dissolves under leakage-blocked OOF. GBDT ΔAUC −200 to −343bp; FM/rule/sparse-LR −9 to −43bp. **FM is 23–37× more leakage-robust than every GBDT.**
- DONE: Option 9 single-bag probe — falsifies "OOF-noise overfit" thesis. K=21 stack complexity JUSTIFIED.
- DONE: Option 3 T1.2 multi-formulation 3-of-3 (censored/ratio/survival LGBMs) — ALL FAIL min-meta. Time-to-event LGBMs are GBDT-class.
- DONE: Option 4 Year-specialist + AV-reweight — BOTH FAIL -4.5 to -5.0bp. AV-AUC=0.502 (i.i.d.); 2023 EASIEST segment; cohort splitting strips cross-Year regularization.
- DONE: Option 5 LambdaRank meta — REGRESSES -86bp under Race grouping; AUC-pairwise XGB -451bp fold-0. Dead-list (Q6 origin).
- INSIGHT (Day-12 unifying): K=21 stack works because LR-meta routes between leakage-eating GBDTs and leakage-robust FM/rules. Public LB row-iid (U3) so PRIMARY survives. Diversification needed WITHIN leakage-robust population.
```

## Pointers — pre-Day-12 audit notes

- `audit/2026-05-04-strategy-critique.md` — Rule 14 origin.
- `audit/2026-05-04-catboost-research.md` — CatBoost lever map.
- `audit/2026-05-04-m5h-l1coef-prune.md` — Day-3 submit candidate.
- `audit/2026-05-04-d3a-te-unified.md` — Step 1 result + M5i/M5j.
- `audit/2026-05-04-d3b-seqfe.md` — Step 2 result + M5k.
- `audit/2026-05-05-d4-yetirank-nb-results.md` — Day-4 base-add probes.
- `audit/2026-05-05-d4-gbdt-meta-breakthrough.md` — Day-4 slot-2 envelope.
- `audit/2026-05-05-nn-stack-priorities.md` — bigger-move ordering.
- `audit/2026-05-07-d6-critic-loop.md` — Rule 14 audit; 5 untried mechanisms.
- `audit/2026-05-07-d6-f5-aux-meta-result.md` — F5 falsified.
- `audit/2026-05-07-d6-move-b-2base-recursive.md` — Move B falsified.
- `audit/2026-05-07-d6-move-c-rule-residual.md` — F1.1 single rule.
- `audit/2026-05-07-d6-f1-2-multi-rule.md` — F1.2 K=18 LB-landed +2.1bp.
- `audit/2026-05-09-d9-math-heuristics.md` — d9 10-approach cohort, all min-meta FAIL vs PRIMARY.
- `audit/2026-05-09-d9b-r14-ladder.md` — d9b R14 ladder L0-L5; K=20 swap+L4 SUBMITTED LB 0.95025 TIE.
- `audit/2026-05-09-d9c-fm.md` — d9c FM passes min-meta +0.18bp; Sd K=20 swap+FM LB 0.95029 (+3bp).
- `audit/2026-05-10-d9d-fm-sweep-bag.md` — FM hparam sweep + bag NULL; bag HURTS stack.
- `audit/2026-05-10-d9e-ffm.md` — FFM strictly worse than FM.
- `audit/2026-05-10-d9f-multi-fm.md` — multi-FM partition K=21 swap LB 0.95031 (+2bp prior PRIMARY).
- `audit/2026-05-10-d9g-3way-multi-fm.md` — 3-way partition REGRESSED.
- `audit/2026-05-10-d9h-fm-augmented.md` — FM_aug12 standalone strongest; K=22 add LB 0.95034 (+3bp).
- `audit/2026-05-10-d9i-augmented-2way.md` — aug 2-way K=21 swap LB 0.95034 (+3bp).
- `audit/2026-05-10-d10-groupkf-audit-fm-real.md` — strict GKF FM bases drop 2.5–54bp vs GBDTs −210bp.
- `audit/2026-05-10-d10b-groupkf-stack-rebuild.md` — FM-class +2.01bp GKF (2.3× AMPLIFIED).
- `audit/2026-05-10-d10d-leak-corrected-meta.md` — leak-corrected LR meta gate-FAILs G3.
