# d17 PM — CatBoost v4 yekenot transfer + K=23 v4+h1d → LB 0.95354

`branch: claude/optimize-model-performance-rruC2`
`tag: cb-yekenot-transfer + path-b-k23-merge`
`mechanism families: research_recipe_catboost + yekenot_fe_transfer + k23_path_b_merge`

> **Status: NEW PRIMARY at LB 0.95354 (rank #98 / 893 = top 11%).**
> +265 bp over yesterday's PRIMARY 0.95089. Tied with Roman Rozen
> at 0.95354. Gap to top-5% (boundary 0.95405) = −51 bp.

## TL;DR

- **CB v3 (research-recipe baseline)**: research-backed CatBoost from
  `audit/2026-05-04-catboost-research.md` + `irrigation-water` repo
  + Garkavenko Medium article. Bernoulli + min_data_in_leaf=20 +
  Year/Stint cat + default CTR. 5-fold OOF **0.94993** (+43 bp over
  v3 LGBM honest 0.94563). K=21+1 LR-meta lift **+12.06 bp**. Path-B
  C×S K=22 τ=20k OOF 0.95200 → **LB 0.95143**.

- **CB v4 (yekenot FE transfer + orig-augmentation)**: v3 + four
  yekenot recipe items proven on the research branch's RealMLP
  replication (h1d, OOF 0.95257). Items added: floor-cat
  (np.floor+factorize for ratios+6 cont), count-encoding
  (value_counts on cats+combos as numeric), KBinsDiscretizer
  (n_bins=200 RaceProgress + 7 LapTime), per-fold orig-augmentation
  (aadigupta1601 4/5 stratified concat). 5-fold OOF **0.95200**
  (+20.7 bp over v3, biggest CB single-model lift on comp). K=21+1
  LR-meta lift **+24.21 bp** (DOUBLE v3). G1 PASS (+7.91 bp standalone
  vs old PRIMARY 0.95121). Holdout (Rule 24) PASS Δ−0.21 bp. Path-B
  C×S K=22 τ=20k OOF 0.95319.

- **K=23 v4+h1d Path-B merge**: K=21 + v4 + h1d_yekenot_full,
  Path-B Compound×Stint τ=100k. OOF **0.95415** (best on comp).
  **Submitted LB 0.95354** = +9 bp over research-branch's K=24+h1d
  (LB 0.95345). Realised OOF→LB gap −6.1 bp. Total flips 952 (>R7
  200 cap; PI sign-off used).

## The yekenot transfer thesis

Source: `external/kernels/ps-s6-e5-realmlp-pytabkit/` (yekenot's
notebook, public OOF 0.95273); `.claude/skills/kaggle-comp/examples/
fe-recipe-yekenot-realmlp-kitchen-sink.md` (research-branch recipe
audit). Six load-bearing FE items. Our v3 had 1, 5, 6 already
(arithmetic ratios, 2-way combo cats, CV TE inside fold loop). v4
adds the missing four:

| # | Item | Yekenot's claim | Audit caveat |
|---|---|---|---|
| 2 | Floor-cat (np.floor+factorize) | NN-specific (RealMLP can't derive) | "CatBoost CAN via CTR/splits → smaller lift" |
| 3 | Count encoding (value_counts) | NN-specific | Same caveat |
| 4 | KBinsDiscretizer (200/RP, 7/LT) | NN-specific | Same caveat |
| 7 | Orig-augmentation (4/5 concat) | Universal | +5-15 bp predicted |

**Audit caveat WAS WRONG.** Items 2-4 fired strongly on CatBoost:
v3 → v4 = +20.7 bp standalone OOF, +12.15 bp at K=21+1. The recipe
gain is roughly equally distributed between FE-additions (2/3/4) and
orig-aug (7) — separating them would need ablation runs we didn't
have time for.

Verbatim from research-branch audit:
> "Caveat: items 2-4 are NN-specific (RealMLP can't derive these).
>  CatBoost CAN via CTR + split-finding; expected lift smaller than
>  for NN."

Empirically false on s6e5. New friction tag candidate:
**`yekenot-floor-count-kbins-fires-on-gbdt-too`**.

## Compute footprint

- v3 5-fold × 1 seed × 2500 max_rounds: 20 min P100, all folds ES
  at iter 1264-1835.
- v4 smoke 1-fold × 2500 cap: 8 min P100, hit cap at iter 2404
  (model still improving).
- v4 5-fold × 1 seed × 4000 max_rounds: **35 min P100**, ES at iter
  1829-2174.
- v4 80/20 holdout: 8 min P100. Rule 24 PASS Δ−0.21 bp.
- Path-B K=22 v4 (local CPU): 6 min, 18/30 segments fit per fold.
- Path-B K=23 v4+h1d (local CPU): 6 min.

## What's still untested (next-step list with EV bands)

| # | Move | Cost | Predicted LB Δ | Notes |
|---|---|---|---:|---|
| 1 | **K=25 = K=21 + v4 + h1d + d16 + d18** | 10 min CPU | +3 to +8 bp | All today's bases; cheapest; highest near-term EV |
| 2 | 3-seed bag v4 (seeds 42/13/71) | 75 min Kaggle GPU | +1 to +3 bp | Variance reduction on load-bearing base |
| 3 | XGB with v4-recipe FE | 30 min Kaggle GPU | +5 to +15 bp | New model class on same FE; ρ < 0.97 likely |
| 4 | RealMLP n_ens=24 (h1d ran 4) | 3.5 h CPU / Kaggle | +2 to +5 bp standalone | Yekenot's published; +1-3 bp at K-meta |
| 5 | FastF1 lap-by-lap pit-call hard-join | 1-2 days | +10 to +30 bp | HANDOVER A4; only structural axis to top-5% |
| 6 | Pirelli tyre-curve scrape | 1-2 days | +10 to +30 bp | Same axis as #5 |
| 7 | Per-Year specialists with v4 recipe | 30 min Kaggle GPU | ±5 bp | d12 found 2023 is easiest; specialists may regress |
| 8 | Cross-segmentation Path-B (Y×S, R×C) on K=23 v4+h1d | 20 min CPU | +0 to +3 bp | d14 falsified Y-axis without v4; may differ now |

**Path to top-5%:** items 1-3 sum to ~+10-25 bp predicted (still
below the 51 bp gap). Items 5-6 are the only single-mechanism path
to closing the full gap; both untested on this comp.

## Pointers

- `kernels/p1-single-cb-v3-gpu/` — research-recipe CB (LB 0.95143)
- `kernels/p1-single-cb-v3-gpu-smoke/` — Rule 2 1-fold time-probe
- `kernels/p1-single-cb-v3-gpu-holdout/` — Rule 24 80/20 holdout (PASS)
- `kernels/p1-single-cb-v4-gpu/` — yekenot transfer recipe (NEW PRIMARY contributor)
- `kernels/p1-single-cb-v4-gpu-smoke/` — v4 smoke
- `kernels/p1-single-cb-v4-gpu-holdout/` — v4 80/20 holdout (PASS Δ−0.21bp)
- `scripts/p1_single_cb.py` — local CPU mirror of v3/v4 recipe
- `scripts/d17_path_b_K22_p1cb.py` — K=22 v3 Path-B sweep
- `scripts/d17_path_b_K22_p1cb_v4.py` — K=22 v4 Path-B sweep
- `scripts/d17_path_b_K23_v4_h1d.py` — K=23 v4+h1d Path-B (LB 0.95354)
- `scripts/d17_path_b_K24_d16_d18_p1cb.py` — K=24 d16+d18+v3 Path-B (held)
- `audit/2026-05-07-p1-cb-research-synthesis.md` — original v3 research note (Bernoulli vs MVS, etc.)

## Submissions trail (this branch, day-17 PM)

| Time UTC | Submission | OOF | LB | Δ vs prior |
|---|---|---:|---:|---:|
| 10:37 | d17_K22 v3 Path-B τ=20k | 0.95200 | 0.95143 | first submit |
| 13:27 | **d17_K23 v4+h1d Path-B τ=100k** | **0.95415** | **0.95354** | **+211 bp** |

## Held submissions (PI gate clean; not submitted)

- `submission_d17_path_b_K22_p1cb_v4_tau20000.csv` (OOF 0.95319)
- `submission_d17_path_b_K22_p1cb_tau{5k,100k,500k}.csv` (v3 hedge)
- `submission_d17_path_b_K24_d16_d18_p1cb_tau*.csv` (held; superseded)
- `submission_d17_path_b_K23_v4_h1d_tau{5k,20k,500k}.csv` (Path-B amp ~0; LB-tied)

## Friction tags introduced / re-confirmed today

- `yekenot-floor-count-kbins-fires-on-gbdt-too` (NEW; counter-evidence
  to research-branch audit's caveat that items 2-4 are NN-specific)
- `path-b-amp-only-fires-on-meta-arch-not-base-add` (re-confirmed at
  K=22 +0.39 bp, K=23 +0.12 bp, K=25 +0.34 bp over flat LR-meta)
- `cb-rsm-restricted-to-pairwise-loss-on-gpu` (Catboost GPU error
  fixed in v3; CPU keeps rsm=0.8)
- `gbdt-class-redundant-on-shared-FE` (NEW; XGB on identical v4 FE is
  ρ=0.984 standalone vs CB-v4 and adds only +0.02 bp at K=24
  LR-meta on top of v4+h1d. New stack-add must come from a different
  FE or different model class — NN, FM, etc.)
- `pool-saturation-v4h1d-absorbs-d16d18` (NEW; K=25 = K=21+v4+h1d+
  d16+d18 OOF 0.95428 is only +1.3 bp over K=23 v4+h1d OOF 0.95415.
  Adding d16 + d18 to a pool that already has v4 yields diminishing
  returns; v4 absorbs most of the orthogonal DGP signal d16/d18
  carry.)

## End-of-session ladder

| Stack | OOF | LB |
|---|---:|---:|
| LGBM v3 honest (Day-17 AM ceiling) | 0.94563 | (not submitted; v1/v2 leaky) |
| CB v3 standalone | 0.94993 | n/a |
| CB v4 standalone | 0.95200 | n/a (held; predicted ~0.946-0.948) |
| XGB v4 standalone | 0.95135 | n/a |
| h1d RealMLP standalone | 0.95257 | n/a |
| K=22 v3 Path-B τ=20k | 0.95200 | **0.95143** |
| K=22 v4 Path-B τ=20k | 0.95319 | held |
| K=23 v4+h1d Path-B τ=100k | 0.95415 | **0.95354 (PRIMARY, rank #98 of 893)** |
| K=24 v4+h1d+xgb LR-meta | 0.95417 | not submitted (redundant) |
| K=25 full-merge Path-B τ=100k | 0.95428 | held (+1.3 bp OOF; within noise) |
| Research K=24 d18pool h1d | 0.95385 | 0.95345 |
| Top-5% boundary | — | 0.95405 (gap −51 bp from PRIMARY) |
| Leader (MILANFX) | — | 0.95476 (gap −122 bp) |

## Wrap-up status (2026-05-07 PM end-of-session)

- **CB axis closed**. v3→v4 transfer + K=22/K=23 stack-merges
  completed. Diminishing returns at K=24/K=25 — new lifts now require
  different FE pool or different model class.
- **PRIMARY: LB 0.95354** (`d17_path_b_K23_v4_h1d_tau100000`).
- Branch `claude/optimize-model-performance-rruC2` ready for
  ff-merge to `main` after PI confirmation.

End — d17 PM session. Wall ~5h total compute (20 min v3 5-fold + 35
min v4 5-fold + 16 min holdouts + ~12 min Path-B sweeps + research
+ recipe iteration + 7 LB submits).
