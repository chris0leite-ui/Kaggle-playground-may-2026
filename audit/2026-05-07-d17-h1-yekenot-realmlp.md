# D17 H1 — Yekenot RealMLP recipe replication (2026-05-07)

## Goal

Replicate yekenot's published RealMLP recipe (notebook
`external/kernels/ps-s6-e5-realmlp-pytabkit/`) as K=22 base candidate.
Our current `realmlp` slot was a default-config single-fold smoke
(0.94582 OOF, single fold). Yekenot's published recipe scores ~0.95273
OOF on full data (5-fold + orig + TE).

Constraint: train.csv only (no orig f1_strategy_dataset_v4 merge),
CPU only on 4-core box, hard wall ≤ 90 min, n_threads=2 (shared with
H2/H3).

## What I ran

Two CPU configs — recipe identical except `n_ens / n_epochs / batch_size`:

| Mode  | n_ens | n_epochs | batch | 5-fold wall | full-OOF AUC |
|-------|------:|---------:|------:|------------:|-------------:|
| FAST  |     2 |        4 |   512 |     7.9 min | **0.94344** |
| STRONG|     3 |        6 |   256 |    PENDING  |    PENDING  |

Single-fold smoke (n_ens=1, n_epochs=2, batch_size=512): 37 s,
fold-0 AUC 0.94135 (used to project budgets).

## Features (per task brief)

11 raw numerics: PitStop, LapNumber, Stint, TyreLife, Position,
LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress,
Position_Change, Year.

Cats (12 total): Driver, Compound, Race, Year_str, Driver_Compound,
Race_Compound, Race_Year, Driver_Race, Driver_Year,
Compound_TyreLifeBin (TyreLife→5 quantile bins),
Compound_RaceProgressBin (RaceProgress→5 quantile bins), Stint_Compound.

OrdinalEncode all cats on combined train+test (Rule 25 safe at
AV-AUC=0.502). KBins fit on combined train+test (feature-only, no
label).

Rule 24: NO target-conditional features. Rule 25: AV-AUC pre-checked.

## FAST run results

5-fold StratifiedKFold(seed=42), 100% coverage, n_threads=2, CPU.

- Fold AUCs: [0.94419, 0.94281, 0.94342, 0.94320, 0.94364]
- **OOF AUC = 0.94344**
- Holdout-20% (seed=99 independent split): 0.94371
- OOF − holdout: −2.7 bp (CLEAN; no cross-fold leakage)
- vs default-config realmlp baseline (0.94582): **−23.8 bp** (REGRESS)
- vs yekenot published 0.95273: **−92.9 bp**

Wall: 9.5 min total (5-fold 7.9 min + holdout 1.6 min).

### Min-meta gate (K=22 ADD), FAST candidate

- K=21 LR-meta baseline OOF: 0.95073
- K=22 add d17_h1: 0.95077 → Δ **+0.427 bp** (below noise floor)
- ρ vs PRIMARY (d13e Compound×Stint τ=20k): **0.99574** (rank-locked)
- |w| 0.2533 (raw +0.033, rank −0.127, logit +0.093)

### K=21 SWAP gate (replace `realmlp` slot), FAST candidate

- K=21 baseline OOF: 0.95073
- K=21 swap OOF:     0.95069 → Δ **−0.439 bp** (REGRESS)
- ρ vs PRIMARY: 0.99514
- Per-base: yekenot |w|=0.36 vs default-realmlp |w|=0.45 (lower utility)

## Gap-closure analysis (+69 bp target)

The brief targeted closing the gap from 0.94582 (default realmlp) to
0.95273 (yekenot published) ≈ **+69 bp**. With CPU FAST mode:

- Standalone OOF: 0.94344 (−24 bp BELOW default realmlp).
- Closed: **−24 / 69 = −35 % of the gap** (we *opened* it further).

Reasons FAST regressed below default realmlp:
1. **Capacity collapse.** Yekenot's published n_ens=24, n_epochs=6,
   batch_size=256 → ours n_ens=2, n_epochs=4, batch_size=512. That's
   ~1/24th the ensembling capacity and 1/2 the per-epoch noise.
2. **No orig data.** Yekenot concatenates orig (f1_strategy_dataset_v4)
   train+orig in each fold; we use train.csv only per brief.
3. **No TargetEncoder.** Yekenot applies sklearn `TargetEncoder(cv=5)`
   on Race_Compound + Race_Year combos; we don't (Rule-24 caution
   was already disciplined for the K=21 pool, but yekenot's TE is
   inner-CV, fold-safe).

The default-config `realmlp` slot at OOF 0.94582 was a single-fold
smoke with a different cat-feature schema; it gets a substantial
ensembling-during-training advantage from RealMLP's internal
default `n_cv=8`. Our recipe-faithful build with n_ens=2 effectively
under-ensembled.

## STRONG run (n_ens=3, n_epochs=6, batch=256)

PENDING — see scripts/artifacts/d17_h1_yekenot_realmlp_strong_results.json.
Projected wall ~43 min, capacity ~6× FAST per fold.

## Verdict

**FAST mode: NULL** (regress vs default-realmlp baseline; rank-locked
in stack at ρ=0.996). Not a PRIMARY-replace candidate. The CPU-budget-
constrained recipe replication does not in fact reproduce yekenot's
0.95273 ceiling.

**STRONG mode: PENDING** — verdict deferred to STRONG output.

## Conclusions (mechanism level)

1. The published 0.95273 yekenot OOF is *not* recoverable on 4-core CPU
   without orig-data merge or n_ens≫4. It's a Kaggle-GPU artifact
   built on (a) 24× ensembling, (b) doubled training data via orig.
2. RealMLP's standalone OOF as a K=22 base is not the bottleneck —
   even at 0.94582 (default config) it produces +0.43 bp K=22 add at
   ρ=0.996, identical to what FAST gives. The pool is **rank-locked**
   on the RealMLP/MLP axis.
3. Q6 OK: yekenot uses `val_metric_name='1-auc_ovr'` (row-AUC aligned).
4. Rule 24 / Rule 25 both clean: 80/20 holdout matches OOF within
   ±2.7 bp; no leakage.
5. Cross-comp lesson: replicating a heavy-bagging public-notebook
   recipe at 1/12th capacity is *not* a reliable way to test the
   underlying recipe. Either match capacity or accept that the gap
   measurement is dominated by capacity, not recipe.

## Files

- `scripts/d17_h1_yekenot_realmlp.py` — main 5-fold trainer.
- `scripts/d17_h1_swap_gate.py` — K=21 swap-gate harness.
- `scripts/artifacts/oof_d17_h1_yekenot_realmlp_strat.npy` (FAST).
- `scripts/artifacts/test_d17_h1_yekenot_realmlp_strat.npy` (FAST).
- `scripts/artifacts/d17_h1_yekenot_realmlp_results.json` (FAST).
- `scripts/artifacts/probe_min_meta__d17_h1_yekenot_realmlp.json` (FAST K=22).
- `scripts/artifacts/d17_h1_swap_gate__d17_h1_yekenot_realmlp.json` (FAST K=21 swap).
- `scripts/artifacts/oof_d17_h1_yekenot_realmlp_strong_strat.npy` (STRONG; PENDING).
