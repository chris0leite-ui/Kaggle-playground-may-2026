# Day-7 RealMLP partial-bag salvage — NULL, do not slot

> Kaggle kernel `realmlp-bag-gpu` cancelled mid-fold-3 of seed 123
> after parallel-branch probes (P10 in `audit/2026-05-08-data-probe-
> results.md`) downgraded bag EV to Tier-3 and freed GPU for higher-
> EV moves (T1.3 Q12 forced-pit, T1.1 TabM).

## Salvaged artifacts (kernel cancellation)

- `oof_realmlp_seed123_partial_3fold_strat.npy` — 60% coverage
  (3/5 fold val partitions filled; 40% remain 0)
- `test_realmlp_seed123_partial_3fold_strat.npy` — sum over 3 folds
  with `/N_FOLDS=5` divisor; rescaled `× 5/3` for valid 3-fold avg
- per-fold seed-123 AUCs from log: f0=0.94724, f1=0.94535, f2=0.94619
  (mean 0.94626 vs seed-42 5-fold OOF 0.94582 = +4.4bp per-fold)

## Two salvage paths tested in `scripts/d7_realmlp_partial_bag.py`

| Path | OOF source | TEST source | K=18 OOF | Δ d6_k18 | ρ vs d6_k18 |
|---|---|---|---:|---:|---:|
| B | seed-42 only | rank-avg(seed-42, seed-123-rescaled) | 0.95065 | **−0.02bp** | **0.99955** |
| C | hybrid (60% bag, 40% seed-42) | same as B | 0.95066 | +0.08bp | 0.99964 |

Both above the tightened 0.9995 ρ threshold (Day-6 friction lesson)
→ **TIE regime, wasted slot**. Verified that seed-123 covered-row
OOF AUC = 0.94624 — matches per-fold mean, no genuine lift over
seed-42 at the K-stack scale.

## Mechanism diagnosis

- RealMLP's L1 weight in K=18 = 0.698 (4th of 18 bases).
- Reducing variance on 1 of 18 bases when the other 17 already have
  full 5-fold OOF gives ≤0.1bp at the stack level.
- Parallel-branch's Tier-3 classification (`T3.4 snapshot
  ensembling on RealMLP, EV +0.5-3bp`) is confirmed: this is base-
  level variance reduction, not pool-level diversity addition.
- Even a complete 2-seed bag (had we let the kernel finish 4h more
  GPU) would have produced ~+0.3-1bp K-stack OOF max — very likely
  tied or +1 quantum LB.

## Held submissions (do NOT submit)

- `submissions/submission_d7_realmlp_bag_partB.csv` — TIE regime
- `submissions/submission_d7_realmlp_bag_partC.csv` — TIE regime

## Next moves (per parallel-branch strategic menu)

1. **CPU — T1.3 Q12 forced-pit rule_residual** (+5-10bp standalone
   prior). Slots into F1.2 mechanism unchanged. K=19 stack rebuild.
2. **GPU — T1.1 TabM 1-fold smoke** (Rule 2). Different inductive
   bias from RealMLP-TD. EV +2-8bp K-stack.
3. **Drop F4 sequence model** from menu — parallel-branch P1
   probe falsified (test groups average 2.25 laps; only 9.7% have
   ≥5 consecutive).

## Pointers

- `scripts/d7_realmlp_partial_bag.py` — salvage script
- `scripts/artifacts/d7_realmlp_partial_bag_results.json` — full result
- `audit/2026-05-08-strategic-menu-wider-steps.md` (parallel branch)
  — Tier-1/2/3 ranking with EV math
- `audit/2026-05-08-data-probe-results.md` (parallel branch) —
  P1-P10 probes that updated priors
