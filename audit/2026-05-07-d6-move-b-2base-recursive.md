# Day-6 Move B: 2-base [M5q, recursive] — FALSIFIED

> Critic-loop §4 Move B / HANDOVER: "K=2 OOF stack was −0.2bp but
> rank structure is structurally different from K=15. Pre-submit-
> diff vs M5q first; if ρ < 0.999, slot." Tested 4 K=2 blend
> variants. **None slot-worthy.**

## Result

| variant | Strat OOF | Δ M5q | ρ vs M5q | pred-LB | verdict |
|---|---:|---:|---:|---:|---|
| V1 LR-expand (raw+rank+logit) | 0.95055 | −0.17bp | **0.99996** | 0.95003 | tie regime |
| V2 prob-avg (no fit) | 0.95039 | −1.75bp | 0.99748 | 0.94977 | OOF regression |
| V3 rank-avg (no fit) | 0.95041 | −1.63bp | 0.99782 | 0.94979 | OOF regression |
| V4 LGBM-shallow meta | 0.95050 | −0.69bp | 0.99771 | 0.94988 | OOF regression |

ρ(M5q OOF, recursive OOF) = 0.98875; ρ(M5q test, recursive test) =
0.99159 (recursive standalone OOF 0.94994).

## What this proves

**The K=2 LR-expand stack is NOT structurally different from K=15
at the rank level.** V1 ρ=0.99996 vs the K=15 LR-stack's 0.99991 —
both are well inside Kaggle 5-decimal tie territory. The HANDOVER
bet on K=2 having different rank structure than K=15 is falsified.

L1 weights tell the story:
```
M5q raw=0.084  rec raw=0.032
M5q rk =0.047  rec rk =0.028
M5q lg =0.918  rec lg =0.071   ← essentially all weight on logit(M5q)
```
Given two correlated bases × 3 expansion views, LR finds `logit(M5q)`
is the optimal univariate predictor. Rank-lock laid bare.

V2/V3 (unfit averages) DO change rank structure (ρ ≈ 0.997–0.998)
but pay a 1.6–1.75bp OOF regression because they give recursive
equal weight despite being 6bp worse standalone. V4 GBDT-meta finds
no non-linear lift that compensates.

## Strategic implication

The recursive base is **structurally redundant with M5q** at the
2-base meta level. Mechanism: recursive was trained on raw features
+ `m5q_oof_proba` as a feature, so it's a residual model conditioned
on M5q. Any 2-base meta that gives M5q full weight (V1) tie-locks;
any meta that re-weights toward recursive (V2/V3/V4) regresses
because recursive's standalone signal is dominated by M5q's
consensus.

**Fourth independent confirmation that base-pool signal is the
binding constraint** (after F5 today + 3 prior rank-lock confirms).
Cannot extract incremental LB from this base via any K=2 meta.

## Re-rank — both Move A (F5) and Move B falsified

| # | Move | Status |
|---|---|---|
| A | F5 aux-meta | FALSIFIED ✗ |
| B | 2-base [M5q, recursive] | FALSIFIED ✗ |
| **C** | **F1 hazard-rate L1 reformulation** | **promoted to TOP next move** |
| D | TabM 1-fold smoke (Kaggle T4) | overnight |
| F | Multi-seed RealMLP bag (Kaggle GPU) | overnight |
| E | F4 sequence-FE LGBM probe | day build |

The two cheap-and-fast options at the meta layer (F5, B) are both
exhausted. Day 6 has produced two clean falsifications and zero
slot candidates — exactly the signal that the binding constraint is
at the L1 base / problem-formulation layer, not the meta. **F1 is
now the only EV-positive direction left at the L1 layer that
doesn't require Kaggle GPU**, and its prior (Deotte's April-2025
winner used 4 problem formulations at L1) is the strongest medium-
term lift on offer.

## Held artifacts (do not submit)

- `submissions/submission_d6_2base_v1_lr_expand.csv` — tie regime
- `submissions/submission_d6_2base_v2_prob_avg.csv` — regression
- `submissions/submission_d6_2base_v3_rank_avg.csv` — regression
- `submissions/submission_d6_2base_v4_lgbm_shallow.csv` — regression
- `scripts/artifacts/d6_two_base_recursive_results.json` — full result
- `scripts/d6_two_base_recursive.py` — script

## Next move per audit ordering

1. **C (F1 hazard-rate reformulation)**: build today/tomorrow as
   the Day-7/8 anchor. Predict `P(pit_at_lap_k | survived to lap k)`
   via discrete-time hazard model on the lap sequence. Different
   target structure → different rank ordering → real L1 diversity.
2. **F (multi-seed RealMLP bag)**: push to Kaggle GPU overnight,
   parallel to F1 build.
3. **E (sequence-FE LGBM probe)**: 2h CPU, can run alongside F1
   build if tools available.
