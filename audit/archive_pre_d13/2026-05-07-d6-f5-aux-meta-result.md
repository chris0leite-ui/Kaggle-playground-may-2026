# Day-6 F5: aux-feature GBDT-meta — FALSIFIED, do not slot

> Critic-loop §3 F5 hypothesis: replacing LR-meta with LGBM-shallow
> meta over (raw + rank + per-row aux disagreement features) breaks
> the LR-meta-rank-lock. Test result: **null at +0.12bp over no-aux
> LGBM baseline; OOF still 0.78bp below M5q anchor.**

## Result

| variant | F | Strat OOF | Δ M5q | Δ no-aux | ρ vs M5q test | gate |
|---|---:|---:|---:|---:|---:|---|
| noaux_replicate (28F) | 28 | 0.95048 | −0.92bp | 0 | 0.99508 | PASS |
| with_aux (38F) | 38 | 0.95049 | −0.78bp | **+0.12bp** | 0.99476 | PASS |

Anchors: M5q LR-meta 0.95057 / LB 0.95005; m5_meta_lgbm_shallow
no-aux 0.95048 / LB 0.95001 (d4 slot-2).

## Top features (with_aux)

```
raw__f1_hgbc_deep         566515  ← still dominant
raw__f2_hgbc_shallow      391653
aux__median               292243  ← aux DID get used (#3 by gain)
raw__e3_hgbc              287293
raw__e5_optuna_lgbm        91043
rank__f1_hgbc_deep         80376
...
aux__mean                  23791  ← #11
```

`aux__median` ranked #3 by gain — disagreement signal IS being
extracted by the meta. But the marginal AUC gain is sub-bp (+0.12bp
over no-aux). The other 8 aux features (std, max, min, range, skew,
iqr, count_hi, count_lo) collectively contributed below the top-15.

## What this proves

**Third independent confirmation that base-pool signal is the
binding constraint, not meta-learner expressiveness.** Prior:
1. `lr-meta-rank-lock-strong-anchor` (Day-4) — LR meta with ρ=0.666
   yetirank base produces 0.99966 stack-level ρ vs M5q.
2. `rho-0.995-not-tie-meta-switch-bounded` (Day-4) — LGBM-meta switch
   regresses 4bp LB at ~50% OOF→LB transfer.
3. **F5 (today)** — adding 10 disagreement aux features lifts OOF
   only 0.12bp over the same LGBM-meta. The disagreement signal that
   LR cannot use is partially extracted by GBDT, but the ceiling is
   set by the L1 pool, not the meta architecture.

## Predicted LB if slotted

OOF 0.95049 vs M5q 0.95057 = −0.78bp.  ρ=0.99476 implies meta-switch
regime (~50% OOF→LB transfer per Day-4 calibration), so predicted
LB ≈ 0.95005 − 0.4bp = **0.95001** (matches m5_meta_lgbm_shallow LB
exactly: 0.95001). Predicted gap −5.6bp, within the systematic
−5.0±0.4bp band. **Slot would burn calibration data, not lift.**

## Verdict — F5 FALSIFIED

Do not slot d6_aux_meta_with_aux. Move B (2-base [M5q, recursive]
LB probe) is now the highest-EV cheap probe; F1 (hazard-rate
reformulation) becomes the Day-7/8 anchor build with elevated
priority.

## Artifacts

- `scripts/d6_aux_meta.py` — script
- `scripts/artifacts/d6_aux_meta_results.json` — full result
- `scripts/artifacts/d6_aux_meta.log` — run log
- `submissions/submission_d6_aux_meta_with_aux.csv` — held, do not push
- `oof_d6_aux_meta_with_aux_strat.npy`, `test_d6_aux_meta_with_aux_strat.npy`

## Strategic re-rank (post-F5)

| # | Move | New status |
|---|---|---|
| A | F5 aux-meta | **FALSIFIED** ✗ |
| B | 2-base [M5q, recursive] LB | promoted to next slot |
| C | F1 hazard-rate L1 reformulation | promoted to Day-7 anchor |
| D | TabM 1-fold smoke | overnight Kaggle |
| F | Multi-seed RealMLP bag | overnight Kaggle |
| E | F4 sequence-FE LGBM probe | day build |
