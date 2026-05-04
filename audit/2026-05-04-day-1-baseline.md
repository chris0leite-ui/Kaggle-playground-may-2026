# Day-1 audit — baseline calibration

## Submission

- File: `submissions/submission_baseline_two_anchor.csv`
- Mechanism: LightGBM, raw 14 features, 5-fold StratifiedKFold(seed=42)
- Hyperparams: lr=0.05, num_leaves=63, min_data_in_leaf=200,
  num_boost_round≤2000 (best ≈ 380 across folds), ES=100
- OOF (StratKFold): **0.94075** ; fold_std 0.00075 (tight)
- OOF (GroupKFold Race): 0.92059 ; fold_std 0.01306 (race-robustness)
- Public LB: **0.94113**
- **Gap LB − OOF_A: +3.8bp** (LB is 3.8bp HIGHER than OOF)

## Calibration verdict

Excellent. The +3.8bp gap is well inside the probe-resolution floor
(~14bp at n_public ≈ 37,633, prior 0.20). Anchor A (StratKFold) was
the right LB proxy — confirms U3's i.i.d. row-split finding.

The 200bp gap to anchor B (GroupKFold Race) is structural
within-race signal exploitable at test time, NOT leakage. Top
public notebooks reach OOF 0.93–0.96 with plain StratKFold and
public LB 0.954, all consistent.

## Headroom analysis

- Top-5% threshold (rank 27 of 542 at kickoff): **0.95345**
- Our baseline LB: 0.94113
- **Gap to top-5%: +123bp** (achievable in 26 days)

Plausible budget (from metric_notes & top-notebook prior_art):
- FE (target encoding + interactions): 30–60bp
- External data join: 10–30bp
- GBDT trio blend (LGB+XGB+CatBoost rank-mean): 10–20bp
- NN integration (RealMLP-style): 5–15bp
- Stacker on rank features: 5–10bp
- Total achievable: 60–135bp

Top-5% is reachable, not guaranteed. Calibration ladder must keep
us at sub-30bp gaps as we add lift; if a candidate drifts >50bp
on OOF→LB we re-run U-series probes.

## Friction (Day-1)

Two events logged in `audit/friction.md`:
1. `tag: stats-error` — bogus "structural lag" claim from a match
   rate that was indistinguishable from independent baseline.
2. `tag: cv-anchor-context` — auto R1 verdict "gap >50bp ⇒ leakage"
   needs a qualifier requiring the test to hold out by group.
   Skill update queued in `pre-baseline-gate.md` /
   `metric_notes` defaults.

## Day-2 queue (PI to prioritise)

```
Candidate A — external data join
  spec: download aadigupta1601/f1-strategy-dataset-pit-stop-prediction;
        DROP Normalized_TyreLife (host removed); concat to train;
        re-fit LGBM with same hyperparams; two-anchor OOF
  expected lift: +10–30bp
  effort: ~30 min (download + join + fit)
  risk: low (host explicitly allows; prior_art uses this pattern)

Candidate B — feature engineering pass 1
  spec: add interactions TyreLife×Compound, LapNumber×RaceProgress,
        Compound×Stint; add target-encoding for Driver,
        (Race, Compound) with OOF discipline (per-fold encoding)
  expected lift: +30–60bp
  effort: ~1–2h (FE + OOF-safe target encoding + CV)
  risk: medium (must avoid target leakage in encoding)

Recommendation: do A first (cheap calibration anchor #2), then B.
```

## Compute budget

- Day-1 spend: ~3 min total LGBM (1m smoke-style + 2m two-anchor)
- 5-fold full-data probe ≈ 25s/fold; comfortable inside 1h GPU cap
- Subs used: 1 / 5 today; 1 / total
