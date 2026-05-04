# M6 — Dirichlet random search blend (3000 candidates, α=1)

Pool: ['baseline', 'd2a_te', 'm2_xgb', 'm3_catboost', 'm4_relstate']

## Two-anchor winners

| anchor | mode | OOF | Δ vs base |
|---|---|---:|---:|
| Strat | raw | 0.94696 | +62.1bp |
| GroupKF | rank | 0.92459 | +40.0bp |

## Best weights — Strat

raw mode: {'baseline': 0.0241, 'd2a_te': 0.0012, 'm2_xgb': 0.2737, 'm3_catboost': 0.6462, 'm4_relstate': 0.0548}

rank mode: {'baseline': 0.0241, 'd2a_te': 0.0012, 'm2_xgb': 0.2737, 'm3_catboost': 0.6462, 'm4_relstate': 0.0548}

## Best weights — GroupKF

raw mode: {'baseline': 0.0032, 'd2a_te': 0.0384, 'm2_xgb': 0.1064, 'm3_catboost': 0.2348, 'm4_relstate': 0.6173}

rank mode: {'baseline': 0.0212, 'd2a_te': 0.0121, 'm2_xgb': 0.2261, 'm3_catboost': 0.2176, 'm4_relstate': 0.523}

## Verdict

Submission file uses Strat winner (raw mode).
Compare against M5 LR meta to choose the LB candidate.
