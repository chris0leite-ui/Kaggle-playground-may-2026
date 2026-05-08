# scripts/archive — historical research log

Scripts here are NOT live tooling. They reproduced results that are now
either folded into the K=27 PRIMARY pool, archived as falsified, or
superseded by newer probes. Kept for reproducibility / archaeology.

## Layout

| Subdir | What |
|--------|------|
| `m_sweeps/` | Day-3/4/5 m-tier LR-meta + GBDT-meta sweeps (m1, m3, m4, m5*, m6) |
| `by_day/` | Day-tagged research scripts d2*-d12* + falsified d13/d14/d15/d16/d18 variants + Day-2 universals (u2, u3) + early diagnostics |
| `dead_probes/` | Probes whose outputs were INVALIDATED by Rule 24 strict-OOF audit (target_reform leaky family) or otherwise superseded |
| `lr_research/` | LR-bank / LR-leverage / LR-torch — Day-17/18 expedition; promoted to skill `.claude/skills/kaggle-comp/lr-diagnostics.md` |
| `probes_d8/` | Day-8 mini-probe suite |
| `eda_deep/` | Day-13 deep-EDA scripts |

## When to consult

- Reproducing a specific calibration-ladder row → look up the script
  by day-tag prefix here.
- Verifying a "Falsified or dead" claim → the falsifying script is
  preserved.
- Otherwise: use live scripts in `scripts/` (top level).
