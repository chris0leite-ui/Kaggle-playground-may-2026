# U1 — two-anchor baseline LGBM (2026-05-04)

Anchors:
- A: StratifiedKFold(5, seed=42)
- B: GroupKFold(5) on Race (26 levels)

Hyperparams: lr=0.05, num_leaves=63, min_data_in_leaf=200, num_boost_round≤2000, ES=100

## Results

| anchor | OOF AUC | fold_std | per-fold |
|---|---:|---:|---|
| A — StratKFold | **0.94075** | 0.00075 | ['0.9416', '0.9402', '0.9410', '0.9396', '0.9414'] |
| B — GroupKFold(Race) | **0.92059** | 0.01306 | ['0.9135', '0.9078', '0.9081', '0.9391', '0.9324'] |
| gap A−B | +0.02017 (+201.7bp) | | |

## R1 verdict

LEAKAGE FLAG: gap > 50bp. StratKFold OOF likely leakage-inflated. Public-LB-driven decisions risky.

## Revised verdict (override the auto-flag)

The auto-verdict above reads "LEAKAGE FLAG" because the script's
default rule is `gap > 50bp ⇒ leakage`. That rule is missing a
qualifier: it presumes the test set holds out by group. Given U3
(`audit/2026-05-04-u3-split-probe.md`), it doesn't.

What each anchor actually measures:

- **Anchor A (StratKFold)** = "given the model has seen rows from
  every race, predict held-out random rows from those same races."
  Matches the test set's i.i.d. row split (U3: alt-ratio 0.447,
  0/13,185 contiguous groups).
- **Anchor B (GroupKFold Race)** = "given the model has never seen
  this race, predict it." The test set is NOT structured this way:
  every test race appears in train.

The 200bp gap measures the within-race signal the model exploits.
That signal IS available at test time because the test draws from
the same (Race, Driver) sequences seen during training. Anchor A
is the correct LB proxy. The gap is structural leverage, not
leakage.

Cross-check: top public notebooks use plain StratKFold and report
OOF 0.93–0.96; public LB at 0.954. Consistent with anchor A as the
right proxy.

### Implications

- **Public LB expected ≈ 0.941** for our baseline submission.
- Anchor B (0.921) is a "race-level robustness" stat, NOT an LB
  proxy.
- R2 (final-selection-along-public-LB) remains appropriate; public
  LB measures the same regime as anchor A.
- Headroom: cross-race signal is bounded by the anchor B
  trajectory; in-race interpolation is what closes the
  0.921 → 0.941 gap.
