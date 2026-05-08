# Day-2 probe #2 — DGP rule probe (2026-05-04)

Method: shallow `DecisionTreeClassifier` (max_depth=3,5,7; min_samples_leaf=200) on PitNextLap. If a meaningful fraction of rows fall into low-entropy leaves (entropy < 0.10), the data has rule-structure and trees rediscover interactions natively.

Reference: irrigation-water PM-03 §1 — closed-form 6-feature rule drove +84bp; hand-FE on top of DGP regressed -52bp.

## Results

| depth | n_leaves | AUC | rows in low-entropy leaves | frac |
|---:|---:|---:|---:|---:|
| 3 | 8 | 0.87990 | 77,233 | 0.1759 |
| 5 | 32 | 0.90554 | 77,164 | 0.1757 |
| 7 | 109 | 0.91754 | 140,430 | 0.3198 |

## Verdict

**PARTIALLY RULE-STRUCTURED.** 32.0% of rows in low-entropy leaves at depth 7. The DGP has some deterministic structure but isn't a simple rule. Trees should still capture it; hand-FE risk is moderate.

## Tree text (depth=3)

```
|--- Stint <= 1.50
|   |--- Compound <= 2.50
|   |   |--- Compound <= 1.50
|   |   |   |--- class: 0
|   |   |--- Compound >  1.50
|   |   |   |--- class: 0
|   |--- Compound >  2.50
|   |   |--- Position <= 9.50
|   |   |   |--- class: 0
|   |   |--- Position >  9.50
|   |   |   |--- class: 0
|--- Stint >  1.50
|   |--- Year <= 2023.50
|   |   |--- Year <= 2022.50
|   |   |   |--- class: 0
|   |   |--- Year >  2022.50
|   |   |   |--- class: 0
|   |--- Year >  2023.50
|   |   |--- TyreLife <= 10.50
|   |   |   |--- class: 0
|   |   |--- TyreLife >  10.50
|   |   |   |--- class: 1

```

## Top 10 leaves at depth=5

| leaf | count | target_rate | entropy |
|---:|---:|---:|---:|
| 43 | 76,037 | 0.0039 | 0.0372 |
| 13 | 73,395 | 0.0169 | 0.1236 |
| 12 | 56,273 | 0.0425 | 0.2535 |
| 58 | 52,919 | 0.7174 | 0.8590 |
| 16 | 32,356 | 0.0672 | 0.3552 |
| 39 | 22,929 | 0.6234 | 0.9556 |
| 54 | 15,708 | 0.3850 | 0.9615 |
| 59 | 14,963 | 0.5244 | 0.9983 |
| 6 | 13,185 | 0.0889 | 0.4328 |
| 5 | 12,209 | 0.0324 | 0.2065 |
