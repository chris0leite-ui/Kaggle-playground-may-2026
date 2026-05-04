# ζ — Deep CatBoost single-fold probe (2026-05-04)

Per new 1-fold-actual-within-1h cap rule.

## Result

- fold-0 AUC: **0.94992** (Δ baseline +91.7bp)
- best_iter: 1999/2000 (hit cap)
- fit wall: 340.9s (5.68 min)
- 5-fold both-anchor projection: 3409s (56.8 min)
- single-fold within 1h cap: YES

## Decision

If fold-0 AUC > 0.94900 (E3 HGBC ballpark), pursue 5-fold both-anchor.
Otherwise skip; not worth the compute.
