# Day-5 Path C — Recursive GBDT with M5q_oof_proba as feature

NH11. Hypothesis: a fresh GBDT base trained on raw features +
`m5q_oof_proba` learns row-level corrections to the M5q stack that
the LR-meta can't represent (LR is linear in expanded base columns;
trees can split on the consensus prediction itself and gate other
features on its value).

## Result — strongest standalone GBDT we've built; null at 2-base stack

| Quantity | Value | Notes |
|---|---:|---|
| Recursive standalone Strat OOF | **0.94994** | +91.9bp vs baseline_two_anchor; +11.8bp vs e3_hgbc |
| ρ(recursive_test, m5q_test) | **0.99159** | inside d4 "real-LB-delta" band (>0.994 / <0.999) |
| [M5q, recursive] LR-stack Strat OOF | **0.95055** | Δ M5q anchor = **−0.2bp (NULL)** |
| Per-fold AUC | 0.94906–0.95119 (std=0.00073) | tight, no folds-anomaly |
| Wall (5-fold HGBC, CPU) | **24s** | dramatically under 30min ETA |

## What the result says

1. **Standalone is strong (+91.9bp)** — the recursive base has access
   to M5q's blended consensus, so its decision tree gets to start
   "near the answer" and refine. Best single GBDT we've ever built.
2. **2-base LR stack is null.** ρ=0.992 is high enough that LR over
   two correlated columns + their rank/logit views cannot extract
   incremental signal. Expected: with K=2 the meta has nowhere to
   route the residual rank shuffle.
3. **Diversity is real-but-small.** ρ=0.99159 sits between the
   TIE_EXPECTED ceiling (≥0.999, d4 yetirank/nb) and the d4 GBDT-meta
   "real LB delta" band (~0.995). LB delta from a full-pool re-stack
   is theoretically plausible.

## Where this leaves the recursive base

The 2-base test is the WRONG test, in retrospect. The right test is:
**add recursive to the K=14 M5q pool and rebuild the LR meta**
(K=15). M5q's LR meta has 14 bases × 3 expansion views = 42 degrees
of freedom; adding a 15th base whose ρ vs M5q is 0.99159 gives the
LR room to route corrections across the full pool that a 2-base
collapse cannot.

Compare d4 yetirank: ρ=0.666 vs M5q standalone, but TIE_EXPECTED
when added to the full M5q pool — the full pool's freedom is
already saturated by linear means. Same gate test applies here.

## Next moves

1. **K=15 stack: M5q pool + recursive.** Reuse the m5qrs LR-meta
   pattern; report Strat OOF + L1 weight on `recursive` base + Δ M5q.
   If +1bp OOF, candidate for slot 2/3 LB submit (with pre-submit-diff).
2. **GBDT-meta over K=15 pool.** d4 finding: GBDT-meta over K=14
   gave ρ=0.995/−4bp LB. With recursive added as a structural
   ρ=0.992 column, GBDT-meta may now pick up cross-base interactions
   the LR can't.
3. **Alternative recursive view: include logit + rank of M5q.** If
   K=15 is null, retry with `m5q_proba`, `m5q_logit`, `m5q_rank` as
   3 added columns. Trees are insensitive to monotone xforms but the
   logit may interact better with `lap_norm` / `Compound`.

## Held artifacts

- `scripts/artifacts/oof_d5_recursive_m5q_strat.npy`
- `scripts/artifacts/test_d5_recursive_m5q_strat.npy`
- `scripts/artifacts/d5_recursive_m5q_results.json`

End — 50 lines.
