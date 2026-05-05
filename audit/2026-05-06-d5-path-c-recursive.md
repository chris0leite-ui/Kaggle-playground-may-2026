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
- `scripts/artifacts/d5_recursive_m5q_strat_results.json`

## K=15 result — recursive eaten by LR rank-lock (3rd confirmation)

`scripts/d5_recursive_stack_k15.py` ran three pool-composition
variants. All TIE-class at meta level:

| variant | K | Strat OOF | Δ M5q bp | ρ M5q test | rec L1 | gate |
|---|---:|---:|---:|---:|---:|---|
| M5_K15a (M5q + rec) | 15 | 0.95056 | −0.06 | **0.99991** | **0.841** | TIE_EXPECTED |
| M5_K15b (drop e3, +rec) | 14 | 0.95053 | −0.35 | 0.99971 | 0.256 | TIE_EXPECTED |
| M5_K15c (drop f1+f2, +rec) | 13 | 0.95045 | −1.23 | 0.99922 | 0.345 | TIE_LIKELY |

The striking fact: in M5_K15a the recursive is the **2nd-highest L1
weight in the pool** (0.841, behind only `cb_slow-wide-bag` at 1.060,
and 2× realmlp's 0.422). The LR meta clearly wants to use it. Yet
the resulting test predictions are ρ=0.99991 vs M5q — Kaggle
5-decimal quantization territory.

**Mechanistic read.** Because the recursive base IS a transformation
of (raw_features + M5q_consensus), an LR meta over the expanded
pool can re-derive M5q's contribution through the recursive's
noisier-but-richer path. Big L1 weight on a near-clone of the
consensus equals the same rank ordering, by construction. LR cannot
exploit the row-level corrections that the recursive's tree splits
encode — those corrections require a non-linear meta to combine
`recursive_proba` with `e3_hgbc_proba` × `cb_lossguide_rank` etc.

**Third confirmation of `lr-meta-rank-lock-strong-anchor`.**
Day-3 friction logged it. Day-4 yetirank (ρ=0.666 standalone /
TIE_EXPECTED stacked) + nb (ρ=0.853 / TIE_EXPECTED stacked) gave
two more confirmations. Today's recursive (ρ=0.99159 standalone /
ρ=0.99991 stacked) is the third. The pattern is invariant to base
diversity at any scale — orthogonal OR near-clone, the LR meta
collapses to M5q's rank ordering.

## Falsified

- "K=15 LR re-stack with recursive added breaks M5q ceiling" → null
- "Pool composition (drop e3 / drop f1+f2) un-locks the LR meta" → null

## Live candidate from this build

The recursive base remains a strong **pool member for a non-linear
meta**. d4 GBDT-meta over K=14 was −4bp LB (ρ=0.995); GBDT-meta over
K=15 with recursive may find the cross-base interactions LR cannot
(`recursive` correlates highly with consensus but encodes different
per-row residual structure that trees can split on).

## Next move

`scripts/d5_gbdt_meta_k15.py` — re-run the d4 GBDT-meta sweep
(LightGBM depth=3/5, HGBC) over the K=15 pool. Decision rule: if
OOF ≥ 0.95048 (matches d4 lgbm_shallow which gave −4bp LB) AND
ρ < 0.999, candidate is real and we have a slot 2 LB probe.

End — 95 lines.
