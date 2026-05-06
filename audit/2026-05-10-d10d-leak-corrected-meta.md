# Day-10 — Leak-corrected LR meta: gate-FAIL, but informative

> Hypothesis from d10b/c: refit the LR meta on GKF OOFs (where FM_B
> is L1=6.96 dominant) and apply those weights to GKF-averaged base
> test predictions. The leak-correction should up-weight FM and
> produce a private-robust submittable prediction.
> Falsified at the G3 gate: rare-class flip ratio is 0.001.

## What we built

`scripts/d10d_leak_corrected_meta.py`. Same K=15 pool as d10b/c
(13 GBDT/baseline + FM_A + FM_B, all under Race-only GKF). Fit
LR meta on GKF OOFs (`F_oof = expand([P_oof])`), apply coefficients
to GKF-averaged test predictions.

Compute under Race-only GroupKFold(5) on Race. ~3 min wall.

## Gate-check result

| Metric | Value | Gate | Verdict |
|---|---:|---|---|
| GKF stack OOF (CV) | 0.92764 | matches d10b | ✓ sanity |
| ρ vs PRIMARY test (d9f K=21 swap) | 0.98740 | ≥ 0.999 tie | **FAIL** |
| Rare-class flips (1% threshold): + → −  | 1751 | — | — |
| Rare-class flips: − → + | 2 | — | — |
| **Flip ratio (min/max)** | **0.001** | **G3 ≥ 0.5** | **FAIL** |
| Predicted LB (ρ-chain) | 0.95001 | ≥ 0.95031 | **FAIL (−3bp)** |

The flip asymmetry is the killer. The leak-corrected meta drops
**1751 rows out of the top-1%-pit-probability** band that PRIMARY
flagged, while only adding 2 rows. That's not a re-weighting of
roughly-equal predictions — it's a **wholesale smoothing-away of
the rare-pit signal** that GBDT bases produce.

## Why the asymmetry

PRIMARY's Strat-meta gives heavy weight to GBDT bases (L1=0.5-0.9
on b_lapsuntilpit, e5_optuna_lgbm, etc.). Those bases capture
**row-specific extremes** — e.g. "this exact row has TyreLife=42
on a Soft tyre at lap 58 of an 80-lap race; pit_rate is 0.85".
That signal is *real* (the test set has the same DGP per U3
i.i.d.), and only ~half of it is within-group leakage.

The leak-corrected meta puts FM_B at L1=6.96. FM_B is built on
{Race, Year, RaceProgress_q5, Position_q5} — coarse 5-quintile
bins. Its predictions are **smoothed** along those bins; it can't
spike on a specific row without a corresponding spike across the
quintile. So when the meta routes through FM, the "spike rows"
get pulled toward their quintile mean.

This means the d10c Strat→GKF lift (+0.87bp → +2.01bp = 2.3×) was
**partly real (FM-class lift transfers under leak-blocking) and
partly artifact (GBDT bases lose value under GKF for reasons that
*don't* transfer to test, namely the inability to hit specific
rows in unseen Race groups)**.

## What remains valid from d10b/c

The d10b/c finding is still load-bearing for the **PRIMARY-stays
direction**:

1. FM-class lift exists under both Strat and GKF (the bp is
   smaller under Strat but the *direction* is positive in both).
2. PRIMARY's +2bp public-LB lift over the FM-less d6_k18 baseline
   is still real (FM-class is doing genuine work in the meta).
3. The HEDGE selection still applies: d6_k18 is leakage-inflated
   like all GBDTs, but the test set is i.i.d., so the leakage is
   "fair" in the sense that it doesn't mis-rank rows that aren't
   in the test set's distribution.

What's *not* valid: using GKF stack lift as a one-shot oracle for
private-LB ranking. The GKF OOFs cannot see test-row-specific
extremes, so they systematically under-credit GBDT bases on the
rare-pit class.

## What about a blend?

Mixing leak-corrected (d10d) with PRIMARY 70/30 would preserve
70% of the rare-pit signals but partially up-weight FM. That's a
heuristic compromise. Two reasons not to pursue:

1. The blend is just LR-meta-on-LR-metas. PRIMARY *already* has
   the FM weight that LR considers optimal *under Strat*. A
   convex blend cannot lift unless it captures information neither
   parent has — and we just showed d10d under-credits rare-pit.
2. We have 5/10 today and a real backlog (Path B Bayesian meta,
   Path C adversarial weights). A "70/30 blend" submit is a
   gut-check spend on a known-low-EV configuration.

## Slot-worthy follow-ups

1. **Bayesian hierarchical stacker (Path B, 2-3h CPU).** Per-segment
   partial-pooling weights mean that for *common* segments (where
   leakage piggybacking from GBDTs is large), GBDT weight stays
   high; for *rare* segments (where Strat OOF for GBDTs is mostly
   leakage), FM weight rises. This is the correct way to combine
   d10d's insight with PRIMARY's row-specific signal.
2. **Adversarial validation instance weights (Path C, 30 min CPU).**
   Re-weight train rows by p(test|x) before fitting the LR meta.
   Cheap, no harm, EV +0-2bp.

## Pointers

- `scripts/d10d_leak_corrected_meta.py` — builder.
- `submissions/submission_d10d_leak_corrected_meta.csv` — HELD
  (gate FAIL).
- `scripts/artifacts/{oof,test}_d10d_leak_corrected_meta_strat.npy`.
- `scripts/artifacts/d10d_leak_corrected_meta.json`.
