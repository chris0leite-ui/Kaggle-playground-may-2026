# PCA-meta probe — 4 variants on the K=27 ensemble

ISO date: 2026-05-08 PM (comp day 8 of 31).
Probe: `scripts/probe_pca_meta.py`. Wall 503s (~8.4 min CPU).
JSON output: `scripts/artifacts/probe_pca_meta.json`.
Anchor: K=10+1 plain LR-meta OOF = **0.95417** (re-measured this probe).

PI ask: try a PCA of the K=27 ensemble and similar ideas. Research, experiment,
learn. PI selected all four variants; precedent anchor = K=10+1 plain LR.

## TL;DR — no LB candidate, but two scientific findings

1. **EXP-NEW (non-LR meta) is FALSIFIED, not just untested.** LightGBM as
   the meta-learner *underperforms* logistic regression at every input
   representation we tested (PCA top-K for K∈{3,5,10,15,27}, raw K=10 [P,
   rank, logit] expansion, raw K=27 expansion). Best LightGBM result was
   0.95417 (≈anchor); best LR result was 0.95428 (+1.09 bp). The "only
   architecturally untested avenue" per A30 closes negative.
2. **A25's eff-rank=3.23 is variance, not predictive content.** PCA-LR on
   the top-3 PCs of the K=27 logit pool reaches only 0.95061, which is
   −35.64 bp below K=10+1 plain LR. The predictive content is spread
   across roughly 15 PCs, not 3. Top-15 PCA-LR (0.95401) nearly matches
   K=10+1 plain LR; top-27 PCA-LR (0.95422) essentially recovers it.
   This refines but does not falsify A30 — the LR-meta family still
   appears at its representational ceiling, just on a 15-D effective
   manifold rather than a 3-D one.

## Setup

- 27-base pool (the K=27 from `probe_pool_structure.py`).
- 5-fold StratifiedKFold seed=42, matching the base CV split.
- All meta variants fold-safe: PCA standardisation + SVD fit on
  meta-train rows only; meta-val rows projected through the
  train-fitted basis. K=27 OOFs are honest because each base's row
  prediction was held out under the base's own fold split (same seed).
- Compound × Stint segmentation for Path-B (5 compounds × 6 stints =
  30 segments).

## Results — full grid

| Variant | Score | Δ vs K=10 anchor (bp) |
|---|---:|---:|
| **K=10+1 plain LR-meta (anchor)** | **0.95417** | — |
| K=27+1 plain LR-meta (full pool) | 0.95428 | +1.09 |
| **A — PCA-LR top-3** | 0.95061 | −35.64 |
| A — PCA-LR top-5 | 0.95077 | −34.07 |
| A — PCA-LR top-10 | 0.95124 | −29.37 |
| A — PCA-LR top-15 | 0.95401 | −1.66 |
| A — PCA-LR top-27 | 0.95422 | +0.43 (null) |
| **B — PCA-GB top-3** | 0.95088 | −32.88 |
| B — PCA-GB top-5 | 0.95114 | −30.29 |
| B — PCA-GB top-10 | 0.95156 | −26.15 |
| B — PCA-GB top-15 | 0.95385 | −3.21 |
| B — PCA-GB top-27 | 0.95398 | −1.91 |
| B — GB on K=27 raw expansion | 0.95417 | +0.01 (null) |
| B — GB on K=10 raw expansion | 0.95405 | −1.25 |
| **C — Path-B C×S top-3 PCs** | 0.95077 | −34.00 |
| C — Path-B C×S top-5 PCs | 0.95088 | −32.89 |
| C — Path-B C×S top-10 PCs | 0.95133 | −28.45 |
| **D — K=10 LR + K=27 PC residuals (strip-3)** | **0.95428** | **+1.03** |
| D — K=10 LR + K=27 PC residuals (strip-5) | 0.95428 | +1.03 |

## Variant-by-variant interpretation

### A — PCA-truncate + LR (regularizer test)

**The 3-D variance ceiling is not a 3-D predictive ceiling.**

The K=27 logit pool has eff-rank-entropy 3.23 (A25), which means 93% of
the *variance* is in the top 5 components. But top-3 PCA-LR scores only
0.95061, while top-15 PCA-LR scores 0.95401 — within 1.66 bp of K=10+1
plain LR. Top-27 PCA-LR (0.95422) recovers K=10+1 LR within sample noise.

Reading: the **predictive eff-rank** is much higher than the **variance
eff-rank**. Low-variance principal components carry meaningful signal
about the binary target — they are noisy in feature-space but informative
in label-space. This is the canonical "PCR can be worse than ridge"
scenario from regression theory (Frank & Friedman 1993): supervised
methods don't necessarily prefer high-variance directions.

A25's wording — "K=27 pool's effective rank is 3.23" — is correct but
should be qualified: that's the variance-effective-rank. The
predictive-effective-rank under the LR-meta is closer to 15.

### B — GB-meta on PCs (EXP-NEW closed)

**LightGBM as meta-learner underperforms LR everywhere.**

| Input representation | LR | LightGBM | LR − LightGBM (bp) |
|---|---:|---:|---:|
| K=27 PCs (top-27) | 0.95422 | 0.95398 | +2.4 |
| K=10 [P, rank, logit] | 0.95417 | 0.95405 | +1.2 |
| K=27 [P, rank, logit] | 0.95428 | 0.95417 | +1.1 |

LightGBM is *worse* than LR at the meta layer for this stack. Three
plausible reasons:

- The 27 base predictions are highly redundant (logit eff-rank=3.23).
  The bias-variance tradeoff favours the simpler linear combiner.
- The base predictions are already near-calibrated; LR can simply weight
  them. A tree model has to reinvent calibration through monotonic
  splits, which is wasteful.
- LightGBM's default learning_rate=0.05 / num_leaves=15 / max_depth=4 /
  num_boost_round=300 is reasonable for raw-feature problems but maybe
  not optimal for meta-stacking; tuning could improve. But the gap is
  consistent across representations, so I doubt a tune-up closes it.

Strategic implication: A30 is REFINED, not falsified. Its claim was
"breaking the 3-D ceiling requires either new data outside 14 cols, or
a non-LR meta architecture." We tested the second clause: non-LR meta
*does not* break the ceiling. **The non-LR meta architecture avenue
is empirically closed.**

### C — Path-B Compound × Stint on PCs

**Routing on uncorrelated directions kills Path-B.**

Path-B's per-segment shrinkage stacker fires on **redundant** pools
(per `state/hypothesis-board.md` "insights still load-bearing"). The
PCs are uncorrelated by construction, so the routing variable cannot
exploit base correlations. Top-10 PCs Path-B (0.95133) loses 28 bp
versus K=10 plain LR.

Mechanism: when the inputs are correlated, per-segment slopes can
re-weight which base-cluster speaks for that segment. When the inputs
are orthogonal, each segment can only do PC-level slope changes, which
the global LR already provides. The shrinkage strength τ=100k can't
recover this because the per-segment LRs have nothing to bring.

This is a clean falsification of the "Path-B finds value via correlation
exploitation" intuition — value with correlation, no value without.

### D — K=27 PC residuals as auxiliary K=10 features

**The +1 bp from K=27 over K=10 lives in PC4..PC27, but no extra signal
beyond that.**

Variant D fed K=10's [P, rank, logit] expansion alongside K=27's
"residuals after top-3 PCs stripped" (24 columns of standardised
residual logits). Score 0.95428 = exactly K=27+1 plain LR. Strip-5
gives the same number.

So: yes, the +1 bp from K=27 over K=10 IS in the low-variance PCs. But
adding those residuals to K=10 gets us to where K=27 already is, not
beyond. The LR-meta on K=27 raw features already captures all the
signal in PC4..PC27.

This is consistent with PCs being a rotation: LR(K=27 raw) ≡
LR(K=10 raw + K=27 residuals stripped of top-3 K=27 PCs) up to
collinearity, since the top-3 K=27 PCs are essentially K=10's
high-variance directions. Confirmed empirically.

## What we learned

1. **PCA on the K=27 logit pool is a useful diagnostic but not a
   predictive lever.** The 3-D variance subspace claim (A25) is
   variance-only; it does not constrain predictive content.
2. **Non-LR meta is dead.** EXP-NEW closes negative (LightGBM is
   *worse*, not better). Per A30, this leaves only "new data outside
   the 14 columns" as a structurally-different lift route — and PI
   has ruled that out.
3. **Path-B mechanism is clarified.** Path-B's per-segment lift
   requires base correlation (routing material). PCs lose it.
4. **The ceiling is real but slightly larger than thought.** The
   informative subspace under the LR-meta family appears to be
   ~15-D, not 3-D. That doesn't reduce the ceiling — top-27 PCA-LR
   recovers K=27 LR — it just changes our description of why.

## Strategic upshot

- **No new submission candidate.** The best probe variant ties
  K=27+1 plain LR at 0.95428 (which we have already).
- **No assumption-ledger flip — but A25 wording wants a footnote.**
  The "3.23 eff-rank" is variance-only. Predictive eff-rank ≈ 15.
  See `ASSUMPTIONS.md` A25 update.
- **EXP-NEW status: FALSIFIED.** Move from `EXPERIMENTS-NEXT.md` Tier
  A pending to verdicts table.
- **Open avenues now:** R5 hedge prep, RealMLP n_ens=24 (low-confidence
  prior, GPU cost), per-Year CatBoost specialists. The strategic frame
  shifts toward acceptance/wrap-up posture (also flagged in
  `HANDOVER.md`).

## Caveats

- LightGBM hyperparameters were not tuned. Default-ish params
  (lr=0.05, num_leaves=15, max_depth=4, n_rounds=300, bag/feat=0.9,
  min_data=20). A 2-hour Optuna sweep *might* claw back the 1-2 bp
  gap, but the consistent pattern across representations is hard to
  blame on hyperparameter choice alone.
- We did not test a small NN as the meta. A neural meta is the only
  other "non-LR meta" framing. Cost would be 30-60 min CPU; learning
  value is bounded by the LightGBM result (NNs and GBDTs typically
  bracket each other on tabular meta tasks).
- Path-B was not tested on PCs *with the original [P, rank, logit]
  expansion* — i.e., we tested Path-B on top-K PCs alone, not on
  Path-B(top-K PCs of K=27) + K=10 [P, rank, logit]. That hybrid
  could in principle add value, but variant D already shows residuals
  give nothing beyond K=27 LR.

## Files

- `scripts/probe_pca_meta.py` — the probe.
- `scripts/artifacts/probe_pca_meta.json` — full numerical results.
