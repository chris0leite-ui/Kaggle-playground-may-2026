# Where we are right now

Updated whenever a new submission lands or the strategic picture shifts.
For history, read `audit/postmortems/` and `audit/research/`.

## Best submission so far

**Score: 0.95368 on the public leaderboard.** Rank 98 of 893 = top 11%.

**What it is:** a stack of 27 base models combined with a per-segment
shrinkage stacker. The 27 bases are:

- 21 models from the original pool (gradient-boosted trees of various
  flavours, factorisation machines, rule-based residual learners, target
  encoders).
- A CatBoost model trained with a published Kaggle recipe ("yekenot
  transfer" — floor-categorical features, count encoding, KBins-discretised
  numerics, stratified concatenation of original-data augmentation).
- A RealMLP neural network trained with the same yekenot recipe.
- A LightGBM model trained on the original (pre-synth) Kaggle dataset
  using continuous-only features.
- A LightGBM model whose features are the per-step log-likelihood of a
  causal chain decomposition of the synthetic data-generating process.
- A k-nearest-neighbours preimage join: each row's distance to its
  nearest neighbours in the original dataset, computed per Compound on
  the seven feature columns the synthesiser left least distorted.
- A constraint-violation feature set: ten physical constraints (e.g.,
  TyreLife should grow monotonically within a stint), with violation
  rates aggregated per row's group.

**The stacker** is a per-segment partial-pooling logistic regression.
Segments are (Compound × Stint number). Each segment's logistic
regression is shrunk toward a global fit with strength τ = 100,000.

**Holdout-honest:** all label-derived features are refit per
cross-validation fold using only training rows.

## Leaderboard ladder (from this team's submissions)

| Date | Score | What changed |
|---|---:|---|
| Day 18 PM | 0.95368 | Added six DGP-class bases; same per-segment stacker. **PRIMARY.** |
| Day 17 PM | 0.95354 | First submission with CatBoost-yekenot + RealMLP-yekenot bases. |
| Day 17 mid | 0.95345 | Earlier configuration of the same idea. Crossed top-5% threshold. |
| Day 17 AM | 0.95149 | Added two original-data and chain-decomposition bases. |
| Day 16 | 0.95089 | First clean per-segment stacker on continuous-only features. |
| Day 15 | 0.95059 | Added a denoising-autoencoder base. |
| (earlier) | 0.95049 | Per-segment stacker first fired (Compound × Stint, τ=20k). |
| Day 1 | 0.94113 | Two-anchor baseline (Stratified + GroupKFold). |

## Submissions

- **Used: 39 of 270.** Plenty of slots left. Per Rule 12, spend them.
- **Today: 0 used.**
- **Plateau: 0 days.** Day 18 PM pushed the leaderboard +1.4 bp.

## Distance to top-5%

- Top-5% boundary: 0.95405. Gap: **−3.7 basis points.**
- Leader: 0.95476 (MILANFX). Gap: −10.8 basis points.

## What axes have been tried — high-level

For the named-experiment-by-experiment ledger, see
`state/mechanism-ledger.md`.

- **Stacking pool growth (A axis):** 7 separate confirmations that
  adding orthogonal new bases doesn't move the meta-stacker rankings
  on test once the stack is saturated. The escape was a feature
  recipe transfer (yekenot), not a new base.
- **Anchor swap (B axis):** XGBoost trained with the yekenot recipe is
  redundant with CatBoost-yekenot (correlation 0.987). RealMLP with 24
  ensembles untested.
- **Meta-architecture redesign (C axis): closed.** Nine variants tested
  across Days 14-19. Compound × Stint with plain shrinkage τ=100,000 is
  the local optimum.
- **External data (D axis): closed.** debashish historical priors
  closed by pre-flight (PI agreed with the harness's null prediction);
  FastF1 capped at 1.4% match rate (synthetic driver codes); aadigupta
  original data already in the stack.
- **Sequence-level fingerprinting (A1, untouched):** every row in every
  base is treated as independent of its stintmates. The synthesiser
  almost certainly broke within-stint sequence coherence. This is the
  one structurally-orthogonal axis still available.

## Held submissions (do not submit)

Audit `audit/2026-05-06-target-reform-leakage-audit.md`. All
target-reformulation candidates collapse 88-100% under strict
fold-safe re-runs. The held files based on them must not be submitted:

- Anything named `path_b_K22_invlaps_*`
- Anything named `path_b_K23_dae_invlaps_*`
- Anything named `path_b_K25_megapool_*`
- `path_b_multilevel_τ_*` (a separate null family)

## Held submissions (safe — hedge candidates)

These don't beat PRIMARY on the leaderboard but are safe:

- `d15b_path_b_K22_dae_only_tau{20k,100k}` — the Day-15 PRIMARY and a
  close runner-up.
- `path_b_K22_d12meta_tau100000` — landed at 0.95045 (eligible for the
  R7 final-window hedge).
- `d15c` (Extra Trees), `d15d` (LightGBM-on-kNN-features) — R5 hedge only.
