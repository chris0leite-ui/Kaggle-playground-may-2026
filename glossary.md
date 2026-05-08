# Glossary

Plain-English definitions for every abbreviation, technical term, and
short-code used in this project's documentation.

This file is for reference between agents and for the PI when reading
dense audit notes. **In live chat, agents must not require the PI to
look anything up here** — speak plainly first, define inline if needed.

## Scoring and validation

- **AUC** — Area Under the Receiver-Operating Characteristic curve. The
  competition metric. Ranges 0.5 (random) to 1.0 (perfect).
- **OOF** — Out-of-fold. A prediction made by a cross-validation model
  on rows it was not trained on. The OOF AUC is the in-house quality
  signal.
- **LB** — Public leaderboard score on Kaggle. The score we actually
  care about, computed on roughly 20% of the held-out test rows.
- **Private LB** — The score computed on the remaining ~80%, revealed
  only after the competition ends.
- **bp** — Basis point. 1 bp = 0.0001 AUC. The natural unit of "how
  much did we move."
- **Stratified K-Fold (Strat / SKF)** — Split rows into 5 folds keeping
  each fold's positive-rate roughly equal. The default cross-validation
  scheme on this comp.
- **GroupKFold (GKF)** — Split rows so that all rows of one group (e.g.,
  one Race) are in the same fold. Used as a leakage-blocking diagnostic.
- **Holdout** — A separate 80/20 split (independent seed). Used to
  audit OOF figures: if `holdout_AUC ≪ OOF_AUC` by >10 bp, leakage is
  present.
- **AV-AUC** — Adversarial-validation AUC. Train a binary classifier to
  tell train rows from test rows. AV-AUC ≈ 0.5 means train and test
  are indistinguishable; AV-AUC > 0.55 means they differ.
- **ρ (rho)** — Spearman rank correlation between two prediction sets.
  ρ = 1.0 means identical rankings; ρ < 0.95 means meaningfully
  different; ρ > 0.999 means the leaderboard will tie within Kaggle's
  display precision.
- **G3 flip ratio** — A pre-submit safety check. Counts how many top-k
  rows the new model promotes vs demotes compared to the previous
  PRIMARY. Below 0.5 = asymmetric (often a bad sign).

## Modelling

- **GBDT** — Gradient-Boosted Decision Tree. The class of model that
  includes LightGBM, XGBoost, and CatBoost.
- **LightGBM (LGBM)** — Microsoft's GBDT implementation. Default
  workhorse here.
- **XGBoost (XGB)** — Tianqi Chen's GBDT.
- **CatBoost (CB)** — Yandex's GBDT. Native categorical handling
  via "CTR" (categorical-to-real) features.
- **HGBC** — sklearn's HistGradientBoostingClassifier.
- **RealMLP** — A multi-layer-perceptron variant from the PyTabKit
  library tuned for tabular data.
- **DAE** — Denoising autoencoder. Trained to reconstruct corrupted
  inputs; the bottleneck layer is a learned representation.
- **TabPFN** — A pretrained transformer model for tabular tasks.
- **FM** — Factorization machine. A linear model with a low-rank bilinear
  interaction term; good when GBDTs can't reach the right interactions.
- **FFM** — Field-aware factorization machine.
- **TE** — Target encoding. Replace a categorical level with the mean
  target of training rows at that level. Must be done out-of-fold.
- **CV TE** — Cross-validated target encoding (the safe version).
- **NN** — Neural network.
- **GRU** — Gated recurrent unit. A type of recurrent neural network.

## Feature engineering and data terms

- **FE** — Feature engineering.
- **Yekenot recipe** — A specific feature-engineering pipeline from a
  public Kaggle notebook. Includes floor-categorical features
  (np.floor + factorize on numeric ratios), value-counts encoding on
  categoricals + numeric combos, KBinsDiscretizer with 200 / 7 bins
  on RaceProgress / LapTime, and stratified concatenation of original-data
  augmentation (4-of-5 per fold).
- **Rozen recipe** — Another public Kaggle notebook's pipeline. ~50
  hand-engineered features + 6 cross-validated target encodings + 9
  raw categoricals = 65 features total. Origin of "kitchen-sink FE."
- **Original / orig data** — The aadigupta1601 Kaggle dataset, the
  original (pre-synthesised) F1 strategy data the host used to derive
  the synthetic competition data.
- **Synth / synthetic** — The competition's training/test rows. The
  host used a CTGAN-class generative model to create them.
- **DGP** — Data-generating process. The mechanism that produces the
  data; here, the synthesiser.
- **Normalized_TyreLife (NTL)** — Original-data column the host removed
  from synth. Forbidden to reintroduce.
- **PitNextLap** — The target column (will the driver pit on the next
  lap?). Float in {0.0, 1.0}.

## Stacking

- **Stack / ensemble** — A combination of multiple models' predictions.
- **Pool** — The set of base models in the stack.
- **K=21, K=22, K=27** — Pool size (number of base models).
- **PRIMARY** — The current best submission (or its underlying stack).
- **HEDGE** — A held-back submission for the final-window override probe.
- **Base / base model** — A first-level model whose OOF predictions
  feed into the meta-stacker.
- **Meta / meta-stacker / meta-learner** — The second-level model that
  combines base predictions into a final score.
- **LR-meta** — Logistic regression as the meta-stacker; the canonical
  one for this comp.
- **Hier-meta / per-segment shrinkage stacker / Path B** — Three names
  for the same trick: fit a separate logistic regression per segment
  of the data (e.g., per Compound × Stint), each shrunken toward a
  global fit. Best lift came from this on Days 13-15.
- **τ (tau)** — Shrinkage strength in Path B. Larger τ = more pull
  toward the global fit. τ ∈ {5k, 20k, 100k, 500k} are the values
  swept in practice.
- **Path B amp / amplification** — The phenomenon where +1 bp OOF gain
  from a Path-B reconfiguration produced +6 to +12 bp on the LB. Mostly
  observed on the 21-base pool; mostly absent at K=27.
- **Rank-lock** — The pattern where adding a new base, however diverse,
  doesn't change the meta-stacker's test rankings. Documented seven
  times on this comp.
- **Effective rank** — A measure of how much new information each base
  in the pool adds. K=10 effective rank ≈ 2 means 10 LR variants
  carry only 2 dimensions of net signal.

## Process / framework

- **BOTE** — Back-of-the-envelope. The pre-experiment lift prediction
  the harness produces (`scripts/probe.py bote NAME`).
- **Gate** — The post-experiment uniform structured report
  (`scripts/probe.py gate NAME`); reports OOF Δ, ρ vs PRIMARY, predicted
  LB Δ, flip ratio, verdict.
- **Q6** — The metric-alignment question. "Does the training objective
  match the row-level AUC metric?" Pairwise-rank or group-rank objectives
  trigger a verdict downgrade.
- **PI** — Principal investigator. The non-coding human reviewer who
  ratifies plans and submissions.
- **R5 / R7** — Final-window override-mechanism rules from the prior-comp
  postmortem (see `## Defaults from prior comp` in CLAUDE.md).
- **Min-meta** — The K=K_pool+1 stack-add gate
  (`scripts/probe_min_meta.py`).
- **Pre-submit-diff** — Mandatory check before any submit
  (`scripts/pre_submit_diff.py`); aborts if Spearman > 0.999 vs the
  previous submission.

## Experiment short-codes you may encounter in older notes

The team used letter-number codes within and between sessions. These are
load-bearing for back-references; new notes should use descriptive names.

- **m5q, m5h, m5g, m5b, m5c, m5d** — Day-3 multi-rule pool variants.
- **e1, e2, e3, e4, e5** — Day-3 alternative-base probes.
- **a, b, f1, f2** — Day-3 hand-coded rule variants.
- **d2a** — Day-2 target-encoding probe.
- **d3a, d3b** — Day-3 unified-TE / sequence-FE probes.
- **d4** — Day-4 alternative-meta probes.
- **d5** — Day-5 recursive-GBDT / pseudo-label probes.
- **d6** — Day-6 multi-rule-residual stack (had been PRIMARY).
- **d9, d9b ... d9i** — Day-9 hashing-LR / FM-class sweep.
- **d10, d10b, d10c, d10d** — Day-10 GroupKFold rebuild.
- **d12** — Day-12 master synthesis (4 falsified, 1 structural finding).
- **d13, d13a, d13b, d13c, d13d, d13e** — Day-13 Path-B exploration.
- **d14** — Day-14 cohort sweep / DGP residuals.
- **d15** — Day-15 four-branch deep dive (DAE became PRIMARY).
- **d15b** — Day-15 DAE-only PRIMARY at LB 0.95059.
- **d15c, d15d** — Day-15 ExtraTrees / kNN-LightGBM hedges.
- **d16** — Day-16 virgin-axes round.
- **d17, d17_h1d, d17_K23, d17_K24** — Day-17 yekenot transfer + Path-B
  variants.
- **d18** — Day-18 reverse-engineer-data-generation arc.
- **d19** — Day-19 overnight (4 axes closed).
- **v3, v4** — CatBoost research-recipe variants. v4 = yekenot transfer.
- **h1d** — RealMLP yekenot full-recipe replication.
- **K=21, K=22, K=23, K=24, K=27** — Pool sizes at successive PRIMARY
  configurations.
- **E1-E5, F1-F6, G/H/I/J, A1, A2, B1, B2, C1, D1** — Day-18/19 axis-and-
  probe codes from the reverse-engineer-data-generation arc and the
  Day-19 overnight closure round.
- **FM_A, FM_B** — Two FM variants from the d9f partition.
- **Probe 1-5** — Day-17 PM diagnostic probes.

## Calibration ladder columns

When you read `state/calibration-ladder.md` or audit notes:

- **Strat OOF** — Stratified K-fold cross-validation AUC.
- **GKF OOF** — GroupKFold cross-validation AUC. Different number; used
  for leak-blocking diagnostics.
- **LB** — Public leaderboard at submission time.
- **Notes** — Free-text; often contains friction tags or probe codes.

## Friction tags

Friction notes are categorised by `tag: short-name`. The current
load-bearing tags are:

- `target-construction-layer-leakage` — Rule 24 origin.
- `transductive-features-need-AV-check` — Rule 25 origin.
- `path-b-amp-only-fires-on-meta-arch-not-base-add` — meta-stacker
  saturation pattern.
- `path-b-amp-needs-orthogonal-signal-not-meta-derivatives` — meta-
  derivative additions don't amplify.
- `lr-meta-rank-lock-strong-anchor` — 7-times-confirmed; adding diverse
  bases doesn't move the stacker output.
- `pool-saturation-v4h1d-absorbs-dgp-class` — once yekenot recipe is in
  the stack, DGP-class probes get absorbed.
- `pathb-amp-dead-when-pool-already-routes-segmentation-variable` — if a
  base in the pool already routes by the segmentation variable, Path-B
  segmentation on that variable is null.
- `lr-meta-rank-lock-strong-anchor` (7×) — rank-lock saturation.
- `lr-eff-rank-bounded-at-2-by-pipeline-not-base-class` — LR-bank
  diversity ceiling.
- `meta-arch-redesign-family-empirically-exhausted-on-k27-pool` — 9
  variants tested.

For the full list, see `audit/friction-archive.md`.
