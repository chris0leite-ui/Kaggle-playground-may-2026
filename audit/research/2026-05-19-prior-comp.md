# 2026-05-19 — Prior-Comp Research (Research-loop, plateau-break)

Context: PRIMARY R15 LB 0.95397, top-5% at 0.95405, leader 0.95476.
Plateau on 4-null day. Strategy-critic identified 120 bp gap on
REAL-F1 drivers (vs synthetic D###) that internal-feature
manipulation cannot extract. Question: how have prior Kaggle
playground winners injected genuinely-external information into a
synthetic-augmented tabular comp where part of the dataset has a
real-world counterpart with rich history?

## Writeups surveyed (3)

### 1. Playground Series S5E3 — Binary Prediction with a Rainfall Dataset (Mar 2025)

**Structural match strength: HIGHEST.**

- Metric: ROC-AUC. Binary target. Class-imbalanced (rainy days minority).
- Data structure: synthetic-generated from public Australian
  meteorological "Rain in Australia" weather dataset. Public original
  dataset exists and is far larger / longer-history than the synthetic
  train.
- Chris Deotte 2nd place: "GBDT + NN + SVR + Original Data."
  Title alone confirms the mechanism family; full writeup gated.
- Approach (per multiple secondary sources, NVIDIA blog +
  Medium playbook synthesis): Deotte's "original data" trick has
  two flavours that have appeared across his recent playground
  writeups:
    (a) ROW-CONCAT: append the original dataset's rows to the
        synthetic training set with their TRUE labels, used as
        extra training data. This is data augmentation, not
        feature engineering. Works when original schema matches
        synthetic schema closely.
    (b) FEATURE-GROUP TARGET-ENCODING-FROM-ORIGINAL: for each
        target-encoded categorical (or category-combination) used
        in synthetic train, look up that same key in the ORIGINAL
        dataset and use the original's per-key mean target as a
        smoothed prior. "Manufacturer suggested retail" framing
        from the S5E2 Backpacks 1st-place NVIDIA writeup
        (`orig_price = mean target per weight-capacity group from
        original`, merged back as a feature).
- Leakage discipline: Deotte uses NESTED CV when target-encoding
  any column with the target — the encoding statistics for fold k
  are computed from out-of-fold rows only. When concat-as-rows is
  used, the original rows are added to the train fold (NOT the
  validation fold) and dedup is checked against test.
- LB lift: not directly quantified in the title; "GBDT + NN + SVR
  + Original Data" framing implies original-data is one of four
  pillars. By analogy to S5E2 ("orig_price" was reported as
  meaningful enough to be one of the top 500 selected features
  out of thousands) and the general claim "when the original
  dataset is public, blending it with synthetic Playground data
  often gives the final boost," typical lift is in the 30-150 bp
  range when the original dataset has TRUE labels and the
  synthetic version is a partial / noisier reflection of it.

### 2. Playground Series S5E2 — Backpack Price Prediction (Feb 2025)

**Structural match strength: HIGH (regression not AUC, but
mechanism transfers directly).**

- 1st place (NVIDIA blog, cuDF-pandas grandmaster-pro-tip).
- Synthetic data was generated from a public backpack-price
  original dataset.
- Single derived feature `orig_price = mean target per
  weight-capacity group from the ORIGINAL dataset`, merged onto
  the synthetic train+test via the weight-capacity key. Framing:
  treat original as "manufacturer suggested retail," synthetic as
  "individual stores' prices."
- Combined with thousands of other features (groupby aggs,
  digit extraction, categorical-combinations). 500 features
  selected from the larger pool.
- Leakage discipline: nested k-fold cross-validation whenever
  target encoding was used.
- LB lift: not isolated per feature, but original-derived
  features made the top-500 cut from a many-thousand pool ⇒
  non-trivial signal.

### 3. Playground Series S4E1 — Bank Customer Churn (Jan 2024)

**Structural match strength: MODERATE (binary AUC, high-cardinality
categorical SUR-NAME analogous to our Driver field).**

- Metric: ROC-AUC. Binary target. Class-imbalanced (~21% positive,
  near-identical to our 19.9% prior).
- Synthetic generated from public "Bank Customer Churn Prediction"
  dataset.
- Top solutions (community consensus across notebooks / GitHub
  repos referenced; specific top-3 writeups gated):
    - **Surname target encoding**: Surname is high-cardinality
      and the standard advice "drop CustomerId & Surname, they
      have no signal" was WRONG. Top finishers smoothed-TE the
      Surname column (sometimes also CustomerId) using NESTED
      K-FOLD encoding (alpha smoothing, hyperparam tuned).
    - Some top finishers also concatenated original-dataset
      rows to training set.
    - Ensemble breadth (8+ models with hill climbing for the
      #2 solution).
- Leakage discipline: raw target encoding leaks; the fix
  documented in multiple writeups is k-fold-out-of-fold mean,
  with smoothing alpha (typical alpha ∈ [5, 100], tuned).
- LB lift specifically from Surname TE: not directly quantified
  in surfaced material, but the field shift from "drop" to "TE-it"
  was a top-5% discriminator in this comp.

## Single highest-EV mechanism family for our case

**Driver-level target encoding from EXTERNAL F1 historical data,
smoothed nested-CV alpha ≈ 50, joined on Driver only — applied
ONLY to is_real_driver == True rows; D### rows get the global
prior.**

Concretely:
1. Source ISSUES 4b debashish F1 1950-2022 dataset (already
   identified in our brainstorm).
2. For each REAL driver present in train.csv + test.csv (the 38%
   real-driver subset), compute career pit-stop rate from the
   external dataset — `pit_rate_career[d] = pits / races` over
   driver d's full career, OR if our target is more granular,
   per-driver target-mean over the external dataset.
3. Smooth: `te[d] = (n[d] * pit_rate[d] + alpha * global_mean) /
   (n[d] + alpha)`, alpha tuned in inner CV. Start alpha=50,
   sweep {10, 25, 50, 100}.
4. For synthetic D### drivers, fill with global_mean (no
   external info → don't fabricate). Add binary `te_is_external`
   indicator so the model can learn the two regimes.
5. NESTED CV: re-fit the lookup table per outer fold using
   ONLY the outer-train portion of train.csv merged with the
   FULL external dataset. The external dataset is fold-invariant
   (it's not in our CV split), so this collapses to: external
   stats are a constant input; only any train-side smoothing
   prior needs fold-rebuilding. Treat external rows as additional
   training data when joining is row-level; treat them as a
   per-Driver aggregate when joining is feature-level. PREFER
   the feature-level (aggregate) variant first — it's cheaper
   and the precedent (S5E2 Backpacks `orig_price`, S5E3 Deotte
   feature-group lookup) is exactly this shape.

Why this and not concat-as-rows: our train.csv has 439k rows
and a featureful schema (pit decision rich context). The
external F1 1950-2022 dataset has different per-row columns
(race results, lap times, but not the same features as our
train). Schema mismatch makes row-concat costly — we'd have to
project the external dataset into our column space and many of
our features would be NULL for those rows. Feature-level
aggregation (per-Driver career prior) sidesteps the schema
problem and is the exact mechanism that worked for Backpacks.

Secondary mechanism if (a) plateaus: **per-(Driver, Circuit)
or per-(Driver, Era) cross-tabulated prior from external
data**, again smoothed. This is the "feature-group" version of
the Deotte trick — combinations of two external attributes
keyed against our internal columns.

## Leakage red flags that force us to re-validate

1. **Adversarial-validation gate (R25)**: if we build the
   per-Driver external prior using TRAIN-AND-TEST drivers (as
   a transductive lookup table), we must re-run AV on the
   augmented feature. Current AV-AUC = 0.502 (safe);
   adding an external-data-derived feature can push it past
   0.55 if the external dataset's Driver-coverage pattern
   differs systematically between train and test rows.

2. **Time-leakage from external dataset**: the F1 1950-2022
   dataset extends beyond train.csv's effective time window
   (we should confirm what year-range the synthetic was
   generated from). If we use, say, a 2021-2022 driver's
   2023-era career stats to predict a 2018-era race in
   train, we've leaked future. Mitigation: cap external
   lookup to year < min(train year for that driver), OR
   use full-career aggregates and accept the leak as
   benign-on-distribution (target-encoding stats are stable
   across a multi-year career for established drivers).

3. **Nested CV is mandatory (R24, R33)**: any
   target-encoding-style transformation that touches labels
   must be re-fit per outer fold. For external priors
   computed from a fold-invariant external dataset, the
   nested CV burden is light (the lookup table is the same
   across folds) BUT if we smooth via global_mean computed
   from train.csv, that global_mean must be re-fit per fold.

4. **Synthetic-D###-vs-real split bias**: the 120 bp gap is
   on real drivers. If our external prior helps a lot on the
   real subset but the inner CV averages across all rows
   (including D### where the prior is just global_mean), the
   CV signal will under-estimate the true LB lift on the
   real-driver subset. Recommend computing two CV AUCs:
   real-only, D###-only, BOTH, before deciding.

5. **R3 4-gate pre-LB filter** still applies — G1 standalone
   OOF, G2 blend lift on REAL subset specifically, G3 net
   rare-class-flip ≥ 0.5, G4 direction asymmetry.

## Implementation note

Start with the cheapest probe: build per-Driver career
pit-rate from external data, join on Driver, smooth alpha=50,
feature-engineer is_real_driver indicator already in place
from prior probe. Run probe.py bote, then 1-fold smoke on
real-only subset (should see ≥ +30 bp OOF lift on real-only
to be worth promoting). If smoke passes, R3 gate, then
PI-approved single-shot submission.

## Sources

- [Chris Deotte S5E3 Rainfall 2nd place](https://www.kaggle.com/competitions/playground-series-s5e3/writeups/chris-deotte-2nd-place-gbdt-nn-svr-original-data) (gated, title-only verified)
- [NVIDIA cuDF-pandas Grandmaster Pro Tip — S5E2 Backpacks 1st](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)
- [NVIDIA Kaggle Grandmasters Playbook — 7 techniques](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Playground S4E1 Bank Churn competition page](https://www.kaggle.com/competitions/playground-series-s4e1)
- [Playground S5E3 Rainfall competition page](https://www.kaggle.com/c/playground-series-s5e3)
- [Medium: Kaggle Playground how top competitors win 2025](https://medium.com/@gauurab/kaggle-playground-how-top-competitors-actually-win-in-2025-c75d4b380bb5)
