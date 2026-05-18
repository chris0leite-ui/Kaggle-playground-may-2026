# Notebooks research — 2026-05-18 (refreshed pass, Rule 7 plateau)

Triggered by: 7-segmentation plateau (1 win, 6 nulls/marginals). Prior pass
above this header was a null-harvest from Kaggle pages (JS/reCAPTCHA gated).
This refreshed pass pivots to **domain-literature + adjacent-Kaggle-comp
proxies** since direct s6e5 notebook scraping remains blocked.

Wall-clock budget: 15 min web-only. No local compute.

## Sources scouted this pass

1. **Frontiers in AI (2025) — "Data-driven pit stop decision support for Formula 1 using deep learning models"**
   - URL: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full
   - Mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC12626961/
   - WebFetch status: full body extracted. Explicit feature list and Bi-LSTM arch.
2. **Tracy Renee / Crystal X — Medium**
   - URL: https://tracyrenee61.medium.com/predict-on-the-probability-a-formula-1-driver-will-pit-on-the-next-lap-e6b8adb4a45f
   - WebFetch status: full body. CatBoost + train_test_split, no CV, claims 94% (likely accuracy not row-AUC). Already logged in prior pass — no new mechanism.
3. **F1 Strategy Dataset (Kaggle) — aadigupta1601/f1-strategy-dataset-pit-stop-prediction**
   - URL: https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction
   - WebFetch: title only (gated). Schema not extractable. We already use a slim-kNN cut of this; no new joinable column surfaced.
4. **GPFans / FlowRacers / Formula1.com — undercut / overcut domain articles**
   - https://www.gpfans.com/en/f1-news/1016512/f1-undercut-overcut-explained/
   - https://flowracers.com/blog/undercut-overcut-double-stack-f1-pit-stops/
   - https://www.formula1.com/en/latest/article/undercut-vs-overcut-why-tyre-strategy-was-so-finely-poised-in-monaco-and-why.1YYMDkEBnFols8bDWtSXiz
   - WebSearch returned readable excerpts. Quantitative rule of thumb: chaser within 2–4 s of leader + ≥1.5 s/lap fresh-tyre delta + 4–5 clear laps to overcome pit delta → undercut threshold zone.
5. **laurence9899/F1_Pitstop_Predict_ML (GitHub)** — TensorFlow per-lap binary; README only, no feature list extractable.
6. **Tapan Babbar Medium — Predicting F1 Lap Times** (https://medium.com/@tapanbabbar/predicting-f1-lap-times-a-comparison-of-ml-models-275ac8a06e19) — lap-time delta features for adjacent context.
7. **Kaggle s6e3 1st-place writeup ("KGMON Playbook")** — gated; not extractable.
8. **s6e5 /discussion** — still gated (reCAPTCHA, as in prior pass).

Net: 1 high-value academic source (Frontiers Bi-LSTM with explicit feature
list) + 3 domain quantification articles. Direct Kaggle notebook discourse
remains dark for s6e5.

## Mechanism extracts

### From Frontiers Bi-LSTM paper (s6e5-shaped sequence framing)

- **Driver-pair features**: `DriverAheadPit` and `DriverBehindPit` — binary
  flags for whether the immediate neighbors in race order pitted on the
  current lap (or recent window). Directly encodes undercut/overcut
  contagion. Per-lap, per-(driver, race) — requires sort-by-position on lap.
- **Delta-lap-time features**: `delta_laptime` = lap-to-lap time difference
  for the same driver. Captures degradation curvature — pit usually preceded
  by a step-up in lap time. We have lap time but not necessarily its
  first-derivative-by-driver-race.
- **CumulativeTimeStint**: cumulative time on current compound (distinct
  from `TyreLife` which is lap count). Non-linear with compound + track
  characteristics.
- **Track-status flag**: yellow flag / safety car. Massive pit-call shifter
  in domain. If s6e5 train doesn't expose this directly, can sometimes be
  inferred from sudden race-wide lap-time inflation in the same lap.
- **Stratified split for class imbalance**: SMOTE used. Not relevant for us
  (AUC, large train, our calibration handles prior).
- **Bi-LSTM 256→128→64 over 10-timestep sequences**: similar to our v2
  transformer; their innovation isn't the arch but the **DriverAhead /
  DriverBehind context** which their sequence input encodes per-row.

### From undercut/overcut domain articles

- Numeric undercut window: chaser within 2–4 s of leader, ≥1.5 s/lap fresh
  delta, 4–5 clear laps. → suggests a continuous `undercut_pressure` feature
  = function of (gap_to_car_ahead, recent_pace_delta, laps_remaining_on_tyre).
- **Double-stack** pattern: a team pitting both drivers same lap (rare but
  high-information). If `Team` + `LapNumber` is in the data, count of
  same-team pit events in last N laps is a leading indicator.

### Domain prior on segments where we're weak

- MEDIUM × Stint 2 weakness (our AUC 0.897) maps to the domain's "midrace
  decision window" — exactly where undercut/overcut pressure is highest.
  This is the segment where DriverAheadPit-style features should bite.
- 2023 anomaly (AUC 0.953 at 1% prior) is consistent with the 2023 F1
  calendar including more low-degradation circuits; a `circuit_deg_class`
  external join (3-class: high/med/low typical tyre wear by track) could
  collapse the year-anomaly signal into a more general feature.

## Punch list — 5 candidate mechanisms

For each: 1-line novelty check vs our mechanism ledger (Path-B per-segment
shrinkage, DriverClass × Stint, Year × Compound, Race-cluster × Stint,
Compound × FirstPitWindow, multi-segmentation rank-blend, K=27 super-base
stack, slim-kNN, r4_segment_fe, HMM, transformer v2, swap-noise DAE).

### C1. Driver-pair contagion features: `AheadPit_lag{1,2,3}` × `BehindPit_lag{1,2,3}`

- **What**: sort rows within (Race, Year) by Position-or-LapTime per lap;
  for each (driver, lap), compute whether the driver immediately ahead /
  behind in track order pitted in lap, lap-1, lap-2. 6 features total.
  Optional: same-team-pitted-in-last-N flag (double-stack proxy).
- **Predicted lift**: 3–8 bp. Strong domain prior (undercut/overcut is the
  central pit-decision mechanism). Directly hits MEDIUM × Stint 2 weakness.
- **Cost**: ~30 min FE (lap-level sort + shift within group) + 1 full 5-fold
  bag. Risk: requires within-race lap ordering — if our row-level data
  doesn't preserve intra-lap order, must reconstruct from LapTime or
  Position columns.
- **Novelty check**: NOT in mechanism ledger. r4_segment_fe is 9 own-row
  interaction features; this is cross-driver context. HMM and transformer
  v2 are sequence-of-self; this is sequence-of-neighbors. Distinct axis.
- **Source**: Frontiers Bi-LSTM paper — `DriverAheadPit` / `DriverBehindPit`
  features (https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full).

### C2. Lap-time derivative features: `delta_laptime_lag{1,2}` + `delta_laptime_rolling3_per_driverrace`

- **What**: per-(Driver, Race, Year), within-group first-difference and
  rolling-3 mean of LapTime. 3 features. Captures the degradation
  curvature step-up that precedes a pit.
- **Predicted lift**: 2–5 bp. Lap time is in our base features; its
  derivative is not in any logged mechanism. Universal across compounds,
  so should not just shore up MEDIUM × Stint 2 but lift global AUC modestly.
- **Cost**: ~20 min FE + 1 bag. Low risk: pure within-group shift, fully
  R24 fold-safe (no label aggregation).
- **Novelty check**: NOT in ledger. r4_segment_fe interactions are
  cross-feature; this is a within-feature temporal derivative. Transformer
  v2 sees the sequence but the GBDT bases never get the explicit Δ scalar.
- **Source**: Frontiers paper feature list, plus Tapan Babbar Medium on
  lap-time lag features.

### C3. Continuous undercut-pressure scalar: `undercut_window_score`

- **What**: a hand-designed continuous feature
  = sigmoid((4s - gap_to_car_ahead) / 1s) × sigmoid((stint_age - threshold)
  / 2) × (laps_remaining > 5). Encodes "in the undercut sweet spot now."
  All ingredients (gap, tyre life, laps remaining) are derivable from
  existing columns. Heuristic, not learned — Rule 6 closed-form first.
- **Predicted lift**: 1–4 bp. Heuristic is brittle; the value is forcing
  the GBDT to see the conjunction explicitly rather than rediscover it.
- **Cost**: ~15 min FE + bag. Very low risk.
- **Novelty check**: NOT in ledger. FirstPitWindow segmentation cuts
  at lap thresholds (categorical); this is a continuous strategic
  pressure indicator combining 3 inputs. Different axis.
- **Source**: GPFans + FlowRacers + Formula1.com undercut/overcut articles
  (numeric thresholds: 2–4 s, 1.5 s/lap, 4–5 laps).

### C4. Per-(Compound × Stint) target-mean stack base with R24 fold-refit

- **What**: a dedicated tiny base model whose only features are 6
  fold-refit target means: `mean_y_by[Compound × Stint]`,
  `mean_y_by[Compound × Stint × Year]`, `mean_y_by[Compound × Stint ×
  CircuitCluster]`, plus their counts (4 features). Train as a single
  shallow tree (depth 3) so it's a calibrated specialist, then add to
  the K=27 stack.
- **Predicted lift**: 1–3 bp via stack diversity, not own AUC.
  Hits MEDIUM × Stint 2 directly (one of the 6 cells).
- **Cost**: ~25 min code + bag + meta refit. Risk: R24 leakage if not
  fold-refit; gated by R33 (inner-CV-validate post-hoc).
- **Novelty check**: We have many segmentation **operators** (Path-B
  shrinkage on K=13). This is a segmentation **base learner** whose only
  signal is the segment prior — semantically a "target-encoding base."
  Not in ledger.
- **Source**: implicit in every undercut/overcut article (compound × stint
  is the strategic atom); concrete framing from Frontiers paper.

### C5. Circuit-degradation-class external join (3-level: high/medium/low)

- **What**: hand-curated 3-level mapping of each Race to a tyre-degradation
  class (e.g., Bahrain/Suzuka = high; Monaco/Hungary = low). Replaces
  high-cardinality race target encoding with a low-cardinality domain
  prior. 1 categorical feature + its 3 interactions with Compound (4
  features). External knowledge — not learned from training labels, so
  R24-trivial.
- **Predicted lift**: 2–4 bp via collapsing the 2023 year-anomaly
  (AUC 0.953 at 1% prior) into a generalizable prior. Year × Compound
  in ledger but Circuit-cluster × Stint there is RACE-clustering by
  signal, not by physical tyre-wear class.
- **Cost**: ~30 min — mostly the curation table (24 races × 4 years).
  Risk: subjective labels; mitigate by sourcing the table from a
  published tyre-degradation report (Pirelli releases these per round).
- **Novelty check**: Race-cluster × Stint in ledger clusters races by
  empirical signal similarity. This clusters by **physical mechanism**
  (degradation class) — orthogonal axis. Year × Compound segmentation
  doesn't decompose the year effect; this hypothesizes the year effect
  is a degradation-class compositional shift.
- **Source**: Frontiers paper (excludes wet/intermediate as "reactive,"
  implying dry-deg class is the controllable axis) + undercut articles
  (tyre-deg is the universal denominator).

## Dedup audit

Against our mechanism ledger entries (Compound × Stint Path-B, K=27 stack,
slim-kNN, r4_segment_fe, HMM, transformer v2, swap-noise DAE, DriverClass
× Stint, Year × Compound, Race-cluster × Stint, Compound × FirstPitWindow,
multi-segmentation rank-blend):

- C1 driver-pair lag: orthogonal to all (cross-driver, not own-row /
  own-sequence).
- C2 lap-time derivative: orthogonal (within-feature temporal derivative;
  not a GBDT input today even though LapTime is).
- C3 undercut-pressure scalar: orthogonal (continuous handcrafted
  conjunction; FirstPitWindow is a categorical cut, r4_segment_fe is
  pairwise products of base features).
- C4 segmentation base learner: orthogonal axis (segmentation as a base
  in the stack, not as an operator on top of the stack).
- C5 circuit-deg-class: orthogonal axis (physical mechanism, not signal
  similarity).

## Caveats

- All 5 candidates assume the row-level data has, or can reconstruct: lap
  ordering within a race, gap-to-neighbor, lap time per row, compound,
  stint index. Verify schema before any commits.
- LB-lift estimates are upper-bounded by the 10–20 bp gap to the leader;
  diminishing returns once we crash the MEDIUM × Stint 2 segment.
- R33 inner-CV mandatory for C4 (any segment-prior base risks in-sample
  illusion).
- Frontiers paper uses 2020–2024 real F1 data; s6e5 is synthetic
  ("playground"). Synthetic generator may have stripped some of the
  Frontiers-paper features. **Schema verification is the gating step.**

## URLs cited (this pass)

- Frontiers AI: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full
- PMC mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC12626961/
- Tracy Renee Medium: https://tracyrenee61.medium.com/predict-on-the-probability-a-formula-1-driver-will-pit-on-the-next-lap-e6b8adb4a45f
- Tapan Babbar Medium: https://medium.com/@tapanbabbar/predicting-f1-lap-times-a-comparison-of-ml-models-275ac8a06e19
- GPFans undercut/overcut: https://www.gpfans.com/en/f1-news/1016512/f1-undercut-overcut-explained/
- FlowRacers undercut/overcut/double-stack: https://flowracers.com/blog/undercut-overcut-double-stack-f1-pit-stops/
- Formula1.com Monaco strategy debrief: https://www.formula1.com/en/latest/article/undercut-vs-overcut-why-tyre-strategy-was-so-finely-poised-in-monaco-and-why.1YYMDkEBnFols8bDWtSXiz
- F1 Strategy dataset (gated): https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction
- laurence9899 GitHub: https://github.com/laurence9899/F1_Pitstop_Predict_ML
- s6e5 /discussion (gated): https://www.kaggle.com/competitions/playground-series-s6e5/discussion
- s6e3 1st-place writeup (gated): https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm
