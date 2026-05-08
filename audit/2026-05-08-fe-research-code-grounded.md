# FE research code-grounded supplement — comp-day 8 (2026-05-08 PM, late)

> Branch: `claude/research-feature-engineering-7oCmj`. Companion to
> `audit/2026-05-08-fe-research-survey.md` — the prior note relied on
> agent summaries that under-counted the FE in `external/kernels/`.
> This note is **code-grounded**: every feature listed below was read
> directly from a notebook in our local mirror.

## How this changes the picture

The agent summaries reported ~10 distinct features per public notebook;
the code shows **80+ features** in svanikkolli's `make_features_A`
alone. Several feature classes that look like they should break our
rank-lock are present in the public code and **fold-safe by
construction** (they use `PitStop`, a feature column from the
prior-lap pit indicator, not `PitNextLap` the target). We have not
tested any of them.

Three additional structural findings change earlier conclusions:

1. **Romanrozen's stack-meta is LightGBM**, not LR. The input matrix
   is ~54 columns wide: 5 OOFs + 5 rank-norm + 5 logit-transform +
   10 pairwise products + 10 pairwise abs-diffs + 13 raw FE columns +
   6 TE columns. **Our EXP-NEW falsification of "non-LR meta" tested
   LightGBM only on PCA top-K of K=27 OOFs and on raw [P, rank,
   logit]-expansion of K=27 = 30 columns. It did NOT test LightGBM
   on a richly-featured stack matrix.** The "non-LR meta closed
   negative" assumption in `state/hypothesis-board.md` is therefore
   narrower than I treated it: refuted for prediction-only inputs,
   open for prediction + raw-FE + TE inputs.
2. **Field-state features ≠ field-state aggregates**. We tested
   per-(Race, Year, LapNumber) aggregates at K=24+1 and they were
   NULL. Svanikkolli v12's "field-state" pack adds **gap-to-car
   features** built from cumulative race-time + position-sorted shifts
   (Section F5 below) — structurally different from groupby
   aggregates. Untried.
3. **The (Driver, Race, Year) TE that was killed in Day-17 P1** was
   killed because the implementation aggregated on full train then
   applied to val rows in the same call. The nested-fold form
   (sklearn `TargetEncoder(cv=5)` or romanrozen's `cv_target_encode`
   function) is a different construction and explicitly safe under
   Rule 24. Romanrozen claims +0.002 OOF AUC from this construction;
   we have NOT retested it in the safe form.

## Code-grounded feature inventory

### Pipeline A (svanikkolli v12 / romanrozen — 80+ features)

The two notebooks share a near-identical feature factory; svanikkolli
v12 adds sections F3–F7 below on top of romanrozen's base.

#### A. Tyre algebra (8 features) — already partially in our K=4

`tyre_life_sq`, `tyre_life_log`, `tyre_life_sqrt`, `deg_per_lap`,
`compound_life_ratio` (`TyreLife / max(TyreLife | Compound)`),
`compound_max_life` (lookup table; **three different tables in the
public corpus** — see "Compound expected-life lookups" below),
`compound_tyre_norm` (`TyreLife / compound_max_life` clipped 0..2),
`tyre_overdue_norm` (`compound_tyre_norm > 0.85`).

#### B. Race progress (10 features)

`est_total_laps` (`LapNumber / RaceProgress`, clipped 30..80),
`laps_remaining`, `tyre_pct_remaining` (`TyreLife / (laps_remaining +
1)`), `is_pit_window` (`0.28 ≤ RaceProgress ≤ 0.62`), `is_late_race`
(`RaceProgress > 0.75`), `position_pressure` (`Position * (1 -
RaceProgress)`), `urgency_score` (`|Cumulative_Degradation| * (1 -
RaceProgress)`), `race_phase` (4-bin pd.cut), `norm_position` (`1 -
(Position - 1) / 19`), `life_x_progress` (`TyreLife * RaceProgress`).

#### C. Lags + rolling within (Driver, Race, Year) (18 features)

`delta_lag1`, `delta_lag2`, `prev_pit` (`PitStop.shift(1)`),
`delta_accel` (`LapTime_Delta - delta_lag1`), `roll{3,5,7,10,15}_lt`
(rolling means of `LapTime`), `roll{3,7}_d` (rolling means of
`LapTime_Delta`), `roll3_std` (rolling std of `LapTime`),
`lap_vs_r{3,5,7,10}` (deviation from rolling mean), `deg_velocity`
(`Cumulative_Degradation.diff(3) / 3`), `is_slow_lap` (`LapTime >
roll5_lt * 1.15`), `lap_in_stint` (cumcount within Stint),
`stint_start_lap`.

#### D. Race-level context — **LABEL-DERIVED, requires per-fold refit per Rule 24**

`race_avg_pit_lap` (mean LapNumber where `PitNextLap=1` per
Race-Year), `compound_avg_life` (mean TyreLife where `PitNextLap=1`
per Compound), `race_total_laps` (max LapNumber per Race-Year — NOT
label-derived), `race_max_stint` (max Stint per Race-Year — NOT
label-derived), and 12 derived: `pit_window_flag`, `tyre_vs_comp_avg`,
`overdue_pit`, `laps_remaining_race`, `tyre_age_pct_race`,
`stint_progress`, `tyre_life_pct`, `stint_end_est`, `laps_until_stop`,
`pit_imminent`, `pit_in_5`, `cliff_flag`. **The romanrozen / svanik
notebooks fit these once on full train then apply to both partitions
in CV — that's the exact Day-17 leak pattern. Per-fold refit is
mandatory for our setting.**

#### E. Strategy interactions (10 features)

`laps_to_stop`, `past_optimal`, `must_pit_or_stay`, `undercut_threat`
(`Position ≤ 10 & is_pit_window`), `relative_stint`, `deg_x_win`,
`over_x_win`, `tyre_x_pres`, `r3_x_life`, `comp_stint`.

#### F2. Cliff detection + window arithmetic — **label-derived (uses race_avg_pit_lap)**

`lap_accel_smooth` (rolling-5 mean of `LapTime_Delta`, then `.diff()`),
`tyre_cliff_imminent` (`lap_accel_smooth > 0.3 & TyreLife > 12`),
`laps_to_window_start` (`race_avg_pit_lap * 0.85 - LapNumber`),
`laps_to_window_end`, `in_optimal_window`. Plus `stint_lt_baseline`
(first lap-time within stint via `groupby.transform('first')`,
**fold-safe**), `stint_degradation_ratio` ((`LapTime -
stint_lt_baseline`) / `stint_lt_baseline`, **fold-safe**).

Plus `compound_race_median_lt` (median LapTime per
Race-Year-Compound, **fold-safe — uses LapTime not target**),
`lap_vs_compound_baseline`. Plus `roll7_std`, `roll3_var_ratio` (=
`roll3_std / roll7_std`). Plus `pos_lag3`, `pos_trend_3`,
`losing_positions`. Plus `race_compound_max_life` (max TyreLife per
Race-Year-Compound, **fold-safe**), `tyre_freshness_pct`,
`must_pit_signal`, `urgency_composite`. Plus `dc_avg_stint_life`
(mean TyreLife where PitNextLap=1 per Driver-Compound,
**LABEL-DERIVED**), `driver_vs_avg_life`, `driver_overdue_personal`.

#### F3. Field-level competitor features (svanikkolli v12 only) — **all FOLD-SAFE**

These use `PitStop` (prior-lap pit indicator, a feature column) NOT
`PitNextLap` (the target). No Rule 24 issue.

```python
g_lap = df.groupby(['Race', 'Year', 'LapNumber'])
df['avg_field_tyre_age']   = g_lap['TyreLife'].transform('mean')
df['max_field_tyre_age']   = g_lap['TyreLife'].transform('max')
df['min_field_tyre_age']   = g_lap['TyreLife'].transform('min')
df['field_tyre_age_pct']   = (df['TyreLife'] / (df['max_field_tyre_age'] + 1)).clip(0, 1)
df['is_oldest_tyre']       = (df['TyreLife'] == df['max_field_tyre_age']).astype(int)
df['cars_older_tyres']     = g_lap['TyreLife'].transform(lambda x: (x > x.mean()).sum())
df['n_diff_compounds']     = g_lap['Compound'].transform('nunique')
df['n_pitted_this_lap']    = g_lap['PitStop'].transform('sum')          # fold-safe
df['n_pitted_race_last5']  = (df.groupby(['Driver','Race','Year'])['PitStop']
                                .transform(lambda x: x.shift(1).fillna(0).rolling(5,1).sum()))
df['field_pit_rate']       = g_lap['PitStop'].transform('mean')         # fold-safe
df['tyre_age_vs_field']    = (df['TyreLife'] - df['avg_field_tyre_age']).clip(-20, 20)
```

#### F4. Safety-car proxy split (svanikkolli v12 only) — **all FOLD-SAFE**

```python
df['field_median_lt']  = g_lap['LapTime (s)'].transform('median')
sc_ratio = df['field_median_lt'] / df['roll5_lt'].fillna(df['LapTime (s)']).clip(lower=60)
df['is_sc_proxy']      = (sc_ratio > 1.08).astype(int)
df['is_vsc_proxy']     = ((sc_ratio > 1.08) & (sc_ratio <= 1.30)).astype(int)
df['is_full_sc_proxy'] = (sc_ratio > 1.30).astype(int)
# laps_since_sc / laps_since_vsc / laps_since_full_sc via cumulative count-since-last-1
# in_sc_window / in_vsc_window / in_full_sc_window thresholds 3 / 2 / 5
```

#### F5. Gap-to-car features (svanikkolli v12 only) — **STRUCTURALLY NOVEL FOR US**

Builds a cumulative race time per driver, then sorts by Position
within (Race, Year, LapNumber) and shifts to get the car ahead /
behind. The gap is then `cum_rt[me] - cum_rt[ahead]`. arXiv 2501.04068
TimeSHAP analysis ranks "Gap Ahead" as the **#1 feature** driving pit
decisions in F1.

```python
PIT_DELTA = 22.0
df['_clean_lt'] = df['LapTime (s)'] - df['PitStop'].fillna(0) * PIT_DELTA
df['_cum_rt']   = df.groupby(['Driver','Race','Year'])['_clean_lt'].cumsum()
gap = (df[['Race','Year','LapNumber','Position','_cum_rt','PitStop']]
         .sort_values(['Race','Year','LapNumber','Position']).reset_index())
glap = gap.groupby(['Race','Year','LapNumber'])
gap['_cum_ahead']  = glap['_cum_rt'].shift(1)
gap['_cum_behind'] = glap['_cum_rt'].shift(-1)
gap['_ahead_pitted']  = glap['PitStop'].shift(1).fillna(0)
gap['_behind_pitted'] = glap['PitStop'].shift(-1).fillna(0)
# reindex back onto df.index, then:
df['gap_to_car_ahead']    = (df['_cum_rt'] - df['_cum_ahead']).clip(-5, 60)
df['gap_to_car_behind']   = (df['_cum_behind'] - df['_cum_rt']).clip(-5, 60)
df['in_drs_range']        = (df['gap_to_car_ahead'] < 1.2).astype(int)
df['undercut_viable']     = ((df['gap_to_car_ahead'] > 0.5) & (df['gap_to_car_ahead'] < 4.0)).astype(int)
df['threat_from_behind']  = (df['gap_to_car_behind'] < 2.0).astype(int)
df['gap_ahead_delta']     = df.groupby(['Driver','Race','Year'])['gap_to_car_ahead'].diff(3).fillna(0).clip(-10, 10)
df['nearby_pit_pressure'] = (df['car_ahead_pitted_now'] + df['car_behind_pitted_now'] +
                              g_drv['car_ahead_pitted_now'].shift(1).fillna(0) +
                              g_drv['car_behind_pitted_now'].shift(1).fillna(0)).clip(0, 4)
```

#### F6. Mandatory 2-compound rule (svanikkolli v12 only) — **FOLD-SAFE, untried**

FIA regulation: in dry races each driver must use ≥2 different
compounds. Once a stop is mandatory, pit probability rises sharply.

```python
df['_comp_first']   = df.groupby(['Driver','Race','Year'])['Compound'].transform('first')
df['_comp_changed'] = (df['Compound'] != df['_comp_first']).astype(int)
df['n_compounds_used'] = (df.groupby(['Driver','Race','Year'])['_comp_changed']
                             .transform('cummax') + 1).astype(int)
df['mandatory_pit_pending'] = ((df['n_compounds_used'] < 2) & (df['RaceProgress'] > 0.45)).astype(int)
df['mandatory_urgency']     = (df['mandatory_pit_pending'] * df['RaceProgress'])
```

#### F7. Fuel load correction (svanikkolli v12 only) — **FOLD-SAFE, untried**

Single physics constant (0.035 s per lap from fuel-burn). Recovers a
cleaner tyre-only degradation signal.

```python
FUEL_GAIN_PER_LAP = 0.035
df['fuel_adj_lt']        = df['LapTime (s)'] + df['LapNumber'] * FUEL_GAIN_PER_LAP
df['fuel_corrected_deg'] = (df['fuel_adj_lt']
                              - df.groupby(['Driver','Race','Year'])['fuel_adj_lt'].transform('first')).clip(-5, 20)
```

### Compound expected-life lookups (three competing tables)

| Compound | romanrozen | driver-FE | baarzenzijncool empirical (median TyreLife at pit) |
|---|---:|---:|---:|
| SOFT | 15 | 25 | **12** |
| MEDIUM | 30 | 35 | **16** |
| HARD | 50 | 45 | **20** |
| INTERMEDIATE | 25 | 30 | **17** |
| WET | 20 | 40 | **11** |

The **baarzenzijncool empirical-from-data table** is most likely
calibrated to the s6e5 synth (the others are population priors from
Pirelli or competing data). Use baarzenzijncool's table as the
default if testing pick #4 in the prior audit note.

driver-FE also publishes `COMPOUND_WINDOW_START` = {SOFT: 0.64,
MEDIUM: 0.68, HARD: 0.72, INTERMEDIATE: 0.62, WET: 0.60} — the
fraction of expected-life at which the pit window opens. This is
**genuinely new information** not in any other notebook or our
internal docs.

### Target-encoding configurations across notebooks

| Notebook | TE columns | smoothing | nested-fold? |
|---|---|---:|---|
| romanrozen | (Driver,Race,Year), (Driver,Race), (Race,Compound), (Driver,Compound), (Race,Year), (Driver,Race,Compound) | 20/30/25/25/20/15 | **yes** (custom `cv_target_encode`) |
| pit-or-stay | Driver, Race_Year, Driver_Race, Driver_Year, Race, Stint_Compound | 30 | **yes** (per-fold) |
| yekenot (already in our K=4) | (Race, Compound), (Race, Year), (Driver, Compound) | sklearn `smooth='auto'` | yes (sklearn `TargetEncoder(cv=5)`) |
| driver-FE | inner-OOF on full TE-column block | alpha=80 | yes |

### Stacking-meta architectures

| Notebook | Meta input | Meta model |
|---|---|---|
| romanrozen | 5 OOF + 5 rank + 5 logit + 10 pairwise prod + 10 pairwise diff + 13 raw FE + 6 TE = ~54 cols | **LightGBM** (num_leaves=31, lr=0.02, n_est=800) |
| svanikkolli v12 | (likely similar; not read end-to-end here) | (likely similar) |
| our K=4 PRIMARY | 4 base predictions only | LR (per-segment Compound × Stint, τ=100k) |

## Reconsiderations from this code-grounded read

### EXP-NEW assumed scope is narrower than the hypothesis-board states

`state/hypothesis-board.md` "Killed" line: "Non-LR meta architecture
(LightGBM on PCA / raw expansion). PCA-meta probe 2026-05-08 PM:
LightGBM meta is *worse* than LR meta by 1-2 bp at every input
representation tested." — **representations tested were prediction-
only**. Romanrozen's stack-meta uses prediction + raw FE + TE; that
input space was not in the falsification set. The hypothesis-board
entry should be amended to: "non-LR meta on **prediction-only**
inputs is falsified; non-LR meta on **richly-featured stack matrix**
is untested."

### Field-state aggregates ≠ gap-to-car features

Day-17 closure of "field-state cross-driver aggregates" tested the F3
class (transform('mean'), transform('max'), etc.) at K=24+1 and got
NULL. The F5 gap-to-car class is structurally different — it builds
a *cumulative race time* per driver and sorts by Position to find
neighbours. The Day-17 closure does not transitively close F5.

### Day-17 P1 falsification scope clarified

The Day-17 P1 (`Driver × Race × Year` 3-way TE) was killed because
of fitting on full train. The proper nested-fold form using sklearn's
`TargetEncoder(cv=5)` or romanrozen's `cv_target_encode` is a
different construction. Yekenot's 2-way TE inside `TargetEncoder(cv=5)`
already passes our Rule 24 audit (per
`external/kernels/ps-s6-e5-realmlp-pytabkit/VALIDATED.md` L34-L38);
the 3-way variant in the same nested-fold framework is open.

## Revised top picks (replaces all prior top-3 lists in this audit)

The expected-value ranking now is:

| # | Pick | Cost | Novelty | Mechanism |
|---|---|---:|---|---|
| 1 | **Gap-to-car features (F5)** | ~15 min CPU | HIGH — TimeSHAP #1, structurally orthogonal | Cumulative race-time + position-sorted shifts → `gap_to_car_ahead`, `in_drs_range`, `undercut_viable`, `gap_ahead_delta` |
| 2 | **Mandatory 2-compound rule (F6)** | ~5 min CPU | HIGH — domain regulation, fold-safe | `n_compounds_used` cummax, `mandatory_pit_pending`, `mandatory_urgency` |
| 3 | **Bigram/trigram nested-fold TE sweep** | ~12 min CPU | MEDIUM — recurring across 3 1st-place writeups | Re-test (Driver, Race, Year) and (Driver, Race, Compound) inside sklearn `TargetEncoder(cv=5)` |
| 4 | **VSC vs Full-SC proxy split (F4)** | ~8 min CPU | MEDIUM — replaces our single-proxy field-state attempt | `field_median_lt / roll5_lt` thresholded at 1.08 / 1.30 + `laps_since_sc` |
| 5 | **Heilmeier `remaining_pit_stops_proxy`** | ~5 min CPU | MEDIUM — TUM 2020 #1 ablation feature | `expected_stops - (Stint - 1)` |
| 6 | **Fuel load correction (F7)** | ~3 min CPU | LOW-MEDIUM — single coefficient, recovers tyre-only signal | `fuel_corrected_deg = LapTime + 0.035*LapNumber - first()` |
| 7 | **Field-state competitor features (F3)** | ~8 min CPU | MEDIUM — uses PitStop not target, structurally fold-safe | `n_pitted_this_lap`, `field_pit_rate`, `cars_older_tyres` |
| 8 | **LightGBM stack-meta on richly-featured matrix** | ~10 min CPU | HIGH — refutes EXP-NEW assumed scope | Build romanrozen-style 54-col stack matrix on K=4 OOFs, fit LGBM meta, gate vs current LR-meta |

Picks 1, 2, 4, 6, 7 are **all fold-safe by construction** (no
label-derived aggregates) — they pass Rule 24 trivially. Picks 3, 5,
8 require explicit per-fold refit infrastructure.

Total smoke + 5-fold OOF probe budget for picks 1-8: **~65 min CPU**.
A natural batching is picks 1+2+6 first (~25 min CPU; the highest-EV
fold-safe-by-construction trio), gate against K=4+1 plain LR-meta;
then picks 3-5-7 (~35 min CPU); then pick 8 (~10 min CPU + needs the
stack matrix already built).

## Open questions for PI

1. **Does the EXP-NEW scope clarification merit a hypothesis-board
   amendment?** The bookkeeper rule is "FALSIFIED is permanent" but
   the falsification was narrower than the entry currently states.
2. **Authorise picks 1+2+6 as a batched smoke-then-OOF run?** They
   are fold-safe by construction so the leakage gate is trivial; the
   only risk is wall time (~25 min CPU).
3. **Does the K=4 PRIMARY architecture admit a LightGBM stack-meta
   without breaking the per-segment Path-B shrinkage τ=100k**? The
   answer determines whether pick #8 is a swap or a parallel path.

## Caveats and discipline reminders

- **`PitStop` ≠ `PitNextLap`.** PitStop is the prior-lap pit
  indicator and a *feature*; PitNextLap is the *target*. Features
  derived from PitStop (F3, F4, F5) are fold-safe by construction.
  Features derived from PitNextLap aggregates require per-fold refit.
- **Rule 24 still binding for D-section features.** romanrozen and
  svanikkolli's notebooks fit `race_avg_pit_lap`, `compound_avg_life`,
  `dc_avg_stint_life` once on full train and apply to val rows in
  CV — that is the Day-17 leak pattern. Any port must refit per fold.
- **Rule 25 cleared.** AV-AUC = 0.502; combined-frame transforms
  (everything in F3/F4/F5/F6/F7) are safe.
- **Q6 metric alignment.** All picks above train direct-binary
  objective.
- **Smoke gate first.** Per Rule 2, every pick must pass 1-fold /
  50k-rows smoke probe before 5-fold full.
- **K=4 absorption capacity is smaller than K=24.** Per A26
  (K=10 captures ~99% of K=27's LB value), retest field-state-style
  candidates at K=4+1 even if they were NULL at K=24+1.

## Sources

All five points above come from reading the following local files
(extracted to `/tmp/notebook_code/<slug>.py` for ease of inspection):

- `external/kernels/romanrozen/f1-pit-driver-race-year-encoding-0-95354.ipynb`
  (LB 0.95354, OOF 0.95357; ~80 features; LightGBM stack-meta;
  pseudo-labeling; DE-anchored blend).
- `external/kernels/f1-lap-by-lap-prediction-engine-v2/f1-lap-by-lap-prediction-engine-v2.ipynb`
  (svanikkolli v12; sections F3-F7 the differentiating content).
- `external/kernels/pit-or-stay-f1-strategy-1/pit-or-stay-f1-strategy-1.ipynb`
  (TE columns + lag features + 2023 anomaly handling).
- `external/kernels/s6e5-driver-s-high-driver-feature-eng/s6e5-driver-s-high-driver-feature-eng.ipynb`
  (compound expected-life + window-start lookups, driver-string
  parsing, z-score-by-group features).
- `external/kernels/ps-s6e5-hb1/ps-s6e5-hb1.ipynb` (LB 0.95400
  weighted blend; no novel FE — just `0.05·rohit + 0.10·mikhail +
  0.15·yekenot + 0.70·anthony`).
- `external/kernels/ps6e5-ensemble-0-95314-best-score/...ipynb`
  (LB 0.95314 weighted blend; no novel FE).
- `external/kernels/predicting-f1-pit-stops-blend/...ipynb`
  (anthonytherrien blend; no novel FE).
- `external/kernels/ps-s6-e5-realmlp-pytabkit/ps-s6-e5-realmlp-pytabkit.ipynb`
  (yekenot — already in our K=4 as `d17_h1d_yekenot_full`).
