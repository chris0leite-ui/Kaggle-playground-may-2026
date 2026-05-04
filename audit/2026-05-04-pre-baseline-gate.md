# Pre-baseline gate audit — 2026-05-04

Items 2 (schema) / 3 (target-rate) / 4 (group keys) of the pre-baseline understanding gate. Companion to `brief.md`, `comp-context.md` (prior_art / domain_notes / metric_notes), and the agent summaries.

## Item 2 — full schema

### train.csv
- shape: (439140, 16)

| col | dtype | n_unique | n_null | top-3 |
|---|---|---|---|---|
| id | int64 | 439140 | 0 | {'0': 1, '1': 1, '2': 1} |
| Driver | str | 887 | 0 | {'MAS': 1682, 'RAI': 1669, 'BAR': 1656} |
| Compound | str | 5 | 0 | {'MEDIUM': 211141, 'HARD': 170518, 'SOFT': 38744} |
| Race | str | 26 | 0 | {'Dutch Grand Prix': 24462, 'Mexico City Grand Prix': 23672, 'Pre-Season Testing': 22492} |
| Year | int64 | 4 | 0 | {'2023': 136147, '2024': 127110, '2025': 92894} |
| PitStop | int64 | 2 | 0 | {'0': 379365, '1': 59775} |
| LapNumber | int64 | 78 | 0 | {'1': 16558, '3': 13988, '6': 13859} |
| Stint | int64 | 8 | 0 | {'1': 216288, '2': 129536, '3': 69238} |
| TyreLife | float64 | 78 | 0 | {'6.0': 20663, '4.0': 20348, '12.0': 19945} |
| Position | int64 | 20 | 0 | {'4': 25267, '11': 25031, '12': 24937} |
| LapTime (s) | float64 | 37719 | 0 | {'83.939': 256, '84.924': 248, '84.813': 231} |
| LapTime_Delta | float64 | 57532 | 0 | {'0.0': 21473, '-0.0919999999999987': 1183, '-0.0979999999999989': 1042} |
| Cumulative_Degradation | float64 | 142701 | 0 | {'0.0': 10462, '-3.9339999999999975': 150, '-19.845': 142} |
| RaceProgress | float64 | 1898 | 0 | {'0.5': 8490, '0.1666666666666666': 5640, '0.25': 5591} |
| Position_Change | float64 | 37 | 0 | {'0.0': 137668, '1.0': 41603, '-1.0': 32075} |
| PitNextLap | float64 | 2 | 0 | {'0.0': 351759, '1.0': 87381} |

### test.csv
- shape: (188165, 15)

| col | dtype | n_unique | n_null | top-3 |
|---|---|---|---|---|
| id | int64 | 188165 | 0 | {'439140': 1, '439141': 1, '439142': 1} |
| Driver | str | 801 | 0 | {'MAS': 743, 'WEB': 741, 'GLO': 735} |
| Compound | str | 5 | 0 | {'MEDIUM': 90897, 'HARD': 72677, 'SOFT': 16615} |
| Race | str | 26 | 0 | {'Dutch Grand Prix': 10340, 'Mexico City Grand Prix': 10296, 'Hungarian Grand Prix': 9721} |
| Year | int64 | 4 | 0 | {'2023': 58160, '2024': 54532, '2025': 40125} |
| PitStop | int64 | 2 | 0 | {'0': 162525, '1': 25640} |
| LapNumber | int64 | 77 | 0 | {'1': 7096, '4': 6066, '6': 5982} |
| Stint | int64 | 8 | 0 | {'1': 93188, '2': 55330, '3': 29381} |
| TyreLife | float64 | 77 | 0 | {'6.0': 9015, '4.0': 8791, '12.0': 8391} |
| Position | int64 | 20 | 0 | {'4': 10942, '11': 10740, '12': 10710} |
| LapTime (s) | float64 | 30286 | 0 | {'84.924': 113, '84.813': 102, '83.939': 94} |
| LapTime_Delta | float64 | 40673 | 0 | {'0.0': 9297, '-0.0919999999999987': 470, '-0.0979999999999989': 434} |
| Cumulative_Degradation | float64 | 86823 | 0 | {'0.0': 4514, '-19.845': 70, '-3.9339999999999975': 60} |
| RaceProgress | float64 | 1556 | 0 | {'0.5': 3669, '0.1666666666666666': 2502, '0.25': 2358} |
| Position_Change | float64 | 37 | 0 | {'0.0': 59067, '1.0': 17631, '-1.0': 13737} |

### train ↔ test column diff
- in train, not in test: ['PitNextLap']
- in test, not in train: []

## CRITICAL — `PitStop` vs `PitNextLap` structural check

Both `PitStop` and `PitNextLap` exist in train; `PitStop` also in test. Hypothesis: `PitNextLap_N == PitStop_{N+1}` within a (Race, Driver) sequence (1-step-ahead lag).

- valid rows (non-last-lap): 424,198 / 439,140
- match rate `PitNextLap_N == PitStop_{N+1}`: **0.724381**
- weak relationship; not a simple lag.

### test — leakage scan
- test rows where `PitStop_{N+1}` exists in test under same (Race, Driver): 174,514 / 188,165
  (if non-zero, that fraction of the target is deterministically recoverable from test alone — competitors can solve it with a join.)

## Item 4 — GroupKFold candidate keys

- **Race**: train=26 groups (avg 16890 rows/group); test=26; overlap=26
- **Driver**: train=887 groups (avg 495 rows/group); test=801; overlap=801
- **Year**: train=4 groups (avg 109785 rows/group); test=4; overlap=4
- **Compound**: train=5 groups (avg 87828 rows/group); test=5; overlap=5
- **(Race, Driver)**: train=14942 groups; test=13651; overlap=13185
- **(Race, Driver, Stint)**: train=47841 groups; test=39297; overlap=35897

## Item 3 — per-feature target rate

### TyreLife (deciles)
- `(0.999, 3.0]`: target_rate=0.0275, n=51993
- `(3.0, 5.0]`: target_rate=0.0827, n=38645
- `(5.0, 8.0]`: target_rate=0.1213, n=57357
- `(8.0, 10.0]`: target_rate=0.1544, n=35186
- `(10.0, 12.0]`: target_rate=0.1755, n=38195
- `(12.0, 15.0]`: target_rate=0.2127, n=49118
- `(15.0, 18.0]`: target_rate=0.2480, n=42885
- `(18.0, 22.0]`: target_rate=0.2923, n=43864
- `(22.0, 27.0]`: target_rate=0.3288, n=38276
- `(27.0, 77.0]`: target_rate=0.3938, n=43621

### LapNumber (deciles)
- `(0.999, 4.0]`: target_rate=0.0539, n=54233
- `(4.0, 7.0]`: target_rate=0.0655, n=39461
- `(7.0, 11.0]`: target_rate=0.0857, n=47535
- `(11.0, 14.0]`: target_rate=0.1218, n=37360
- `(14.0, 19.0]`: target_rate=0.1569, n=45213
- `(19.0, 25.0]`: target_rate=0.2114, n=44878
- `(25.0, 32.0]`: target_rate=0.2952, n=42223
- `(32.0, 40.0]`: target_rate=0.3623, n=44006
- `(40.0, 49.0]`: target_rate=0.3398, n=45560
- `(49.0, 78.0]`: target_rate=0.3302, n=38671

### Stint (deciles)
- `(0.999, 2.0]`: target_rate=0.1839, n=345824
- `(2.0, 3.0]`: target_rate=0.2931, n=69238
- `(3.0, 8.0]`: target_rate=0.1448, n=24078

### RaceProgress (deciles)
- `(0.011800000000000001, 0.0526]`: target_rate=0.0554, n=44715
- `(0.0526, 0.1]`: target_rate=0.0688, n=43534
- `(0.1, 0.154]`: target_rate=0.0940, n=44789
- `(0.154, 0.208]`: target_rate=0.1242, n=42670
- `(0.208, 0.269]`: target_rate=0.1687, n=44890
- `(0.269, 0.359]`: target_rate=0.2301, n=42960
- `(0.359, 0.462]`: target_rate=0.3377, n=43927
- `(0.462, 0.577]`: target_rate=0.3722, n=44562
- `(0.577, 0.722]`: target_rate=0.3844, n=43545
- `(0.722, 1.0]`: target_rate=0.1558, n=43548

### Cumulative_Degradation (deciles)
- `(-274.565, -75.748]`: target_rate=0.3285, n=43915
- `(-75.748, -52.838]`: target_rate=0.2370, n=43914
- `(-52.838, -39.973]`: target_rate=0.2524, n=43931
- `(-39.973, -29.583]`: target_rate=0.2927, n=43896
- `(-29.583, -20.994]`: target_rate=0.2034, n=43938
- `(-20.994, -16.538]`: target_rate=0.1118, n=43890
- `(-16.538, -8.739]`: target_rate=0.1462, n=43923
- `(-8.739, -3.9]`: target_rate=0.1163, n=43938
- `(-3.9, 10.491]`: target_rate=0.1597, n=43884
- `(10.491, 2412.026]`: target_rate=0.1419, n=43911

### LapTime_Delta (deciles)
- `(-2403.896, -17.433]`: target_rate=0.2465, n=43916
- `(-17.433, -10.858]`: target_rate=0.2778, n=43951
- `(-10.858, -6.904]`: target_rate=0.2778, n=43875
- `(-6.904, -2.828]`: target_rate=0.3109, n=43937
- `(-2.828, -0.295]`: target_rate=0.2208, n=43930
- `(-0.295, -0.067]`: target_rate=0.0318, n=43893
- `(-0.067, 0.014]`: target_rate=0.0609, n=43994
- `(0.014, 0.931]`: target_rate=0.0324, n=43820
- `(0.931, 9.73]`: target_rate=0.2550, n=43910
- `(9.73, 2423.932]`: target_rate=0.2757, n=43914

### Position (deciles)
- `(0.999, 2.0]`: target_rate=0.1738, n=45268
- `(2.0, 4.0]`: target_rate=0.1855, n=49552
- `(4.0, 6.0]`: target_rate=0.1915, n=49243
- `(6.0, 8.0]`: target_rate=0.2012, n=49467
- `(8.0, 10.0]`: target_rate=0.2010, n=49116
- `(10.0, 11.0]`: target_rate=0.2038, n=25031
- `(11.0, 13.0]`: target_rate=0.2178, n=49787
- `(13.0, 15.0]`: target_rate=0.2249, n=47729
- `(15.0, 17.0]`: target_rate=0.2091, n=41655
- `(17.0, 20.0]`: target_rate=0.1760, n=32292

### Position_Change (deciles)
- `(-18.001, -5.0]`: target_rate=0.2446, n=46675
- `(-5.0, -2.0]`: target_rate=0.2787, n=62598
- `(-2.0, -1.0]`: target_rate=0.2454, n=32075
- `(-1.0, 0.0]`: target_rate=0.0575, n=137668
- `(0.0, 1.0]`: target_rate=0.1689, n=41603
- `(1.0, 3.0]`: target_rate=0.2870, n=49484
- `(3.0, 5.0]`: target_rate=0.3129, n=31660
- `(5.0, 18.0]`: target_rate=0.3104, n=37377

### Compound (5 levels)
- `HARD`: target_rate=0.3275, n=170518
- `SOFT`: target_rate=0.1935, n=38744
- `INTERMEDIATE`: target_rate=0.1523, n=17382
- `MEDIUM`: target_rate=0.1011, n=211141
- `WET`: target_rate=0.0251, n=1355

### Driver (887 levels)
- top-5: {'VET': 0.5655, 'MSC': 0.4732, 'HAD': 0.4621, 'STR': 0.4275, 'ANT': 0.4101}
- bottom-5: {'D609': 0.0, 'D608': 0.0, 'D607': 0.0, 'D606': 0.0, 'D601': 0.0}

### Race (26 levels)
- top-5: {'Chinese Grand Prix': 0.3886, 'Monaco Grand Prix': 0.3574, 'Spanish Grand Prix': 0.32, 'Bahrain Grand Prix': 0.2875, 'Belgian Grand Prix': 0.2804}
- bottom-5: {'British Grand Prix': 0.1335, 'Italian Grand Prix': 0.132, 'United States Grand Prix': 0.114, 'Miami Grand Prix': 0.1036, 'Mexico City Grand Prix': 0.0907}


---

## Item 5 — prior_art (web-research agent, 2026-05-04)

```yaml
prior_art:
  realmlp_pytabkit:
    title: "PS|S6|E5: RealMLP · PyTabKit"
    author: Vladimir Demidov (yekenot)
    votes: 56
    url: https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit
    cv: StratifiedKFold(n_splits=5, shuffle=True, random_state=42); split applied independently to comp data and original data, then concatenated per fold
    feature_engineering:
      - Drops Normalized_TyreLife from original before merging
      - Arithmetic interactions LapNumber/RaceProgress, TyreLife/LapNumber
      - Floor-binning of every numeric → categorical
      - Count encoding on cats and binned cats
      - KBinsDiscretizer (200 quantile bins) on RaceProgress
      - Race x Compound and Race x Year combo cats with TargetEncoder (cv=5, smooth='auto')
    model: RealMLP_TD_Classifier (PyTabKit) — 5-fold, n_ens=8, hidden=[512,256,128], silu, lr=0.03, wd=0.018
    oof_score: not stated (computed but not displayed in pulled cells)
    lb_score: not stated
    warnings: none flagged; relies on competition + original concatenated training without group-aware CV
  drivers_high_feature_eng:
    title: "[S6E5] Driver's High - Driver Feature Eng."
    author: Pilkwang Kim
    votes: 41
    url: https://www.kaggle.com/code/pilkwang/s6e5-driver-s-high-driver-feature-eng
    cv: StratifiedKFold(n_splits=5); counterfactual OOF "AUC delta" ladder, no GroupKFold
    feature_engineering:
      - Driver OHE/identity, driver-structure stats, driver_original_vocab match against original-data Driver set
      - Race x tyre algebra; compound structure (hardness, expected life, pit-window start)
      - Public-domain pit-window features; nonlinear interactions; frequency encoding; field-relative context
      - Target encoding for Driver-Compound, Race-Compound, Race-Stint, Compound-Stint
      - Logit-space Driver residual correction stage; original-data weight ablation
    model: LGBM ladder by default; CatBoost / XGB upgrade pass at end
    oof_score: 0.93208 (E0 raw+freq, 40k rows) climbing to ~0.948-0.960 with public reference stacks (XGB+CatBoost ensemble = 0.96076)
    lb_score: not stated; references public stacks at LB ~0.948-0.961
    warnings: explicitly cautions Driver signal needs counterfactual OOF; flags interaction-branch experiments may not improve baseline
  slowest_kaggle_pit_stop:
    title: "Slowest Kaggle Pit Stop"
    author: Marília Prata (mpwolke)
    votes: 37
    url: https://www.kaggle.com/code/mpwolke/slowest-kaggle-pit-stop
    cv: train_test_split(test_size=0.2, stratify=y, random_state=42) — single holdout, no K-fold
    feature_engineering:
      - LabelEncoder on object cols, fillna(-999) for cats, mean for floats
      - SimpleImputer(median) + StandardScaler for nums; SimpleImputer(most_frequent) for cats
      - No derived columns; EDA-only on Compound, numerical histograms, skewness, correlation heatmap
    model: LogisticRegression(max_iter=1000) only (xgb imported but not fit)
    oof_score: not stated
    lb_score: not stated
    warnings: notebook is EDA + single-holdout starter, not a competitive baseline
```

## Item 6 — domain_notes (domain-research agent, 2026-05-04)

```yaml
domain_notes:
  paragraph: |
    F1 pit calls balance tyre degradation against track position. Each
    compound (Soft/Medium/Hard) has a wear-rate curve; once lap times
    fall off the "cliff," a stop becomes net-positive. Teams undercut
    (pit early for fresh-tyre pace) or overcut (stay out as rivals
    pit) based on gap to cars ahead/behind. Safety Car / VSC windows
    halve pit-loss time, triggering opportunistic stops. In a ~60-lap
    race, 1-stop optima cluster lap 22-32; 2-stops near 18/40. Rain
    or drying track forces Inter/Wet crossover stops regardless of
    wear.
  citations:
    - "https://en.wikipedia.org/wiki/Formula_One_tyres — compound hierarchy and wear-cliff behaviour by Soft/Medium/Hard."
    - "https://www.formula1.com/en/latest/article/explained-the-undercut-and-overcut-strategies.6Nf3uPmeXOlJgCspljvWmd — undercut vs overcut definitions and gap dependence."
    - "https://en.wikipedia.org/wiki/Safety_car — SC/VSC reduces pit-loss, creating opportunistic stop windows."
    - "Heilmeier et al., 'A Race Simulation for Strategy Decisions in Circuit Motorsports,' IEEE ITSC 2018 — optimal 1-stop/2-stop lap-window modelling."
  column_mapping: |
    | column | likely driver | conf | reasoning |
    |---|---|---|---|
    | TyreLife | wear-cliff proximity | high | laps on current set drives degradation |
    | LapNumber | strategy-window timing | high | absolute lap aligns with 1/2-stop optima |
    | Stint | stint-count strategy | high | stint index encodes 1-stop vs 2-stop plan |
    | RaceProgress | normalised stop-window | high | lap/total handles variable race length |
    | Cumulative_Degradation | wear-cliff proximity | high | engineered degradation proxy |
    | LapTime_Delta | pace fall-off signal | high | slowing laps trigger pit decision |
    | Year | regulation/compound era | med | tyre construction changed across seasons |
    | Compound | compound life curve | high | Soft/Med/Hard sets baseline wear rate |
    | Driver | driver-style wear | med | individual wear/defending tendencies |
    | Race | track pit-loss & SC rate | high | circuit sets pit-loss + SC probability |
    | Position | strategic flexibility | med | leaders can defend / followers can attack |
    | Position_Change | recent overtakes / pit aftermath | med | non-zero often follows a stop |
    | PitStop | current-lap pit indicator | high | structural correlate of next-lap target (matched 0.724, not 1.0) |
```

## Item 7 — metric_notes (metric-research agent, 2026-05-04)

```yaml
metric_notes:
  imbalance_handling: |
    AUC is rank-based and invariant to class prior; scale_pos_weight / is_unbalance
    rescale leaf gradients but rarely lift AUC at prior=0.20 — they often hurt by
    distorting probability calibration without changing rank order
    (LightGBM docs, Parameters-Tuning §"For Better Accuracy"; XGBoost FAQ on
    scale_pos_weight). Verdict: leave default, sweep only if early OOF stalls.
    Caveat: helps PR-AUC / log-loss, not AUC (Davis & Goadrich, ICML 2006).
  calibration: |
    AUC is threshold-free and rank-only, so monotonic transforms are no-ops
    (sklearn docs, roc_auc_score). Calibration matters only if (a) we stack with
    a meta-learner that consumes probabilities (logistic blender benefits from
    isotonic), (b) downstream threshold needed. Plan: skip Platt/isotonic for
    rank-mean blends; apply isotonic only on stacker inputs if logistic meta used.
  cv_scheme: |
    Within-stint autocorrelation makes StratifiedKFold leak across (driver,race,lap)
    neighbours, inflating OOF AUC (sklearn user-guide §3.1.2 "Cross-validation
    iterators for grouped data"; cf. M5/Riiid winners using GroupKFold).
    R1 two-anchor at 439k rows is fine: StratifiedKFold seed=42 PLUS GroupKFold
    on Race (26 levels — 5-fold = ~5 races/fold) — gap >50bp between the two
    flags leakage. NOTE: schema audit shows (Race, Driver) overlap train/test
    is 96.6%, so within-train GroupKFold(Race) is the meaningful anchor.
  blend_topology: |
    Past Playground binary-AUC winners: rank-averaged LGB+XGB+CatBoost is the
    workhorse, +NN gives ~5-15bp on tabular-with-interactions
    (e.g. PS-S3E23 1st place "Software Defects" writeup; PS-S4E1 "Bank Churn"
    top-3 solutions on Kaggle Discussions). Linear/logistic stacker on rank
    features is the typical top-end. Plan: rank-mean GBDT trio first, NN later.
  gotchas: |
    (a) AUC = rank only, so blend at rank level (scipy.stats.rankdata) not raw
    probs — avoids one model dominating via scale.
    (b) Probe floor at n=37,633 public rows, prior 0.20: SE(AUC) ≈ sqrt(p(1-p)/
    (n_pos·n_neg)) ≈ 0.0014, so <14bp LB moves are noise (Hanley & McNeil 1982).
    (c) Tiny positive slices (~7.5k public positives) — single-fold spikes
    common; require both anchors to confirm.
  lgb_tactics: |
    400k-row binary-AUC Playground norms (from PS-S4E1, S3E23 public kernels):
    learning_rate 0.02-0.05, num_leaves 63-255, max_depth -1 or 8-12,
    min_data_in_leaf 100-500, feature_fraction 0.7-0.9, bagging_fraction 0.7-0.9,
    early_stopping_rounds 100-200, num_iterations 3-10k capped by ES.
    Citation note: exact ranges are kernel-empirical, not from a paper.
```

## Gate-cleared summary (for PI sign-off)

- **brief.md**: 149 lines verbatim from host (rules, evaluation, data, prizes,
  timeline). Original dataset's `Normalized_TyreLife` is host-removed — DO NOT
  reintroduce.
- **schema**: 16 train cols / 15 test cols (target = `PitNextLap`); 3 categoricals
  are `Driver` (887/801), `Compound` (5: MEDIUM/HARD/SOFT/INTERMEDIATE/WET),
  `Race` (26 grand prix names). All test-set Compound/Race/Year levels appear
  in train.
- **structural lag** (PitStop ↔ PitNextLap): 0.724 match — not deterministic;
  but `lead(PitStop)` within (Race, Driver) is recoverable for 92.7% of test
  rows → strong but bounded signal competitors can exploit.
- **CV plan (R1 two-anchor)**: (a) StratifiedKFold(seed=42) — public-notebook
  norm; (b) GroupKFold on `Race` (26 levels). >50bp gap = leakage flag.
- **prior_art**: 3 top notebooks read; none use group-aware CV; OOF range
  0.932 → 0.961 with public stacks.
- **domain_notes**: pit-call drivers cited (Wikipedia, F1.com, IEEE ITSC 2018);
  column mapping table built.
- **metric_notes**: AUC rank-only → skip class-weighting & calibration; blend
  at rank level; probe floor ≈14bp; LGBM hyperparam priors logged.
