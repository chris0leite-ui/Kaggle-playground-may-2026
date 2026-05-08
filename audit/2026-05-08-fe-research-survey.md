# FE research survey — comp-day 8 (2026-05-08 PM)

> Branch: `claude/research-feature-engineering-7oCmj`. Three parallel
> research subagents over (a) top public notebooks for s6e5, (b) similar
> competitions / F1-domain literature, (c) top-Grandmaster recipes.
>
> Triggered by: PI direction "research FE among the notebooks and write
> ups among similar competitions and contributions of the top
> competitors". Aligns with Rule 22 (public-notebook scan at every
> plateau) and Rule 7 (research before saturation). Top-5% gap is
> −5.4 bp on PRIMARY (LB 0.95351); top public-notebook blends sit at
> 0.95402–0.95412 i.e. 5–6 bp ahead of us.

## Headline

Public-notebook cluster ceiling is **0.95412** (rank-blend of
flexon_t85 + sohail + deeplearnerrr). Top single-model is **yekenot
RealMLP-PyTabKit n_ens=24 = 0.95356** — already in our K=4 PRIMARY
as `d17_h1d_yekenot_full`. The 5-bp gap to the public ceiling is
**not from a single missing feature**; it is the cumulative effect of
several recipes we have either not ported, partially ported, or tested
NULL on a *dense* pool but never on **K=4 sparse** (where absorption
capacity is smaller per EXPERIMENTS-NEXT EXP-1 reasoning).

The Heilmeier 2020 ablation (TUM Virtual Strategy Engineer) flags one
single-line domain feature — `remaining_pit_stops_proxy = expected_stops
- (Stint - 1)` — as the **most-impactful engineered feature in their
study**. It is fold-safe under Rule 24, takes <5 minutes CPU to probe,
and we have not tried it.

## Top public notebooks for s6e5 (mirrored locally at `external/kernels/`)

| Notebook | Author | Public LB | Local mirror |
|---|---|---:|---|
| `ps-s6-e5-realmlp-pytabkit` (RealMLP n_ens=24) | yekenot | 0.95356 | `external/kernels/ps-s6-e5-realmlp-pytabkit/` (VALIDATED) |
| `ps-s6e5-hb1` (rank blend of 4 CSVs) | nina2025 | **0.95400** (v2 → 0.95409) | `external/kernels/ps-s6e5-hb1/` |
| `predicting-f1-pit-stops-blend` | anthonytherrien | 0.95388 | `external/kernels/predicting-f1-pit-stops-blend/` |
| `f1-lap-by-lap-prediction-engine v2/v12` | svanikkolli | ~0.9530–0.9534 | `external/kernels/f1-lap-by-lap-prediction-engine-v2/` |
| `f1-pit-driver-race-year-encoding-0-95354` | romanrozen | OOF 0.95357 / blend 0.95354 | `external/kernels/romanrozen/` |
| `pit-or-stay-f1-strategy-1` | (anon) | ~0.953 | `external/kernels/pit-or-stay-f1-strategy-1/` |
| `s6e5-driver-s-high-driver-feature-eng` | (anon) | ~0.953 | `external/kernels/s6e5-driver-s-high-driver-feature-eng/` |

GitHub blend logs (Beiciccc) show ~0.95412 from rank-blending CSVs of
flexon_t85 + sohail + deeplearnerrr; those notebooks themselves are
not mirrored. Public discussion threads are JS-rendered and not
WebFetch-accessible.

## Already-tried (do not retry, with evidence)

- **Romanrozen 3-way Driver × Race × Year target-encoding**. Day-17 P1
  audit falsified this in our per-fold-aggregated-on-full-train
  setup; flagged in `external/kernels/ps-s6-e5-realmlp-pytabkit/VALIDATED.md`
  L62-64 as "leak-prone in per-fold-aggregated-on-full-train setups".
  Yekenot's 2-way `(Race, Compound)` + `(Race, Year)` TE inside
  `TargetEncoder(cv=5)` is the safe variant we already use.
- **Field-state cross-row aggregates per (Race, Year, LapNumber, ±Compound)**.
  Standalone +13.7 bp vs raw-14, but absorbed at K=24+1 hier-meta
  (Day-17 audit). **Not retested at K=4 sparse pool.**
- **Inter-stint memory features** (`prev_stint_length`, `prev_compound`,
  `prev_pit_lap_in_race`, `stints_completed_so_far`,
  `race_pit_count_so_far`). Standalone OOF 0.814, ρ 0.47 vs K=10, NULL
  Δ −0.011 bp at K=10+1 plain (EXP-3 in EXPERIMENTS-NEXT). Logit-level
  rank-lock, not rank-correlation.
- **Sequence / hazard / dual-head reframings** — LambdaRank per-stint
  (EXP-2), GRU sequence (EXP-1), stint-completion dual-head (EXP-4),
  pit-horizon 4-class, inv-laps-until-pit, reverse-cumulative pits.
  All NULL or LEAKY under Rule 24. Target reformulations are a
  leakage trap on this comp.
- **Yekenot recipe core**: floor-categorical, count-encoding,
  KBinsDiscretizer, 2-way combo categoricals via
  `TargetEncoder(cv=5, smooth='auto')`, original-data merge. Already
  the strongest base in our K=4 (`d17_h1d_yekenot_full`).

## Untried — ranked by expected value

EV ranking weights: (i) novelty given our internal ledger, (ii) Rule 24
fold-safety achievable, (iii) cost in CPU minutes for a 5-fold OOF
probe, (iv) precedent strength in source literature. **All cost
estimates are for a 5-fold OOF on the full 439k-row train**; smoke
gate (1-fold / 50k rows) per Rule 2 first.

### Tier A — high-EV, cheap (≤30 min CPU each)

1. **Heilmeier `remaining_pit_stops_proxy`** *(highest single-feature EV)*.
   `expected_stops_for(Race, Year) - (Stint - 1)` where
   `expected_stops_for(Race, Year)` is fold-safe groupby-max of `Stint`
   on training-fold rows only. **Source:** Heilmeier 2020 Applied
   Sciences 10(21):7805 (TUM Virtual Strategy Engineer) flags this as
   their most-impactful engineered feature. **Cost:** ~5 min CPU. **Why
   we missed it:** monotone counter, looks trivial; never explicitly
   tested. Probe at K=4+1 plain LR-meta gate.
2. **`cars_pitting_same_lap` field-state, fold-safe**. For each row at
   `(Race, Year, LapNumber)`, count of OTHER drivers in *training-fold*
   rows where `PitNextLap=1` at the same `(Race, Year, LapNumber)`.
   **Source:** Heilmeier 2020 (FCY/SC proxy) + svanikkolli v12 ("safety-
   car proxy"). **Cost:** ~10 min CPU. **Why interesting:** synthesises
   SC/VSC clustering + weather-driven compound-transition waves into
   one fold-safe signal. **Rule 24:** label-derived; mandatory per-fold
   refit; smooth with prior `m≈20` to handle small (Race, Year, Lap)
   cells.
3. **`driver_ahead_pitted_recently` undercut response**. For each row,
   look at the row at `Position - 1` in the same `(Race, Year,
   LapNumber)`; binary indicator if their `Stint` increased in the
   last 1-2 observed laps. **Source:** Frontiers AI 2025
   (`DriverAheadPit`); F1 strategy literature; svanikkolli v12.
   **Cost:** ~10 min CPU. **Why interesting:** rank-orthogonal to
   within-row features; direct mechanism for the dominant pit-decision
   driver in F1 strategy (TimeSHAP analysis in arXiv 2501.04068 ranks
   "Gap Ahead" #1).
4. **Compound expected-life lookup + tyre-overdue indicator**. Hard-code
   `{SOFT:15, MEDIUM:30, HARD:50, INTERMEDIATE:25, WET:20}`; derive
   `compound_tyre_norm = TyreLife / compound_max_life`,
   `tyre_overdue_norm = (compound_tyre_norm > 0.85)`. **Source:**
   Pirelli compound stint guidance; svanikkolli; romanrozen. **Cost:**
   ~3 min CPU. **Why interesting:** GBDT can't infer this normalisation
   from a single split when Compound has 5 levels and TyreLife is
   continuous; the closed-form ratio is a different function-form than
   one tree-split per Compound level.
5. **Race-progress fraction + pit-window flag**. `race_progress =
   LapNumber / max(LapNumber | Race, Year)`; `is_pit_window = 0.28 ≤
   race_progress ≤ 0.62`; `urgency_score = Cumulative_Degradation × (1
   − race_progress)`. **Source:** svanikkolli v12; romanrozen;
   pit-or-stay. **Cost:** ~3 min CPU. **Why interesting:** rectifies
   variable race length; `race_progress` differs from raw `LapNumber`
   when total race lap count varies across rounds.
6. **Conditional baseline hazard table at unusual granularities**.
   Fold-safe groupby-mean of target on `(Race, LapNumber, Compound)`
   and `(Compound, TyreLife)` cells, smoothed via Halford additive
   `μ = (n·x̄ + m·w)/(n+m)`, m≈300. **Source:** Halford blog;
   discrete-time hazard literature (Tutz BMC 2022). **Cost:** ~6 min
   CPU. **Why interesting:** adds finer hazard slices than our existing
   `(Race, LapNumber)` 2-way TE.

### Tier B — medium cost (30–90 min CPU)

7. **Cartesian groupby pair-aggregation sweep (Deotte S5E2 trick)**.
   For every (C1, C2) pair in `{Driver, Race, Compound, Year, Stint,
   LapNumber-bucket, TyreLife-bucket}` compute `groupby(C1)[C2].agg(
   ['mean', 'std', 'count', 'nunique', 'min', 'max', 'skew'])`. ~150-200
   features; feed to a single LightGBM, OOF-gate at K=4+1. **Source:**
   Chris Deotte S5E2 NVIDIA blog (1st place via 10k features of this
   shape). **Cost:** ~45 min CPU. **Why interesting:** systematic
   sweep we have NOT done at this fan-out; Deotte's published winning
   recipe.
8. **Pseudo-labeling à la romanrozen**. Soft test-row predictions where
   `P > 0.97` or `P < 0.015` get added to RealMLP training only.
   **Source:** romanrozen notebook; Deotte's pseudo-labeling QDA
   notebook. **Cost:** ~60 min GPU on Kaggle (RealMLP retrain).
   **Why interesting:** transductive-safe under our AV-AUC=0.502;
   never tried in our pipeline.
9. **Rolling 3-lap and 5-lap stats within (Race, Driver, Year, Stint)**.
   Of TyreLife and any cumulative-degradation numeric; rolling mean,
   max, std. Combined-frame safe per AV-AUC = 0.502. **Source:**
   svanikkolli v12; sports-analytics writeups. **Cost:** ~15 min CPU
   for the FE + 5-fold OOF. **Why interesting:** *within-stint
   momentum* is structurally different from the per-row features
   GBDT sees; possibly rank-orthogonal in a way that prev-stint
   features (which we tested NULL) were not.
10. **Cumulative pit-count features**. `cum_pits_in_race_so_far`,
    `cum_pits_for_compound_in_race`, fold-safe per Rule 24 (only
    training-fold target shifts). **Source:** survival-analysis
    literature (RFM, monotone counters). **Cost:** ~10 min CPU.

### Tier C — speculative (CPU ≤30 min, low-confidence)

11. **Genetic-programming feature search via gplearn**. `SymbolicTransformer`
    on top 8 most-important raw + engineered features, n_components=20,
    population=2000. **Source:** TPS-S5E10 1st place ("I think it was
    Genetic Programming"). **Cost:** ~30 min CPU. **Why speculative:**
    high variance; sometimes finds non-trivial polynomial combos but
    typically null on synthetic data with weak interactions.
12. **Float-digit extraction**. `((x * 10**k) % 10).astype(int)` for
    each continuous feature. **Source:** Deotte; Playground series
    folklore (synthetic CTGAN comps sometimes leak digit patterns).
    **Cost:** ~5 min CPU. **Why speculative:** s6e5 may not have CTGAN
    digit-leakage; quick to falsify.

### Tier D — closed externally per PI direction

- All historical-priors hard-joins (romanrozen `_pit_drv`, `_pit_ckt`
  from F1 1950–2022). External-data axis is closed per PI direction
  2026-05-08; FastF1 hard-join capped at 1.4% match earlier.
- Lap-time-delta degradation features (S3, S5, S6 academic sources) —
  require telemetry columns we do not have.

## External GitHub repos worth mining

A second pass via a fifth research agent surfaced four GitHub repos
that mirror or extend public-notebook work and are NOT auth-walled:

| Repo | URL | Best LB / OOF | What's in it |
|---|---|---|---|
| `baarzenzijncool/playground-series-s6e5` | `https://github.com/baarzenzijncool/playground-series-s6e5` | OOF 0.9486 (XGBoost only) | **`src/features.py` (~40 features) + `docs/eda_insights.md`** — the most complete domain-FE module we have access to |
| `leechanwoo-kor/kaggle-playground-series-s6e5` | `https://github.com/leechanwoo-kor/kaggle-playground-series-s6e5` | LB 0.94979 | EXP-001..007 ablation log; data-audit findings |
| `Beiciccc/predicting-f1-pit-stops` | `https://github.com/Beiciccc/predicting-f1-pit-stops` | LB **0.95412** | Public-notebook rank-blender + LB log |
| `oivler/kaggle-f1-pit-stops` | `https://github.com/oivler/kaggle-f1-pit-stops` | LB 0.94342 | Plain CatBoost; nothing novel |

### Untried features from `baarzenzijncool/src/features.py`

These add to the Tier-A list above and are concrete enough to copy
verbatim:

13. **Per-compound TyreLife cliff thresholds** *(static lookup, no
    Rule-24 risk if hard-coded; Rule-24-binding if re-derived per
    fold)*. The repo publishes empirically-derived medians and Q75s
    of `TyreLife` at pit:

    | Compound | median (pit) | Q75 (pit) |
    |---|---:|---:|
    | SOFT | 12 | 16 |
    | MEDIUM | 16 | 22 |
    | HARD | 20 | 27 |
    | INTERMEDIATE | 17 | 24 |
    | WET | 11 | 17 |

    Derived columns: `tyre_life_ratio = TyreLife / compound_median`,
    `tyre_life_over_cliff = (TyreLife > compound_q75)`,
    `tyre_life_remaining_to_cliff = compound_q75 - TyreLife`. Treat the
    lookup tables as **fold-derived** (refit per CV fold) to keep Rule
    24 satisfied. Replaces / supersedes Tier-A pick #4 (Pirelli
    population priors) — `baarzenzijncool` empirical thresholds are
    likely better calibrated to the synthetic data than Pirelli's
    population values.
14. **Two-window pit-window flags** (empirical, not the
    svanikkolli-fixed `[0.28, 0.62]`): `in_pit_window = (rp ∈ [.25, .45])
    | (rp ∈ [.52, .72])`, `too_late_to_pit = rp > 0.85`,
    `closing_lap_flag = rp > 0.90`. `race_phase ∈ {OPENING, EARLY, MID,
    LATE, CLOSING}` via `np.select`.
15. **`tyre_stress = tyre_life_ratio × |degradation_rate|`** with
    `degradation_rate = Cumulative_Degradation / max(TyreLife, 1)`.
16. **`undercut_zone = (Position > 6) & (rp ∈ [0.30, 0.70])`** —
    midfield × mid-race interaction flag.
17. **`laptime_delta_clipped = clip(LapTime_Delta, -20, 30)`** to
    suppress safety-car outliers (raw max 2,400 s); pit-vs-no-pit
    median delta is −4.3 s vs −0.14 s in the clipped view.
18. **`degradation_above_compound_q90`** + `degradation_vs_compound_median
    = Cumulative_Degradation - compound_median_cumulative_deg`. EDA-flagged
    but not actually implemented in either GitHub repo.
19. **`Year_2023_flag`** (NOT ordinal-encode Year). Pit rate 0.96% in
    2023 vs 26-30% other years — strong anomaly.

### Critical data-audit finding from `leechanwoo-kor`

**The synthetic data violates time order in ~71% of (Driver, Race,
Year) groups when sorted by `LapNumber`.** Sort by `(Stint, TyreLife)`
instead. This is independent corroboration of our GRU / LambdaRank /
sequence-model nulls — the sequence is broken in the synth, so
sequence models *cannot* extract a within-stint trajectory. Any
within-stint rolling-stats feature (Tier-B pick #9) MUST sort by
`(Stint, TyreLife)`, NOT by `LapNumber`.

### Independent CV-scheme confirmation from `leechanwoo-kor`

| CV scheme | OOF | LB gap |
|---|---:|---:|
| Random 5-fold | 0.94906 | 0.00032 |
| GroupKFold (Year, Race) | 0.93263 | ~0.016 |

This independently confirms our R1/Strat-as-LB-proxy choice and the
~3 bp OOF→LB gap calibration. GroupKFold over-penalises by ~16 bp;
Stratified is the right anchor.

## Prior-comp 1st-place writeup synthesis

A sixth focused agent attempted to read the actual *bodies* of recent
TPS 1st-place writeups (S6E1, S6E2, S6E3, S6E4 irrigation, S5E10,
S5E11, S5E12). Kaggle SPA renders behind JS for unauth fetches;
Wayback Machine egress-blocked, archive.ph CAPTCHA-walled, Google /
Bing cache returned 404. The agent recovered content via two indirect
paths:

1. **NVIDIA developer blogs** — KGMON team's S6E3 win + Grandmasters
   Playbook + cuDF FE + cuML stacking blogs, written by Chris Deotte
   et al., are open-access and substantively describe their methods.
2. **BlamerX/Kaggle-Playground-Predection-Competition** GitHub agent
   logs — explicitly quote and port from the writeup bodies of
   S6E1/S6E2/S6E3/S6E4 with `Source: SXEY 1st place` attributions.

**Honesty caveat:** the agent did NOT read the literal HTML of the
S6E1/S6E2/S5E10/S5E11/S5E12/S6E4 1st-place writeup bodies. The
recurrence of patterns across the indirect sources is the
confidence anchor — multiple independent ports and quotes converge
on the same 6 patterns.

### Patterns recurring across ≥3 1st-place writeups (high confidence)

1. **Inner-fold target encoding on bigram / trigram categorical
   interactions.** S6E2 1st (`Contract×IS×OnlineSecurity` was the
   #1 importance feature at 0.155), S5E11 1st (multi-target encoding
   variant), S6E4 1st (145 pairwise TE features), Deotte across multiple
   comps. Nested-fold (Deotte-style) to prevent leakage. **This is
   the highest-confidence port we have evidence for.** Adjusts our
   existing yekenot 2-way TE on `(Race, Compound)` + `(Race, Year)` to
   include 3-way and exhaustive 2-way over more pairs.
2. **Digit-level / modulo / rounding features on numericals.** S5E11
   1st, S6E2 1st (`tenure % 10`, decimal-extraction from
   `MonthlyCharges`), S6E4 1st (`Field_Area % 1` "magic anchor"
   detecting synthetic-generation artifacts). Reproduced with
   **+0.00028 OOF** in independent S6E3 port. **Particularly relevant
   for synthetic data** — CTGAN / similar generators leak digit
   patterns in low-order decimals.
3. **Original (parent) dataset injection as anchor.** S6E2, S6E4,
   S5E11. Target-encode against the source dataset, weight the anchor
   ~0.30-0.40 in the final blend. We already do this via the yekenot
   recipe's `aadigupta1601` original-data merge — confirms the pattern.
4. **Multi-level / many-model stacking with hill-climbing weights.**
   S6E3 KGMON (4-level stack of 150 models from 850 candidates), S5E12
   1st ("Hill Climbing + Ridge Ensemble"), S6E2 1st ("Gap-Aware
   Blend"). Hill-climb chosen over linear/NN stackers when bases are
   highly correlated (avg ρ=0.997 in KGMON). We use forward-greedy
   K=4; **un-explored adjacent: hill-climb with re-weighting (not just
   inclusion) on the K=4 logit pool**.
5. **Pseudo-labeling / knowledge distillation.** S6E1 1st (Deotte
   quote: "we don't benefit from the fake targets; we benefit from
   the new real features"), ported into S6E2 (V58/V59) and S6E3
   (V53-V57). Self-distillation works for 1-2 iterations only before
   degrading. **Already in Tier-A #8 (romanrozen variant).**
6. **Diversity discipline — depth-2 stumps + OHE + LR scaling.** S6E2
   1st champion class. Beats deeper single models. Inclusion of
   weaker-but-uncorrelated bases (RealMLP, NODE, KAN, RGF) for
   ensemble lift.

### Outlier ideas worth testing on s6e5 (single-comp evidence)

7. **Genetic programming feature discovery (S5E10 1st).** `gplearn`
   to auto-discover arithmetic combinations on continuous columns
   (LapNumber, TyreLife, Stint, RaceProgress). Strong fit when synth
   has hidden generation rules. **Already Tier-C #11; S5E10 1st-place
   evidence elevates this from "speculative" to "moderate-EV" if
   we have the CPU budget.**
8. **Decimal-modulo "magic anchors" on numericals (S6E4 trick).**
   `LapNumber % 1`, `RaceProgress * 1000 % 10`, `TyreLife % 1`.
   **Already Tier-C #12 (float-digit extraction); evidence
   strengthens this from "speculative" to "moderate-EV"** because
   S6E4 winner explicitly used `Field_Area % 1` as a magic anchor.
9. **DVAE / DAE bottleneck features (S6E2 1st).** Denoising
   variational autoencoder on train+test concatenated; AV-AUC=0.502
   on s6e5 means combined fit is safe. 64-dim latent as auxiliary
   input to GBDT, or as a pure diversity model. NB: this **failed
   when ported to S6E3** (bottleneck destroyed signal in highly-
   redundant input space). We already tested simple DAE; DVAE
   specifically is untried.
10. **Recursive knowledge distillation, 1-2 iterations only (S6E1
    1st).** Train s6e5 best XGB on pseudo-labels of test rows where
    `P > 0.98` or `P < 0.02` with weight 0.5. Per Deotte: more than 2
    iterations degrade.
11. **`baseline=` margin transfer (S6E1).** Convert XGB log-odds to
    CatBoost `Pool(baseline=...)` so CatBoost trains on the residual
    margin. Cheaper than full stacking, untried in our pipeline.
12. **Per-(Driver, Race, Year) "AllCat" string concatenation (S6E2 /
    Deotte).** Treat the tuple as a single high-cardinality
    categorical; inner-fold TE. The 71% time-shuffle within (Driver,
    Race, Year) groups means group ID itself is leak-free. **NB:
    Day-17 P1 audit falsified the romanrozen Driver × Race × Year
    TE in our per-fold-aggregated-on-full-train setup. The S6E2 /
    Deotte variant uses NESTED-FOLD TE which is the safe form.**
    Worth re-attempting with strict nested-fold sklearn
    `TargetEncoder(cv=5)`.

## What the leader is probably doing

Synthesising the three agents' findings: the most likely mechanism for
the 0.95476 leader is **a fold-safe field-state composite** — same-lap
pit clustering (SC/VSC proxy) + 1-2-lap-lagged "rival-just-pitted"
indicator, layered on top of a Heilmeier-style `remaining_pit_stops`
counter and a per-Compound `tire_age_progress` normalisation. None of
these are individually flashy; their combined lift is plausibly the
~12 bp gap. The leader's likely *not* using a structurally different
model class — every public top-cluster notebook uses the same
LGBM+XGB+CB+RealMLP stack we do.

## Implementation sketch — top-3 picks for next session

```python
# Pick 1 — Heilmeier remaining_pit_stops_proxy (Tier A #1)
# Fold-safe under Rule 24: expected_stops fitted on train fold only.
def add_remaining_pit_stops(train_fold, val_fold):
    expected_stops = (train_fold.groupby(['Race', 'Year'])['Stint']
                      .max().rename('expected_stops'))
    train_fold = train_fold.merge(expected_stops, on=['Race', 'Year'], how='left')
    val_fold = val_fold.merge(expected_stops, on=['Race', 'Year'], how='left')
    train_fold['remaining_pit_stops_proxy'] = (
        train_fold['expected_stops'] - (train_fold['Stint'] - 1))
    val_fold['remaining_pit_stops_proxy'] = (
        val_fold['expected_stops'] - (val_fold['Stint'] - 1))
    return train_fold, val_fold

# Pick 2 — cars_pitting_same_lap field-state (Tier A #2)
# Fold-safe: groupby aggregates use train-fold target only.
def add_cars_pitting_same_lap(train_fold, val_fold, smooth_m=20):
    g = (train_fold.groupby(['Race', 'Year', 'LapNumber'])['PitNextLap']
         .agg(['sum', 'count']).reset_index())
    prior = train_fold['PitNextLap'].mean()
    g['cars_pitting_rate'] = (
        (g['sum'] + smooth_m * prior) / (g['count'] + smooth_m))
    train_fold = train_fold.merge(g[['Race','Year','LapNumber','cars_pitting_rate']],
                                  on=['Race','Year','LapNumber'], how='left')
    val_fold = val_fold.merge(g[['Race','Year','LapNumber','cars_pitting_rate']],
                              on=['Race','Year','LapNumber'], how='left')
    return train_fold, val_fold

# Pick 3 — compound_tyre_norm + tyre_overdue (Tier A #4)
# No label dependency; safe to compute combined-frame.
COMPOUND_LIFE = {'SOFT': 15, 'MEDIUM': 30, 'HARD': 50,
                 'INTERMEDIATE': 25, 'WET': 20}
def add_compound_tyre_norm(df):
    df['compound_max_life'] = df['Compound'].map(COMPOUND_LIFE)
    df['compound_tyre_norm'] = df['TyreLife'] / df['compound_max_life']
    df['tyre_overdue_norm'] = (df['compound_tyre_norm'] > 0.85).astype(int)
    return df
```

Suggested smoke order: Pick 3 (no label, 3 min) → Pick 1 (5 min) →
Pick 2 (10 min). Each as a single-feature add-on into a re-fit
yekenot-recipe LGBM, OOF-gate at K=4+1 plain LR-meta.

## Caveats and Rule-discipline reminders

1. **Rule 24 is binding for picks 1, 2, and 6.** Any per-group
   aggregate of the label MUST refit per CV fold using training rows
   only. Public notebooks (svanikkolli v12, romanrozen) compute these
   on the *full* train set — that's the exact pattern that collapsed
   our reverse-cumulative / inv-laps / pit-horizon under Day-17
   strict-OOF audit (88-100% collapse).
2. **Rule 25 cleared.** AV-AUC = 0.502 means combined-frame transforms
   are safe. Compound-life lookup, race-progress fraction, and rolling
   stats can be computed across train+test concatenated.
3. **Rank-lock prior.** EXP-1/2/3/4 closed NULL despite predictions
   with ρ down to 0.41 vs K=10. Stack-add gates at K=10+1 absorb most
   genuinely-orthogonal signals. **Test at K=4+1** instead — sparse
   pool has lower absorption capacity per A26 (K=10 captures ~99% of
   K=27's LB value).
4. **Q6 metric alignment.** PitNextLap binary → AUC. All picks above
   train direct-binary objective, matching the LB metric. No risk of
   the "minimised-RMSE-but-evaluated-on-AUC" failure mode.
5. **Smoke gate first.** Per Rule 2, every pick must pass 1-fold /
   50k-rows smoke probe before 5-fold full. Kill if smoke OOF gain
   < 1 bp; full-data gain almost never exceeds the smoke gain by
   more than 2×.
6. **PI ask before submit.** Every LB submission single-shot, Rule 1
   + Rule 26.

## Sources

Public notebooks (local mirrors at `external/kernels/`):
- yekenot RealMLP-PyTabKit, romanrozen, svanikkolli v12, anthonytherrien
  blend, nina2025 hb1, pit-or-stay, driver-feature-eng.

Academic / industry literature:
- Heilmeier et al. 2020, "Virtual Strategy Engineer", *Applied Sciences*
  10(21):7805.
- Thomas et al. 2025, "Explainable RL for F1 Race Strategy", arXiv
  2501.04068 (TimeSHAP analysis).
- Ahmed et al. 2025, *Frontiers in AI* 8:1673148 (Bi-LSTM pit
  decision support).
- TUMFTM/race-simulation (GitHub, 2018–2023; FCY-adjusted tyre age).
- arXiv 2512.00640 (state-space tyre-degradation, F1).

Top-Grandmaster recipes:
- Chris Deotte, NVIDIA cuDF blog (TPS-S5E2 1st), NVIDIA cuML stacking
  blog (TPS-S5E4 1st), "Grandmasters Playbook" (NVIDIA, 7 techniques).
- KGMON team (Deotte, Puget et al.), NVIDIA blog "Winning a Kaggle
  Competition with Generative AI–Assisted Coding" (TPS-S6E3 1st).
- `cdeotte/KGMON-Playbook-2026` GitHub (notebooks 02_Feature_Engineering,
  04_Stacking).
- `BlamerX/Kaggle-Playground-Predection-Competition` GitHub (agent
  logs porting / quoting from S6E1, S6E2, S6E3, S6E4 1st-place
  writeups with explicit attribution).
- `AdilShamim8/Kaggle_Competitions` GitHub (Top-1% reproduction of
  S5E10 Road Accident Risk; confirms genetic-programming features).
- Max Halford, "Target encoding done the right way" (closed-form
  additive smoothing).
- TPS-S6E2 / S6E3 / S6E4 / S5E10 / S5E11 1st-place writeup titles
  (bodies JS-rendered, accessed indirectly via NVIDIA blogs and
  GitHub agent logs as above).
- Medium: Sanjay Bista S5E11 loan-prediction class-imbalance writeup.

GitHub blend logs:
- Beiciccc/predicting-f1-pit-stops (experiment_log.md, leaderboard_history.csv).

## Next-action recommendation

Smoke-gate Picks 3 → 1 → 2 in sequence; each ≤10 min CPU. If any
single pick passes K=4+1 plain LR-meta gate by ≥ +0.5 bp, promote to
5-fold full-data probe and stack-add candidate. If all three NULL, the
F1-domain feature axis is empirically closed and we accept the wrap-up
posture per `state/hypothesis-board.md` Open Priority #4.

PI sign-off needed before any compute spend — present this audit's
top-3 picks with the cost/EV table, ask whether to proceed with smoke
or defer to wrap-up.

**Revised top-3 after sixth-agent prior-comp synthesis:**

The prior-comp synthesis surfaced two patterns that recur across ≥3
TPS 1st-place writeups (the highest-confidence ports we have
evidence for) which beat the F1-domain candidates on prior strength.
Final top-3:

1. **Bigram / trigram inner-fold target encoding sweep** (writeup
   pattern #1; S6E2 1st + S5E11 1st + S6E4 1st all use it). For
   pairs and triples in `{Driver, Race, Compound, Year, Stint}`,
   build string-concat columns and fit `TargetEncoder(cv=5)`. ~12
   min CPU. The Day-17 P1 falsification was the *non-nested* variant;
   strict sklearn `TargetEncoder(cv=5)` is the safe nested-fold form.
2. **Heilmeier `remaining_pit_stops_proxy`** (single-line domain
   feature; Tier-A #1; TUM 2020 ablation flags as #1 most-impactful
   engineered feature). ~5 min CPU.
3. **`baarzenzijncool` per-compound TyreLife cliff features** (item
   #13; ~5 min CPU + per-fold refit). The empirical thresholds are
   calibrated to s6e5 synth.

Reserve runner-ups (one slot each if any of top-3 lifts):
- **Decimal-modulo "magic anchors"** (writeup pattern #8). `LapNumber
  % 1`, `RaceProgress * 1000 % 10`, etc. ~3 min CPU. Cheap,
  high-variance probe; explicit S6E4 1st-place precedent.
- **`cars_pitting_same_lap` field-state, fold-safe** (Tier-A #2;
  ~10 min CPU).

Total smoke + 5-fold OOF probe budget: ~25-35 min CPU for the top-3.
Within the Rule-2 envelope.

