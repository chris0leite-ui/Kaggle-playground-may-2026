# Round 9 execution — 2026-05-18 (Day-19)

## Plan reference

`/root/.claude/plans/cached-frolicking-rivest.md` (R9 dual-track:
NB4 + C1). PI-approved at session-start. R8 had falsified the
multi-seg sweep; EOD strategy-critic verdict: priority-queue
Σ × P(real) ≈ 0.058 bp vs 1.6 bp gap → structural shortfall.
Posture pivot scout: +1 bp single-mechanism class OR hedge-prep.

R8 PM research-loop (`audit/research/2026-05-18-research-pm-addendum.md`)
top-3 dedup result (Explore agents at R9 session-start):
- **C4 UID magic-features** smoke-FAILED today at −16.2 bp
  (`audit/2026-05-18-tier-a-batch.md:92-102`) — SKIP.
- **Competitor pit cascade** tested in 3 prior variants
  (A3-1 / F3 / graph-pit-pressure all absorbed) — SKIP.
- **NB4** confirmed NOVEL — go.

PI chose dual-track: NB4 (cheap TE-as-base diagnostic) + C1
(external-info structural lever).

## Phase A — NB4: Per-(Compound × Stint) target-mean as BASE

Script: `scripts/probe_r9_compound_stint_te_base.py`.
Uses `cv_target_encode` from `scripts/p1_features.py:302-333`
(fold-safe; refits stats per fold's training rows).
Feature matrix: raw 14 + 1 TE column. 5-fold StratifiedKFold seed 42.
LGBM `LGB_PARAMS` standard (probe_r4 template).

### Standalone

Fold AUCs: 0.94762 / 0.94757 / 0.94863 / 0.94774 / 0.94891
OOF AUC: **0.94850** fold-std 0.00077 wall 150.6 s.
**G1 PASS** (≥ 0.948 yekenot-level).

Diagnostic on Strategy-critic Section 1 weak segment
(MEDIUM × Stint 2, 25 363 rows): **0.8849** vs PRIMARY R7.1 0.8975.
NB4 standalone is WORSE on that segment than the stacked PRIMARY —
confirms the segment is at noise floor for row-level features
(re-falsified the segment-FE thesis a second time, after the
session-start 5-min specialist probe).

### K=14 + Path-B DriverClass×Stint τ=100k

`python scripts/build_K13_pathb_multiseg.py --segs driverclass_stint
--tau 100000 --extra-bases NB4_compound_stint_te`. Wall 105.5 s.

K=14 OOF: **0.954469** (Δ vs R5.2 +0.084 bp; **Δ vs R7.1 PRIMARY
0.954471 = −0.022 bp REGRESSION**).
ρ_test vs R5.2: 0.999769 (TIE_ZONE band).

**G2 FAIL** (Δ < +0.01 bp). KILL.

### Phase A interpretation

The TE-broadcast-at-base-layer mechanism is **absorbed at the
K=13+Path-B meta layer**. Path-B's per-(DriverClass × Stint)
shrinkage already extracts the per-(Compound, Stint) signal at the
META layer; the same signal injected at the BASE layer (NB4) is
redundant and slightly dilutes the rank-lock. The segmentation-as-
base axis (R9 priority-queue item 1 from postmortem) is **closed**.

## Phase B — C1: External per-Race scalars (Aadigupta)

Script: `scripts/probe_r9_race_external_scalars.py`.
Source: `data/original/f1_strategy_dataset_v4.csv` (101 371 rows,
28 Race levels, FORBIDDEN col `Normalized_TyreLife` dropped).
26/26 s6e5 Race levels overlap — no fallback needed.

Per-Race scalars (5, FEATURE-derived, not target-derived):
`lap_time_median_race`, `lap_time_std_race`, `cum_deg_max_race`,
`pos_chg_std_race`, `race_len_max_lap`.

Feature matrix: raw 14 + 5 ext scalars (broadcast by Race name).
5-fold StratifiedKFold seed 42. LGBM standard.

### Standalone

Fold AUCs: 0.95027 / 0.94811 / 0.94912 / 0.94839 / 0.94921
OOF AUC: **0.94902** fold-std 0.00075 wall 162.1 s.
**G1 PASS** (≥ 0.948); ~5 bp better standalone than NB4.

Per-Race diagnostic (Strategy-critic Section 1 weak races):
- Spanish GP : 0.9082 (PRIMARY 0.9176 K=13 stack)
- Bahrain GP : 0.9143 (PRIMARY 0.9215)
- Emilia GP  : 0.9100 (PRIMARY 0.9224)
- Saudi Ar GP: 0.9711 (PRIMARY 0.9738)

C1 standalone lifts NOTHING per-race vs PRIMARY (expected; standalone
single LGBM vs K=13 stack). Diagnostic is for asymmetric meta-add
expectation only.

### K=14 + Path-B DriverClass×Stint τ=100k

`python scripts/build_K13_pathb_multiseg.py --segs driverclass_stint
--tau 100000 --extra-bases C1_race_external`. Wall 92.8 s.

K=14 OOF: **0.954466** (Δ vs R5.2 +0.061 bp; **Δ vs R7.1 PRIMARY
−0.045 bp REGRESSION**).
ρ_test vs R7.1 PRIMARY: **0.999981** (deep TIE_ZONE).

**G2 FAIL** (Δ < +0.01 bp). KILL.

### Phase B interpretation

External-data injection — the only structural lever left after the
EOD strategy-critic Section 5 verdict — **also absorbs at the
K=13+Path-B meta layer**. C1 regressed MORE than NB4 (−0.045 vs
−0.022 bp). Likely cause: yekenot's existing 6 `TE_CONFIGS` in
`scripts/p1_features.py:336-342` already touch Race in 5 of 6
configs (`te_drv_race_yr`, `te_drv_race`, `te_race_comp`,
`te_race_yr`, `te_drv_race_comp`) — the existing pool absorbs
per-Race signal density that C1's scalars duplicate, then the
K=14 LR-meta over-fits the now-correlated 14th column and
slightly dilutes the rank-lock.

## R9 net verdict

**Rank-lock at K=13+Path-B is structurally confirmed across
three axes**:

1. Operator family — D15 DAE (R7 swap-noise), v2 transformer
   (R6 seq2seq) — all absorbed or G2-fail.
2. Mechanism class — segment-FE (R4 G2-fail), HMM solo (R4 null),
   pit cascade (3 variants all null), NB4 TE-base (R9) all absorbed.
3. Data class — C1 external Aadigupta scalars (R9) absorbed.

This is strong evidence the K=13+Path-B pool has reached its
**structural ceiling** for ROW-LEVEL features. The remaining
mechanism-expansion candidates lie OUTSIDE row-features:

- **A1 sequence-level**: real seq2seq transformer on
  per-(Driver, Race) lap sequences (HMM K=13 base was Baum-Welch
  one-shot on 4 states; full seq2seq untried).
- **Graph mechanism**: edge-attributed GNN on (Race, Driver, Lap)
  with competitor edges within (Race, Lap).
- **Mixture / survival**: explicit time-to-pit hazard with
  driver-style mixture component.

PI directive: **hold all 3 daily slots; pivot to mechanism expansion
for R10**. Comp Day-19 of 32 (13 days left).

## Numerical summary

| Probe | Standalone OOF | Wall | K=14 OOF | Δ vs R7.1 | ρ vs R7.1 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| NB4 (Compound×Stint TE base) | 0.94850 | 150 s | 0.954469 | −0.022 bp | 0.99977 | NULL |
| C1 (Aadigupta per-Race scalars) | 0.94902 | 162 s | 0.954466 | −0.045 bp | 0.99998 | NULL |

PRIMARY unchanged: **R7.1 K=13 + Path-B DriverClass × Stint τ=100k
LB 0.95389 OOF 0.954471**.

Slot use: 0 of 3.

## Artifacts saved

- `scripts/probe_r9_compound_stint_te_base.py` (NB4 builder)
- `scripts/probe_r9_race_external_scalars.py` (C1 builder)
- `scripts/artifacts/oof_NB4_compound_stint_te_strat.npy` + test
- `scripts/artifacts/oof_C1_race_external_strat.npy` + test
- `scripts/artifacts/oof_K14_pathb_driverclass_stint_tau100000.npy`
  (last write = C1 run; NB4 K=14 OOF is in the JSON only)
- `audit/2026-05-18-round-9-K14-nb4.{log,json}`
- `audit/2026-05-18-round-9-K14-c1.{log,json}`
- `submissions/submission_K14_pathb_driverclass_stint_tau100000.csv`
  (last write = C1; not submitted)
- `scripts/build_K13_pathb_multiseg.py` modified — added
  `--extra-bases` CLI flag for K-extension experiments.

## Frictions surfaced

1. **ISSUES.md is Day-19-era stale** — does not list R6/R7/R8/R9
   leaves. Rule 18 claim convention can't be followed without
   leaf entries. Need EOD ISSUES.md refresh.
2. **`build_K13_pathb_multiseg.py` ref-baseline is R5.2** (hardcoded
   `K13_seghmm_pathb_tau100000`). Δ vs R7.1 PRIMARY requires manual
   computation. Friction note: parametrise `--ref-oof` for honest
   comparison in future R10+ runs.
3. **Submission file name collision**: `submission_K14_pathb_*`
   was overwritten by C1 run after NB4 run; only one survives on
   disk. Not critical (neither was submitted) but worth a name
   suffix for any future K=14 multi-probe day.

## R10 priority queue (mechanism expansion)

A. **A1 seq-class**: per-(Driver, Race) lap sequence transformer
   (small 2-3 layer attention + LSTM hybrid). Predicts PitNextLap
   as a sequence task. ~2 hr Kaggle T4. Genuinely orthogonal axis.
B. **Graph mechanism**: (Race, Lap) per-row graph with competitor
   edges; LightGCN / GAT 2-layer. ~3 hr local CPU or Kaggle T4.
C. **Survival**: Cox PH on stint-life with TyreLife / Compound
   covariates → hazard score as base. ~30 min CPU. Fastest of
   the three but lowest novelty.

Out of scope: any further row-feature TE / external-scalar variants
(R9 closed that axis).
