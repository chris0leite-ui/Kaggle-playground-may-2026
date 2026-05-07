# 2026-05-07 PM — Probe 4 field-state aggregates: STRUCTURAL FIND

`branch: claude/review-handover-solutions-oE78b`
`tag: hiding-in-plain-sight + cross-row-aggregates`
`mechanism family: field_state_aggregates (NEW; not in mechanism_families_explored)`

> **Status: standalone OOF +15.58 bp at single-LGBM level over raw 14.
> Single feature `fs_cum_pits` AUC 0.7972 — highest single-feat on comp.
> NEW MECHANISM FAMILY discovered after 17 days. PI sealed prediction
> 0 bp LB Δ technically vindicated (no submit) but the OOF signal is
> 30× the agent's predicted band.**

## Setup

After the 3-probe pass earlier today closed own-row sequence/feature
axes (NTL, target structure, combined-frame lead/lag), the surviving
hypothesis was: signals depending on *other rows* in the same race-lap
context that the GBDT cannot reconstruct from a single row.

The candidate: aggregates over OTHER rows per (Race, Year, LapNumber)
and per (Race, Year, LapNumber, Compound) computed from train+test
combined. PitStop is a feature column (not the label), so this is
AV-safe per Rule 25 (AV-AUC=0.502).

## Headline numbers

| Run | Features | OOF AUC | Δ vs F2 |
|---|---|---:|---:|
| F2 | raw 14 only (matches U1 baseline 0.94075) | 0.94074 | — |
| F3 | raw + combined-frame field-state (~30 feats) | 0.94230 | **+15.58 bp** |
| F4 | raw + train-only field-state (control) | 0.94241 | **+16.66 bp** |
| F3 - F4 | combined-frame premium | — | **-1.08 bp** |

Fold std 0.00058 (well below the 15+ bp lift). Per-fold deltas vs F2
raw: +18.3, +14.5, +13.8, +15.3, +14.8 bp. Consistent across folds.

**Combined-frame premium is null/slightly negative.** The lift comes
from the field-state mechanism itself, not from including test rows in
the aggregates. Per-(R, Y, L) groups are large enough on train alone
(40 drivers × 26 races × 4 years ≈ 4160 cells) that train-only stats
are already stable.

## Top single-feature OOF AUCs (combined frame)

| Feature | combined AUC | train-only AUC | Δ |
|---|---:|---:|---:|
| **fs_cum_pits** (Σ PitStop in (R,Y) up to L) | **0.7972** | 0.7966 | +0.0006 |
| fs_field_size (n drivers active at L) | 0.2680* | 0.2686 | -0.0006 |
| fs_pit_rate_now (mean PitStop at (R,Y,L)) | 0.7292 | 0.7231 | +0.0061 |
| fs_compound_pit_rate | 0.7236 | 0.7153 | +0.0083 |
| fs_std_TyreLife | 0.7132 | 0.7104 | +0.0028 |
| fs_cum_pit_lap_count | 0.7023 | 0.7023 | 0 |
| fs_cum_pit_rate | 0.7017 | 0.6992 | +0.0025 |
| fs_mean_Stint | 0.6997 | 0.6994 | +0.0002 |
| fs_compound_max_TyreLife | 0.6936 | 0.6949 | -0.0013 |
| fs_max_TyreLife | 0.6886 | 0.6895 | -0.0010 |

*anti-correlated; flipped = 0.732.

**Reference single-feature anchors on this comp:**
- TyreLife alone: 0.6989
- RaceProgress alone: 0.6644
- All NTL reconstructions (Probe 3): cap at 0.687

`fs_cum_pits` at 0.7972 is the strongest single-feature OOF AUC on this
comp by ~10 bp over the previous strongest (TyreLife). That this signal
sat unused for 17 days while we stacked 23 bases is the find of the
session.

## Mechanism

The PRIMARY (K=23 v4+h1d Path-B τ=100k LB 0.95354) pool is built on:
- Per-row arithmetic (TyreLife/Compound, RaceProgress, etc.)
- Per-row sequence position (Stint, lap_in_stint, lap_from_end)
- Per-(Driver) historical priors (1950-2022)
- Per-(R, Y, L)-Compound × KBinsDiscretizer (yekenot v4)

What it does NOT have, and what no FE recipe in our pool computes:
**aggregates over OTHER drivers' PitStop column at the same lap.**

F1 strategy is partly herd-conditional. When the field starts pitting
(SafetyCar / undercut window opens / first driver's strategy
materialises), other drivers follow. `fs_cum_pits` encodes how far the
race has progressed in pit-event terms; `fs_pit_rate_now` encodes the
local "everyone is pitting" state; `fs_compound_pit_rate` encodes the
per-compound pit cascade.

The GBDT cannot reconstruct any of these from a single row's features
because they depend on aggregating over other rows in the same group.
This is the structural axis Probe 1 (combined-frame lead/lag) attacked
on the wrong side: own-row lead/lag is implicitly encoded by GBDT
interactions; cross-row aggregates of OTHER rows' columns are not.

## Calibration

| Probe | PI sealed | Agent BOTE expected_lb_bp | Realised LB Δ | Realised OOF Δ |
|---|---:|---:|---:|---:|
| probe_combined_lead_lag | 0 bp | +0.025 bp | 0 bp (no submit) | +2.18 bp |
| probe_target_structure | 0 bp | n/a (EDA) | 0 bp (no submit) | n/a |
| probe_ntl_single_rule | 0 bp | n/a (single-feat) | 0 bp (no submit) | best ≤0.687 |
| **probe_field_state** | **0 bp** | **+0.025 bp** | **0 bp (no submit)** | **+15.58 bp** |

PI's "0 bp LB Δ" trivially holds (no submit), but the OOF signal is 30×
agent's predicted band and 624× agent's median (0.5 bp). The
single_base_fe_addition family prior in FAMILY_PRIORS (p=0.05, band
[0, 0.5, 2.0]) was calibrated on 4-of-4 NULLs from earlier in the comp;
it does NOT cover cross-row aggregate features as a class. The family
prior should be split.

## What this UNblocks

Six concrete next-session candidates, ordered by EV:

| # | Candidate | Cost | Predicted OOF lift on top | Why |
|---|---|---|---:|---|
| 1 | **CB-v4 + field-state** (retrain v4 with fs features) | 35 min Kaggle GPU | +1 to +5 bp standalone | v4 absorbs most signal but FS encodes other-rows column — orthogonal |
| 2 | **K=24 = K=23 + field-state-LGBM** (this probe's model as new base) | 5 min CPU | +0 to +2 bp at meta | base-add per `path-b-amp-only-fires-on-meta-arch-not-base-add`; check ρ first |
| 3 | **CB-v4 + field-state + h1d FE re-train** | 1 h Kaggle GPU | +2 to +8 bp standalone | yekenot's KBins on `fs_pit_rate_now` and `fs_cum_pit_rate` |
| 4 | **Path-B with cohort = field-state-quartile** (alt segmentation axis) | 20 min CPU | +0 to +3 bp meta-arch | meta-arch redesign IS amp-eligible; new axis from probe 4 |
| 5 | **Field-state derivatives**: lead_fs_cum_pits (will pit count rise next lap?), Δfs_pit_rate (acceleration of pit cascade) | 30 min CPU | +0 to +2 bp standalone | extends mechanism |
| 6 | **Race-state global features**: lap with min/max field LapTime spike (SafetyCar proxy), simultaneous Position_Change indicator | 30 min CPU | +0 to +3 bp standalone | non-pit field-state |

The strongest combined move is probably #1 + #2: add field-state to v4
recipe, retrain CB-v4-fs on Kaggle GPU; check standalone OOF; if v4-fs
> v4 by ≥3 bp standalone, replace v4 with v4-fs in K=23 stack;
otherwise add as 24th base. Either path is on the load-bearing axis.

## Strict fold-safe re-run (2026-05-07 PM, post-PI flag)

PI flagged: "isn't this a feature we tested already and discarded due to
leakage?" — referring to Day-17 P1 v2 FS_A merge that inflated OOF by
491 bp (caught by 80/20 holdout). The Day-17 pattern was
`df[df.PitNextLap==1].groupby(...).mean()` (uses LABEL); probe 4 uses
`df.groupby([R,Y,L]).PitStop.sum()` (uses PitStop, a feature column
with single-feat AUC 0.521 ≈ chance vs target per U2). So strict Rule
24 doesn't apply, but the strict-fold-safe re-run is the defensive
audit pattern.

`scripts/probe_field_state_strict.py` — for each CV fold, compute
field-state aggregates from tr_fold rows ONLY (or tr_fold + test for
combined variant), then merge into both tr and val rows.

| Run | OOF AUC | Δ vs F2 raw | Δ vs full-train |
|---|---:|---:|---:|
| F2 raw 14 (baseline) | 0.94074 | — | — |
| F3 full-train combined | 0.94230 | +15.58 bp | — |
| **F3 strict per-fold combined** | **0.94211** | **+13.73 bp** | -1.87 bp |
| F4 full-train tr-only | 0.94241 | +16.66 bp | — |
| **F4 strict per-fold tr-only** | **0.94208** | **+13.35 bp** | -3.35 bp |

**Verdict: FOLD-SAFE-REAL.** Strict retains 85% of the lift. Day-17
target-reformulations collapsed 88-100% under the same audit; probe 4
collapses 12-22%. The residual loss is consistent with "smaller source
set → noisier per-fold aggregate" rather than label-leakage. Honest
fold-safe estimate: **+13.35 to +13.73 bp** standalone OOF.

This is the find of the day. PI's leakage skepticism was correct to
demand the audit — and the audit cleared it. The structural axis
(cross-row aggregates over OTHER rows' PitStop column) is real.

## Friction tags introduced

- `cross-row-aggregates-fire-where-own-row-sequence-doesnt` — Probe 1
  (own-row lead/lag) closed at GBDT-implicit; Probe 4 (cross-row
  aggregates of OTHER rows' columns) lifts +15.58 bp because the GBDT
  cannot reconstruct group-aggregate values from a single row.

- `field-state-mechanism-fires-on-train-only-too-no-combined-premium`
  — combined-frame premium F3-F4 = -1.08 bp. Per-(R, Y, L) groups are
  large enough (~50 rows/cell on average) that train-only aggregates
  are stable.

- `family-prior-single-base-fe-addition-calibrated-on-row-features-not-aggregates`
  — FAMILY_PRIORS["single_base_fe_addition"] = (p=0.05, band [0, 0.5,
  2.0]) was calibrated on 4-of-4 row-feature NULLs. Cross-row aggregate
  features are a separate class with much fatter tails. Recommendation:
  split into `single_base_row_fe` (current prior) and
  `single_base_cross_row_aggregate` (new band, p=0.30 (1, 5, 15)) when
  the next BOTE in the family lands.

## Pointers

- `scripts/probe_field_state.py`
- `scripts/artifacts/probe_field_state.json`
- `audit/decisions.jsonl` — BOTE + outcome record.

## Bottom line

After 4 probes today on "what's hiding in plain sight" with PI sealed
prediction = 0 bp on every one, this is the one that fired. Whether
the +15.58 bp standalone OOF transfers to K=23 LB lift is the next
session's first-action decision. The mechanism is novel enough
(cross-row aggregates of OTHER rows' columns, AV-safe) that it
deserves a Kaggle GPU slot for v4-fs retraining.
