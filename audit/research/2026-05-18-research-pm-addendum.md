# Research-loop PM addendum — 2026-05-18

Companion to `2026-05-18-research.md` (AM synthesis). The plateau
triggered Rule 7 again at EOD R8 (Strategy-critic verdict: priority
queue Σ << headroom). Two additional research agents ran in PM
(notebooks, domain) and an updated prior-comp summary reuses the
AM artifact.

Status of AM candidates:
- **C1** (Per-Race OpenF1/FastF1 scalar join) — UNTESTED. Still
  priority candidate.
- **C2** (DAE swap-noise) — TESTED in R7 Phase A. Standalone OOF
  0.94665; absorbed at K=14+Path-B (Δ −0.09 to −0.15 bp). CLOSED.
- **C3** (Per-(Race, LapNumber) Bayesian shrinkage) — **FALSIFIED**
  in `audit/2026-05-18-tier-a-batch.md`. 3 variants tested; all
  regress at every w > 0. Mechanism error: any per-group shrinkage
  trades row-level rank discriminability for group smoothness, and
  AUC measures ranking. CLOSED.
- **C4** (UID magic-feature groupby base, IEEE-CIS Deotte precedent)
  — UNTESTED. Still priority candidate.

## PM new candidates (10 total, deduped below to 5 novel)

Combined output from notebooks + domain agents. Renaming to
prevent collision with AM C1-C4 codes.

| Code | Mechanism | Predicted lift | Cost | Source |
|---|---|---:|---:|---|
| **NB1** | Driver-pair contagion (`AheadPit_lag{1,2,3}`, `BehindPit_lag{1,2,3}`, same-team double-stack) | +0.3-0.8 bp meta | 30 min CPU | Notebooks agent |
| **NB2** | Lap-time derivative features (within-(Driver, Race) Δ + rolling-3 of `LapTime (s)`) | +0.2-0.5 bp meta | 20 min CPU | Notebooks agent |
| **NB3** | `undercut_window_score` continuous scalar (sigmoid conjunction of gap × tyre-age × laps-remaining; thresholds from undercut literature) | +0.1-0.4 bp meta | 15 min CPU | Notebooks agent |
| **NB4** | Per-(Compound × Stint) target-mean as **base learner** (R24 fold-refit + R33 inner-CV) — segmentation-as-base, not as Path-B operator | +0.1-0.3 bp meta | 30 min CPU | Notebooks agent |
| **NB5** | Circuit-degradation-class external join (3-level Pirelli-style abrasiveness rating) | +0.2-0.4 bp meta | 1-2 h external pull | Notebooks agent |
| **DM1** | Race abrasiveness scalar (Pirelli 1-5 × Compound × TyreLife triple interaction) | +0.05-0.15 bp | 5 min CPU (26-row lookup) | Domain agent |
| **DM2** | Compound-cliff-lap ratio (TyreLife / expected-cliff-lap[Race, Compound] from FastF1 historical) | +0.10-0.30 bp | 1-2 h external pull | Domain agent |
| **DM3** | Competitor pit cascade — within-(Race, LapNumber) count of OTHER drivers' pits on L-1/L-2 | +0.15-0.40 bp | 25 min CPU | Domain agent |
| **DM4** | Undercut break-even surrogate (`pit_loss_sec / fresh_delta_per_lap` × `LapsRemaining`) | +0.05-0.20 bp | 1 h CPU | Domain agent |
| **DM5** | VSC/SC base-rate prior per (Race, LapNumber-bucket) from FastF1 race_control | +0.05-0.15 bp | 3-4 h external pull | Domain agent |

## Deduplication

- **NB1 ≈ DM3**: both about cross-driver pit cascade. NB1 lagged
  per-row of named neighbours; DM3 within-row aggregate count.
  Merge into one mechanism: **"competitor pit cascade"**, two
  flavour variants (per-row lag vs within-row aggregate) — pick
  the simpler within-row aggregate first.
- **NB5 ≈ DM1**: both Pirelli-style abrasiveness. NB5 is 3-level
  categorical; DM1 is Pirelli 1-5 scalar. Merge into **"race
  abrasiveness external scalar"**.
- **NB2 vs LapTime_Delta column**: `LapTime_Delta` already exists
  in the schema. NB2's "Δ within-(Driver, Race)" is essentially
  what `LapTime_Delta` captures (need to verify the definition).
  If `LapTime_Delta` already = consecutive lap-time delta, NB2's
  Δ feature is REDUNDANT — only the rolling-3 statistic is new.
- **NB3 / DM4**: both encode undercut-break-even logic at different
  abstraction levels. NB3 is a single sigmoid scalar; DM4 is a
  raw ratio × LapsRemaining. Both worth testing as a pair.

## Q6 metric-alignment for each surviving candidate

| Candidate | Q6 verdict | Reasoning |
|---|---|---|
| **C1** Per-Race OpenF1 scalar | PASS | Adds row-feature; LightGBM with log_loss is loss-optimal for row-AUC per past s6e5 finding. |
| **C4** UID magic-features | PASS | Same; standalone LGBM base, fold-safe groupby aggregates. |
| **NB4** Per-(C×S) target-mean base | PASS (conditional on R24/R33) | Target-encoded base ranks fed to LR meta; fold-refit eliminates leak. |
| **NB1+DM3** Competitor pit cascade | PASS | Cross-row aggregate of `PitStop[L-1]` (confirmed exposed in train+test) → LGBM input. |
| **NB2** Lap-time derivative | PASS | Standard derivative FE. |
| **NB3** `undercut_window_score` | PASS | Continuous scalar input. |
| **NB5/DM1** Race abrasiveness | PASS | Single per-Race scalar; LGBM consumes. |
| **DM2** Compound-cliff-lap | PASS | Single ratio; LGBM consumes. |
| **DM4** Undercut break-even surrogate | PASS | Single ratio. |
| **DM5** VSC/SC base-rate | PASS | Single rate per (Race, LapBucket); LGBM. |

All PASS Q6. No survival hazard / sequence-loss framing in this list.

## Top-3 candidates for next session (EV / cost ranked)

After dedupe and excluding closed (C2, C3):

### Rank 1 — NB4 — Per-(Compound × Stint) target-mean as BASE learner

**Why now**: NOVEL axis we haven't explored. The Path-B operator
uses Compound × Stint segmentation as a META-LEVEL shrinkage
operator. NB4 turns the same partition into a **BASE-LEVEL TE
feature** that becomes a new K=14 ingredient. The two are
mechanistically distinct: Path-B is convex per-segment LR; TE-base
is target-mean broadcast as a feature for downstream LightGBM. The
target-mean signal might add diversity at the K-pool layer that
Path-B's meta-level shrinkage can't replicate.

**Mechanism details**:
- 5-fold CV; on each train fold, compute per-(Compound, Stint)
  mean(PitNextLap); broadcast to val fold + test (Rule 24 fold-refit).
- Add as a single column to existing base-LightGBM (yekenot recipe).
- Inner-CV on the TE base (Rule 33) to confirm OOF lift before
  passing to K-pool gate.

**Cost**: 25-30 min CPU.

**Predicted**: standalone OOF +5-15 bp (TE bases typically); meta
add at K=13+Path-B +0.1-0.3 bp (saturation discount).

**Kill criterion**: standalone OOF < 0.945 in 5-fold (TE leak
indicator), OR meta-add Δ < +0.01 bp at K=14+Path-B C×S τ=100k.

### Rank 2 — C4 — UID magic-feature groupby base

**Why now**: IEEE-CIS Fraud +109 bp precedent. UID =
`Driver + '_' + Race + '_' + floor(LapNumber / W)` for W ∈
{5, 10, 20}; on each fold's training rows, compute 20-40 per-UID
groupby aggregates of LapTime, TyreLife, Position,
Cumulative_Degradation (means / stds / max / min / count) — fed
as columns into a single LGBM base.

**Cost**: 25 min CPU.

**Predicted**: standalone OOF +5-30 bp; meta add at K=13+Path-B
+0.1-0.5 bp.

**Kill criterion**: standalone OOF < 0.945 OR meta-add Δ < +0.01 bp.

### Rank 3 — Competitor pit cascade (NB1+DM3 merged)

**Why now**: PitStop column exposed in BOTH train and test (verified
this session). Within-(Race, LapNumber) cross-driver pit aggregates
encode F1's documented undercut/overcut dynamic — a mechanism class
not currently in K=13. Strategy-critic Section 3 showed the K=13
pool error pattern is uniform (no row-level diversity at the hard
segments); cross-driver signal is the missing piece.

**Mechanism details**:
- For each (Race, LapNumber) row, compute aggregate over OTHER
  drivers at L-1 and L-2: count_pits_other, count_drivers_other,
  pit_rate_other. R24 fold-safe (uses only feature column, not target).
- Add 6 columns to LightGBM base.

**Cost**: 25 min CPU.

**Predicted**: standalone OOF +2-8 bp; meta add at K=13+Path-B
+0.1-0.4 bp.

**Kill criterion**: standalone OOF < 0.945 OR meta-add Δ < +0.01 bp.

## Mechanism-axis observation (cross-cutting)

**Strategy-critic Section 1 surfaced MEDIUM × Stint 2 as the
worst segment (AUC 0.897, 5.8 % of train, 44.8 % prior).**
The session's 5-min specialist probe (kitchen-sink LightGBM on
the MEDIUM-S2 subset only, 18 features incl. 6 targeted
interactions) yielded OOF AUC 0.881 — 16 bp BELOW PRIMARY on the
same subset. **The segment IS at noise floor for row features**;
specialised FE on this segment is REFUTED.

Implication: the worst segments are *intrinsically hard*, not
*feature-deficient*. The +1.6 bp top-5% gap won't close by
targeting MEDIUM-S2; it requires a mechanism class that lifts
signal-richer segments (HARD, INTER, MEDIUM × Stint ≥3) or
adds cross-row context (top-3 candidates above).

## Strategic posture

**Headroom math (from Strategy-critique Section 5)**: priority
queue midpoint Σ × P(real) = 0.058 bp realistic; gap = 1.6 bp.
Even with all 3 new candidates landing at their midpoint
predictions and stacking sub-linearly, queue Σ ≈ 0.3-0.5 bp —
**still well short of 1.6 bp**. The plan is no longer "reach
top-5% by stacking research candidates"; it's **"maximise the
hedge ladder for final-window R7d while continuing to scout for
a +1 bp single mechanism."**

Final-window R7d (Days 28-31) will see:
- PRIMARY R7.1 LB 0.95389
- Hedge-1: R7.2 fold-bag (LB 0.95389 tied; structurally distinct)
- Hedge-2: R8 60/20/20 multi-seg rank-blend (saved, untested at LB)
- Hedge-3: K=27 + Path-B τ=100k (LB 0.95368; different operator pool)
- New hedge candidates from research: best-meta-add survivor of
  NB4 / C4 / cascade if any of these land at +0.2 bp OOF (TIE_ZONE
  on public LB but distinct on private).

## Open questions for PI

1. Did the morning research-loop's Tier-A batch finish? `tier-a-batch.md`
   shows C3 FALSIFIED + a2_2 and a3_1 both WEAK at K=4+1. The other
   8 items (rank-sorted gaps, lagged DriverAheadPit, Heilmeier
   residual, per-track fuel coef, nested TE, C4 UID magic, KNN-target-mean,
   F3 field-state, quantile/histogram groupby) — were they all run, or
   did the session pivot to R5 K=11 rebuild before completing?
2. Should next session attack NB4 (target-mean BASE) first, or wait
   for kaggle-cli-401-auth fix so we can do the proper R22 notebook
   scan that's now 18+ days overdue?
