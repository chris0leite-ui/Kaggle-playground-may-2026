# 2026-05-08 PM — four-lane research-extension audit

Branch: `claude/research-model-extensions-Ibwvn`. PI prompt: pursue
the four directions sketched in the senior-ML-researcher analysis
(downsampling, F1 priors, compound routing, 4th latent construct).

PRIMARY substitute used: **plain K=4 LR meta on [P, rank, logit]**,
OOF AUC 0.95399. The K=4+Path-B PRIMARY composite (LB 0.95351) is
not on the published artifact dataset; the substitute was used as the
in-script comparison anchor. Relative deltas vs Path-B PRIMARY would
shift by ≤0.04 bp (A9c: Path-B amp is statistical noise at this pool
size).

## Headline result — ceiling closes harder than expected

**0 of 13 active probes PASS the ≥+0.5 bp gate.**
**6 AMBIG (within ±0.2 bp of zero), 7 NULL or strongly negative.**

The senior-lens framework (four lanes) was the right map. The answer
in every lane was: **K=4 has already absorbed the signal**. New
information cannot enter via meta-features, gap-aware features,
deterministic priors, compound routing, or any non-LR meta architecture
on K=4 [P, rank, logit].

## Lane-by-lane

### Lane 4 — 4th latent construct (D4.1, P4.1, P4.2, P4.4)

The **single most informative diagnostic** of this session.

- **K=4 effective rank in logit = 1.33** (entropy on singular values).
  Component 1 alone captures **93.6% of variance**. Compare K=27 = 3.23
  (A25). The forward-greedy pool collapse is dramatic.
- Component 1 loadings: all 4 bases load similarly (-0.44 to -0.56) —
  this is the consensus direction.
- Component 1 correlations: TyreLife=−0.33, LapNumber=−0.30,
  is_HARD=−0.27, is_MEDIUM=+0.26, Cumulative_Degradation=+0.23.
  **The dominant direction is "tyre-degradation pressure × compound."**
- Component 2 (4.6%): d16_orig_continuous_only loads −0.89 vs synth-
  trained bases positive — this is the **"original-vs-synth"** axis.
  Weakly correlated with TyreLife (−0.14) and is_SOFT (+0.10).
- Component 3 (1.0%): d17_h1d (NN) +0.63 vs f1_hgbc (HGBC) −0.77 —
  **NN-vs-HGBC architectural disagreement**. No strong feature corr.
- Component 4 (0.7%): noise.

Probe results:
- **P4.1 GBM meta: NULL (−1.20 bp).** Gradient boosting on 12-feature
  meta input regresses vs LR even with strong regularisation
  (num_leaves=15, lr=0.03, lambda_l2=1.0).
- **P4.2 MLP meta: strongly NULL (−7.77 bp).** Two-hidden-layer MLP
  overfits the projection. Confirms: at the K=4 meta level, LR is the
  right model class.
- **P4.4 augmented LR + raw row features: AMBIG (−0.04 bp).** Adding
  TyreLife, LapNumber, Stint, Position, LapTime_Delta,
  Cumulative_Degradation, RaceProgress as columns alongside the
  K=4 expansion does NOT add a direction.

**Lane 4 falsifies A30.** Non-LR meta is not the constraint. The
constraint is the 1.33-D pool collapse — any meta on K=4 learns the
same 1-D consensus projection.

### Lane 1 — downsampling / censoring (D1.1, D1.2, P1.1, P1.3)

- **D1.1 gap distribution shows real signal:** P(pit | gap=1) = 8.5%
  vs P(pit | gap≥11) = 29.9%. **3.5× monotonic gradient.** The gap is
  a strong marginal predictor.
- **D1.2 per-gap calibration is NEAR-PERFECT.** Per-bucket ECE between
  0.0001 and 0.0015 across all 7 gap buckets. The K=4 LR meta has
  fully absorbed the gap-conditional probability.
- **P1.1 gap features as meta input: AMBIG (+0.02 bp).** Adding
  `gap_to_next_obs`, `gap_to_prev_obs`, `stint_lap_idx`,
  `is_last_in_stint`, `stint_density`, `stint_size` doesn't help.
- **P1.3 per-gap isotonic recalibration: NULL (−2.18 bp).** Splitting
  isotonic per gap-bucket overfits — already-optimal calibration
  cannot be improved by more parameters.
- Per-gap AUC drops from 0.9628 (gap=1) to 0.9316 (gap≥11) — a
  residual that no probe in this lane recovered.

**W3 (downsampling) is NOT the bottleneck.** TyreLife + Stint +
LapNumber + Compound implicitly encode enough that the K=4 pool
recovers the gap-conditional structure without seeing gap directly.

### Lane 2 — F1 pit-decision priors (D2.1, P2.1, P2.2, P2.3)

- **D2.1 hazard tables.** P(pit | TyreLife pctile in compound) goes
  from 0.035 (p10) to 0.32 (p100) — strong gradient, already learned.
- **CRITICAL synth quirk: P(pit | is_last_lap_of_race) = 0.38**, not
  ~0. F1 reality says drivers don't pit in the last lap; this synth's
  labelling diverges. n=21 but the direction is opposite to F1
  intuition. **Senior-lens domain heuristics partially misfire here.**
- **P2.1 heuristic features as meta input: AMBIG (−0.02 bp).** 9
  hand-crafted F1 heuristics (compound_tier, tyre_life_pctile_in_
  compound, laps_to_race_end, is_last_3_laps, race_progress,
  n_distinct_compounds_so_far, field_size_at_lap, stint_overrun)
  added as meta features: ZERO lift.
- **P2.2 deterministic rule clamps: NULL (−9.48 bp).** "Last-lap
  clamp" misfires because of the D2.1 quirk; tyre-cliff clamp
  also doesn't help.
- **P2.3 monotonic LGBM K=4+1: AMBIG (+0.19 bp).** Compound-tier
  monotone single base (TyreLife+, pctile+) standalone AUC 0.805;
  K=4+1 lift +0.19 bp — within noise.

**Domain priors are fully absorbed by the K=4 GBDT bases.** The
"heuristic features as meta inputs bypass A30" hypothesis fails at the
empirical level: the K=4 [P, rank, logit] linear span already covers
those heuristics.

### Lane 3 — compound routing/gating (D3.1, D3.2, P3.1, P3.2, P3.3)

- **D3.1 per-Compound AUC:** MEDIUM 0.957 (n=211k, easiest), HARD
  0.937 (n=171k), INTERMEDIATE 0.940 (n=17k), SOFT 0.935 (n=39k),
  **WET 0.832** (n=1,355). For WET, p1_single_cb_v4_gpu wins
  (0.856) — different from other compounds where d17_h1d wins.
- **D3.2 per-Compound calibration:** ECE for HARD/INT/MED/SOFT in
  range 0.0015–0.0054. **WET ECE = 0.0112** (3-7× higher).
  WET p̂_mean=0.017 vs ȳ_mean=0.025 — PRIMARY underpredicts WET pits.
- **P3.1 per-Compound LR meta heads: AMBIG (+0.11 bp).** Per-Compound
  routing buys ≤1 bp at this K=4 pool size.
- **P3.2 per-Compound isotonic: NULL (−1.78 bp).** Same overfit
  pattern as P1.3.
- **P3.3 rain-row meta blend: AMBIG (+0.03 bp global, +0.02 bp within
  rain).** Refitting K=4 meta on rain-only and blending 50/50 doesn't
  meaningfully differ from global.

**W1 (rain residual) survives this lane.** The WET miscalibration is
real but the segment is too small (n=1,355 train) for any
Compound-conditional meta to recover at the global level.

## Cross-lane synthesis

The four lanes were the right hypothesis tree. The empirical answer is
unified:

> The K=4 forward-greedy pool collapses to a 1.33-D logit subspace
> dominated by tyre-degradation × compound. Every meta-architecture
> variant we tested (LR, GBM, MLP, augmented LR) and every meta-input
> augmentation (gap features, F1 heuristics, raw row features) lies
> within that subspace's reach.

Implications:

1. **A30 dropped from `live` to `FALSIFIED`.** Non-LR meta is not the
   architecturally untested avenue — it was tested, it regresses.
2. **A29 holds and tightens.** The "logit-direction-level rank-lock"
   IS the binding constraint, but it's even tighter at K=4 (1.33-D)
   than was inferred at K=27 (3.23-D).
3. **The 12.5-bp gap to leader cannot be closed at the meta or
   feature-engineering layer.** It must come from BASE-level
   structural change — different bases trained on:
   (a) different data (closed by PI: external data off-table),
   (b) different objective (EXP-2 lambda-rank, EXP-4 dual-head — null),
   (c) **different sample windows**: e.g., a base trained on the
       gap-conditional decomposition (P1.2 hazard reformulation,
       not yet run as a base); a base trained on dense-trajectory
       reconstruction (would need FastF1 conditioning, also closed).
4. **The actually-untested lane is now the gap-conditional hazard at
   the BASE level.** P1.2 was sketched in the script as a TODO. If
   we want to make one more swing, that's it.

## Recommended posture

The PI's read that "leader gap is real mechanism, not lottery" is
consistent with this finding: the leader has a 4th direction we cannot
manufacture from K=4's pool. But the mechanism we'd need lives at the
**base** layer with **information we don't have** (FastF1, full lap-
by-lap trajectories). All meta-layer and feature-engineering paths
explored by these four lanes are within the existing 1.33-D span.

**Hedge / wrap-up posture is now justifiable.** The R5 hedge ladder
(`state/hypothesis-board.md` "Hedge ladder") plus the K=27 PRIMARY
re-promotion as PRIMARY = +1.7 bp over current PRIMARY for free; that
brings us to LB 0.95368 = 8.4 bp from top-5%, 10.8 bp from leader.
Both gaps remain partially inside ±12 bp public-LB sample-noise.

## Artifacts

- `scripts/artifacts/probe_lane{1,2,3,4}_*.json` — full numerical
  results.
- `scripts/artifacts/oof_lane*_strat.npy` — 11 OOF arrays for any
  downstream stack-add probes.

## Friction events for `audit/friction.md`

- **`pool-collapse-K4-effective-rank-1.33`** (NEW). Forward-greedy
  K=4 collapses to 1.33-D in logit, far below K=27's 3.23-D.
  Implication: forward-greedy reduces effective rank faster than it
  reduces base count.
- **`synth-divergence-from-F1-realism-on-last-lap`** (NEW).
  P(pit | is_last_lap_of_race) = 0.38 in this synth. F1 domain
  intuitions about race-end strategy partially misfire.
- **`isotonic-overfits-when-base-already-calibrated`** (CONFIRMED, 2nd
  time). P1.3 per-gap isotonic and P3.2 per-Compound isotonic both
  regress despite ECE diagnostic showing well-calibrated input.
  Friction tag candidate.
