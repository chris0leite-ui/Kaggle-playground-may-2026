# RESEARCH-PLAN-extensions.md — four-lane research plan

Branch: `claude/research-model-extensions-Ibwvn`. Author: senior-ML lens
re-read of state at 2026-05-08 PM. Companion probe scripts:

- `scripts/probe_lane1_downsampling.py`
- `scripts/probe_lane2_priors.py`
- `scripts/probe_lane3_routing.py`
- `scripts/probe_lane4_nonlr_meta.py`

## The constraint, restated

The K=27 logit pool sits in a 3.23-D subspace (A25). Three different
inductive biases (LambdaRank, inter-stint memory, dual-head) absorb at
K=10+1 LR-meta within ±0.05 bp despite Spearman ρ as low as 0.41 (A29).
The 30-feature [P, rank, logit] expansion can reconstruct any new base's
logit linearly. Conclusion (A30): to break ceiling we need either new
**information** outside the 14 columns or a **non-linear meta** that
projects beyond the linear span.

Because top-of-LB has few submissions, the gap to leader (12.5 bp) is
**most likely a real mechanism**, not lottery. That sharpens the search:
we are looking for one structurally-different signal, not for
incremental tuning.

The four lanes below each target a *specific* unstated assumption that
the current pool relies on. Each lane has a diagnostic (always-cheap,
always-informative) and one or two active probes whose null result tells
us something about the next lane.

---

## Lane 1 — Downsampling / censoring is structured, not random

**The unstated assumption:** rows are i.i.d. snapshots. Every model in
the pool treats `(row → PitNextLap)` as an independent classification.

**The fact that breaks it (Probe C, 2026-05-08; W3):** mean observed
stint length = 3.87 vs 19.80 in original; gap-1 fraction = 28% vs 99.6%.
The synthesiser temporally downsamples — each row is a sparse sample of
a longer trajectory. PitNextLap therefore conflates *two* events:
- "the next observed lap is the pit lap" (gap = 1), and
- "the next observed lap is K laps later, and the pit happened
  somewhere in those K-1 unseen laps plus the pit lap" (gap > 1).

Under censoring, **P(PitNextLap | row) ≈ P(stint ends in next g laps |
row)**, where g is the row-specific lap gap to the next observed row.
The model never sees `g`, so it averages across a mixture of horizons.

**What we test (D1 + P1):** does explicitly modelling the gap recover a
4th logit direction?

- **D1.1 (diagnostic).** Distribution of `gap_to_next_obs` per row, per
  Compound. Conditional `P(PitNextLap | gap)` curve. If `P(PitNextLap |
  gap=1)` is materially different from `P(PitNextLap | gap=5)`, then the
  current pool is collapsing real horizon structure.
- **D1.2 (diagnostic).** Per-(gap-bucket) calibration of PRIMARY: is
  PRIMARY systematically over/under per gap?
- **P1.1 (active).** Add `gap_to_next_obs`, `gap_to_prev_obs`,
  `gap_z_in_stint` as **meta features** alongside K=4 [P, rank, logit].
  K=4+gap-meta gate. Cost: ~25 min.
- **P1.2 (active).** Discrete-time-hazard LGBM with k ∈ {1,2,3,5,10}
  binary heads (`P(stint ends ≤ k obs-laps from now)`); marginalise via
  the actual gap. K=4+1 gate as a single base. Cost: ~45 min.
- **P1.3 (active).** Per-gap isotonic recalibration of PRIMARY. Cost:
  ~15 min. Floor risk; either fires or it doesn't.

**What null tells us:** the 14 columns themselves don't encode gap-
relevant information beyond what the pool already uses; sequence-lane
is closed for real and Lane 4 is the only remaining structural avenue.

**EV:** P1.1 is the cheapest 4th-direction probe with structural
backing. Predicted +0.5 to +2 bp if gap is real signal.

---

## Lane 2 — F1 pit-decision priors aren't fully encoded

**The unstated assumption:** GBDTs on 14 columns will discover all
deterministic structure (last-3-laps no-pit, tyre-cliff at p99(Compound),
mandatory two-compound rule, pit-window centring per (Compound, Race)).

**The fact that makes this dubious:** EXP-2/3/4 produced low-ρ
predictions that absorbed at the LR meta. That tells us the existing
pool's *linear span* is fixed, but it does NOT tell us those features
are present *as columns* the meta can route on. They are mixed into base
predictions.

**What we test (D2 + P2):** does presenting the priors **as meta
features alongside base predictions** add a direction the LR meta
cannot reconstruct from base outputs alone?

- **D2.1 (diagnostic).** Empirical pit hazard curve per (Compound):
  `P(pit | TyreLife = t)` and `P(pit | laps_to_race_end = k)`. Identify
  bins where prior is near-deterministic (P > 0.8 or P < 0.05).
- **P2.1 (active).** Build heuristic meta features (fold-safe per
  Rule 24):
  - `tyre_life_pctile_in_compound` — percentile of TyreLife in Compound
  - `laps_to_race_end` — race_max_lap − LapNumber
  - `is_last_3_laps`, `is_last_lap` — indicators
  - `compound_tier_ordinal` — SOFT=1 … WET=5
  - `n_distinct_compounds_so_far` — Driver-Race-Year cumulative
  - `race_lap_pit_density` — cross-driver field-state at this Lap
  - `stint_progress` = TyreLife / quantile_99(TyreLife | Compound)
  Concatenate with K=4 [P, rank, logit]; LR meta gate. Cost: ~30 min.
- **P2.2 (active).** Deterministic post-hoc clamps on PRIMARY output:
  if `is_last_3_laps and stint_lap_idx > 0` → P_pit clamped to bin
  empirical mean; if `TyreLife > p99(Compound)` → P_pit floored at
  empirical mean. Compute OOF Δ. Cost: ~15 min.
- **P2.3 (active).** Compound-tier monotonic LGBM (single base):
  `monotone_constraints={"TyreLife": +1}` on (TyreLife, Compound,
  LapNumber). Standalone weak; K=4+1 gate. Tests if a *constrained*
  hypothesis class adds a direction unconstrained GBDTs cannot.
  Cost: ~20 min.

**What null tells us:** GBDT really did absorb all deterministic
priors; the missing 4th direction is not in the 14 columns at all
(reinforces Lane 4).

**EV:** P2.1 most likely to fire. Heuristic features as meta inputs is
the cheapest mechanism on the entire menu — bypasses A30's "non-LR
meta required" inference by routing new info *outside* the base
predictions. Predicted +1 to +3 bp if priors are leaked.

---

## Lane 3 — Compound class imbalance via routing/gating

**The unstated assumption:** a single global meta-projection is optimal
across all Compounds. Path-B Compound × Stint shrinkage *softens* this
via per-segment LR with prior-pool toward global, but it doesn't allow
**different bases to win in different regimes**.

**The fact that makes this dubious:** Probe A (2026-05-08) showed WET
rows have AUC 0.845 vs global 0.954. The rain residual is real. The
specialist-replacement test (A22) failed because a single-LGBM rain
specialist lost cross-Compound transfer (−152 bp). But routing — *use
the global pool for dry, blend for wet* — was never tested.

**What we test (D3 + P3):** does the optimal meta projection differ
across Compounds, and can that difference produce a 4th direction at
the global level?

- **D3.1 (diagnostic).** Per-Compound K=4 standalone AUC and
  forward-greedy K=1 winner. If Compound-conditional winners differ,
  routing has structural support.
- **D3.2 (diagnostic).** Per-Compound calibration of PRIMARY. ECE per
  Compound; identify Compound where miscalibration > global mean.
- **P3.1 (active).** Per-Compound LR meta heads (5 separate LRs, one
  per Compound), routed at inference. Cost: ~25 min. Different from
  Day-18 A4 (per-Compound LR on raw features, absorbed at K=10+1) —
  this is **on the K=4 [P, rank, logit] expansion**, post-pool.
- **P3.2 (active).** Per-Compound isotonic recalibration of PRIMARY.
  Cost: ~15 min. Different from prior conformal-isotonic schemes (which
  recalibrated globally with per-bin shrinkage) — this is per-Compound
  flat isotonic.
- **P3.3 (active).** Rain-row meta blend: refit K=4 meta on
  Compound ∈ {INTERMEDIATE, WET} only; at inference, blend `0.5 ×
  PRIMARY + 0.5 × rain_meta` for rain rows. Retains full-pool transfer.
  Cost: ~30 min.

**What null tells us:** the global meta really is uniformly optimal;
rain-row residual is intrinsic per Probe A.

**EV:** P3.1 is the directly-testable variant. Predicted ±1 bp; rain
rows are 4.3% of the data, so even sharp local lift bounds global
delta. P3.3 is the least likely to fire but the most diagnostic for
the W1 rain weakness.

---

## Lane 4 — Identify and break the 3-D logit subspace

**The unstated assumption:** the 30-feature [P, rank, logit] expansion
plus an LR meta is sufficient. Any non-linear interaction between bases
is invisible.

**Where the bottleneck actually is:** A30 explicitly says "non-LR
meta architecture is the only architecturally untested avenue." Yet
the team has tested 11+ Path-B segmentations and zero non-LR metas.
This is the single most underpriced experiment on the board.

**What we test (D4 + P4):**

- **D4.1 (diagnostic, structural).** SVD on K=27 logit pool, recover
  the top 3 right-singular vectors. For each, compute correlation with:
  raw features (TyreLife, LapNumber, Stint, Compound-OHE, Position),
  inter-stint features (n_distinct_compounds_so_far, laps_since_pit),
  field-state features (race_lap_pit_density). **This identifies what
  the 3 latent constructs are**, which directly informs what the
  missing 4th is. Cost: ~10 min.
- **P4.1 (active).** Gradient-boosted meta on K=4 [P, rank, logit]
  (12 features). Small LGBM (num_leaves=15, lr=0.05, min_data=200).
  K=4+GBM_meta gate; if it lifts global OOF by ≥1 bp over LR meta,
  ceiling is LR-specific not data-intrinsic. Cost: ~30 min.
- **P4.2 (active).** Small MLP meta on K=4 [P, rank, logit]. 2 hidden
  layers (32, 16), dropout 0.2, sigmoid out. Different nonlinearity
  from GBDT. Cost: ~30 min CPU.
- **P4.3 (active).** kNN-on-predictions meta. K=20 nearest neighbours
  in K=4 logit space; output mean(label[neighbours]). Non-parametric;
  bypasses any linear-or-tree model class assumption. Cost: ~30 min
  (FAISS or sklearn BallTree).
- **P4.4 (active).** Augmented LR meta: [P, rank, logit, raw row
  features]. 12 + 14 = 26 features. Tests whether row context routes
  the linear projection. Cost: ~15 min.

**What null tells us:** the 3-D ceiling is **fundamental to the 14
column data under any meta**; Lane 1+2+3 are the only paths and the
LB ceiling is reachable only via Lane 1's hazard/sequence path or
Lane 2's structural priors.

**EV:** P4.1 + P4.4 are the most likely to fire. P4.4 in particular is
the cheapest experiment on the entire plan that is *not* yet on the
EXPERIMENTS-NEXT.md menu.

---

## Recommended sequence (3-day window)

Day 1 (cheap, parallel):

1. D1.1 + D1.2 + D2.1 + D3.1 + D4.1 — **all diagnostics, in one batch**.
   ~1.5 hours. Output: structural picture of *what each lane is
   actually working with*. Decisions below depend on this.
2. P4.4 (augmented LR meta with raw features). 15 min. Single cheapest
   experiment that isn't on the menu yet.
3. P2.1 (heuristic meta features) and P4.1 (GBM meta). Both ~30 min.

Day 2 (medium):

4. P1.1 (gap-meta features) and P1.3 (per-gap calibration).
5. P3.1 (per-Compound LR heads) and P3.2 (per-Compound isotonic).
6. P4.2 (MLP meta).

Day 3 (capstone):

7. P1.2 (discrete-time hazard). Most distinctive structural reframe.
8. P3.3 (rain-row blend) if D3 diagnostics show Compound-routing
   opportunity.

---

## Pre-flight notes

**Rule 16 Q6 (metric alignment):** all probes target row-level AUC on
PitNextLap. Hazard reformulation (P1.2) marginalises explicitly back
to PitNextLap before the meta gate.

**Rule 24 (fold-safety):** every probe that uses label-conditional
aggregates re-fits per fold using training rows only. Lane 2's
`tyre_life_pctile_in_compound` and `n_distinct_compounds_so_far`
are computed combined-frame from row indices, not from labels — they
are AV-safe per A3.

**Rule 18 (claim board):** these probes target the open EXP-NEW
(non-LR meta) avenue and the W1 (rain residual) / W3 (downsampling)
weakness items. Suggested ISSUES.md leaves for the active session
to claim:

- Lane 1 → new leaf under §3 (target reformulation) + W3
- Lane 2 → new leaf under §1 (new model class — heuristic meta input)
- Lane 3 → 5b updated (rain blend) + new leaf under §2 (meta-layer)
- Lane 4 → 2e (new leaf, non-LR meta) — the explicit EXP-NEW.

**Risk register:**

- Heuristic features (Lane 2) need fold-safety check; the leakage trap
  Rule 24 came from exactly this class. Flag: any `*_so_far` cumulative
  feature is computed from row indices not labels, so safe; but
  `tyre_life_pctile_in_compound` is computed from features only, also
  safe.
- Non-LR meta (Lane 4) overfits easily on a 30-feature meta input
  with ~440k rows. Use shallow trees (num_leaves ≤ 15) and strong
  regularisation; cross-validate the meta itself per outer fold.
- Per-Compound routing (Lane 3) reduces effective sample size for the
  rain heads (rain ≈ 4.3% × 350k train ≈ 15k rows). Use shrinkage
  toward the global meta when fitting per-Compound heads.
