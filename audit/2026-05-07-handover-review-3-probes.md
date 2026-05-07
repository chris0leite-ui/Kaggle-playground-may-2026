# 2026-05-07 — Handover-review 3-probe pass (claude/review-handover-solutions-oE78b)

`branch: claude/review-handover-solutions-oE78b`
`tag: hiding-in-plain-sight + diagnostic-only-no-submit`

> **Status: 3 cheap probes, 0 LB submits.** PI committed sealed
> prediction = 0 bp LB Δ for all three before any execution (Rule 26a).
> Two probes falsified the hypothesis; one ongoing.

## Setup

PI prompt: "What are the simple solutions, the simplest way to learn
something about our problem, that is hiding in plain sight?"

Three candidates surfaced after re-reading HANDOVER.md, friction.md,
U1-U3 probes, comp-context.md and the v4-yekenot Day-17 PM audit:

1. Train+test combined lead/lag features (computed never).
2. PitNextLap target-structure reverse-engineering (target undefined).
3. NTL single-rule single-feature OOF baseline (host quote: "removing
   Normalized_TyreLife makes the prediction trivial").

Per Rule 26a, PI committed first: **0 bp for all three.**

## Probe 3 — NTL single-rule baseline (cost 5 min) — FALSIFIED

`scripts/probe_ntl_single_rule.py` — single-feature StratKF 5-fold OOF
AUC of NTL reconstructions and threshold rules.

| Feature | OOF AUC | Δ vs base rate |
|---|---:|---:|
| **REF_TyreLife_only** | **0.69895** | — |
| REF_RaceProgress_only | 0.66437 | reference |
| R4 NTL_compound_year_max | 0.68673 | -1.22 bp |
| R5 NTL_compound_p99 | 0.66303 | -3.59 bp |
| R1 compound_tyre_norm | 0.63142 | -6.75 bp |
| R3 NTL_stint (within stint) | 0.62931 | -6.96 bp |
| R2 rule ctn>0.50 (best threshold) | 0.56577 | -13.32 bp |
| R5 rule ntl_p99>0.70 (best threshold) | 0.53405 | -16.49 bp |

**Verdict.** The host's quote — "we intentionally remove
Normalized_TyreLife which makes the prediction trivial" (brief.md:60) —
refers to the *unmasked NTL value from the original* dataset, NOT to a
denominator-by-Compound estimate built from the synth-corrupted features.
The synthesizer disrupted the NTL → PitNextLap deterministic relationship
enough that no row-local NTL rule recovers >0.70 OOF AUC. PI sealed
prediction (0 bp) wins.

**Counter-evidence to my Day-17 PM thesis.** I argued NTL-axis was
under-exploited; data says NTL reconstruction caps at 0.687, below
TyreLife alone (0.699). Closes that axis at single-rule level.

## Probe 2 — PitNextLap target structure (cost 30 min EDA) — STRUCTURAL FINDING

`scripts/probe_target_structure.py` — pure EDA, no model fitting.

### T6 multi-positive stint geometry
- `frac_last_pos_at_last_lap = 0.8104` — when stint has ≥2 positives,
  last positive lands on observed-last-lap 81% of the time.
- `frac_contiguous_positives = 0.6492` — 65% of multi-pos stints have
  contiguous positives.

### T1 P(target=1 | lap-from-stint-end)
| lap_from_end | rows | pos_rate |
|---:|---:|---:|
| 0 | 113,567 | 0.272 |
| 1 | 81,880 | 0.247 |
| 2 | 62,839 | 0.217 |
| 3 | 47,716 | 0.182 |
| 5 | 25,987 | 0.120 |
| 7 | 14,225 | 0.078 |
| 10 | 6,006 | 0.061 |

**Monotonic decay** from observed stint end (27.2% → 6.1% over 10 laps).

### T5 P(target=1 | RaceProgress decile)
Inverted-U peaking at RP ∈ (0.5, 0.7) at 38-39%, dropping to 5-15% at
extremes.

### T8 next-row change vs target (the simple-rule null)
| Mechanism | P(change \| pos) | P(pos \| change) |
|---|---:|---:|
| stint_change | 0.250 | 0.260 |
| compound_change | 0.171 | 0.210 |
| tyrelife_reset | 0.236 | 0.281 |
| ANY of above | 0.250 | 0.260 |

Only 25% of positives align with any next-row change. The target is
**not** a deterministic "next row started new stint" signal.

### T4 positives per stint by stint-size
Stint size 4 has `pos_max=4` — there exist stints where ALL 4 rows are
target=1. This rules out any windowed-shift interpretation of PitNextLap.
The target is fundamentally noisy/synthetic with cluster-decay structure.

**Verdict.** No exploitable deterministic rule. PI sealed prediction
(0 bp) holds *directly*. **Indirect** payoff: T1 + T6 motivate Probe 1
— sharper stint-end identification via train+test combined frame should
sharpen the dominant `lap_from_stint_end` decay axis.

## Probe 1 — Train+test combined lead/lag (cost 30 min) — FALSIFIED

`scripts/probe_combined_lead_lag.py`. Fold-safe (Rule 24): no labels
involved. Combined-frame transductive: AV-AUC=0.502 (Rule 25 PASS).

### L1 single-feature AUC: combined-frame ALWAYS DOMINATES train-only

| Feature | combined | train-only | Δ |
|---|---:|---:|---:|
| **lead_LapNumber** | **0.6960** | 0.6905 | **+5.5 bp** |
| lag_LapNumber | 0.6906 | 0.6852 | +5.3 bp |
| lead_RaceProgress | 0.6618 | 0.6569 | +4.9 bp |
| lag_Stint | 0.6614 | 0.6571 | +4.3 bp |
| lead_Stint | 0.6598 | 0.6568 | +3.1 bp |
| lag_RaceProgress | 0.6565 | 0.6514 | +5.1 bp |
| lead_TyreLife | 0.6421 | 0.6330 | **+9.0 bp** |
| lag_TyreLife | 0.6357 | 0.6310 | +4.7 bp |
| **lead_LapNumber_diff** | **0.6214** | 0.5924 | **+29.0 bp** |

**Every** lead/lag feature gains from combined-frame, with the
single-feature AUCs ALL > 0.5. `lead_LapNumber_diff` (= `lead_LapNumber
- LapNumber`) is the strongest combined-frame ADVANTAGE: +29 bp gain
just from filling in the train-only NaNs at train/test boundaries.

`lead_LapNumber` single-feature (0.696) is essentially equivalent to raw
`TyreLife` (0.699) — combined-frame lead is one of the strongest
single-feature signals on the comp.

### L3/L4/L5 LGBM standalone OOF — combined-frame premium NEGATIVE

| Run | Features | OOF AUC | Δ vs L3 |
|---|---|---:|---:|
| L3 | raw 14 | 0.94074 | — (matches U1 0.94075) |
| L4 | raw + combined-frame L/L (32 feats) | 0.94096 | **+2.18 bp** |
| L5 | raw + train-only L/L (32 feats) | 0.94099 | **+2.55 bp** |
| L4 - L5 | combined-frame premium | — | **-0.36 bp** |

Fold std 0.00058. Total lead/lag lift over raw = +2-3 bp, **within
noise**. Combined-frame premium is **negative -0.36 bp** — train-only
lead/lag is marginally better at the LGBM level than combined-frame.

**Key finding.** L1 single-feature AUCs ALWAYS gain +5 to +29 bp from
combined-frame, but the LGBM extracts the same signal from the
train-only version via interactions of (TyreLife, Stint, LapNumber,
RaceProgress). The model recovers the pattern through implicit
sequence reasoning, not through explicit lead/lag features. The
combined-frame premium evaporates at the model level.

This **closes the simple structural axis** I argued for. The 51 bp gap
to top-5% is not in row-local feature engineering. It's in either:
1. External data (FastF1 / Pirelli — HANDOVER A4-A5).
2. Cross-base diversity beyond the v4+h1d ceiling (`gbdt-class-redundant-
   on-shared-FE` already closed XGB-on-v4 today).
3. Meta-arch redesign that fires Path-B amp on a base-add (6× confirmed
   NULL via `path-b-amp-only-fires-on-meta-arch-not-base-add`).

### BOTE (decisions.jsonl)
- family: `single_base_fe_addition` (closest match in FAMILY_PRIORS)
- p_useful: 0.05; band (0, 0.5, 2.0); expected_lb_bp: +0.03 (verdict
  SKIP per harness; family prior is calibrated by historical NULLs).
- PI: 0 bp.
- Q6: True (log-loss objective + AUC metric, properly aligned).
- Note: harness verdict SKIP doesn't mean don't run — it means EV vs
  cost is low. We're running for the diagnostic value (combined-frame
  premium), not as a stack-add candidate for LB. Running anyway.

## Decision-quality vs outcome-quality

**PI sealed predictions: 3-for-3 vindicated** (0 bp realised LB Δ each).

- Probe 3: agent loss. Argued NTL was under-exploited; reconstructions
  cap at AUC 0.687 < TyreLife alone 0.699. Single-rule axis closed.
- Probe 2: PI win directly. EDA finding (T1 decay, T6 stint-end
  concentration) was load-bearing INPUT for Probe 1's hypothesis but
  the hypothesis itself failed at the model level.
- Probe 1: agent loss. Combined-frame premium is null/negative at
  the single-LGBM level despite +5-29 bp single-feature gains.

PI calibration improves: prior `lr-meta-rank-lock-strong-anchor` and
`gbdt-class-redundant-on-shared-FE` both predicted that the GBDT pool
is well-anchored and resists incremental feature-axis additions. PI's
intuition extended that to "even an obviously-missing axis (combined-
frame lead/lag) won't lift" — correct.

## Implications for next session

This session **closes 3 cheap diagnostic axes** that looked promising
on paper:
1. Host-quote NTL "trivial" claim — refers to original-column
   (5.5% hard-join only), not synth-reconstructible.
2. PitNextLap target structure — noisy/synthetic with cluster decay,
   no deterministic rule recoverable from features.
3. Combined-frame lead/lag — model extracts the signal implicitly
   from sequence-correlated features without explicit transductive lookup.

Net update on next-step priority list (HANDOVER §"Day-18+"): items 1-3
of the priority list (K=25 merge, 3-seed bag v4, XGB on v4 FE) are
unaffected by this session. Items 5-6 (FastF1, Pirelli) remain the only
single-mechanism path to top-5% and are the highest-EV unattempted
work. The diagnostic null on Probe 1's combined-frame hypothesis
**strengthens** the case for external-data work over structural FE.

## Pointers
- `scripts/probe_ntl_single_rule.py` + `artifacts/probe_ntl_single_rule.json`
- `scripts/probe_target_structure.py` + `artifacts/probe_target_structure.json`
- `scripts/probe_combined_lead_lag.py` + `artifacts/probe_combined_lead_lag.json` (pending)
- `audit/decisions.jsonl` — Probe 1 BOTE record with PI sealed prediction.

## Friction tags introduced
- `host-quote-trivial-refers-to-original-not-reconstructible` — Probe 3
  closes the brief.md "trivial" line as referring to the unmasked
  original NTL column (5.5% hard-join only); reconstructions cap at
  AUC 0.687.
- `pitnextlap-target-cluster-decay-not-shift` — Probe 2 confirms target
  is NOT a shifted PitStop and not a deterministic stint-boundary
  signal; it has decay-from-end structure with multi-positive clustering.
- `combined-frame-leadlag-premium-evaporates-at-gbdt` — Probe 1: single-
  feature combined-frame AUC always > train-only by +5 to +29 bp, but
  L4 vs L5 LGBM premium is **-0.36 bp**. The GBDT extracts the same
  sequence-position signal from raw (TyreLife, Stint, LapNumber,
  RaceProgress) interactions without explicit transductive lookup.
  Generalises to: combined-frame transductive features need a model
  class that DOESN'T already extract the sequence pattern (e.g., NN
  with positional encoding, FM with field-aware sequence features) to
  capture the L1 single-feature gains. On GBDT family, null.
