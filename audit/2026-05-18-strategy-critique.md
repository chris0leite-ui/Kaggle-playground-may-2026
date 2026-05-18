# Strategy critique — Day-18 EOD / plateau (2026-05-18)

Triggered by: plateau (R7+R8 = 1 WIN + 3 marginal + 3 NULL across 7
Path-B segmentation variants; PRIMARY LB unchanged R7→R8) AND EOD
auto-fire (Rule 14). Plateau-mode ordering: section 5 first.

## 5. Headroom math vs plan

- PRIMARY OOF 0.954471, LB **0.95389**
- Top-5% boundary **0.95405**; gap **+1.6 bp**
- Leader **0.95476**; gap **+8.7 bp**

Priority-queue lift midpoint × probability-real:

| Lever | midpoint | P(real) | EV bp |
|---|---:|---:|---:|
| C1 OpenF1 per-Race scalar join | +0.15 | 0.30 | +0.045 |
| DAE v2 (deeper + masked-column) | +0.20 | 0.20 | +0.040 |
| Public-notebook scan (blocked) | +0.10 | 0.30 | +0.030 |
| R8 60/20/20 rank-blend hedge | +0.00 | TIE_ZONE | 0 (public) |
| R7.2 fold-bag hedge | +0.00 | TIE_ZONE | 0 (public) |

Sum optimistic Σ = **0.115 bp** EV; 50% additivity discount =
**0.058 bp**. **Discounted Σ << headroom (1.6 bp)**. Structural
shortfall — the priority queue **mathematically cannot reach
top-5%**.

**Required action**: research-loop must surface a NEW mechanism
class with +1+ bp LB potential OR strategic posture pivots from
"reach top-5%" to "maximise hedge ladder for final-window R7d".

**Contingency** if research-loop returns thin: accept top-X%
(not top-5%) outcome; redirect remaining compute to
private-LB-variance hedge candidates (multi-seed bagging,
cross-mechanism rank-blends, alternative operator-pool composition).

## 1. Per-segment failure map

PRIMARY OOF AUC = 0.95447. Bottom segments dragging the mean:

| segment | n_rows | AUC | Δ vs mean | prior |
|---|---:|---:|---:|---:|
| **MEDIUM × Stint 2** | 25,363 | 0.8975 | **−319.5 bp** | 0.448 |
| SOFT × Stint 4 | 6,222 | 0.9000 | −293.8 bp | 0.177 |
| MEDIUM × Stint 3 | 12,217 | 0.9074 | −220.0 bp | 0.278 |
| Spanish GP | 20,483 | 0.9176 | −321.0 bp | 0.320 |
| Bahrain GP | 19,535 | 0.9215 | −282.2 bp | 0.288 |
| Year 2022 | 82,989 | 0.9200 | −156.5 bp | 0.267 |
| HARD (overall) | 170,518 | 0.9372 | −57.0 bp | 0.328 |
| Stint 2 (overall) | 129,536 | 0.9229 | −109.7 bp | 0.391 |

**Surfaced gap**: MEDIUM × Stint 2 (5.8 % of train, 44.8 % prior,
AUC **0.897**) is the single biggest learning opportunity. Path-B
segmented by Compound × Stint TREATS this segment but the model
still underperforms there. Per-segment FE specialised on
MEDIUM × Stint 2 (e.g., Position_Change × Cumulative_Degradation
interactions, MEDIUM-S2-conditional TyreLife splines) is a
**completely untried** mechanism axis. Predicted standalone-OOF
lift on the segment: +20-50 bp; predicted global OOF lift on K=13
+ Path-B meta: +0.05-0.20 bp.

## 2. Calibration

Brier = **0.0667**, ECE (10 equal-width bins) = **0.0011**. Very
well-calibrated globally. Local bin gaps:

| bin | n | p_pred | p_true | gap bp |
|---|---:|---:|---:|---:|
| [0.6, 0.7) | 14,371 | 0.6512 | 0.6423 | **+89** (over) |
| [0.7, 0.8) | 17,052 | 0.7514 | 0.7579 | **−64** (under) |
| [0.4, 0.5) | 13,037 | 0.4498 | 0.4467 | +32 |
| [0.9, 1.0) | 18,853 | 0.9400 | 0.9363 | +36 |

**Verdict**: no lift for AUC LB (rank-invariant). Calibration
adjustment doesn't change ranks. Skip — no isotonic / Platt fix
required.

## 3. Disagreement localization (K=13 pool)

Closest base pairs (ρ_OOF):
- yekenot ↔ K27_100k 0.988
- seg_fe ↔ HMM 0.987 (R4 mechanism-orthogonality pair — still ρ 0.987 at OOF)
- cb_v4 ↔ K27_100k 0.981

Most diverse pairs:
- qAT ↔ qAF 0.390
- qAF ↔ qAK 0.479
- qAO ↔ qAF 0.519
- (qAX family carries the pool's internal diversity)

Row-level rank-spread (std of normalized rank across 13 bases):
q50 = 0.121, q95 = 0.219, q99 = 0.262.

**Top-5 % rank-disagreement subset** (21,957 rows):
- PRIMARY AUC = 0.9547 — *better* than the complement (0.9541)
- class prior 0.099 (half of overall 0.199)
- Compound mix: SOFT 1.83× over-represented
- Stint mix: S3 1.18×, S4 1.51× over-represented

**Surfaced gap**: disagreement-difficult rows ≠ PRIMARY-difficult
rows. The MEDIUM × Stint 2 weakness is **NOT** captured by
inter-base disagreement — every base in the pool gets it wrong
in the same direction. Adding more bases of the same kind will
not lift this segment; the lift requires NEW row-feature signal
specific to MEDIUM-S2 dynamics.

## 4. Unexploited structural-finding scout

`comp-context.md` `structural_findings` (4 entries) all settled
or noise-floor framing — pitstop_pitnextlap_match_rate 0.724 (≈
chance baseline), lead_pitstop AUC 0.512 (no leak), train/test
iid row-level, test_lead_pitstop_computable_pct 0.974. Nothing
unexploited there.

**New structural finding (this critique)**: MEDIUM × Stint 2 is
a **44.8%-prior high-volume low-AUC segment** invisible in the
current FE / pool. Mechanism class to test: per-segment LightGBM
sub-model + stack as new base. Per-segment FE was last attempted
in R4 (`r4_segment_fe.py` — 9 interaction features) but targeted
the WET+S1, INTER+S2, VET-driver weak segments diagnosed BEFORE
the R5 Path-B + slim-kNN rebuild. Refresh with current PRIMARY's
weak-segment list.

## Concrete plan re-rank

Re-ranked priority queue (was: C1 / DAE v2 / notebook / hedges):

1. ~~**MEDIUM-S2-targeted FE**~~ — **REFUTED** by 5-min probe (this
   session). Specialised LightGBM on MEDIUM-S2 subset only (25k
   rows, 18 features incl. 6 targeted interactions: TyreLife ×
   Position_Change, TyreLife × CumDeg, PosChange × CumDeg, etc.)
   yields OOF AUC **0.881** — 16 bp BELOW PRIMARY's 0.8975 on the
   same subset. The pool (K=13 trained on 439k rows) extracts more
   signal on this segment than a 25k-row specialist can. MEDIUM ×
   Stint 2 is **at noise floor for row features**; the 20% synthetic
   noise concentrates here (44.8 % prior + low-AUC + can't-beat-
   pool = noise floor signature). **Drop this candidate.**

   Implication: Section 1's failure map identifies *intrinsically
   hard* segments, not *feature-deficient* segments. The lift surface
   is NOT in segment-specialised FE.

2. **Race-axis targeted FE** — similar diagnostic needed. Spanish
   / Bahrain / Emilia GP weakness might be intrinsic (e.g., wet
   conditions, safety-car-heavy) rather than fixable; race-aware
   specialist probe (~5 min CPU) should run before any FE work.
   **Status: untested; lower priority post-probe.**
3. **Research-loop fires** (mandatory per Rule 7; plateau triggered).
   Section 5 verdict says queue Σ << headroom — need a new
   mechanism class. Loop dispatched 2026-05-18; 3 agents in
   flight (notebooks, prior-comp, domain). Output → next-session
   experiment queue.
4. **C1 OpenF1 per-Race scalar join** (unchanged, ~45 min CPU).
5. **DAE v2 architecture** (unchanged, ~3h Kaggle T4).
6. **Public-notebook scan** — still BLOCKED on kaggle CLI 401;
   needs creds refresh first.

**Strategic posture pivot**: from "queue priority 1+2+3 will get
us to top-5%" to "queue priority is exploratory; reach top-5%
requires a +1 bp mechanism class that's not yet on the queue
(i.e., research-loop output) — AND the segment-FE thesis (the
plausible 'easy' lever from Section 1) is refuted, so we lean
harder on external scouts."

## Delta vs prior critique

No prior strategy-critique doc exists for this comp (Rule 14
auto-trigger missed at earlier plateaus; friction
`research-loop-overdue` from 2026-05-18 documented the same
gap). All sections above are NEW findings; the per-segment
MEDIUM × Stint 2 surface, the seg_fe ↔ HMM ρ=0.987 despite
mechanism-orthogonality, and the disagreement-≠-difficulty
finding are all first-time observations.
