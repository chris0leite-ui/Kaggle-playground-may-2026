# Row-feature structural ceiling — when to stop bargaining

**Date**: 2026-05-18 (Day-19 of comp; PRIMARY R7.1 LB 0.95389)

## Substance

2026-05-18 spent the entire day adding mechanisms to the K=13+Path-B
stack: R4 segment-FE, R4 HMM, R5 K=11+seg+HMM, R5.2 Path-B Compound
×Stint, R6 fold-fit bag, R7 DAE, R7 DriverClass×Stint, R8 multi-seg
sweep, R9 NB4, R9 C1.

The cleanest result of the day: **R9 falsified both EOD-critic
structural levers (TE-base AND external-data) within ~30 min CPU
each**:

- R9 NB4 (Compound × Stint target-mean as base learner) — standalone
  OOF 0.94850 (G1 PASS, yekenot-level), K=14+Path-B Δ vs R7.1 PRIMARY
  **−0.022 bp NULL**. Internal TE absorbed.
- R9 C1 (Aadigupta external per-Race FEATURE scalars; the only
  structural lever named in R8 EOD strategy-critic Section 5) —
  standalone OOF 0.94902 (G1 PASS, ~5 bp stronger than NB4),
  K=14+Path-B Δ vs R7.1 PRIMARY **−0.045 bp NULL**. External data
  absorbed *more* than internal TE.

The pool absorbs everything that lives in row-feature space — and
at this point the absorption is so consistent across operator,
mechanism class, and data class axes that it's not noise; it's a
STRUCTURAL CEILING.

## The pattern

Three structurally distinct mechanisms fell to rank-lock in <1 week:

1. **R7 swap-noise DAE** (operator family: embedding-class anchor swap)
   — absorbed at every Path-B segmentation tested.
2. **R9 NB4 TE-as-base** (mechanism class: segmentation-as-base
   instead of segmentation-as-meta-operator) — absorbed at K=14.
3. **R9 C1 external scalars** (data class: cross-domain feature
   injection from Aadigupta dataset) — absorbed at K=14, more
   strongly than NB4 because yekenot's TE_CONFIGS touch Race in
   5 of 6 entries.

Each of these was, in its own session, the highest-EV remaining
candidate. The fact that they all absorb at K=13+Path-B means the
pool has saturated the row-feature representation; adding more row
features just gives the meta-LR more collinear inputs to overfit.

## The takeaway

For future comps: when a pool absorbs 3 structurally distinct
mechanisms within a week (operator class + mechanism class + data
class), the right move is NOT to try a 4th row-feature variant.
It's to either:

A. **Commit to mechanism EXPANSION outside row-features**: seq2seq
   on lap sequences, graph models on competitor edges, survival /
   hazard models on stint life. These inject inductive biases the
   row-feature meta-LR cannot represent.

B. **Pivot to hedge-prep posture and accept top-X%**: maximise
   private-LB variance reduction via structurally-distinct ladder,
   accept that the row-feature ceiling is the public-LB ceiling
   for this synth generator.

R8 EOD already said this. R9 *proved* it. Next time, the third
null in a 3-axis sweep IS the auto-trigger for posture pivot —
don't wait for a 4th data point.

## Related friction

- `audit/friction.md`: `rank-lock-confirmed-three-axes` (2026-05-18).
  Promotion candidate for skill `kaggle-comp/self-improvement.md`:
  add 3-axis-rank-lock as a posture-pivot auto-trigger condition.
