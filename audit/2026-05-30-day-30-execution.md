# Day-30 audit — 2026-05-30

Starting state: PRIMARY = V8 LB 0.95456, rank 21/2791.

Plan reference: `/root/.claude/plans/it-is-the-next-hazy-newt.md`.
Pre-built candidates per `scripts/r22_d30_candidates.py` and
`scripts/r22_distill_stack.py`.

## Submissions

| Slot | Cand | File | R27 ρ vs V8 | LB | Δ vs V8 | Note |
|------|------|------|-------------|----|---------|------|
| 1 | I2 | submission_d30_R22I2_raunakdey90_d22raw_10.csv | 0.9974 | 0.95379 | **−77 bp** | Raw d22 at 10% catastrophic; K=20 chassis is necessary filter |
| 2 | W3 | D30_W3_A80_P20.csv | 0.99992 | **0.95457** | **+1 bp** | V8 slightly P-underweight in chassis |
| 3 | I3 | submission_d30_R22I3_raunakdey90_d21raw_10.csv | 0.9978 | 0.95384 | −72 bp | Same lesson as I2; raw NN axis class is too noisy |
| 4 | W2 | D30_W2_A85_P15.csv | 0.99998 | **0.95457** | **+1 bp** | TIE with W3 — plateau confirmed at P=0.15 |
| 5 | W4 | D30_W4_A75_P25.csv | 0.99981 | **0.95457** | **+1 bp** | TIE with W3 — plateau extends to P=0.25 |
| 6 | M1 | D30_M1_A85_C5_C7_R31.csv | 0.99998 | **0.95457** | **+1 bp** | TIE — multi-ours adds no new info beyond C5 |
| 7 | L1 | D30_L1_LOGIT_A90_P10.csv | 0.99996 | **0.95457** | **+1 bp** | TIE — logit-space matches linear plateau |
| 8 | E3 | D30_E3_PMEAN_q3_A90_P10.csv | 0.99987 | 0.95456 | 0 | Cubic mean ties V8 — curvature doesn't help |
| 9 | C1 | D30_C1_CONFCOND_A_mid_P20_ext_P05.csv | 0.99994 | 0.95456 | 0 | Conf-cond ties V8 — heuristic non-linear doesn't help |
| 10 | HOLD | — | — | — | — | Plateau firm; reserve for Day-31 final-pick re-verify |

## Day-30 summary

**New PRIMARY plateau at LB 0.95457 (+1 bp vs V8 0.95456).**
5 tied candidates: W2, W3, W4, M1, L1.

**Rank moved from #21 → #20 of 2791 (top-0.72%).**

## FINAL pick recommendation

- **FINAL_PRIMARY = W2 (A=0.85 P=0.15)** — smallest weight perturbation
  from V8 (known to generalize), simplest formula, lowest overfit risk.
- **FINAL_HEDGE = V0 (raw raunakdey, LB 0.95453)** — naturally R2d-compliant
  (4 bp inside 30 bp cap), structurally distinct from W2 (zero ours weight),
  hedges against "our K=20 axis doesn't generalize" scenario.

## Branch decisions

**Slots 1-3:** "W3 wins, I2/I3 regress" (not in original matrix but cleanly
determined). Raw NN axis closed; W-sweep direction opened.

**Slots 4-6:** Confirmed V8-family plateau at 0.95457 across P=0.15-0.25
and multi-ours variant. Direction saturated.

**Slots 7-9:** Tested sophistications. L1 (logit) matched plateau; E3
(cubic) and C1 (conf-cond) regressed to V8 level — non-linearity costs
slightly. Plateau is firm in linear-blend space only.

## Key lessons

1. **Raw NN axes only work through chassis filtering.** d22 raw at 10%
   = -77 bp; same d22 routed through K=20 PathB at 10% = +1 bp inside V8.
   The chassis IS the filter; raw NN is too noisy for direct blending.

2. **V8 family has a flat plateau at +1 bp.** P=0.15 to P=0.25 all give
   0.95457. The 1-bp lift is the realistic ceiling of mixing C5 (or
   structurally similar multi-ours) with the public anchor.

3. **Curvature-based blends (cubic mean, conf-cond) don't help.** The
   ranking structure is well-suited to linear weighting; sophisticated
   blends slightly regress.

4. **Top-10 boundary (0.95466) requires +9 bp more lift not achievable
   from blend tuning alone.** Would need a substantially stronger
   original-work model or true OOF-aligned meta-stacking.

## Branch decisions

**After slots 1-3:** "W3 wins, I2/I3 regress" branch (not in original
matrix but cleanly determined). Raw NN axis closed; W-sweep direction
opened. Slots 4-6 fire weight-curve bisection (W2/W4) + multi-ours
control (M1).

**Key lesson:** the d22 axis only adds value when routed through the
K=20 PathB chassis. Raw d22 at 10% blend weight overshoots noise
tolerance by ~25×. V8's structural success ≠ d22 IS the lift; the
CHASSIS that contains d22 is the lift.
