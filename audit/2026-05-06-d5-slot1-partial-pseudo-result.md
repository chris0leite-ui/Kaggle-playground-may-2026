# Day-5 slot 1: partial-pseudo K=14 — LB 0.94963 (−4.2bp), gap WIDENED

## Result

| Metric | M5q anchor | Partial-pseudo K=14 | Δ |
|---|---:|---:|---:|
| Strat OOF | 0.95057 | 0.95082 | **+2.54bp** (real) |
| LB (public) | 0.95005 | **0.94963** | **−4.2bp** (regressed) |
| OOF→LB gap | −5.2bp | **−12.0bp** | **gap widened by 6.8bp** |
| ρ vs M5q test | — | 0.998358 | REAL_DELTA |

## What this proves

Path B's pseudo channel **gains OOF AUC by amplifying M5q's
systematic biases** in the rebuilt bases. The OOF measure rewards
this (rebuilt bases now agree more with M5q's training-time error
structure). The LB penalizes it (M5q's biases don't generalize to
the held-out rank ordering).

L1 reshuffling confirms the mechanism:
- Pseudo-rebuilt bases (e3 +116%, m2_xgb +130%, baseline +67%, f1
  +69%) got DOUBLE / TRIPLE the meta weight they had in M5q.
- Original "anchor" bases that were NOT pseudo-rebuilt got DEMOTED
  (cb_slow-wide-bag −72%, a_horizon −85%, d2a_te −45%).
- But the ORIGINAL bases were the ones encoding clean LB-relevant
  rank structure. The meta optimized for OOF and chose wrong.

This is the HANDOVER's "overconfidence collapse" risk — pseudo
generated from M5q amplifies M5q's biases, OOF rewards the
amplification, LB does not.

## Falsified

- "Partial-pseudo K=14 (6 pseudo + 8 orig) breaks the M5q LB ceiling
  via base-pool reformation" → null. OOF +2.54bp, LB −4.2bp.
- "30bp-class move in prior comps when Path B lands" — does not apply
  to s6e5's broad pseudo-gate setup; gap widens instead of narrows.

## Implications for Phase 3 queue (CB rebuilds, d2a_te, RealMLP GPU)

Each Phase 3 rebuild uses the SAME pseudo-gate (180k/188k rows,
union of M5q-confidence + multi-base vote) that produced the
slot-1 gap-widening. Continuing to rebuild more bases on this gate
amplifies the same systematic error. **Phase 3 as designed is
EV-NEGATIVE.**

## Re-rankable next moves (require PI direction)

1. **Pivot: tighter pseudo gate.** Re-run Phase 1+2 with M5q ∈ [0.99,
   1.0] ∪ [0, 0.01] AND ≥12/13 multi-base vote (intersection, not
   union). Expected pseudo set: ~30-50k rows (vs 180k today).
   Hypothesis: tighter gate keeps the LB-relevant signal, drops the
   over-amplification. Cost: 1-2h CPU. 1-2 slot probes to calibrate.
2. **Pivot: pseudo via row weighting, not augmentation.** Train
   bases on full real-train + pseudo-test with pseudo rows getting
   sample_weight=0.3 (vs 1.0 for real). Reduces pseudo's OOF-gaming
   without losing the test-row exposure. ~Phase 2 wall + 5min.
3. **Park Path B entirely.** Pivot to multi-seed RealMLP bag
   (HANDOVER A.1, +1-3bp prior at known cost) or to truly orthogonal
   meta architectures. Day-5 calibration says s6e5's M5q LB is at
   or near the achievable ceiling for this base pool family.
4. **Re-test Path C with the pseudo-info gate disabled.** The
   recursive base was strong standalone (+92bp baseline) but null
   in K=15 because LR couldn't extract its row-correction signal.
   GBDT-meta over K=15 was −1bp uniformly. Two unmined variants:
   (a) 2-base [M5q, recursive] LB submit — different rank structure
   than the K=15 stack we tested.

## Held: do not push RealMLP pseudo-rebuild kernel until PI directs

The Day-6 HANDOVER plan called for pushing the RealMLP pseudo-rebuild
kernel as fork-3 part 2. The slot-1 LB result says this would burn
~6h Kaggle GPU credits chasing the same gap-widening. Held pending
PI re-prioritization.

End — 70 lines.
