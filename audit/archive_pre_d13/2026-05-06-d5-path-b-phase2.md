# Day-5 Path B Phase 2 — first META-LEVEL lift on Day 5

Phase 1 (e3_hgbc MVP) cleared both Phase-B gates. Phase 2 rebuilt 5
more fast CPU GBDT bases on (train ∪ pseudo-test) and stacked them
with 8 unchanged originals into a K=14 partial-pseudo M5q.

## Per-base rebuilds (Phase 2)

| Base | Anchor | Rebuilt OOF | Δ bp | ρ vs orig | Wall |
|---|---:|---:|---:|---:|---:|
| baseline_two_anchor (LGBM) | 0.94075 | 0.94265 | **+19.0** | 0.99594 | 75s |
| m2_xgb | 0.94507 | 0.94639 | **+13.3** | 0.99520 | 133s |
| e5_optuna_lgbm | 0.94736 | 0.94792 | **+5.6** | 0.99709 | 317s |
| f1_hgbc_deep | 0.94870 | 0.94914 | **+4.4** | 0.99532 | 87s |
| f2_hgbc_shallow | 0.94861 | 0.94882 | **+2.1** | 0.99656 | 122s |
| e3_hgbc (Phase 1) | 0.94876 | 0.94917 | +4.1 | 0.99593 | 125s |

**Every base lifts.** Smaller-anchor bases lift more (headroom for
pseudo signal compresses against ceiling). Per-base ρ is below 0.998
universally — pseudo channel meaningfully shifts each base's test
rank ordering.

## Partial-pseudo K=14 M5q stack

6 pseudo-rebuilt + 8 original bases, LR meta with expand().

| Quantity | Value | Anchor | Δ |
|---|---:|---:|---:|
| Strat OOF | **0.95082** | M5q 0.95057 | **+2.54bp** |
| ρ vs M5q test | **0.99836** | tie threshold 0.999 | REAL_DELTA |

This is the **first non-null meta-level Day-5 result**. Six prior
nulls (path-c standalone-stack, K=15 LR re-stack 3 variants, K=15
GBDT-meta 3 variants, TabNet smoke) confirmed the meta-add and
meta-switch ceilings against the original M5q pool. Path B works
because it changes the POOL composition itself — every base sees
new training data.

## L1 reshuffling — the pseudo channel rerouted meta weight

| Base | M5q L1 | partial-pseudo L1 | Δ |
|---|---:|---:|---:|
| cb_slow-wide-bag (orig) | 1.060 | 0.295 | **−72%** |
| a_horizon (orig) | 0.658 | 0.096 | **−85%** |
| d2a_te (orig) | 0.671 | 0.370 | −45% |
| f1_hgbc_deep (pseudo) | 0.357 | 0.602 | **+69%** |
| baseline (pseudo) | 0.343 | 0.573 | **+67%** |
| e3_hgbc (pseudo) | 0.245 | 0.530 | **+116%** |
| m2_xgb (pseudo) | 0.178 | 0.409 | **+130%** |

The original "high-L1 anchor" bases (`cb_slow-wide-bag`, `a_horizon`,
`d2a_te`) were carrying meta weight that the pseudo-rebuilt bases
now subsume. The HGBC variants (e3, f1, f2) and `baseline` quadruple
or double their L1 weight. Two readings:

1. The pseudo channel makes the simpler GBDTs **as informative** as
   the prior anchor bases — the rank-corrections that previously
   came from the slow CatBoost / target-reformulation bases now
   come from any sufficiently-trained LGBM/HGBC fed pseudo data.
2. Rebuilding the slow bases (Phase 3 candidate list) would
   compound this: cb_slow-wide-bag with pseudo could push back its
   L1 weight at the expense of its noisier rebuilt partners.

## Phase 3 candidates (slow + GPU rebuilds)

Decision rule MET: partial-pseudo OOF (0.95082) > M5q + 1bp (0.95068).
Expand to slow rebuilds. EV-ranked queue:

1. **CatBoost CPU bases** (~30-60min each, run sequentially):
   `cb_slow-wide-bag` (Kaggle GPU bag in original; CPU rebuild
   without bagging is fastest first probe), `cb_lossguide`,
   `cb_year-cat`, `e1_catboost_sub`. Combined original L1 was 1.66
   in M5q. If rebuilds reroute even half, +5-10bp meta lift plausible.
2. **RealMLP rebuild** on Kaggle GPU T4x2 (~6h overnight). The
   RealMLP base in M5q gave the original +14bp LB lift. Pseudo-
   rebuild expected to compound but variance is real. Kernel push
   tomorrow morning.
3. **d2a_te rebuild** with pseudo-aware TE (TE within outer-train
   only; pseudo rows get TE values from original mappings, NOT from
   their own fold-out TE). Conservative, ~10min CPU.
4. **a_horizon / b_lapsuntilpit rebuild** — these target
   reformulations don't take pseudo-PitNextLap directly. Could
   reconstruct PitInNext3Laps via M5q's predictions on neighboring
   laps (sequence reconstruction). Defer — high effort, modest EV.

## Slot 1 candidate (today)

**`submissions/submission_d5_partial_pseudo_m5q.csv`** — the K=14
partial-pseudo M5q LR-stack. ρ=0.99836 vs M5q test (REAL_DELTA);
Strat OOF +2.54bp. Pre-submit-diff verified-able. LB delta is
genuinely uncertain: the d4 calibration ρ=0.99508 → −4bp LB does
not extrapolate cleanly to ρ=0.99836 with positive OOF lift.

Plausible LB outcomes: −2 to +5bp from M5q's 0.95005. With 0/10
slots used today and 22 days remaining, this is a high-information
single-shot probe.

## Held artifacts

- `oof_d5_<base>_pseudo_strat.npy` × 6 (e3 from Phase 1, 5 from Phase 2)
- `test_d5_<base>_pseudo_strat.npy` × 6
- `oof_d5_partial_pseudo_m5q_strat.npy`, `test_d5_partial_pseudo_m5q_strat.npy`
- `submissions/submission_d5_partial_pseudo_m5q.csv`
- `scripts/artifacts/d5_pseudo_phase2_results.json`

End — 90 lines.
