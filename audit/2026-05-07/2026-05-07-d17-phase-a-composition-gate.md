# Day-17 PM — d17 Phase-A composition gate (K=23/K=24 stack-add of d16 Phase-4 winners)

**Date.** 2026-05-07. **Branch:** `claude/read-handover-62BCt`.
**PRIMARY:** `d16_path_b_K22_continuous_only_tau20000` LB 0.95089 / OOF 0.951208.
**ISSUES leaf claimed:** 7f.

## Setup

Inherited from commit `1f442e8` (sibling branch
`claude/autoencoder-synthetic-data-pEMB6`): Phase A re-arranges the four
d16 Phase-4_v2 orig-LGBM feature-subset winners + cross-branch strict-OOF
inv_laps into K=22/K=23/K=24 stacks via canonical 5-fold LR meta. The
sibling session bailed mid-run after writing C1-C5 OOFs to disk; summary
JSON and C6/C7 were missing.

This branch re-ran `scripts/d17_phase_a_compose.py` to completion
(818 s wall, 1 core; 5-fold LR-lbfgs on 439k×3K-feature design matrix
per combo × 7 combos × 5 folds + baseline).

## Result table

Two PRIMARY columns: the **script-internal** PRIMARY (`oof_PRIMARY_K22_strat.npy`,
which still points to the OLD d15b DAE PRIMARY at OOF 0.95074, LB 0.95059)
and the **actual current** PRIMARY (`oof_d16_path_b_K22_continuous_only_tau20000_strat.npy`
at OOF 0.951208, LB 0.95089).

Predicted LB Δ uses `probe.predicted_lb_delta_bp(d_oof_bp, ρ_test)`:
ρ ≥ 0.999 → Δ−0.5 · ρ ≥ 0.995 → Δ−1.5 · ρ ≥ 0.99 → Δ−3.0.

| Combo | K | OOF | Δ vs d15b old (bp) | Δ vs d16 PRIMARY (bp) | ρ_test vs d16 PRIM | pred LB Δ |
|---|---:|---:|---:|---:|---:|---:|
| C1 cont | 22 | 0.95106 | +3.23 | **−1.45** | 0.99581 | −2.95 REGRESS |
| C2 cont+nolaptime | 23 | 0.95120 | +4.60 | −0.09 | 0.99557 | −1.59 REGRESS |
| C3 cont+notyrerp | 23 | 0.95122 | +4.80 | +0.11 | 0.99517 | −1.39 REGRESS |
| C4 cont+catonly | 23 | 0.95115 | +4.15 | −0.54 | 0.99515 | −2.04 REGRESS |
| C5 cont+invlaps_strict | 23 | 0.95107 | +3.27 | −1.42 | 0.97555 | −6.42 REGRESS |
| C6 cont+nolaptime+invlaps | 24 | 0.95122 | +4.78 | +0.09 | 0.97714 | −4.91 REGRESS |
| **C7 cont+nolaptime+notyrerp** | **24** | **0.95129** | **+5.50** | **+0.81** | **0.99506** | **−0.69** TIE |

Per-base |w| (sum-of-3 LR weights raw+rank+logit) on C7:
- d16_orig_continuous_only |w|=1.72 (dominant, matches d16 PRIMARY |w|≈1.5)
- d16_orig_no_laptime |w|=0.26
- d16_orig_no_tyrelife_rp |w|=0.44

## Key findings

### 1 — vs OLD d15b DAE PRIMARY: C7 looks dominant (+5.5 bp OOF)

The script's printed Δ_PRIMARY is computed against the stale
`oof_PRIMARY_K22_strat.npy` file (d15b DAE, OOF 0.95074), not the
current d16 cont_only Path B file. Read in isolation, C7's +5.50 bp
OOF + ρ 0.99760 looks like a strong PRIMARY-replace candidate
(predicted LB Δ +4.0 bp via probe.py band).

### 2 — vs ACTUAL d16 cont_only Path B PRIMARY: C7 is TIE

Once compared against the real current PRIMARY OOF 0.951208,
**none of C1-C7 LR-meta stacks beat it**. C7 is the strongest at
+0.81 bp OOF / ρ_test 0.99506 → predicted LB Δ −0.69 bp. C1 is **−1.45 bp
below current PRIMARY** because Path-B Compound×Stint τ=20k segmentation
on K=22 (cont_only) does +0.15 bp OOF more work than the canonical
LR-meta on the same pool, and adding 3 more orig-LGBM bases via LR-meta
does not close the gap.

This is the 5th cross-confirmation of friction tag
`path-b-amp-only-fires-on-meta-arch-not-base-add`: meta-arch
(per-segment hier-shrinkage) is the load-bearing mechanism, not pool
size. Naive K=24 LR-meta reasoning that "more bases ⇒ more lift"
overstates by ~5 bp here.

### 3 — Strict-OOF inv_laps adds nothing on top of cont_only

C1 vs C5: K=22 cont_only OOF 0.95106 → K=23 cont_only + inv_laps_strict
OOF 0.95107 (+0.04 bp). Strict-OOF inv_laps standalone gates were +0.234
bp K=21+1 per the d17-phase-0 audit; when stacked on top of cont_only
(which already explains most of the orig-data signal), the marginal
contribution collapses to noise. C5/C6 (with invlaps) underperform their
non-invlaps siblings by ~1 bp at lower ρ.

This refines `target-construction-layer-leakage` finding: even the
audit-cleaned strict-OOF inv_laps is not differentiated enough from the
orig-LGBM cont_only signal to warrant K_pool inclusion.

### 4 — Feature-subset orig-LGBM family is real but ceiling is near

C2/C3/C4 individually add +1.4 to +1.6 bp over C1 (K=22 cont_only LR-meta)
when paired with a different feature-subset variant. Two variants stacked
together (C7) add +2.3 bp over C1 — sub-additive (~70% synergy retention),
indicating the no_laptime / no_tyrerp signals are partially redundant
with each other but each carries some unique orig-data structure.

Implication: pool surgery isn't dead. The path forward to a real
PRIMARY-advance is **applying Path B Compound×Stint τ=20k over the K=24
pool**, not extending K=24 LR-meta further.

## Next step (NOT YET RUN — awaiting PI direction)

Build `scripts/d17_path_b_K24_C7.py` modeled on
`scripts/d16_path_b_K22_continuous_only.py` with K=24 base pool
(= K=22 + d16_orig_no_laptime + d16_orig_no_tyrelife_rp). Sweep
τ ∈ {5000, 20000, 100000}. Compare against d16 PRIMARY OOF/test directly.

Cost: ~15 min CPU (5-fold × 30-segment per-fold LR refit on K=24×3=72
features).

Family: `meta_arch_redesign` (FAMILY_PRIORS p=0.30, (1, 4, 8) bp).
Q6: log-loss objective, row-AUC aligned, True. PI sealed-prediction
required per Rule 26(a) before submission.

## Artifacts

- `scripts/artifacts/d17_phase_a_summary.json` — full per-combo summary
  with per-base |w|.
- `scripts/artifacts/oof_d17_C{1..7}_*_strat.npy` + matching `test_*` —
  C6, C7 newly produced this run.
- `scripts/artifacts/oof_d17_C7_K24_cont_nolaptime_notyrerp_strat.npy`
  is the C7 stack OOF; will become the input pool's effective K=24 OOF
  for the Path-B follow-up.

## Operating-rule check

- Rule 18 (issue-tree claim): leaf 7f claimed 96a871e; status `wip`
  pending Path-B follow-up.
- Rule 19a (BOTE): not run for the Phase A re-run because it was
  recovery (artifacts already on disk from `1f442e8`); BOTE will run
  before the K=24 Path-B follow-up.
- Rule 24 (fold-safe label-conditional aggregates): all C7 components
  pass — orig-LGBM with restricted features, no target-conditional
  aggregates. Strict-OOF inv_laps already audit-cleared (88% collapse
  ⇒ +0.234 bp legitimate).
- Rule 25 (transductive features need AV check): AV-AUC=0.502 globally,
  i.i.d. — pass.
- Rule 1 (single-shot submit, PI-approved): nothing submitted this
  session.

## Submissions used

0/10 today (this branch). Total: unchanged from handover (32/270).
