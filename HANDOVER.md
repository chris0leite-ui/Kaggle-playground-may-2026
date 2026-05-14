# HANDOVER

Next-session brief. **PI says "handover"** → agent reads this file
and proceeds. **PI says "prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

This file is rewritten (not tail-appended) every wrap-up. Prior
versions: `audit/archive-YYYY-MM-DD-handover-*.md`.

---

## Where we are

**Active PRIMARY: rank-blend 70/30 K=11 + K=9. LB 0.95386.**
Set 2026-05-12; reconfirmed 2026-05-14 after 4 NULL submissions.
+3.5 bp lift over the original K=4 PRIMARY (LB 0.95351).

- Top-5% boundary 0.95405 → gap **−1.9 bp**.
- Leader 0.95476 → gap **−9.0 bp**.
- Bootstrap CI on a 20% public draw is ±12 bp wide, so both gaps
  fall partly inside public-LB sample noise. Cross-submission
  relative deltas trust to ~±1 bp.

Submissions: **42 / 270** total; **4 used 2026-05-14**. Comp-day
**14 of 31**; days remaining **17**. Daily cap 10
(`comp-context.md: submission_budget`).

File: `submissions/submission_blend_K11_K9_w_70_30.csv`.

## Strategic state — saturation regime

Five distinct mechanism classes tested overnight 2026-05-14 all
NULL or REGRESSION (K=12 wide-ρ base, observable lead-feature,
τ-trio harness, K=11 tree recalibrator, adaptive blend). Audit:
`audit/2026-05-14-overnight-iteration.md`.

Strategic conclusion: K=11 + LR-meta + Path-B is at or near the
**Bayes-optimal ceiling** for row-feature prediction on this
synthetic dataset (synth generator decouples PitStop/PitNextLap
with ~20% stochastic disagreement).

Next-session pivots to **NEW-INFORMATION mechanisms** (cross-domain
training, multi-seed bagging of the full pipeline, Bayesian
group-prior with smoothing) rather than weight/feature refinement
on the existing K=11 stack.

## 🔴 Critical: held submissions — DO NOT submit

Day-17 strict fold-safe audit collapsed all target-reformulation
single-add results 88-100% (Rule 24 origin). Files still on disk
but invalidated:

- `path_b_K22_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K25_megapool_tau{5k,20k,100k}.csv`
- `path_b_multilevel_τ_*.csv`

Origin: `audit/2026-05-06-target-reform-leakage-audit.md`.

## Read order on session start

1. `CLAUDE.md` — rules + pointers.
2. `state/current.md` — current PRIMARY, LB ladder, axes status.
3. **THIS FILE** — held-submissions warning above.
4. `state/hypothesis-board.md` — open ideas, killed list.
5. `audit/2026-05-14-overnight-iteration.md` — most recent state.
6. `audit/friction.md` — current-week friction summary.

For detail when needed:

- `state/mechanism-ledger.md` — every mechanism family probed.
- `state/calibration-ladder.md` — OOF / LB anchors per family.
- `audit/2026-05-06-target-reform-leakage-audit.md` — Rule 24 origin.
- `glossary.md` — abbreviations, frozen short-codes.
- `audit/friction-archive.md` — full historical friction (1,450 lines;
  do not read by default).

## Empirical transfer bands (Rule 27, 2026-05-14)

Encoded in `scripts/probe_blend_harness.py::RULE_27_*_THRESHOLD`:

| Band | ρ_test vs PRIMARY | Expectation |
|---|---|---|
| TIE_ZONE | ≥ 0.9999 | LB ties within ±0.05 bp |
| OK transfer | 0.999 ≤ ρ < 0.9999 | Sub-bp to few-bp LB movement |
| REGRESSION_RISK | < 0.999 | Wide-ρ adds overfit CV patterns |

The K=12 result (cross-val gate +18.194 bp; LB −15.4 bp; ρ_test
0.928) is the cleanest demonstration of the cross-validation-gate
transfer-trap at this saturation level.

## Next-session first actions (EV / cost order)

1. **Blend operator sweep** — arithmetic / geometric / log-odds /
   trimmed mean across the 4 LB-confirmed PRIMARYs. Cost 5 min.
2. **3-way and 4-way blends of LB-confirmed PRIMARYs** —
   hill-climb the weights. Cost 1 min per blend.
3. **Path-B C×S on fresh 27-base + slim-kNN union (τ=100k only)** —
   qBG attempted this, timed out at 16+ min. Skip τ=5k / τ=20k to
   halve compute.
4. **R5 hedge ladder preparation** — list OOF-best candidates that
   were rejected for LB regression. 30 min.
5. **NEW-INFORMATION mechanism scouting** — if (1)-(3) are NULL,
   trigger Research-loop (loops.md). Three parallel research agents
   + grep-ledger dedup.

## Falsified / dead — do not retry

Deduplicated list in `state/hypothesis-board.md ## Killed`.
Highlights:

- All target-reformulation single-add variants (leaky).
- All `path_b_*_invlaps_*` and `path_b_*_megapool_*` candidates.
- Multi-level 4-tier per-segment stacker.
- Day-16 virgin-axes round (11 of 11 null).
- TabPFN v2.5 / v2.6, 16+ field FMs, drop-GBDT pool refactor.
- Simple K=21 blending, α-calibrated τ-resweep, multi-target NN,
  masked-column self-prediction, twin-pool 2-meta blend, conformal
  isotonic, AV-weighted LightGBM, Yao/Vehtari covariance-modulated
  per-segment stacker, non-LR meta on K=4 (LightGBM/MLP/RF), kernel
  SVM family (8 variants), kNN-target-mean variants V5/V6.
- qBA Manhattan distance kNN (LB regress despite +0.13 bp OOF).
- K=34 unrolled at any C-sweep value (over-parameterises LR-meta).

## Today's calibration data (2026-05-14)

| Probe | OOF Δ | LB | Lesson |
|---|---:|---:|---|
| K=8 rebuilt (3-of-6 slim-kNN + K=27) | n/a | 0.95382 | Reproduction tie (ρ 0.999901). |
| Blend 70/10/20 K=11/K=10/K=27 | +0.065 | 0.95386 | Harness top-1 in TIE_ZONE. |
| K=12 + control LightGBM | +18.194 | **0.95232** | Cross-val gate fooled by wide-ρ (0.928); REGRESSION_RISK band confirmed. |
| Blend 60/15/25 K=11/K=10/K=27 | +0.059 | 0.95386 | Deeper-in-OK-zone tie. |
