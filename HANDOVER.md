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
with ~20% stochastic disagreement). Per Rule 4, this claim was
re-interrogated 2026-05-18 via a formal Research-loop; result is
4 deduped untried candidates (see "Research-loop completed"
section below) — the "ceiling" is row-feature-exhausted but NOT
task-framing- or information-source-exhausted.

Next-session pivots to the **Tier-A batch** (8 unconsumed picks
from 2026-05-08 FE research + 2 new candidates from 2026-05-18
loop) before returning to NEW-INFORMATION mechanisms.

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

## Empirical transfer bands (Rule 27)

See `state/current.md` for the full table. TIE_ZONE ≥ 0.9999;
OK transfer [0.999, 0.9999); REGRESSION_RISK < 0.999.

## 2026-05-18 — Research-loop completed (Rule 7)

This session ran a fresh Research-loop on the post-2026-05-14
plateau. Three parallel agents + dedup against the ledger. Outputs:

- `audit/research/2026-05-18-notebooks.md` — null harvest (Kaggle
  pages reCAPTCHA-walled; next plateau scan should use authenticated
  `kaggle kernels list -c <slug> --sort-by voteCount`).
- `audit/research/2026-05-18-prior-comp.md` — Porto Seguro / IEEE-CIS
  Fraud / Otto analogues; quoted lifts (downscale for our saturation).
- `audit/research/2026-05-18-domain.md` — arXiv 2512.00640 state-space
  tyre-deg paper, TUM FTM features, OpenF1 per-Race aggregate join
  is fold-safe at the (Race) level.
- `audit/research/2026-05-18-research.md` — SYNTHESIS. 4 new
  deduped candidates (C1-C4); 6 confirmations of Tier-A2/A3 picks
  already on the menu.

Strategy-critic-loop is **still pending** (needs `data/` and
`scripts/artifacts/` populated; this session ran in an ephemeral
container with empty disk). The next session-with-bootstrap should
fire it before any compute.

## Next-session first actions (EV / cost order — REVISED 2026-05-18)

1. **Bootstrap** (`bash bootstrap.sh`) to pull `data/` + the
   `chrisleitescha/s6e5-artifacts` dataset into
   `scripts/artifacts/`. Required for everything below.
2. **Strategy-critic-loop** (`strategy-critic.md`, ~30 min) — runs
   the 5-question template on PRIMARY OOF. May re-rank the Tier-A
   batch below.
3. **Tier-A feature batch** (~110 min CPU; 1-2 submission slots):
   - C3 per-(Race, LapNumber) Bayesian shrinkage post-process (5 min).
   - EXP-A3-1 rank-sorted gaps (18 min).
   - EXP-A3-3 lagged DriverAheadPit + tirechange_pursuer (10 min).
   - EXP-A3-4 Heilmeier residual (8 min + per-fold).
   - EXP-A3-2 per-track fuel coef (6 min + per-fold).
   - EXP-A2-3 nested-fold TE 3/4-way (12 min).
   - C4 UID magic-features as LGBM base (25 min).
   - EXP-A3-6 KNN-target-mean-500 (20 min).
   - EXP-A2-7 F3 competitor field-state at K=11+1 (8 min).
   - EXP-A3-8 quantile/histogram groupby (15 min).
   - Each gates at K=11+1 plain LR-meta with G1-G4 + Rule-27.
4. **Follow-ups** (only if Tier-A clears a survivor):
   - Two-meta-in-parallel + Ridge top blend (~30 min).
   - Caruana forward-selection-with-replacement (~10 min).
   - C2 DAE with swap-noise (Porto Seguro pattern, ~2-3 hr T4×2).
   - C1 per-(Race) OpenF1 aggregate join (~45 min).
   - Cross-domain training on `d16_orig` (~30-60 min) — **first
     check whether Aadigupta dataset is already in Path-B via
     yekenot recipe** (see flag in `2026-05-18-research.md`).
5. **Hedge ladder maintenance** — list R5/R7 candidates per
   `state/current.md`. 30 min.
6. **Final-window-only**: EXP-9 gap-aware sequence transformer
   on Kaggle T4×2 (~4-6 hr) if Tier-A all-null and ≥5 days remain.

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

## Today's calibration data (2026-05-14, see audit for full)

See `audit/2026-05-14-overnight-iteration.md`. Headline: K=12 +
control LightGBM regressed -15.4 bp at LB despite +18.194 bp at
the cross-val gate (ρ_test=0.928 below the OK-transfer band).
