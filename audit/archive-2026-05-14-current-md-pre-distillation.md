# Where we are right now

Updated whenever a new submission lands or the strategic picture shifts.
For history, read `audit/postmortems/` and `audit/research/`.

**Date convention:** prose in this file uses **ISO dates** (e.g.
"2026-05-08") or **comp-day-N** anchored to comp start 2026-05-01
(so today = 2026-05-08 = comp day 8). The `d13`..`d19` labels in
script names and audit prose are *frozen code/file prefixes*, not
calendar days — see `glossary.md` and `audit/friction.md`
under `day-counter-drift`.

## Status as of 2026-05-14 (overnight iteration on bootstrap branch)

**PRIMARY unchanged at LB 0.95386.** Four submissions used today (2026-05-14);
all five tried mechanism classes were NULL or REGRESSION. Audit:
`audit/2026-05-14-overnight-iteration.md`. Strategic conclusion:
K=11 + LR-meta + Path-B is at or near the Bayes-optimal ceiling for
row-feature prediction on this synthetic dataset. The synth generator
decouples PitStop and PitNextLap with a stochastic component (~20%
label disagreement on observable pairs), revealing a noise floor that
no row-level model can recover.

Today's submission ladder:
| Submit | Mechanism | OOF lift | ρ_test vs K=11 | LB |
|---|---|---:|---:|---:|
| K=8 rebuilt (3-of-6 slim-kNN + K=27) | reproduction | n/a | 0.999901 | 0.95382 |
| Blend 70/10/20 K=11/K=10/K=27 | harness top-1 (TIE_ZONE ρ) | +0.065 | 0.999955 | 0.95386 (tie) |
| K=12 = K=11 + control_logloss LGBM | wide-ρ base add | +18.194 | **0.928** | **0.95232 (-15.4 bp REGRESSION)** |
| Blend 60/15/25 K=11/K=10/K=27 | deeper-in-OK-zone | +0.059 | 0.99992 | 0.95386 (tie) |

Empirical transfer bands updated (Rule 27 recalibration): TIE_ZONE at
ρ_test ≥ 0.9999, REGRESSION_RISK at ρ_test < 0.999, OK transfer in
between. The K=12 result is the cleanest demonstration of the
cross-validation-gate transfer-trap at this saturation level.

## PRIMARY (active) — set 2026-05-12 (blend 70/30); RECONFIRMED 2026-05-14

**Score: 0.95386 on the public leaderboard.** Direct LB-confirmed.
**+3.5 bp lift over original K=4 PRIMARY** (LB 0.95351). +0.1 bp over
the K=11 + K=27 PRIMARY (LB 0.95385).

**What it is:** rank-blend at weights 0.7 / 0.3 of two LB-confirmed
predictions:
- K=11 + K=27 + Path-B τ=100k (LB 0.95385) — 70% weight
- K=9 qAX (qAT+qAV+qAO+qAA+qAF + Path-B τ=20k, LB 0.95375) — 30% weight

The cross-mechanism blend cancels small errors: K=11 uses K=27 super-base
+ slim-kNN; K=9 uses slim-kNN only. Their errors are partially
orthogonal even though ρ_test = 0.9998. PI-authorized "all 3" override
of Rule 27 abort threshold (ρ > 0.999) yielded LB 0.95386.

**Submitted:** 2026-05-12 08:14 UTC. LB 0.95386, public COMPLETE.
File: `submissions/submission_blend_K11_K9_w_70_30.csv`

## Calibration data — qBI experiments (2026-05-12)

Three candidates submitted to extract calibration data:

| Submission | OOF Δ | LB | vs K=11 |
|---|---:|---:|---:|
| qBI K=12 + qBA Manhattan + Path-B τ=100k | +4.161 | 0.95380 | -0.5 bp |
| qBI K=34 C=0.1 (tighter LR reg) | +4.237 | 0.95374 | -1.1 bp |
| **Blend 70/30 K=11+K=9** | — | **0.95386** | **+0.1 bp** |

Key learnings:
- **qBA Manhattan kNN HURTS at LB despite +0.13 bp OOF** — different
  distance metric introduces test-time noise vs euclidean kNN.
- **K=34 unrolled never transfers** — C-sweep (1.0, 0.1, 0.03, 0.01) all
  give LB 0.95373-0.95374. The 27 individual base predictions provide
  redundant signal at the meta layer; over-parameterization is real but
  regularization can't fix it.
- **Rank-blending different-mechanism PRIMARYs ESCAPES the rank-lock**
  even at ρ=0.9998 — small probability-rank differences in the K=9 vs
  K=11 predictions, when blended, cancel uncorrelated errors.

**Mechanism background:** see
`audit/2026-05-09/2026-05-09-qAK-breakthrough.md` and
`audit/2026-05-09/2026-05-09-final-results-summary.md`.

## Reference of prior PRIMARY for hedge-eligibility

The K=4 + Path-B C×S τ=100k at LB 0.95351 remains hedge-eligible per
Rule R7 if the new K=9 PRIMARY regresses on private LB. See
`submissions/submission_K4_PRIMARY_pathb_cs_tau100000.csv`.

## Leaderboard ladder (with new PRIMARY)

| ISO date | Pool | Score | What changed |
|---|---|---:|---|
| **2026-05-09 PM** | **K=9 + Path-B C×S τ=20k** | **0.95375** | **NEW PRIMARY**. orig-kNN K=1 6-axis cell + Path-B amp. +2.4 bp over K=4. |
| 2026-05-09 AM | K=5 (K=4 + V4 kNN-aug) + Path-B τ=100k | 0.95359 | V4 kNN-target-mean ingested at base level. |
| 2026-05-08 PM | K=4 + Path-B τ=100k | 0.95351 | Sparse-pool reduction. |
| 2026-05-07 PM | K=27 + Path-B τ=100k | 0.95368 | DGP-class bases. |
| 2026-05-07 AM | K=23 + Path-B τ=20k | 0.95154 | First with CB-yekenot + RealMLP-yekenot. |
| Earlier dates | (see HANDOVER for full ladder) | | |

**Top-5% boundary: 0.95405. Gap from PRIMARY: -3.0 bp** (was -5.4 bp).
Leader: 0.95476. Gap from PRIMARY: -10.1 bp (was -12.5 bp).

## Submissions

- **Used: 42 of 270.** Plenty of slots left.
- **Today (2026-05-09): 2 used** (K=5 V4 kNN-aug AM + K=9 PM PRIMARY).
- **Comp-day:** 9 of 31. **Days remaining: 22.**



**Why we promoted from K=27 → K=4 at a deliberate −1.7 bp LB cost:**

1. **K=4 captures 99% of the bank's LB value with 15% of the bases**
   (LB 0.95351 vs prior K=27 PRIMARY 0.95368, Δ −1.7 bp).
2. **The K=27 pool's logit effective rank is 3.23.** SVD shows it
   spans a 3-D subspace; 17 of the 27 bases were dead weight.
3. **Path-B amp is a myth at this pool size.** The Compound × Stint
   shrinkage stacker lifts only +0.04 bp over a plain global LR-meta
   on K=27 — within fold noise. (Day-13 era +18 bp claim was
   pool-specific and never replicated.)
4. **Cleaner reference for any new direction.** A 4-base baseline is
   easier to reason about, faster to retrain, and exposes new bases'
   contributions more cleanly than a saturated 27-base pool.

**Reference of the prior PRIMARY for hedge-eligibility:** the K=27 +
Path-B Compound × Stint τ=100k submission at LB 0.95368 remains
hedge-eligible per Rule R7 (it's the highest-LB prediction we have).
See `submissions/submission_PRIMARY_d18_K27_pathb_tau100k.csv` (rebuild
of the LB 0.95368 artefact).

**Holdout-honest:** all label-derived features are refit per
cross-validation fold using only training rows.

## Leaderboard ladder (from this team's submissions)

| ISO date | Pool | Score | What changed |
|---|---|---:|---|
| 2026-05-09 AM | K=5 (K=4 + V4 kNN-aug LightGBM base) + Path-B C×S τ=100k | **0.95359** | First lift on K=4 PRIMARY in 8 days (+0.8 bp). Mechanism: tree splits ingesting kNN-target-mean (K=20 standardised feature space) at the BASE layer, not as a meta feature. Same feature at meta extracted +0.01 bp; at base extracted +0.8 bp on LB. ρ_test_vs_K4 0.99989 — Rule 27 abort threshold exceeded but PI-authorised override produced a real LB lift. Calibration data: ρ in 0.999-0.9999 zone is NOT auto-tie. |
| 2026-05-08 PM | K=4 forward-greedy + Path-B C×S τ=100k | **0.95351** | Prior PRIMARY. Sparse-pool reduction; 99% of the bank's LB value with 15% of the bases. |
| 2026-05-08 PM | K=10 forward-greedy + Path-B C×S τ=100k | 0.95356 | Sparse-pool calibration probe; precise OOF→LB transfer. |
| 2026-05-07 PM | K=27 + Path-B C×S τ=100k | 0.95368 | Prior PRIMARY. Six DGP-class bases on top of K=21+v4+h1d+d16. |
| 2026-05-07 PM (earlier) | K=23 v4+h1d + Path-B | 0.95354 | First with CatBoost-yekenot + RealMLP-yekenot. |
| 2026-05-07 mid | K=24 LR-meta v3+h1d | 0.95345 | Crossed top-5% threshold. |
| 2026-05-07 AM | K=23 + d16 + d18_chain Path-B τ=20k | 0.95149 | Original-data + chain-decomp bases. |
| Earlier | K=22 + DAE + Path-B Compound×Stint τ=20k | 0.95089 | First clean per-segment stacker on continuous-only features. |
| Earlier | K=22 + DAE Path-B τ=20k | 0.95059 | DAE base added. |
| Earlier | K=21 + Path-B Compound × Stint τ=20k | 0.95049 | Per-segment stacker first fired. |
| 2026-05-01 (kickoff) | Two-anchor baseline | 0.94113 | Stratified + GroupKFold baseline. |

## Submissions

- **Used: 42 of 270.** Plenty of slots left. Per Rule 12, spend them.
- **Today (2026-05-09): 1 used** (K=5 V4 kNN-aug base — new PRIMARY).
- **Comp-day:** 9 of 31. **Days remaining: 22.**

## Distance to top-5%

- Top-5% boundary: 0.95405. Gap from PRIMARY (K=4 LB 0.95351): **−5.4 bp**.
- Gap from K=5 hedge candidate (LB 0.95359): −4.6 bp.
- Leader: 0.95476. Gap from PRIMARY: −12.5 bp.
- Bootstrap CI on a 20% public draw is ±12 bp wide, so both gaps fall
  partly inside the public-LB sample-noise band. Cross-submission
  *relative* deltas trust to ~±1 bp at this scale.

## What axes have been tried — high-level

For the named-experiment-by-experiment ledger, see
`state/mechanism-ledger.md`.

- **Stacking pool growth (A axis): closed.** Multiple confirmations
  that adding orthogonal new bases doesn't move the meta-stacker
  rankings once the stack is saturated. 2026-05-08 PM extended this
  to: rank-lock holds at K=10+1 *and* at K=22+1 *and* at K=24+1 — i.e.
  pool-size-independent. The escape was a feature-recipe transfer
  (yekenot), not a new base.
- **Anchor swap (B axis): closed.** XGBoost+yekenot is redundant with
  CatBoost-yekenot (ρ=0.987). RealMLP n_ens=24 is the only untested
  variant; classical sqrt(n_ens) law gives ≤1 bp from variance
  reduction alone, so the EV is bounded.
- **Meta-architecture redesign (C axis): closed.** Eleven+ variants
  tested across Compound × Stint, Compound × Year, Compound ×
  RaceProgress-bin, Compound × Stint × Year, Yao/Vehtari covariance,
  multi-level. All within ±0.5 bp of plain global LR-meta on the
  current pool. Path-B amp itself is +0.04 bp at K=27 — within fold
  noise.
- **External data (D axis): closed per PI direction.** aadigupta
  original is in the pool; FastF1 hard-join capped at 1.4% match;
  PI directed 2026-05-08 to NOT pursue further external data.
- **Sequence / structural inductive bias (A1):** **closed 2026-05-08
  PM.** Three different-task-framing probes (LambdaRank per-stint,
  inter-stint memory features, stint-completion dual-head) all NULL
  at K=10+1 within ±0.05 bp despite low rank-correlation (ρ
  0.41–0.73). The d16 GRU re-tested at K=10+1 also NULL (Δ −0.045 bp
  matching prior K=22+1 result). **Rank-lock is at the logit-direction
  level, not at rank-correlation.** See `ASSUMPTIONS.md` A29, A30.
- **Non-LR meta architecture (E axis): closed 2026-05-08 PM
  (FALSIFIED).** PCA-meta probe (4 variants, K=27 pool) shows
  LightGBM as meta-learner is *worse* than LR at every input
  representation tested. Best LightGBM 0.95417 ≈ K=10 anchor; best
  LR 0.95428 (K=27+1). EXP-NEW closes negative. Bonus findings:
  A25's 3.23 eff-rank is variance-only (predictive eff-rank ≈ 15);
  Path-B C×S on uncorrelated PCs loses 28-34 bp (Path-B needs base
  correlations to route). See `audit/2026-05-08-pca-meta-probe.md`,
  `scripts/probe_pca_meta.py`, A25b/A30b/A30c.
- **EXP-NEW Phase 1-5b FE/meta campaign (issues leaf 11): closed
  2026-05-08 PM.** 7 of 8 tier-A2/A3 picks NULL in Phase 1 smoke;
  A2-2 (mandatory compound rule) +9.3 bp smoke, +1.4 bp full 5-fold,
  +0.302 bp at K=4+1 plain LR-meta (below +0.5 strict gate);
  TIE_EXPECTED on 4-gate filter. **Path-B amp test (Phase 4b):**
  K=4 + A2-2 + Path-B C×S τ=100k OOF 0.95405 vs K=4 PRIMARY 0.95403
  (Δ +0.26 bp); ρ 0.999893; WEAK. **A2-8 LightGBM stack-meta (Phase
  5b):** 43 meta features incl. pairwise prediction products + abs-
  diffs + logit-diffs + raw side info; 5-fold OOF 0.95390 vs Path-B
  PRIMARY 0.95403 (Δ −1.30 bp); fold-std 0.00080 elevated; FAIL.
  Tree stackers overfit interaction noise on K=4 pool; convex LR +
  Path-B partial-pooling regularize better. **Rule 7 saturation
  research scan:** Frontiers AI 2025 Bi-LSTM peer-effect features
  (DriverAheadPit/Behind) are duplicative of A3-1 RankSortedGaps
  (already null in smoke); only genuinely untried mechanism is a
  Bi-LSTM/GRU sequence base on 10-lap windows (GPU-heavy, ~30-60 min
  Kaggle T4); deferred. See `scripts/probe_a2_2_pathb_K4.py`,
  `scripts/probe_a2_8_stack_meta.py`, `audit/decisions.jsonl` for
  recorded outcomes; ISSUES leaf 11 closed `null`.

## Held submissions (do not submit)

Audit `audit/2026-05-06-target-reform-leakage-audit.md`. All
target-reformulation candidates collapse 88-100% under strict
fold-safe re-runs. The held files based on them must not be submitted:

- Anything named `path_b_K22_invlaps_*`
- Anything named `path_b_K23_dae_invlaps_*`
- Anything named `path_b_K25_megapool_*`
- `path_b_multilevel_τ_*` (a separate null family)

## Held submissions (safe — hedge candidates)

These don't beat PRIMARY on the leaderboard but are safe:

- `d15b_path_b_K22_dae_only_tau{20k,100k}` — the Day-15 PRIMARY and a
  close runner-up.
- `path_b_K22_d12meta_tau100000` — landed at 0.95045 (eligible for the
  R7 final-window hedge).
- `d15c` (Extra Trees), `d15d` (LightGBM-on-kNN-features) — R5 hedge only.
- `path_b_K5_rf_yekenot_tau{5k,20k,100k}` (2026-05-08 evening) —
  K=5 = K=4 + RF-yekenot Path-B refit; τ=100k ρ=0.999917 vs PRIMARY
  (tie-band per Rule 27, PI held); τ=20k/5k asymmetric flips, R7-style
  override risk. R5 hedge candidates.
