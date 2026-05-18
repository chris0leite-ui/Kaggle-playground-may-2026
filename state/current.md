# Where we are right now

Single source of truth for the current PRIMARY, LB ladder, axes
status, and submission count. **Rewrite this file when PRIMARY
changes** — do not tail-append. Prior versions live in
`audit/archive-YYYY-MM-DD-current-md-*.md`.

**Date convention:** ISO dates ("2026-05-14") or comp-day-N anchored
to comp start 2026-05-01. The `d13`..`d19` labels in script names
and old audit prose are FROZEN code prefixes — never calendar days
(per `glossary.md` and the `day-counter-drift` friction).

## PRIMARY (active) — set 2026-05-12, reconfirmed 2026-05-14

**LB 0.95386** (rank-blend 70/30 of two LB-confirmed submissions).

- **Constituent A (70%)**: K=11 + K=27 + Path-B τ=100k. LB 0.95385.
- **Constituent B (30%)**: K=9 qAX (qAT+qAV+qAO+qAA+qAF + Path-B
  τ=20k). LB 0.95375.

File: `submissions/submission_blend_K11_K9_w_70_30.csv`.

The cross-mechanism blend cancels small errors: K=11 uses K=27
super-base + slim-kNN; K=9 uses slim-kNN only. PI-authorised
override of Rule 27 abort threshold (ρ_test ≥ 0.999) yielded the
+0.1 bp lift despite ρ = 0.9998 — uncorrelated-error cancellation
at the 5-decimal Kaggle quantisation.

## Today's status (2026-05-18)

- Submissions used this comp: **43 / 270**. Daily cap: 10
  (`comp-context.md: submission_budget`).
- Today (2026-05-18): **1 used**:
  - K=4 + r4_segment_fe + r4_hmm_seq LR-meta (R4 plateau-break
    probe) — LB **0.95354** (OOF 0.95405, transfer −5.1 bp).
    Beats K=4+Path-B by 0.3 bp; 3.2 bp behind PRIMARY.
- Comp-day **18 of 31**. Days remaining: **13**.
- Top-5% boundary: **0.95405**. Gap to PRIMARY: **−1.9 bp**.
- Leader: **0.95476**. Gap to PRIMARY: **−9.0 bp**.

## Transfer bands (Rule 27 recalibration, 2026-05-14)

Encoded in `scripts/probe_blend_harness.py::RULE_27_*_THRESHOLD`:

| Band | ρ_test vs PRIMARY | Expectation |
|---|---|---|
| TIE_ZONE | ≥ 0.9999 | LB ties within ±0.05 bp |
| OK transfer | 0.999 ≤ ρ < 0.9999 | Sub-bp to few-bp LB movement |
| REGRESSION_RISK | < 0.999 | Wide-ρ adds overfit CV patterns |

## Strategic conclusion (2026-05-14)

K=11 + LR-meta + Path-B is **at or near the Bayes-optimal ceiling**
for row-feature prediction on this synthetic dataset. The synth
generator decouples PitStop and PitNextLap with a stochastic
component (~20% label disagreement on observable pairs), revealing
a noise floor that no row-level model can recover (audit:
`audit/2026-05-14-overnight-iteration.md`).

Pivots that remain untried:
1. NEW-INFORMATION mechanisms (cross-domain training, FastF1
   hard-join, multi-seed bagging of the full pipeline).
2. Bayesian group-prior with smoothing for sequence-conditional
   error cancellation.
3. Different blend operators (geometric mean, log-odds mean) over
   the 4 LB-confirmed PRIMARYs (cost 5 min total).

## Held submissions — DO NOT SUBMIT

Day-17 strict fold-safe audit collapsed all target-reformulation
single-add results 88-100% (Rule 24 origin). The CSVs are still on
disk but must not be submitted:

- `path_b_K22_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K25_megapool_tau{5k,20k,100k}.csv`
- `path_b_multilevel_τ_*.csv` (5 configs, all null anyway)

Origin: `audit/2026-05-06-target-reform-leakage-audit.md`.

## Hedge ladder (R5 / R7 final-window candidates)

These don't beat PRIMARY but are eligible for the final-3-day
hedge probe:

- K=4 + Path-B C×S τ=100k (LB 0.95351) — clean reference base.
- K=9 qAX + Path-B τ=20k (LB 0.95375) — slim-kNN solo.
- K=27 + Path-B τ=100k (LB 0.95368) — pre-sparse-pool PRIMARY.
- `d15b_path_b_K22_dae_only_tau{20k,100k}` — Day-15 PRIMARY runner-up.
- `path_b_K5_rf_yekenot_tau{5k,20k,100k}` — Forest-base τ-sweep;
  τ=100k is balanced flips, τ=5k/20k asymmetric (R7 override risk).

## Leaderboard ladder (this team)

| ISO date | Mechanism | LB | Δ vs K=4 baseline |
|---|---|---:|---:|
| **2026-05-12 AM** | **Blend 70/30 K=11+K=9** | **0.95386** | **+3.5** |
| 2026-05-09 PM | K=11 + K=27 + Path-B τ=100k | 0.95385 | +3.4 |
| 2026-05-09 PM | K=10 + K=27 + Path-B τ=100k | 0.95384 | +3.3 |
| 2026-05-09 PM | K=9 qAX (slim-kNN) | 0.95375 | +2.4 |
| 2026-05-09 AM | K=5 (K=4 + V4 kNN-aug) Path-B τ=100k | 0.95359 | +0.8 |
| 2026-05-18 AM | K=4 + r4_segment_fe + r4_hmm_seq LR-meta | 0.95354 | +0.3 |
| 2026-05-08 PM | K=4 + Path-B τ=100k | 0.95351 | 0 |
| 2026-05-07 PM | K=27 + Path-B τ=100k | 0.95368 | (pre-sparse) |
| 2026-05-07 AM | K=23 + d16 + d18_chain Path-B τ=20k | 0.95149 | |
| 2026-05-01 | Two-anchor baseline | 0.94113 | |

## Closed axes (high-level — see mechanism-ledger.md for detail)

- **Stacking pool growth** (A): closed. Rank-lock pool-size-independent.
- **Anchor swap** (B): closed. RealMLP n_ens=24 the only untested
  variant; sqrt(n_ens) bounds ≤1 bp EV.
- **Meta-architecture redesign** (C): closed. 11+ Path-B variants,
  all within ±0.5 bp of plain LR-meta on the current pool.
- **External data** (D): closed per PI direction.
- **Sequence / structural inductive bias** (A1): closed. LambdaRank
  per-stint, inter-stint memory, stint-completion dual-head all NULL
  at K=10+1 despite ρ 0.41–0.73.
- **Non-LR meta architecture** (E): closed (FALSIFIED). LightGBM /
  MLP / RF metas all regress vs LR-meta. Predictive eff-rank ≈ 15,
  variance eff-rank = 3.23 (A25/A30b/A30c).
- **Forest family**: characterised end-to-end. +0.25 bp K=4+1 ceiling
  set by meta architecture, not RF; Path-B absorbs to +0.02 bp.

## What axes remain open

1. **NEW-INFORMATION mechanisms** (next-session pivot per 2026-05-14
   audit) — anything that injects signal not derivable from existing
   row features: cross-domain pretrain, FastF1 hard-join (capped at
   1.4% match), multi-seed bagging of the full pipeline.
2. **Blend operator sweep** — arithmetic / geometric / log-odds /
   trimmed mean over 4 LB-confirmed PRIMARYs. Cost 5 min.
3. **Path-B C×S on a fresh 27-base + slim-kNN union** — qBG attempt
   timed out at 16+ min. Rerun with τ=100k only.
4. **R5 hedge preparation** — 30 min; ladder above is the candidate set.
