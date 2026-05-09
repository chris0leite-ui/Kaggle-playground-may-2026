# Where we are right now

Updated whenever a new submission lands or the strategic picture shifts.
For history, read `audit/postmortems/` and `audit/research/`.

**Date convention:** prose in this file uses **ISO dates** (e.g.
"2026-05-08") or **comp-day-N** anchored to comp start 2026-05-01
(so today = 2026-05-08 = comp day 8). The `d13`..`d19` labels in
script names and audit prose are *frozen code/file prefixes*, not
calendar days — see `glossary.md` and `audit/friction.md`
under `day-counter-drift`.

## PRIMARY (active) — set 2026-05-09 PM (autonomous loop BtmFl + K=27 ensemble)

**Score: 0.95385 on the public leaderboard.** Direct LB-confirmed.
**+3.4 bp lift over original K=4 PRIMARY** (LB 0.95351). +1.0 bp over
the prior K=9 PRIMARY (LB 0.95375).

**What it is:** K=11 = K=4 + qAT + qAV + qAO + qAA + qAF + qAK +
K27_100k, with **Path-B Compound × Stint τ=100,000**.

The K=27 base is the prior PRIMARY-of-old
(`d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000`) treated as a single
super-base — collapses 27 bases (incl. d18 chain decomp, F2 constraint,
E2 preimage-kNN) into one prediction. Adding K=27_100k as a single
base lifts +2.624 bp at K=4+1 plain LR-meta — proving prior K=27
information was being lost in K=4-only PRIMARY.

The 6 new orig-kNN slim-bases (qAK/qAO/qAA/qAF/qAT/qAV) ride on top
of K=27 to give per-row attribution at the host's decoded 6-axis
cell key.

**Submitted:** 2026-05-09 13:40 UTC. LB 0.95385, public COMPLETE.
File: `submissions/submission_qBF_K11_qAT_qAV_qAO_qAA_qAF_qAK_K27_100k_pathb_tau100000.csv`

**OOF→LB transfer for this candidate**: OOF +4.032 bp → LB +3.4 bp,
ratio 0.84× (lower than qAX's 1.19× because K=27 brings more
redundancy with K=4 already-used bases).

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
