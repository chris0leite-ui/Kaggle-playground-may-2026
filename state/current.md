# Where we are right now

Single source of truth for the current PRIMARY, LB ladder, axes
status, and submission count. **Rewrite this file when PRIMARY
changes** — do not tail-append. Prior versions live in
`audit/archive-YYYY-MM-DD-current-md-*.md`.

**Date convention:** ISO dates ("2026-05-14") or comp-day-N anchored
to comp start 2026-05-01. The `d13`..`d19` labels in script names
and old audit prose are FROZEN code prefixes — never calendar days
(per `glossary.md` and the `day-counter-drift` friction).

## PRIMARY (active) — set 2026-05-18 Round 7

**LB 0.95389** — R7.1 K=13 + Path-B DriverClass × Stint τ=100k.

K=13 pool = K=11 (4 K=4 trees + 6 slim-kNN + K=27 super-base) +
r4_segment_fe + r4_hmm_seq. Path-B applied with **DriverClass ×
Stint** segmentation (named-vs-D0XX driver classification × 6
stint values = 12 segments, 10 above MIN_ROWS=1000).

File: `submissions/submission_K13_pathb_driverclass_stint_tau100000.csv`.
OOF 0.95447 (+0.106 bp over default Compound × Stint baseline);
OOF→LB transfer -5.8 bp (similar to all K=13+Path-B variants).

The Round-7 finding: the named-driver pit-rate differential (32-43%
vs 16-22% for anonymous D0XX) is captured by driver-class
segmentation; default Compound × Stint missed it.

## Prior PRIMARY R5.2 (2026-05-18 Round 5) — retained for hedge

**LB 0.95387** — K=13 + Path-B Compound × Stint τ=100k.
File: `submissions/submission_K13_seghmm_pathb_tau100000.csv`.

## Hedge candidate R7.2 (2026-05-18 Round 7)

**LB 0.95389** (ties R7.1) — R7.1 + 5-seed fold-fit bag.
File: `submissions/submission_K13_dcs_pathb_foldbag.csv`. OOF 0.95450
(+0.264 bp over R7.1 single-seed; largest OOF lift of session).
Structurally distinct hedge for private-LB variance.

## Pre-Round-5 PRIMARY (2026-05-12) — retained for hedge ladder

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

## Today's status (2026-05-19)

- Submissions used this comp: **47 / 270**. Daily cap: 10.
- Today (2026-05-19): **1 used**, 9 unused at session-end.
  - **R10 HEDGE 3: 75/25 arith blend (R7.2 5-seed bag + K=27 wide-pool) → LB 0.95387** (-0.02 bp vs PRIMARY 0.95389). True .npy ρ=0.999882 (OK band) registered as -0.02 bp LB delta. First cross-mechanism diversity hedge confirmed; HEDGE-3 slot in final-window ladder filled.
  - R10 blend-operator sweep over {R7.1, R7.2, R5.2, R6.1, K27} × {arith, gmean, logit_mean, rank_mean}: only OK-band (ρ <0.9999) candidate was the R7.2+K27 75/25 arith blend submitted above. All others TIE_ZONE.
  - R10 multi-constituent LR-meta alt-stack (LambdaRank stint + race + rolling LGBM + kernel hazard, 4 constituents): blended w/ R7.1 returned Δ < 0 at every weight (best Δ=-0.045 bp). **Closes alt-stack as 4th rank-lock axis.** See `audit/2026-05-19-round-10-hedge-prep.md`.
- Kaggle CLI auth fix (session-start blocker): KGAT_-prefixed access tokens must be exported as `KAGGLE_API_TOKEN`, NOT placed in `kaggle.json`'s `key` field (legacy HTTP Basic). Working invocation: `KAGGLE_API_TOKEN="$KaggleAPIToke" kaggle ...`.

## Today's status (2026-05-18, prior session)

- Submissions used this comp: **46 / 270**. Daily cap: 10.
- Today (2026-05-18): **7 used**, 3 unused at session-end:
  - R4: K=4 + seg + HMM LR-meta → LB **0.95354**
  - R5.1: K=11 + seg + HMM LR-meta → LB **0.95382**
  - R5.2: K=13 + Path-B Compound×Stint τ=100k → LB **0.95387**
  - R5.3: 70/30 rank-blend R5.2 + K=27+Path-B → LB **0.95385**
  - R6.1: R5.2 + 5-seed fold-fit bag → LB **0.95387** (ties R5.2)
  - **R7.1: K=13+Path-B DriverClass×Stint τ=100k → LB 0.95389** ← PRIMARY
  - R7.2: R7.1 + 5-seed fold-fit bag → LB **0.95389** (ties R7.1; hedge)
  - R8 multi-seg sweep: 4 segs / none > +0.10 bp (`audit/2026-05-18-round-8-multiseg.json`).
  - R8 60/20/20 multi-seg blend: OOF +0.079 bp, ρ TIE_ZONE, NOT submitted.
  - **R9 NB4** (Compound×Stint TE-as-base): K=14 Δ vs R7.1 PRIMARY **−0.022 bp NULL**; standalone 0.94850 G1✓.
  - **R9 C1** (Aadigupta per-Race scalars): K=14 Δ vs R7.1 PRIMARY **−0.045 bp NULL**; standalone 0.94902 G1✓.
- Comp-day **18 of 31**. Days remaining: **13**.
- Top-5% boundary: **0.95405**. Gap to PRIMARY: **−1.8 bp**.
- Leader: **0.95476**. Gap to PRIMARY: **−8.9 bp**.

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
| 2026-05-19 PM R10 | 75/25 arith R7.2 + K=27 (HEDGE 3 OK-band) | 0.95387 | +3.6 |
| **2026-05-12 AM** | **Blend 70/30 K=11+K=9** | **0.95386** | **+3.5** |
| 2026-05-09 PM | K=11 + K=27 + Path-B τ=100k | 0.95385 | +3.4 |
| 2026-05-09 PM | K=10 + K=27 + Path-B τ=100k | 0.95384 | +3.3 |
| 2026-05-09 PM | K=9 qAX (slim-kNN) | 0.95375 | +2.4 |
| **2026-05-18 PM R7** | **K=13+Path-B DriverClass×Stint τ=100k (R7.1)** | **0.95389** | **+3.8** |
| 2026-05-18 PM R7 | K=13+Path-B DC×S 5-seed fold-bag (R7.2) | 0.95389 | +3.8 (tied; hedge) |
| 2026-05-18 PM R5 | K=13 (K=11+seg+HMM)+Path-B Compound×Stint τ=100k (R5.2) | 0.95387 | +3.6 |
| 2026-05-18 PM R6 | R5.2 + 5-seed fold-fit bag (R6.1) | 0.95387 | +3.6 (tied) |
| 2026-05-09 AM | K=5 (K=4 + V4 kNN-aug) Path-B τ=100k | 0.95359 | +0.8 |
| 2026-05-18 PM | K=11 + seg + HMM LR-meta | 0.95382 | +3.1 |
| 2026-05-18 PM | 70/30 R5.2 + K=27+Path-B rank-blend | 0.95385 | +3.4 |
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

1. **MECHANISM EXPANSION beyond row-feature ceiling** (R9 forced
   pivot 2026-05-18). R9 closed BOTH residual row-feature axes:
   NB4 TE-as-base (−0.022 bp NULL) and C1 external Aadigupta scalars
   (−0.045 bp NULL). Rank-lock at K=13+Path-B is structurally
   confirmed across operator / mechanism / data-class. Three
   structurally orthogonal candidates remain (none tested):
   1. **A1 seq2seq transformer** on per-(Driver, Race) lap
      sequences (HMM K=13 base was BW one-shot on 4 states; full
      attention-LSTM untried). ~2 hr Kaggle T4.
   2. **Graph mechanism** (Race, Lap) per-row graph with competitor
      edges; LightGCN / GAT 2-layer. ~3 hr local CPU or Kaggle T4.
   3. **Survival** Cox PH on stint-life as hazard-base. ~30 min CPU.
2. **Blend operator sweep** — arithmetic / geometric / log-odds /
   trimmed mean over 4 LB-confirmed PRIMARYs. Cost 5 min.
3. **R7d hedge ladder finalisation** (final-window posture, days
   28-31 = May 28-31). Ladder candidate set documented above.
4. **NEW-INFO mechanism via cross-domain pretrain** (still on the
   2026-05-14 list, but FastF1 hard-join capped at 1.4% match;
   C1 falsified the simpler per-Race scalar variant of this axis).
