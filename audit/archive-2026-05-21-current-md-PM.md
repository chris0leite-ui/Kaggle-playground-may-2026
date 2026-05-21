# Where we are right now

Single source of truth for the current PRIMARY, LB ladder, axes
status, and submission count. **Rewrite this file when PRIMARY
changes** — do not tail-append. Prior versions live in
`audit/archive-YYYY-MM-DD-current-md-*.md`.

**Date convention:** ISO dates ("2026-05-21") or comp-day-N anchored
to comp start 2026-05-01. The `d13`..`d19` labels in script names
and old audit prose are FROZEN code prefixes — never calendar days
(per `glossary.md` and the `day-counter-drift` friction).

## PRIMARY (active) — set 2026-05-21 Round 25

**LB 0.95402** — R25 rank-mean blend: K=18 × 0.89 + R21 (CB YetiRank
standalone OOF on (Year, Race, LapNumber) cohort) × 0.11.

File: `submissions/submission_R25_K18_R21_rankmean_w89.csv`. OOF
**0.954508** (+0.179 bp vs R15 PRIMARY 0.954490; +0.087 bp vs K=18
0.954499). .npy ρ_test vs K=18 = 0.99981 (OK band), vs R15 = 0.99967
(OK band). **Flips vs K=18: 351**; **vs R15: 865** (both ≥200 → R7d
PRIMARY-swap eligible). Submission ref 52882020.

**Top-5% gap: −0.7 bp** (from R15's −0.8 bp). Leader gap: −7.4 bp.

Round-25 finding: cohort-listwise axis was structurally CLOSED at
Path-B base-add layer today (R20 lambdarank K=19 +0.011 bp sub-G2; R21
YetiRank K=19 -0.061 bp regression; R23 cohort-aggregate K=24 -0.054
bp regression; K=20 R20+R21 combo -0.086 bp). **BUT the rank-blend
of K=18 PRIMARY-equivalent + R21 standalone OOF (the orthogonal base
that didn't survive Path-B's L1-meta) extracted +0.179 bp OOF /
+0.05 bp LB**. Mechanism: Path-B LR-meta absorbs cohort-listwise as
one direction; submission-level rank-blend bypasses this by combining
PRE-LR-meta R21 with POST-LR-meta K=18.

R26 stacker swap also tested: XGB meta -6.79 bp / LGB meta -2.78 bp
vs Path-B K=20. Path-B's LR-meta + per-segment shrinkage is BETTER
than non-linear stackers — not the bottleneck.

## Prior PRIMARY R15 (2026-05-19 Round 15) — retained for hedge

**LB 0.95397** — K=17 = R14 K=16 pool + R15_xendcg_per_seg base +
Path-B DCS τ=100k. File:
`submissions/submission_R15_K17_xendcgbase_pathb_dcs_tau100000.csv`.
OOF 0.954490. Hedge: pure Path-B without R21 ingredient.

## Prior PRIMARY R14 (2026-05-19 Round 14) — retained for hedge

**LB 0.95395** — K=16 (K=13 + cb_horizon + cb_stint_completion + TabM) +
Path-B DCS τ=100k. OOF 0.954487. Hedge: "remove xendcg-base" ablation.

## Prior PRIMARY R13 (2026-05-19 Round 13) — retained for hedge

**LB 0.95393** — K=15 (K=13 + cb_horizon + cb_stint_completion) +
Path-B DCS τ=100k. OOF 0.954485.

## Prior PRIMARY R12-2 (2026-05-19 Round 12-2) — retained for hedge

**LB 0.95392** — K=14 (K=13 + cb_horizon) + Path-B DCS τ=100k. OOF
0.954475.

## Prior PRIMARY R7.1 (2026-05-18 Round 7) — retained for hedge

**LB 0.95389** — K=13 + Path-B DriverClass × Stint τ=100k. OOF
0.954471.

## Today's R-series catalog (2026-05-21)

| Round | Mechanism | OOF Δ vs R15 | Verdict |
|-------|-----------|--------------|---------|
| **R25** | **K=18 + R21 rank-mean 0.89/0.11** | **+0.179 bp** | **LB 0.95402 PRIMARY** |
| R20 | LGB lambdarank meta on (Y,R,L) cohort, K=19 add | +0.103 bp OOF / +0.011 vs K=18 | sub-G2, TIE_ZONE vs K=18 |
| R21 | CB YetiRank standalone on (Y,R,L) cohort, K=19 add | +0.031 bp / -0.061 vs K=18 | K-add NULL; standalone became R25 ingredient |
| R23 | 5 cohort-aggregate features (mean/max/std/rank/centered), K=24 | -0.054 vs K=18 | NULL — LR-meta confused |
| K=20 | R17 + R20 + R21 combo | -0.086 vs K=18 | combo NULL |
| R26 | XGB / LGB meta swap on K=20 | -6.79 / -2.78 vs Path-B | stacker-swap axis CLOSED |

## Submissions

- 50 of 270 total; **1 used 2026-05-21** (R25); **9 daily slots
  available** at writing.
- Comp-day **21 of 31**; days remaining **10**. Final-3-day lock
  window opens **2026-05-29** (8 days out).

## Active axes

| Axis | Status | Last probe |
|------|--------|------------|
| **Submission-level rank-blend** | **OPEN, lifted** | R25 K=18+R21 w=0.89 → LB 0.95402 |
| Path-B K-add (cohort-listwise) | CLOSED | 4 nulls today: R20/R21/R23/K=20 combo |
| Non-linear stacker swap | CLOSED | R26 XGB -6.79 / LGB -2.78 bp |
| Cross-family standalone bases | OPEN | R21 CB YetiRank produced strong standalone (OOF 0.95396, ρ 0.978 vs R15) |
| Wide-pool blending (K=27) | OPEN | K27 ρ 0.998 vs R15; not yet blended w/ K=18 |
| Inventor track NN family | 1 lift (TabM) / 2 nulls (R18 multitask, R19 aleatoric) | R19 K=20 LR-meta -0.084 bp |

## Next-experiment shortlist

1. **More 2-/3-way rank-blends** with K=18 anchor + diverse standalone
   ingredients (R21, K=27 wide-pool, K=19_R20, R7.2 fold-bag). Cheap;
   each LB-submittable as a hedge or PRIMARY-swap candidate.
2. **More cross-family standalone bases** via Kaggle GPU (XGBoost
   rank:ndcg, CatBoost QueryCrossEntropy) — each new standalone is a
   fresh blend ingredient. Same axis as R21 worked.
3. **Different blend operators** on the same K=18 + R21 pair: arith,
   gmean, logit_mean (R25 used rank_mean).
