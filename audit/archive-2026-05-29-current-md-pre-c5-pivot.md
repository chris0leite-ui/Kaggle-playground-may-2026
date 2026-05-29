# Where we are right now

Single source of truth for the current PRIMARY, LB ladder, axes
status, and submission count. **Rewrite this file when PRIMARY
changes** — do not tail-append. Prior versions live in
`audit/archive-YYYY-MM-DD-current-md-*.md`.

**Date convention:** ISO dates ("2026-05-21") or comp-day-N anchored
to comp start 2026-05-01. The `d13`..`d19` labels in script names
and old audit prose are FROZEN code prefixes — never calendar days.

## PRIMARY (active) — set 2026-05-21 Round 25

**LB 0.95402** — R25 rank-mean blend: K=18 × 0.89 + R21 × 0.11.

File: `submissions/submission_R25_K18_R21_rankmean_w89.csv`. OOF
0.954508. ρ_test vs K=18 = 0.99981 (OK band); vs R15 = 0.99967 (OK band).

**Realistic LB position: rank 297 / 2063 — top-14.4%.** Top-5% boundary
0.95449 (+4.7 bp gap); top-10% boundary 0.9420 (+1.8 bp gap). Strategy-
critic Section 5 (2026-05-21 PM): own-bases discounted lift ~0.30 bp —
**top-10% unreachable without R22 public-notebook scout or external help**.

## Hedge ladder candidates (LB-confirmed)

| Rank | File | LB | Mechanism |
|------|------|-----|-----------|
| PRIMARY | `submission_R25_K18_R21_rankmean_w89.csv` | 0.95402 | K=18 × 0.89 + R21 × 0.11 rank-mean |
| HEDGE 0 | `submission_R31_K18_R30_R72_rankmean_50_15_35.csv` | 0.95402 | 3-way K=18 + R30 + R7.2 rank-mean (different structure) |
| HEDGE 1 | `submission_R27_4way_K18_R21_K27_R72_70_12_10_08.csv` | 0.95402 | 4-way K=18 + R21 + K=27 + R7.2 rank-mean |
| HEDGE 2 | `submission_K18_pathb_driverclass_stint_tau100000.csv` | 0.95398 | Pure K=18 Path-B (no rank-blend) |
| HEDGE 3 | `submission_R15_K17_xendcgbase_pathb_dcs_tau100000.csv` | 0.95397 | K=17 + R17 listwise xendcg + Path-B |

## Today's R-series catalog (2026-05-21 PM)

| Round | Mechanism | OOF | LB | Verdict |
|-------|-----------|-----|----|---------|
| R30 | 5-seed bag of R21 YetiRank cohort | 0.954027 | n/a | LIFT INGREDIENT (+0.07 bp vs R21) |
| R31 (cohort) | (Driver,Y,R) CB YetiRank K=17 | 0.951779 | n/a | WEAK NULL (cohort too fine) |
| R32 (cohort) | (Y,R,Stint) CB YetiRank K=17 | ERROR | n/a | CB GPU 1023 max-query limit blocked |
| R31 (submit) | 3-way K=18+R30+R7.2 rank-mean 0.50/0.15/0.35 | 0.954529 | **0.95402** | LB TIE — rank-blend ceiling confirmed |
| R33 | Per-cohort isotonic of R25 (R33 inner-CV) | -13 to -44 bp | n/a | NULL across 3 cohort defs |
| R34 | Pseudo-label CB YetiRank K=17 (Y,R,L) | 0.953709 | n/a | NULL (cohort contamination) |
| R35 | FT-Transformer (3L×96D, num+cat+K17 tokens) | **0.954086** | n/a | STRONGEST NN ever; ρ vs K18=0.9832; blend regression |
| R37 | CB PairLogit cohort K=17 (Y,R,L) | 0.951919 | n/a | NULL (ρ vs R21=0.9499 most-orthogonal cohort but standalone gap) |
| R36 | 10-seed bag of R21 (variance reduction) | 0.954039 | n/a | +0.012 bp vs R30; indistinguishable in blends |

## Submissions

- 53 of 270 total; **3 used 2026-05-21** (R25, R27, R31); **7 daily slots
  available** at session-end.
- Comp-day **21 of 31**; days remaining **10**. Final-3-day lock window
  opens **2026-05-29** (8 days out).

## Active axes

| Axis | Status | Last probe |
|------|--------|------------|
| Submission-level rank-blend | **SATURATED at LB 0.95402** | R31 3-way TIE; R30/R35/R37/R34 all blend-NULL |
| Path-B K-add (cohort-listwise) | CLOSED | R20/R21/R23/K=20 combo all sub-G2 (Day 21) |
| Cohort-listwise (Y,R,L) sweet spot | UNIQUE — alternate cohorts/losses all sub-R21 | R31 (Driver,Y,R), R37 (PairLogit), R32 (Y,R,Stint blocked) |
| Per-cohort isotonic | CLOSED | -13 to -44 bp on all 3 cohort defs (Day 22) |
| Pseudo-label cohort YetiRank | CLOSED | R34 -0.025 bp vs R21 standalone |
| NN-class FT-Transformer | OPEN as orthogonality | R35 OOF 0.954086, ρ_K18=0.9832 — strongest NN ever |
| Non-linear stacker swap | CLOSED | R26 XGB -6.79 / LGB -2.78 bp |
| Final-window hedge ladder | OPEN — build by 2026-05-29 | 4 LB-confirmed candidates ready |
| R22 public-notebook IDEA-scan | DEFERRED | PI authorization pending |

## Next-experiment priorities

1. **R36 result + final blend search** (in-progress).
2. **Hedge-ladder validation** (R5d / R7d): private-LB regression-risk
   probe at Day 28 (3 days before close). Verify PRIMARY+HEDGE pair
   doesn't both regress on the same private-LB perturbation.
3. **R22 public-notebook IDEA-scan** if PI authorizes — extract structural
   ideas from `safar1/lb-score-0-95449`, `cdeotte EDA`, `leonchani 02_NN`
   (READ ONLY — no CSV blending per PI directive).
4. **Strategic accept**: top-5% (+4.7 bp) and top-10% (+1.8 bp)
   unreachable on own bases. Aim for stable top-15% (current rank ~297).
