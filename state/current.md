# Where we are right now

Single source of truth for the current PRIMARY, LB ladder, axes
status, and submission count. **Rewrite this file when PRIMARY
changes** — do not tail-append. Prior versions live in
`audit/archive-YYYY-MM-DD-current-md-*.md`.

**Date convention:** ISO dates ("2026-05-29") or comp-day-N anchored
to comp start 2026-05-01. The `d13`..`d22` labels in script names
and old audit prose are FROZEN code prefixes — never calendar days.

## PRIMARY (active) — set 2026-05-29 Day-29 (C5)

**LB 0.95404** — C5 4-way rank-mean: K=20 PathB (DriverClass×Stint,
τ=100k) × 0.82 + R21 YetiRank × 0.08 + K=27 PathB × 0.10.

File: `submissions/submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv`.
OOF 0.954534. ρ_test vs C1 (LB 0.95403) = 0.999930.
ρ_test vs R25 (LB 0.95402) = 0.999897.

**+2 bp cumulative LB gain Day-29 vs R25 0.95402.** Probe.py "TIE_EXPECTED"
warning broken twice today by membership-inference axis (d21, d22).
Realized OOF→LB amp factor on d22 path: 10× (OOF +0.10 → LB +1 bp from
C1→C5).

**K=20 pool composition (20 bases):** K=18 set (yekenot, cb_v4, hgbc_deep,
d16_orig, qAT, qAV, qAO, qAA, qAF, qAK, K27_100k, seg_fe, HMM, R12, R13,
R14_tabm, R15_xendcg, R17_listwise) + d21_membership_inference (k=5 NN,
ρ_test_vs_K18 = 0.7453) + d22_membership_exact_k1 (k=1 NN exact-copy,
ρ_test_vs_K18 = 0.7014 — lowest ever).

## Hedge ladder candidates (LB-confirmed) — Day-29 PM

| Rank | File | LB | Mechanism |
|------|------|-----|-----------|
| **PRIMARY** | `submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv` | **0.95404** | K=20 × 0.82 + R21 × 0.08 + K=27 × 0.10 (d21+d22 mem-inf in K=20) |
| HEDGE 0 | `submission_d21_d22_C7_K20_R21_K27_R28_renorm.csv` | 0.95404 | C5 + R28 xgb-ndcg @ 0.05 (TIE — adds nothing but doesn't hurt) |
| HEDGE 1 | `submission_d21_C1_K19_R21_rankmean_89_11.csv` | 0.95403 | K=19 PathB × 0.89 + R21 × 0.11 (K=19 = K=18+d21) |
| HEDGE 2 | `submission_R25_K18_R21_rankmean_w89.csv` | 0.95402 | K=18 × 0.89 + R21 × 0.11 (prior PRIMARY) |
| HEDGE 3 | `submission_R31_K18_R30_R72_rankmean_50_15_35.csv` | 0.95402 | 3-way K=18 + R30 + R7.2 (structurally diverse) |

**Flip counts C5 vs R25:** 577 raw (sign at 0.5), 274 top-5% rank flips.
R7d threshold >200 — would need explicit PI sign-off for HEDGE pick,
but C5 IS the PRIMARY so this is informational only.

## Today's submissions (2026-05-29 Day-29 of 31)

| Slot | File | OOF | LB | Notes |
|------|------|-----|----|------|
| 1 | C1 K=19 PathB × R21 89/11 | 0.954524 | **0.95403** | +1 bp vs R25 — d21 axis activated |
| 2 | C2 K=19 plain PathB | 0.954515 | 0.95399 | Plain Path-B regressed at LB |
| 3 | C3 3-way K=18 + K=5d21 + R21 | 0.954518 | 0.95402 | Tied R25 — secondary blend null |
| 4 | C5 K=20 + R21 + K=27 82/8/10 | 0.954534 | **0.95404** | +1 bp vs C1 — d22 axis activated |
| 5 | C7 C5 + R28 xgb-ndcg @ 0.05 | 0.954537 | 0.95404 | TIE — xgb-ndcg adds nothing |
| 6 | C8 C5 + R20 listwise-lambdarank @ 0.05 | 0.954539 | 0.95402 | REGRESSED 2 bp — OOF-grinding overfit |

**6 of 7 daily slots used. 1 slot remains** (preserve for Day-30 anchor
verification or hedge probe).

## Active axes — Day-29 PM

| Axis | Status | Last probe |
|------|--------|------------|
| Membership-inference (d21+d22) | **ACTIVE — +2 bp Day-29** | C5 confirmed; new lowest-ever ρ_vs_K18 (d22 0.7014) |
| K=20 Path-B DriverClass×Stint | ACTIVE — produces PRIMARY | K=20 OOF 0.954519, Path-B amp ~10x on d22 axis |
| Blend-layer reweighting (K=27 inside K=20 + K=27 again at blend) | CONFIRMED — +0.7 bp marginal | C5 vs K=20 PathB delta in blend layer |
| OOF-grinding additions (R28, R20, R36, R30) | CLOSED — no LB lift | C7 TIE, C8 -2bp, R30/R36 in K=27 sweep TIE |
| Submission-level rank-blend | OPEN — extends to 4-way | C5 4-way structurally diverse helps |
| Cohort-listwise alternate cohorts | CLOSED (Day-23) | R31/R37 sub-G2 |
| Per-cohort isotonic | CLOSED (Day-22) | -13 to -44 bp |
| NN-class FT-Transformer (R35) | OPEN-as-orthogonality | OOF 0.954086, ρ_K18=0.9832 |
| Final-window hedge ladder | LOCKED — 5 LB-confirmed candidates | C5 PRIMARY, C7 HEDGE_0 |
| R22 public-notebook IDEA-scan | DEFERRED | PI authorization pending |

## Day-30 (tomorrow) priorities

1. **Hold C5 as PRIMARY.** Don't ladder-grind further (C7 TIE, C8 -2 bp
   confirms OOF-grinding fails past C5).
2. **Final-3-day lock window opens 2026-05-29 EOD.** Day-30 + Day-31 are
   verification + hedge selection.
3. **Verify C5 vs alternate seed/fold of K=20** (anti-overfit probe) — 1
   slot.
4. **Hedge selection:** C5 PRIMARY, C7 HEDGE (LB-tied 0.95404 same family),
   C1 HEDGE_2 (different structure, ρ 0.999930). R2d: best public + best
   OOF regressed ≤30 bp.
5. **Avoid:** more OOF-grinding 4-way/5-way (C7/C8 falsified this
   direction); more weight sweeps in (K=20, R21, K=27) basin (local-best).

## Strategic posture

**Top-15% achieved.** R25 LB 0.95402 was top-14.4% (rank ~297/2063);
C5 LB 0.95404 likely improves rank by ~30-50 places (top-12-13%
estimated; verify via LB download Day-30 AM).

Top-10% boundary 0.9420 — still ~+1.6 bp away. Top-5% boundary
0.95449 — +4.5 bp away. **C5 ladder closed.** Further lift requires
R22 public-notebook scout (PI authorization gated) or a new mechanism
family not yet tried (e.g., structured graph model on Race × Driver
× Compound co-occurrence).

## Key learnings Day-29

1. **Membership-inference axis is real and large.** d21 (k=5 NN) and
   d22 (k=1 NN exact-copy) each delivered +1 bp at LB despite probe.py
   "TIE_EXPECTED" warning at ρ=0.9999. The K=18 pool did NOT route
   this signal — ρ_d22_vs_K18=0.7014 is the lowest ever recorded.
2. **Probe.py ρ-band is empirical-data-conditioned.** Same ρ band
   (0.99993) gave +10× amp (C5) and 0× amp (C7) within hours. The
   amp factor depends on what NEW signal the candidate brings, not
   on ρ alone.
3. **OOF-grinding past a structural lift fails fast.** C5 (structural
   d22 axis) lifted +1 bp; C7 (OOF +0.03 via R28) tied; C8 (OOF +0.05
   via R20) regressed -2 bp. Higher OOF ≠ higher LB once the
   structural axis is consumed.
4. **KGAT_ token auth requires `KAGGLE_API_TOKEN` env (Bearer), not
   `KAGGLE_KEY` (Basic).** Documented in
   `.claude/skills/kaggle-comp/improvements.md` 2026-05-19 entry —
   re-encountered Day-29 13:25 UTC, fixed in 3 min via grep of
   improvements.md.
