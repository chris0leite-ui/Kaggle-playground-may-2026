# Where we are right now

Single source of truth for the current PRIMARY, LB ladder, axes
status, and submission count. **Rewrite this file when PRIMARY
changes** — do not tail-append. Prior versions live in
`audit/archive-YYYY-MM-DD-current-md-*.md`.

**Date convention:** ISO dates ("2026-05-29") or comp-day-N anchored
to comp start 2026-05-01. The `d13`..`d22` labels in script names
and old audit prose are FROZEN code prefixes — never calendar days.

## PRIMARY (active) — set 2026-05-29 Day-29 PM (R22-V4)

**LB 0.95446** — R22-V4 public-blend rank-mean:
0.70 raunakdey07_95454 (public LB 0.95454)
+ 0.15 arunklenin_solo (most-diverse public ρ_anchor=0.974)
+ 0.15 PRIMARY_C5 (original-work hedge, LB 0.95404).

File: `submissions/submission_d29_R22V4_raunakdey70_arunsolo15_C5_15.csv`.
ρ_test vs C5 (prev PRIMARY) = 0.99476 (R27 OK band).
100% rows differ; mean rank shift 3764 positions.

**+42 bp LB lift in 1 submission** — PI authorized 2026-05-29 R22
CSV-blend reversal of Day-8/Day-22 'original work only' directive.
Predicted band was +3 to +5 bp → realized +42 bp (8.4× upper-band
break). Mechanism: public-cluster signal (raunakdey lineage) sat at
+50 bp above K=18/K=20 stack ceiling; our PRIMARY had never tapped
that signal.

**R22 distinct-lineage finding:** 38 scraped public submissions reduce
to 16 distinct lineages. Of those 16, only `arunklenin_solo` (ρ ~0.974
vs anchor) is materially orthogonal to the raunakdey-blender cluster.
All other 15 sources are mutually ρ ≥ 0.997 — same upstream signal
re-blended. Our PRIMARY C5 sits at ρ ~0.992 vs the public cluster
(orthogonal enough to add hedge value).

## PREV PRIMARY (C5) — held as HEDGE candidate

**C5 LB 0.95404** — K=20 PathB × 0.82 + R21 × 0.08 + K=27 × 0.10.
File: `submissions/submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv`.
K=20 = K=18 set + d21_mem_inf + d22_mem_inf (lowest ρ_vs_K18 ever 0.7014).
Most likely FINAL_HEDGE per R2d (best OOF, original work, ≤30 bp behind
new PRIMARY 0.95446).

## Hedge ladder candidates (LB-confirmed) — Day-29 PM

| Rank | File | LB | Mechanism |
|------|------|-----|-----------|
| **PRIMARY** | `submission_d29_R22V4_raunakdey70_arunsolo15_C5_15.csv` | **0.95446** | 0.7 raunakdey + 0.15 arunsolo + 0.15 C5 (R22 public-blend) |
| HEDGE 1 (best-original) | `submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv` | 0.95404 | C5 K=20 + R21 + K=27 82/8/10 (best original-work LB) |
| HEDGE 2 (TIE family) | `submission_d21_d22_C7_K20_R21_K27_R28_renorm.csv` | 0.95404 | C5 + R28 xgb-ndcg @ 0.05 (TIE) |
| HEDGE 3 (d21-only mem-inf) | `submission_d21_C1_K19_R21_rankmean_89_11.csv` | 0.95403 | K=19 PathB × 0.89 + R21 × 0.11 |
| HEDGE 4 (R25 prior PRIMARY) | `submission_R25_K18_R21_rankmean_w89.csv` | 0.95402 | K=18 × 0.89 + R21 × 0.11 |

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
| 7 | R22-V4 raunakdey70+arunsolo15+C5_15 | n/a | **0.95446** | **+42 bp — public-blend axis activated** |

**All 7 daily slots used.** Day-30 + Day-31 = ~14 more probes available.

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
| R22 public-blend (PI-reauthorized 2026-05-29 PM) | **ACTIVE — +42 bp Day-29 PM** | V4 = raunakdey + arunsolo + C5 → LB 0.95446; +42 bp largest-single-slot lift this comp |

## Day-30 (tomorrow) priorities — R22 public-blend axis ACTIVE

Realized V4 +42 bp blows past every prior single-slot lift in this
comp. Top-5% boundary (0.95449) is 3 bp above current PRIMARY (0.95446).
Top public LB 0.95454 = 8 bp above. ~14 slots over Day-30 + Day-31.

1. **V0/V1 — pure raunakdey passthrough probe.** Confirm the public
   ceiling 0.95454. If so, defines our upper bound; weight sweeps then
   place V4 on a known interpolation curve. 1 slot.
2. **Weight sweep around V4** — A75+B15+P10, A80+B10+P10, A60+B20+P20,
   A65+B20+P15 — find the marginal-curve optimum. 3-4 slots Day-30.
3. **V2 (A70+B30) no-PRIMARY variant** — measures whether arunsolo
   alone (without our hedge) lifts further on public. 1 slot.
4. **V9 5-equal incl ours** — most-diversified bet; private-LB hedge.
   1 slot.
5. **Day-31:** narrow on the 2 best LB candidates as FINAL_PRIMARY
   (highest public LB) and FINAL_HEDGE (most robust per R2d ≤30 bp).
   Most likely C5 stays as FINAL_HEDGE (original work, -42 bp from
   V4 PRIMARY = within R2d 30 bp cap? **NO** — 42 bp > 30 bp, so C5
   would violate R2d cap. Pick a heavier-PRIMARY blend as hedge.)
6. **Avoid:** scouting more public sources (16 distinct lineages done;
   no other diverse axis); rebuilding K=20 pool variants (V4 dominates).

## Strategic posture

**Top-12% → within 3 bp of top-5% boundary in 1 submission.**
V4 LB 0.95446 vs top-5% boundary 0.95449. Top public LB 0.95454.
Highest realistic ceiling: ~0.95455-0.95460 if blend optimum sits
slightly above pure raunakdey.

**R2d hedge constraint check:** V4 PRIMARY at 0.95446; C5 at 0.95404.
Gap = 42 bp > 30 bp R2d cap. **C5 cannot be FINAL_HEDGE under R2d.**
Need a hedge ≥ 0.95416. Likely candidates: V5 (heavy hedge 0.50 ours)
or V6 (0.80 anchor + 0.10 each) or similar interpolation.

## Key learnings Day-29

0. **R22 public-blend axis: +42 bp single-slot lift.** PI 2026-05-29 PM
   authorized scrape+blend (reversing Day-8/Day-22 'original work only'
   directive). V4 = 0.7 raunakdey + 0.15 arunsolo + 0.15 C5 → LB 0.95446.
   Distinct-lineage scan (38 → 16 distinct → only 1 truly orthogonal:
   `arunklenin_solo`, ρ=0.974). Public-blender cloud is mutually
   ρ ≥ 0.997 — blending more than 1 cluster member is a no-op. Predicted
   band +3 to +5 bp; realized +42 bp = 8.4× upper-band break. Mechanism:
   the public cluster sat at +50 bp above our K=18/K=20 stack ceiling
   that no in-pool axis ever tapped.
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
