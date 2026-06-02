# Where we are right now

Single source of truth for the current PRIMARY, LB ladder, axes
status, and submission count. **Rewrite this file when PRIMARY
changes** — do not tail-append. Prior versions live in
`audit/archive-YYYY-MM-DD-current-md-*.md`.

## COMP CLOSED — private reveal 2026-06-02

- **Final standing: rank 148 / 3023 → top 4.90%** (cleared top-5%).
- **Official score:** 0.95461 (HEDGE W2_R31 scored as best-of-selected).
- **PRIMARY W2 private:** 0.95460 ; **HEDGE W2_R31 private:** 0.95461
  (hedge beat primary +1 bp — distinct-backbone logic validated).
- **Best of all our submissions (not selected):** W6 (P=0.35) and E3
  PMEAN q=3 both at private 0.95466 ≈ top-2.5%. ~5–6 bp left on table.
- **Closing postmortem:** `audit/2026-06-02-postmortem-research-improvements-jjI84.md`.
- **R8d log:** end-of-comp percentile appended to
  `.claude/skills/kaggle-comp/improvements.md`.


**Date convention:** ISO dates ("2026-05-29") or comp-day-N anchored
to comp start 2026-05-01. The `d13`..`d22` labels in script names
and old audit prose are FROZEN code prefixes — never calendar days.

## PRIMARY (active) — set 2026-05-30 Day-30 (R22-W2 / plateau pick)

**LB 0.95457** — R22-W2 rank-mean:
0.85 raunakdey07_95454 (public anchor)
+ 0.15 PRIMARY_C5 (K=20 mem-inf stack).

File: `submissions/public_scrape/_blends_d30/D30_W2_A85_P15.csv`.
**Rank 20 / 2791 (top-0.72%) — was 21.**

5-way TIE plateau at 0.95457: W2 (P=0.15), W3 (P=0.20), W4 (P=0.25),
M1 (multi-ours), L1 (logit-space V8). W2 picked as least-perturbation
from known-V8.

## FINAL pick set (Day-31 lock — LOCKED 2026-05-31)

- **FINAL_PRIMARY:** **W2** — `D30_W2_A85_P15.csv` (Kaggle ref 53180053).
  0.85 raunakdey + 0.15 C5 (K=20+R21+K=27). LB **0.95457**.
- **FINAL_HEDGE:** **W2_R31** — `D31_W2R31_A85_R31_15.csv` (Kaggle ref 53216236).
  0.85 raunakdey + 0.15 R31 (K=18+R30+R7.2). LB **0.95457**.

Both at plateau max (0.95457). HEDGE uses entirely-distinct ours-backbone
(K=18+R30+R7.2 vs K=20+R21+K=27) — diversifies the C5-specific-private-bias
failure mode without sacrificing public LB. Selected over V0 (0.95453, -4 bp)
because private LB is row-iid (per comp-context) → V0's "blends don't
generalize" tail hedge is low-probability and not worth the 4 bp deficit.

PI lock-in confirmed by user via Kaggle UI.

## V0/V4/V8/V10 day-29 result table

| Cand | Weights | LB | Δ vs V8 | Note |
|------|---------|----|---------|------|
| V0 | raunakdey raw | 0.95453 | -3 bp | public ceiling confirmed |
| V4 | A=0.70, B=0.15, P=0.15 | 0.95446 | -10 bp | arunsolo drag identified |
| **V8** | **A=0.90, P=0.10** | **0.95456** | **0** | NEW PRIMARY |
| V10 | A=0.90, B=0.10 | 0.95450 | -6 bp | arunsolo HURTS at 10% too |

**Key A/B finding (V8 vs V10):** at fixed anchor weight 0.90, swapping
the 0.10 hedge from PRIMARY (V8 0.95456) to arunsolo (V10 0.95450)
costs **−6 bp**. arunsolo signal was already absorbed by the raunakdey
blender; our K=20 stack carries genuine new info to public-anchor.

**Mechanism interpretation:** the +3 bp from PRIMARY hedge over pure
anchor (V8 vs V0) is the membership-inference + K=20 PathB signal
leaking through at 10% weight. Our prior K=20 work was load-bearing
for breaking past the public ceiling — public top 0.95453 is a hard
limit for pure-public stacks.

## R2d hedge constraint

V8 PRIMARY 0.95456. R2d cap = 30 bp → HEDGE ≥ 0.95426.
- V0 (0.95453) ✓ — structurally distinct (pure public, no original work)
- V4 (0.95446) ✓ — different blend recipe
- V10 (0.95450) ✓ — anchor + arunsolo
- C5 (0.95404) ✗ — 52 bp below, violates R2d cap

**V0 is the natural FINAL_HEDGE: highest LB, structurally distinct from
V8 (no PRIMARY weight), no original-work commitment.**

## Hedge ladder candidates (LB-confirmed) — Day-29 PM

| Rank | File | LB | Mechanism |
|------|------|-----|-----------|
| **PRIMARY** | `submission_d29_R22V8_raunakdey90_C5_10.csv` | **0.95456** | 0.9 raunakdey + 0.1 C5 (R22 public-blend) |
| HEDGE A (best-public-only) | `submission_d29_R22V0_raunakdey_RAW.csv` | 0.95453 | raunakdey raw passthrough |
| HEDGE B (arunsolo blend) | `submission_d29_R22V10_raunakdey90_arunsolo10.csv` | 0.95450 | 0.9 raunakdey + 0.1 arunsolo |
| HEDGE C (V4 original) | `submission_d29_R22V4_raunakdey70_arunsolo15_C5_15.csv` | 0.95446 | 0.7 raunakdey + 0.15 arunsolo + 0.15 C5 |
| out-of-R2d | C5 / C7 / C1 / R25 family | 0.95402-0.95404 | original-work stack (violates 30 bp R2d cap vs V8) |

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
| 7 | R22-V4 raunakdey70+arunsolo15+C5_15 | n/a | 0.95446 | +42 bp — public-blend axis activated |
| 8 | R22-V0 raunakdey raw passthrough | n/a | 0.95453 | public ceiling confirmed (+7 bp vs V4) |
| 9 | R22-V8 raunakdey90+C5_10 | n/a | **0.95456** | **NEW PRIMARY — +3 bp ABOVE public top** |
| 10 | R22-V10 raunakdey90+arunsolo10 | n/a | 0.95450 | arunsolo at 10% HURTS by -6 bp vs V8 |

**10 of 10 daily slots used.** Day-30 + Day-31 = ~14 more probes available.

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

## ACTUAL LB RANK (pulled 2026-05-29 19:15 UTC)

**Rank 21 of 2791 teams (top-0.76%)** at V8 LB 0.95456.
- Rank 1: 0.95494 (+38 bp gap to leader)
- Rank 8 / 0.95470: Don Mani (+14 bp gap)
- Rank 10 / 0.95466: Andreas Palmgren (top-10 boundary, +10 bp gap)
- Rank 16 / 0.95459: Ravi Ramakrishnan + `arunklenin` + `ravi20076` (uses
  arunsolo successfully — proof arunsolo IS valuable when blended right)
- **Rank 21 (us): 0.95456 — V8**
- Rank 22-56: 35 teams clustered at 0.95454-0.95455 (public-blender ceiling)

V8's +3 bp put us above 35 teams sitting at the raunakdey-cluster ceiling.

## HIDDEN AXIS — d22 raw orthogonality

Found 2026-05-29 PM by inspecting blend dilution:
- ρ_d22_vs_anchor = **0.682** (k=1 NN exact-copy raw)
- ρ_d21_vs_anchor = **0.727** (k=5 NN membership raw)
- ρ_C5_vs_anchor  = 0.991
- ρ_K20_alone_vs_anchor = 0.990

V8 dilutes d22 to ~0.4% effective weight (10% C5 × 82% K=20 × ~5% d22 share).
Pure d22 raw at 10% blend weight = **25× more concentrated** — this is
our most-orthogonal axis sitting under-used.

Built I1/I2/I3 isolation probes for Day-30 slot 1-3:

| Slot | File | ρ_vs_V8 |
|------|------|---------|
| 1 | `submission_d30_R22I2_raunakdey90_d22raw_10.csv` (d22 raw @ 10%) | 0.99744 |
| 2 | `submission_d30_R22I3_raunakdey90_d21raw_10.csv` (d21 raw @ 10%) | 0.99779 |
| 3 | `submission_d30_R22I1_raunakdey90_K20pathb_10.csv` (K=20 alone @ 10%) | 0.99999 |

I2/I3 are well below the 0.9990 TIE band — structural LB delta plausible.
I1 will likely tie V8 (control probe).

## Day-30 queue (built 2026-05-29 evening, fire at UTC reset)

20 candidates pre-built in `submissions/public_scrape/_blends_d30/`.
**R27 caveat: ALL Day-30 candidates ρ_vs_V8 ∈ [0.99981, 1.00000].** Every
single one trips TIE_EXPECTED on the standard threshold. Override
authorization is per-slot, justified by today's V4/V8/V10 axis
activations that broke similar bands by +42 bp and +3 bp.

Sharpened 8-slot priority (PI authorizes at fire-time):

| Slot | Cand | Mechanism | ρ_vs_V8 |
|------|------|-----------|---------|
| 1 | W3 A=0.80 P=0.20 | More-ours monotonicity | 0.99992 |
| 2 | O2 A=0.90 R31=0.10 | R31 vs C5 as "ours" | 1.00000 |
| 3 | M1 A=0.85 + 3-way ours (C5+C7+R31) | Multi-ours diversification | 0.99998 |
| 4 | E1 q=2 power-mean | Quadratic mean curvature | 0.99995 |
| 5 | E3 q=3 power-mean | Cubic mean (most curvature) | 0.99984 |
| 6 | L1 logit-space V8 | Logit-space blend | 0.99999 |
| 7 | C1 confidence-conditional | Mid-range up-weight ours, tails down | 0.99995 |
| 8 | W1 A=0.95 P=0.05 | Min-ours calibration | 0.99998 |

Reserve (2 slots): W2 A=0.85 P=0.15, M3 A=0.90 C5_07 R31_03.

**Generator:** `scripts/r22_d30_candidates.py`.

## Strategic posture

**ON TOP of the leaderboard.** V8 LB 0.95456 vs public top 0.95453.
Top-5% boundary 0.95449 = -7 bp below us. Top-1% boundary unknown;
likely ~0.9546 area.

R2d hedge candidates LB-confirmed (all ≥ 0.95426 cap):
- V0 0.95453 ✓ (pure public, structurally distinct)
- V10 0.95450 ✓
- V4 0.95446 ✓

**FINAL selection plan (Day-31):**
- FINAL_PRIMARY = best LB found Day-30 (V8 or weight-sweep winner)
- FINAL_HEDGE = highest-LB structurally-distinct candidate (likely V0
  pure public; or a V8-variant with different "ours" source).

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
