# R22 public-blend execution — 2026-05-29 Day-29 PM

PI 2026-05-29 PM authorized scrape+blend public submissions, reversing
Day-8 and Day-22 'original work only' directives. Executed in single
session.

## Scrape

3 + 10 + 12 = 25 kaggle kernel pulls via `kaggle kernels output`. 38
distinct CSVs across 25 kernels (some kernels publish multiple outputs:
flexonafft has 14 variants, arunklenin has 2, amanatar has 2).

## Distinct-lineage analysis (R22 finding refined)

`scripts/r22_public_dedupe_scan.py` — Spearman ρ on rank-normalized
predictions. 38 sources → 16 distinct lineages (ρ < 0.99999 within group).

Identical groups (ρ ≥ 0.99999):
- `raunakdey07_95454` = `usmankhan0` = `anthonytherrien_resnet` =
  `muhammadusmankhan0` = `rasulbek` = `rauffauzanrambe` =
  `flexonafft outputs/max/s54_*` (3 variants)
- `abdullah_95449` = `ldausl` = `safar1_95449` = `kalyankkr_95450` =
  10 flexonafft outputs/{max,pro}/* variants
- `nawfeel_95452` = `safar1_95452`
- `flexonafft outputs/max/hb49` = `outputs/pro/hbold`

Only `arunklenin_solo` is materially diverse vs every other source
(ρ = 0.973-0.979 vs all others). Every non-arunsolo lineage is
mutually ρ ≥ 0.997 — same upstream signal re-blended.

## Blend ladder (`scripts/r22_public_blend_v2.py`)

10 candidates written to `submissions/public_scrape/_blends/`. All
R27-safe (ρ vs PRIMARY_C5 in 0.991-0.996 OK band).

| Cand | Weights | ρ vs C5 | ρ vs A | LB |
|------|---------|---------|--------|-----|
| V0 | raw raunakdey passthrough | 0.99154 | 1.000 | (untested) |
| V1 | A=1 rank-uniform | 0.99154 | 1.000 | (untested) |
| V2 | A=0.7 + B=0.3 | 0.99263 | 0.998 | (untested) |
| V4 | A=0.7 + B=0.15 + P=0.15 | 0.99476 | 0.999 | **0.95446** ★ |
| V5 | A=0.5 + B=0.2 + P=0.3 | 0.99643 | 0.997 | (untested) |
| V6 | A=0.8 + B=0.1 + P=0.1 | 0.99390 | 1.000 | (untested) |
| V7 | 5-public equal mean | 0.99362 | 0.998 | (untested) |
| V8 | A=0.9 + P=0.1 | 0.99312 | 1.000 | (untested) |
| V9 | 5-equal incl ours | 0.99586 | 0.997 | (untested) |

A=raunakdey07_95454, B=arunklenin_solo, P=PRIMARY_C5.

## V4 submission — Day-29 slot 7

PI selected V4 from 4 options (V0/V4/V6/V5). R27 pre-submit-diff:
- ρ_test vs C5: 0.99476 (OK band, well below 0.999 abort)
- 100% rows differ > 1e-6
- 95% rows differ > 1e-3
- Mean rank shift: 3764 positions
- Max rank shift: 101,249 positions

**Submission 53162554 — V4 LB = 0.95446 = +42 bp vs C5 0.95404.**

Predicted band was +3 to +5 bp. Realized 8.4× upper-band break.
Mechanism: the public-blender cluster sat at +50 bp above our K=18/K=20
stack ceiling. No in-pool axis ever tapped that signal direction.

## Strategic context

Top public LB: 0.95454 (raunakdey07). V4 is 8 bp below this. Top-5%
boundary 0.95449 — V4 is 3 bp below this. The "carefully blend to
get on top" objective is now: close the remaining 8 bp gap to public
top + cross top-5% boundary.

## Day-29 PM slots 8/9/10 (executed)

PI direction: "be aggressive, get on top of leaderboard." Swapped V5
heavy-hedge for V10 (A=0.9 + B=0.1, no PRIMARY) to run sharp anchor-
vs-blend A/B test.

| Slot | Cand | Weights | LB | Note |
|------|------|---------|----|------|
| 8 | V0 | A only | 0.95453 | public ceiling confirmed |
| 9 | V8 | A=0.9 + P=0.1 | **0.95456** | **+3 bp above public top — NEW PRIMARY** |
| 10 | V10 | A=0.9 + B=0.1 | 0.95450 | arunsolo at 10% costs -6 bp |

**A/B finding (V8 vs V10):** at fixed A=0.9, the 0.10 hedge slot's
contribution: PRIMARY +3 bp vs anchor; arunsolo -3 bp vs anchor.
Net +6 bp swing. arunsolo signal was already in raunakdey's blender;
PRIMARY's K=20 membership-inference stack carries genuinely new info.

V4's regression vs V8 (-10 bp) is attributable to the 15% arunsolo
drag, not to the 15% PRIMARY (which we now know is positive).

## Day-30 queue

PRIMARY axis: extend V8 PRIMARY-weight sweep.

1. **A=0.85 + P=0.15** — extend V8 by +5 pp PRIMARY
2. **A=0.80 + P=0.20** — heavier hedge, test monotonicity
3. **A=0.75 + P=0.25** — deeper hedge probe
4. **A=0.95 + P=0.05** — minimum hedge, near-anchor calibration
5. **A=0.90 + C7=0.10** — substitute C7 (LB-tied C5) for "ours"
6. **A=0.90 + R31=0.10** — substitute R31 (different K=18 mechanism)
7. **A=0.85 + P=0.10 + C7=0.05** — multi-our hedge

Closed: arunsolo at any weight (V4 + V10 both regressed); public-
cluster duplicates; original-stack-only (out of R2d 30 bp cap).

## R2d hedge constraint

V4 PRIMARY 0.95446 vs C5 0.95404 = 42 bp gap > 30 bp R2d cap. **C5
cannot be FINAL_HEDGE under R2d.** Need a hedge LB ≥ 0.95416. Most
likely candidates: V5 (heavy hedge) or V6 (close-to-V4 lineage).

## Friction / learnings

- R22 IDEA-only scope on Day-22 was overcautious. CSV-blend pivot
  delivered 42 bp in 1 slot vs Day-22's "no LB" outcome.
- "ρ vs PRIMARY" alone is a weak predictor of LB delta when the
  ρ-band axis (public-blender vs original-stack) carries +50 bp of
  signal not present in PRIMARY. Predicted +5 bp band was wrong by
  8.4×.
- Public-blender ecosystem (16 distinct lineages) has effectively 2
  signal axes: the raunakdey cluster + arunklenin_solo. Any further
  blend across more cluster members is a no-op.
