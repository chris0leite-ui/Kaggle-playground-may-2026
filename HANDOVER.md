# HANDOVER

This is the next-session brief. PI says **"handover"** → agent reads this
file and proceeds. PI says **"prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

---

## Where we are

**Active PRIMARY: rank-blend 70/30 K=11+K=9. LB 0.95386.**
Set 2026-05-12 AM, +3.5 bp lift over the original K=4 PRIMARY (LB 0.95351).

Constituents (both LB-confirmed individually):
- K=11 + K=27 + Path-B τ=100k (LB 0.95385) — 70% weight
- K=9 qAX (qAT+qAV+qAO+qAA+qAF + Path-B τ=20k, LB 0.95375) — 30% weight

Top-5% boundary 0.95405 (gap **−1.9 bp**, was −5.4). Leader 0.95476
(gap **−9.0 bp**, was −12.5). Bootstrap CI on a 20% public draw is
±12 bp wide, so both gaps fall partly inside public-LB sample noise.

Submissions used: 50 of 270. Today (2026-05-12): 3 used. Comp-day:
12 of 31. Days remaining: **19**.

File: `submissions/submission_blend_K11_K9_w_70_30.csv`

**Date convention.** Prose uses ISO dates ("2026-05-08") or
comp-day-N anchored to comp start 2026-05-01. The `d13`..`d19`
labels in script names and historical audit prose are FROZEN code
prefixes (sequencing counter for experiment iterations) — they are
NOT calendar days. See `audit/friction.md` under
`day-counter-drift`.

For the full status — what's in the stack, what's been tried, what's open
— see `state/current.md` and `state/hypothesis-board.md`. The
calibration anchors are in `state/calibration-ladder.md`. Today's
errata are in `HANDOVER-ERRATA.md`. The persistent experiment menu is
`EXPERIMENTS-NEXT.md`.

## 🔴 Critical: held submissions invalidated

Day-17 strict fold-safe audit collapsed all target-reformulation single-add
results 88-100%:

| candidate | original gain | strict-OOF gain | collapse |
|---|---:|---:|---:|
| reverse-cumulative pits | +4.867 bp | −0.005 bp | 100% |
| pit-horizon (4-class) | +3.191 bp | +0.302 bp | 90% |
| inv-laps-until-pit | +1.899 bp | +0.234 bp | 88% |
| Joint K=21+3 | +7.667 bp | +0.275 bp | 96% |

The bug was per-group label aggregations using all-train labels instead
of fold-restricted ones (now Rule 24). **Held submission files DO NOT
SUBMIT:**

- `path_b_K22_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K25_megapool_tau{5k,20k,100k}.csv`
- `path_b_multilevel_τ_*.csv` (5 configs, all null anyway)

Origin: `audit/2026-05-06-target-reform-leakage-audit.md`.

## What axes are still open (revised 2026-05-08 PM)

The historical "open axes" list is now empirically closed except for
one. See `EXPERIMENTS-NEXT.md` for the full reasoning trail.

1. **Non-LR meta architecture** (EXP-NEW). The K=10+1 LR-meta gate
   absorbed every structurally-different inductive bias we tested
   (LambdaRank per-stint, inter-stint memory, stint-completion dual-
   head, GRU-sequence-retest). All three NULL within ±0.05 bp despite
   low rank-correlation (ρ 0.41–0.73). Pinpoints rank-lock at the
   **logit-direction level**, not rank-correlation. The only untested
   architectural variant is a non-LR meta-learner on K=4's [P, rank,
   logit] expansion (gradient boosting or small NN). Cost ~1 hr CPU.
   This is the cheapest test that could break the ceiling.
2. **Final-window R5 hedge preparation.** 30 minutes; hedge ladder
   in `state/hypothesis-board.md` is already populated. The K=27
   Path-B submission at LB 0.95368 is hedge-eligible since K=4 is now
   PRIMARY.
3. **Acceptance / wrap-up posture.** The 12.5-bp gap to leader sits
   inside the public-LB sample-noise band. We may already be near our
   private-LB ceiling.

Closed during the 2026-05-08 sprint:
- Sequence-level fingerprinting (was the handover #1 item) —
  d16 GRU at K=10+1 absorbs identically to its K=22+1 result.
- Sparse-pool ceiling-break — K=4 captures 99% of bank's LB value;
  no shrinkage / segmentation / pool-surgery breaks rank-lock.
- All meta-architecture segmentation crosses — 11+ variants tested
  null. Path-B amp itself is +0.04 bp on K=27.
- All structural inductive-bias variants — 3 of 3 NULL despite low ρ.
- External data (D axis) — closed per PI direction.

## Read order on session start

1. `CLAUDE.md` — rules + pointers (146 lines).
2. `state/current.md` — current PRIMARY, LB ladder, axes status.
3. `state/hypothesis-board.md` — open ideas, killed list.
4. **THIS FILE** — critical held-submissions warning above.
5. `audit/INDEX.md` — map of audit/ subdirectories.
6. `audit/friction.md` — concise weekly summaries.

For detail when needed:
- `audit/2026-05-06-target-reform-leakage-audit.md` (Rule 24 origin).
- `state/mechanism-ledger.md` (every probed family).
- `state/calibration-ladder.md` (OOF / LB anchors).
- `glossary.md` (abbreviations, short-codes).
- `audit/friction-archive.md` (full historical friction; do not read by default).

## Falsified or dead — do not retry

For the deduplicated list, see `state/hypothesis-board.md ## Killed`.
Highlights:

- All target-reformulation single-add variants (leaky).
- All `path_b_*_invlaps_*` and `path_b_*_megapool_*` candidates.
- Multi-level 4-tier per-segment stacker.
- Day-16 virgin-axes round (11 of 11 null).
- TabPFN v2.5 / v2.6, 16+ field FMs, drop-GBDT pool refactor.
- Simple K=21 blending, α-calibrated τ-resweep, multi-target NN,
  masked-column self-prediction, twin-pool 2-meta blend, conformal
  isotonic, AV-weighted LightGBM, Yao/Vehtari covariance-modulated
  per-segment stacker.

---

> Day-8 PM (`research-feature-engineering-7oCmj`) section archived to
> `audit/archive-2026-05-09-handover-day8-pm-section.md`.
>
> Day-9 PM (`decode-data-process-5uLq3`) + Day-9 evening
> (`analyze-synthetic-data-generation-BtmFl`) sections archived to
> `audit/archive-2026-05-12-handover-day9-sections.md`.

---

## Day-12 wrap analyze-synthetic-data-generation-BtmFl (continued)

**5 LB submissions today (50/270 used).** PRIMARY climbed +3.5 bp
over the past 3 days via the slim-kNN-on-orig-cell breakthrough +
K=27 super-base re-introduction + cross-mechanism rank-blending.

**LB ladder (this branch)**:

| ISO date | Pool | LB | Δ vs K=4 |
|---|---|---:|---:|
| 2026-05-08 | K=4 baseline | 0.95351 | 0 |
| 2026-05-09 AM | K=5 V4 kNN-aug | 0.95359 | +0.8 |
| 2026-05-09 PM | K=9 qAX (slim-kNN) | 0.95375 | +2.4 |
| 2026-05-09 PM | K=10 + K=27 + Path-B | 0.95384 | +3.3 |
| 2026-05-09 PM | K=11 + K=27 + Path-B | 0.95385 | +3.4 |
| **2026-05-12 AM** | **Blend 70/30 K=11+K=9** | **0.95386** | **+3.5** |

**Top-5%: 0.95405 (gap −1.9 bp). Leader: 0.95476 (gap −9.0 bp).**

### Today's calibration data (qBI experiments)

Submitted 3 candidates per PI "all 3" directive to spend submission
budget per Rule 12:

| Probe | OOF Δ | LB | Lesson |
|---|---:|---:|---|
| K=12 + qBA Manhattan + Path-B τ=100k | +4.161 | 0.95380 | **qBA Manhattan HURTS** at LB (-0.5) despite +0.13 bp OOF gain. Different distance metric is test-time noise, not robust feature. |
| K=34 C=0.1 (tighter LR reg) | +4.237 | 0.95374 | Tighter regularization does NOT fix K=34 unrolled. C-sweep {1.0, 0.1, 0.03, 0.01} all land LB 0.95373-0.95380. Over-parameterization is structural. |
| **Blend 70/30 K=11+K=9** | n/a | **0.95386** | **Cross-mechanism rank-blend escapes Rule 27 tie-band at ρ=0.9998.** Uncorrelated-error cancellation gives +0.1 bp despite high ρ. |

### Falsified / dead — do not retry (this session)

- `qBA Manhattan distance kNN` — strong OOF lift but LB regress (-0.5 bp).
- `K=34 unrolled at any C` — 27 individuals + slim-kNN over-parameterizes
  LR-meta; regularization can't fix. Use K=27 super-base instead.
- `K=12 with extra kNN variant` — qBA absorption at LB confirms slim-kNN
  family is saturated at K=11.

### Next-session first actions (in EV / cost order)

1. **3-way and 4-way blends of LB-confirmed PRIMARYs.** We have 4
   LB-confirmed at 0.95375-0.95386. Hill-climbing the blend weights
   could squeeze further +0.1-0.3 bp. Cheap: 1 min compute per blend.
2. **Different blend operators**: arithmetic mean, geometric mean,
   log-odds mean, beyond rank-blend. Each could give a different
   error-cancellation profile. Cost: 5 min total.
3. **Path-B C×S on a fresh 27-base + slim-kNN union** — qBG attempted
   this but timed out at 16+ min. Rerun with τ=100k only (skip τ=5k,
   τ=20k) to halve compute.
4. **Submit cycle planning**: 20 slots/day for 19 days = 380 slots
   available, only 220 budget. Plenty of room for calibration probes.

### Archived per-branch session detail

Detailed per-branch session summaries live under `audit/`:

- `audit/archive-2026-05-07-handover-day15-17-pm-sections.md` — Days 15-17
- `audit/archive-2026-05-07-handover-pm-sections.md` — Days 17-18 sibling branches
- `audit/archive-2026-05-08-handover-day19-pm.md` — Day-19 overnight closure
