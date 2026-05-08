# HANDOVER

This is the next-session brief. PI says **"handover"** → agent reads this
file and proceeds. PI says **"prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

---

## Where we are

**Best leaderboard score so far: 0.95368.** Rank 98 of 893 = top 11%. Top-5%
boundary is 0.95405 (gap −3.7 basis points). Leader is at 0.95476.

**Current PRIMARY** is a stack of 27 base models combined with a per-segment
shrinkage stacker (segments are tire compound × stint number, shrinkage
strength 100,000). Submissions used: 39 of 270.

For the full status — what's in the stack, what's been tried, what's open
— see `state/current.md` and `state/hypothesis-board.md`. The
calibration anchors are in `state/calibration-ladder.md`.

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

## What axes are still open

In priority order — see `state/hypothesis-board.md` for full reasoning:

1. **Sequence-level fingerprinting** of within-stint structure (Compound
   transitions, stint lengths, TyreLife progression). Predicted +1 to +3 bp.
   2-3 hours CPU. Only structurally-orthogonal axis remaining.
2. **RealMLP with 24 ensembles** instead of the current 4. Predicted
   +1 to +3 bp standalone. 3.5 hours GPU.
3. **Per-Year CatBoost specialists.** Predicted ±2 bp. 30 minutes GPU.
4. **Final-window R5 hedge preparation.** 30 minutes; hedge ladder
   already populated.
5. **Wrap-up posture.** Top-11% achieved; reserve compute for the next
   competition.
6. **FastF1 hard-join.** The only path to top-5%, but capped at 1.4%
   match rate by synthetic driver codes; cost 1-2 days.

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

## Archived per-branch session detail

Detailed per-branch session summaries live under `audit/`:

- `audit/archive-2026-05-07-handover-day15-17-pm-sections.md` — Days 15-17
- `audit/archive-2026-05-07-handover-pm-sections.md` — Days 17-18 sibling branches
- `audit/archive-2026-05-08-handover-day19-pm.md` — Day-19 overnight closure
