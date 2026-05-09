# HANDOVER

This is the next-session brief. PI says **"handover"** → agent reads this
file and proceeds. PI says **"prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

---

## Where we are

**Active PRIMARY: K=4 forward-greedy + Path-B Compound × Stint, τ=100k.**
LB **0.95351**. Set 2026-05-08 PM as the new working baseline at a
deliberate −1.7 bp cost vs the prior 27-base PRIMARY (LB 0.95368).
Bases: yekenot-RealMLP (`d17_h1d_yekenot_full`), CatBoost-yekenot
(`p1_single_cb_v4_gpu`), HGBC deep (`f1_hgbc_deep`), LightGBM on the
original pre-synth data (`d16_orig_continuous_only`) — one per model
class.

Top-5% boundary is 0.95405 (gap −5.4 bp from K=4 PRIMARY). Leader is
at 0.95476 (gap −12.5 bp). Bootstrap CI on a 20% public draw is ±12 bp
wide, so both gaps fall partly inside public-LB sample noise.

Submissions used: 41 of 270. Today (2026-05-08): 2 used. Comp-day:
8 of 31. Days remaining: **23**.

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

---

## Day-9 PM decode-data-process-5uLq3

DGP-decode session (no LB submissions). Twenty-three probes
(Q1-Q10 + qB-qZ), twenty-five commits, twelve audit docs in
`audit/2026-05-09/`. Pushed disc-AUC gap host-vs-candidate from
0.999 (off-the-shelf SDV CTGAN baseline) to **0.7160** (analytic
resample-and-cond pipeline) — half of the way to the perfect-mimicry
lower bound of 0.4944.

**Read first**: `audit/2026-05-09/2026-05-09-EXEC-SUMMARY.md`
(plain-English) and `audit/2026-05-09/2026-05-09-PHASE-B-FINAL-and-plan-v3.md`
(full ledger).

**DGP picture (final):** input aadigupta1601 → custom marginal that
suppresses PitStop=1 by 0.54× → per-cell NN generator (the unsolved
residual; produces 73% novel values per cell, no noise, NOT BGMM /
KDE / affine / global / cross-cell-mixed) → structured per-cell
Driver/Stint sampling → drop Norm_TyreLife → ship.

**Architecture exclusion ledger** (host generator NOT any of):

| Architecture | disc-AUC |
|---|---:|
| SDV CTGAN (5/10/20 ep + synth-marginal cond) | 0.9993-0.9997 |
| SDV GaussianCopula | 0.9988 |
| SDV TVAE 10 ep | 0.9991 |
| SDV CopulaGAN | conclusive by pattern |
| noisy-orig + Gaussian sigma > 0 | monotone-worse |
| per-cell BGMM (4 floats) | 0.8643 |
| per-cell KDE bw 0.05-0.5 | 0.7448-0.7657 |
| global float sampling | 0.9907 |
| cross-cell mixing fraction > 0 | monotone-worse |
| affine moment-matching | 0.9883-0.9979 |

**New findings F11-F15:** Driver/Stint structured per-cell (qH +14 pp);
continuous columns strictly per-cell (qU); 73% novel `(Y, C, LapTime)`
keys per cell (qR); per-cell mean shifts -2.81 with std ratio 0.87 but
non-affine (qX/qY); d16++ standalone synth AUC 0.940 (+2.5 pp over
d16) but only +0.149 bp at K=4+1 (rank-lock saturates).

**qZ d16++ artifacts saved** at
`scripts/artifacts/dgp_v3_qZ_{oof_strat, test, train_synth}.npy`.
Stack-add gate measured at +0.149 bp (below +0.5 strict threshold).

**Next-session first actions** (in EV / cost order):
1. **TabDDPM-on-orig** if the install-debug session can land it. The
   single most likely candidate to close the 0.22 disc-AUC residual.
   ~30 min GPU.
2. **Normalising flow (RealNVP / NSF) per-cell** with cell-key
   conditioning. Skew-sensitive; matches the qX skewness diffs.
3. **Re-decompose K=4 PRIMARY** to swap d16 for qZ and re-measure;
   small expected lift but free.
4. **Accept structural decode as the answer** and wrap the comp;
   the rank-lock cap on K=4+1 means decode-derived features are
   bounded ≤1 bp at the LB.

**Friction tags promoted (this session, in audit/friction.md):**
`synth-rows-are-not-literal-copies-of-orig-rows` (retract P1c),
`host-not-in-sdv-library`, `noise-on-continuous-cols-makes-disc-worse-not-better`,
`cond-driver-stint-on-cell-saves-14pp`,
`extending-cond-axes-monotonic-down-to-LapN-then-sparsity-bites`,
`affine-moment-matching-fails-skewness-non-trivial`,
`host-cont-vals-strictly-per-cell-no-cross-cell-mixing`,
`rank-lock-saturation-puts-cap-on-K4plus1-with-decode-features`.

Postmortem: `audit/2026-05-09-postmortem-decode-data-process-5uLq3.md`.

---

## Archived per-branch session detail

Detailed per-branch session summaries live under `audit/`:

- `audit/archive-2026-05-07-handover-day15-17-pm-sections.md` — Days 15-17
- `audit/archive-2026-05-07-handover-pm-sections.md` — Days 17-18 sibling branches
- `audit/archive-2026-05-08-handover-day19-pm.md` — Day-19 overnight closure
