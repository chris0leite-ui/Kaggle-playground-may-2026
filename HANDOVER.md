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

## Day-8 PM research-feature-engineering-7oCmj

EXP-NEW Phase 1-5b FE/meta campaign (ISSUES leaf 11) closed `null`.
PRIMARY unchanged @ LB 0.95351; 0 of 270 submission slots used.

| Probe | OOF Δ vs PRIMARY 0.95403 | Verdict |
|---|---:|---|
| A3-7 UID smoothing dry-run | −124 bp | leakage FAIL |
| 6 of 7 Phase-1 smoke picks | null/regress | FAIL |
| A2-2 mandatory_compound_rule smoke | +9.3 bp | smoke-only |
| A2-2 single-LGBM 5-fold | +1.4 bp standalone | partial absorb |
| A2-2 K=4+1 plain LR-meta | +0.302 bp; TIE_EXPECTED | < +0.5 PASS |
| A2-2 K=4+1 Path-B C×S τ=100k | +0.26 bp; ρ 0.999893 | WEAK |
| A2-8 LightGBM stack-meta on K=4 | −1.30 bp | FAIL |

**Rule 7 research scan** (Frontiers AI 2025 Bi-LSTM, Optimum Racing
IJRASET 2025): 4 of 5 mechanisms duplicate A3-1 RankSortedGaps
(already null). Only genuinely untried lever: Bi-LSTM/GRU sequence
base on 10-lap windows, ~30-60 min Kaggle T4, deferred.

**Next-session first actions.** (1) Bi-LSTM/GRU sequence base on
K=4 — cheapest untried lever, +0.5 bp gate. (2) R5 hedge prep
(~30 min CPU). (3) Acceptance posture — −12.5 bp gap to leader
inside sample-noise band; private-LB ceiling may be near.
Artifacts: `scripts/probe_a2_2_pathb_K4.py`,
`scripts/probe_a2_8_stack_meta.py`, postmortem
`audit/2026-05-08-postmortem-research-feature-engineering-7oCmj.md`.

---

## Archived per-branch session detail

Detailed per-branch session summaries live under `audit/`:

- `audit/archive-2026-05-07-handover-day15-17-pm-sections.md` — Days 15-17
- `audit/archive-2026-05-07-handover-pm-sections.md` — Days 17-18 sibling branches
- `audit/archive-2026-05-08-handover-day19-pm.md` — Day-19 overnight closure
