# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 7 (2026-05-08)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R14)
2. `audit/2026-05-07-d6-f1-2-LB-result.md` — F1.2 LB win (+2.1bp PRIMARY)
3. `audit/2026-05-07-d6-f1-2-multi-rule.md` — K=18 build details
4. `audit/2026-05-07-d6-critic-loop.md` — Rule 14 audit + 5 untried mechs
5. `scripts/pre_submit_diff.py` — MANDATORY before every submit.
   **NOTE**: tighten ρ threshold to 0.9995 OR diff on `.npy` arrays.
   CSV precision loss flagged TIE_EXPECTED on the Day-6 +2.1bp lift.

Open with a 3-bullet read-back of state + first action.

## Where we are

- **Day 7, 0/10 used today.** Day-6 closed at 1/10 used → **LB
  0.95026 +2.1bp**.
- **PRIMARY** = `d6_k18_multi_rule` LB 0.95026 (M5q + 4
  rule-residual bases). Strat OOF 0.95065. Gap NARROWED from
  −5.2bp (M5q) to −3.9bp (K=18).
- **Headroom to top-5%** (0.95345): **31.9bp**.
- **20 days remaining** (deadline 2026-05-31). 9 slots/day.
- **Active parallel branch**: PI is exploring a different angle on
  another branch. This branch's continuation is the RealMLP-bag
  thread; coordinate at merge points.

## RealMLP bag thread — CLOSED, NULL salvage

`kernels/realmlp-bag-gpu/` was cancelled mid-fold-3 of seed 123
after parallel-branch probes (P10 in
`audit/2026-05-08-data-probe-results.md`) downgraded bag EV to
Tier-3. Salvage in `scripts/d7_realmlp_partial_bag.py` tested two
paths:

| Path | K=18 OOF | Δ d6_k18 | ρ vs d6_k18 |
|---|---:|---:|---:|
| B (bagged TEST + seed-42 OOF) | 0.95065 | −0.02bp | 0.99955 |
| C (hybrid OOF + bagged TEST) | 0.95066 | +0.08bp | 0.99964 |

Both above the 0.9995 tightened tie threshold. **Confirmed Tier-3
classification**: variance-reduction on 1 of 18 bases caps stack
lift at ≤0.1bp. Both submissions HELD — see
`audit/2026-05-08-d7-realmlp-partial-bag-null.md`. **Do not retry
RealMLP bagging.**

## Updated priors from parallel branch (read first)

`origin/claude/read-handover-850hm` published 10 data probes +
strategic menu Day-8. Material updates to our priors:

- **P1 falsifies sequence models.** Test groups average 2.25 laps;
  only 9.7% have ≥5 consecutive laps. Big LSTM/Transformer is
  bounded. **Drop F4 from the menu.** What survives: 1-step lookup
  features (next_compound, prev_compound, laps_into_stint).
- **P2 falsifies retrieval.** kNN distances too large → TabR /
  Hopular / TabPFN-context bounded.
- **P5**: 68% of test rows have computable `next_compound` —
  large unused signal.
- **P10**: pool extracts what's extractable from the 14 raw
  features. **No residual cohort with |bias|≥2pp.** Lift requires
  NEW signals, not better extraction.
- **P6**: StratifiedKFold has 80% within-group leakage. OOF is
  optimistic by ~5bp (matches our gap). Use Strat as LB proxy
  per R1 but add GroupKFold(Race, Driver, Year, Stint) as
  diagnostic.
- **C6**: compute is NOT the binding constraint.

## Day-7 first action — T1.3 Q12 forced-pit rule_residual

Per parallel-branch strategic menu T1.3 (highest EV/hour). F1
regulation requires ≥2 distinct dry compounds per driver per race.

1. **Verify (Driver, Race, Year)-group atomicity** — check
   `train.groupby(['Driver','Race','Year'])['Compound'].nunique()`
   distribution. If groups are atomic in the synth (≥1 distinct
   compound per group consistently), proceed.
2. **Build rule_proba via lookup**:
   - `compounds_used_so_far[Driver, Race, Year]` (within-group
     cumulative distinct count up to lap k).
   - `must_change_compound = (n_distinct == 1) AND
     (LapNumber > race_total_laps × 0.6)`.
   - `forced_pit_pressure = must_change_compound × Stint`.
3. **Train HGBC residual GBDT** on raw features predicting
   `(target − rule_proba)`, à la `scripts/d6_rule_residual.py`.
4. **Pool-add to K=18 → K=19** stack. Reuse
   `scripts/d6_multi_rule.py` (add the new rule_residual to the
   `RULES` list; change file paths).
5. **Pre-submit-diff vs `submission_d6_k18_multi_rule.csv`**
   (NEW PRIMARY ref). Threshold **ρ < 0.9995** (NOT 0.999).
6. **If OOF > 0.95068 AND ρ < 0.9995 → slot Day-7 slot-1.**

EV prior: +5-10bp standalone, +1.5-3bp K=19 stacked. Cost 3-5h CPU.

## Re-rankable next moves (post-T1.3)

| # | Move | Cost | EV (bp) | Notes |
|---|---|---|---:|---|
| A | T1.3 Q12 forced-pit rule | 3-5h CPU | +1.5-3 stack | TOP — Day-7 slot-1 |
| B | T1.1 TabM 1-fold smoke | 1h GPU | gate | Rule 2; only proceed to 5-fold if smoke OK |
| C | T1.1 TabM 5-fold | 6-10h GPU | +2-8 | new NN family |
| D | T1.2 Multi-formulation L1 | 6-10h CPU | +2-10 | Deotte April-2025 winner pattern |
| E | T1.4 Hazard-rate reformulation | 3-4h | +1-7 | attacks Stint-2 −341bp blind spot |
| F | T1.5 Deotte L2 std/mean meta | 30min CPU | +0.5-3 | distinct from F5 (L3 weighted avg) |
| G | T2.1 next_compound feature | 1-2h CPU | +1-4 | 68% test computable per P5 |
| H | T2.2 prev_compound × laps_into_stint rule | 1-2h CPU | +1-3 | targeted Stint-2 attack |

## Falsified / dead — do NOT retry

- **Big sequence models** — P1 (parallel branch).
- **kNN / retrieval / TabR / Hopular** — P2.
- **TabPFN-2.5 ICL ensemble** — same regime issue as P2.
- **RealMLP bagging** — Day-7 partial-bag null (this audit).
- **Broad pseudo-labeling** — Day-5 partial-pseudo K=14.
- **F5 aux-feature GBDT-meta** — Day-6.
- **Move B 2-base [M5q, recursive]** — Day-6.
- **Per-Race / per-Stint isotonic** — Day-3 in-CV regress.
- **Reintroduce `Normalized_TyreLife`** — host-removed.

## Critical operating rules (freshly used Day-6)

1. **Pre-submit-diff before EVERY submit.** **Tighten ρ to 0.9995**
   or diff on `.npy` — CSV precision loss flipped verdict on Day-6
   K=18.
2. **Mechanism-class-only**: pool-tweaks via LR-meta are dead
   (3× rank-lock). New slots must change L1 formulation, meta
   family, OR add orthogonal model class.
3. **Predicted-gap gate**: pred-gap <−7bp needs PI sign-off.
4. **Minimal-input-meta sanity check**: for every base-add, train
   2-comp meta on `anchor + new` only. If 2-comp OOF < anchor,
   K-comp lift was memorization. Reject.
5. **Strat-only Day-3+** (R1; U3 confirmed i.i.d.).
6. **Track gap direction** — Day-6 K=18 narrowed gap (opposite of
   d5 over-amp). Real positive transfer.

## Calibration ladder snapshot (Day 7 morning)

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | gap −5.2bp |
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | **gap −3.9bp NARROWED** |

## Held submissions (do not blindly submit)

Carry-forward TIE/NULL: `m5x_yetirank`, `m5z_yetirank_nb`,
`m5_meta_lgbm_*`, `m5_meta_hgbc`, `d5_meta_k15_*`, `m5_k15a/b/c`.
Burned: `d5_partial_pseudo_m5q` (−4.2bp).
Day-6 falsified: `d6_aux_meta_with_aux`, `d6_2base_v[1-4]_*`,
superseded `d6_k15_rule_residual`/`d6_k16_two_diverse`.
Day-7 NULL salvage: `d7_realmlp_bag_part{B,C}.csv`.

## Pointers

- `audit/2026-05-07-d6-f1-2-LB-result.md` — Day-6 LB win
- `audit/2026-05-07-d6-f1-2-multi-rule.md` — K=18 build
- `audit/2026-05-08-d7-realmlp-partial-bag-null.md` — bag null
- `audit/2026-05-08-strategic-menu-wider-steps.md` (parallel branch) — Tier-1/2/3 EV
- `audit/2026-05-08-data-probe-results.md` (parallel branch) — P1-P10 priors
- `scripts/d6_multi_rule.py` — F1.2 builder (reuse for T1.3 K=19)
- `scripts/d6_rule_residual.py` — F1.1 builder
- `scripts/d7_realmlp_partial_bag.py` — bag salvage (closed)
- `scripts/pre_submit_diff.py` — MANDATORY (use ρ < 0.9995)
