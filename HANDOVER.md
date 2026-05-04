# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file with the next
> session's brief.

---

## Today's session — Day 3 endgame (2026-05-04 ~23:00 UTC)

**Read order on session start** (skip the default; this file is the
synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R14)
2. `audit/2026-05-04-day3-endgame.md` — full Day-3 retrospective + Day-4 plan
3. `audit/friction.md` — load-bearing failure modes
4. `scripts/pre_submit_diff.py` — MANDATORY before every submit

Open with a 3-bullet read-back of state + the first mechanism to run.

## Where we are

- **Day 3 done. 10/10 used today. LB best = 0.94991** (M5h, also
  matched by M5h2 and M5j — all tied at quantization limit).
- **Day-4 starts at 00:00 UTC** (~60 min from this writing).
- **All Day-3-evening orthogonal-pool experiments REGRESSED on LB**:
  M5p −237bp, M5n_3b −291bp. Minimal-basis hypothesis FALSIFIED.
  GBDT consensus IS load-bearing for OOF→LB transfer.

## What completed overnight

| Base | Strat OOF | ρ vs M5h | Notes |
|---|---:|---:|---|
| RealMLP-TD (Kaggle T4) | 0.94582 | 0.972 | Strong, low diversity |
| H1 (pseudo-label LGBM) | 0.94265 | 0.965 | +19bp baseline; modest |
| EBM (interpret-core) | 0.93361 | 0.931 | Diverse but weak |
| LR with FE | 0.89684 | 0.869 | Most-diverse but very weak |

## Pre-built Day-4 slot-1 candidate: M5q

`submissions/submission_m5q_realmlp.csv` is **READY**.

  M5q = M5h pool + RealMLP (K=14)
  Strat OOF: 0.95057 (+1.4bp vs M5h)
  ρ vs M5h test: 0.99865 (PASS pre-submit-diff gate, just barely)
  RealMLP L1 in meta: 0.573 (6th-highest of 14 — meaningful weight)

Expected LB: 0.94991 ± 2bp. ρ borderline → 50/50 tie vs lift.

**Submit M5q as Day-4 slot 1** assuming no overnight changes by PI.

## Day-4 priority sequence

### Slot 1 (immediate): M5q
Submit pre-built `submission_m5q_realmlp.csv`. Pre-submit-diff vs M5h
already done (ρ=0.99865 → PASS). Expected LB 0.94991 ± 2bp.

### Slot 2: depends on M5q LB outcome
- If M5q LB > 0.94991 → build **M5q + H1 + EBM** (K=16). All
  strong-orthogonal bases on M5h. New build: ~5 min LR-meta refit.
- If M5q LB = 0.94991 (tie) → ρ-gate is too weak; need harder
  rank-shifts. Try **M5h + RealMLP_logit_only** (drop the rank/raw
  channels, force RealMLP to express only via probability scale)
  to amplify its rank influence. Or move to slot 3.
- If M5q LB < 0.94991 → very surprising; pivot to slot 3.

### Slot 3: HGBC multi-seed bag
E3, f1, f2 are single-seed. Bag 3 seeds (seeds 42/123/456 like
cb_slow-wide-bag did successfully). New pool member. ~30 min CPU.
Could give +1-3bp via variance reduction.

### Slot 4: Pseudo-label H1 RUN-2 with RealMLP-anchored agreement
H1 today used 13-GBDT pool agreement. Re-run with M5q's K=14 pool
(includes RealMLP). May produce different pseudo-labels. ~30 min CPU.

### Slot 5: Sequence-aware base (LSTM with Driver embedding)
**1-fold smoke probe FIRST** (per Rule 2 — applying tonight's lesson).
Then full 5-fold on Kaggle GPU. ~3-4h roundtrip.
Driver embedding (887 → 16d) + LapNumber + per-lap features over
(Race, Driver) sequence.

### Slot 6+: Stint-2-targeted FE base (NH9)
Pool is uniformly wrong on Stint 2. New features:
  - lap_since_last_pit × tyre_compound interaction
  - relative_pace = (LapTime − race_min) / race_std
  - within-Stint-2 conditional TE
Train ONE LGBM with these + raw features. New pool member.

### Slots 7-10: hedge / R5 final-window probes
Conservative submissions:
  - Best Strat OOF that regressed ≤30bp on public LB (R5 mandate)
  - PRIMARY = best public LB; HEDGE = best-OOF-that-regressed
  - Final-window lock at Day-30.

## Critical operating rules (FRESHLY VIOLATED Day-3 — read these)

1. **Pre-submit-diff before EVERY submit.** Run
   `python3 scripts/pre_submit_diff.py <candidate.csv>`. If ρ > 0.999,
   abort. Saved would-be-tied submissions on Day-3 retrospectively.

2. **1-fold smoke before any GPU 5-fold.** RealMLP took 175 min on
   Kaggle T4. We didn't smoke first; could have been killed early
   if it was 5h. Always 1-fold first, project, then full.

3. **Strat-only Day-3+** (Rule R1). GroupKF was DROPPED. Wasted 50%
   of compute on multiple Day-3 scripts running both anchors.

4. **In-pool tweaks of GBDT-heavy LR meta tie at LB 0.94991** within
   Kaggle's 5-decimal quantization. Don't burn slots on rho > 0.999
   variants.

## Falsified hypotheses (DON'T retry)

- Smaller pool → tighter LB gap (M5h2 v1, K=12 → tied)
- TE-key swap → LB delta (M5j d3a/d2a swap → tied)
- Calibration (per-Race/Year isotonic) → LB lift (overfit, inner-CV negative)
- Stint-2 specialist → in-segment lift (specialist −124bp on its segment)
- Hill-climb / LGBM-meta / L1-LR → beats LR meta (all within fold-noise)
- **Minimal-orthogonal-basis → break LB tie** (M5p −237bp, M5n_3b −291bp)
- 2-way TE (Driver×Compound, Race×LapBin, K=7) → orthogonal lift (+0.1bp)
- Sequence-FE (cum_pits, laps_since_last_pit, rolling-TE) → stack lift (+0.2bp)

## Open hypotheses for Day-4+

- **NH8**: RealMLP brings general LB lift, not Stint-2 fix
- **NH9**: Stint-2 needs feature engineering, not a new model family
- **NH10**: Distillation — train a small model to mimic M5h
- **NH11**: M5h_oof_probability as a feature (recursive base)
- **NH12**: Per-(Year, Stint) mini-models (segment ensemble)
- **NH13**: Logit-only stacker variant on M5h pool

## Calibration ladder (today's additions)

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| **m5h** | 0.95043 | **0.94991** | **CURRENT PRIMARY** |
| m5h2 (drop a_horizon, K=12) | 0.95044 | 0.94991 | tied |
| m5j (d3a swaps d2a, K=13) | 0.95044 | 0.94991 | tied |
| **m5p** (orth K=6) | 0.94839 | **0.94754** | **−237bp** REGRESSED |
| **m5n_3b** (min-orth K=4) | 0.94808 | **0.94700** | **−291bp** REGRESSED |
| RealMLP (standalone) | 0.94582 | n/a | strong; held |
| H1 pseudo-LGBM (standalone) | 0.94265 | n/a | held |
| EBM (standalone) | 0.93361 | n/a | held |
| LR-FE (standalone) | 0.89684 | n/a | held |
| **m5q** (M5h + RealMLP, K=14) | **0.95057** | n/a | **DAY-4 SLOT 1 CANDIDATE** |
| m5r (M5h2 + RealMLP, K=13) | 0.95056 | n/a | held; ρ=0.999 → tie expected |
| m5s (M5n_3b + RealMLP, K=5) | 0.94854 | n/a | held; minimal-basis is falsified |

## Pointers

- `audit/2026-05-04-day3-endgame.md` — Day-3 retrospective + Day-4 plan
- `audit/2026-05-04-day3-learnings.md` — pool weaknesses + new hypotheses
- `audit/2026-05-04-d3-pool-disagreement.md` — diversity diagnostic
- `audit/2026-05-04-d3-per-segment-analysis.md` — Stint-2 blind spot
- `audit/friction.md` — operating-rule failure modes (READ BEFORE NEW PROBE)
- `scripts/pre_submit_diff.py` — MANDATORY pre-submit gate
