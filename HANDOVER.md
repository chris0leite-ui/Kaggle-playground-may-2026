# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file with the next
> session's brief.

---

## Today's session — Day 4 early (2026-05-05 ~05:30 UTC)

**Read order on session start** (skip the default; this file is the
synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R14)
2. `audit/2026-05-04-day3-endgame.md` — Day-3 retrospective
3. `audit/friction.md` — 25 logged failure modes; consult before
   any new probe (especially the new R-tagged ones)
4. `scripts/pre_submit_diff.py` — MANDATORY before every submit

Open with a 3-bullet read-back of state + first action.

## Where we are

- **Day 4, 1/10 used today.** New PRIMARY: **M5q LB 0.95005** —
  +14bp over M5h's 0.94991. **First substantive lift in many days.**
- **Headroom to top-5%** (0.95345): **34.0bp**.
- **Slot 2 pending PI direction.** All 4 stack candidates built
  overnight (M5t, M5u, M5v, M5w) — all ρ ≥ 0.997 vs M5q →
  TIE_EXPECTED. New base build needed to make slot 2 informative.

## The big find: NN-family bases have disproportionate LB amplification

M5q = M5h + RealMLP-TD (K=14). Strat OOF +1.4bp vs M5h. **LB +14bp.**
**10× amplification of OOF→LB delta.** Possible explanations:
1. GBDT OOFs are slightly optimistic on the test distribution; RealMLP
   "honest-corrects" without showing up in OOF AUC.
2. Test set has rows that benefit from smooth NN-style predictions
   that GBDT-heavy pool was biased away from.
3. Sample-size: 188k test rows allow rank corrections that don't
   surface in OOF AUC (which is integrated over a different
   distribution).

**Strategic implication**: NN-family bases provide LB EV that the
OOF metric understates. Adding more NN-family bases (TabNet,
sequence LSTM with embeddings, multi-seed RealMLP bag) is HIGH-EV
even if their standalone OOF is weak.

## Day-3 falsified hypotheses (DO NOT retry)

- Smaller pool → tighter LB gap (M5h2 K=12 → tied)
- TE-key swap → LB delta (M5j d3a/d2a swap → tied)
- Per-group calibration (per-Race / per-Year isotonic) → LB lift
  (in-sample +24.6bp, inner-CV −10.9bp; overfit)
- Stint-2 specialist → in-segment lift (specialist −124bp on its
  segment vs M5h)
- Hill-climb / LGBM-meta / L1-LR → beats LR meta (within fold-noise)
- **Minimal-orthogonal-basis → break LB tie** (M5p −237bp,
  M5n_3b −291bp). The 10 GBDT clones earn their slot.
- 2-way TE (Driver×Compound, Race×LapBin, K=7) → orthogonal lift
  (+0.1bp stacked)
- Sequence-FE (cum_pits, laps_since_last_pit, rolling-TE) → stack
  lift (+0.2bp)
- **Layered orthogonal bases on M5q anchor → break rank lock**
  (M5t/M5u/M5v all ρ ≥ 0.9997 vs M5q)

## Day-4 slot 2-10 plan (pending PI direction)

### Slot 2: build new base FIRST, then submit

Layered candidates all expected to tie. Need a NEW BASE to make slot
2 informative. Options ranked:

1. **HGBC multi-seed bag** (~30 min CPU). E3, f1, f2 are single-seed.
   Bag 3 seeds (42/123/456) following cb_slow-wide-bag pattern.
2. **Multi-seed RealMLP bag** (Kaggle GPU, ~6h overnight).
   Re-run RealMLP with seeds 123 + 456. Rank-bag the 3 OOF/test.
   Likely +1-3bp on top of M5q's +14bp lift.
3. **TabNet on Kaggle GPU** (~3h roundtrip; 1-fold SMOKE FIRST per
   Rule 2 lesson). Different NN family from RealMLP.
4. **Stint-2-targeted FE base** (NH9, ~30 min CPU). Features GBDT
   pool is missing on the shared blind spot:
   - lap_since_last_pit × tyre_compound interaction
   - relative_pace = (LapTime − race_min) / race_std
   - within-Stint-2 conditional TE.

### Slot 3+
Depends on slot-2 outcome and which new base lands first.

## Untried CatBoost levers (research note + YetiRank)

Stage-A research note `audit/2026-05-04-catboost-research.md` lives on
the catboost branch (`origin/claude/explore-catboost-options-jWhDD`).
Mechanisms 1, 5, 7 became `cb_year-cat`, `cb_slow-wide-bag`,
`cb_lossguide` (all in M5h). **Mechanisms 3, 4, 6, 8 were probed at
fold-0 only and never 5-folded or stack-incorporated.**

Probe results (all hit the 799-iter cap → 0.947-0.948 are FLOORS):

| Variant (CatBoost branch) | Fold-0 AUC | Promising? |
|---|---:|---|
| **MVS bootstrap** (mech #6) | **0.94792** | YES — equals cb_slow-wide-bag floor |
| One-hot (max_size=10) (mech #2 follow-up) | 0.94749 | maybe |
| Counter-only CTR (mech #4) | 0.94741 | diversity — no target in CTR |
| CTR-complex (max_ctr=6, mech #3) | 0.94731 | maybe |
| Ordered (mech #8, smoke) | 0.93908 | abandoned — slow |

**Plus the new untried recent-CatBoost feature**:

- **`loss_function='YetiRank'`** (or `YetiRankPairwise`) — pairwise
  ranking objective, structurally aligned with AUC. All current
  CatBoost bases use `Logloss` → adding a YetiRank base introduces
  a different loss surface, likely orthogonal predictions. Build:
  `scripts/d4_cb_yetirank.py` (TBD). ~30-60 min CPU; faster on GPU.

**Recommended ordering for Day-4 slot 2**:
1. **MVS 5-fold + slow+wide hyperparams** (`bootstrap_type=MVS,
   subsample=0.7, mvs_reg=0.1, lr=0.03, iter=4000`). Echoes the
   slow-wide-bag pattern that worked. ~30 min CPU.
2. **YetiRank** with same slow+wide config. ~30-60 min CPU.
3. **Counter-only CTR** as a diversifier (no target in CTR =
   different signal source).

These give us 3 new CatBoost bases of distinct character. Add to
M5q pool individually and check L1 contribution + ρ vs M5q.

### Final-window plan
- M5q stays as PRIMARY unless dethroned.
- HEDGE candidate per R2 (best-OOF that regressed ≤30bp on public).
- R5 final-window OOF probe.
- Lock-in by Day-30.

## Critical operating rules (FRESHLY VIOLATED Day-3 — read these)

1. **Pre-submit-diff before EVERY submit.** Run
   `python3 scripts/pre_submit_diff.py <candidate.csv>`. If ρ ≥ 0.999,
   abort. ρ ≥ 0.9997 → guaranteed tie at LB 5-decimal precision.
2. **1-fold smoke before any GPU 5-fold.** Codified after the
   RealMLP 175-min run (Rule 2 violation logged).
3. **Strat-only Day-3+** (Rule R1). No GroupKF in new scripts.
4. **In-pool tweaks of GBDT-heavy LR meta tie at LB 0.95005** within
   Kaggle's 5-decimal quantization. Pre-submit-diff prevents waste.
5. **Don't drop bases purely on L1/diversity grounds.** Minimal-
   basis falsified Day-3.

## Calibration ladder snapshot

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| baseline_two_anchor | 0.94075 | 0.94113 | LB-proxy ✓ gap +3.8bp |
| e3_hgbc | 0.94876 | 0.94870 | best single GBDT pre-CB |
| m5b | 0.94926 | 0.94891 | gap −3.5bp (anchor) |
| m5d | 0.95023 | 0.94963 | gap −6.0bp (widened) |
| m5h | 0.95043 | 0.94991 | gap −5.2bp |
| m5h2 (K=12) | 0.95044 | 0.94991 | tied |
| m5j (swap) | 0.95044 | 0.94991 | tied |
| m5p (orth K=6) | 0.94839 | **0.94754** | **−237bp** REGRESSED |
| m5n_3b (min-orth K=4) | 0.94808 | **0.94700** | **−291bp** REGRESSED |
| **m5q (M5h + RealMLP, K=14)** | **0.95057** | **0.95005** | **NEW PRIMARY**; +14bp; 10× LB amplification |
| RealMLP standalone | 0.94582 | (held) | strong, low diversity |
| H1 pseudo-LGBM | 0.94265 | (held) | +19bp baseline |
| EBM | 0.93361 | (held) | weak alone, GA²M family |
| LR-FE | 0.89684 | (held) | most-diverse, very weak alone |

## New base candidates ready (artifacts in scripts/artifacts/)

- `oof_realmlp_strat.npy` / `test_realmlp_strat.npy` — Day-4 anchor
- `oof_d3e_ebm_strat.npy` / `test_d3e_ebm_strat.npy`
- `oof_d3f_pseudo_lgbm_strat.npy` / `test_d3f_pseudo_lgbm_strat.npy`
- `oof_d3g_lr_fe_strat.npy` / `test_d3g_lr_fe_strat.npy`

## Held submissions (built but not submitted)

- `submission_m5t_layered.csv` — TIE_EXPECTED (K=15, +H1)
- `submission_m5u_layered.csv` — TIE_EXPECTED (K=16, +H1+EBM)
- `submission_m5v_lr_fe_layered.csv` — TIE_EXPECTED (K=15, +LR-FE)
- `submission_m5w_blend_50.csv` — PASS but lower OOF (risky)
- `submission_realmlp_standalone.csv` — RealMLP single-base
- `submission_d3e_ebm.csv` — EBM single-base
- `submission_d3f_pseudo_lgbm.csv` — H1 single-base
- `submission_d3g_lr_fe.csv` — LR-FE single-base

## Open hypotheses (NH8-NH13)

- **NH8**: RealMLP brings GENERAL LB lift, not Stint-2 fix
  (|Δ|@Stint2 only 0.0368 — confirmed on M5q).
- **NH9**: Stint-2 needs feature engineering, not new model family.
- **NH10**: Distillation — train small model to mimic M5q.
- **NH11**: M5q_oof_probability as a feature (recursive base).
- **NH12**: Per-(Year, Stint) mini-models (segment ensemble).
- **NH13**: Logit-only stacker variant on M5q pool.
- **NH14**: CatBoost YetiRank — pairwise ranking objective directly
  optimizes the AUC-equivalent metric. All 3 existing CB bases use
  Logloss; YetiRank introduces a different loss surface and likely
  orthogonal predictions. Build alongside MVS slow+wide.
- **NH15**: CatBoost MVS slow+wide — fold-0 probe (0.94792) on the
  catboost branch was never extended to 5-fold or stacked. With
  slow+wide hyperparams (lr=0.03, iter=4000, l2=8) following the
  cb_slow-wide-bag recipe, expect 0.948+ Strat OOF.
- **NH16**: CatBoost Counter-only CTR — pure frequency encoding,
  no target dependency in the CTR layer. Different signal source
  from existing CB bases (which all use default target-based CTRs).
  Fold-0 probe was 0.94741.

## Pointers

- `audit/2026-05-04-catboost-research.md` (on catboost branch
  `origin/claude/explore-catboost-options-jWhDD`) — Stage-A research
  note: 10 untried CatBoost mechanisms with citations. Mechanisms
  3/4/6/8 were probed-only and never 5-folded.
- `audit/2026-05-04-day3-endgame.md` — Day-3 retrospective + Day-4 plan
- `audit/2026-05-04-day3-learnings.md` — pool weaknesses + new hypotheses
- `audit/2026-05-04-d3-pool-disagreement.md` — diversity diagnostic
- `audit/2026-05-04-d3-per-segment-analysis.md` — Stint-2 blind spot
- `audit/friction.md` — 25 operating-rule failure modes
- `scripts/pre_submit_diff.py` — MANDATORY pre-submit gate
- `scripts/diag_pool_disagreement.py` — base-diversity diagnostic
- `scripts/diag_new_base_diversity.py` — new-base scorecard
- `scripts/m5qrs_realmlp_stacks.py` — M5q/M5r/M5s builders
- `scripts/m5tu_layered.py` — M5t/M5u builders
- `scripts/m5vw_diversity_blends.py` — M5v/M5w builders
