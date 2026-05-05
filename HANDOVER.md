# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file with the next
> session's brief.

---

## Today's session — Day 5 (2026-05-06)

**Read order on session start** (skip the default; this file is the
synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R14)
2. `audit/2026-05-05-d4-gbdt-meta-breakthrough.md` — slot-2 envelope
3. `audit/2026-05-05-d4-yetirank-nb-results.md` — base-add probes
4. `audit/2026-05-05-nn-stack-priorities.md` — bigger-move ordering
5. `audit/friction.md` — 30 logged failure modes (4 new from Day-4 PM)
6. `scripts/pre_submit_diff.py` — MANDATORY before every submit

Open with a 3-bullet read-back of state + first action.

## Where we are

- **Day 5, 0/10 used today.** PRIMARY = **M5q LB 0.95005** (M5h +
  RealMLP-TD, K=14, LR meta).
- **Headroom to top-5%** (0.95345): **34.0bp**.
- **22 days remaining** (deadline 2026-05-31). 8 slots/day available.

## Day-4 outcomes (single-paragraph synthesis)

Slot 1 = M5q at LB 0.95005 (+14bp over M5h; 10× OOF→LB amplification
from RealMLP). Slot 2 = m5_meta_lgbm_shallow (LGBM d=3 meta over same
K=14 base pool) at LB 0.95001 (-4bp; meta-switch costs). Probed two
new structurally-different bases (YetiRank ρ=0.666, NB ρ=0.853) — both
TIE_EXPECTED at LR-meta-stack level. Three independent confirmations
this session of `lr-meta-rank-lock-strong-anchor`. **Strategic
finding: base-pool signal ceiling is the binding constraint, not the
meta-learner.** ρ=0.995→4bp empirically validates the 0.999 tie
threshold.

## Day-5 plan — bigger moves only (PI directive)

**Stop sub-1bp tuning.** With 34bp headroom and 22 days remaining,
EV calculus = multi-bp moves only. Save seed-bagging / Optuna for
the final 3-day R5 window.

### Three high-EV candidate paths

#### Path A: NN-family multiplication (PRIMARY recommendation)

RealMLP gave 10× OOF→LB amplification adding ONE NN to a GBDT-heavy
pool. Hypothesis: a SECOND NN family with different inductive bias
compounds. Priority order from `audit/2026-05-05-nn-stack-priorities.md`:

1. **Multi-seed RealMLP bag** (Kaggle GPU T4x2, ~6h overnight). Same
   model, seeds 42 + 123 + 456, rank-bag the 3 OOF/test. Replaces
   M5q's RealMLP base; rebuild stack. Expected +1-3bp on top of M5q.
   *Note: this is the smallest of the three "bigger" moves; consider
   only as a parallel-stream while a fresh family is in flight.*
2. **TabNet on Kaggle T4x2** (~3h roundtrip; 1-fold SMOKE FIRST per
   Rule 2 — Day-3 RealMLP burned 175min skipping smoke). Attention-
   based feature selection. Distinct from RealMLP's MLP+embedding.
3. **FT-Transformer / SAINT** (Kaggle T4x2). Transformer family for
   tabular. Different inductive bias again.

Each new NN base added to M5q pool with LR meta first; then with
GBDT-meta when the pool composition changes (so a re-test of the
GBDT meta becomes informative again).

#### Path B: Pseudo-labeling at scale (HIGHEST CEILING)

H1 from CLAUDE.md hypothesis board, NH11 from the prior HANDOVER.
Use M5q's high-confidence predictions on the 188k test rows + multi-
base agreement gates (≥10/13 of M5h bases agree on pos/neg call OR
M5q proba in [0.95, 1.0] ∪ [0, 0.05]) to construct ~50-100k pseudo-
labels. Rebuild ALL 14 bases on train + pseudo-test (each base needs
its own re-OOF). Restack. **30bp-class move in prior comps when it
lands**; null risk is real (can over-amplify systematic errors).

Build artifact: `scripts/d5_pseudo_label_pool_rebuild.py`. ETA ~3-4h
local CPU for the GBDT bases; RealMLP rebuild needs Kaggle GPU.

Risks to gate against:
- Per-row leakage: ensure pseudo-test rows go ONLY to the OOF folds
  they wouldn't naturally appear in (i.e., add to all 5 train folds,
  evaluate OOF only on real labels).
- Overconfidence collapse: validate that test_rho_pseudo_vs_orig <
  0.998 so we're not just re-ranking the same scores.
- 4-gate leakage filter (G1-G4) before LB submit.

#### Path C: Recursive base (NH11)

Train a fresh GBDT base that includes `M5q_oof_proba` as a feature
plus all original features. The GBDT learns ROW-LEVEL CORRECTIONS
to M5q's predictions — cross-row interactions the original bases
never saw. Then re-stack on M5q + recursive_gbdt + (any new bases).

Build artifact: `scripts/d5_recursive_m5q_gbdt.py`. ETA ~30min CPU.
Smaller move than B, faster to test. Expected +2-5bp on top of M5q
if M5q leaves any row-level systematic error to correct.

### Recommended sequencing

1. **Day-5 morning**: launch Path C (recursive base, 30min CPU) AND
   Path A.2 TabNet smoke kernel (1-fold) in parallel. Smallest
   commitment, fastest data.
2. **Day-5 afternoon**: launch Path B (pseudo-label rebuild) — heavy
   compute, runs while we evaluate slots from morning.
3. **Day-5 slots**: 1-2 single-shots only; preserve 6+ slots for the
   pseudo-label restack candidate (highest variance, highest reward).

### Slot 2 candidate ALREADY built (tie-margin material)

`submission_m5_meta_lgbm_medium.csv` — most-divergent of the three
GBDT-meta variants (ρ=0.99436 vs M5q vs lgbm_shallow's 0.99508).
OOF -1bp; held. NOT a primary candidate, but if PI wants to extract
more LB-point information from the meta-switch theory, this is the
held option.

## Falsified hypotheses (DO NOT retry)

(carried forward from Day-3 + Day-4)

- Smaller pool → tighter LB gap (M5h2 K=12 → tied)
- TE-key swap → LB delta (M5j d3a/d2a swap → tied)
- Per-group calibration (per-Race / per-Year isotonic) → LB lift
- Stint-2 specialist → in-segment lift
- Hill-climb / LGBM-base-meta / L1-LR → beats LR meta
- Minimal-orthogonal-basis → break LB tie (M5p, M5n_3b regressed)
- 2-way TE / Sequence-FE → stack lift
- Layered orthogonal bases on M5q anchor → break rank lock
  (M5t/M5u/M5v/M5x/M5z all ρ ≥ 0.9995 vs M5q)
- **GBDT-meta over M5q pool → break LB ceiling** (Day-4 slot-2:
  -4bp LB; meta-switch is bounded; base-pool ceiling is binding)
- External F1 strategy dataset → recover features (Day-2 d2-probe1:
  5.6% test match rate, host-shuffled)
- Optuna sweep on RealMLP-TD as bigger move (lower EV per GPU-hour
  than seed bagging; lower than fresh NN family)
- Hand-crafted FE specifically for NN branch (RealMLP's internal
  embeddings re-derive most hand-features; d3a/d3b were null at
  stack level)

## Calibration ladder snapshot

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| baseline_two_anchor | 0.94075 | 0.94113 | LB-proxy ✓ gap +3.8bp |
| e3_hgbc | 0.94876 | 0.94870 | best single GBDT pre-CB |
| m5b | 0.94926 | 0.94891 | gap −3.5bp (anchor) |
| m5d | 0.95023 | 0.94963 | gap −6.0bp (widened) |
| m5h | 0.95043 | 0.94991 | gap −5.2bp |
| m5p (orth K=6) | 0.94839 | 0.94754 | −237bp REGRESSED |
| m5n_3b (min-orth K=4) | 0.94808 | 0.94700 | −291bp REGRESSED |
| **m5q (M5h + RealMLP, K=14)** | **0.95057** | **0.95005** | **PRIMARY**; +14bp; 10× LB amplification |
| m5_meta_lgbm_shallow | 0.95048 | 0.95001 | -4bp; meta-switch costs |
| RealMLP standalone | 0.94582 | (held) | strong, single-seed |
| H1 pseudo-LGBM | 0.94265 | (held) | +19bp baseline |
| EBM | 0.93361 | (held) | weak alone, GA²M family |
| LR-FE | 0.89684 | (held) | most-diverse, very weak alone |
| d4_cb_yetirank | 0.90508 | (held) | ρ=0.666 vs M5q (most diverse) |
| d4_nb (mixed) | 0.87984 | (held) | ρ=0.853 vs M5q |

## Held submissions (built but not submitted)

- `submission_m5x_yetirank.csv` — M5q + YetiRank, TIE_EXPECTED
- `submission_m5z_yetirank_nb.csv` — M5q + YetiRank + NB, TIE_EXPECTED
- `submission_m5_meta_lgbm_medium.csv` — meta-switch most-divergent
- `submission_m5_meta_hgbc.csv` — meta-switch HGBC variant
- `submission_m5t_layered.csv` / `m5u_layered.csv` / `m5v_lr_fe_layered.csv`
- `submission_m5w_blend_50.csv` — PASS but lower OOF (risky)
- `submission_realmlp_standalone.csv` / `d3e_ebm.csv` / `d3f_pseudo_lgbm.csv`
  / `d3g_lr_fe.csv` — single-base candidates

## Critical operating rules (FRESHLY VIOLATED Day-3/4 — read these)

1. **Pre-submit-diff before EVERY submit.** Run
   `python3 scripts/pre_submit_diff.py <candidate.csv>`. ρ ≥ 0.999 → tie.
   ρ in [0.994, 0.999] → real LB delta possible (Day-4 slot-2 calibration).
2. **1-fold smoke before any GPU 5-fold.** Codified after the
   RealMLP 175-min run (Rule 2 violation logged).
3. **Strat-only Day-3+** (Rule R1). No GroupKF in new scripts.
4. **Don't drop bases purely on L1/diversity grounds.** Minimal-
   basis falsified Day-3.
5. **Bigger-moves rule: weight candidates by EV_bp / day_invested.**
   Sub-1bp tuning saved for final-window R5 probe (Day-4 friction).
6. **Strategy review before propose: grep audit/ for prior probes on
   the proposed mechanism.** External data already-tested-d2 friction.

## Pointers

- `audit/2026-05-05-d4-gbdt-meta-breakthrough.md` — slot-2 envelope
- `audit/2026-05-05-d4-yetirank-nb-results.md` — base-add probes
- `audit/2026-05-05-nn-stack-priorities.md` — bigger-move ordering
- `audit/2026-05-04-day3-endgame.md` — Day-3 retrospective
- `audit/2026-05-04-d2-probe1-external-join.md` — external join FAILED
- `audit/2026-05-04-catboost-research.md` (catboost branch) — CB levers
- `audit/friction.md` — 30 logged failure modes
- `scripts/pre_submit_diff.py` — MANDATORY pre-submit gate
- `scripts/d4_cb_yetirank.py` — YetiRank base build
- `scripts/d4_naive_bayes.py` — NB base build
- `scripts/d4_gbdt_meta.py` — GBDT-meta sweep over M5q pool
- `scripts/m5qrs_realmlp_stacks.py` — M5q/M5r/M5s builders
