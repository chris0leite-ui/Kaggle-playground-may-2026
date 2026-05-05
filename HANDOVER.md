# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file with the next
> session's brief.

---

## Today's session — Day 6 (2026-05-07)

**Read order on session start** (skip the default; this file is the
synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R14)
2. `audit/2026-05-06-d5-path-b-phase2.md` — Path B PASS, partial-pseudo K=14
3. `audit/2026-05-06-d5-gbdt-meta-k15.md` — meta-add ceiling confirmed
4. `audit/2026-05-06-d5-tabnet-smoke-fail.md` — TabNet parked
5. `audit/2026-05-06-d5-path-c-recursive.md` — recursive null + 3rd rank-lock
6. `scripts/pre_submit_diff.py` — MANDATORY before every submit

Open with a 3-bullet read-back of state + first action.

## Where we are

- **Day 6, 0/10 used today.** Slot-1 candidate held: `submission_d5_partial_pseudo_m5q.csv`.
- **PRIMARY** = M5q LB 0.95005 (until partial-pseudo K=14 lands an LB).
- **Headroom to top-5%** (0.95345): **34.0bp**.
- **21 days remaining** (deadline 2026-05-31). 8 slots/day available.
- **Phase 3 in flight or queued** — see "Day 6 plan" below.

## Day-5 outcomes (single-paragraph synthesis)

Day 5 stress-tested the ceiling against the M5q pool and found it.
Six probes against M5q (recursive K=15 LR stack 3 variants, K=15
GBDT-meta sweep 3 variants, recursive standalone 2-base stack,
TabNet default smoke) all NULL or FAIL. **The 3rd independent
confirmation of `lr-meta-rank-lock-strong-anchor` AND a fixed-ceiling
GBDT-meta finding** (-1bp uniformly when adding any base to K=14).
Then Path B (HANDOVER's HIGHEST-CEILING move) was launched: Phase 1
MVP on e3_hgbc rebuilt with pseudo-labels showed +4.1bp OOF /
ρ=0.99593 — both gates PASS. Phase 2 expanded to 5 more fast CPU
GBDT bases (baseline_two_anchor, m2_xgb, e5_optuna_lgbm, f1, f2);
EVERY base lifted (+2 to +19bp). Partial-pseudo M5q K=14 (6 pseudo
+ 8 original): **Strat OOF 0.95082, +2.54bp vs M5q anchor, ρ=0.99836
REAL_DELTA**. **FIRST non-null Day-5 meta-level result.** L1 reshuffled
away from `cb_slow-wide-bag` (1.06→0.30) and `a_horizon` (0.66→0.10)
toward pseudo-rebuilt HGBC/LGBM bases (e3 +116%, m2_xgb +130%).

## Day-6 plan — Phase 3 + slot 1 calibration probe

PI directive Day 5 close: merge to main, prepare handover, do "first 3"
of Phase 3 queue. Three rebuild streams in flight or queued:

### Phase 3A: CatBoost CPU rebuilds (sequential, ~1-2h CPU)
`scripts/d5_pseudo_phase3_catboost.py` rebuilds 4 CatBoost bases on
(train ∪ pseudo-test):
1. `cb_slow-wide-bag` — original L1=1.06 in M5q (top weight); CPU
   single-seed rebuild without GPU bagging is the fastest first probe.
2. `cb_lossguide` — L1=0.24 original; lossguide grow_policy.
3. `cb_year-cat` — L1=0.26 original; Year ∈ CAT_COLS.
4. `e1_catboost_sub` — L1=0.27 original; row-subsampled CB.

### Phase 3B: d2a_te TE-aware rebuild (~10min CPU)
`scripts/d5_pseudo_phase3_d2a.py`. Original d2a_te uses within-fold
target encoding. Pseudo handling: TE built from outer-train ONLY;
pseudo rows get TE values from the outer-train mappings (no leak).
L1=0.67 original → 0.37 in partial-pseudo K=14; rebuilding may
recover that drop.

### Phase 3C: RealMLP Kaggle GPU rebuild (~6h Kaggle T4x2 overnight)
`kernels/realmlp-pseudo-gpu/`. Pushed at session start; runs while
CPU rebuilds finish. RealMLP base in M5q gave the original +14bp LB
on M5h→M5q. Pseudo-rebuild expected to compound but variance is real.

### Slot 1 calibration probe (PI-approve)
`submissions/submission_d5_partial_pseudo_m5q.csv` (built Day 5).
Pre-submit-diff vs M5q expected ρ=0.99836 → REAL_DELTA. Plausible
LB outcomes: −2 to +5bp from M5q's 0.95005. The OOF→LB calibration
this gives is load-bearing for Phase 3 sequencing decisions.

### Sequencing recommendation
1. **PI approves slot 1 partial-pseudo K=14 submit** (single-shot, Rule 1).
2. **Push RealMLP pseudo-rebuild kernel** (instant; ~6h on Kaggle).
3. **Run Phase 3A + 3B sequentially** while #2 runs. ~1-2h total CPU.
4. **End-of-day**: rebuild K=14 with all phase-3 OOFs in pool;
   compare full-pseudo OOF vs partial-pseudo (0.95082) and M5q (0.95057).
5. **Slot 2 candidate** if full-pseudo lifts >+1bp OOF over partial.

## Falsified Day-5

- TabNet at default pytorch-tabnet config (n_d=32, cat_emb_dim=4,
  120 epochs) — fold-0 0.93532, FAIL gate. Under-trained, not
  under-priced. Re-test only after Path B's ceiling is mapped.
- Recursive GBDT (HGBC + M5q_oof_proba feature) at standalone +92bp
  baseline; null at 2-base, K=15 LR-stack, K=15 GBDT-meta. 3rd
  independent rank-lock confirmation.
- GBDT-meta over K=15 (recursive in pool) — all 3 variants worse
  than d4 K=14 by ~1bp uniformly. Meta divergence ceiling fixed.
- "Pool composition (drop e3 / drop f1+f2) un-locks the LR meta" — null.

## Calibration ladder snapshot (Day 6 morning)

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| baseline_two_anchor | 0.94075 | 0.94113 | LB-proxy ✓ gap +3.8bp |
| e3_hgbc | 0.94876 | 0.94870 | best single GBDT pre-CB |
| m5h | 0.95043 | 0.94991 | gap −5.2bp |
| **m5q (M5h + RealMLP, K=14)** | **0.95057** | **0.95005** | **PRIMARY**; +14bp; 10× LB amplification |
| m5_meta_lgbm_shallow (slot 2) | 0.95048 | 0.95001 | -4bp; meta-switch costs |
| d5_recursive_m5q (HGBC + M5q feat) | 0.94994 | n/a | std-alone +92bp; K=15 stacks NULL |
| d5_tabnet_smoke fold0 | 0.93532 | n/a | FAIL gate; parked |
| d5_e3_pseudo (Phase 1 MVP) | 0.94917 | n/a | +4.1bp anchor; ρ=0.99593 PASS |
| d5_baseline_pseudo | 0.94265 | n/a | +19.0bp anchor |
| d5_m2_xgb_pseudo | 0.94639 | n/a | +13.3bp anchor |
| d5_e5_optuna_lgbm_pseudo | 0.94792 | n/a | +5.6bp anchor |
| d5_f1_hgbc_deep_pseudo | 0.94914 | n/a | +4.4bp anchor |
| d5_f2_hgbc_shallow_pseudo | 0.94882 | n/a | +2.1bp anchor |
| **d5_partial_pseudo_m5q (K=14)** | **0.95082** | **(slot 1)** | **+2.54bp**; ρ=0.99836 REAL_DELTA |

## Held submissions

- **`submission_d5_partial_pseudo_m5q.csv`** — slot-1 candidate (REAL_DELTA)
- (carried forward from Day 5)
  - `submission_m5x_yetirank.csv` / `m5z_yetirank_nb.csv` — TIE_EXPECTED
  - `submission_m5_meta_lgbm_medium.csv` / `m5_meta_hgbc.csv` — meta variants
  - `submission_d5_meta_k15_*.csv` × 3 — K=15 GBDT-meta NULLs
  - `submission_m5_k15a/b/c.csv` — K=15 LR stack NULLs

## Critical operating rules (FRESHLY VIOLATED — READ THESE)

1. **Pre-submit-diff before EVERY submit.** Run
   `python3 scripts/pre_submit_diff.py <candidate.csv>`. ρ ≥ 0.999 → tie.
   ρ in [0.994, 0.999] → real LB delta possible.
2. **1-fold smoke before any GPU 5-fold.** Codified after the
   RealMLP 175-min run; reaffirmed Day 5 by TabNet smoke saving 5-fold cost.
3. **Strat-only Day-3+** (Rule R1). No GroupKF in new scripts.
4. **Don't drop bases purely on L1/diversity grounds.** Minimal-
   basis falsified Day-3.
5. **Bigger-moves rule: weight candidates by EV_bp / day_invested.**
   Sub-1bp tuning saved for final-window R5 probe.
6. **Strategy review before propose: grep audit/ for prior probes on
   the proposed mechanism.** External data already-tested-d2 friction.
7. **Pseudo gate union of M5q-confidence ([0.95,1] ∪ [0,0.05]) and
   multi-base vote (≥10/13 of M5h pool agree).** Day-5 yielded
   180k/188k rows, 0 conflicts, pos_rate 18.5% (real 19.9%). Reuse
   exactly across Phase 3 rebuilds.
8. **Pin SAME fold split across all rebuilds** (StratifiedKFold(5,
   shuffle=True, random_state=42)) — load-bearing for stack legitimacy.

## Pointers

- `audit/2026-05-06-d5-path-b-phase2.md` — Path B PASS (THIS THREAD)
- `audit/2026-05-06-d5-gbdt-meta-k15.md` — meta-add ceiling
- `audit/2026-05-06-d5-tabnet-smoke-fail.md` — TabNet parked
- `audit/2026-05-06-d5-path-c-recursive.md` — recursive + K=15 nulls
- `audit/2026-05-05-d4-gbdt-meta-breakthrough.md` — d4 slot-2 envelope
- `audit/friction.md` — logged failure modes
- `scripts/pre_submit_diff.py` — MANDATORY pre-submit gate
- `scripts/d5_pseudo_label_mvp.py` — Phase 1 MVP
- `scripts/d5_pseudo_phase2_rebuild.py` — Phase 2 driver (5 fast bases)
- `scripts/d5_recursive_m5q_gbdt.py` — Path C base build
- `scripts/d5_recursive_stack_k15.py` — K=15 LR sweep
- `scripts/d5_gbdt_meta_k15.py` — K=15 GBDT-meta sweep
- `kernels/tabnet-smoke-gpu/` — failed smoke (do not promote)
