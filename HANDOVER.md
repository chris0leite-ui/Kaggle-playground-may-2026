# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 9 (2026-05-09)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16 (Rule 16 NEW Day-8: 5-question pre-flight)
2. `audit/2026-05-09-d9c-fm.md` — Factorization Machine PASSES min-meta; K=20 swap+FM CANDIDATE
3. `audit/2026-05-09-d9b-r14-ladder.md` — R14 ladder; K=20 swap+L4 SUBMITTED LB 0.95025 TIE
4. `audit/2026-05-09-d9-math-heuristics.md` — d9 10-approach cohort all FAIL min-meta
5. `audit/2026-05-08-data-probe-results.md` — load-bearing probes (P1, P5, P10)
6. `audit/2026-05-07-d6-f1-2-LB-result.md` — F1.2 PRIMARY landed +2.1bp
7. `scripts/pre_submit_diff.py` — MANDATORY before submit. ρ threshold 0.9995.

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 10 morning)

- **Day 10, 0/10 used today.** Day-9 used 1/10 (d9b L4 SUBMITTED, LB 0.95025 TIE).
- **PRIMARY** = `d6_k18_multi_rule` LB **0.95026** (M5q + 4 rule_residuals). Strat OOF 0.95065. Gap −3.9bp.
- **CANDIDATE HELD** = `d9c_K20_swap_FM` (drop 2 redundant rules + R6/R10/R7 + FM). Strat OOF **0.95070** (+0.53bp), ρ=0.99973, predicted LB Δ **+0.53bp**. ABOVE +0.5bp slot threshold.
- **Headroom to top-5%** (0.95345): **31.9bp**.
- 21 days remaining (deadline 2026-05-31). 9 slots/day available.

## Day-9 d9 cohort — 10 simple math/heuristic rule_residuals, all FAIL min-meta

| Approach | Std OOF | ρ vs PRIMARY | Min-meta Δ | Verdict |
|---|---:|---:|---:|---|
| R5 weibull_compound | 0.94600 | 0.943 | −0.09 | FAIL |
| R6 next_compound | 0.94443 | 0.908 | −0.12 | FAIL |
| R7 prev_compound | 0.94481 | 0.914 | −0.10 | FAIL |
| R8 position_progress | 0.94554 | 0.931 | −0.11 | FAIL |
| R9 laptime_delta_z | 0.94558 | 0.942 | −0.09 | FAIL |
| R10 driver_eb (Beta-Binom) | 0.94463 | 0.912 | −0.10 | FAIL |
| R11 stint_overdue | 0.94557 | 0.925 | −0.09 | FAIL |
| R12 cumdeg_knee | 0.94535 | 0.934 | −0.09 | FAIL |
| R13 race_lapbin | 0.94539 | 0.925 | −0.12 | FAIL |
| **R14 hash_lr_3way** | **0.79377** | **0.444** | **−0.02** | FAIL by hair (most-diverse) |

Reading: PRIMARY's 4-rule cohort has saturated the rule_residual
mechanism. New rule_residuals are diverse but informationally
redundant. **5th independent confirmation of P10**: lift requires
NEW SIGNALS or NEW MODEL CLASS, not better extraction.

R14 is the lone structural outlier — different model class (sparse
LR over hashed Driver × Compound × Stint), ρ=0.444, *almost* passes.

## Day-9b R14 strength ladder — L2/L3/L4 PASS, L4 swap SUBMITTED LB 0.95025 TIE

| Level | Std OOF | ρ vs PRIMARY | Min-meta Δ | Verdict |
|---|---:|---:|---:|---|
| L0 (baseline R14) | 0.79368 | 0.444 | −0.02 | FAIL |
| L1 (+Race × Year) | 0.89200 | 0.845 | −0.06 | FAIL (worse) |
| L2 (+binned numerics) | 0.91449 | 0.874 | +0.01 | **PASS ✓** |
| L3 (+ Compound × num) | 0.91626 | 0.875 | +0.01 | **PASS ✓** (best std) |
| L4 (+ Driver × num) | 0.91369 | 0.869 | +0.01 | **PASS ✓** |
| L5 (kitchen sink) | 0.90852 | 0.854 | −0.10 | FAIL (overfit) |

K=20 swap+L4 (drop 2 most-redundant rules, add R6+R10+R7+R14_L4)
SUBMITTED at predicted +0.19bp → **actual LB 0.95025 (−0.01bp TIE)**.

Reading: K=N rearrangement of the rule_residual + sparse-LR family is
quantization-bounded. The +0.19bp OOF prediction did not transfer.

## Day-9c Factorization Machine — PASSES at +0.18bp, K=20 swap pred +0.53bp

| Quantity | FM (d9c) | R14_L3 (d9b best) | R14_L0 (d9 R14) |
|---|---:|---:|---:|
| Std OOF | **0.92069** | 0.91626 | 0.79377 |
| ρ vs PRIMARY | **0.89858** | 0.87487 | 0.44358 |
| Min-meta Δ | **+0.18bp** | +0.01bp | −0.02bp |
| Verdict | **PASS ✓ (18× R14_L3 lift)** | PASS ✓ | FAIL |
| Wall (5-fold) | 56s | ~22 min | ~75s |

FM uses the same 8 main-effect features as R14_L2 (no hand-engineered
interactions); learns cross-feature pairwise interactions in a
k=8-dim low-rank space. Strictly dominates the R14 ladder on both
strength AND diversity simultaneously.

K=N stack experiments:

| Stack | K | Δ PRIMARY OOF | ρ vs PRIMARY | pred-LB Δ |
|---|---:|---:|---:|---:|
| Sa K=21 (R6+R10+R7+R14_L4+FM) | 21 | +0.41 | 0.99973 | +0.41 |
| Sb K=18 (R7+FM) | 18 | +0.43 | 0.99968 | +0.43 |
| Sc K=17 (FM solo) | 17 | +0.37 | 0.99977 | +0.37 |
| **Sd K=20 swap (R6+R10+R7+FM, no R14)** | **20** | **+0.53** | **0.99973** | **+0.53** |

Sa < Sd: R14_L4 and FM occupy the same model-class slot — including
both is double-counting; FM dominates.

## Day-10 first-action plan

### If submitting today:

1. **HOLD candidate**: `submission_d9c_K20_swap_FM.csv` (Sd K=20).
   Predicted +0.53bp; ρ vs PRIMARY 0.99973 (passes 0.9995 gate).
   Ask PI for single-shot sign-off (Rule 1).

### Cheap pre-submit refinements (run BEFORE submit if PI delays):

1. **FM hyperparameter sweep** — embed_dim ∈ {4, 8, 16}, weight decay
   ∈ {0, 1e-6, 1e-5}. ~10 min CPU. Could push std OOF from 0.921 to
   ~0.925 → Sd lift to +0.6–0.8bp.
2. **FM 3-seed bag with rank-average** — variance reduction. ~3 min
   CPU. Typical +0.1–0.3bp OOF.
3. **FM with longer epochs + early-stop on val-AUC** — current loss
   curves plateaued by epoch 4; could squeeze another 0.5bp.

### CPU queue (post-d9c)

| # | Move | Cost | EV (bp) | Notes |
|---|---|---|---:|---|
| C1 | Pirelli pit-window scrape (Tier-2 ext data) | 6-8h scrape + 2h CPU | 3-12 | new info NOT in pool; highest absolute EV |
| C2 | EmbMLP on CPU (PyTorch nn.Embedding for Driver) | 4-6h CPU (8-core) | 1-3 | distinct from FM/RealMLP |
| C3 | FM with field-aware variant (FFM) | 2h CPU | 0.5-2 | refinement of FM, may push OOF |
| C4 | External-data Q3 SC probability per track | 2-3h CPU + 1h scrape | 2-5 | rule_residual base; race-marginal |
| C5 | Hierarchical Bayesian stacking (Yao 2021) | 2-3h CPU (PyMC+JAX) | 1-6 | different META structure |

### GPU queue (when GPU returns)

| # | Move | Cost | EV (bp) | Notes |
|---|---|---|---:|---|
| G1 | T1.1 TabM 1-fold smoke (Rule 2) | 1-2h T4 | gate test | NOT YET FALSIFIED; sole unbroken Tier-1 |
| G2 | T1.4 Hazard-rate NN, careful build | 4-6h T4 | 1-7 | structurally different; note d9 hazard_nn_stack regressed 315bp — implementation matters |
| G3 | T2.6 SCARF pretrain on aadigupta1601 | 6-10h T4 | 1-6 | different unlabeled corpus |

## Critical operating rules (reaffirmed Day-9)

1. **Pre-submit-diff before EVERY submit**, ρ < 0.9995.
2. **Mechanism-class-only**: pool tweaks AND meta-only changes AND
   single-rule residuals on raw features AND duplicate
   reformulations AND single-base bag rebuilds AND **all 9 d9 rule_-
   residual variants** all dead vs PRIMARY.
3. **Predicted-gap gate**: pred-gap < −7bp needs PI sign-off.
4. **Minimal-input-meta sanity check** is sharp: 0.02bp made the
   difference between R14_L0 (FAIL) and R14_L2 (PASS).
5. **Strat-only Day-3+** (R1; U3 confirmed i.i.d.).
6. **Track gap direction** — F1.2 narrowed gap −5.2 → −3.9bp; d9b
   tied at LB 0.95025 (essentially −0.01bp from PRIMARY).
7. **Rule 16 pre-flight (5 questions)** — applied retroactively, the
   d9 cohort scored 0/5 on Q5 (clones of existing rule_residual).
   FM scored 5/5 (truly new model class) — and lifted accordingly.

## Falsified / dead — do NOT retry

- Big sequence models (P1 Day-8).
- kNN / retrieval / TabR / Hopular (P2 Day-8).
- TabPFN-2.5 ICL ensemble (regime mismatch).
- RealMLP bagging (Day-7 partial-bag NULL).
- Broad pseudo-labeling (Day-5 partial-pseudo K=14).
- F5 aux-feature GBDT-meta (Day-6).
- Move B 2-base [M5q, recursive] (Day-6).
- Per-Race / per-Stint isotonic (Day-3 in-CV regress).
- Reintroduce `Normalized_TyreLife` (host-removed).
- T1.5 Deotte L2 stacking (Day-8).
- T1.3 Q12 single-rule rule_residual (Day-8).
- T1.2 Poisson laps-until-next-pit (Day-8).
- **d9 simple-math rule_residual cohort (9 of 10 variants)** —
  rule_residual mechanism saturated within PRIMARY's 4-rule cohort.
- **d9 hazard_nn_stack** (parallel-agent submission at LB 0.94711,
  −315bp regression) — the *concept* of hazard NN remains alive but
  this particular kernel imploded; debug before retry.

## Held submissions (do NOT blindly submit)

- **Day-7 NULL salvage**: `d7_realmlp_bag_partB.csv`, `d7_realmlp_bag_partC.csv`
- **Day-8 holds**: `d8_l3_blend.csv`, `d8_k19_q12.csv`, `d8_k19_poisson.csv`
- **Day-9c HOLDS** (NEW): `submission_d9c_K20_swap_FM.csv` (Sd, pred +0.53bp,
  CANDIDATE pending PI sign-off), `submission_d9c_K17_FM_solo.csv` (Sc).
- **Burned**: `d5_partial_pseudo_m5q` (−4.2bp), `d9b_k20_swap_l4` (−0.01bp TIE).
- **Carry-forward TIE/NULL**: `m5x_yetirank`, `m5z_yetirank_nb`,
  `m5_meta_lgbm_*`, `m5_meta_hgbc`, `d5_meta_k15_*`, `m5_k15a/b/c`.

## Calibration ladder snapshot (Day 10 morning)

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | gap −5.2bp |
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | **gap −3.9bp** |
| d9b_k20_swap_l4 | 0.95067 | 0.95025 | TIE −0.01bp; pred +0.19bp; quantization-bounded |
| d9c_FM (Factorization Machine) | 0.92069 | n/a | std-alone; ρ=0.899 most-diverse base since RealMLP |
| **d9c_Sd_K20_swap_FM (CANDIDATE)** | **0.95070** | n/a | **pred LB +0.53bp; HELD pending PI** |

## Pointers

- `audit/2026-05-09-d9-math-heuristics.md` — d9 cohort min-meta saturation
- `audit/2026-05-09-d9b-r14-ladder.md` — d9b R14 ladder + K=20 swap LB tie
- `audit/2026-05-09-d9c-fm.md` — d9c FM PASS + Sd K=20 swap CANDIDATE
- `audit/2026-05-08-d8-cpu-probes-falsified.md` — Day-8 CPU audit
- `audit/2026-05-08-data-probe-results.md` — P1-P10 probes
- `audit/2026-05-07-d6-f1-2-LB-result.md` — F1.2 K=18 LB win
- `scripts/d9_math_heuristics.py`, `scripts/d9_kn_stack.py`
- `scripts/d9b_r14_ladder.py`, `scripts/d9b_kn_stack.py`
- `scripts/d9c_fm.py`, `scripts/d9c_kn_stack.py`
- `scripts/pre_submit_diff.py` — MANDATORY (ρ < 0.9995)
