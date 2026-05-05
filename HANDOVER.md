# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 10 (2026-05-11)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16 (Rule 16 Q7 PROPOSED below; needs PI go to land in CLAUDE.md)
2. `audit/2026-05-09-d9-hazard-nn-LB-postmortem.md` — load-bearing failure mode discovery
3. `audit/2026-05-10-d10-hazard-nn-leakfree-confirmed-dead.md` — diagnostic close-out
4. `audit/2026-05-09-d9-three-nulls-tabm-c5-c1.md` — Day-9 main-branch CPU/GPU sweep
5. `audit/2026-05-09-d9c-fm.md` — **NEW: d9c parallel-branch FM CANDIDATE submitted today**
6. `audit/2026-05-09-d9b-r14-ladder.md` — d9b R14 ladder; K=20 swap+L4 burned LB 0.95025 TIE
7. `audit/2026-05-09-d9-math-heuristics.md` — d9 10-approach cohort all FAIL min-meta
8. `audit/2026-05-08-data-probe-results.md` — load-bearing P1-P10 (P6 explains the leak mechanism)
9. `scripts/pre_submit_diff.py` — MANDATORY before submit (ρ < 0.999)

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 10 morning)

- **NEW PRIMARY** = `d9f_K21_swap_partA_partB` LB **0.95031** (+2bp from d9c).
- **Prior PRIMARY** = `d9c_K20_swap_FM` LB 0.95029 (held as hedge).
- **Gap NARROWED**: −2.6bp → **−2.4bp**.
- **Headroom to top-5%** (0.95345): **31.4bp**. To leader (0.95435): 40.4bp.
- **20 days remaining** (deadline 2026-05-31). 9 slots/day after submit reset.
- **Submits used today**: **3/10** (d9b L4 TIE + d9c FM +3bp + d9f multi-FM +2bp).
- **Total comp: 17.**

## Day-9 main-branch — load-bearing hazard-NN failure mode discovery

Submitted `submission_d9_k19_hazard_nn_stack.csv` (M5q + 4 d6 rules +
hazard_nn_bag, K=19). Strat OOF 0.95446 (+38.1bp vs K=18). Predicted LB
0.95367. **Actual LB 0.94711. Gap −73.5bp** (vs calibrated −5bp).

**Root cause**: hazard target's `bfill` propagates val-row PitNextLap
backward through every earlier row in the same stint group. Per P6,
80% of consecutive-lap pairs cross fold boundaries → 80% of tr rows'
hazard buckets directly encode val labels. Hazard NN's K=20 buckets
per row carry 20× the per-row leak signal vs scalar reformulations.

**Day-10 leak-free re-run** (stint-drop within Strat fold): leak-free
seed-42 OOF **0.92013** vs leaky 0.94319. **Leak magnitude: 230bp.**
Hazard NN architecture itself adds **zero usable signal** beyond
the binary pool. **Hazard NN is DEAD; do not retry.**

## Day-9 parallel-branch (d9 / d9b / d9c) — math heuristics + R14 ladder + FM

Independent thread on `claude/math-heuristics-ml-62fpM`, merged today.

**d9 — 10 simple math/heuristic rule_residual variants**

| Approach | Std OOF | ρ vs PRIMARY | Min-meta Δ | Verdict |
|---|---:|---:|---:|---|
| R5 weibull_compound | 0.94600 | 0.943 | −0.09 | FAIL |
| R6 next_compound | 0.94443 | 0.908 | −0.12 | FAIL |
| R7 prev_compound | 0.94481 | 0.914 | −0.10 | FAIL |
| R8 position_progress | 0.94554 | 0.931 | −0.11 | FAIL |
| R9 laptime_delta_z | 0.94558 | 0.942 | −0.09 | FAIL |
| R10 driver_eb | 0.94463 | 0.912 | −0.10 | FAIL |
| R11 stint_overdue | 0.94557 | 0.925 | −0.09 | FAIL |
| R12 cumdeg_knee | 0.94535 | 0.934 | −0.09 | FAIL |
| R13 race_lapbin | 0.94539 | 0.925 | −0.12 | FAIL |
| R14 hash_lr_3way | 0.79377 | 0.444 | −0.02 | FAIL (most diverse, miss by hair) |

5th independent confirmation of P10: rule_residual mechanism is
saturated within PRIMARY's 4-rule cohort.

**d9b — R14 hash-LR strength ladder**

L0 → L5 ladder; L2/L3/L4 PASS at +0.01bp; L1 (+Race/Year) and L5
(kitchen sink) FAIL. K=20 swap+L4 SUBMITTED → **LB 0.95025 TIE −0.01bp**
(pred +0.19bp, quantization-bounded). Same-mechanism rearrangement
of rule_residual + sparse-LR doesn't transfer.

**d9c — Factorization Machine baseline**

PyTorch CPU FM, k=8 embeddings, 8 main-effect features (D, C, R, Y,
S + TyreLife/RaceProgress/Position quintiles), 6 epochs SparseAdam.
56s wall total (5 folds).

| Quantity | FM | R14_L3 (best ladder) | R14_L0 |
|---|---:|---:|---:|
| Std OOF | **0.92069** | 0.91626 | 0.79377 |
| ρ vs PRIMARY | **0.89858** | 0.87487 | 0.44358 |
| Min-meta Δ | **+0.18bp** | +0.01bp | −0.02bp |
| Verdict | **PASS ✓ (18× R14_L3 lift)** | PASS | FAIL |

FM strictly dominates R14 on BOTH strength AND diversity. The auto-
learned low-rank pairwise interactions share statistical strength
across all feature pairs vs R14's per-bucket weights. **No leak risk**
(binary PitNextLap target, no within-group future-label propagation).

K=N stack experiments:

| Stack | K | Δ PRIMARY OOF | ρ | pred-LB Δ |
|---|---:|---:|---:|---:|
| Sa K=21 (R6+R10+R7+R14_L4+FM) | 21 | +0.41 | 0.99973 | +0.41 |
| Sb K=18 (R7+FM) | 18 | +0.43 | 0.99968 | +0.43 |
| Sc K=17 (FM solo) | 17 | +0.37 | 0.99977 | +0.37 |
| **Sd K=20 swap (R6+R10+R7+FM, no R14)** | **20** | **+0.53** | **0.99973** | **+0.53** |

Sa < Sd: FM and R14_L4 occupy the same model-class slot. FM
dominates; including R14 is double-counting.

**Sd SUBMITTED at 18:56 UTC** as `submission_d9c_K20_swap_FM.csv`.
Predicted +0.53bp; **actual LB 0.95029 (+3.0bp), 5.7× upside.** NEW
PRIMARY. Gap narrowed −3.9 → −2.6bp. FM is the first genuinely new
model class to land LB lift since RealMLP joined M5q in Day-3.

## Pool audit finding (from main hazard-NN postmortem, still valid)

`b_lapsuntilpit` (in K=18) and `a_horizon` (in K=18) use the SAME
bfill mechanism. b_lapsuntilpit groups by `(Race, Driver)` — broader
than hazard's `(Race, Driver, Year, Stint)`. Both carry within-group
label leakage but at much smaller magnitude (their L1 weights 0.57 /
0.67 in the K=18 LR-meta — diluted by 14 leak-free binary
classifiers). Hazard NN entered K=19 with **L1=9.52**, dominating the
meta and exposing its 230bp leak.

K=18's calibrated −3.9bp gap is robust because all bases share
similar leak signatures and L1 weights are anchored by the binary
classifiers. **Don't disturb the K=18 pool composition** unless we
have a clean replacement. FM in d9c is **leak-free** (binary target,
8 main-effect features, no group-future propagation).

## Proposed Rule 16 Q7 (needs PI go for CLAUDE.md)

> **Q7 — Target-construction leak test.** If a candidate base's
> target/feature construction depends on FUTURE same-group labels
> (e.g., `bfill`, `next_pit_lap`, horizon-shifted target with group
> propagation), apply ×0.1 EV downgrade UNLESS its OOF was computed
> under GroupKFold by the relevant group. Apply PRE-FLIGHT, not
> post-submit. Catches the failure mode that pre_submit_diff,
> min-meta, and Q6 all missed (all are sensitive to predictions, not
> targets).

## Day-10 / Day-11 first-action plan

### Path A0 — d9c K=20 swap+FM LANDED at LB 0.95029 (+3bp)

Sd is **NEW PRIMARY**. 5.7× upside on +0.53bp prediction. Confirms FM
model class transfers to LB. Next:

1. **FM bagging** — 3 seeds rank-averaged; predicted +0.1–0.5bp.
2. **FM hyperparameter sweep** — embed_dim k ∈ {4, 8, 16}, weight
   decay ∈ {0, 1e-6, 1e-5}, epoch ∈ {4, 6, 10}. Find best single FM.
3. **Field-aware FM (FFM)** — different field-pair embedding tables;
   typically +0.5–1bp std OOF over plain FM.
4. **Multi-FM diversity** — train 2 FMs with different feature
   partitions or different k; stack both. Tests whether the FM
   model-class diversity bonus extends with more FM bases.

### Path A — TabM v3 with extended training (top-priority main-branch)

Day-9 TabM smoke v2 stopped at val cross-entropy −0.4248 at epoch 5,
then 20 epochs of oscillation. Possibly converged to a local basin;
possibly under-trained at the margin.

**Build**: copy `kernels/tabm-smoke-gpu/` → `kernels/tabm-smoke-v3-gpu/`
with explicit longer training:
- introspect `pytabkit.TabM_D_Classifier.__init__` for `n_epochs` /
  `patience` / `lr` knobs; pass max-epochs ≥200, patience ≥50
- if pytabkit doesn't expose those, fall back to standalone `tabm`
  package (simpler API)
- ~10 min on T4

Same gates as smoke v2: fold-0 AUC ≥ 0.945 PROMOTE; else HOLD.

**No leak risk** (binary PitNextLap target). If it clears the gate,
this is the cleanest next-base candidate. EV +1-7bp at K=19.

### Path B — G4 SCARF/VIME pretrain on aadigupta1601 (heavy)

6-10h T4. Different unlabeled corpus avoids d5 partial-pseudo amp.
Different inductive bias (contrastive pretraining). No leak risk.

Defer until Path A resolves.

### Path C — FM refinements (cheap, parallel)

If d9c Sd lifts, the next-step FM moves are:
1. **FM bagged across 3 seeds (rank-average)** — ~3 min; +0.1–0.3bp.
2. **FM hyperparameter sweep** — embed_dim ∈ {4, 8, 16}, weight
   decay ∈ {0, 1e-6, 1e-5}. ~10 min CPU.
3. **Field-aware FM (FFM)** — different field-pair embedding tables.
   ~2h CPU.

If d9c Sd doesn't lift, FM model-class hypothesis is bounded; pivot
back to TabM v3 / external data.

### Path D — F2 multi-rule rebuild with Q6 enforcement (cheap CPU)

After today's confirmation that K=18 pool is at the LR-meta info
ceiling (8 nulls + d9 cohort), the only way to add a base via
rule_residual is to find a rule whose ρ-orthogonality is preserved
AGAINST EVERY existing pool member, not just M5q. Cheap experiment
but predicted NULL by analogy with C5/C1.

## Falsified / dead — do NOT retry

(keep this list growing)
- **Big sequence models** — P1
- **kNN / retrieval / TabR / Hopular** — P2
- **TabPFN-2.5 ICL ensemble** — P2 regime issue
- **RealMLP bagging** — Day-7
- **Broad pseudo-labeling** — Day-5
- **F5 aux-feature GBDT-meta** — Day-6
- **Move B 2-base [M5q, recursive]** — Day-6
- **Per-Race / per-Stint isotonic** — Day-3
- **Reintroduce `Normalized_TyreLife`** — host-removed
- **T1.5 Deotte L2 stacking** — Day-8
- **T1.3 Q12 single-rule rule_residual** — Day-8
- **T1.2 Poisson laps-until-pit** — Day-8
- **TabM-D smoke (default config)** — Day-9
- **C5 prev/next compound multi-rule** — Day-9 (K=20 TIE, main-branch)
- **C1 SC-prob curated lookup** — Day-9 (K=19 TIE, main-branch)
- **T1.4 Hazard-rate NN** — Day-9 LB −31.5bp / Day-10 leak-free OOF 0.92013
- **d9 parallel-branch 9 rule_residual variants (R5–R13)** — all min-meta FAIL
- **d9b R14 strength ladder K=20 swap+L4** — LB 0.95025 TIE
- **d9b R14 single-base** (any L0–L5 alone, no K=N stack) — quantization-bounded

## Held submissions (do NOT submit)

- `submission_d9_k19_hazard_nn_stack.csv` — burned at LB 0.94711
- `submission_d9_hazard_nn.csv` (single-base hazard) — predict LB ~0.926
- `submission_d10_hazard_nn_leakfree.csv` — predicted LB ~0.916
- Day-9 `submission_d9_k20_neighbor.csv`, `submission_d9_k19_sc_prob.csv` — TIE-locked
- Day-7 `d7_realmlp_bag_partB.csv`, `d7_realmlp_bag_partC.csv` — TIE-locked
- Day-8 `d8_l3_blend.csv`, `d8_k19_q12.csv`, `d8_k19_poisson.csv`
- Day-5 `d5_partial_pseudo_m5q.csv` — burned at LB 0.94963
- Day-9 `submission_d9b_k20_swap_l4.csv` — burned at LB 0.95025 TIE
- Day-9 `submission_d9c_K17_FM_solo.csv` — held (Sc K=17, pred +0.37bp)

## Calibration ladder snapshot (Day 10 evening)

| Mechanism | Strat OOF | LB | Gap | Notes |
|---|---:|---:|---:|---|
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | **−3.9bp** | UNCHANGED |
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | −5.2bp | parent |
| d9_k19_hazard_nn (LEAKY) | 0.95446 | **0.94711** | **−73.5bp** | burned (main) |
| d9_k19_sc_prob | 0.95065 | n/a | n/a | TIE held (main) |
| d9_k20_neighbor | 0.95065 | n/a | n/a | TIE held (main) |
| Day-10 hazard leak-free | 0.92013 | n/a | n/a | DEAD (main) |
| d9b_k20_swap_l4 | 0.95067 | 0.95025 | TIE | burned -0.01bp; pred +0.19bp |
| d9c_FM (Factorization Machine) | 0.92069 | n/a | n/a | most-diverse since RealMLP; passes min-meta +0.18bp |
| d9c_Sd_K20_swap_FM (hedge) | 0.95070 | 0.95029 | −2.6bp | +3bp LB lift; 5.7× upside; demoted by d9f |
| d9f_FM_A (D/C/S/T_q5) | 0.82505 | n/a | n/a | most-diverse single base since R14 (ρ=0.487 vs PRIMARY) |
| d9f_FM_B (R/Y/Rp_q5/P_q5) | 0.88438 | n/a | n/a | min-meta +0.04bp PASS |
| **d9f_K21_swap_partA_partB (NEW PRIMARY)** | **0.95073** | **0.95031** | **−2.4bp** | **+2bp LB lift; 6.25× upside on +0.32bp pred; multi-FM partition replaces unified FM** |

## Critical operating rules (re-emphasised)

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **Predicted-LB calibration is fragile in low-ρ regime.** The
   `pred_lb` function's offset table was calibrated on ρ ≥ 0.99
   submits; at ρ ≈ 0.96 the offset extrapolation was 20× too small.
   **At ρ < 0.99, downgrade pred-LB by an additional 30bp** until we
   have more sub-0.99 submits to recalibrate.
3. **Rule 16 Q7 (PROPOSED): target-construction leak test.** Apply
   ×0.1 EV downgrade pre-flight unless GroupKF OOF available.
4. **Strat-only Day-3+** (R1) BUT: GroupKFold is now load-bearing
   for any base whose target depends on within-group future labels.
5. **Mechanism-class-only**: confirmed Nth time. Pool tweaks AND
   meta-only changes AND single-rule residuals on raw features AND
   duplicate reformulations AND single-base bag rebuilds AND
   leak-saturated NN bases are ALL dead. **FM (d9c) is the first
   genuinely new model class to PASS min-meta since RealMLP.**
6. **NEVER-GIVE-UP / saturation-is-bounded**. Today's failures are
   data points about WHICH new bases work, not that no new bases
   work. TabM (binary target) is unfalsified. FM is alive pending LB.

## Pointers

- `audit/2026-05-09-d9-hazard-nn-LB-postmortem.md`
- `audit/2026-05-10-d10-hazard-nn-leakfree-confirmed-dead.md`
- `audit/2026-05-09-d9-three-nulls-tabm-c5-c1.md`
- `audit/2026-05-09-d9-c5-neighbor-rules-marginal.md`
- `audit/2026-05-09-d9-math-heuristics.md` (parallel-branch d9)
- `audit/2026-05-09-d9b-r14-ladder.md` (parallel-branch d9b)
- `audit/2026-05-09-d9c-fm.md` (parallel-branch d9c — FM CANDIDATE)
- `audit/2026-05-08-data-probe-results.md` (P6 = leak mechanism)
- `audit/2026-05-08-strategic-menu-wider-steps.md` (apply Q7 overlay)
- `kernels/hazard-nn-leakfree-gpu/hazard_nn_leakfree.py` (stint-drop pattern)
- `kernels/tabm-smoke-gpu/tabm_smoke_gpu.py` (Day-9 v2; copy for Path A)
- `scripts/d9c_fm.py`, `scripts/d9c_kn_stack.py` (FM builder + stack)
- `scripts/pre_submit_diff.py` (MANDATORY)
