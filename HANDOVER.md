# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 13 (2026-05-13)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16 (Q7 still PROPOSED)
2. `audit/2026-05-12-d12-master-synthesis.md` — Day-12 6-option overnight; load-bearing
3. `audit/2026-05-12-d12-groupkf-rebuild.md` — Option 1 STRUCTURAL FINDING
4. `audit/2026-05-12-d12-tabpfn-finetune-prep.md` — Option 2 Kaggle GPU kernel ready
5. `audit/2026-05-08-data-probe-results.md` — P1-P10 priors (P3 inverted by Option 4)
6. `scripts/pre_submit_diff.py` — MANDATORY before submit (ρ < 0.999)

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 13 morning)

- **PRIMARY** = `d9h_K22_add_aug12` / `d9i_S1_K21_swap_aug2way` (TIED) LB **0.95034**.
- **HEDGE candidates**: `d9f_K21_swap_partA_partB` LB 0.95031; `d12_groupkf_meta` (held; ρ=0.9914 vs PRIMARY — most diverse meta-output to date; reserve for R5).
- **Gap**: −1.7bp (narrowed from −2.4 on Day-11).
- **Headroom to top-5%** (0.95345): **31.1bp**.
- **14 days remaining** (deadline 2026-05-31). 9 slots/day available.
- **Submits used**: 19/270 total; Day-12 0/9 (subagent night, no submits per Rule 1).

## Day-12 close: 6 wider-step options, 1 structural finding

Per `audit/2026-05-12-d12-master-synthesis.md`. Of 6 parallel
subagents: 1 structural (Option 1), 1 kernel-ready (Option 2), 4
falsified (Options 3/4/5/9 — see `mechanism_families_explored`).

**THE finding (Option 1):** under GroupKFold(Race,Driver,Year,Stint),
K=21 LR-meta produces test predictions with ρ=**0.9914** vs Strat-meta
(below the 0.999 RHO_TIE threshold). Bases split into 2 populations:
- 13 GBDTs drop **−200 to −343bp** under GKF (leakage eaters)
- 7 FM/rule/sparse-LR drop only **−9 to −43bp** (leakage robust)
- **FM is 23–37× more leakage-robust than every GBDT.**

L1 reshuffles dramatically: cb_slow-wide-bag −17 ranks,
e5_optuna_lgbm −13; **FM jumps +15 (#20→#5)**, R6_next_compound +11.

**Reframe:** rank-lock at ρ=0.9999 is substantially a Strat-leakage
artifact. Pool's diversification frontier lives entirely WITHIN the
leakage-robust population.

## Day-13 first-action plan (parallel: A + B; then C)

### Move A — re-push TabPFN-2.5 fine-tune kernel (PI provides token THIS session)
**Status: BLOCKED on Day-12; PI confirmed will provide TABPFN_TOKEN
this session.** Day-12 attempted push (`kaggle kernels push`) errored
fast at the license check because no `TABPFN_TOKEN` Kaggle Secret
was set. Kernel is at `kernels/d12-tabpfn-finetune-gpu/` — verified
correct (handles token via `kaggle_secrets.UserSecretsClient` per
script lines 117-129). **First action this session:**
  1. PI provides token from https://ux.priorlabs.ai (one-time license accept).
  2. Set Kaggle Secret named `TABPFN_TOKEN` on PI's Kaggle account
     (web UI → Settings → Add-ons → Secrets, or via `kaggle config secret`).
  3. Re-push: `cd kernels/d12-tabpfn-finetune-gpu && kaggle kernels push`.
  4. Verify status: `kaggle kernels status chrisleitescha/d12-tabpfn-finetune-strat`.
  5. T4×2 wall 5-7h. **Only live 10bp shot.** EV +5-15bp std-alone,
     +1-3bp stack median, +3-9bp tail. Runs in background.
  6. On completion: `kaggle kernels output chrisleitescha/d12-tabpfn-finetune-strat -p kernels/d12-tabpfn-finetune-gpu/output/`.

### Move B — build 3 FM-class diversification variants (cheap CPU)
Each 2-3h CPU. Builds on d9f's 2-way 4/4 sweet spot finding:
1. **5/3 multi-FM partition** — 5 driver-state + 3 race-context fields
2. **4/4 alternative split** — Compound × TyreLife axis with new
   partner field (variant of d9i's aug-2way)
3. **Augmented 2-way (different fields)** — pick top-12 augmented
   fields from d9h_aug12 by L1, split into 2 partitions

Each candidate: standalone OOF, ρ vs PRIMARY, min-meta gate, K=22
add stack. **Use both Strat AND GroupKF as gates** (first time GKF
is a secondary gate — Day-12 finding mandates this).

### Move C — pool refactor (after Move B lands ≥1 winner)
Drop 3 most-leakage-eating GBDTs (e5_optuna_lgbm, cb_slow-wide-bag,
e1_cb_sub — all ΔAUC ≤ −215bp under GKF) and replace with Move B
winners. Build K=21 stack with swapped pool. Submit only if OOF
Strat ≥ PRIMARY AND ρ < 0.9995.

### Day-14 — Move D + E (second-tier)
- **D: DeepFM-lite** (FM + 2-layer MLP head). 4-6h CPU.
- **E: Regularised FFM re-attempt** (k=4, L2=0.5, dropout-on-fields).
  d9e FFM died from overfit at 4× FM params; reduced capacity may
  earn slot.

## Falsified / dead — do NOT retry (Day-12 additions in **bold**)

- Big sequence (P1); kNN/retrieval/TabR/Hopular/TabPFN-ICL (P2)
- RealMLP bagging (Day-7); broad pseudo-labeling (Day-5)
- TabM-D smoke + extended 200ep lr=3e-4 (Day-9/11)
- T1.4 Hazard NN — Day-9/10 (230bp leak; leakfree zero signal)
- d9 10 math heuristic rule_residuals; d9b R14 K=20 swap+L4 (TIE)
- d9d FM hparam sweep + 3-seed bag; d9e FFM (overfit at default cap);
  d9g 3-way multi-FM (REGRESSION −0.46bp); d9h K=22 add (TIE_EXPECTED
  but LB +3bp from FM-class amplification)
- **T1.2 multi-formulation 4-of-4** (Day-8 Poisson + Day-12
  censored/ratio/survival)
- **Year-segmented specialist** (cohort-split strips cross-Year reg.;
  Year=2023 is EASIEST segment, P3 inverted)
- **Adversarial validation reweighting** (AV-AUC=0.502; train/test
  i.i.d.; no shift to exploit)
- **LambdaRank meta** (-86bp); **AUC-pairwise XGB base** (-451bp
  fold-0); LR-meta on [raw,rank,logit] retains seat
- **Single-bag e3 5seed** (-19bp; K=21 complexity JUSTIFIED, not
  OOF-noise overfit)
- **External real-world priors on synth** — C1 TIE + C2 Pirelli
  DEPRECATED Day-12

## Held submissions (do NOT submit)

- BURNED: `d5_partial_pseudo_m5q.csv` (LB 0.94963), `d9_k19_hazard_nn_stack.csv` (LB 0.94711)
- TIE/DEAD held: `d7_realmlp_bag_part[BC]`, `d8_{l3_blend,k19_q12,k19_poisson}`,
  `d9_{hazard_nn,k20_neighbor,k19_sc_prob}`, `d9b_K20_swap_R14_L4` (LB 0.95025 TIE),
  `d10_hazard_nn_leakfree`, `d9d_*` FM sweep variants, **d12 t12c/d/e bases**,
  **d12 year_specialist / advweight bases**, **d12 lambdarank_meta**,
  **d12 e3/cb single bags**, **d12_groupkf_meta** (HEDGE for R5 only)

## Critical operating rules

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **At ρ < 0.99**, downgrade pred-LB by additional 30bp until recalibrated.
3. **Rule 16 Q6** (ρ vs FULL pool, not just PRIMARY): binding gate; pre-flight.
4. **Rule 16 Q7** (target-construction leak): ×0.1 EV unless GroupKF OOF.
5. **NEW Day-12: GroupKF as secondary gate.** New candidate must
   either pass Strat AND not regress GroupKF, OR pass GroupKF if it
   is a leakage-robust class (FM/rule/sparse-LR).
6. **Strat-only Day-3+ (R1)** — but GroupKF is now a secondary gate,
   not a primary. Public LB stays the truth (U3: i.i.d. row split).
7. **Submit budget** 19/270; 14 days × 9 slots = 126 remaining. Spend
   3-4 calibration probes Day-13-15 on held TIEs at sub-0.99 ρ.
8. **Model-class > tuning AND > external info on synth.** FM proved
   model-class diversity lifts; Day-12 confirms within-class
   diversification is the live lever.

## Pointers

- `audit/2026-05-12-d12-master-synthesis.md` — Day-12 unifying frame
- `audit/2026-05-12-d12-groupkf-rebuild.md` — Option 1 detail
- `audit/2026-05-12-d12-tabpfn-finetune-prep.md` — Move A specifics
- `audit/2026-05-09-d9c-fm.md`, `audit/2026-05-10-d9f-multi-fm.md` — FM thread
- `scripts/{d9c_fm.py, d9c_kn_stack.py, d9f_multi_fm.py}` — Move B templates
- `kernels/d12-tabpfn-finetune-gpu/` — Move A kernel
- `scripts/{d12_groupkf_meta.py, d6_multi_rule.py, pre_submit_diff.py}`
