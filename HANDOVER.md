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

### Move B — DONE Day-13. **V1 5/3 PASS_BOTH_GATES.**
Built 3 FM-class variants. Result table (Strat Δ / **GKF Δ** vs
leakage-blocked baseline 0.94776):
  - **V1 5/3 (D,C,S,T,Ln + R,Y,Rp): +0.06bp / +0.97bp PASS_BOTH_GATES**
    — ρ A vs B = 0.402 (d9f sweet spot); FM_5fA in L1 top-15.
  - V2 4/4 alt (C,T,S,Rp + D,R,Y,P): −0.36bp / +0.85bp PASS_GKF_ONLY
    — ρ A vs B = 0.186 (most-orthogonal pair to date).
  - V3 6/6 aug alt: −0.27bp / +0.54bp PASS_GKF_ONLY.

**Day-13 submit candidate: V1** at
`submissions/submission_d13_V1_5_3_K22_add.csv` (HELD per Rule 1).
Pred-LB Strat 0.95035 (TIE PRIMARY 0.95034); FM-class LB
amplification precedent (d9c +3bp / d9f +2bp / d9h +3bp on similar
OOF deltas) suggests realistic +0.5-3bp lift. PI decision per Rule
1; pre-submit-diff first (ρ=0.99963 vs PRIMARY).

V2 and V3 reserved as R5 final-3-day GKF-robust HEDGE candidates.

### Move C — pool refactor BLOCKED (no swap-class winners)
None of the 3 V variants beats PRIMARY on Strat OOF (V1 ties at
+0.06bp). Drop-and-replace can't proceed without a +1bp+ Strat
winner. **Pivoted to Move D + feature engineering instead.**

### Day-14 — Move D + E + F (second-tier; Move C falsified)
- **D: DeepFM-lite** (FM + 2-layer MLP head; d9h_aug12 fields). 4-6h CPU.
- **E: Regularised FFM** (k=4, L2=0.5, dropout); d9e overfit at 4× cap.
- **F: New FM-input features** (Move B falsified same-12-field reshuffle):
  pit-window-since-last-pit, hazard-decay, compound-pressure, race-stage.

## Falsified / dead — do NOT retry (Day-12 additions in **bold**)

- Big sequence (P1); kNN/retrieval/TabR/Hopular/TabPFN-ICL (P2)
- RealMLP bagging (Day-7); broad pseudo-labeling (Day-5)
- TabM-D smoke + extended 200ep lr=3e-4 (Day-9/11)
- T1.4 Hazard NN — Day-9/10 (230bp leak; leakfree zero signal)
- d9 10 math heuristic rule_residuals; d9b R14 K=20 swap+L4 TIE
- d9d FM sweep + 3-seed bag; d9e FFM overfit; d9g 3-way multi-FM −0.46bp
- **T1.2 multi-formulation 4-of-4** (Poisson Day-8 + censored/ratio/survival Day-12)
- **Year-segmented specialist** (cohort-split strips reg.; Year=2023 is EASIEST, P3 inverted)
- **Adversarial validation reweighting** (AV-AUC=0.502; train/test i.i.d.)
- **LambdaRank meta** (-86bp); **AUC-pairwise XGB base** (-451bp fold-0)
- **Single-bag e3 5seed** (-19bp; K=21 complexity JUSTIFIED)
- **External real-world priors on synth** — C1 TIE + C2 Pirelli DEPRECATED
- **Day-13 Move B V2/V3 K=22 add** Strat regress; held as GKF-only HEDGE only

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
- `audit/2026-05-13-d13-move-b-fm-variants.md` — Move B (V1 PASS_BOTH)
- `audit/2026-05-12-d12-tabpfn-finetune-prep.md` — Move A kernel
- `scripts/{d13_move_b_fm_variants.py, d12_groupkf_meta.py, pre_submit_diff.py}`
