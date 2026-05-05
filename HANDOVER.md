# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 12 (2026-05-12)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16 (Q7 still PROPOSED)
2. `audit/2026-05-09-d9c-fm.md` — load-bearing FM breakthrough (NEW PRIMARY)
3. `audit/2026-05-10-d9d-fm-sweep-bag.md` — FM sweep flat; pivot to FFM
4. `audit/2026-05-11-d11-strategy-critique.md` — revised pivot (FM-aware)
5. `audit/2026-05-11-d11-tabm-v3-extended-training-dead.md` — TabM closed
7. `audit/2026-05-08-data-probe-results.md` — P1-P10 priors
8. `scripts/pre_submit_diff.py` — MANDATORY before submit (ρ < 0.999)

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 12 morning)

- **NEW PRIMARY** = `d9f_K21_swap_partA_partB` LB **0.95031** (Day-10 multi-FM, +2bp over d9c).
- **HEDGE** = `d9c_K20_swap_FM` LB 0.95029 (held; demoted by d9f).
- **Prior PRIMARY** = `d6_k18_multi_rule` LB 0.95026.
- **Gap**: **−2.4bp** (narrowed from −2.6bp; from −3.9bp two days ago).
- **Headroom to top-5%** (0.95345): **31.4bp**.
- **15 days remaining** (deadline 2026-05-31). 9 slots/day.
- **Submits used**: 17/270 total; Day-11 0/10 (d9f executed Day-10).

## Day-11 close: TabM v3 dead, strategy reframed

TabM v3 extended training (n_epochs=200, patience=50, lr=3e-4) returned
fold-0 AUC **0.93926**, **−11.3bp worse than v2** (0.94039). Best val at
epoch 11; 189 epochs of monotone drift after. Patience never tripped.
**TabM-D dead at any plausible training schedule on this DGP.**

Strategy critique (Rule 14) — **REVISED** in light of d9c FM PASS. Of
12 submits since Day-6 K=18 PRIMARY: 11 NULL, 1 PASS (FM at +3bp). The
pattern is NOT "all bases tie"; it is "**bases inside K=18 hypothesis
class tie; structurally different model classes can lift**". FM's
low-rank pairwise interaction surface was the missing inductive bias
GBDTs/MLPs couldn't reach.

**Day-12 update (PI flag): C2 Pirelli DEPRECATED before execution.**
External real-world priors don't transfer to the CTGAN synth (Year=2023
mode-collapse + 4+-way joints broken + pool already absorbs the synth's
empirical analog). C1 SC-prob TIE is direct prior evidence of the same
failure mode. **Pivot direction: more model-class diversity (3-way
multi-FM partition) AND different inductive bias (G4 SCARF) — both
robust to synth artifacts. Skip new external info.**

## Day-12 first-action plan (synth-robust only; external info DEPRECATED)

### Path A — 3-way multi-FM partition (cheap; extends d9f +2bp win)
Disjoint 3-way split {D,C,S} + {R,Y} + {T_q5,Rp_q5,P_q5}; 3× FM k=8
6-ep, K=N stack as add or swap. ~10 min CPU. Pred +0.2-1bp.
Template: `scripts/d9f_multi_fm.py`.

### Path B — G4 SCARF on aadigupta1601 (overnight T4)
Contrastive pretraining on aadigupta's 101k unlabeled rows (NOT in
comp train) → fine-tune on labeled comp. Different inductive bias;
synth-robust (depends on aadigupta joint structure, not real-world).
6-10h T4. EV +1-4bp. Sole un-falsified GPU post-TabM closure.

### Path C — Bayesian hierarchical stacking (Yao 2021)
2-3h CPU PyMC+JAX. Per-segment (Compound, Stint, Year×Compound)
weights under Gaussian partial-pooling. Different META structure;
uses only existing pool OOFs. EV +1-3bp.

### Path D — Adversarial validation instance weights (HIGH FLOOR)
30 min CPU. p(test|x) density ratio as instance weight on K=N stack
rebuild. EV 0-2bp, no harm. Year=2023 mixture rescue.

### DEPRECATED Day-12 — C2 Pirelli + F2 Q6 rebuild
C2: synth-DGP incompatible (`audit/2026-05-12-d12-c2-pirelli-prep.md`
deprecation header). F2: 6× P10 confirmation (d9 10-cohort + d9b
ladder + C5 + C1) = dead-list category, not a probe.

### Path E — Calibration submits on held TIEs (1-2 day overlap)
3-4 of 253 unused submits on held candidates (m5x, m5z, d6_k15,
d8_k19_q12, d9_k19_sc_prob, d9_k20_neighbor, d9d_*) to recal
`pred_lb` at sub-0.99 ρ where uncertainty is ~30bp. Single-shot R1.

## Falsified / dead — do NOT retry (Day-11/12 additions in **bold**)

- Big sequence (P1); kNN/retrieval/TabR/Hopular/TabPFN-ICL (P2)
- RealMLP bagging (Day-7); broad pseudo-labeling (Day-5)
- F5 aux-meta / Move B 2-base / per-Race-per-Stint isotonic / reintroduce Normalized_TyreLife
- T1.5 Deotte L2/L3 (Day-8); T1.3 Q12 + T1.2 Poisson single-rule (Day-8)
- TabM-D smoke default (Day-9); **TabM-D extended 200ep lr=3e-4** (Day-11)
- C5 prev/next compound multi-rule (Day-9 K=20 TIE); C1 SC-prob lookup (Day-9 K=19 TIE)
- T1.4 Hazard NN — Day-9/10 (230bp leak)
- d9 10 math heuristic rule_residuals (cohort); d9b R14 hash-LR K=20 swap+L4 (TIE)
- d9d FM hparam sweep + 3-seed bag (flat); d9e FFM (overfit)
- **In-pool NN variants (whole class) 4×**; **single-rule rule_res on raw features 5×**
- **External real-world priors on synth** — C1 TIE + **C2 Pirelli pre-flight DEPRECATED Day-12**

## Held submissions (do NOT submit)

- BURNED: `d5_partial_pseudo_m5q.csv` (LB 0.94963), `d9_k19_hazard_nn_stack.csv` (LB 0.94711)
- TIE/DEAD held: `d7_realmlp_bag_part[BC]`, `d8_{l3_blend,k19_q12,k19_poisson}`,
  `d9_{hazard_nn,k20_neighbor,k19_sc_prob}`, `d9b_K20_swap_R14_L4` (LB 0.95025 TIE),
  `d10_hazard_nn_leakfree`, `d9d_*` FM sweep variants

## Calibration ladder snapshot (Day 12 morning)

| Mechanism | Strat OOF | LB | Gap | Notes |
|---|---:|---:|---:|---|
| **d9f_K21_swap_partA_partB (PRIMARY)** | **0.95073** | **0.95031** | **−2.4bp** | +2bp LB; multi-FM partition |
| d9c_K20_swap_FM (HEDGE) | 0.95070 | 0.95029 | −2.6bp | +3bp LB; 5.7× upside; demoted by d9f |
| d6_k18_multi_rule (great-grandparent) | 0.95065 | 0.95026 | −3.9bp | initial K=18 stack |
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | −5.2bp | grandparent |
| d9f_FM_A (D/C/S/T_q5) std | 0.82505 | n/a | n/a | ρ=0.487 vs PRIMARY (most diverse since R14) |
| d9f_FM_B (R/Y/Rp_q5/P_q5) std | 0.88438 | n/a | n/a | min-meta +0.04bp PASS |
| d9c FM std | 0.92069 | n/a | n/a | most-diverse single base since RealMLP |
| TabM v2 / **v3** fold-0 | 0.94039 / **0.93926** | n/a | n/a | gate FAIL; DEAD |
| d9_k19_hazard_nn LEAKY | 0.95446 | **0.94711** | **−73.5bp** | burned |
| d10 hazard leak-free | 0.92013 | n/a | n/a | DEAD (architecture zero-signal) |

## Critical operating rules

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **At ρ < 0.99**, downgrade pred-LB by additional 30bp until recalibrated.
3. **Rule 16 Q6** (ρ vs FULL pool, not just PRIMARY): binding gate; pre-flight.
4. **Rule 16 Q7** (target-construction leak): ×0.1 EV unless GroupKF OOF.
5. **Strat-only Day-3+** (R1) BUT GroupKFold load-bearing for within-group-future targets.
6. **Mechanism-class-only**: 11× confirmed nulls + 1× model-class break (FM).
   **Stop running NN arch variants in MLP+embed family + single-rule rule_residuals.**
7. **Submit budget** 17/270; spend 3-4 calibration probes Day-12-14.
8. **Model-class > tuning AND > external info on synth.** FM proved
   model-class diversity lifts; C2 deprecation confirms external
   real-world priors do not transfer to CTGAN synth.

## Pointers

- `audit/2026-05-09-d9c-fm.md`, `2026-05-10-d9d-fm-sweep-bag.md` — FM thread
- `audit/2026-05-11-d11-strategy-critique.md` — revised pivot
- `audit/2026-05-11-d11-tabm-v3-extended-training-dead.md` — TabM close
- `audit/2026-05-08-data-probe-results.md` — P1-P10 priors
- `scripts/{d9c_fm.py, d9c_kn_stack.py, d9d_fm_sweep_bag.py}` — FM templates
- `scripts/{d6_multi_rule.py, pre_submit_diff.py}` — F2 template + MANDATORY gate
