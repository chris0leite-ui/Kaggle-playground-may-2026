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
GBDTs/MLPs couldn't reach. **Pivot direction: more model-class
diversity (FFM, multi-FM partitions) AND new info (C2 Pirelli).**

## Day-12 first-action plan

### Path A — C2 Pirelli pit-windows (highest absolute EV; promoted Day-11)

FFM falsified Day-10 (`audit/2026-05-10-d9e-ffm.md`); FM hparam
neighborhood flat (`d9d`); FM bagging stacks worse (d9d). With all
in-class FM follow-ups dead, **C2 is the clear top-priority CPU
move.** **Do NOT defer past Day-13.** 6-8h scrape (F1.com strategy
guides + Pirelli press kits, 26 races × 4 years) + 2h CPU build of
4-6 rule_residual bases following F1.2 template + Q6 filter.
EV +3-8bp median. Only un-explored move bringing genuinely new info.

### Path B — Multi-FM partition diversity → STRONG CANDIDATE (d9f, EXECUTED Day-10)

`audit/2026-05-10-d9f-multi-fm.md`. FM_A (driver-dynamics: D,C,S,T_q5)
+ FM_B (race-context: R,Y,Rp_q5,P_q5), each k=8 6-ep. ρ FM_A vs FM_B =
**0.406** (~orthogonal). FM_A ρ vs PRIMARY = **0.487** (most-diverse
single base since Day-9 R14). K=22 add: OOF **+0.30bp**, ρ=0.99968,
pred-LB **+0.32bp**. **Awaiting PI sign-off for LB submit.** Apply
+30bp pred-LB downgrade per Rule 2 (sub-0.99 → 0.9997 ρ band still
fragile).

### Path D — F2 multi-rule rebuild w/ Q6 (cheap falsification)

Predicted-NULL by d9 10-cohort precedent (5th confirmation of P10).
Tier-3; only run if A/B return TIE. ~3h CPU.

### Background — G4 SCARF on aadigupta1601 (Day-13 overnight T4)

6-10h T4. Different inductive bias (contrastive pretraining); only
un-falsified GPU candidate post-TabM closure.

### Path E — Calibration submits on held TIEs (1-2 day overlap)

3-4 of 256 unused submits on held candidates (m5x, m5z, d6_k15,
d8_k19_q12, d9_k19_sc_prob, d9_k20_neighbor) to recalibrate `pred_lb`
at sub-0.99 ρ where current uncertainty is ~30bp. Single-shot per R1.

## Falsified / dead — do NOT retry

(Day-11 additions in **bold**)
- Big sequence models — P1
- kNN / retrieval / TabR / Hopular — P2
- TabPFN-2.5 ICL ensemble — P2
- RealMLP bagging — Day-7
- Broad pseudo-labeling — Day-5
- F5 aux-feature GBDT-meta — Day-6
- Move B 2-base [M5q, recursive] — Day-6
- Per-Race / per-Stint isotonic — Day-3
- Reintroduce `Normalized_TyreLife` — host-removed
- T1.5 Deotte L2 stacking — Day-8
- T1.3 Q12 / T1.2 Poisson single-rule — Day-8
- TabM-D smoke (default) — Day-9; **TabM-D extended (200ep, lr=3e-4)** — Day-11
- C5 prev/next compound multi-rule — Day-9 (K=20 TIE)
- C1 SC-prob curated lookup — Day-9 (K=19 TIE)
- T1.4 Hazard-rate NN — Day-9/10 (230bp leak)
- d9 10 math heuristic rule_residuals — Day-9 (cohort)
- d9b R14 hash-LR ladder K=20 swap+L4 — Day-9 (LB 0.95025 TIE)
- d9d FM hparam sweep + 3-seed bag — Day-10 (flat hparam neighborhood)
- d9e FFM (field-aware FM, 6ep + 2ep) — Day-10 (overfit; std OOF below FM)
- **In-pool hypothesis-class NN variants (whole class)** — 4× confirmed
- **Single-rule rule_residuals on raw features (whole class)** — 5× confirmed

## Held submissions (do NOT submit)

- BURNED: `d5_partial_pseudo_m5q.csv` (LB 0.94963), `d9_k19_hazard_nn_stack.csv` (LB 0.94711)
- TIE/DEAD held: `d7_realmlp_bag_part[BC]`, `d8_{l3_blend,k19_q12,k19_poisson}`,
  `d9_{hazard_nn,k20_neighbor,k19_sc_prob}`, `d9b_K20_swap_R14_L4` (LB 0.95025 TIE),
  `d10_hazard_nn_leakfree`, `d9d_*` FM sweep variants

## Calibration ladder snapshot (Day 12 morning)

| Mechanism | Strat OOF | LB | Gap | Notes |
|---|---:|---:|---:|---|
| d6_k18_multi_rule (great-parent) | 0.95065 | 0.95026 | −3.9bp | initial K=18 stack |
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
| d9f K=22 add (FM_A+FM_B+FM_d9c) | 0.95073 | n/a | n/a | tied with K=21; d9c FM demoted out of L1 top-15 |
| d6_k18_multi_rule (great-parent) | 0.95065 | 0.95026 | −3.9bp | original PRIMARY before d9c jump |
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | −5.2bp | grandparent |
| TabM v2 fold-0 | 0.94039 | n/a | n/a | gate FAIL |
| **TabM v3 fold-0** (extended) | **0.93926** | n/a | n/a | **−11bp from v2; DEAD** |
| d9d_bag3_seeds (FM) | 0.92253 | n/a | n/a | std OOF +1.84bp; stack TIE; bag HURTS stack |
| d10 hazard leak-free | 0.92013 | n/a | n/a | DEAD |

## Critical operating rules

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **At ρ < 0.99**, downgrade pred-LB by additional 30bp until recalibrated.
3. **Rule 16 Q6** (ρ vs FULL pool, not just PRIMARY): binding gate; pre-flight.
4. **Rule 16 Q7** (target-construction leak): ×0.1 EV unless GroupKF OOF.
5. **Strat-only Day-3+** (R1) BUT GroupKFold load-bearing for within-group-future targets.
6. **Mechanism-class-only**: 11× confirmed nulls + 1× model-class break (FM).
   **Stop running NN arch variants in MLP+embed family + single-rule rule_residuals.**
7. **Submit budget** 16/270; spend 3-4 calibration probes Day-12-14.
8. **Information > bases AND model-class > tuning.** FM proved both:
   structurally different model class with no new info still lifted.

## Pointers

- `audit/2026-05-09-d9c-fm.md`, `2026-05-10-d9d-fm-sweep-bag.md` — FM thread
- `audit/2026-05-11-d11-strategy-critique.md` — revised pivot
- `audit/2026-05-11-d11-tabm-v3-extended-training-dead.md` — TabM close
- `audit/2026-05-08-data-probe-results.md` — P1-P10 priors
- `scripts/{d9c_fm.py, d9c_kn_stack.py, d9d_fm_sweep_bag.py}` — FM templates
- `scripts/{d6_multi_rule.py, pre_submit_diff.py}` — F2 template + MANDATORY gate
