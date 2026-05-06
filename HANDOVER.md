# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 14 (2026-05-14)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16
2. `audit/2026-05-13-d13-path-b-hier-meta.md` — Path B mechanism (load-bearing)
3. `audit/2026-05-13-d13d-path-b-gkf-probe.md` — GKF amplification confirms private-robust
4. `audit/2026-05-12-d12-master-synthesis.md` — Day-12 leakage-robust thesis
5. `audit/2026-05-12-d12-tabpfn-finetune-prep.md` — TabPFN GPU kernel READY
6. `scripts/pre_submit_diff.py` — MANDATORY before submit

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 14 morning)

- **PRIMARY** = `d13_path_b_stint_tau100000` LB **0.95041** (NEW Day-13).
- **HEDGE** = `d9h_K22_add_aug12` / `d9i_S1_K21_swap_aug2way` (TIED) LB 0.95034.
- **Gap to top-5%** (0.95345): **30.4bp** (narrowed from 31.1bp).
- **13 days remaining** (deadline 2026-05-31). 9 slots/day.
- **Submits used**: 21/270 total; Day-13 2/9 (d13c LB 0.95033, d13 Stint LB 0.95041).

## Day-13 close: Path B hier-meta is a NEW MODEL CLASS

`d13_path_b_stint_tau100000` is the load-bearing finding. Per-segment
empirical-Bayes hierarchical LR meta over the K=21 PRIMARY pool. Stint
segmentation (5 levels), shrinkage τ=100000.

| | Pre-submit | Result |
|---|---|---|
| OOF Δ vs d9f | +0.86bp | (built-in) |
| ρ vs d9f | 0.998 (sub-tie) | (built-in) |
| G3 flip ratio | 0.211 (FAIL) | benign — aligned with public LB |
| **LB** | predicted ~−1bp | **+10bp vs d9f, +7bp vs d9h/d9i** |
| **Upside** | n/a | **11.6× OOF→LB** |

**d13d GKF probe confirms mechanism is leakage-robust:** Strat lift
+0.90bp → GKF lift +2.59bp = **2.9× AMPLIFIED** (stronger than the
FM-class 2.3× amplification in d10b/c). Three independent leak-blocking
probes converge: public-LB +7bp transfers to private at median +4-6bp,
conservative +2-3bp, bull +6-8bp.

**Insight**: hierarchical-meta heuristics are NOT the same as base-class
heuristics. G3/ρ/R7 thresholds derived from d9c/d9f/d9h all failed for
d13 Stint. New mechanism family → recompute the GKF probe BEFORE
applying prior gates. (See `audit/friction.md` tag
`pred-lb-heuristics-broken-for-hier-meta`.)

## Day-14 first-action plan (40bp gap, structural moves only)

PI directive Day-13 evening: **"we want to improve by 40bp not 2."**
+1-3bp tuning probes are off the table for the comp middle. Each
candidate below is a structural model-class lift.

### Move A — push TabPFN-2.5 fine-tune kernel to Kaggle (PI action)
Kernel ready at `kernels/d12-tabpfn-finetune-gpu/`. Set `TABPFN_TOKEN`
secret (license accept on ux.priorlabs.ai). T4×2 wall 5-7h. **Only
live single-shot ≥10bp candidate.** EV +5-15bp std-alone, +1-5bp
stack tail. Single PI action; runs in background. 

### Move B — pseudo-label cascade at K=21+hier-meta level (CPU)
~3-4h CPU. d5 partial pseudo got +14bp standalone on m5q but only
+2.5bp at K=14 stack — tighter cascade at the new K=21 ceiling could
compound:
1. Generate test pseudo-labels from d13 Stint τ=100000 PRIMARY (LB 0.95041)
2. Round 1: confidence-filter (top 30% by predicted prob spread); retrain 5 fastest bases on (train + filtered pseudo)
3. Re-stack K=21 with pseudo-trained bases + hier-meta
4. Round 2 if Round 1 ρ < 0.999 vs PRIMARY

EV +5-10bp incremental. Risk: pseudo over-amp (d5 widened gap on m5q).

### Move C — DeepFM-lite (CPU)
~3-4h CPU. FM low-rank pairwise + 2-layer MLP head over the 21-base
expanded space. Fundamentally new base class:
- FM: ⟨v_i, v_j⟩ pairwise interaction surface (current d9c/d9f)
- + MLP head: f(W₂ · σ(W₁ · concat(v_i)) + b) for higher-order interactions

EV +3-8bp standalone, +1-3bp stacked. Risk: 2-layer MLP may overfit
without dropout (d9e FFM precedent — 4× params died).

### Move D — Compound × Stint hier-meta SUBMIT (held variants)
HELD. Two candidates per `audit/2026-05-13-d13-path-b-hier-meta.md`:
- `d13e_compound_stint_tau20000.csv` (+1.00bp OOF, ρ=0.996, projects ~+2bp LB)
- `d13e_compound_stint_tau100000.csv` (+0.82bp OOF, ρ=0.9996 vs Stint winner; HEDGE-grade)

ONLY submit if Day-14 doesn't produce a structural Move A/B/C winner.

### Research-loop trigger (Rule 7) IF Day-14 yields no ≥+5bp move
Pause submits. Web-search top-5 finishers' writeups from comparable
playground tabular comps. Identify untried mechanism families. Honest
read: **40bp gap may not be tractable from current architecture**.
Top finishers likely have a structural insight (target reformulation /
unique FE / external data) we haven't found.

## Falsified / dead — do NOT retry

- Big-sequence (P1); kNN/TabR/Hopular/TabPFN-ICL (P2)
- RealMLP bagging; broad pseudo-labeling; TabM-D extended 200ep
- Hazard NN (230bp leak; leakfree zero signal)
- d9 10 math heuristic rule_residuals; d9b R14 K=20 swap+L4 (TIE)
- d9d FM sweep+bag; d9e FFM (overfit); d9g 3-way; d9h K=22 add (TIE)
- T1.2 multi-formulation 4-of-4 (Poisson/censored/ratio/survival)
- Year-specialist; AV reweighting (i.i.d.); LambdaRank meta (-86bp)
- AUC-pairwise XGB (-451bp); external real-world priors on synth
- **d10d leak-corrected meta** (G3 flip ratio 0.001; over-credits FM)

## Held submissions (do NOT submit)

- BURNED: `d5_partial_pseudo_m5q.csv` (LB 0.94963), `d9_k19_hazard_nn_stack` (LB 0.94711)
- TIE/DEAD held (full list in CLAUDE.md): d7-d10 variants, d12 t12/year/AV/lambda
  bases, **d12_groupkf_meta** (HEDGE-eligible R5), **d10d** (G3 fail),
  **d13b Stint τ=20000**, **d13c Compound τ=100000** (LB 0.95033 result),
  **d13e Compound×Stint** 4 τ-variants (only submit if no Move A/B/C win)

## Critical operating rules

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **NEW Day-13: ρ/G3/R7 heuristics DO NOT apply to new mechanism
   families.** Confirm via GKF probe before assuming a sub-tie ρ
   candidate will under-perform.
3. **Rule 16 Q6** (ρ vs FULL pool, not just PRIMARY): binding gate.
4. **GroupKF as secondary gate** (Day-12 finding). For new bases:
   pass Strat AND not regress GKF, OR pass GKF if leakage-robust class.
5. **Strat-only Day-3+ (R1)** for primary OOF. Public LB stays the
   truth (U3: i.i.d. row split). GKF is the private-LB proxy probe.
6. **Submit budget** 21/270; 13 days × 9 = 117 remaining. **40bp gap
   means structural moves only**; tuning probes deferred to R5.
7. **Model-class diversification > tuning** (d9c FM, d13 hier-meta both
   confirmed). New base class > new τ on existing meta.

## Pointers

- `audit/2026-05-13-d13-path-b-hier-meta.md` + `d13d-path-b-gkf-probe.md`
- `audit/2026-05-12-d12-master-synthesis.md` + `d12-tabpfn-finetune-prep.md`
- `audit/2026-05-10-d10b-groupkf-stack-rebuild.md` + `d10d-leak-corrected-meta.md`
- `audit/friction.md` — Day-13 frictions appended
- `scripts/d13{,b,c,d,e}_path_b_*.py` — Path B family
- `kernels/d12-tabpfn-finetune-gpu/` — Move A kernel
