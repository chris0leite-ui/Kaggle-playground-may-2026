# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 14 (2026-05-14)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `scripts/probe.py` — entry point. `bote()` for BOTE, `gate()` for uniform gate report
3. `scripts/probe_min_meta.py` — K=21+N stack-add gate
4. `audit/2026-05-06-blend-and-rho-probes.md` — most recent rule-out + ρ inventory
5. `audit/2026-05-06-alpha-asymmetry-verification.md` — Path B α-asymmetry verified
6. `audit/2026-05-13-d13-path-b-hier-meta.md` — Path B mechanism (load-bearing)
7. `audit/2026-05-13-d13d-path-b-gkf-probe.md` — GKF amplification confirms private-robust
8. `audit/2026-05-12-d12-master-synthesis.md` — leakage-robust thesis
9. `scripts/pre_submit_diff.py` — MANDATORY before submit

Open with a 3-bullet read-back of state + first action.

**Harness usage cheatsheet (Rule 19):**
```bash
# BEFORE writing code for a candidate ≥10 min CPU:
python scripts/probe.py bote NAME --family X --cost_min N \
    [--std_oof_lift_bp Y] [--prob_useful U] [--note "rationale"]

# AFTER artifacts exist (under scripts/artifacts/oof_<NAME>_strat.npy):
python scripts/probe.py gate NAME \
    --oof scripts/artifacts/oof_NAME_strat.npy \
    --test scripts/artifacts/test_NAME_strat.npy

# Stack-add probe (K=21 + candidate(s)):
python scripts/probe_min_meta.py --candidates NAME1 NAME2 ...
```
Family priors are in `scripts/probe.py FAMILY_PRIORS`. Rule-out is
a valid result; cheap NULL findings get audit notes too.

## Where we are (Day 14 evening)

- **PRIMARY** = `d13e_compound_stint_tau20000` LB **0.95049** (+8bp Day-13 PM).
- **HEDGE** = `d13_path_b_stint_tau100000` LB 0.95041 (R5 candidate).
- **Gap to top-5%** (0.95345): **29.6bp**. 13 days remaining.
- **Submits used**: 24/270 total (Day-13 used 6/9, Day-14 used 0).

## 2026-05-06 PM addendum (branch `claude/ml-handover-alignment-xvUN0`)

PI redirect: experimentation culture; many small probes; BOTE-first;
"the solution is probably simple, maybe a code-quality fix".

**Built:** experimentation harness — `scripts/probe.py` (`bote` +
`gate`), `scripts/probe_min_meta.py` (K=21+N stack-add gate),
`scripts/probe_blends_K21.py`, `scripts/probe_rho_inventory.py`.
Rule 19 added to CLAUDE.md codifying BOTE-first / gate-after.

**Cheap probes (all via harness):**

1. **α-asymmetry verification.** OOF uses fold-train counts in α=n/(n+τ);
   test uses full-train counts. Bayesian-correct shrinkage, NOT a fixable LB
   cap. **PURSUE**: α-calibrated τ-resweep (~30 min).

2. **K=21 simple-blend probe.** mean/gmean/rank_mean/trimmed all regress
   19–32bp standalone vs PRIMARY. LR-meta-stays-best CONFIRMED.

3. **ρ inventory of 22 held candidates.** Best near-tie HEDGE: **`d12_lr_meta`**
   (OOF 0.95073, ρ=0.996, flip ratio 0.297).

4. **K=21 + d6_rule_compound_stint min-meta.** Δ −0.020bp NULL (already absorbed).

5. **K=21 + 3 (`d12_lr_meta` + `d10d_leak_corrected_meta` + `blend_rank_mean_K21`).**
   **Δ +1.298bp OOF** (0.95073 → 0.95086). `d12_lr_meta` dominates. **First non-NULL.**

**Open candidates (NOT YET RUN, BOTE-graded):**
- α-calibrated τ-resweep on PRIMARY hier-meta (PURSUE; ~30 min).
- `d12_lr_meta` single-candidate ablation (was in flight at session-end).
- Within-Race quantile-rank of LapTime_Delta as FM input (DEFER; H5 z-score leak fix needed).
- Per-Driver historical pit rate smoothed EB (DEFER; ~10 min).
- Year×Stint sparse-LR / FM partition (DEFER; ~30 min).

## Day-14 session — TabPFN + Move D results

### Move A — TabPFN fine-tune: DEAD

- **v2.5 @ 150k rows** (kernel v10): fold-0 AUC **0.94446** — identical to 50k-row result
  (0.94439). Training loss flat from epoch 1; fine-tuning not learning. Wall 6829s, no gain.
- **v2.6 @ any row count**: OOM at epoch 1 (model weights ≈15.37GB, P100 = 16GB). Dead.
- **Verdict**: TabPFN ceiling ~0.944 (-64bp vs PRIMARY). ρ=0.960 diverse but gap too large.
  **Both versions dead-listed.**

### Move D — FM new inputs (F1-F4): DEAD

`scripts/d13_move_f_fm_aug16.py` 16-field FM (12 d9h + 4 new: PitWindow/HazardDecay/
CompoundPressure/RaceStage). Standalone +20.1bp (0.92741 vs aug12 0.92540), ρ=0.919.
Min-meta: **-0.07bp FAIL**. Confirms FM-field-augmentation saturated at 12 fields.

## Remaining live moves (Day 15)

### PURSUE: α-calibrated τ-resweep (~30 min CPU)
τ chosen on OOF may not be τ-optimal for test (fold-train vs full-train counts differ).
Re-sweep τ ∈ {5k, 10k, 20k, 50k, 100k, 200k} on d13e and d13b Stint under corrected
α formula. EV +0-3bp. Cheap — run via harness first.

### PURSUE: d12_lr_meta single-candidate stack-add
K=21 + d12_lr_meta alone. If Δ > 0bp OOF → HEDGE candidate. Cost ~5 min.

### Move B — Pseudo-label cascade at K=21+hier-meta level (~3-4h CPU)
EV +5-10bp. Use d13e PRIMARY preds, top-30% confidence filter, retrain 5 fastest
bases, re-stack K=21+hier-meta. Risk: d5 widened gap on m5q (-4.2bp LB).

### Move C — DeepFM-lite (~3-4h CPU)
FM pairwise + 2-layer MLP head. New model class. EV +3-8bp standalone, +1-3bp stacked.
Risk: overfit (d9e FFM precedent). Mitigation: dropout + batch-norm + depth=2.

### Research loop trigger (Rule 7)
If no ≥+5bp structural move found: pause submits, web-search top-5 finisher writeups
from comparable playground comps, identify untried mechanism families.

## Falsified / dead — do NOT retry

All prior entries remain. Additional dead from Day-14:
- **TabPFN v2.5 fine-tune** — AUC ceiling 0.9444 regardless of row count
- **TabPFN v2.6 fine-tune** — OOM on P100 at any row count
- **FM-field-augmentation** (Move D / d14 aug13 / aug16) — saturated at 12 fields
- d14 Path B cohort sweep (Year/Year×Stint/Race × τ) — all NULL vs Compound×Stint PRIMARY

## Critical operating rules

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **BOTE before code** (Rule 19). Cost-gate: expected OOF lift × prob_useful > 0.1bp.
3. **NEW Day-13: ρ/G3/R7 heuristics DO NOT apply to new mechanism families.**
4. **Rule 16 Q6** (ρ vs FULL pool, not just PRIMARY): binding gate.
5. **GroupKF as secondary gate**. Strat AND not-regress-GKF, or pass GKF directly.
6. **Submit budget** 24/270; ~115 remaining. 40bp gap → structural moves only.
7. **Model-class diversification > tuning** (FM + hier-meta both confirmed).

## Pointers

- `audit/2026-05-13-d13-{path-b-hier-meta,d13d-path-b-gkf-probe}.md` — load-bearing
- `audit/2026-05-06-{blend-and-rho-probes,alpha-asymmetry-verification}.md` — Day-14 probes
- `scripts/d13_move_f_features.py` + `scripts/d13_move_f_fm_aug16.py` — Move D
- `scripts/artifacts/d13_move_f_fm_aug16_results.json` + `d12_tabpfn_finetune_150k_results.json`
- `kernels/d12-tabpfn-finetune-gpu/` (v2.5) + `kernels/d13-tabpfn-v26-strat/` (v2.6) — archived

---

## Day-15 PM read-handover-LgbQ4 (deep-dive 4-branch + GPU revival + submit)

**Result: NEW PRIMARY LB 0.95059** (+1.0bp over d13e 0.95049). Submission `d15b_path_b_K22_dae_only_tau20000` (52394353 COMPLETE 2026-05-06 15:38). Gap to top-5%: 28.6bp. Day-15 used 1/9 slots; total 26/270.

**Branches run** (4-parallel + GPU revival):
- **A** `d15a_alpha_tau_resweep` (`code_fix_calibration`) — **FALSIFIED**. ρ=1.000000 vs d13e at τ=20000; α-fix is no-op (segments ≥1000 rows have α≈1 in both regimes). PI's "code-quality fix" intuition wrong on this lever.
- **B-CPU** `d15b_dae_smoke` — KILLED, DAE-on-CPU non-feasible (smoke 30 min on 70k×782 incomplete).
- **B-GPU v1** `d15b-dae-lgbm-gpu` — ERROR (P100 sm_60 fallback, friction `kaggle-p100-fallback-reproduced-day15`).
- **B-GPU v2** torch 2.4 force-reinstall fix — **SUCCESS**. DAE 256-512-256 swap-noise frac=0.15 on (train+test 627k) 20 epochs batch=4096; 768-d latent (h2+h3 concat). LGBM-on-latent-only std OOF 0.94007, ρ_test 0.9477 (most-diverse since FM_A_53). Min-meta +0.793bp at ρ 0.99547. K=22 Path B Compound×Stint τ=20000 OOF 0.95090 (+0.715bp), ρ=0.99973, flips 59/53 R7-eligible. Submit landed +1.0bp LB.
- **C** `d15c_extra_trees` (4000 trees max_features=sqrt) — borderline. std OOF 0.92967, min-meta +0.059bp at ρ 0.99599. **R5 HEDGE only.**
- **D** `d15d_lgbm_knn` (k=5 NN per-Compound + per-Driver, 10 features) — borderline. std OOF 0.94166, min-meta +0.056bp at ρ 0.99586. C+D K=23 add additive +0.095bp; ρ between C/D raw 0.9325 but LR-meta routes both to ρ≈0.996. **R5 HEDGE only.**

**Load-bearing finding**: friction `path-b-amp-only-fires-on-meta-arch-not-base-add`. Realised LB amp 1.4× on +0.715bp OOF base-add — well below Path-B-amp 6-11.6× on meta-arch redesigns (d13 Compound 6.7×, d13e 8×, d13 Stint 11.6×). Refines the prior `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`: even genuinely orthogonal base-add (DAE ρ 0.9477 standalone) does NOT fire amp; only meta-arch redesign (segmentation refinement) does. Cross-confirmed by main-branch agent same day: K=22 + orig_transfer hier-meta τ=20k LB 0.95049 (TIE +1.127bp OOF).

**Cross-branch awareness** (do NOT touch — note only):
- `claude/ml-handover-alignment-xvUN0` has unsubmitted candidate `path_b_K22_invlaps τ=20k` OOF 0.95110 (+2.75bp), claimed largest-of-session. Untested at LB; under my new friction tag would predict ~+4bp LB at 1.4× realised amp.
- Multiple branches also active: knowledge-base-setup-LIxXm, eda-deep-dive-qFByN, etc. Scribe will consolidate next morning.

**Artifacts persisted (all committed for later use)**:
- `scripts/d15a_alpha_tau_resweep.py` + 7 OOF/test pairs at τ ∈ {2000,5000,10000,20000,50000,100000,200000}
- `scripts/d15b_dae_{smoke,encoder,lgbm_on_dae}.py` (CPU killed) + `kernels/d15b-dae-gpu/` (v2 working, 1.4GB latents .gitignored)
- `scripts/d15b_path_b_K22_dae_only.py` + 3 τ-sweep OOF/test pairs
- `oof_d15b_lgbm_dae_{full,only}_strat.npy` + `test_*` (re-usable in any K_pool+1 probe)
- `scripts/d15c_extra_trees.py` + `oof_d15c_extra_trees_strat.npy` + test
- `scripts/d15d_{knn_features,lgbm_on_knn}.py` + `d15d_knn_X_{train,test}.npy` (KNN distance feature matrices) + LGBM OOF/test
- `submissions/submission_d15b_path_b_K22_dae_only_tau20000.csv`
- 6 `probe_min_meta__d15*.json` gate reports
- `scripts/probe.py` extended with 3 new families (`dae_unsupervised`, `extra_trees_ensemble`, `knn_distance_features`)

**Next steps (Day-16 priority)**:
1. **PI decision needed**: do we submit `claude/ml-handover-alignment-xvUN0`'s K=22+inv_laps τ=20k? +2.75bp OOF projects ~+4bp LB at 1.4× realised amp = LB ~0.95093. Stronger candidate than today's d15b. Slot cost 1/9.
2. **Day-16 axis pivot**: meta-architecture redesign (amp-eligible per friction `path-b-amp-only-fires-on-meta-arch-not-base-add`). Concrete candidates:
   - **Non-Gaussian shrinkage** on Path B (Beta-Binomial / Student-t). Replaces Gaussian-τ; same amp axis as d13e Compound×Stint (8×).
   - **Yao/Vehtari covariance-modelled BMA** (LKJ prior on inter-base Σ, GP prior on segment index). Proper "Path B done correctly". 4h CPU or 30 min Kaggle GPU PyMC-JAX.
   - **Alternative segmentation cross**: Year×Compound (4×5=20 seg), Compound×TyreLife_q5 (5×5=25 seg), Driver_clustered×Stint (4 driver clusters × 5 = 20 seg).
3. **DAE pool composition**: `d15b_lgbm_dae_full` had similar lift profile to base-add (no amp). Both d15b artifacts useful for any future K_pool+N probe — no need to retrain.
4. **R5 HEDGE ladder accumulating**: d15b_path_b_K22_dae_only_tau{20000,100000} (PRIMARY + close-second), d15c (ExtraTrees), d15d (LGBM-on-KNN), d15c+d15d K=23.
5. **Skill amendment** (already in friction): start any new torch GPU kernel from `kernels/hazard-nn-smoke-gpu/` boilerplate (sm_60 force-reinstall pre-wired). Don't dispatch general-purpose subagents for python jobs >5 min wall.

**Pointers added today**:
- `audit/2026-05-15-d15-4branch-results.md` — full 4-branch + B-GPU + K=22 Path B + submit-result audit
- `scripts/d15{a,b,c,d}_*.py` — branch implementations
- `kernels/d15b-dae-gpu/` — Kaggle GPU kernel (v2 with sm_60 fix)
- `submissions/submission_d15b_path_b_K22_dae_only_tau20000.csv` — submitted artifact
