# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are

- **PRIMARY** = `d13e_compound_stint_tau20000` LB **0.95049** (Day-13 PM advance).
- **TOP SUBMISSION CANDIDATE (HELD):** `path_b_K22_invlaps_tau20000.csv`
  - OOF **0.95110** (+2.75 bp); largest non-meta-derivative single-add OOF advance ever.
  - **Target-derived** signal (LGBM regression on `1/(1+laps_until_pit)`); orthogonal-signal criterion satisfied.
  - ρ vs PRIMARY 0.99753; flips 57/96 (ratio 0.594).
  - Path B family amp prior (8-11.6×) → predicted LB band **+1.25 to +32 bp**.
- **HEDGE candidates held:** `path_b_K22_invlaps_tau100000.csv` (asymmetric flips 45/189),
  `path_b_K22_d12meta_tau100000.csv` (LB 0.95045 R7-eligible),
  `d12_lr_meta` (+1.348 bp OOF meta-derivative).
- **Gap to top-5%** (0.95345): −29.6 bp from PRIMARY. Bull projection on the held
  τ=20k candidate (11.6× amp) → 0.95370 (top-1% range).
- **Submissions used total:** 25/270; 1 LB submission today.

## Read order on session start (skip default; this is the synthesis)

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `scripts/probe.py` — entry point. `bote()` for BOTE, `gate()` for uniform gate report
3. `scripts/probe_min_meta.py` — K=K_pool+N stack-add gate
4. `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-pred NULL + load-bearing DGP diagnostic (joint-explains all per-row-FE NULLs)
5. `audit/2026-05-06-synthetic-data-batch.md` — 7-probe batch (today's session-1 NULLs)
6. `audit/2026-05-06-do-all-4-probes.md` — TE-audit CLEAN, α-resweep NULL, sparse-LR/lt-q5 NULL
7. `audit/2026-05-06-blend-and-rho-probes.md` — K=21 blends ruled out + ρ inventory
8. `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB-failure record
9. `audit/2026-05-06-alpha-asymmetry-verification.md` — α verification, harness intro
10. `audit/friction.md` — 5 new tags from this session at top
11. `scripts/pre_submit_diff.py` — MANDATORY before submit

**Harness usage cheatsheet (Rule 19):**
```bash
python scripts/probe.py bote NAME --family X --cost_min N \
    [--std_oof_lift_bp Y] [--prob_useful U]
python scripts/probe.py gate NAME --oof PATH --test PATH
python scripts/probe_min_meta.py --candidates NAME1 NAME2 ...
```

## Today's progress (2026-05-06, both sessions)

**Per-row feature-engineering family CLOSED.** Across both sessions today,
13+ single-base candidates tested at K=21+1 meta gate. All NULL when not
target-derived: NN-with-embeddings (ρ=0.918 most-diverse, NULL), Year×Stint
sparse-LR (ρ=0.844, NULL), within-Race LapTime_Δ q5, lap-mod/id-mod features
(566 bp marginal pattern absorbed), confidence-extreme pseudo-cascade,
multi-target NN with shared trunk, blend aggregators (mean/gmean/rank/trimmed).
**The 5th independent NULL** — `d14_dgp_residuals` (masked-column self-prediction)
— produced a load-bearing diagnostic: across all 4 self-pred targets, OOF RMSE
≈ marginal σ within 3 sig figs. **The synthetic NN-DGP is conditionally
near-independent within rows.** Per-row FE / SSL pretraining cannot break
the K=21 + Path-B ceiling — joint-explains FM-aug12 saturation, Move D NULL,
Day-13/14 alt-axis 4-of-4, TabPFN's 0.944 ceiling.

**The breakthrough.** `inv_laps_until_pit` = LGBM regression on
`1/(1+laps_until_pit)` (computed from PitNextLap labels per Driver-Race-Year
group). K=21+1 OOF +1.899 bp (largest non-meta-derivative single-add measured).
**Path B Compound×Stint over K=22+inv_laps τ=20k → OOF 0.95110** (+2.75 bp
vs PRIMARY). Pre-submit-diff PASS (ρ=0.99753; 53% rows shifted >1e-3).
**HELD pending submission decision.** Mechanistic distinction from
d12_lr_meta failure (LB −4 bp): inv_laps is target-derived (orthogonal),
not convex-combo of pool predictions.

**Day-14 also confirmed:** TabPFN v2.5/v2.6 DEAD (AUC ceiling 0.944);
FM-field-augmentation saturated at 12 fields (Move D / aug16 -0.07 bp NULL).

**Harness installed:** `scripts/probe.py` (bote + gate), `probe_min_meta.py`,
`probe_path_b_K22_invlaps.py`, `probe_target_reform.py` etc. CLAUDE.md
Rule 19 codifies the workflow.

## Falsified or dead — do NOT retry

See `ISSUES.md ## Falsified or dead` (full list, 20+ entries). Highlights:
- Per-row feature-engineering of any kind (5 NULLs jointly explained by `tag: synthetic-dgp-conditionally-near-independent`).
- Meta-derivative-as-base 2-level stacking (d12_lr_meta LB −4 bp confirmed; KD same family).
- TabPFN fine-tuning v2.5 / v2.6.
- FM-field-augmentation beyond 12 fields (Move D aug16, d14 H1 aug15).
- Drop-GBDT pool refactor (d13c −2.5 to −2.6 bp Strat).
- α-calibrated τ-resweep (τ=20k empirically optimal).
- TE fold-leak audit (d2a/d3a CLEAN).
- ρ-alone diversity heuristic (NULL across 3 probes today).

## Next-session first-action — RANKED by EV/cost

**Daily slot status (2026-05-06):** 10/10 used. Submission queue resumes
2026-05-07 with `path_b_K22_invlaps_tau20000.csv` as primary candidate.

### A1 — SUBMISSION (queued for next day, PI-approved)

`submission_path_b_K22_invlaps_tau20000.csv`. Re-checked vs new PRIMARY
(d15b_path_b_K22_dae_only LB 0.95059): OOF 0.95110 (+2.03 bp), ρ=0.99733
(PASS gate), flips 77/121 ratio 0.636. Per `path-b-amp-only-fires-on-meta-
arch-not-base-add` realised amp 1.4× → predicted LB ~+2.85 bp → ~0.95088.

### Day-15+ experiment menu (build on wins + try similar)

Three orthogonal-base mechanism classes are now empirically alive (each
realises ~1.4× LB amp, NOT Path-B 6-11.6×):
1. **Manifold learning** — Jahrer DAE swap-noise → LGBM-on-latent (LB +1bp)
2. **Target reformulation** — `inv_laps_until_pit` (held; +2.03 bp OOF)
3. **External data** — `orig_transfer` LGBM on aadigupta (regressed -1bp)

**Tier 1 — COMBINE the orthogonal bases (highest EV/cost; probes ready)**

- **T1a. K=23 = K=21 + DAE_only + inv_laps** under Path B Compound×Stint
  τ ∈ {5k, 20k, 100k}. ~10 min CPU. Both OOF/test artifacts on disk
  (`oof_d15b_lgbm_dae_only_strat.npy`, `oof_inv_laps_until_pit_strat.npy`).
  EV: P=0.40, expected +0.6 bp LB. **EXECUTE FIRST.**
- **T1b. K=24 = K=21 + DAE + inv_laps + orig_transfer** Path B sweep.
  Risk: orig_transfer regressed alone, may dilute. ~10 min CPU.
  EV: P=0.30, expected +0.3 bp LB.

**Tier 2 — More target reformulations** (`probe_target_reform.py` scaffold)

- T2a. `pit_horizon_multiclass` (4-class softmax: this / 1-2 / 3-5 / >5)
- T2b. `next_pit_lap_number` (regression on absolute lap number)
- T2c. `stint_index_within_race` (regression on completed-stints count)
- T2d. `reverse_cumcount_pits` (# remaining pits per Driver-Race-Year)

Each ~5 min CPU; 4 probes = 20 min. EV per: P=0.20, +0.3 bp LB expected.

**Tier 3 — DAE variants**

- T3a. Mask-noise DAE (zero-out instead of swap-noise) — ~15 min
- T3b. Stacked-2-layer DAE (deeper encoder, larger latent) — ~30 min
- T3c. CatBoost-on-DAE-latent (different downstream model class) — ~15 min
- T3d. DAE on K=21 OOFs themselves (autoencode pool predictions) — ~10 min

**Tier 4 — META-ARCH redesign (the friction's amp-eligible priority)**

Per `path-b-amp-only-fires-on-meta-arch-not-base-add`, meta-arch redesign
is the ONLY axis where Path-B's 6-11.6× amp can fire. Highest tail EV.

- T4a. Non-Gaussian shrinkage (Student-t / Beta-Binomial prior on segment LRs) — ~30 min
- T4b. Multi-level hierarchy (Stint within Compound within Year nested) — ~30 min
- T4c. Multi-cohort meta-blend (LR over multiple Path B OOFs as joint inputs) — ~30 min
- T4d. Yao/Vehtari covariance-modelled BMA — ~45 min (research first)

### A2b — Re-test d15_orig_transfer against the NEW K=22 pool (~30 min CPU)

`branch claude/decode-synthetic-data-uoPIn` confirmed `aadigupta1601` is
the synth's source dataset and built `d15_orig_transfer` (LGBM trained on
the original, predicts synth). Standalone synth-AUC 0.85138, ρ=0.565 vs
OLD PRIMARY (most-diverse single base since d9f FM_A 0.487). At hier-meta(K=22,
Compound×Stint, τ=20k) = K=21 + d15_orig_transfer: OOF +1.127 bp,
ρ=0.99844, flips 180 (R7 ✓), submitted LB 0.95049 TIE with OLD PRIMARY.
Branch pushed full artifacts. NEW PRIMARY (d15b_dae) brings K=22 inv_laps
to a different baseline — re-test orig_transfer against `K=22 = K=21
+ inv_laps_until_pit` (NEW baseline) to see if the two orthogonal
mechanism families stack. Audit: `audit/2026-05-06-d15-decode-synthesizer.md`.

### A3 — Pool composition: STRUCTURED replace, not naive drop

d13c falsified naive drop. Structured swap (drop 2 leak-eaters AND add 2
target-derived bases) untested. ~30 min CPU. EV +1 to +5 bp.

### A4 — External data revisit (Pirelli scrape)

ISSUES leaf 4a still open. Pirelli pit-window per (Compound, Race) historical.
Aggregate-prior (not row-join) integration pattern. EV +0.5 to +3 bp.

### Research-loop trigger (Rule 7) IF A1 misses on LB

If A1 LB ≤ PRIMARY (no advance): pause submits, redecompose ISSUES,
web-search top finisher writeups for synthetic-tabular Playground.

## Operating rules (load-bearing)

1. Pre-submit-diff before EVERY submit; ρ < 0.999 mandatory.
2. **Per-row feature engineering is dead** (`synthetic-dgp-conditionally-near-independent`). Don't propose new ones.
3. **ρ alone NOT sufficient for meta-utility** (3 NULLs at low ρ today triangulate this).
4. **Target-derived single-bases pass orthogonality**; meta-derivatives FAIL Path B amp.
5. Strat-only Day-3+ (R1) for primary OOF; public LB row-iid per U3.
6. Cap ≤3 concurrent CPU-heavy probes; schedule cheap probes first.

## Pointers (audit notes added today)

- `audit/2026-05-06-alpha-asymmetry-verification.md` — α verification + harness intro
- `audit/2026-05-06-blend-and-rho-probes.md` — K=21 blends ruled out + ρ inventory
- `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB-failure record
- `audit/2026-05-06-do-all-4-probes.md` — TE-audit / α-resweep / sparse-LR / lt-q5
- `audit/2026-05-06-synthetic-data-batch.md` — 7-probe batch
- `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-pred NULL + DGP diagnostic (load-bearing)
- `scripts/probe.py` / `probe_min_meta.py` — harness
- `scripts/probe_path_b_K22_invlaps.py` — THE breakthrough probe
- `scripts/probe_target_reform.py` — target reformulation scaffold (extensible)
- `scripts/d14_dgp_residuals.py` — DGP-residual probe

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
