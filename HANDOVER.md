# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are (Day-17 PM, 2026-05-07 evening)

- **PRIMARY** = `d17_path_b_K23_v4_h1d_tau100000` LB **0.95354** (Day-17 PM
  advance via `claude/optimize-model-performance-rruC2`, scored 2026-05-07
  13:27 UTC). K=23 = K=21 + `p1_single_cb_v4_gpu` (yekenot transfer
  recipe; see CB-axis arc below) + `d17_h1d_yekenot_full` (RealMLP-
  pytabkit replication imported from `claude/read-handover-62BCt`),
  Path-B Compound×Stint hier-meta τ=100k. **Realised OOF→LB gap −6.1 bp.**
- _Previous PRIMARIES (Day-17 trail):_
  - `d17_K24_d18pool_h1d` LB 0.95345 (research-branch midday;
    K=21 + d16_cont_only + p1_single_cb_v3 + h1d).
  - `d18_path_b_K23_d16_d18_tau20000` LB 0.95149 (research-branch AM;
    K=21 + d16_cont_only + d18_chain_decomp).
  - `d16_path_b_K22_continuous_only_tau20000` LB 0.95089 (yesterday).
  - `d15b_path_b_K22_dae_only_tau20000` LB 0.95059 (Day-15 PM).
- **Rank: #98 of 893 = top 11%.** Tied with [QWERTY] Roman Rozen at
  0.95354. Top-5% boundary moved up 60 bp today: was 0.95345 (stale),
  now **0.95405** (rank 44). **Gap to top-5%: −51 bp.** Leader MILANFX
  0.95476 (gap −122 bp).
- **Submissions used total:** 38/270; today 7/10.
- **Branches active recently (Day-17):**
  - `claude/optimize-model-performance-rruC2` — **NEW PRIMARY LB 0.95354**
    via CB-axis closure (v3→v4 yekenot transfer, +20.7 bp standalone OOF).
  - `claude/read-handover-62BCt` — h1d (yekenot RealMLP replication,
    OOF 0.95257) + K=24+h1d LB 0.95345 (intermediate Day-17 PM PRIMARY).
  - `claude/reverse-engineer-data-generation-Hu8EK` — d18 chain_decomp
    DGP probe (K=21+1 +7.37 bp) + d18 K=23 LB 0.95149 (Day-17 AM PRIMARY).
  - `claude/research-agentic-kaggle-W6IAP` — yekenot recipe extraction
    + h1d full-recipe replication.
  - `claude/autoencoder-synthetic-data-pEMB6` — d16 cont_only LB 0.95089.

## ⭐ Day-17 PM CB-axis arc + new PRIMARY

This branch (`claude/optimize-model-performance-rruC2`) closed the CB
recipe axis with a +20.7 bp single-model lift via yekenot FE transfer.
Full audit: `audit/2026-05-07-d17-cb-v4-yekenot-transfer.md`.

| Stage | OOF | LB | Key fact |
|---|---:|---:|---|
| LGBM v3 honest (Day-17 AM, P1) | 0.94563 | n/a | single-model honest ceiling |
| **CB v3 (research-recipe)** | 0.94993 | **0.95143** | +43 bp over LGBM v3; K=21+1 +12.06 bp |
| **CB v4 (yekenot transfer + orig-aug)** | **0.95200** | (held) | +20.7 bp over v3; K=21+1 **+24.21 bp** |
| K=22 v4 Path-B C×S τ=20k | 0.95319 | (held) | Path-B amp +0.39 bp (friction re-confirmed) |
| **K=23 v4+h1d Path-B τ=100k = NEW PRIMARY** | **0.95415** | **0.95354** | submitted; gap to LB top −122 bp; rank 98/893 |

**Recipe transfer items 2/3/4 (floor-cat, count-encoding, KBins) fired
on CatBoost despite research-branch audit caveat that they are
"NN-specific."** New friction tag candidate
`yekenot-floor-count-kbins-fires-on-gbdt-too`.

## 🎯 Next-step priority list (Day-18+)

To close the −51 bp gap to top-5% (boundary 0.95405), ranked by EV/cost.

| # | Move | Cost | Predicted LB Δ | Notes |
|---|---|---|---:|---|
| 1 | **K=25 = K=21 + v4 + h1d + d16 + d18** | 10 min CPU | +3 to +8 bp | Cheapest near-term; combines all today's bases |
| 2 | 3-seed bag of v4 (seeds 42/13/71) | 75 min Kaggle GPU | +1 to +3 bp | Variance reduction on load-bearing base |
| 3 | XGB with v4-recipe FE | 30 min Kaggle GPU | +5 to +15 bp | New model class on same FE; ρ < 0.97 likely |
| 4 | RealMLP n_ens=24 (h1d ran 4) | 3.5 h CPU/GPU | +2 to +5 bp standalone | Yekenot's published; +1-3 bp at K-meta |
| 5 | **FastF1 lap-by-lap pit-call hard-join** | 1-2 days | +10 to +30 bp | HANDOVER A4; only single-mechanism path to top-5% |
| 6 | Pirelli tyre-curve scrape | 1-2 days | +10 to +30 bp | Same axis as #5 |
| 7 | Per-Year specialists with v4 recipe | 30 min Kaggle GPU | ±5 bp | d12 found 2023 is easiest segment |
| 8 | Cross-segmentation Path-B (Y×S, R×C) on K=23 v4+h1d | 20 min CPU | +0 to +3 bp | d14 falsified Y-axis without v4 |

Items 1-3 sum to ~+10-25 bp predicted; still below the 51 bp gap.
Items 5-6 are the only single-mechanism path.

## 🔴 CRITICAL — held candidates INVALIDATED

End-of-day strict-OOF audit on this branch **collapsed all
target-reformulation single-add results 88-100%**:

| candidate | original Δ at K=21+1 | strict-OOF Δ | collapse |
|---|---:|---:|---:|
| reverse_cum | +4.867 bp | −0.005 bp | 100% |
| pit_horizon | +3.191 bp | +0.302 bp | 90% |
| inv_laps_until_pit | +1.899 bp | +0.234 bp | 88% |
| Joint K=21+3 | +7.667 bp | +0.275 bp | 96% |

**Bug:** `compute_targets()` in `scripts/probe_target_reform.py` and
`_v2.py` aggregates per (Driver, Race, Year) group using ALL train
labels — leaking val-row labels into tr-row regression targets via
`total_pits` + `cumsum`. New friction tag
`target-construction-layer-leakage`. Same family as
`d12_lr_meta` 2-level stacking (LB regress on +1.348 bp inflated OOF).

**Held candidates DO NOT submit:**
- `path_b_K22_invlaps_tau{5k,20k,100k}.csv` — 88% leaky
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv` — partially leaky (inv_laps component)
- `path_b_K25_megapool_tau{5k,20k,100k}.csv` — 96% leakage mirage
- `path_b_multilevel_τ_*.csv` — 5 configs NULL anyway

**Held candidates safe (no target-leakage):**
- `d15b_path_b_K22_dae_only_tau{20k,100k}.csv` (PRIMARY + close-second)
- `path_b_K22_d12meta_tau100000.csv` (LB 0.95045, R7-eligible HEDGE)
- `d15c` (ExtraTrees), `d15d` (LGBM-on-KNN) — R5 HEDGE only

Audit: `audit/2026-05-06-target-reform-leakage-audit.md`.

## Read order on session start

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `audit/2026-05-06-target-reform-leakage-audit.md` — **load-bearing** —
   strict-OOF audit collapse table
3. `audit/2026-05-16-d16-virgin-axes-results.md` — Day-16 virgin-axes
   (11 probes, all NULL / falsified / parked)
4. `audit/friction.md` — top tags `target-construction-layer-leakage`,
   `path-b-amp-only-fires-on-meta-arch-not-base-add`,
   `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`,
   `lr-meta-rank-lock-strong-anchor`
5. `scripts/probe.py` — `bote()` + `gate()` harness (Rule 19)
6. `scripts/probe_min_meta.py` — K=21+N stack-add gate
7. `scripts/probe_target_reform_strict_oof.py` — strict-OOF audit pattern
8. `scripts/pre_submit_diff.py` — MANDATORY before submit

## Day-15 PM (read-handover-LgbQ4): NEW PRIMARY LB 0.95059 via DAE

Submission `d15b_path_b_K22_dae_only_tau20000` (52394353 COMPLETE
2026-05-06 15:38) — DAE 768d latent → LGBM-on-latent → K=22 + Path B
Compound×Stint τ=20k. Realised LB amp **1.4×** on +0.715 bp OOF —
load-bearing for the new friction tag
`path-b-amp-only-fires-on-meta-arch-not-base-add`.

DAE artifacts re-usable for any future K_pool+N probe (no need to
retrain): `oof_d15b_lgbm_dae_{full,only}_strat.npy` + test variants.

## Day-16 (read-handover-lA8Nr): virgin-axes complement, all NULL

11 probes covering α/β/δ/ε/ζ/η axes from the d13 problem decomposition
tree. Highlights:
- **α4 GRU sequence on (Driver, Race) lap windows**: std OOF 0.93066,
  ρ=0.919 (most-diverse base of session). K=22+1 LR-meta Δ=−0.043 bp NULL.
  **5th cross-confirmation of `lr-meta-rank-lock-strong-anchor`.**
- **ε2 twin parallel-pool 2-meta blend**: ρ(metaA, metaB)=0.967 real
  disagreement; top-level LR vs single LR-meta(K=11): FALSIFIED Δ=−1.79 bp.
  Friction `twin-pool-2-meta-collapses-rank-info`.
- **δ2/3 conformal isotonic 4 schemes**: All regress −2.5 to −9.6 bp NULL.
  Friction `primary-hier-meta-globally-calibrated`.
- **ζ6 transductive pseudo (full-test soft labels)**: marginal +0.63 bp
  at LR-meta-K22 but −0.30 vs PRIMARY hier. R5 HEDGE only.
- 2 parked, 3 killed (DeepGBM ε4 over-engineered, etc.)

Full audit: `audit/2026-05-16-d16-virgin-axes-results.md`.

## This branch (ml-handover-alignment-xvUN0): harness + target-reform leakage

**Three significant deliverables:**
1. **Harness installed** (`scripts/probe.py`, `probe_min_meta.py`,
   18+ probe scripts). CLAUDE.md Rule 19 codifies BOTE-first / gate-after.
2. **Target reformulation thesis FALSIFIED via strict-OOF audit** (above).
3. **4-tier multi-level Path B (T4a)**: 5 (τ_0, τ_1, τ_2) configs all NULL.
   Simple multi-tier-shrinkage variant doesn't fire Path-B amp.

**Per-row feature engineering family CLOSED** (5 NULLs jointly explained
by `tag: synthetic-dgp-conditionally-near-independent`).

## Falsified or dead — do NOT retry

See `ISSUES.md ## Falsified or dead` (full list). Highlights:
- **target_reformulation_invlaps / pit_horizon / reverse_cum / stintprog**
  — all leaky; strict-OOF audit 88-100% collapse
- **path_b_K22_invlaps_*, path_b_K23_dae_invlaps_*, path_b_K25_megapool_***
  — all built on leaky targets
- **multi_level_path_b_4tier** — 5 configs NULL
- **Day-16 virgin-axes** — 11 of 11 NULL/falsified/killed
- TabPFN v2.5/v2.6, FM-aug16+, drop-GBDT pool refactor, simple K=21
  blends, α-calibrated τ-resweep, multi-target NN, masked-column
  self-prediction (DGP-residual)

## Next-session first-action — RANKED by EV/cost

### A1 — SINGLE-MODEL HYPOTHESIS TEST (PI-directed for next session)

PI hypothesis: leader at LB ~0.955 likely uses ONE strong model with
a structural mechanism we missed. Our 25-base stacking chases inflated
OOF that doesn't transfer.

**Procedure:** train ONE LightGBM (or CatBoost) with a wide feature
set including raw + strict-OOF target reformulations as FEATURES (not
separate bases). Measure standalone OOF AUC. If it beats ~0.945
(baseline+) significantly → "single model" path alive. If standalone
is unchanged → the +50 bp gap requires a structural insight we
haven't found.

Inputs available:
- Raw features (11 numeric + 3 cat)
- `oof_target_reform_{reverse_cum,pit_horizon,inv_laps}_strict_strat.npy`
  (strict-OOF, leak-free per `probe_target_reform_strict_oof.py`)
- DAE 768d latent (`oof_d15b_lgbm_dae_only_strat.npy`)

Cost: ~10-30 min for one wide-feature LightGBM 5-fold + standalone AUC.

### A2 — Pirelli external data scrape (ISSUES leaf 4a; untouched)

Aggregate-prior pattern (per (Compound, Race, Year) historical),
NOT row-join (d2 row-join failed at 5.6% match rate). Tier-2 EV per
Day-8 research. EV +0.5 to +3 bp.

### A3 — Examine raw data structure for missed leak

`id_mod_1000` 568 bp marginal span absorbed by GBDT interactions when
added as feature. Question: is there a **non-feature** structure (row
order, group ordering) that encodes pit_next_lap directly?

### A4 — Web search top-finisher Playground writeups

Pattern-match leader's "single model at 0.955" against published
synthetic-tabular Playground writeups.

### Meta-arch redesign (still alive at structural level)

Untested in T4a's simple multi-tier:
- Non-Gaussian shrinkage prior (Beta-Binomial / Student-t)
- Yao/Vehtari covariance-Σ BMA (LKJ + GP prior)
- Alternative segmentation cross (Year×Compound, Compound×TyreLife_q5,
  Driver-cluster × Stint)

### Research-loop trigger (Rule 7)

If A1 + A2 + A3 all NULL: pause submits, re-decompose ISSUES.md (3+
plateau-days now confirmed; per-row FE family closed; target-reform
family closed via leakage-audit; meta-arch redesign still untouched
at proper Bayesian level).

## Operating rules (load-bearing)

1. **Pre-submit-diff before EVERY submit**; ρ < 0.999 mandatory.
2. **Strict-OOF audit any per-group y-derived target before submission**
   (`tag: target-construction-layer-leakage`).
3. **Per-row feature engineering is dead**
   (`tag: synthetic-dgp-conditionally-near-independent`).
4. **ρ alone NOT sufficient for meta-utility** (5 cross-confirmations).
5. **Path B amp does NOT fire on base-adds** (1.4× realised, not 6-11.6×;
   `tag: path-b-amp-only-fires-on-meta-arch-not-base-add`).
6. **Path B amp REQUIRES orthogonal signal** (meta-derivatives FAIL;
   `tag: path-b-amp-needs-orthogonal-signal-not-meta-derivatives`).
7. Strat-only Day-3+ (R1) for primary OOF; public LB row-iid per U3.
8. Cap ≤3 concurrent CPU-heavy probes; schedule cheap probes first.

## Pointers (audit notes added today)

- `audit/2026-05-06-target-reform-leakage-audit.md` — **load-bearing**
- `audit/2026-05-16-d16-virgin-axes-results.md` — Day-16 11-probe NULL audit
- `audit/2026-05-06-alpha-asymmetry-verification.md` — α verification + harness intro
- `audit/2026-05-06-blend-and-rho-probes.md` — K=21 blends ruled out + ρ inventory
- `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB-failure record
- `audit/2026-05-06-do-all-4-probes.md` — TE-audit / α-resweep / sparse-LR / lt-q5
- `audit/2026-05-06-synthetic-data-batch.md` — 7-probe synth-data batch
- `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-pred NULL + DGP diagnostic (load-bearing)
- `audit/2026-05-15-d15-4branch-results.md` — 4-branch + B-GPU + DAE submit audit
- `scripts/probe.py` + `probe_min_meta.py` + `probe_target_reform_strict_oof.py` — harness
- `scripts/probe_path_b_K22_invlaps.py` — INVALIDATED (`target-construction-layer-leakage`)
- `scripts/probe_target_reform.py` — INVALIDATED (`target-construction-layer-leakage`)
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

---

## Day-16 PM autoencoder-synthetic-data-pEMB6 (overnight gauge-p-synth sweep)

**🎯 NEW PRIMARY: LB 0.95089 🎯** (`submission_d16_path_b_K22_continuous_only_tau20000.csv`,
ref 52410696, scored 2026-05-07). +3.0 bp over previous PRIMARY 0.95059.
Top-5% gap closes −28.6 bp → −25.6 bp (leader 0.95345). Realised amp ~1.0×
on +3.10 bp OOF advance.

**Headline.** **`d16_path_b_K22_continuous_only_tau20000` OOF 0.95121** =
HIGHEST OOF EVER (+3.10bp vs prior PRIMARY OOF 0.95090). Standalone base
`d16_orig_continuous_only` K=21+1 +3.331bp = LARGEST single-base K=21+1 of
session (beats inv_laps +1.90 by 1.75×).

**Mechanism.** Selective feature-restriction transfer. Orig-trained LGBM on
the 7 features the synthesizer left marginal-aligned (TyreLife KS=0.017,
Position 0.019, LapTime 0.056, LapTime_Delta 0.179, Cumulative_Degradation
0.071, RaceProgress 0.186, LapNumber 0.188). Phase-1 KS-divergence diagnostic
*literally* guided the choice. ρ vs PRIMARY 0.9946 — most-diverse positive
single base since d15_orig_transfer (0.5653).

**Sweep done overnight** (5 phases × 19 probes; CPU-only; 0 submits per
Rule 1; 12h wall):
  - **Phase 1 ✅** SDV overall 0.803; class-conditional structure SHARPER
    in synth than orig (synth Stint y0-vs-y1 KS 0.43 vs orig 0.24).
  - **Phase 2 v2 ✅** Density ratio r̂(x) (Driver/Race excluded after v1
    AUC 0.9985 from ghost-Driver tells). r̂ as feature NULL; as sample
    weight +0.78bp; as cohort router +1.32bp. New friction
    `density-ratio-routes-or-weights-but-fails-as-feature`.
  - **Phase 3 ✅** GMM 16-comp single-feat: ρ=0.503 (most-diverse single
    base ever) but K=2 NULL — 4th confirmation of `rho-alone-insufficient-for-meta-utility`.
    BGMM at reg_covar=1.0 oversmoothed (AUC 0.55 near-random) — new friction
    `bgmm-default-oversmooths-at-reg-covar-1`.
  - **Phase 4 v2 ✅ KEY WIN** Orig-transfer feature-subset variants. All
    4 PASS K=21+1; continuous_only +3.33, no_laptime +1.87, no_tyrelife_rp
    +0.86, dr_split +1.32. New friction
    `feature-subset-orig-transfer-passes-where-arch-bag-fails`.
  - **Phase 5 null+caveat** All 6 ran r̂_q5/logp_q5 cohort axes regress
    -3 to -4bp. CAVEAT: ran on K=14 sub-pool (only 14/21 named bases
    matched filenames). New friction
    `path-b-on-pool-subset-conflates-cohort-axis-with-pool-size`.
  - **Phase 6 ✅** K=21+1 individual gates + K=21+7 panel (+5.94bp;
    continuous_only |w|=1.48 dominates) + K=22 Path B Compound×Stint
    sweep with continuous_only as 22nd base.

**Submission result**:
  - τ=20k variant SUBMITTED 2026-05-07 (PI authorized; 468 flips override).
    OOF 0.95121 → **LB 0.95089** (+3.0 bp over previous PRIMARY).
  - τ=100k variant HELD as HEDGE-eligible candidate (R5 final-window probe).

**Cross-branch awareness**: claude/ml-handover-alignment-xvUN0 has held
candidate `path_b_K22_invlaps τ=20k` OOF 0.95110. d16's continuous_only
variant lifts another +1.10bp OOF over inv_laps and is mechanistically
distinct (selective feature-restriction transfer vs target reformulation).
Independent mechanism families — should consider K=23 stack-add of both
in next session.

**Pointers added today**:
- `audit/2026-05-07-overnight-gauge-p-synth.md` — full overnight sweep audit (synthesis section finalized)
- `audit/friction.md` — 5 new friction tags from this branch
- `ISSUES.md` § 7 — umbrella `gauge-p-synth-overnight` with 5 sub-leaves (4 done, 1 null)
- `scripts/d16_gauge_phase{1,2_v2,3_likelihood,3b_bgmm_fix,4_v2,5_pathb,6_wrapup}.py`
- `scripts/d16_path_b_K22_continuous_only.py` — submission candidate generator
- `submissions/submission_d16_path_b_K22_continuous_only_tau20000.csv` — submission CSV (NOT submitted; Rule 1)
- 16+ d16 OOF/test pairs in `scripts/artifacts/oof_d16_*` and `test_d16_*`
- `scripts/artifacts/d16_phase{1..5}_summary.json` + `d16_overnight_consolidated.json`

**Next-session priorities (handover):**
1. **PI submission decision**: τ=100k (HEDGE-safe) vs τ=20k (PI sign-off); predicted LB +4-5bp lift.
2. **K=23 stack-add**: K=21 + d16_orig_continuous_only + path_b_K22_invlaps (cross-branch). Both PASS independently; orthogonal mechanisms; predict additive ~+4-5bp K=23 OOF.
3. **Phase-5 re-test on full K=21**: cleanly disambiguate cohort-axis failure from missing-bases artifact.
4. **Tune continuous_only LGBM**: feature subset is fixed; tune n_leaves, min_data, subsample on orig.
5. **Multi-arch on continuous_only feature subset**: CatBoost / XGB with same 7 features (different from d15_orig_multi_arch which varied arch on full features).

---

## Day-17 AM autoencoder-synthetic-data-pEMB6 (status-only wrap)

**No compute this session.** Session opened with a misgrounded clarifying
question (proposed "Phase F" without first reading branch state); PI
responded **"stop here. wrap up."** Inherited d17 Phase 0 + Phase A in flight
from prior commit `1f442e8`. No LB submissions; no new OOF/ρ measurements.

**Inherited artifacts staged in this wrap** (under `scripts/artifacts/`):
- Phase 0 leakage cleanup: `d17_phase0_leakage_summary.json`,
  `oof_d17_dr_weighted_orig_v2_strat.npy` + test pair.
- Phase A K=22/K=23 stack-add OOF/test pairs (5 candidates, `_strat`):
  `oof_d17_C1_K22_cont`, `oof_d17_C2_K23_cont_nolaptime`,
  `oof_d17_C3_K23_cont_notyrerp`, `oof_d17_C4_K23_cont_catonly`,
  `oof_d17_C5_K23_cont_invlaps_strict` (+ matching `test_*` files).

**Phase B / Phase C unrun.** Scripts present and committed in `1f442e8`:
- `scripts/d17_phase_b_extend.py` — multi-arch + N-sweep + physics
  specialists + synth-restricted variants.
- `scripts/d17_phase_c_meta_arch.py` — Path-B-amp meta-arch redesigns
  (Student-t shrinkage, 3-level hierarchy, 75-seg Compound×Stint×r̂_q3).

**Next agent:** read `1f442e8` commit message + the C1-C5 artifacts above,
then either (a) gate the C-candidates with `probe_min_meta.py` to decide
PRIMARY-advance vs HEDGE, or (b) execute Phase B / Phase C scripts.
PRIMARY remains `d16_path_b_K22_continuous_only_tau20000` LB **0.95089**.

**File-size flag:** HANDOVER.md is at 317 lines, over the 150-line cap in
WRAPUP.md step 5. This bloat predates today's session; not archived here
to avoid touching other branches' Day-N PM sections (Rule 15). Flag for
the next merge-target scribe.

---

## Day-17 AM read-kaggle-handover-rsi2Q

P1 single-model thesis (PI hypothesis "leader at LB ~0.955 likely
uses ONE strong model") — tested end-to-end via Rozen 0.95354 recipe
replication. **CONCLUSIVELY FALSIFIED** under strict OOF discipline.

### What ran
- Pulled top 8 public s6e5 notebooks under `external/kernels/` as
  reference (incl. `romanrozen/f1-pit-driver-race-year-encoding-0-95354`).
- Pulled external datasets: `aadigupta_orig`, `f1_official_1950_2022`
  (driver/circuit historical priors), `weather_woodshole`.
- Built `make_features_A` v1 (50 engineered + 6 CV TE incl Driver×Race×Year).
- v1 single LGBM OOF 0.94970 → submitted alone LB **0.94107** (gap −863 bp);
  K=22 LR-meta-add OOF 0.95404 → submitted LB **0.94933** (−126 bp vs PRIMARY).
- v2 fixed `stint_size_far` per-split-count cluster + added FS_A merge
  aggregates. OOF 0.95128 (+38 bp over PRIMARY OOF — too good).
  K=2 LR(PRIMARY, v2) submitted LB **0.94996** (−63 bp vs PRIMARY).
- 80/20 honest holdout test (`scripts/p1_holdout.py`) caught FS_A target
  leak: holdout AUC **0.94637** vs OOF 0.95128 = **−491 bp gap**.
- v3 with **fold-safe FS_A** (`fit_fs_a` per-fold, `apply_fs_a` merge):
  OOF **0.94563** matches holdout. Honest single-LGBM ceiling on this
  comp.

### What we now know
- Single-LGBM with kitchen-sink Rozen-style FE achieves OOF ~0.946.
- PRIMARY (K=22 + Path-B hier-meta, OOF 0.95090) is +52 bp ahead.
- **Stacking is necessary for our LB position.** P1 thesis FALSIFIED.
- Rozen's published 0.95241 single-LGB OOF is likely similarly inflated
  by FS_A leak in his pipeline (he uses the same `df[df['PitNextLap']==1]
  .groupby(...).mean()` pattern fit on full train); his real single-LGB
  LB is probably ~0.946, blend wins via 5 external sources.

### Other branch's win
- **`claude/.../d16_path_b_K22_continuous_only_tau20000` LB 0.95089**
  (+30 bp over PRIMARY 0.95059). Clean Path-B base-add candidate using
  KS-divergence-identified marginal-aligned features; this is the
  Day-17+ PRIMARY-replacement candidate to confirm.

### Lessons captured (skill `improvements.md` + local CLAUDE.md R20-R25)
- R20 single-model-first / kitchen-sink FE before stacking
- R21 family falsification requires ≥3 variants
- R22 public-notebook scan at every plateau
- R23 framework is scaffolding, not authorship
- **R24 fold-safe label-conditional aggregates** (NEW Day-17)
- **R25 transductive features need AV check** (NEW Day-17 PI lesson)

`scripts/p1_holdout.py` — 80/20 honest holdout test (independent seed).
Mandatory before any new-FE-family LB submit.

### Submissions used (all UTC days combined)
Day-17: 4/10 used (3 by this branch + 1 d16 from another).
Total: 32/270.

### Files
- `scripts/p1_features.py` — `make_features_static` + `fit_fs_a` +
  `apply_fs_a` (v3 fold-safe). Legacy `make_features_A` flagged.
- `scripts/p1_single_lgbm_v3.py` — fold-safe trainer.
- `scripts/p1_single_lgbm.py` — v1/v2 trainer (legacy).
- `scripts/p1_single_cb.py` — single CatBoost (deferred, not run).
- `scripts/p1_holdout.py` — 80/20 honest holdout.
- `scripts/p1_post.py`, `scripts/p1_gate_all.py` — gate harnesses.
- `scripts/artifacts/oof_p1_single_lgbm_v3_feA_te_strat.npy` (+test).
- `audit/2026-05-06-p1-single-model-{plan,results}.md`.
- `external/kernels/{romanrozen,...}/` — 8 reference notebooks.
- `external/{aadigupta_orig,f1_official_1950_2022,weather_woodshole,
  makimakiai_idsafe,gkanamoto_tabm,pavloivanin_baseline}/`.

### Open candidates from other branches
- `d16_path_b_K22_continuous_only_tau20000` LB 0.95089 — verify and
  consider as new PRIMARY.
- v3 single LGBM OOF 0.94563 itself — too low standalone, but ρ=0.953
  diversity. Genuine K=22+v3 stack-add lift only +3.40 bp OOF
  (vs leaky +30.79 bp). Held; probably not worth a slot.

---

## Day-17 PM read-handover-62BCt — TOP-5% AT-THRESHOLD via yekenot recipe (LB 0.95345)

**🎯 NEW PRIMARY: LB 0.95345 (AT TOP-5% THRESHOLD) 🎯**
`submission_d17_K24_d18pool_h1d.csv` (ref 52420646, scored 2026-05-07
11:39 UTC). +19.6 bp over d18 PRIMARY 0.95149 = **BIGGEST single-submit
lift of comp**. Headroom to top-5% closes from −19.6 bp → **0**.

### What worked

Full yekenot RealMLP recipe replication (`scripts/d17_h1d_yekenot_full_recipe.py`):
- 5-fold StratKF OOF AUC 0.95257 (matches yekenot pub 0.95273 within 1.6 bp)
- ρ_test vs PRIMARY 0.972 (single base) — first base to break ρ < 0.99
  in 5+ months
- All 6 load-bearing FE items: arithmetic ratios, floor-cat, count enc,
  KBins(200/7), per-fold stratified orig concat, **CV TargetEncoder on
  (Race,Compound)+(Race,Year) inside fold loop** (load-bearing).
- `n_ens=4` on 4-core CPU; yekenot's `n_ens=24` on Kaggle GPU is +5 bp
  ceiling at most.

K=24 d18pool+h1d submission stack:
- K=21 + d16_orig_continuous_only + p1_single_cb_v3_gpu + d17_h1d_yekenot_full
- LR-meta (Path B over K=22 with h1d was TIE per 6th cross-confirmation
  of `path-b-amp-only-fires-on-meta-arch-not-base-add`)
- OOF 0.95385, ρ_test vs d18 PRIMARY 0.989, predicted LB Δ +15 bp
- Realised LB Δ +19.6 bp (PI sealed prediction +10 bp; agent +15.11 bp;
  both conservative)

### Calibration outcomes (audit/decisions.jsonl)

| Probe | PI pred | Agent pred | Actual |
|---|---:|---:|---:|
| H1 (initial 3 variants) | 0 bp | +27 bp | NULL across 3 variants — recipe-gap misdiagnosis |
| H2 FastF1 | +5 bp | +3.6 bp | ~0 bp (1.4% match rate cap from synth D### codes) |
| H3 ID-shift | 0 bp | +0.6 bp | 0 bp (PI win — id_div_N AV is labeling convention only) |
| H1d full-recipe (final) | +10 bp | +15.11 bp | **+19.6 bp** (both beat) |

### What didn't work (this branch)

- H1 v1/v2/v3 (yekenot-hyperparams + orig-merge alone): all NULL.
  Misdiagnosed +69 bp standalone gap as hyperparameter+orig only;
  actual gap is the FULL FE pipeline.
- H2 FastF1: 1.4% match rate due to 60% synthetic D### driver codes
  + sandbox 403 on livetiming.formula1.com.
- H3 ID-shift: train ids 0..439139 / test 439140..627304 = labeling
  convention with zero overlap; sparse-LR base on id-div features =
  chance level.
- C7 K=24 LR-meta (without h1d): predicted LB Δ −0.69 bp (TIE/regress).

### Files

- `scripts/d17_h1d_yekenot_full_recipe.py` — verified replication
- `scripts/artifacts/oof_d17_h1d_yekenot_full_strat.npy` + test
- `scripts/artifacts/oof_d17_K24_d18pool_h1d_strat.npy` + test
  (the SUBMITTED stack)
- `submissions/submission_d17_K24_d18pool_h1d.csv`
- `external/kernels/ps-s6-e5-realmlp-pytabkit/VALIDATED.md`
- `.claude/skills/kaggle-comp/examples/fe-recipe-yekenot-realmlp-kitchen-sink.md`
- `audit/2026-05-07-d17-strategy-critique.md`
- `audit/2026-05-07-d17-h1-verdict.md`
- `audit/2026-05-07-d17-h2-fastf1-external.md`
- `audit/2026-05-07-d17-h3-id-shift.md`
- `audit/2026-05-07-d17-phase-a-composition-gate.md`

### Submissions used (Day-17, all UTC days combined)

7/10 today (this branch +1; 6 prior including 3 sibling submits).
Total: 35/270.

### Next-session priorities

1. **PI submission discussion**: do we need to submit anything else
   today? K=24 LR-meta variants with Path B Compound×Stint segmentation
   are unlikely to lift (6th confirmation of meta-arch friction).
2. **Tier-2 follow-ups for the yekenot recipe**:
   - n_ens=8 or 12 variant of h1d (~1-2 h CPU); +2-5 bp standalone OOF
     ceiling. EV +1-3 bp LB.
   - Apply CV-TE / engineered-cat FE pipeline to a second base
     architecture (CatBoost or LGBM on the same yekenot FE set).
     Could yield a structurally different base.
3. **PRIMARY-replace candidates pending sibling integration**: we have
   not yet tested K=25+ unions with sibling-branch new bases (d18
   already includes d16 + p1cb; if siblings produce d19+ candidates,
   re-stack).

---

## Day-17 PM read-handover-62BCt — d17 Phase-A composition gate

**0 submits this session.** Bootstrapped repo (deps + Kaggle data),
claimed ISSUES leaf 7f, re-ran inherited `scripts/d17_phase_a_compose.py`
to completion (sibling branch had bailed mid-run after C1-C5 OOFs were
written but before summary JSON / C6 / C7).

**Result.** Best K=24 LR-meta combo C7 (cont_only + no_laptime +
no_tyrerp) OOF **0.95129**, +5.50 bp over the script's printed PRIMARY
column — but that column was the OLD `oof_PRIMARY_K22_strat.npy` (d15b
DAE LB 0.95059, OOF 0.95074), not the actual current d16 cont_only
Path B PRIMARY (LB 0.95089, OOF 0.951208). Vs the actual current
PRIMARY, **C7 is +0.81 bp OOF at ρ_test 0.99506 → predicted LB Δ −0.69
bp (TIE/regress). All other Cn combos REGRESS −0.09 to −1.45 bp OOF.**

| Combo | K | OOF | Δ vs d16 PRIM (bp) | ρ_test | pred LB Δ |
|---|---:|---:|---:|---:|---:|
| C1 cont | 22 | 0.95106 | −1.45 | 0.99581 | −2.95 |
| C2 cont+nolaptime | 23 | 0.95120 | −0.09 | 0.99557 | −1.59 |
| C3 cont+notyrerp | 23 | 0.95122 | +0.11 | 0.99517 | −1.39 |
| C4 cont+catonly | 23 | 0.95115 | −0.54 | 0.99515 | −2.04 |
| C5 cont+invlaps_strict | 23 | 0.95107 | −1.42 | 0.97555 | −6.42 |
| C6 cont+nolaptime+invlaps | 24 | 0.95122 | +0.09 | 0.97714 | −4.91 |
| **C7 cont+nolaptime+notyrerp** | **24** | **0.95129** | **+0.81** | 0.99506 | **−0.69** |

**Mechanism.** Path-B Compound×Stint τ=20k segmentation on K=22 cont_only
adds +0.15 bp OOF over canonical LR-meta on the *same* pool. Stacking 3
more orig-LGBM bases via LR-meta does not close that gap. **5th
cross-confirmation of `path-b-amp-only-fires-on-meta-arch-not-base-add`.**
Strict-OOF inv_laps adds essentially nothing on top of cont_only (C5 vs
C1 = +0.04 bp); refines `target-construction-layer-leakage` finding —
even audit-cleaned strict-OOF inv_laps is not differentiated enough.

**Next step (NOT RUN — awaiting PI sealed prediction).** Path B
Compound×Stint τ=20k over the C7 K=24 pool. Cost ~15 min CPU. Family
`meta_arch_redesign` (p=0.30, (1, 4, 8) bp). Q6: log-loss / row-AUC =
True. Per Rule 26(a) PI commits LB Δ prediction first.

**Files**:
- `audit/2026-05-07-d17-phase-a-composition-gate.md` — full audit
- `scripts/artifacts/d17_phase_a_summary.json` — per-combo |w| + ρ
- `scripts/artifacts/oof_d17_C{1..7}_*_strat.npy` + `test_*` (C6/C7
  produced this run)
- `data/{train,test,sample_submission}.csv` re-hydrated via `bootstrap.sh`

---

## Day-17 PM review-handover-solutions-oE78b — 5-probe diagnostic pass, 1 structural find, 0 submits

**Result: 0 LB submits, 0 LB Δ.** PI sealed prediction = 0 bp for every
probe (Rule 26a); 5/5 vindicated at LB level (no submission).
PRIMARY remains **`d17_path_b_K23_v4_h1d_tau100000` LB 0.95354**.

### What ran

PI prompt: "What's hiding in plain sight?" Two rounds:

**Round 1 — 3 probes** (`52b00f2`):
- `probe_ntl_single_rule.py` (5 min): 5 NTL reconstructions + 13
  thresholded rules. Best AUC 0.687 < raw TyreLife alone 0.699.
  Host's brief.md "trivial" refers to unmasked original column.
- `probe_target_structure.py` (30 min EDA): P(target=1 | lap_from_
  observed_stint_end) decays 0.272 → 0.061 over 10 laps; 81% of
  multi-pos stints have last pos at observed-last-lap; 65% contiguous.
  Target is decay-from-end, not shifted PitStop, no deterministic rule.
- `probe_combined_lead_lag.py` (30 min): single-feat L1 +29 bp combined
  vs train-only on `lead_LapNumber_diff`. LGBM-level F4-F5 = -0.36 bp.
  Combined-frame premium evaporates at GBDT.

**Round 2 — probe 4 + leakage audit + K=24 gate** (`22ffbc0`,
`c548fda`, `1ca661a`, `e28db35`):
- `probe_field_state.py` (12 min): cross-row aggregates over
  (Race,Year,LapNumber) and (R,Y,L,Compound) from train+test combined.
  Standalone single-LGBM raw 14 + ~30 fs feats: full-train OOF
  0.94230 (+15.58 bp). Top single-feat **`fs_cum_pits` AUC 0.7972 —
  highest single-feat OOF on comp** (raw TyreLife alone 0.6989).
- PI flag: "isn't this leaky like Day-17 FS_A merge?"
- `probe_field_state_strict.py` (7 min): per-fold strict re-run.
  F3-strict OOF 0.94211 (+13.73 bp), F4-strict tr-only 0.94208
  (+13.35 bp). 12% collapse vs Day-17 88-100%. **Audit cleared.**
- K=24 stack-add gate: K=23 LR-meta (v4+h1d) OOF 0.95414, K=24
  (+ field-state) OOF 0.95414, marginal **-0.015 bp NULL**.
  field-state |w|=0.0807 vs v4 0.55 / h1d 0.48. **6th cross-
  confirmation of `lr-meta-rank-lock-strong-anchor`**.

### Calibration outcomes

| Probe | PI sealed | Agent BOTE | Realised LB | Realised OOF |
|---|---:|---:|---:|---:|
| probe_ntl_single_rule | 0 bp | n/a | 0 (no submit) | best 0.687 (≤ raw) |
| probe_target_structure | 0 bp | n/a (EDA) | 0 (no submit) | n/a |
| probe_combined_lead_lag | 0 bp | +0.025 bp | 0 (no submit) | +2.18 bp |
| probe_field_state | 0 bp | +0.025 bp | 0 (no submit) | +13.73 bp strict |
| probe_field_state K=24 gate | 0 bp | n/a | 0 (no submit) | -0.015 bp marginal |

PI 5/5 vindicated at LB. Family prior `single_base_fe_addition` was
mis-priced for cross-row aggregates (30× under at standalone OOF level)
but correctly predicted 0 bp at K=24 stack transfer.

### Friction tags introduced

- `cross-row-aggregates-fire-where-own-row-sequence-doesnt`
- `cross-row-aggregates-survive-strict-fold-safe-audit` (PROMOTED to
  improvements.md G18)
- `field-state-mechanism-fires-on-train-only-too-no-combined-premium`
- `family-prior-single-base-fe-addition-mis-calibrated-for-cross-row`
- `lr-meta-rank-lock-strong-anchor` (6th cross-confirm)
- `host-quote-trivial-refers-to-original-not-reconstructible`
- `pitnextlap-target-cluster-decay-not-shift`
- `combined-frame-leadlag-premium-evaporates-at-gbdt`

### Next-session first-action candidates

1. **CB-v4 + field-state retrain on Kaggle GPU** (~35 min). Add ~24
   field-state features INTO the yekenot recipe; retrain CB; predict
   v4-fs standalone OOF lift +1 to +5 bp over current v4 OOF 0.95200.
   If lifts ≥+3 bp, replace v4 in K=23 and re-gate. Bypasses rank-lock
   by changing the strong anchor itself.
2. h1d + field-state retrain (RealMLP n_ens=4 with fs feats added).
3. **External Pirelli/FastF1 hard-join** (HANDOVER A4) — only
   single-mechanism path to top-5% (51 bp gap to 0.95405 boundary).
4. K=25 full-merge (+d16 cont_only, +d18 chain_decomp) Path-B held
   +1.3 bp OOF — within noise but cheap to re-submit.

### Files
- `audit/2026-05-07-handover-review-3-probes.md` — Round 1 audit
- `audit/2026-05-07-probe4-field-state-aggregates.md` — Round 2 audit
  including strict fold-safe + K=24 gate
- `audit/2026-05-07-postmortem-review-handover-solutions-oE78b.md`
- `scripts/probe_ntl_single_rule.py`
- `scripts/probe_target_structure.py`
- `scripts/probe_combined_lead_lag.py`
- `scripts/probe_field_state.py`
- `scripts/probe_field_state_strict.py`
- `scripts/probe_field_state_lgbm_artifact.py`
- `scripts/artifacts/probe_*` JSONs + `oof_field_state_lgbm_strat.npy`
  + `test_field_state_lgbm_strat.npy`
- `.claude/skills/kaggle-comp/improvements.md` G18 (PI ratified)

### Submissions used (Day-17, all UTC days combined)
0 this branch. Day-17 total: 7/10 (unchanged).

### Branch state
Top of `52b00f2 → ca40a76 → 22ffbc0 → 0241693 → c548fda → 1ca661a → e28db35`.
Ready for ff-merge to `main` after PI sign-off.
