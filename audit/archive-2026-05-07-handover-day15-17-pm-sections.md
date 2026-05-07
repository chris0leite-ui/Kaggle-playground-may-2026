# HANDOVER archive — Day-15/16/17 PM detailed sections

Archived 2026-05-07 PM during Day-18 wrap-up (Rule 5 file-size guard).
Source: `HANDOVER.md` lines 224-586 (pre-archive).

These are per-branch detailed PM sections from prior sessions. The
load-bearing summaries are already preserved in `HANDOVER.md`'s
top-level `## Where we are` and `## 🔴 CRITICAL — held candidates
INVALIDATED` sections. The detailed reports below remain accessible
for archaeology.

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


---

## Additional archive: stale front-matter sections (Day-15/16/17 short summaries + stale Next-session + older Pointers)

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

