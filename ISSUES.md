# ISSUES.md — live problem decomposition

> BPS step-2 (disaggregate) + step-3 (prioritise) output. Refreshed
> when the strategy-critic-loop fires (plateau, saturation, kickoff,
> 50% checkpoint). Cap ≤150 lines.

**Problem (BPS step 1; Day-19, 2026-05-07).** Close the **−3.7 bp** gap
to top-5 % boundary (0.95405) from PRIMARY 0.95368
(`d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000`) in ~8 remaining days.
In-pool axes empirically exhausted on K=27 with v4+h1d anchors: meta-arch
redesign closed (C axis, 9 variants tested), external-data closed
(D axis), gbdt-class redundant (B2). Only structurally distinct axis
remaining is **A1 sequence-level** (HMM Compound transitions + AR(1)
within-stint TyreLife — leaf 7h below). All other open leaves are
optional / R5 HEDGE prep.

**Claim convention.** Pick an unclaimed `open` leaf. Edit its
`[owner: ...]` field to your branch slug (part after `claude/`).
Commit + push. One open leaf per branch at a time. Update status
at wrap-up via `WRAPUP.md` section A.

Status values: `open`, `wip`, `done`, `null` (falsified), `parked`.

---

## 1. New model class (existing classes saturated within pool)

- **1a.** TabPFN fine-tuned (v2.5 or v2.6). **DEAD.**
  v2.5 @ 50k and 150k rows: fold-0 AUC 0.9444 both (flat training loss;
  fine-tuning not learning). v2.6 OOM on P100 at any row count (model
  weights ≈15.37GB > 16GB). AUC ceiling -64bp vs PRIMARY. ρ=0.960
  diverse but gap too large. `[owner: read-handover-8hsZh | status: null]`
- **1b.** EmbMLP CPU baseline → reframed as Jahrer swap-noise DAE +
  LGBM-on-latent. **DONE 2026-05-06**: GPU kernel d15b-dae-lgbm-gpu v2
  ran on P100 (torch 2.4 sm_60 fix). std OOF 0.94007, ρ_test 0.9477
  (most-diverse since FM_A_53), min-meta +0.793bp at ρ 0.99547.
  K=22 Path B Compound×Stint τ=20000 OOF +0.715bp. **SUBMITTED LB
  0.95059 (+1.0bp NEW PRIMARY).** Realised amp 1.4× (well below Path-B-amp
  6-11.6× central) — new friction `path-b-amp-only-fires-on-meta-arch-not-base-add`.
  `[owner: read-handover-LgbQ4 | status: done]`
- **1c.** DeepFM-lite (FM + 2-layer MLP head). Extends d9c FM with
  non-linear interactions. Cheap CPU.
  `[owner: unclaimed | status: open]`
- **1d.** Regularised FFM re-attempt. d9e FFM strictly worse, but
  with stronger regularisation (dropout, L2) may flip the tie.
  `[owner: unclaimed | status: parked]`
- **1e.** Kernel SVM family. **DONE 2026-05-08 — FAIL on both K=10
  sparse and K=27 PRIMARY pools.** Two full-data variants on the
  vanilla-LR 45-feature recipe with γ-swept Nyström-RBF approximation
  (n_components=800, γ=0.02 chosen from a 5-point γ-sweep at smoke):
  - Nyström-RBF + LinearSVC (squared-hinge): standalone OOF 0.91395.
    K=10+1 Δ −0.09 bp / K=27+1 Δ −0.02 bp.
  - Nyström-RBF + LogisticRegression (kernel-logistic, calibrated):
    standalone OOF 0.91203. K=10+1 Δ −0.06 bp / K=27+1 Δ −0.00 bp.
  Standalone OOF lifts +56 bp over matched-feature linear LR (0.85588)
  — the kernel non-linearity does real work — but standalone is still
  −400 bp from PRIMARY 0.95431, and the meta can't route diversity
  gain when the AUC gap to GBDT-class is that large. ρ_test 0.82-0.84,
  G3 flip ratio 0.00 (linsvc) / 0.13 (klogreg). 8th rank-lock
  confirmation. New friction
  `kernel-class-fails-when-standalone-AUC-gap-to-gbdt-exceeds-300bp`.
  Smoke γ-sweep (n_components=1500): γ=0.005 → 0.00, γ=0.01 → 0.916,
  γ=0.02 → 0.918, γ=0.04 → 0.914, γ=0.1 → 0.897, γ=0.5 → 0.847,
  γ=1.0 → 0.815. `scripts/svm_kernel_probe.py`, `scripts/svm_gate.py`.
  `[owner: explore-svm-kernels-TRcuo | status: null]`

## 2. Meta-layer innovations beyond Path-B hier-meta

- **2a.** Non-Gaussian shrinkage prior on hier-meta. Path B is
  Gaussian-τ; try Beta-Binomial or Student-t shrinkage.
  `[owner: unclaimed | status: open]`
- **2b.** Calibration-aware stacker for private-LB row-iid structure
  (test is i.i.d. per U3). Re-fit calibration on holdout.
  `[owner: unclaimed | status: open]`
- **2c.** GroupKF-meta as R5 HEDGE. d12 K=20 GKF clean; ρ vs
  Strat-meta 0.9856; private-LB-likely-real per d10/d13d probes.
  `[owner: unclaimed | status: open]`
- **2d.** Yao/Vehtari covariance-modulated Path-B (V3 in c1 audit).
  Per-segment LR with prior covariance Σ from inter-base sample
  correlation; shrinks weights along low-eigenvalue (highly-correlated)
  directions more than plain Path-B. K=27 implementation `scripts/
  c1_yao_vehtari_bma.py` (V0 plain LR / V1 Path-B τ=100k / V2 plain BMA
  / V3 Yao/Vehtari τ ∈ {10k, 50k, 200k}). Result: **V3 REGRESSES vs V1
  by −0.47 to −0.59 bp across all τ.** V3 over-shrinks correlated
  base-routing directions LR uses. Friction
  `covariance-modulated-path-b-overshrinks-correlated-base-routing-directions-vs-plain-tau`.
  Day-19 overnight: 9th meta-arch variant tested; family closed.
  `[owner: ml-competition-analysis-rwD3f | status: null]`
- **2e.** Non-LR meta on K=4 [P, rank, logit] (EXP-NEW; the architecturally-
  untested avenue per A30). Tested 2026-05-08 PM
  (research-model-extensions-Ibwvn). **All three variants NULL or
  regressing:** gradient-boosted meta (LGBM num_leaves=15) Δ −1.20 bp;
  2-hidden-layer MLP meta Δ −7.77 bp; augmented LR with raw row
  features Δ −0.04 bp. **A30 dropped from `live` to `FALSIFIED`.**
  Mechanism: K=4 logit pool collapses to 1.33-D (A31); non-linearity
  at the meta layer overfits a 30-feature input on a sub-2-D signal.
  See `audit/2026-05-08-four-lane-research-extension.md`,
  `scripts/probe_lane4_nonlr_meta.py`,
  `scripts/artifacts/probe_lane4_nonlr_meta.json`. **Closes the
  meta-architecture lane completely.**
  `[owner: research-model-extensions-Ibwvn | status: null]`
- **2f.** Heuristic / domain-prior features as **meta inputs** (not as
  base). Tested 2026-05-08 PM. 9 hand-crafted F1 features
  (compound_tier, tyre_life_pctile_in_compound, laps_to_race_end,
  is_last_3_laps, race_progress, n_distinct_compounds_so_far,
  field_size_at_lap, stint_overrun) added alongside K=4 [P, rank,
  logit]. K=4+heuristic-meta Δ −0.02 bp NULL. Deterministic rule
  clamps regress −9.48 bp (synth label diverges from F1 realism on
  is_last_lap; friction `synth-divergence-from-F1-realism-on-last-lap`).
  Compound-tier monotonic LGBM K=4+1 Δ +0.19 bp within noise.
  `scripts/probe_lane2_priors.py`,
  `scripts/artifacts/probe_lane2_priors.json`.
  `[owner: research-model-extensions-Ibwvn | status: null]`
- **2g.** Per-Compound LR meta heads (routed at inference; distinct
  from base-level per-Compound specialists which absorbed at K=10+1
  per A4). Tested 2026-05-08 PM. Per-Compound LR Δ +0.11 bp; per-
  Compound flat isotonic Δ −1.78 bp; rain-row meta blend Δ +0.03 bp.
  WET segment too small (n=1,355) for any routing variant to fire.
  W1 (rain residual) survives. `scripts/probe_lane3_routing.py`,
  `scripts/artifacts/probe_lane3_routing.json`.
  `[owner: research-model-extensions-Ibwvn | status: null]`

## 3. Target reformulation upstream of K=21 pool

- **3a.** Within-stint relative-progress as TARGET (not feature).
  d13 G1 was a feature-add NULL; reframe as target variable.
  `[owner: ml-handover-alignment-xvUN0 | status: done]`
  Result: stint_progress LGBM (TyreLife/max-stint) std OOF 0.65,
  ρ=0.252 (most-diverse single base ever), but K=21+1 alone NULL;
  joint with inv_laps adds only +0.10 bp. Concept lives in pool
  via inv_laps_until_pit.
- **3b.** Single-task GBDT on transformed targets (NEW FINDING).
  `[owner: ml-handover-alignment-xvUN0 | status: done]`
  Result: **inv_laps_until_pit** LGBM regression on 1/(1+laps_until_pit);
  K=21+1 alone +1.899 bp OOF, ρ=0.99392. Path B over K=22+inv_laps
  Compound×Stint τ=20k OOF 0.95110 (+2.75 bp vs PRIMARY). LARGEST
  OOF advance of session. HELD; submission decision pending.
  Predicted LB band +1.25 to +32 bp depending on Path B amp factor.
- **3c.** Other single-task target reformulations untested.
  `[owner: unclaimed | status: open]` Untried framings:
  pit_horizon_multiclass (4-class horizon bucket); reverse cumcount
  of pits in race; stint_index_within_race.

## 4. External data

- **4a.** Pirelli pit-window scrape. Tier-2 highest absolute EV per
  Day-8 research. Compound × track × season metadata.
  `[owner: unclaimed | status: open]`
- **4b.** debashish311601/formula-1-official-data-19502022 historical-
  priors aggregate join. Untested by us; deployed by Rozen at LB
  0.95354. Aggregate-key (Driver/Constructor/Circuit) join — synthetic-
  augmentation perturbation doesn't bite at join level (vs FastF1 H2
  which capped at 0.55 TyreLife correlation on row-level join). 60%
  synthetic D### still won't match. Career-level priors (1950-2022)
  outside any K=27 base's TE window. PI sealed band −2 to 0 bp;
  agent BOTE +0.5 bp midpoint band (−1, +0.5, +2). Cost ~45 min CPU.
  Harness verdict **SKIP** at 0.20 bp expected / 0.004 bp/min; PI
  agreed pre-flight. Closed null-by-pre-flight; calibration logged
  to `audit/decisions.jsonl` 2026-05-07. Friction tag candidate
  `external-data-axis-closed-by-pre-flight-when-pi-and-harness-agree`.
  `[owner: ml-competition-analysis-rwD3f | status: null-pre-flight]`

## 5. Final-3-day-window strategy (R5 HEDGE)

- **5a.** d13e Compound×Stint τ=20k vs τ=100k HEDGE. Flip ratio
  55/98 < 200; HEDGE-eligible per R7. Decide at start of final
  window.
  `[owner: unclaimed | status: open]`
- **5b.** path_b_K22_invlaps τ=20k as PRIMARY-replacement candidate.
  OOF 0.95110, ρ=0.99753 vs PRIMARY, 53% rows differ >1e-3, flip
  ratio 0.594. Submission held pending PI decision.
  `[owner: unclaimed | status: open]`
- **5c.** path_b_K22_invlaps τ=100k as HEDGE candidate (asymmetric
  flips 45/189 echoes d13 Stint Path B which lifted +7 bp despite
  G3 fail). Hold for final-window decision.
  `[owner: unclaimed | status: open]`

## 6. Single-model path (PI hypothesis P1)

External Kaggle notebook `romanrozen/f1-pit-driver-race-year-encoding-0-95354`
publishes a single LGBM at OOF AUC **0.95241** (and single XGB 0.95232,
RealMLP-A 0.95260). Recipe: ~118 engineered features (tyre/compound/
race-progress/lag-rolling/within-stint/combo-cats) + CV target encoding
of (Driver×Race×Year), (Driver×Race), (Driver×Compound), (Race×Compound),
(Race×Year), (Driver×Race×Compound). Their final blend lands LB **0.95354**.

- **8a.** Replicate Rozen single LGBM with our 5-fold StratifiedKFold(seed=42).
  Variants: raw_only / feA / feA_te / feA_te_orig × {Rozen-hparams, project-hparams}.
  Expected OOF ≈ 0.952; +160 bp over PRIMARY 0.95059 if LB transfer holds.
  `[owner: read-kaggle-handover-rsi2Q | status: wip]`
- **8b.** Single CatBoost with the same recipe (Rozen OOF 0.95127). HEDGE.
  `[owner: unclaimed | status: open]`
- **8c.** Single RealMLP (PyTabKit RealMLP_TD) with Rozen MLP_PARAMS;
  Rozen OOF-A 0.95260 / OOF-B 0.95259. Needs Kaggle GPU.
  `[owner: unclaimed | status: open]`
- **8d.** Add the new single-model OOF as a 23rd base in K=22 + Path-B
  hier-meta to test if the ~+200 bp standalone signal also amplifies stacked
  (separate hypothesis from 8a; depends on 8a passing).
  `[owner: unclaimed | status: open]`

## 7. Pool composition surgery

- **6a.** Replace 3 most leakage-eating GBDTs with FM-class bases.
  cb_slow-wide-bag (-17 GKF rank), e5_optuna_lgbm (-13 rank).
  Risk: public-LB row-iid leak-eaters carry signal (d13c T2/T3).
  `[owner: unclaimed | status: open]`

## 8. Gauge p_synth (overnight research sweep, 2026-05-06/07)

Umbrella: translate "what is the synthesizer's learned p(X,y)" into
prediction signal. 5 phases × 19 probes. CPU-only. 0 submits. Audit at
`audit/2026-05-07-overnight-gauge-p-synth.md`.

- **7a.** Diagnostic measurement of synth↔orig divergence — DONE.
  SDV overall 0.803; class-conditional structure SHARPER in synth than
  orig (Stint y0-vs-y1 KS 0.43 synth vs 0.24 orig — synthesizer
  strengthened y-conditional structure); 2023 lowest divergence (mean-KS
  0.094); HARD compound lowest (0.068). Most-corrupted joints are
  LapNumber × {RaceProgress, LapTime_Delta, Cumulative_Degradation}.
  `[owner: autoencoder-synthetic-data-pEMB6 | status: done]`
- **7b.** Density ratio r̂(x)=p_synth/p_orig — DONE (Driver/Race
  excluded after v1 hit AUC 0.9985 from ghost-Driver tells).
  AV-AUC 0.844 over natural joint. r̂(x) as feature K=21+1 NULL; as
  sample weight (P2.3) K=21+1 +0.78 bp PASS; as cohort router (P2.4)
  K=21+1 +1.32 bp PASS. New friction
  `density-ratio-routes-or-weights-but-fails-as-feature`.
  `[owner: autoencoder-synthetic-data-pEMB6 | status: done]`
- **7c.** Generative model on orig → log p_orig — DONE.
  GMM 16-comp single-feat AUC 0.759, ρ=0.503 (most-diverse single base
  ever); K=2 gate NULL. 4th confirmation of `rho-alone-insufficient-for-meta-utility`.
  BGMM oversmoothed at reg_covar=1.0 (AUC 0.55, near-random); skip
  sklearn BGMM in future.
  `[owner: autoencoder-synthetic-data-pEMB6 | status: done]`
- **7d.** Orig-transfer feature-subset diversification — **DONE, KEY
  WIN**. 4 variants; all 4 PASS K=21+1 gate:
    - **continuous_only +3.33 bp** (LARGEST single-base K=21+1 of session,
      beats inv_laps +1.90 by 1.75×)
    - no_laptime +1.87 bp
    - no_tyrelife_rp +0.86 bp
    - categorical_only PASS via meta-stack
  ρ continuous_only vs PRIMARY 0.9946. Mechanism: orig-LGBM restricted
  to features the synthesizer left marginal-aligned (TyreLife KS=0.017,
  Position KS=0.019). Refines friction
  `external-data-arch-bag-redundant-when-shared-training-data`: arch
  variation redundant, FEATURE-SUBSET variation is not. K=22 Path B
  Compound×Stint τ-sweep submission candidate in flight.
  `[owner: autoencoder-synthetic-data-pEMB6 | status: done]`
- **7e.** Path B with r̂ / log p_orig as cohort axis — null. 6/7 variants
  done (P5.3 Compound×r̂_q5 crashed on single-class segment); all r̂_q5
  and logp_q5 cohort axes regress -3 to -4 bp vs PRIMARY. **CAVEAT**:
  Phase 5 ran on K=14 sub-pool (only 14 of 21 named bases existed under
  exact filenames searched), so gap is partly missing-bases artifact.
  New friction `path-b-on-pool-subset-conflates-cohort-axis-with-pool-size`.
  Re-test on full K=21 pool TODO.
  `[owner: autoencoder-synthetic-data-pEMB6 | status: null]`
- **7f.** d17 Phase-A composition gate: K=23/K=24 stack-add of d16
  Phase-4 winners (continuous_only / no_laptime / no_tyrelife_rp /
  categorical_only) ± cross-branch strict inv_laps.
  `[owner: read-handover-62BCt | status: done]` Result: all C1-C7 LR-meta
  combos TIE/regress vs current d16 PRIMARY; C7 (K=24) +0.81 bp OOF /
  ρ_test 0.99506 / pred LB Δ −0.69 bp. 5th cross-confirmation of
  `path-b-amp-only-fires-on-meta-arch-not-base-add`.
- **7g.** Iterative chain-decomposition of P(X) on orig (E1).
  `[owner: reverse-engineer-data-generation-Hu8EK | status: done]`
  Result: d18 v1 K=21+1 **+7.365 bp** (largest single-base of session).
  Combined with d16+d18+E2+F2 + main's v4+h1d → K=27 Path-B τ=100k
  LB **0.95368** (NEW PRIMARY +1.4 bp over 0.95354).
- **7h.** Sequence-level DGP fingerprinting (NEW; biggest remaining
  blind spot). HMM on per-Year Compound transition matrices + AR(1)
  on within-stint TyreLife; per-(Driver,Race,Year)-group sequence
  log-likelihood under orig's transition model. Run-length stats per
  group. Untested mechanism layer (v4 treats rows i.i.d. internally).
  Predicted +1-3 bp K=21+1. Cost ~2-3 h CPU.
  `[owner: unclaimed | status: open]`
- **7i.** Cross-feature joint mode-id (extends G/H). Mode-tuple as
  GAN's discrete latent VECTOR; frequency-table comparison orig vs
  synth + cluster-id (k-prototypes) + EB(cluster).y_mean. Cost ~30 min CPU.
  `[owner: unclaimed | status: open]`
- **7j.** Membership inference + exact-row copy detection. Per synth
  row, min-distance to orig over all 16 columns; below ε, predicted
  P(y=1) = orig's actual y (leak-free). Cost ~1 h CPU.
  `[owner: unclaimed | status: open]`
- **7k.** Class-conditional CTGAN replay with explicit cond-vector
  spec [PitStop, Compound, Stint, Year]. Cost ~3 h Kaggle GPU.
  `[owner: unclaimed | status: open]`
- **7l.** Per-Year DGP heterogeneity / specialists test under K=27
  pool. Cost ~30 min CPU.
  `[owner: unclaimed | status: open]`

## 9. Day-17 PM strategy-critic top-3 hypotheses (Rule 14 + Rule 7)

PI directive 2026-05-07: "small probes don't matter. Revisit
problem-solving loop. Find focus." Audit at
`audit/2026-05-07-d17-strategy-critique.md`.

- **9a.** H1 — Yekenot RealMLP recipe replication.
  `[owner: read-handover-62BCt | status: null]` 3 variants all NULL at
  K=22 meta. Promoted to 9d. Audit `audit/2026-05-07-d17-h1-verdict.md`.
- **9d.** Full yekenot RealMLP recipe replication.
  `[owner: read-handover-62BCt | status: done-CONFIRMED-WIN]` Standalone
  5-fold OOF 0.95257 (matched yekenot pub 0.95273 within 1.6 bp at n_ens=4).
  K=24 d18pool+h1d → SUBMITTED LB 0.95345 = AT top-5% threshold.
- **9b.** H2 — FastF1 / Ergast external join. **Match rate 1.42% only**.
  Friction `synthetic-augmented-driver-codes-cap-external-data-coverage`.
  `[owner: read-handover-62BCt | status: null]`
- **9c.** H3 — ID-shift / row-position structural probe.
  Sparse-LR base OOF 0.50039 (chance).
  `[owner: read-handover-62BCt | status: null]`

## 10. Virgin axes complement to HANDOVER T1–T4 (Day-16 RESOLVED)

Day-15 PM Conn–McLean re-entry. T1–T4 (combine-bases / target-reform /
DAE-variants / meta-arch redesign) is owned by other branches; this leaf
covered the orthogonal axes from the d13 problem decomposition tree
(α/β/δ/ε/ζ/η). Day-16 executed all 9 candidates. **All 9 NULL,
falsified, parked, or killed.** Full audit
`audit/2026-05-16-d16-virgin-axes-results.md`.

- **7a.** α4 GRU sequence on (Driver, Race) lap windows. Std OOF
  0.93066, ρ=0.919 (most-diverse base of session). K=22+1 LR-meta
  Δ=-0.043 bp NULL. **5th cross-confirmation of `lr-meta-rank-lock-strong-anchor`.**
  Friction `temporal-axis-also-rank-locked-at-K22`.
  `[owner: read-handover-lA8Nr | status: null]`
- **7b.** ε2 twin parallel-pool 2-meta blend. ρ(metaA, metaB)=0.967
  real disagreement; top-level LR vs single LR-meta(K=11):
  **FALSIFIED Δ -1.79 bp.** Friction `twin-pool-2-meta-collapses-rank-info`.
  `[owner: read-handover-lA8Nr | status: null]`
- **7c.** ε3 K=5 pool on aadigupta1601 original. **PARKED**:
  reconsidered redundant per `external-data-arch-bag-redundant-when-shared-training-data`.
  `[owner: read-handover-lA8Nr | status: parked]`
- **7d.** η1 Year=2023 ∩ rare-Driver hard-mask. Best K=5 +0.004 bp
  ceiling NULL. PRIMARY routes 2023 rare rows to near-zero already.
  Friction `primary-hier-meta-globally-calibrated`.
  `[owner: read-handover-lA8Nr | status: null]`
- **7e.** β3 ROC-Star / rank_xendcg. **PARKED** (LightGBM rank
  objectives ill-fit — LambdaRank already failed -86 bp; ROC-Star
  needs custom PyTorch impl, GPU was occupied with H1 GRU).
  `[owner: read-handover-lA8Nr | status: parked]`
- **7f.** δ2/3 conformal isotonic 4 schemes (inner-CV-validated).
  All schemes regress -2.5 to -9.6 bp NULL. PRIMARY hier-meta
  globally well-calibrated. Friction `primary-hier-meta-globally-calibrated`.
  `[owner: read-handover-lA8Nr | status: null]`
- **7g.** α5 two-stage stint (logistic stage-2). Std OOF 0.625 NULL —
  stage-2 1-D logistic too restrictive. Methodological miss; α5 axis
  not falsified. Friction `two-stage-stint-needs-richer-stage-2`.
  `[owner: read-handover-lA8Nr | status: null]`
- **7h.** ζ3 decoded-data KNN nearest-orig PitStop label. **PARKED**
  (reconsidered redundant w/ orig_transfer).
  `[owner: read-handover-lA8Nr | status: parked]`
- **7i.** ε4 DeepGBM. cat-LGBM stage-2 KILLED 16 min over-engineered
  (627 cats × num_leaves=255). Lean variant ε4b (sparse-LR head,
  9300-dim) fold-0 AUC 0.92507 weak; KILLED ~20 min/fold.
  `[owner: read-handover-lA8Nr | status: null]`

**Bonus probes added during execution:**
- **7j.** ζ6 transductive full-test pseudo (lean LGBM, 500 boost / 31
  leaves, half-weight pseudo on all test rows). Std OOF 0.93433,
  ρ=0.872. K=22+1 LR-meta Δ +0.631 bp PASS at LR-meta-K22; **Δ -0.30
  bp regress vs PRIMARY hier-meta** (Path-B-amp doesn't fire on
  base-add). Marginal HEDGE candidate. Friction
  `h9-transductive-pseudo-lifts-LR-meta-but-not-PRIMARY-hier`.
  `[owner: read-handover-lA8Nr | status: marginal]`
- **7k.** Multi-add gates K=22+2: H9+H2 +0.671 / H9+GRU +0.629 bp ≈
  H9 alone (+0.631). Friction `lr-meta-multi-add-no-better-than-single-add`.
  `[owner: read-handover-lA8Nr | status: null]`

## 12. Untested-mechanism sweep (2026-05-08 night)

Origin: PI directive 2026-05-08 night "what shall we experiment now? we
have a lot of time, you can take all night". Initial plan was three-
pronged (FastF1 weather aggregates, per-segment LGBM head, sequence
transformer with gap embedding); all three returned SKIP at probe.py
BOTE (cost-efficiency 0.001-0.009 bp/min vs 0.01 DEFER threshold).

RES-1 public-notebook + Kaggle discussion forum re-scan (Rule 22, last
scan 8 days stale) replaced the initial plan with a stronger one. Pulled
top kernels by score and votes; 4 modeling kernels dominate the 0.954+
public LB:

- **masayakawamata/s6e5-stacking-vibe-coding** — 2-layer GPU stack:
  L0 = (yekenot RealMLP + own LogReg-FE) base OOFs;
  L1 = (XGB / LGBM / CatBoost / Torch-MLP) stackers;
  L2 = cuML L2 LogReg over (L0 + L1) OOFs.
  89 variants in 0.9543-0.9545 range. Best: mlp_32x16 at 0.954471.
- **svanikkolli/f1-lap-by-lap-prediction-engine-v3** — internal 5-model
  stack (LGB + XGB-Optuna + 3 RealMLP variants) → stacking → 5-way
  rank-blend with 4 public notebooks.
- **flexonafft/f1-submission-blender-0-95418** — pure rank-blend of
  high-LB public submissions; output is `0.95 * anchor + 0.05 * support`.
- **sohailkhanlml/f1-pit-stop-blending-the-leaderboard-0-95418** —
  same pattern; aggregates external public submissions.

Our K=4 PRIMARY rank-correlates 0.991-0.996 with every 0.954+ public
file pulled (mean / std distributions are nearly identical: ours
0.199/0.302 vs public-super-0.95418 at 0.204/0.307). PI declined the
blend route on principle (original work only); approved full
replication of the masayakawamata 2-layer stack architecture (Prong S)
and per-segment LGBM head (Prong B) as parallel CPU jobs.

- **12a.** ~~Prong A — FastF1 weather aggregates~~ DROPPED (BOTE 0.001
  bp/min SKIP; lower EV than emerging Prong S/B/blend mechanisms).
  `[owner: ml-model-experiments-gbKiI | status: parked]`
- **12b.** ~~Prong B — Per-segment LightGBM head replacing inner LR
  in Path-B Compound × Stint.~~ PARKED 2026-05-09 — PI redirected
  the night to original creative work (see leaf 13); Prong B not
  executed. Could be revisited if a leaf without a transductive
  base is needed.
  `[owner: ml-model-experiments-gbKiI | status: parked]`
- **12c.** ~~Prong D — Sequence transformer with explicit gap embedding~~
  DROPPED (BOTE 0.006 bp/min SKIP; A32 narrowing makes gap-aware
  attention unlikely to break rank-lock; superseded by Prong S).
  `[owner: ml-model-experiments-gbKiI | status: parked]`
- **12d.** ~~Prong S — 2-layer stack architecture replication on our
  base OOFs.~~ PARKED 2026-05-09 — PI redirected to original work
  ("do not replicate or copy"). Replication-only intent doesn't fit
  the directive. Falsified-by-redirection rather than by execution.
  `[owner: ml-model-experiments-gbKiI | status: parked]`
- **12e.** ~~Prong P — blend with public submissions~~ DROPPED
  (PI declined on principle; original work only).
  `[owner: ml-model-experiments-gbKiI | status: parked]`

## 13. Sequence-coupled meta + transductive base (2026-05-09 night)

Origin: PI directive 2026-05-08 night — "show me you can do original
work, investigate, be creative, work all night autonomously, surprise
me, do not replicate or copy." Agent identified the hidden
row-independence premise in A29's rank-lock framing and tested
sequence-coupled meta features (V1-V3) and transductive bases
(V4-V6).

Findings: rank-lock at K=4 is at the **conditional-target-correlation
level** given row data, not just the logit-direction level. Refines
A29. Tree-base ingestion of transductive label info DOES extract a
small lift; linear/RankNet/LGBM-meta absorption blocks it.

- **13a.** V1 sequence-coupled meta (look-ahead/behind, session
  aggregates, permutation-disagreement). LR row-local 0.95400, LR
  full 36-feat 0.95401 (+0.03 bp WEAK), RankNet 0.95399 (-0.12 FAIL).
  Loss-function variation doesn't escape the ceiling.
  `[owner: ml-model-experiments-gbKiI | status: null]`
- **13b.** V2 sharpenings: LR+interactions TIE (0.95400), residual-LR
  REGRESS (-3.43 bp), LightGBM-meta REGRESS (-0.59 bp). Tree-meta on
  K=4+seq-coupled overfits.
  `[owner: ml-model-experiments-gbKiI | status: null]`
- **13c.** V3 LR meta + kNN-target-mean (transductive label feature
  at meta) — bare y_knn AUC 0.898 standalone, +0.01 bp at meta NULL.
  `[owner: ml-model-experiments-gbKiI | status: null]`
- **13d.** **V4 kNN-augmented LightGBM base** (transductive label at
  base via tree splits). Standalone 0.94163, K=4+1 LR meta +0.24 bp,
  K=5 + Path-B C×S τ=100k OOF 0.95405, **public LB 0.95359** (+0.8 bp
  vs K=4 PRIMARY). Submitted (Rule 27 override authorised). PI
  directive 2026-05-09: keep K=4 as PRIMARY; K=5 file held as +0.8 bp
  hedge candidate.
  `[owner: ml-model-experiments-gbKiI | status: done]`
- **13e.** V5 V4 + multi-cell target encoding (5 cells, alpha=50
  smoothing). Standalone +6.10 bp over V4 but at meta only +0.07 bp
  (V5 replaces V4) / +0.06 bp K=6. ρ_spearman(V5, V4) = 0.989 —
  TE absorbed by V4's logit direction.
  `[owner: ml-model-experiments-gbKiI | status: null]`
- **13f.** V6 V4 + task-learned-embedding kNN (sklearn MLP 64→32→16,
  per-fold encoder, BallTree on penultimate-layer embedding).
  Standalone +3.75 bp over V4 but at meta only +0.18 bp (V6 replaces
  V4) / -0.02 bp K=6. ρ_spearman(V6, V4) = 0.988 — same absorption
  as V5 despite different similarity metric. Transductive-feature
  mechanism is one-dimensional in K=4 logit space.
  `[owner: ml-model-experiments-gbKiI | status: null]`

## 11. Multi-model FE testing campaign (2026-05-08 PM, late)

Origin: PI directive after FE research synthesis. 16 candidate FE
picks queued in `EXPERIMENTS-NEXT.md` Tier-A2 + Tier-A3 from three
audit notes (`audit/2026-05-08-fe-research-{survey,code-grounded,
extended}.md`). Plan at `/root/.claude/plans/now-carefully-plan-how-
polished-dewdrop.md` (PI-approved). Funnel: Stage 0 BOTE → Stage 1
LR ceiling probe → Stage 2 LGBM smoke → Stage 3 LGBM 5-fold OOF +
80/20 holdout audit → Stage 4 K=4+1 stack-add gate. Only the TOP-1
survivor goes through full multi-model build-out (LR + LGBM + CB +
RealMLP + HGBC, including Kaggle GPU per PI authorisation 2026-05-
08). EXP-A2-8 stack-meta architecture probe runs as an independent
track regardless of FE-add outcomes.

- **11a.** Phase 0 — BOTE pre-flight for ~13 picks. Done.
  `[owner: research-feature-engineering-7oCmj | status: done]`
- **11b.** Phase 1-4 — funnel execution. 7 picks tested in smoke;
  A2-2 (mandatory_compound_rule) only winner @ +9.3 bp smoke,
  +1.4 bp full 5-fold. Phase 4 K=4+1 plain LR-meta gate: +0.302 bp
  (below +0.5 PASS); G3 flip 0.195 asymmetric; TIE_EXPECTED.
  Phase 4b Path-B C×S τ=100k amp test: K=4+A2-2 OOF 0.95405 vs
  K=4 PRIMARY 0.95403 (Δ +0.26 bp); ρ 0.999893 (TIE_EXPECTED).
  WEAK verdict; below strict gate. No survivor for Phase 5a.
  `[owner: research-feature-engineering-7oCmj | status: null]`
- **11c.** Phase 5a — multi-model build-out for TOP-1 survivor.
  Skipped: no Stage-4 survivor.
  `[owner: research-feature-engineering-7oCmj | status: parked]`
- **11d.** Phase 5b — EXP-A2-8 LightGBM stack-meta. Tested 2026-
  05-08 PM: 43 meta features (4 P + 4 logits + 4 ranks + 6 pairwise
  products + 6 abs-diffs + 6 logit-diffs + 8 raw side-info + 5
  Compound one-hot). 5-fold OOF 0.95390 vs Path-B PRIMARY 0.95403
  (Δ −1.30 bp); also below plain LR 0.95399 (−0.96 bp). Fold-std
  0.00080 elevated (range 0.95309–0.95506). Tree stacker overfits
  interaction noise on K=4 pool; convex LR + Path-B partial-pooling
  regularize better. FAIL.
  `[owner: research-feature-engineering-7oCmj | status: null]`
- **11e.** Phase 6 — submission decision. No surviving lift; no
  submission. PRIMARY remains K=4 + Path-B C×S τ=100k @ LB 0.95351.
  `[owner: research-feature-engineering-7oCmj | status: null]`

---

## Falsified or dead (do not re-claim)

- Day-16 leaf 7 — all virgin axes from d13 problem-decomposition tree
  (α/β/δ/ε/ζ/η) are rank-locked at K=22 + Path-B-hier-meta. 4 NULL
  (H4 mask, H7 isotonic, H10 stint logistic, H1 GRU sequence at
  ρ=0.919 most-diverse), 1 falsified (H2 twin-pool -1.79 bp), 3
  parked (orig pool, ROC-Star, KNN-orig), 2 killed (ε4 cat-LGBM,
  ε4b lean), 1 marginal (H9 transductive +0.63 LR-meta but -0.30
  vs PRIMARY hier). Multi-add gates ≈ single-add. Full audit
  `audit/2026-05-16-d16-virgin-axes-results.md`.
- Time-to-event LGBM variants (d12 T1.2 4-of-4 dead).
- Year-segmented specialist + AV-reweight (d12 falsified; AV-AUC 0.502).
- LambdaRank meta (d12 -86bp).
- AUC-pairwise XGB base (d12 -451bp fold-0).
- 6-shape FM partition variants (Day-13 PM saturated across shapes).
- Drop-GBDT leak-eaters (d13c T2/T3 falsified -2.5/-2.6bp Strat).
- Single-base FE additions in LGBM/FM class (d13/d14 alt-axis G1/G2'/G3/H1
  4-of-4 NULL).
- FM-field-augmentation beyond 12 fields (Move D / d14 FM_aug16): +20bp
  standalone but min-meta -0.07bp FAIL. New input types (PitWindow, HazardDecay,
  CompoundPressure, RaceStage) confirm saturation — FM interactions already
  cover this signal space. Dead across aug13/aug16.
- TabPFN fine-tuning v2.5 + v2.6 (see 1a above).
- Path B cohort sweep on Year, Year×Stint, Race axes (d14 9 variants NULL).
- Hazard NN (d9 -315bp; main-branch leakfree confirmed dead OOF 0.92013).
- 2-level stacking via meta-as-base (2026-05-06: K=22 + d12_lr_meta
  +1.348 bp OOF, but K=22 Path B Compound×Stint τ=100k SUBMITTED
  LB 0.95045, regressed -4 bp; Path B amp does NOT fire on
  meta-derivative additions per friction tag
  `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`).
- Masked-column self-prediction / DGP-residual features (2026-05-06,
  Day-14: SAINT/TabNet/VIME class). 4 LGBM regressors → z-residuals +
  L1 anomaly as 5 new LGBM features. Std OOF 0.94200 (-88bp), K=2
  min-meta -0.025bp NULL, K=22 add +0.172bp at ρ=0.9958 (pred LB
  -1.3bp). Load-bearing diagnostic: OOF RMSE ≈ marginal σ for all 4
  targets — synthetic NN-DGP is conditionally near-independent within
  rows. Friction tag `synthetic-dgp-conditionally-near-independent`.
  Joint-explains FM-aug12 saturation, Day-13/14 alt-axis 4-of-4 NULL,
  TabPFN 0.944 ceiling. See `audit/2026-05-06-d14-dgp-residuals.md`.
- KD-distilled-LGBM-of-K=21-meta (2026-05-06: meta-derivative class;
  +0.526 bp OOF predicted same LB-regress pattern; held).
- NN-with-embedding-layers single-base (2026-05-06: ρ=0.918
  most-diverse measured but K=21+1 -0.025 bp NULL; ρ alone
  insufficient to clear meta-absorption ceiling).
- Lap-mod / id-mod features in LGBM (2026-05-06: 566 bp marginal
  span on LapNumber_mod_10 absorbed by existing GBDT interactions;
  K=21+1 +0.002 bp NULL).
- Confidence-extreme pseudo-cascade (2026-05-06: K=21+1 +0.019 bp NULL).
- K=21 simple aggregators (mean/gmean/rank_mean/trimmed) (2026-05-06:
  -19 to -32 bp standalone; LR meta with [raw,rank,logit] expand
  is doing real work, simple aggregators ruled out).
- α-calibrated τ-resweep on PRIMARY (2026-05-06: τ=20k unchanged at
  ρ=1.0; Bayesian-correct asymmetry is not a fixable LB cap).
- Driver-cluster Path B cohort axis (2026-05-06: -0.4 to -0.9 bp NULL).
- Within-Race quantile of LapTime_Delta (2026-05-06: K=21+1 +0.20 bp
  NULL/marginal; +922 bp single-feat leak signal absorbed by pool).
- Year×Stint as sparse-LR feature (2026-05-06: K=21+1 +0.05 bp NULL
  despite ρ=0.844; structurally diverse but signal redundant).
- Multi-target NN with shared trunk (2026-05-06: K=21+1 +0.086 bp NULL;
  auxiliary inv_laps head ineffective at meta gate vs single-task NN).
- TE fold-leak audit on d2a/d3a (2026-05-06: CLEAN; no leakage; OOF
  discipline correct).
- Path-B alternative segmentation crosses on K=24 (Day-18, branch
  `claude/logistic-regression-ensemble-0PNkA`). Compound × Year
  (d18) ALL τ ≤ baseline (-2.05 to -0.01 bp); 3 alt axes (d18b)
  Driver_cluster × Stint best τ=100k +0.36 bp <gate, Race_class ×
  TyreLife_q5 best +0.01 bp NULL, Position_q5 × Compound best +0.00
  bp NULL. Friction `pathb-amp-dead-when-pool-already-routes-segmentation-variable`:
  K=24 has cb_year-cat → Year axis dead; d16/v3/v4 carry Position
  continuously → Position×Compound dead. The amp-eligible Path-B
  axis on this pool is uniquely (Compound × Stint). HANDOVER A4
  "alternative seg crosses" closed null. See
  `audit/2026-05-07-d18-pathb-compound-year-result.md` and
  `audit/2026-05-07-d18b-pathb-alt-axes-result.md`.
- LR-bank diversity-via-FE alone is bounded (Day-17 PM, branch
  `claude/logistic-regression-ensemble-0PNkA`). 15-variant LR bank
  + 5 rich-FE variants, lr_mega ceiling OOF 0.92776; LR-bank
  eff_rank 2.0-2.19; combined GBDT+LR pool eff_rank 3.33 (only
  +1.3 directions per LR class). K=24 + mega LR sweep all NULL or
  regress (-0.89 to -36.7 bp). Random subspace REDUCES diversity
  to eff_rank 1.67. Friction `lr-eff-rank-bounded-at-2-by-pipeline-not-base-class`.
  Per-segment mega LR (Probe-5) +60.8 bp standalone (Compound×Year)
  but transfer to meta on K=24 = NULL — friction
  `per-segment-mega-LR-fires-only-at-LR-class-not-meta-class`.
  See `audit/2026-05-07-lr-leverage-six-probes.md` and
  `audit/2026-05-07-simple-lr-playbook.md`.

---

## Re-decomposition trigger

Rewrite this file when ANY of:
- All open leaves resolved (`done` / `null`), OR
- Plateau ≥3 days at unchanged PRIMARY LB, OR
- Saturation count ≥5 at same LB, OR
- 50% comp checkpoint, OR
- PI says "redecompose" / "new issue tree".

The agent that owns the rewrite archives this file to
`audit/archive-YYYY-MM-DD-issues-pre-redecomp.md` before replacing.
