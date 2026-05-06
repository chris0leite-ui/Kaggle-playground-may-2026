# ISSUES.md — live problem decomposition

> BPS step-2 (disaggregate) + step-3 (prioritise) output. Refreshed
> when the strategy-critic-loop fires (plateau, saturation, kickoff,
> 50% checkpoint). Cap ≤150 lines.

**Problem (BPS step 1).** Close the −28.6bp gap to top-5% (leader
LB 0.95345 vs ours 0.95059 = d15b_path_b_K22_dae_only_tau20000) in
12 remaining days. Day-15 confirmed that orthogonal new-base additions
land at +0–1bp LB regardless of standalone diversity (DAE-class ρ
0.9477 standalone → +1bp LB; new friction
`path-b-amp-only-fires-on-meta-arch-not-base-add`). Day-16+ priority
is META-ARCH REDESIGN axis (non-Gaussian shrinkage, Yao/Vehtari
covariance-Σ BMA, alternative segmentation cross), which IS
Path-B-amp-eligible. Pending PI decision: submit
`path_b_K22_invlaps τ=20k` (OOF +2.75bp, predicted ~+4bp LB).

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

## 6. Pool composition surgery

- **6a.** Replace 3 most leakage-eating GBDTs with FM-class bases.
  cb_slow-wide-bag (-17 GKF rank), e5_optuna_lgbm (-13 rank).
  Risk: public-LB row-iid leak-eaters carry signal (d13c T2/T3).
  `[owner: unclaimed | status: open]`

## 7. Gauge p_synth (overnight research sweep, 2026-05-06 PM)

Umbrella: translate "what is the synthesizer's learned p(X,y)" into
prediction signal. Three threads × 19 probes. CPU-only. 0 submits.

- **7a.** Diagnostic measurement of synth↔orig divergence (5 probes:
  SDV QualityReport, marginal KS/chi-sq, pairwise chi-sq grid, class-
  conditional divergence, per-stratum divergence).
  `[owner: autoencoder-synthetic-data-pEMB6 | status: wip]`
- **7b.** Density ratio r̂(x)=p_synth/p_orig (4 probes: AV-AUC + SHAP,
  r̂ as feature, r̂-weighted orig_transfer, r̂-median split).
  `[owner: autoencoder-synthetic-data-pEMB6 | status: wip]`
- **7c.** Generative model on orig → log p_orig(x_synth) feature
  (3 probes: GMM, KDE, MAF normalizing flow).
  `[owner: autoencoder-synthetic-data-pEMB6 | status: wip]`
- **7d.** Orig-transfer feature-subset diversification (4 probes:
  drop-LapTime, drop-TyreLife/RP, cat-only, cont-only).
  `[owner: autoencoder-synthetic-data-pEMB6 | status: wip]`
- **7e.** Path B with r̂ / log p_orig as cohort axis (3 probes:
  r̂_q5, log p_orig_q5, Compound × r̂_q5 cross). Amp-eligible per
  `path-b-amp-only-fires-on-meta-arch-not-base-add`.
  `[owner: autoencoder-synthetic-data-pEMB6 | status: wip]`

---

## Falsified or dead (do not re-claim)

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
