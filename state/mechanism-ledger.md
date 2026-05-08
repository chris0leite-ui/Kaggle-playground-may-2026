# Mechanism ledger

The full enumeration of every mechanism family the team has tried. One
line per family; the result, with descriptive names instead of letter-
number codes.

For the original short-codes, see `glossary.md`. For the full per-probe
audit notes, see `audit/`.

## Baseline pool (Days 1-3)

- **Plain LightGBM on raw features** — baseline.
- **Out-of-fold target encoding** — d2a; the Day-3 closure on this
  family was premature; the load-bearing 3-way variant came back later.
- **XGBoost native categoricals** — m2.
- **CatBoost native categoricals** — e1 (row-subsample variant) plus
  three winners (cb_year-cat, cb_lossguide, cb_slow-wide-bag).
- **Relative-state feature engineering** — m4.
- **Logistic-regression stacker over 3-view expansion** (raw / rank /
  logit) — became the canonical meta-stacker.
- **Dirichlet random search over weights** — calibration baseline.

## Single-model winners pre-stacking

- **Histogram-gradient-boosting with label-encoded Driver** — e3;
  best single model before CatBoost.
- **L1 meta-coefficient pruning** — e2 / m5g / m5h; only the
  L1-coefficient prune preserved OOF.

## Rule-class and FM-class additions (Days 7-10)

- **Multi-rule residual learner** — d6 multi-rule; landed at LB 0.95026
  (+2.1 bp) as 18-base PRIMARY for several days.
- **Hashed logistic regression on 3-way crosses** — d9 / d9b L4; null.
- **Plain factorisation machine on 12 fields** — d9c FM; first FM-class
  base; LB 0.95029 as a 20-base swap.
- **Factorisation-machine partition** (driver-dynamics + race-laptime) —
  d9f; LB 0.95031 as a 21-base swap; correlation 0.487 was the most
  diverse base since the early baselines.
- **Augmented-FM with 12 fields** — d9h aug-12; LB 0.95034 (+3 bp;
  300× upside on a near-zero predicted gain).
- **Augmented-FM with 2-way fields** — d9i aug-2way; LB 0.95034.
- **Augmented-FM with 15 fields** — d13 H1 aug-15; standalone strongest
  ever, but 23-base add regressed at meta.
- **Field-aware factorisation machine** — d9e FFM; held; FFM doesn't
  beat plain FM in this setting.
- **Multi-FM with disjoint feature partitions** — d9f; the strong-est
  partition variant.
- **3-way concat-field FM** — d14 H1 aug-13; correlation 0.917 most
  diverse; min-meta −0.13 bp.
- **16-field FM** — d14 Move D aug-16; +20 bp standalone but min-meta
  −0.07 bp; saturation of the FM-aug axis.

## Per-segment shrinkage stacker (Days 13-19)

- **First per-segment Compound stacker, τ=100k** — d13; LB 0.95033
  (+2 bp).
- **Per-segment Stint stacker, τ=100k** — d13; LB 0.95041 (+7 bp;
  11.6× upside).
- **Per-segment Compound × Stint stacker, τ=20k** — d13e; LB 0.95049
  (+8 bp); old PRIMARY.
- **Compound × Stint, τ=100k** — held; +0.82 bp OOF; hedge.
- **GroupKFold stack rebuild audit** — d10b/c; FM-class lift +0.87 bp
  Stratified became +2.01 bp GroupKFold (2.3× amplification under
  leak-blocking).
- **Leak-corrected meta** — d10d; G3 flip ratio 0.001; held.
- **Per-Year specialist** — d12; falsified; AV-AUC 0.502 confirms
  i.i.d. test; year 2023 was easiest.
- **Adversarial-validation reweighting** — d12; min-meta −4.92 bp;
  train/test i.i.d.
- **LambdaRank race-meta** — d12; −86 bp regression (origin of Q6
  metric-mismatch rule).
- **Pairwise-AUC XGBoost base** — d12; −451 bp on fold 0 (origin of
  same Q6 rule).
- **Single-bag (5-seed) of HGBC** — d12; −19 bp every segment;
  K=21 complexity is justified.
- **GroupKFold full-pool meta** — d12; correlation 0.9914 with
  Stratified meta; rank-lock partial dissolution.
- **FM partition variants** (5/3, 4/4 CT-axis, 6/6 alt) — d13a/d; noise
  floor; partition saturated.
- **Move-C strat pool refactor** — d13c; T2/T3 −2.5/−2.6 bp falsified;
  leak-eaters carry signal.
- **Within-stint LightGBM features / cross-driver features** — d13
  G1/G2'; min-meta null.
- **Stint-grouped LambdaMART** — d13 G3; killed (63% all-zero stints).
- **Path-B cohort sweep** (Year, Year × Stint, Race × τ) — d14;
  9 variants all below PRIMARY OOF.
- **Two-level stacking (meta as base)** — K=22 LB 0.95045; falsified
  (meta-derivative class).
- **TabPFN v2.5 / v2.6** — Day 14; AUC ceiling 0.944; v2.6 ran out of
  memory on Kaggle P100.
- **FM with new input features** (Day 14 Move D) — aug-16 standalone
  +20 bp; min-meta −0.07 bp fail.
- **Masked-column self-prediction** (DGP-residuals) — d14; 5th per-row
  feature-engineering null; load-bearing diagnostic that the synthetic
  DGP is conditionally near-independent.

## Day-15+ exotic bases

- **Knowledge-distillation LightGBM** — Day 15 PM; +0.526 bp at meta
  (held; meta-derivative class).
- **Neural network with embedding layers** — Day 15 PM; correlation
  0.918 most diverse; K=21+1 −0.025 bp null (correlation alone fails).
- **Lap-modulo features (lap mod 10, etc.)** — Day 15 PM; +0.002 bp
  null (566 bp marginal absorbed by GBDT interactions).
- **Pseudo-label confidence-extreme** — Day 15 PM; null.
- **Within-race LapTime quantile** — Day 15 PM; null.
- **Year × Stint sparse logistic regression** — Day 15 PM; correlation
  0.844; null.
- **Blend aggregators** (mean / geometric-mean / rank / trimmed-mean
  on K=21) — Day 15 PM; −19 to −32 bp; the LR-meta does real work.
- **Driver-cluster per-segment cohort** — Day 15 PM; null across τ;
  cohort axis exhausted.
- **Alpha-calibrated τ resweep** — Day 15A; correlation 1.0 with d13e;
  τ=20k is empirically optimal.
- **id-order synth artefact probe** — Day 15 PM; the marginal target
  span is not predictive lift.

## Target reformulation (the leakage trap)

All four were inflated by target-construction-layer leakage:

- **inv-laps-until-pit** — Day 15 PM; OOF +1.899 bp inflated → +0.234
  bp strict (88% collapse). HELD. DO NOT SUBMIT.
- **stint-progress** — Day 15 PM; correlation 0.252 most diverse base
  ever, but K=21+1 null; same leakage.
- **multi-target NN with pit-aux head** — Day 15 PM; null.
- **path-B 22-base + inv-laps Compound × Stint** — Day 15 PM; OOF
  inflated by leakage; INVALIDATED.
- **pit-horizon (4-class)** — Day 15 PM; +3.191 bp inflated → +0.302
  bp strict (90% collapse).
- **reverse-cumulative pits** — Day 15 PM; +4.867 bp inflated → −0.005
  bp strict (100% collapse). The biggest leak.

The discovery of this whole leakage class is in
`audit/2026-05-06-target-reform-leakage-audit.md`. It is the origin
of Rule 24 (fold-safe label-conditional aggregates).

## Day-16 virgin-axes round (all null)

- **22-base original-data continuous-only Path-B, τ=20k** — Day 16;
  LB 0.95089 (+3 bp). PRIMARY-replace candidate, succeeded by Day 17.
- **Year=2023 ∩ rare-Driver hard mask** — null; PRIMARY already
  routes 2023.
- **Conformal isotonic per-bin recalibration** (4 schemes) — null
  to −9.6 bp; PRIMARY is globally calibrated.
- **Two-stage Stint logistic** — methodology miss (stage-2 starved
  of features).
- **Twin-pool 2-meta blend** — −1.79 bp; rank info collapses.
- **DeepGBM leaf-encoding** — killed (over-engineered + sparse-LR
  weak).
- **AV-probability sample-weighted LightGBM** — killed (12 min on AV
  stage; bounded by AV-AUC = 0.502).
- **Transductive pseudo-labeling 627k+** — +0.631 bp K=22 LR-meta but
  −0.30 bp vs hier-meta; marginal hedge.
- **GRU sequence model** — correlation 0.919 most diverse; K=22+1
  −0.043 bp null; 5th rank-lock confirmation from a structurally
  distinct angle.

## Day-17 single-model thesis

- **Single LightGBM kitchen-sink (Rozen recipe)** — Day 17; thesis
  falsified; v1 LB 0.94107 was leaky; v3 fold-safe OOF 0.94563 honest;
  ceiling roughly 0.946 = −52 bp from PRIMARY OOF.
- **CatBoost research-recipe (v3)** — Day 17 PM; OOF 0.94993; +12.06
  bp at K=21+1.
- **CatBoost yekenot transfer (v4)** — Day 17 PM; OOF 0.95200; +24.21
  bp at K=21+1 (double v3); biggest CB single-model lift; landed in
  PRIMARY.
- **23-base + v4 + h1d Compound × Stint, τ=100k** — Day 17 PM; PRIMARY
  before Day-18 supersession; LB 0.95354.

## Day-17 PM cross-row feature-state probes

- **Field-state cross-row aggregates** — per-(Race, Year, LapNumber)
  aggregates; cum-pits AUC 0.7972 highest single-feat on the comp;
  passed strict fold-safe audit; 24-base stack-add −0.015 bp null
  (6th rank-lock confirmation). NOT submitted.
- **Combined train+test lead/lag** — combined-frame premium evaporates
  at GBDT (−0.36 bp); Rule 25 PASS via AV-AUC 0.502.
- **Target-structure EDA** — P(target | lap-from-end) decays
  monotonically 0.272 → 0.061 over 10 laps; closes the
  reverse-engineer-the-target axis.
- **NTL single-rule baseline** — 5 NTL reconstructions + 13 thresholded
  rules; cap at AUC 0.687 < raw TyreLife 0.699; below useful.

## Day-18 LR-diagnostic expedition

- **LR-diagnostic suite (10 scripts)** — Day 18 PM on the K=24 pool;
  effective rank 2.88 of 24; forward-selection confirmed K=10 = K=24
  in OOF AUC. Promoted to skill.
- **Bagged-LR with stint-cross interactions** — A2; correlation 0.71
  with PRIMARY (lowest ever) but K=10+1 null.
- **Per-Compound LR specialists** — A4; OOF 0.87386; K=10+1 null;
  6th rank-lock confirmation.
- **K=10 forward-selected core, LR-meta** — T2; OOF −0.4 bp vs old
  K=24; no-cost simplification candidate; not LB-tested.
- **Path-B 3 alt segmentations × 3 τ on K=10** — T1#3; all 9 within
  sub-bp of K=10 baseline. Friction:
  per-segment-stacker-amp-requires-large-redundant-pool.

## Day-18 reverse-engineer-data-generation arc

- **Chain-decomposition base** (causal + Gaussian, original-DGP
  log-likelihood) — d18 v1; K=21+1 +7.37 bp (largest single-base of
  the arc); landed in PRIMARY.
- **Pre-image kNN** — d18 E2; kNN(K=10) per-Compound on 7 KS-low feats;
  passed; landed in PRIMARY.
- **Constraint-violation features** — d18 F2; 10 physical constraints;
  y=1 rows break within-stint TyreLife consistency 19.6% more than
  y=0; passed; landed in PRIMARY.
- **Class-conditional GMM Bayes-factor base** — d18 F5; 2 GMMs(8) per
  class on 5 KS-low feats; standalone AUC 0.873; passed.
- **Conditional-vector tuple lookup** — d18 J; 5-way empirical-Bayes
  on (Position × Compound × Stint × Race × Year); passed.
- **CTGAN mode-id features** (G/H/I) — passed individually but FULLY
  ABSORBED by CatBoost-yekenot; absorbed at the 23-base level.
- **Synthesiser architecture replay** — d18 F1; lowest mean KS = 0.134
  to CTGAN replay; non-GAN→GAN P(replay-like) jump 0.06→0.13;
  characterises the host as a CTGAN-class GAN.

## Day-18 LR-bank ensemble

- **15-variant LR bank + 5 rich-FE** — Day 17 PM; lr_mega OOF 0.92776
  is the LR ceiling; combined GBDT+LR effective rank 3.33 (+1.3 dim);
  K=24 + mega LR sweep null/regress.
- **Per-segment mega LR (Compound × Year)** — +60.8 bp standalone but
  routes through cb_year-cat at meta — friction
  `per-segment-mega-LR-fires-only-at-LR-class-not-meta-class`.
- **Path-B Compound × Year on K=24** — Day 18; all τ null/regress
  because cb_year-cat already routes Year.
- **Path-B 3 alt axes on K=24** — Day 18; (Driver-cluster × Stint),
  (Race-class × TyreLife-q5), (Position-q5 × Compound) — all null.
  Path-B amp axis on K=24 is uniquely Compound × Stint.

## Day-19 overnight (4 axes closed)

- **Debashish historical-priors external join** (D1) — closed null-
  by-pre-flight (PI sealed −2 to 0 NULL; harness 0.20 bp predicted).
- **XGBoost yekenot transfer** (B2) — verify at K=27; +0.143 bp null;
  GBDT-class redundant on shared FE.
- **LightGBM yekenot + cross-row aggregates** (A5) — proxy std OOF
  below v3 ceiling; K=27+1 −0.106 bp null (7th rank-lock confirmation).
- **Yao/Vehtari covariance-modulated per-segment stacker** (C1) —
  3 τ all regress vs plain shrinkage; per-segment-stacker family
  closed (9 variants tested over Days 14-19).

## Day-20 PM forest sweep (3 angles)

- **Random forest as meta-stacker** on K=4 [P, rank, logit] (12
  features) — OOF 0.95384 vs LR-meta 0.95399 = **−1.54 bp falsified**.
  Bagged-tree variant of the Day-20 PCA-meta probe finding for
  LightGBM; closes the "non-LR meta" clause across both boosted
  and bagged tree-class metas.
- **Random forest on combined input** (K=4 [P, rank, logit] + 6 raw
  numerics = 18 feat) — RF 0.95393 vs LR-on-same 0.95400 = **−0.70 bp
  falsified**. Adding raw numerics to the meta does not rescue
  tree-class meta — confirms `combined-input-meta-stacker-absorbed`.
- **Random forest base on yekenot recipe** (no orig concat, 38 feat,
  5-fold StratifiedKF) — standalone OOF **0.94178**; K=4+1 LR-meta
  **+0.26 bp** at **ρ=0.9595** vs PRIMARY. Most-diverse positively-
  gating base in the K=4 era. +12 bp standalone over d15c ET-on-raw,
  4.4× larger min-meta lift. Hedge-eligible per R5; Path-B refit
  on K=5 is the natural next probe.
