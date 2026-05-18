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
- **Kitchen-sink RF** (yekenot + 12 constraint-violations + 7 inter-
  stint memory = 57 feat) — standalone OOF **0.94054** (−1.24 bp vs
  yekenot-only); K=4+1 LR-meta **+0.25 bp** at ρ=0.9580. **Feature
  breadth hurts RF on this data** (weak features dilute split
  capacity at the random-subset level). **K=4+1 lift is unchanged
  within fold noise** vs Angle A — first reproducibility check on
  the forest-base lift. PI hypothesis that RF scales with feature
  breadth (per irrigation 14-bank meta precedent) is empirically
  refuted on s6e5: the irrigation gain came from already-distilled
  base predictions, not raw + engineered features.
- **Optuna-tuned RF on kitchen-sink** (15 TPE trials, single-fold
  proxy + 2-seed full-5-fold validation). Best config: log2 features,
  max_samples=0.7, max_depth=15, leaf=100, entropy. **K=4+1 LR-meta
  Δ +0.268 bp seed=42, +0.238 bp seed=7** (cross-seed |Δ|=0.030 bp).
  Standalone OOF dropped further (0.93957) — log2/max_samples=0.7
  trades calibration for tree diversity. **Optuna yields zero
  meaningful improvement past the natural +0.25 bp ceiling.** Across
  4 independent RF runs (Angle A, Kitchen-sink, Optuna seed 42,
  Optuna seed 7) the K=4+1 lift sits in +0.24-0.27 bp with std
  0.013 bp. **The +0.25 bp signal is robust to feature width,
  hyperparameters, and seed; it cannot be tuned higher because it
  is set by the meta architecture (3-D logit subspace ceiling),
  not by RF itself.** Path-B refit on K=5 = K=4 + RF-yekenot is
  the single remaining forest-family move with non-trivial EV.
- **Path-B C×S τ-sweep on K=5 = K=4 + RF-yekenot** (5k/20k/100k).
  Compound × Stint segmentation, MIN_ROWS=1000 — same mechanism
  as the K=4 PRIMARY. K=5 LR-meta global OOF 0.95402; K=5 + Path-B
  C×S τ=100k OOF 0.95405; **Path-B amp on K=5 = +0.03 bp within
  fold noise**. ρ vs K=4 PRIMARY = 0.999917 → tie-band at LB per
  Rule 27 (abort threshold 0.999). PI held submission per Rule 27;
  saved as R5 hedge candidate. **Confirms Day-15 friction
  `path-b-amp-only-fires-on-meta-arch-not-base-add`:** per-segment
  shrinkage absorbs the +0.25 bp K=4+1 forest lift down to +0.02 bp
  vs PRIMARY. Path-B amplifies meta-architecture redesigns, not
  single-base orthogonal additions below ~+0.5 bp OOF. Forest
  family characterized end-to-end on s6e5: structural diversity
  benefit caps at +0.25 bp K=4+1; LB transfer is tie-band.

## 2026-05-08 PM EXP-NEW Phase 1-5b (research-feature-engineering)

- **A3-7 user-id smoothing on PitNextLap target** — dry-run −124 bp
  FAIL (target leakage destroys within-group variance). Killed
  before launching at scale.
- **A2-2 mandatory compound rule** — only Phase 1 smoke winner
  (+9.3 bp 50k×1F). Full 5-fold single-LGBM 0.94577 vs 0.94563
  (+1.4 bp partial absorb). K=4+1 plain LR-meta +0.302 bp (below
  +0.5 PASS); G3 flip 0.195 asymmetric; TIE_EXPECTED on 4-gate.
  K=4+1 Path-B C×S τ=100k +0.26 bp; ρ 0.999893; WEAK.
- **A2-3, A2-4, A2-6, A2-7, A3-1, A3-2** — Phase 1 smoke null /
  regress at 50k×1F. Six picks closed in one batch; A3-1 RankSorted-
  Gaps already covers the Frontiers AI 2025 `DriverAheadPit`/Behind
  peer-effect family.
- **A2-8 LightGBM stack-meta on K=4** — 43 meta features (P + ranks
  + logits + 6 pairwise products + 6 abs-diffs + 6 logit-diffs +
  raw side info). 5-fold OOF 0.95390 vs Path-B PRIMARY 0.95403
  (Δ −1.30 bp); also below plain LR 0.95399 (−0.96 bp). Fold-std
  0.00080 elevated. **Friction:** tree stackers overfit interaction
  noise on small-K pools; convex LR + Path-B partial-pooling
  regularize better. Closed FAIL.
- **Bi-LSTM / GRU sequence base** — research scan only. Frontiers
  AI 2025 Bi-LSTM achieved F1 0.81 with 10-lap windows; deferred to
  next session (GPU-heavy ~30-60 min Kaggle T4). Cited as the only
  genuinely untried mechanism after the Rule 7 saturation scan.

## 2026-05-18 Round 4 — Rule 23 free-form FE + mechanism-orthogonal stacking

- **R4 segment-FE v1** (`probe_r4_segment_fe.py`) — 9 hand-coded
  interaction features (Cumulative_Degradation × Compound × 5,
  Position_Change × is_named_driver, is_named, is_WET × Stint1,
  is_INTER × Stint2). Standalone OOF 0.94878. K=4+1 Δ +0.263 bp
  (strongest in Round 4 single-bases, but G2-fails 0.30). Per-segment
  AUCs DID NOT improve on the targeted weak segments (WET-S1 went
  0.81 → 0.77); the meta-lift came from the named-driver × position-
  change interaction (LR-meta logit-coef +0.114).
- **R4 segment-FE v2** (`probe_r4_segment_fe_v2.py`) — drop WET features,
  add TyreLife × Compound + Position_Change × Stint (13 features
  total). Standalone OOF 0.94873. K=4+1 Δ +0.211 bp; TyreLife features
  redundant with the existing TyreLife raw column.
- **R4 v1+v2 2-base stack** — K=4+2 Δ +0.282 bp; v1 and v2 are
  largely redundant (0.019 bp gain from combining).
- **R4 HMM sequence base** (`probe_r4_hmm_seq.py`) — Gaussian HMM
  K=8 hidden states on per-(Year, Race, Driver) sequences with
  observations (Compound_int, TyreLife, RaceProgress, Stint,
  Position_Change, Cumulative_Degradation). 8-dim posterior +
  entropy added as features to downstream LightGBM. HMM fit time
  223s (Baum-Welch 30 EM iter on 40,869 sequences). Standalone OOF
  0.94713. **K=4+1 alone Δ −0.005 bp — NULL.** Sequence-class
  features fully absorbed at the LR-meta when used alone.
- **R4 mechanism-orthogonal 2-base stack (seg_fe + HMM) — Δ +0.542
  bp at K=4+1, FIRST G2 PASS of Round 4 after 18 nulls.** LR-meta
  logit-column coefficients point in opposite directions (seg
  +0.200, HMM −0.127); the bases provide corrections in orthogonal
  prediction directions — genuine cross-mechanism diversity.
  Anchor-attenuation: 0.542 @ K=4 → 0.275 @ K=5(+K27super). LB
  calibration probe submitted → LB 0.95354 (OOF→LB drop 5.1 bp,
  matching K=4+Path-B transfer pattern). **The simple lesson:
  row-feature ceiling holds per single-mechanism, but ORTHOGONAL
  mechanism families combine super-additively.** Next-session
  priority: retest at REAL K=11+1 anchor after slim-kNN rebuild.

## 2026-05-18 Round 5 — multi-class super-stack + Path-B operator

- **R5 slim-kNN rebuild** (Phase A). All 6 dgp_v3 builders ran
  successfully after `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`
  was pulled (the local snapshot didn't include the f1_strategy_dataset_v4.csv
  file the builders reference). K=11 LR-meta plain OOF reconstructed at
  0.95443 — matches historical PRIMARY value exactly. Per-builder K=4+1
  gates: qAT +1.172 bp, qAO +0.730 bp, qAA +0.143 bp, qAF +0.149 bp,
  others positive/neutral.
- **R5 retest at REAL K=11+1** (Phase B). r4_segment_fe + r4_hmm_seq
  combination at K=11+1: Δ +0.245 bp OOF. Anchor-attenuation pattern
  CONFIRMED (0.542 @ K=4 → 0.275 @ K=5 → 0.245 @ K=11). LB
  submission **0.95382** at OOF→LB transfer -6.3 bp. G2 marginal
  pass; ρ vs K=27+Path-B = 0.9989 (just under OK band).
- **R5 graph-class per-(Race, Lap) pit-pressure features**
  (Phase C). 4 features capturing cross-driver pit timing pressure.
  Standalone OOF 0.93344; at K=11 LR-meta alone: -0.012 bp.
  Marginally REGRESSES the seg+HMM combo (-0.030 bp). Closes the
  graph-class axis at K=11+seg+HMM anchor.
- **R5 multi-class super-stack sweep** (Phase D). 15 combinations
  of {seg-v1, seg-v2, HMM, graph, TRF} at K=11+N LR-meta. Best:
  seg+HMM at +0.245 bp. 4-way (seg+HMM+graph+TRF) at +0.254 bp —
  negligible diff. The seg+HMM 2-mechanism combo is the operator-
  invariant winner.
- **R5 Path-B operator on K=13 — THE BREAKTHROUGH** (Phase D).
  Same OOF as LR-meta (0.95446 vs 0.95445), but **5 bp better
  LB transfer**. K=11+seg+HMM under LR-meta → LB 0.95382; under
  Path-B C × Stint τ=100k → **LB 0.95387 (new PRIMARY, +0.01 bp
  over prior PRIMARY 0.95386).** Per-segment shrinkage operator
  preserves the mechanism-orthogonality signal that global LR meta
  absorbs.
- **R5 gap-aware transformer** (Phase F, Kaggle T4). 4-layer
  D=128 transformer with attention over LapNumber positional
  encoding on per-(Year, Race, Driver) sequences. Two errors
  fixed (data path + sm_60 PyTorch incompat). Standalone OOF
  0.91974 (weak; 35 bp below K=11 baseline). Absorbed at K=11+
  Path-B; null contribution. Saved for next-session v2 retry
  with larger architecture + GroupKFold split.
- **R5 5-seed Path-B bagging — FAILED design**. Multi-seed via
  monkey-patching SEED constant in `run_pathb()` only changes the
  Stratified fold splits; test predictions are seed-INVARIANT
  because they come from a full-train fit (line 116 of
  build_K11_full_pathb.py). 5-seed test predictions identical
  (ρ=1.0). True bagging needs fold-fit averaging or sub-sample
  variation. Friction logged for next-session implementation.
- **R5 70/30 rank-blend** of R5.2 + K=27+Path-B → LB 0.95385
  (-0.02 bp vs R5.2). The 30% K=27 weight pulls toward K=27+Path-B's
  lower LB. Rank-blending with structurally-different operator
  doesn't help when one source dominates by both OOF and LB.

## 2026-05-18 Round 6 — operator-axis retest + proper bagging

- **R6 Phase A: 5-candidate operator-axis retest at K=13+Path-B
  τ=100k — 5/5 NULL.** Re-gated conformal_widths, rrf_k60,
  meta_lgbm_rank, trimmed_rank, r4_segment_fe_v2 — all previously
  null under K=4 LR-meta. Δ −0.026 to −0.090 bp under Path-B too.
  The R5 +5 bp Path-B-vs-LR-meta swing is POOL-SPECIFIC (seg+HMM
  × Path-B interaction), not a general operator advantage.
  The "test all prior nulls under Path-B" hypothesis is FALSIFIED
  for these 5 candidates.
- **R6 Phase B: K=13+Path-B 5-seed fold-fit bag** —
  `scripts/build_K13_seghmm_pathb_foldbag.py`. Replaces `run_pathb`'s
  full-train test path with per-fold per-seed test-prediction
  averaging (25 fits across 5×5). Bag OOF 0.95448 (+0.212 bp over
  single-seed R5.2). Predictions DIFFER from single-seed (ρ=0.999988
  vs ρ=1.0 in R5's broken bag — true bagging now). LB submission
  **0.95387 — ties R5.2 within 5-decimal quantization (TIE_ZONE
  prediction confirmed).** Variance reduction works mechanically;
  may register on private LB.
- **R6 Phase C: Transformer v2 (Kaggle T4)** — D_MODEL=256, 6 layers,
  15 epochs, GroupKFold by (Year, Race, Driver) sequence (was
  Stratified per row in v1). Standalone OOF 0.93330, **+13.5 bp**
  over v1's 0.91974 DESPITE the structurally harder GroupKFold
  split. Bigger arch + proper split worked. At K=14+Path-B (R5.2
  + TRFv2): Δ −0.014 bp — absorbed at meta. Standalone OOF still
  21 bp below K=11 baseline (0.95443); doesn't reach meta-utility
  threshold. v3 with even larger arch + pretraining might cross.
- **R6 Phase D: K=14 fold-fit bag (R5.2 + TRFv2 + bagged)** — OOF
  0.95448 (same as Phase B alone). Transformer adds nothing on top
  under Path-B. Phase B captures the full mechanism diversity.

## 2026-05-18 Round 7 — Path-B segmentation sweep + DAE

- **R7 Phase A swap-noise DAE** (`kernels/r7-swapnoise-dae-gpu/`).
  Porto Seguro recipe: 3-layer MLP encoder [23→256→256→128] + 15%
  swap-noise + MSE on train+test combined. Standalone OOF 0.94665
  (stronger than transformer v1 0.91974, comparable to HMM 0.94713).
  At K=14+Path-B for EVERY segmentation tested: Δ −0.09 to −0.15 bp.
  Embedding-class diversity ABSORBED at meta for K=11-pool — closes
  this version of DAE; v2 with deeper bottleneck or contrastive loss
  may cross.
- **R7 Phase B multi-segmentation Path-B sweep**
  (`scripts/build_K13_pathb_multiseg.py`). 3 alternative segmentations
  tested vs default Compound × Stint (R5.2 baseline OOF 0.95446):
  - Year × Compound (20 seg): Δ −0.149 bp NULL
  - **DriverClass × Stint (12 seg): Δ +0.106 bp WIN** (R7.1)
  - Compound × Stint × LapBucket (120 seg): Δ +0.065 bp marginal
  τ sweep on winner: τ=100k optimal (+0.106 bp), τ=20k regresses
  (-0.129 bp), τ=500k marginal (+0.017 bp).
- **R7.1 K=13 + Path-B DriverClass × Stint τ=100k — NEW PRIMARY,
  LB 0.95389** (+0.02 bp over R5.2 PRIMARY). The named-vs-anonymous
  driver split (named codes like VET vs synthetic D0XX) captures
  pit-rate variance that Compound × Stint misses. First non-Compound
  segmentation to beat the default in 6 weeks.
- **R7 Phase D R7.2 cross-pollination**: 5-seed fold-fit bag of R7.1
  (`scripts/build_K13_pathb_multiseg.py` + R6 fold-fit harness logic).
  OOF 0.95450 (+0.264 bp over R7.1 single-seed; largest OOF lift of
  session). ρ vs R7.1 = 0.999973 → TIE_ZONE; LB tied R7.1 at 0.95389.
  Structurally distinct (5-seed averaged on alt segmentation) →
  retained as private-LB hedge.

## 2026-05-18 Round 9 — Closure of row-feature axis

After R8 multi-seg sweep null (4 segs, none > +0.10 bp) and R8 EOD
strategy-critic verdict "structural shortfall vs 1.6 bp top-5% gap",
R9 ran dual-track on the last 2 viable research-loop candidates.

- **R9 NB4 — Per-(Compound × Stint) target-mean as BASE learner.**
  Novel TE grouping (mechanism-ledger had no Compound × Stint entry;
  6 existing TE_CONFIGS in `scripts/p1_features.py:336-342` do not
  include this combo). Built via `cv_target_encode` (fold-safe;
  re-fits stats per fold's training rows). Standalone OOF **0.94850**
  (G1 PASS, between hgbc_deep 0.94870 and HMM 0.94713). K=14 + Path-B
  DriverClass × Stint τ=100k: OOF 0.954469 (**Δ vs R7.1 PRIMARY
  −0.022 bp NULL**). The TE-broadcast-at-base mechanism is absorbed
  at the K=13+Path-B meta layer — Path-B already extracts the per-
  (Compound, Stint) signal at META; same signal injected at BASE is
  redundant and slightly dilutes rank-lock. Segmentation-as-base
  axis CLOSED.

- **R9 C1 — Aadigupta external per-Race feature scalars.** 5 scalars
  (lap-time median / std, cum-deg max, pos-change std, race-length
  max) joined to s6e5 train/test by Race name (26/26 overlap). NOT
  target-derived → trivially Rule 24 safe; no transductive footprint.
  Standalone OOF **0.94902** (G1 PASS; ~5 bp better standalone than
  NB4). K=14 + Path-B DriverClass × Stint τ=100k: OOF 0.954466
  (**Δ vs R7.1 PRIMARY −0.045 bp NULL**; ρ_test 0.999981 TIE_ZONE).
  External-data injection — the only structural lever surfaced by
  the EOD strategy-critic — ALSO absorbs at the K=13+Path-B meta
  layer. C1 regressed MORE than NB4 because yekenot's existing 6
  TE_CONFIGS touch Race in 5 of 6 configs; the per-Race scalars
  duplicate signal density already absorbed by the pool. Data-class
  axis (D) for K=13+Path-B CLOSED.

- **R9 strategic conclusion.** Rank-lock at K=13+Path-B is
  structurally confirmed across three axes: operator family (R6 v2
  transformer, R7 DAE absorbed); mechanism class (R4 segment-FE
  G2-fail, R4-R5 HMM/pit-cascade null, R9 NB4 absorbed); data class
  (R9 C1 absorbed). The remaining structurally distinct mechanism
  classes lie OUTSIDE row-features: A1 seq2seq transformer on
  per-(Driver, Race) lap sequences, graph mechanism with competitor
  edges, survival/hazard model on stint-life. PRIMARY R7.1 unchanged;
  3 daily slots held for R10 mechanism-expansion probes.
