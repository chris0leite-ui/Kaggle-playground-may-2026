# 2026-05-07 — Day-17 PM strategy critique (Rule 14 + Rule 7 research-loop)

PI directive 2026-05-07: "These improvements [K=24 Path B follow-up etc.] do
not matter — ±0 bp gut-feel. Revisit problem-solving loop. Check most pressing
hypotheses. Decide where to put focus."

This note is the strategy-critic-loop output (Rule 14) plus a Rule 7
research-loop sweep, triggered at PI's instruction with plateau_days=2 (no LB
advance over d16 PRIMARY since 2026-05-06 evening).

## 1. Where we are

- **PRIMARY** = `d16_path_b_K22_continuous_only_tau20000` LB **0.95089**
  / OOF 0.951208 (advanced 2026-05-06 PM via sibling branch's KS-divergence
  feature-restriction-transfer trick).
- Top of LB ~0.955 (PI obs); top-5% threshold 0.95345; gap **−25.6 bp**.
- 12 days remaining (deadline 2026-05-31); 32/270 submissions used.
- **Plateau-days = 2** (no PRIMARY advance Day-16 PM → Day-17 PM despite
  4 sessions of compute across 4 branches).
- Closed mechanism families (do NOT retry):
  - Per-row FE — `tag: synthetic-dgp-conditionally-near-independent`.
  - Target reformulation — `tag: target-construction-layer-leakage` (88-100%
    strict-OOF collapse).
  - Single-LGBM thesis — falsified Day-17 AM (Rozen replication; honest
    fold-safe ceiling 0.94563 OOF, −52 bp from PRIMARY).
- Saturated mechanisms (5 cross-confirmations of `lr-meta-rank-lock-strong-anchor`):
  K=22 + new diverse base via LR-meta no longer fires amp.
- This branch's Phase-A K=24 LR-meta C7 = +0.81 bp OOF / ρ_test 0.99506 /
  predicted LB Δ −0.69 bp TIE — confirms the saturation pattern at K=24.

## 2. Convergent finding: the structural gap is RECIPE, not POOL

Two parallel research tracks executed today:

**Track A — public-notebook gap analysis (Rule 22):** read all 8 reference
kernels under `external/kernels/`. Verified actual reported scores (the
Explore agent's "0.95816" for driver-eng was hallucinated — that notebook is
a CV ladder analysis with diagnostic OOF 0.93208, NOT a top-LB scorer).
Verified scores by inspection:
- pit-or-stay-f1-strategy-1 — quotes "proven public 0.95273 model family"
  for the **yekenot RealMLP recipe** + 4-model blend.
- f1-lap-by-lap-prediction-engine-v2 — LB 0.9531 with **GRU + competitor
  field aggregates + Safety-Car proxy**.
- ps-s6e5-hb1 — h_blend (rank-aggregated harmonic blend) 0.95400 over a
  bag of 0.95229 / 0.95282 / 0.95356 / 0.95388 OOFs.
- ps6e5-ensemble-0-95314 — blend-only over a published "vault" of others'
  submissions. Score 0.95314.
- predicting-f1-pit-stops-blend — blend-only on the same vault.
- ps-s6-e5-realmlp-pytabkit — yekenot's original RealMLP recipe.
- s6e5-driver-s-high-driver-feature-eng — CV ladder analysis (Pilkwang).
- romanrozen — already replicated and falsified (LGBM kitchen-sink leaky).

**Track B — web research-loop:** GM playbooks (NVIDIA / Deotte /
Gauurab) + arXiv search.
- s5e12 2nd place winning trick = "Winning based on **ID Shift Analysis**"
  (synthetic-CTGAN id-structure leak). Our id_mod_1000 568bp marginal span
  is a flag (currently absorbed by GBDT).
- Frontiers AI 2025 Bi-LSTM ([frai.2025.1673148](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full))
  identifies **DriverAheadPit** (lead driver pitted last lap) as a 2.6%→5.7%
  pit-prob shift = +1.1 logit lift.
- TabbyFlow ([arXiv 2512.00698](https://arxiv.org/abs/2512.00698)) is the
  current SoTA tabular DGP modeller; no Kaggle deployment yet.

## 3. The diagnosed gap: yekenot RealMLP standalone vs ours

Loaded `oof_realmlp_strat.npy` and computed:

```
oof_realmlp_strat.npy            AUC 0.94582  (our K=21 base)
oof_d16_orig_continuous_only     AUC 0.91483  (single 7-feature LGBM)
```

`scripts/e4_realmlp_probe.py` shows our `realmlp` base is a **default-config
single-fold smoke test**: `RealMLP_TD_Classifier(device='cpu', n_cv=1,
val_metric_name='cross_entropy', use_ls=False)`. No hyperparameter pack;
no engineered cat set; no ensembling; trained on fold-0 only.

**yekenot's published recipe (cell 58 of `pit-or-stay-f1-strategy-1.ipynb`):**
```python
REALMLP_PARAMS = {
  'val_metric_name': '1-auc_ovr',     # AUC, not cross-entropy
  'n_ens': 8, 'n_epochs': 4,           # 8-ensemble vs ours 1-ensemble
  'lr': 0.07, 'wd': 0.018, 'sq_mom': 0.98,
  'lr_sched': 'cos_anneal',
  'first_layer_lr_factor': 0.25,
  'embedding_size': 6, 'max_one_hot_cat_size': 18,
  'hidden_sizes': [512, 256, 128],
  'act': 'silu',
  'p_drop': 0.05, 'p_drop_sched': 'expm4t',
  'plr_hidden_1': 16, 'plr_hidden_2': 8,
}
CAT_COLS_FINAL = ['Driver','Compound','Race','Year_str','Driver_Compound',
                  'Race_Compound','Race_Year','Driver_Race','Driver_Year',
                  'Compound_TyreLifeBin','Compound_RaceProgressBin','Stint_Compound']
```
+ 5-fold StratifiedKFold + ordinal-encoded engineered cats. Reported OOF
~0.95273.

**Standalone gap = +69 bp on the same model class.** The cause is
unambiguously recipe (hyperparameters + ensembling + engineered cat features),
not data: pit-or-stay confirms yekenot trains on the standard train.csv +
the same `f1_strategy_dataset_v4` we already use elsewhere.

This is the structural mechanism PI hypothesized. We never deployed RealMLP
properly — our K=21 `realmlp` slot has been a 0.946-class smoke test for the
entire competition.

## 4. Top 3 hypotheses ranked by EV/cost

### H1 — Deploy yekenot RealMLP recipe (HIGHEST EV; cheap; lowest risk)

**Mechanism.** RealMLP-TD with the published hyperparam pack + engineered
cat set + 8-ensemble + 5-fold OOF; replaces our default-config `realmlp` slot.

**BOTE.** Family `tuning_existing` priors (0.20, (0, 0.5, 1.5)) bp UNDERSTATES
because the gap is +69 bp standalone — this is a recipe-replacement, not
hyperparameter tuning. Re-frame as `new_model_class` priors (0.40, (3, 8, 15))
bp. Q6: log-loss / row-AUC aligned = True (RealMLP_TD_Classifier is
log-loss-trained). Cost: 30-90 min CPU on 4 cores or 10-30 min P100 GPU.

**Predicted standalone OOF.** 0.952-0.953 (replicating yekenot's published
0.95273; published 0.95260 / 0.95259 fold-A/B from Rozen's notebook agree).

**Predicted at-meta-add lift.** ρ_test vs PRIMARY likely 0.95-0.97 (RealMLP
is structurally distinct from GBDT/FM/DAE pool; far more diverse than
default-RealMLP at ρ 0.998). At ρ 0.95 with +5-10 bp standalone lift over
our `realmlp` slot, K=22 swap predicts +3-8 bp OOF, predicted LB Δ +1.5-6.5
bp via probe.py band.

**Devil's advocate (Rule 26c).** (a) yekenot's "0.95273" may carry CV-TE
fold-leakage similar to Rozen's published 0.95354; first thing to check is
fold-safety of CAT_COLS_FINAL engineered features. (b) RealMLP at higher
n_ens may overfit to public-LB folds. (c) `1-auc_ovr` val metric on
binary-classification target ≠ binary AUC; confirm before deploying.

### H2 — F1 telemetry external join (DriverAheadPit + TrackStatus + CumulativeTimeStint)

**Mechanism.** FastF1 / Ergast historical telemetry pull keyed on (Driver,
Race, Year, Lap). Specifically: the lap-N "did the driver in front of me
pit on lap N−1?" flag.

**BOTE.** Family `external_data_aggregate` priors (0.20, (0, 1, 4)) bp
UNDERSTATES per Frontiers AI 2025 evidence. Override: (1, 8, 20) bp at
P=0.30. Cost: 60-180 min CPU (FastF1 telemetry pull + key-merge + base
training). Q6: log-loss-aligned binary feature, True.

**Predicted.** +8-20 bp LB if join coverage ≥ 80% AND DGP doesn't already
encode this signal.

**Devil's advocate.** (a) **Highest risk:** the synthetic DGP (aadigupta1601's
generator) may already train on FastF1 telemetry — joining back amounts to
label retrieval, voids strict-OOF. (b) Year coverage — test rows likely
include 2024-25 races, FastF1 has gaps. (c) Competition rules check needed
before any submit (Playground series default allows external data).

### H3 — ID-shift / row-position structural probe (cheap diagnostic)

**Mechanism.** Replicate s5e12 2nd-place: AV-AUC at `id_mod_N` for
N ∈ {7, 11, 100, 1000, Driver-card}. If any granularity has AV-AUC > 0.502,
fit sparse-LR on those features as a K=22-swap or K=22+1-add candidate.

**BOTE.** Family `single_base_fe_addition` priors (0.05, (0, 0.5, 2)) bp
UNDERSTATES the structural-leak case. Override: (1, 4, 10) bp at P=0.15.
Cost: 20-40 min CPU. Q6: log-loss aligned, True.

**Predicted.** +0 to +5 bp LB if structural shift exists at any modular
granularity; pure NULL if AV-AUC stays at 0.502 across all N.

**Devil's advocate.** (a) GBDT already absorbs id_mod_1000 568bp marginal
span; the residual at sparse-LR should be small. (b) Public LB is row-iid
per U3, so any structural leak found may not transfer to private LB.

## 5. What NOT to chase (closed; do not re-claim)

- C7 + Path B Compound×Stint follow-up — predicted ±0 bp by PI; +0.23 bp
  OOF over PRIMARY too small to rely on against ρ-noise.
- More K=22+1 base-adds at ρ ≥ 0.999 — 5 cross-confirmations of rank-lock.
- Per-row FE additions — `synthetic-dgp-conditionally-near-independent`.
- Target reformulations — `target-construction-layer-leakage` (all collapse).
- TabPFN v2.5/v2.6 — DEAD (AUC ceiling 0.944, OOM on P100).
- Pure single-model thesis — falsified Day-17 AM.
- 70+ model brute-force stack scaling — saturated under LR-meta rank-lock.
- Pirelli press-release compound priors — already absorbed by 3-way TE.
- Harmonic / rank-aggregator blends (hb1) — published lift +0.04 bp;
  noise floor.
- ExtraTrees / KNN R5-HEDGE-only items — keep for final-window hedge.

## 6. Recommendation

**Run H1 first** — yekenot RealMLP recipe replication. Three reasons:

1. **Cheapest diagnostic by far** (30-90 min CPU; can fit on local 4-core).
   Either it lifts standalone OOF from 0.94582 → 0.95273 (closing the recipe
   gap) or it doesn't — both outcomes are load-bearing data.
2. **Lowest risk**. Pure supervised RealMLP retrain; no external data, no
   target reformulation, no leak-prone FE, no rules-check needed.
3. **Highest single-shot EV.** If +69 bp gap closes even partially:
   - K=22 swap (drop our `realmlp`, add yekenot RealMLP) → predicted +1-5 bp
     PRIMARY LB.
   - K=22 swap + Path B Compound×Stint τ=20k → predicted +2-8 bp PRIMARY LB.
   - K=22+1 add (keep both) → predicted +1-3 bp PRIMARY LB.

If H1 lands, **H2 (FastF1 join)** is next as a parallel-week probe (after
rules-check and DGP-source verification). H3 stays as a 30-min cheap
diagnostic that can run alongside H1 on a separate core.

## 7. Open questions for PI

- Approve H1 (yekenot RealMLP recipe deployment) as next compute? 30-90 min CPU.
- Sealed prediction (Rule 26a) — what's PI's gut for closing the +69 bp
  recipe gap (in standalone OOF) and the resulting K=22-swap LB Δ?
- Approve H2 (FastF1 external join) for next session, conditional on
  Playground-series external-data rules check + DGP-source verification?

## Pointers

- `audit/2026-05-07-d17-phase-a-composition-gate.md` — Phase-A C1-C7
  (this morning's gate).
- `external/kernels/ps-s6-e5-realmlp-pytabkit/ps-s6-e5-realmlp-pytabkit.ipynb`
  — yekenot's original recipe.
- `external/kernels/pit-or-stay-f1-strategy-1/pit-or-stay-f1-strategy-1.ipynb`
  — secondary cite of the same recipe at OOF 0.95273.
- `scripts/e4_realmlp_probe.py` — our default-config smoke (TO REPLACE).
- ISSUES.md leaf 1c, 4a, 7f.
- Frontiers AI 2025 Bi-LSTM paper, s5e12 2nd writeup — web-research
  citations.
