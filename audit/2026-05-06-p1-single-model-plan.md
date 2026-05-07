# P1 single-model hypothesis — 7-step problem-solving plan

Branch: `claude/read-kaggle-handover-rsi2Q`
Date: 2026-05-06 PM
Trigger: PI hypothesis "leader at LB ~0.955 likely uses ONE strong model
with a structural mechanism we missed". HANDOVER A1.

---

## Step 1 — Define

**L1.** Can a SINGLE model achieve LB AUC ≥ 0.95345 (top-5%) — or at
minimum beat current PRIMARY (K=22 + Path-B hier-meta) at 0.95059
without any stacking?
**L2.** If yes: which feature/architecture mechanism unlocks the
+15-30 bp jump that our 25-base stack didn't find?
**L3.** Reusable single-model template for future synthetic
Playground comps.

**Decision-maker:** PI.
**Criteria:**
1. standalone OOF AUC > current best single-model (e3_hgbc 0.94876).
2. predicted LB Δ ≥ +1 bp vs PRIMARY 0.95059 (single-shot calibration probe).
3. mechanism understandable + reproducible (publishable single-model story).

**Constraints.**
- 10/day submit budget; 0/10 used today; need PI sign-off per submit (Rule 1).
- Sandbox is CPU-only; Kaggle GPU available for NN class.
- Strict-OOF discipline (Rule per `target-construction-layer-leakage`).
- 50k token CLAUDE.md cap.
- Comp deadline 2026-05-31 (25 days remaining).

**Boundary.**
- IN: single model class, raw + engineered features, target reformulations
  as features (strict-OOF only), external row-level data joins, single-model
  pseudo-labelling, single-model bagging (multi-seed avg).
- OUT: stacking, multi-model blends, ensemble of K bases (those are P2/P3).

---

## Step 2 — Disaggregate (MECE logic tree)

```
ROOT: SINGLE MODEL achieves LB AUC ≥ 0.95345 (or close)
│
├── A. Same model class (GBDT) + WIDE FE       [HIGHEST EXPECTED — PI direction]
│   ├── A1. Engineered features (~118 per Rozen): tyre/compound/race-progress/lag-rolling
│   ├── A2. CV target encoding of high-card combos (Driver×Race×Year etc.)
│   ├── A3. Cross-row aggregates within (Driver,Race,Year), (Driver,Race), (Race,Compound)
│   ├── A4. Quantile binning + pairwise/3-way cross categoricals
│   ├── A5. Frequency / count encoding of categoricals
│   ├── A6. Aggregate priors (mean PitNextLap by Compound × RaceProgress_q5) — strict OOF
│   ├── A7. DAE 768d latent as input to single GBDT (already exists)
│   └── A8. KNN-density features per Compound, per Driver
│
├── B. Different model class (single)
│   ├── B1. RealMLP single (PyTabKit; Rozen MLP_PARAMS) — Rozen OOF 0.95260
│   ├── B2. FT-Transformer / SAINT single (Kaggle GPU)
│   ├── B3. CatBoost depth=10+ with aug FE (Rozen OOF 0.95127)
│   ├── B4. Single XGBoost Optuna-tuned (Rozen OOF 0.95232)
│   └── B5. TabPFN v2.6 [DEAD per ISSUES 1a — model OOM, AUC 0.944 ceiling]
│
├── C. Training-loss reformulation (single-model)
│   ├── C1. LambdaRank / pairwise within (Driver,Race) [DEAD: -86 bp]
│   ├── C2. Survival / Cox regression on time-to-pit
│   ├── C3. Multi-task shared-trunk NN: {PitNextLap, inv_laps, pit_horizon}
│   ├── C4. Focal loss / class-imbalance-aware
│   ├── C5. Soft labels / label smoothing
│   └── C6. Auxiliary self-sup task (predict next-Compound)
│
├── D. Sample weighting / fold-construction
│   ├── D1. AV-density sample weight [DEAD: AV-AUC 0.502]
│   ├── D2. Compound-rarity / Race-rarity weighted
│   └── D3. Curriculum learning order
│
├── E. Synthetic-data structure exploitation
│   ├── E1. Decode synth → orig via KNN; attach orig PitNextLap soft-label
│   ├── E2. Use original aadigupta1601 as ADDITIONAL training data
│   ├── E3. F1-official 1950-2022 historical pit priors per (Driver, Circuit)
│   └── E4. Train on (synth + decoded-orig) hybrid
│
├── F. Pseudo-labelling on test (single-model)
│   └── F1. Confidence-threshold pseudo (0.97/0.015) — Rozen pattern
│
└── G. Massive HP tuning + multi-seed bag
    ├── G1. Optuna 100-trial sweep on single LGBM with feA_te
    └── G2. Multi-seed bag (still "one model class")

DEAD/FALSIFIED branches: B5, C1, D1.
```

MECE check:
- A vs B: same class vs different class — disjoint by definition.
- C is loss-side, D is weight-side, F is data-side (test) — orthogonal levers.
- E is data-side (external) — disjoint from F (transductive within test).
- G is hp-side — applies to A or B.

---

## Step 3 — Prioritise (2×2 impact × effort)

| | Easy / cheap (≤30 min CPU) | Hard / expensive (>1 h or GPU) |
|---|---|---|
| **High impact (+10-50 bp)** | **DO NOW**: A1+A2+A5 wide-LGBM (Rozen recipe replication); B3 single CatBoost; A7 DAE-single | **PLAN FOR**: B1 RealMLP (Kaggle GPU); B2 FT-Transformer; E2/E4 hybrid orig+synth; G1 Optuna 100-trial |
| **Low impact (≤5 bp)** | OPPORTUNISTIC: G2 multi-seed bag; A3 group-aggregates; A4 quantile binning | PRUNE: F1 (already proven low ROI); E3 standalone (best as feature, not separate model) |

**Rationale.** Rozen's published recipe gives a directly testable
single-LGBM at OOF 0.95241 — a +362 bp jump over our best single
model. P(useful) for "new_model_class" family per `probe.py` is 0.40
with median +8 bp; we override predicted std-OOF lift to +150 bp
because of the published evidence, yielding expected LB +60 bp at
25 min cost, cost-efficiency 2.4 bp/min → PURSUE.

---

## Step 4 — Workplan

### Phase 1 (Day-16/17 today)
- **W1. Build feature factory** (`scripts/p1_features.py`):
  `make_features_A` (~50 columns) + `cv_target_encode` + `TE_CONFIGS`. ✓
- **W2. Build training script** (`scripts/p1_single_lgbm.py`):
  4 variants × 2 hp-regimes. ✓
- **W3. Run smoke** `--variant raw_only` (14 raw features, 1500 rounds)
  to verify the harness and get a baseline single-LGBM OOF on raw.
  Expected ~0.94 OOF.
- **W4. Run main** `--variant feA_te` (~50 engineered + 6 TE features,
  Rozen hparams, 6000 rounds with early stopping). Expected OOF ~0.952.
- **W5. Gate vs PRIMARY** (`probe.py gate`). Compute ρ vs PRIMARY,
  predicted LB Δ. If standalone OOF > 0.948 and ρ < 0.999 → submit candidate.

### Phase 2 (Day-17, conditional on Phase 1)
- **W6. Run** `--variant feA_te_orig` (concat aadigupta original).
  Tests E2/E4 lift from external data.
- **W7. Run single CatBoost** (8b) with same recipe; Rozen 0.95127 reference.
- **W8. Pre-submit-diff** (`scripts/pre_submit_diff.py`) on the winning
  candidate vs PRIMARY. Mandatory ρ check.
- **W9. PI-approved single-shot submit.** Single-model story.

### Phase 3 (Day-17/18, only if W9 LB result is +)
- **W10. RealMLP-PyTabKit single** (8c) on Kaggle GPU; Rozen OOF 0.95260.
- **W11. Stack-add probe (8d)**: K=22+1 LR-meta and K=22+1 hier-meta
  with the new single-model OOF as 23rd base. Tests if standalone
  signal still amplifies in stacking.

### Phase 4 (synthesis)
- **W12.** Audit: does the +150 bp single-model gap come from
  (i) FE alone, (ii) TE alone, or (iii) FE+TE interaction? Run
  ablations per variant table.
- **W13.** Friction tags + WRAPUP.md.

### Stop conditions
- Phase 1 OOF ≤ 0.948 → pivot: pull the actual Rozen LGBM submission
  from `external/makimakiai_idsafe/submission_v8_solo.csv`, score it
  locally vs PRIMARY (ρ, flips), gate manually.
- Phase 2 OOF ≤ Phase 1 OOF + 1 bp → drop orig-concat + drop CatBoost arm.
- Submit lift = NULL → re-decompose at this leaf level (Step 1 re-entry).

---

## Step 5 — Analyse (in progress; see Steps 6-7 for results)

Probe results land in `audit/2026-05-06-p1-single-model-results.md`
(separate doc, written on Phase 1 completion).

---

## Step 6 — Synthesise (deferred)

So-what slots:
1. If Phase 1 lands +150 bp single-model OOF → "single model can
   close gap" thesis is **CONFIRMED** at proper FE level. Friction
   to retire: `lr-meta-rank-lock-strong-anchor` (need to re-test
   if K=22 stack absorbs the new diverse single).
2. If Phase 1 lands but LB does not transfer → "single-LGBM OOF
   over-credits public LB" — friction tag candidate
   `single-lgbm-fe-oof-not-LB-transferable`.
3. If Phase 1 NULL → falsifies P1; pivot to A2/A3 partial recipes,
   then RealMLP class (B1).

---

## Step 7 — Communicate (deferred)

Pyramid: governing thought (single-model thesis result) → 3 supporting
arguments (OOF lift, ρ vs PRIMARY, predicted LB) → evidence (this doc)
→ decision (submit / hold / pivot).

WRAPUP.md and HANDOVER.md updates happen at Phase 1 completion.

---

## Pointers / external assets (Day-16 PM)

- `external/kernels/romanrozen/f1-pit-driver-race-year-encoding-0-95354.ipynb`
  — full pipeline; Rozen OOF table.
- `external/makimakiai_idsafe/submission.csv` — LB **0.95372** blended;
  `submission_v8_solo.csv` — single-pipeline fallback.
- `external/makimakiai_idsafe/run_report.json` — reports single-model OOFs:
  LGB 0.95238, XGB 0.95232, CB 0.95126, v8_best 0.95267, expected_lb_floor
  0.95267.
- `external/aadigupta_orig/f1_strategy_dataset_v4.csv` — original (101k rows,
  31/887 driver overlap).
- `external/f1_official_1950_2022/{pitstops,driver_details,...}.csv`
  — 1950-2022 historical priors (debashish311601).
- `external/weather_woodshole/F1_Weather_2022_2025.csv` — track conditions
  feed (not yet wired).
- `external/kernels/{ps-s6-e5-realmlp-pytabkit,ps6e5-ensemble-0-95314-best-score,
  pit-or-stay-f1-strategy-1,predicting-f1-pit-stops-blend,
  ps-s6e5-hb1,f1-lap-by-lap-prediction-engine-v2,
  s6e5-driver-s-high-driver-feature-eng}/`
  — six top public notebooks (5-72 votes) for cross-reference + blend sources.
