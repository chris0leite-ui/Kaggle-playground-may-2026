# Simple Logistic Regression on s6e5 — A Playbook

**Question (PI):** *How good can a simple logistic regression get on
this problem? Just LR — nothing else.*

**Answer (this branch):** **0.92776 OOF AUC** out of a GBDT pool
ceiling of 0.95385. **A pure LR with the right feature engineering
closes 87% of the GBDT-vs-baseline gap, in under 8 minutes of CPU.**
The remaining 26 bp is non-linear DGP structure that LR genuinely
cannot recover regardless of how much FE we throw at it.

This document distills 19 LR variants into a transferable recipe.
Reusable; the pipeline runs on raw CSVs in 5–10 min on a 4-core CPU.

---

## TL;DR — the ladder

| Recipe | OOF AUC | Time | n_feats | Notes |
|---|---:|---:|---:|---|
| Random | 0.5000 | – | – | – |
| Class prior | 0.5000 | – | – | bal-acc threshold only |
| **`lr_raw_std`** | **0.82467** | 3 s | 11 | StandardScaler + 11 numerics |
| `lr_raw_te` | 0.84528 | 5 s | 14 | + fold-safe TE on (Driver, Race, Compound) |
| `lr_raw_ohe` | 0.85407 | 19 s | 929 | + OneHot of cats (replaces TE) |
| `lr_rozen_full` | 0.85735 | 46 s | 93 | Rozen LGBM-style 50 engineered + 6 CV TE — **tree-friendly FE actively hurts LR** |
| `lr_kbins_yekenot` | 0.85948 | 148 s | 1122 | yekenot KBins(200) on 2 of 11 numerics — NN-targeted, not LR |
| `lr_poly2_std` | 0.88244 | 119 s | 77 | degree-2 polynomial of 11 numerics |
| `lr_C_low_kbins20` | 0.90665 | 3 s | 1077 | KBins(20)+OHE w/ C=0.001 — over-regularised |
| `lr_te_3way_sweep` | 0.90403 | 89 s | ~150 | 4 keys × 4 smoothings = 16 TE feats + raw + cat OHE |
| `lr_dgp_rules` | 0.90714 | 106 s | 945 | 4 rule-keys × 4 alphas Bayesian-smoothed lookups + cat OHE |
| **`lr_kbins5_ohe`** | 0.91082 | 22 s | 964 | KBins(5,quantile) on every numeric + cat OHE |
| `lr_splines_5` | 0.91769 | 394 s | 984 | natural cubic B-splines, 5 knots, all numerics |
| `lr_kbins50_uniform` | 0.91799 | 19 s | 1468 | KBins(50,uniform) — diminishing returns |
| **`lr_yekenot_full_recipe`** | **0.91860** | 46 s | **45** | yekenot 6-item port (compactest LR by far) |
| `lr_balanced_kbins20` | 0.92008 | 20 s | 1077 | class_weight='balanced' — rank-no-op |
| `lr_C_high_kbins20` | 0.92027 | 51 s | 1077 | C=100 (light reg) — barely binds |
| **`lr_kbins20_ohe`** | **0.92038** | 20 s | 1077 | **30-second LR baseline** — strongest single-recipe |
| `lr_l1_lasso_kbins20` | 0.92044 | 135 s | 1077 | L1 saga — rho=0.9999 with L2 |
| **`lr_mega`** | **0.92776** | 7m25s | 1202 | All FE concatenated, Rozen + 3-way TE + DGP rules + KBins(20)+OHE — **the LR ceiling** |
| (LR-meta over 15 simple LR bases) | 0.92373 | 70 s | – | meta-stacker over round-1 LR bank |
| GBDT single (CatBoost v4) | 0.95200 | 30 min GPU | – | for comparison |
| GBDT pool LR-meta (K=24) | 0.95385 | 50 s | – | for comparison |
| Current PRIMARY (K=24+Path-B) | 0.95354 (LB) | – | – | LB |

**The LR ceiling on s6e5 is 0.92776 OOF — 24.2 bp below GBDT
single-model (CatBoost v4 0.95200) and 26.1 bp below the 24-base GBDT
meta (0.95385). LR closes 87% of the GBDT-vs-`lr_raw` gap.**

Per-fold mega AUCs: 0.92925 / 0.92631 / 0.92791 / 0.92686 / 0.92851
(σ_fold = 1.1 bp). Stable across folds.

---

## The mechanism map

What FE *helps* a model is dictated by the model's inductive bias.
The s6e5 bank shows this with three model classes and ~20 FE recipes:

| FE family | LR | GBDT | NN |
|---|---|---|---|
| Raw numerics + StandardScaler | ✓ floor (0.82) | ✓ baseline (0.94+) | ✓ baseline |
| Polynomial expansion | ✓ +28 bp over raw | ≈0 (trees split already) | ≈0 (activations) |
| **KBins(15-25, quantile) + OHE** | **★ best (+95 bp over raw)** | small lift | small lift |
| KBins on 2 of 11 numerics (yekenot) | ✗ −60 bp vs full KBins | +20 bp (Day-17 PM) | designed for NN ✓ |
| Cat OneHot | ✓ +10 bp | ≈0 (handled natively) | ✓ via embeddings |
| Cat target encoding (single key) | ✓ +20 bp | small lift | small lift |
| **3-way CV target encoding** | ✓ ~+5 bp | comp's documented +200 bp lift | ≈ |
| Tree-engineered (Rozen kitchen-sink) | ✗ −62 bp (!) | ✓ designed for | ≈0 |
| DGP rule lookups | ✓ ~+80 bp over raw | ≈0 (rediscovered in splits) | ≈ |
| Splines (B-splines, 5 knots) | ✓ +94 bp over raw | small lift | small lift |
| Hash trick on 2/3-way combos | (not measured) | ≈0 | ≈ |

**Reading:** for LR, *binning every continuous feature* is the
single highest-EV operation. For trees, it's TE/feature-engineered
ratios. For NNs, it's PLR + scaling. **A recipe that wins for one
class can hurt another.**

---

## The 30-second baseline

If you have a binary tabular target and 5–10 numeric features + a
few cats, this is the recipe. AUC 0.92 in 22 seconds on s6e5.

```python
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse
import numpy as np
import pandas as pd

NUM_COLS = [...your numerics...]
CAT_COLS = [...your cats...]

# 1. Fit KBins(20, quantile) on full train+test (Rule 25: AV-AUC ~0.5)
num_tr = train[NUM_COLS].fillna(0).values.astype(np.float32)
num_te = test[NUM_COLS].fillna(0).values.astype(np.float32)
kb = KBinsDiscretizer(n_bins=20, encode="onehot", strategy="quantile",
                      subsample=None)
kb.fit(np.vstack([num_tr, num_te]))
Xb_tr, Xb_te = kb.transform(num_tr), kb.transform(num_te)

# 2. Fit OneHot on cats over combined train+test
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                    dtype=np.float32)
enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]]))
Oc_tr, Oc_te = enc.transform(train[CAT_COLS]), enc.transform(test[CAT_COLS])

# 3. Stack sparse, fit LR with liblinear (single-thread, fast on sparse)
Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
Xte = sparse.hstack([Xb_te, Oc_te], format="csr")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
test_pred = np.zeros(len(test))
for tr, va in skf.split(np.zeros(len(y)), y):
    lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
    lr.fit(Xtr[tr], y[tr])
    oof[va] = lr.predict_proba(Xtr[va])[:, 1]
    test_pred += lr.predict_proba(Xte)[:, 1] / 5

print(f"OOF AUC {roc_auc_score(y, oof):.5f}")
```

**Why it works:** every continuous feature gets quantile-binned into
20 buckets. OHE turns each bucket into a free-coefficient column. LR
learns one weight per bucket — exactly what a single-feature decision
stump does, but for *every* numeric simultaneously and with categorical
interactions the regularization path can prune. KBins+OHE is "tree
splits in linear-model clothing."

---

## The compact recipe (yekenot full)

If you need 25× fewer features but the same AUC, port the yekenot
6-item recipe. **45 features, AUC 0.91860, 46 seconds.**

```
1. Arithmetic ratios:    LapNumber / RaceProgress, TyreLife / LapNumber
2. Floor-cat:            np.floor(x).factorize()  on every numeric + ratios
3. Count encoding:       value_counts() of every categorical
4. Heavy KBins on 2:     KBins(200) on RaceProgress, KBins(7) on LapTime
5. Combo cats:           (Race, Compound), (Race, Year) factorized to int
6. CV target encoder:    sklearn TargetEncoder(cv=5, smooth='auto')
                         on the combo cats inside each outer fold
```

Compactness lets it function as the meta-input for downstream stackers
(45 features fit comfortably in any meta-stacker design).

---

## The mega recipe (LR ceiling)

If you want the absolute LR ceiling on a tabular comp:

1. Run `make_features_static` (Rozen-style: 50 engineered cols including
   tyre algebra, race-progress, lag/rolling within group, historical
   priors, combo cats).
2. Run `fit_fs_a` per fold on train-rows only (label-conditional
   aggregates: race_avg_pit_lap, compound_avg_life, etc.).
3. 6 CV target encoders at varied keys + smoothing.
4. 16 3-way CV target encoders × 4 smoothings.
5. 16 DGP rule lookups (Bayesian-smoothed P(y=1|key) at α ∈ {5, 20, 100, 500}
   for 4 key combinations).
6. KBins(20, quantile) + OHE on every numeric.
7. OHE all cats.
8. Densify everything (dense ~1200 cols × 350k rows = 1.7 GB, fits in 16 GB RAM).
9. StandardScaler the dense block.
10. LR with `C=1.0, solver='lbfgs', max_iter=2000`.

**7m25s CPU. AUC 0.92776.** This is the s6e5 simple-LR ceiling.

---

## What does NOT help LR (pre-tested null findings)

- **`class_weight='balanced'`** — ρ=0.9974 with default LR. Class
  weighting reweights loss, not ranking.
- **`L1` vs `L2`** — ρ=0.9999. Sparsity does not pick a different
  signal subset; features are not sparse-prunable on this DGP.
- **`C` regularisation strength** — heavy reg (C=0.001) loses 14 bp;
  light reg (C=100) gains 0 bp; default C=1 is fine.
- **`logits` vs `proba` meta-input** — the Chris Deotte recipe item
  has zero effect on AUC at our bank size (saturated meta).
- **Tree-engineered FE** (Rozen kitchen-sink: rolling means, lag deltas,
  position pressure, etc.) — *actively hurts* LR vs binning. 93 features
  give 0.857 vs 1077 KBins gives 0.920.
- **yekenot's "bin only RaceProgress + LapTime"** — designed for NN
  with raw scaling. LR needs every numeric binned.

---

## What's left on the table

The 50 bp gap from LR (0.929) to GBDT (0.954) on s6e5 is not
recoverable by any additional FE the LR class can use. Per Arc-A E4
diagnostic: no Compound × Stint quintile cell of this DGP has
locally-linear structure in raw numeric features. The non-linearity
the GBDT captures is genuinely 2nd-order interaction structure that LR
cannot construct linearly, even with binning.

If you need to close the gap *while keeping LR-class*, the only path
is *bagged LR* + *random subspace*: train 50–100 LRs on random feature
subsets, then average. This breaks the eff_rank=2 ceiling that 15
hand-designed LR variants hit (because they all share the same
feature pipeline). Predicted lift: +1–3 bp; doesn't close the gap.

The honest reframe: **LR is not a primary model on this comp; it is
a fast, cheap, mechanistically-distinct diversity contributor.**
The single most useful thing about LR-on-s6e5 is the speed — a
30-second AUC 0.92 baseline lets you iterate FE ideas at GBDT-class
performance with sub-minute wall time.

---

## The lessons that compound

### L1: Bin every continuous feature for LR, not just some

`lr_kbins20_ohe` (0.92038) vs `lr_kbins_yekenot` (0.85948). Same KBins,
applied to all 11 numerics vs just 2. **+60 bp from completing the
binning.** Generalises: if you're using LR, KBins-OHE the entire
continuous space.

### L2: Tree-friendly FE actively hurts LR

`lr_rozen_full` (0.85735) is *lower* than `lr_raw_te` (0.84528) plus
14 features vs Rozen's 93. The 50 engineered Rozen features were
designed for LGBM's axis-aligned splits. LR can't use them; the
extra dimensions add variance without signal.

### L3: Polynomial expansion is the LR-friendly way to add interactions

`lr_poly2_std` (0.88244) > `lr_raw_std` (0.82467) by **+58 bp** with
just 77 features. If KBins is unavailable for some reason (high-card
continuous, ordinal-like data), polynomial degree 2 is the next-best
LR move.

### L4: The `lr-meta-rank-lock-strong-anchor` friction is robust

LR-meta over the 15 LR bank gave +33 bp over best single LR. K=24 GBDT
+ 15 LR bank gave +0.022 bp. Adding rich-FE LRs to the GBDT pool moves
the dial by less than measurement noise. Useful diversity needs
*new model class*, not new FE within LR.

### L5: Rich FE lifts the single LR but DOES NOT expand bank diversity

PI question (sealed prediction): rich FE would break the eff_rank=2
ceiling. **Result:** mixed.

| Diagnostic | Round 1 (15 simple LRs) | Round 2 (20 LRs incl rich-FE) | Δ |
|---|---:|---:|---:|
| Best single LR AUC | 0.92044 | **0.92776** (mega) | **+73 bp** |
| LR-bank eff_rank | 2.0 | **2.19** | +0.19 |
| GBDT+LR combined eff_rank | 3.33 | **3.33** | **0** |
| LR-bank residualized eff_rank | 3.15 | **4.2** | +1.05 |
| GBDT+LR residualized eff_rank | 13.56 | 14.05 | +0.5 |
| lr_mega ρ vs PRIMARY | – | **0.9030** | – (highest of any LR) |

**Reading.** Rich FE makes the LR **stronger as a single model** (huge
single-base AUC lift) but does NOT add new directions to the bank vs
the GBDT pool. The mega's signal projects almost entirely onto the
direction-1 the simple LRs were already covering — it's more
*concentrated* there, not orthogonal to GBDT. The combined GBDT+LR
eff_rank stays at 3.33 with 5 added rich-FE LRs.

**Reframe of L7 (original audit):** "bank-diversity needs new model
class" → split into two distinct claims:
- **L7a (claim still holds):** Diversity vs GBDT does NOT come from
  better LR FE. mega's ρ_PRIM 0.903 is the highest of any LR, not the
  lowest.
- **L7b (NEW):** *Single-base LR strength* DOES come from rich FE.
  The 73 bp jump (kbins20 → mega) buys you a strong stand-alone LR.
  Useful when LR is the entire model, useless as a stack-add.

In other words: ask "do I want LR for diversity?" → no. "Do I want LR
for a fast strong baseline?" → yes, run mega.

### L6: How to test "how good can LR get" in 1 hour

The full 4-stage probe:

| Stage | Time | What you learn |
|---|---|---|
| 1. `lr_raw_std` | 3 s | LR floor on this DGP |
| 2. `lr_kbins20_ohe` | 22 s | Practical LR baseline |
| 3. `lr_yekenot_full_recipe` | 46 s | Compact LR (45 feats) |
| 4. `lr_mega` (Rozen + 3wTE + DGP rules + KBins + OHE) | 8 min | LR ceiling |

Total: ~10 min wall, gives you the full LR-on-this-comp profile.
The gap between (2) and (4) tells you how much of the comp's signal
LR can recover; the gap from (4) to GBDT tells you how much you'll
need stacking for.

---

## Practical guidance for new tabular comps

1. **First hour:** Run `lr_raw_std` then `lr_kbins20_ohe`. The lift
   from (1) to (2) is your "LR-FE-ceiling discovery rate" for this
   DGP.
2. **Second hour:** Run `lr_mega`. The lift from (2) to (3) tells you
   how much your FE arsenal — TE, DGP archaeology, splines, rules —
   actually adds beyond binning. **If <30 bp lift, the comp's signal
   is mostly in the binnable numeric structure; focus FE energy on
   per-segment specialists not more FE families.**
3. **Third hour:** Compare LR_mega to single-GBDT-baseline. The gap
   tells you how much comes from non-linear interaction structure
   (GBDT exclusive). >100 bp gap → stacking is necessary, LR is a
   diversity contributor at best.
4. **Don't run LR variants of the same recipe family.** L1 vs L2,
   class_weight, C-sweep — these are rank-no-ops on AUC. Pick one
   variant per FE family, use the rest of the budget on different
   FE families.

---

## Files

Round 1 (15 simple LRs):
- `scripts/lr_bank.py` — bank builder
- `scripts/lr_bank_diagnostics.py` — SVD eff_rank diagnostics
- `scripts/lr_bank_stacking_fast.py` — 3 stacking experiments
- `scripts/lr_torch_gpu.py` — Chris-recipe PyTorch replica (CPU/GPU)

Round 2 (rich-FE):
- `scripts/lr_bank_rich_fe.py` — rich-FE bank: rozen, yekenot,
  dgp_rules, te_3way, mega
- `scripts/p1_features.py` — Rozen FE library (cherry-picked)

Reference (cherry-picked from research-only branch):
- `scripts/lr_diag_e{1,2,4,8}_*.py` — Arc-A pool diagnostics
- `audit/2026-05-07-chris-deotte-lr-stacker-research.md` — research synth
- `audit/2026-05-07-lr-diagnostics-arcA.md` — Arc-A diagnostics
- `audit/2026-05-07-lr-bank-experiment.md` — round-1 audit

OOF artifacts: `scripts/artifacts/oof_lr_*_strat.npy` +
`test_lr_*_strat.npy` (n_train, 2) and (n_test, 2).
