# P1 single-model results — Day-16 PM (final)

Branch: `claude/read-kaggle-handover-rsi2Q`. Plan doc:
`audit/2026-05-06-p1-single-model-plan.md`. ISSUES leaf: 8a.

## TL;DR

**P1 thesis (PI hypothesis "single model can close the gap")**: PARTIALLY
CONFIRMED. A single LGBM with Rozen-recipe kitchen-sink FE + CV target
encoding alone is **−12 bp below PRIMARY** (OOF 0.94970 vs 0.95090). BUT
its diversity (ρ=0.947 vs PRIMARY) makes it the **single most valuable
base-add of the entire competition**: when added as 22nd base to K=21
LR-meta, OOF lifts **+33.09 bp** to 0.95404. Gate vs PRIMARY: **Δ +31.37
bp, predicted LB +26 bp → ~0.95323** (within reach of top-5% 0.95345).

Submission ready: `submissions/submission_K22_add_p1_feA_te.csv`.
Awaiting PI sign-off.

## Phase 1 — replicate Rozen single LGBM

Recipe ported from `external/kernels/romanrozen/f1-pit-driver-race-year-encoding-0-95354.ipynb`:
- `make_features_A`: 50 engineered cols (tyre/compound algebra,
  race-progress, lag/rolling within (Driver,Race,Year), within-stint,
  3 combo categoricals).
- `cv_target_encode`: 6 high-card combos with smoothing 15-30
  (incl. Driver×Race×Year — the load-bearing single trick of the comp).
- LGBM hparams (Rozen): lr=0.025, num_leaves=255, min_child_samples=25,
  ff=0.65, λ1=1.2, λ2=2.5, max_depth=10, path_smooth=0.1, n_est=6000.

| Variant | Std OOF | ρ vs PRIMARY | Δ OOF vs PRIMARY |
|---|---:|---:|---:|
| (e3_hgbc — best prior single) | 0.94876 | — | −21.4 bp |
| feA_te (our recipe, no orig) | **0.94970** | 0.947466 | −12.0 bp |
| feA_te_orig (with aadigupta concat) | 0.94968 | 0.945984 | −12.2 bp |
| Reference: Rozen single LGB (T4 GPU) | 0.95241 | n/a | +15 bp |
| Reference: PRIMARY K=22 + Path-B | 0.95090 | 1.0 | 0 |

ρ(feA_te, feA_te_orig) = 0.991 OOF / 0.995 TEST → near-duplicates;
orig-concat adds zero standalone signal. Synth Driver codes (D109 etc.)
overlap with orig real codes (HAM, BOT, ...) only 31/887, so orig rows
cold-key on (Driver×Race×Year) TE features.

## Phase 2 — stack-add gate

`scripts/probe_min_meta.py` K=21 + N candidates with [raw, rank, logit]
expand:

| Pool | OOF | Δ vs K=21 baseline |
|---|---:|---:|
| K=21 baseline | 0.95073 | 0 |
| K=22 = K=21 + p1_feA_te | **0.95404** | **+33.09 bp** |

Gate vs PRIMARY (`scripts/p1_post.py K22_add_p1_feA_te`):
- standalone OOF: 0.95404 (Δ +31.37 bp)
- ρ vs PRIMARY: 0.986534
- predicted LB band: +26.37 bp → LB ~**0.95323**
- G3 flips: +→− 605, −→+ 435 (R7-eligible, >200)
- K=2 LR-meta(PRIMARY, K22_add) OOF: 0.95404 (saturated; +K22_add ≡ blend)
- DECISION: ✅ pursue submit (PI sign-off).

Pre-submit-diff vs PRIMARY (`scripts/pre_submit_diff.py`):
- 188098 / 188165 rows differ > 1e-6 (99.96%)
- 128767 (68.4%) differ > 1e-3
- max abs 0.384, mean 0.022, Spearman 0.985 ≤ 0.999 → DIFFERS

## ρ inventory (PRIMARY vs external submissions, for context)

| Pair | ρ |
|---|---:|
| PRIMARY (LB 0.95059) vs makimakiai_v8_solo | 0.96952 |
| PRIMARY vs makimakiai_blend (LB **0.95372**) | 0.98146 |
| PRIMARY vs gkanamoto tabM | 0.97860 |
| PRIMARY vs pavlo baseline (0.942) | 0.95470 |
| makimakiai_v8 vs makimakiai_blend | 0.98649 |

The +313 bp gap between PRIMARY (0.95059) and makimakiai_blend (0.95372)
at ρ=0.981 is mechanistically consistent with a single-model FE recipe
(~+150 bp standalone) plus blending with public-LB ensemble sources.

## Why was Rozen 0.95241 standalone unreproduced (only 0.94970)?

Rozen's notebook reports 118 tree features; our `make_features_A` has 50.
Possible gaps:
- Additional rolling windows (we have 3,5,7,10,15)
- Position-related interactions
- Race × Year × Compound 3-way TE (we have it; smoothing 15)
- Multi-seed LGBM bag (Rozen runs single seed, we match)
- GPU `device='gpu'` numerics differ slightly from CPU
- Rozen trains on FOLD train + ALL orig (we test this in feA_te_orig — null)

The standalone gap (~+27 bp Rozen vs ours) is interesting but **not
load-bearing for the submission**: K22_add_p1_feA_te already gives
+31 bp OOF lift even at our 0.94970 standalone.

## Why we missed this earlier (root cause)

Lessons committed to skill `improvements.md` (6 entries) and local
CLAUDE.md (R20-R23):
- R20 single-model-first / kitchen-sink FE before stacking
- R21 family falsification requires ≥3 variants
- R22 public-notebook scan at every plateau
- R23 framework is scaffolding, not authorship

See `audit/friction.md` 2026-05-06 PM section for friction tags.

## Submission decision (awaiting PI)

Submit candidate: `submissions/submission_K22_add_p1_feA_te.csv`
- Predicted LB ~0.95323 (band [+21, +33] bp at ρ=0.987)
- Top-5% threshold 0.95345 (within reach if upper band)
- Submit budget: 0/10 used today

If LB transfers cleanly: jumps from current PRIMARY (LB 0.95059) to
0.953+ in a single submit, closing 90%+ of the gap to top-5% in one
shot. PI sign-off needed per Rule 1.

## Files

- `scripts/p1_features.py` — feature factory + CV TE
- `scripts/p1_single_lgbm.py` — 4-variant trainer
- `scripts/p1_single_cb.py` — single CatBoost (not run; deferred)
- `scripts/p1_gate_all.py`, `scripts/p1_post.py` — gate harnesses
- `scripts/artifacts/oof_p1_single_lgbm_feA_te_strat.npy` (+test)
- `scripts/artifacts/oof_p1_single_lgbm_feA_te_orig_strat.npy` (+test)
- `scripts/artifacts/oof_K22_add_p1_feA_te_strat.npy` (+test)
- `submissions/submission_K22_add_p1_feA_te.csv` (✅ pre-submit-diff)
- `submissions/submission_p1_single_lgbm_feA_te.csv` (HEDGE; predicted LB regression)
- `submissions/submission_p1_single_lgbm_feA_te_orig.csv` (near-dup of feA_te)
- `external/kernels/{romanrozen,...}/` — 8 reference notebooks
