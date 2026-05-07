# VALIDATED — yekenot RealMLP-PyTabKit

This notebook is a **validated reference recipe** for s6e5
(playground-series-s6e5, F1 PitNextLap binary AUC). Keep as a
permanent example for cross-comp recipe library.

## Validation outcome (2026-05-07, branch `claude/read-handover-62BCt`)

Replicated faithfully via `scripts/d17_h1d_yekenot_full_recipe.py` on
4-core CPU box with `n_ens=4` (vs yekenot's GPU `n_ens=24`):

| Metric | Yekenot pub | Our replication | Gap |
|---|---:|---:|---:|
| 5-fold StratKF OOF AUC | 0.95273 | **0.95257** | **−1.6 bp** (within fold variance) |
| Per-fold AUC | n/a | [0.95366, 0.95153, 0.95232, 0.95189, 0.95375] | mean 0.95263 |
| Standalone vs default-config realmlp | n/a | +675 bp (0.94582 → 0.95257) | n/a |
| K=22 ADD via canonical LR-meta | n/a | OOF 0.95355 (+28.16 bp over K=21 baseline 0.95073) | n/a |
| K=21 SWAP (drop default realmlp slot) | n/a | OOF 0.95354 (+28.08 bp) | n/a |
| ρ vs current PRIMARY (d16 cont_only Path B τ=20k) | n/a | **0.97180** standalone / **0.98728** at K=22 stack | first base to break ρ < 0.99 in 5+ months |

Wall: 35 min for 5-fold @ n_ens=4 + orig + full FE pipeline + TE.

## Why this notebook is the reference

1. **Highest published OOF on s6e5** (0.95273) at the time of audit;
   pit-or-stay-f1-strategy-1 cites it as "the proven public 0.95273
   model family".
2. **Recipe is reproducible from the notebook source** — no hidden
   datasets, no auth-walled dependencies, no custom infra. The full
   FE + hyperparameter pack are all in cell 6 + cell 8 + cell 10.
3. **All 6 load-bearing FE items documented** in
   `.claude/skills/kaggle-comp/examples/fe-recipe-yekenot-realmlp-kitchen-sink.md`
   for cross-comp use.
4. **Strict-OOF compliant** — `TargetEncoder(cv=5)` uses sklearn's
   internal cross-fitting; no val-row labels leak into TE features.
   Per-fold orig concat is independent supervised samples (not
   target-aggregated). Passes Rule 24 audit. Passes Rule 25 (AV-AUC
   train/test = 0.502).
5. **Cross-comp portable** — the FE pipeline (arithmetic ratios,
   floor-cat, count-encoding, KBins(N) on selected numerics, 2-way
   combo cats with CV TE) generalises to any synthetic-tabular
   Playground series with engineered cat structure.

## Do NOT delete

This notebook + the FE recipe doc are now permanent project assets.
End-of-comp wrap-up should:
- Promote the recipe doc to skill `examples/` (already done at
  `.claude/skills/kaggle-comp/examples/fe-recipe-yekenot-realmlp-kitchen-sink.md`).
- Cross-link from the next-comp's kickoff `comp-context.md` under
  "recipes to try first".
- Tag this as `recipe-reference-validated` in the friction log.

## Lessons learned from this validation cycle (s6e5 Day-17 PM)

- **`recipe-gap-misdiagnosis-when-public-author-FE-not-fully-replicated`**
  — H1 v1/v2/v3 all NULL because we deployed the hyperparameter pack
  alone (and orig-merge in V3) without the full FE. +69 bp standalone
  gap is the FE pipeline (especially CV TE on 2-way combos), not the
  hyperparameters. Read the FULL notebook source before BOTE.
- **2-way TE on `(Race, Compound)` + `(Race, Year)` is more robust than
  3-way `(Driver, Race, Year)`** in NN+CV-TE setting. Rozen's 3-way is
  the famous +200 bp single-LGBM trick but is leak-prone in
  per-fold-aggregated-on-full-train setups (s6e5 Day-17 P1 falsified).
- **`torch.set_num_interop_threads` can only be called once per
  process** — don't put it inside a fold loop.
- **n_ens=4 captures most of n_ens=24** when the FE pipeline is
  complete. Yekenot 0.95273 (n_ens=24, GPU) vs ours 0.95257 (n_ens=4,
  CPU) gap is −1.6 bp = within fold variance. Ensemble size is
  secondary; FE is primary.

## Files this references

- `scripts/d17_h1d_yekenot_full_recipe.py` — our replication script
- `scripts/artifacts/oof_d17_h1d_yekenot_full_strat.npy` + test pair
- `scripts/artifacts/d17_h1d_yekenot_full_results.json` — fold AUCs +
  walls
- `audit/2026-05-07-d17-h1-verdict.md` — diagnosis of why H1 v1/v2/v3
  failed before this full-recipe success
- `.claude/skills/kaggle-comp/examples/fe-recipe-yekenot-realmlp-kitchen-sink.md`
  — portable FE recipe for future comps
