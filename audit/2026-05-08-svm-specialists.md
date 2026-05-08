# 2026-05-08 — SVM specialists probe (PI-requested follow-up)

**Result: all 5 variants null at K=27 PRIMARY pool. Strongest is
gaussian-kernel-per-Year at K=10+1 +0.05 bp / K=27+1 +0.00 bp.
Family closed across global + specialist axes.**

## What was tested

After the global kernel-SVM probe came back null on both pools (see
`audit/2026-05-08-svm-kernel-probe.md`), the PI asked to try cheap
linear and per-segment specialist variants.

Five variants on the same 45-feature vanilla-LR recipe used by
`scripts/svm_kernel_probe.py`:

| variant | description | full OOF | wall |
|---|---|---:|---:|
| `linear_global` | one LinearSVC on all fold-train rows | 0.85388 | 44 s |
| `linear_per_year` | one LinearSVC per Year (4 levels) | 0.85578 | 62 s |
| `linear_per_compound` | one LinearSVC per Compound (5 levels) | 0.86178 | 41 s |
| `linear_per_stint` | one LinearSVC per Stint clipped 1..5 | 0.86307 | 42 s |
| `rbf_per_year` | Nyström-RBF + LinearSVC per Year | 0.90950 | 1229 s |

For specialists: each fold trains one model per segment level on
training rows of that level only. Predictions for a row come from the
model fit on that row's segment. Decision scores → sigmoid for the
[0,1] OOF/test convention.

## Gate results (vs K=10 sparse + K=27 PRIMARY)

`scripts/svm_gate.py` — LR-meta with raw + rank + logit feature
expansion, 5-fold StratifiedKFold.

| variant | std OOF | ρ_test | G3 flip | K=10+1 Δ | K=27+1 Δ |
|---|---:|---:|---:|---:|---:|
| linear_global | 0.85388 | 0.681 | 0.00 | +0.02 | −0.09 |
| linear_per_year | 0.85578 | **0.548** | 0.00 | −0.04 | −0.02 |
| linear_per_compound | 0.86178 | 0.725 | 0.00 | +0.02 | −0.05 |
| linear_per_stint | 0.86307 | 0.733 | 0.00 | +0.01 | −0.05 |
| **rbf_per_year** | **0.90950** | 0.762 | 0.02 | **+0.05** | **+0.00** |

## Findings

1. **Every variant nulls at K=27 PRIMARY.** Best is rbf_per_year at
   +0.00 bp (full absorption). The −0.05 to −0.09 bp regressions on
   the linear variants are within meta-fit noise.
2. **Per-segment specialisation gave standalone lift but no meta
   lift.** linear_per_stint was +92 bp standalone over linear_global
   (0.853 → 0.863) but the meta-stacker can't extract gain from a
   base that's still 900+ bp behind PRIMARY.
3. **Lowest ρ_test in project history.** linear_per_year hit ρ_test
   0.548 vs prior project low of 0.71 (bagged LR). Both nulled — 9th
   rank-lock confirmation that diversity alone is not sufficient
   meta-utility on this stack.
4. **rbf_per_year doesn't beat rbf_global.** Standalone 0.90950 vs
   0.91395 for the global kernel SVM → per-year stratification costs
   ~5 bp at the kernel-class level (smaller training data per model
   outweighs the year-conditional signal).
5. **The kernel-class is +56 bp over matched-feature linear at the
   standalone level** (0.914 vs 0.854) but the gap to GBDT-class
   PRIMARY is still −400 bp.

## Net read

Combined with the earlier `audit/2026-05-08-svm-kernel-probe.md` runs
(global LinearSVC and global kernel-logistic), 7 distinct SVM-family
configurations have now been falsified on this comp. The K=27 stack
absorbs everything SVM-class can produce on the vanilla 45-feature
recipe.

Friction `low-rho-alone-is-not-meta-utility` formalised in CLAUDE.md
already (5+ confirmations); this is the 9th.

## What would change the verdict (untouched)

The 400-bp standalone gap is the binding constraint. Three escalation
paths the harness flagged but did not run:

1. **Yekenot feature-recipe transfer to SVM** — apply the same
   feature transformation that lifted CatBoost +24 bp at K=21+1
   (KBins-discretised numerics, count encoding, floor categoricals).
   Closes some of the standalone gap. ~2 h CPU.
2. **Polynomial-kernel sketch (degree 2/3)** — explicit pair- /
   triplet-feature interactions instead of local-similarity RBF.
   Different angle from gaussian. ~1 h CPU.
3. **Per-Compound exact RBF SVC** — `libsvm.SVC` per Compound
   (~50-100k rows each, exact kernel SVM is tractable). ~3 h CPU.

Given the consistent absorption by K=27, the harness's prior on these
escalation paths drops to roughly 5-10 % probability of meta-positive
lift; the SVM family is empirically closed on this comp.

## Artifacts

- `scripts/svm_specialists.py` — 5-variant probe driver.
- `scripts/artifacts/oof_svm_linear_global_strat.npy` etc. (10 files).
- This audit note.
