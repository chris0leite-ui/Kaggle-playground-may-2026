# 2026-05-08 — Kernel SVM family probe (ISSUES.md 1e)

**Result: FAIL on both K=10 sparse and K=27 PRIMARY pools. 8th rank-lock confirmation.**

## What was tested

Two full-data SVM variants on the vanilla-LR feature recipe (45 columns:
11 numeric standardised + Compound one-hot + Race one-hot + Driver
frequency-encoded + 4 cheap pair interactions). γ chosen from a 5-point
sweep at smoke (1-fold @ 50k rows): γ=0.02 was the sweet spot.

| variant | full OOF | fold std | total wall |
|---|---:|---:|---:|
| Nyström-RBF + LinearSVC (squared-hinge) | 0.91395 | 0.00119 | 1704 s |
| Nyström-RBF + LogReg (kernel-logistic) | 0.91203 | 0.00114 | 161 s |

Nyström `n_components=800` (reduced from 1500 after first run OOM-killed
with `n_jobs=-1` worker-process duplication). Decision-function passed
through sigmoid for the [0,1] OOF/test convention.

γ-sweep at smoke (linsvc, n_components=1500):

| γ | smoke OOF (50k, fold 0) |
|---:|---:|
| 0.01 | 0.91632 |
| **0.02** | **0.91820** |
| 0.04 | 0.91434 |
| 0.10 | 0.89704 |
| 0.50 | 0.84709 |
| 1.0 | 0.81475 |

## Gate results (probe.py / scripts/svm_gate.py)

Both pools: LR-meta with raw + rank + logit feature expansion, 5-fold
StratifiedKFold (matches `scripts/probe_min_meta.py`).

### K=10 forward-selected core (E9 pick order from `scripts/t2_k10_primary.py`)

| variant | base OOF | +1 OOF | Δ bp |
|---|---:|---:|---:|
| linsvc | 0.95381 | 0.95380 | **−0.09** |
| klogreg | 0.95381 | 0.95381 | **−0.06** |

### K=27 PRIMARY pool (`d18_path_b_K27_v4h1d_d16_d18_e2_f2`)

| variant | base OOF | +1 OOF | Δ bp |
|---|---:|---:|---:|
| linsvc | 0.95428 | 0.95428 | **−0.02** |
| klogreg | 0.95428 | 0.95428 | **−0.00** |

## Diagnostics

- ρ_test vs PRIMARY: **0.83** (linsvc), **0.82** (klogreg).
  Moderately diverse but not extreme — smoke ρ was 0.68–0.75, dropped
  to 0.82–0.83 once trained on full 350k rows.
- G3 flip ratio at top-1%:
  - linsvc: **0.000** (0 / 1882 — SVM never picks rare positives that
    PRIMARY misses; SVM's top-1% is a strict subset of PRIMARY's).
  - klogreg: **0.128** (4797 / 615 — calibrated probabilities make
    asymmetric rare-class picks; doesn't translate to meta lift).
- Standalone OOF gap to LR-mega ceiling 0.92776: **−138 bp** (linsvc),
  **−157 bp** (klogreg).
- Standalone OOF gap to PRIMARY 0.95431: **−403 bp** (linsvc),
  **−423 bp** (klogreg).
- Standalone OOF lift over matched-feature linear LR (0.85588):
  **+58 bp** (linsvc), **+56 bp** (klogreg) — the kernel non-linearity
  does real work on this 45-feature recipe; it's just not enough work
  to close the GBDT-class gap.

## Mechanism

Kernel-class introduces local-similarity structure that linear-class
cannot represent. The +56–58 bp standalone lift over matched-feature
linear LR confirms this. But standalone AUC is bounded by the
representational ceiling of the 45-feature recipe — well below the
GBDT-yekenot ceiling at 0.95+. When the meta-stacker sees a base that
is 400 bp behind on row-AUC, the LR coefficient on that base
collapses toward zero, regardless of structural-distance signal.

## Friction

`kernel-class-fails-when-standalone-AUC-gap-to-gbdt-exceeds-300bp`.
Codifies the 8th rank-lock confirmation. Generalises beyond kernel-SVM:
any new family whose standalone AUC is far below the GBDT-yekenot
baseline cannot fire at the meta on this comp, even with structural
diversity.

## What would change the verdict

Three escalation paths the harness flagged but did not run, ordered by
expected EV:

1. **Yekenot-recipe transfer to SVM** — apply the same feature-recipe
   transformation that lifted CatBoost +24 bp (KBins-discretised
   numerics, count encoding, floor-categorical features). Could lift
   standalone from ~0.91 to ~0.93; still likely below GBDT-class but
   closes some of the 400 bp gap. Cost: ~2 h CPU.
2. **Per-Compound RBF-SVC specialists** — exact `libsvm.SVC` per
   Compound level (5 splits, ~50–100k rows each — exact kernel SVM is
   tractable at this size). Echoes the per-Compound LR result (which
   fired at the LR-class but not the meta). Cost: ~3 h CPU.
3. **Polynomial-kernel sketch (degree 2/3)** — `PolynomialCountSketch
   + LinearSVC`. Encodes feature-space interactions that GBDT
   discovers via splits. Different angle from RBF. Cost: ~1 h CPU.

Given the 400-bp standalone gap is the binding constraint and Path 1
addresses it most directly, Path 1 is the strongest escalation if the
PI elects to revisit the family.

## Artifacts

- `scripts/svm_kernel_probe.py` — three-variant probe driver
  (rff_sgd_hinge, nystroem_linsvc, nystroem_klogreg).
- `scripts/svm_gate.py` — K=10 sparse + K=27 dense gate report.
- `scripts/artifacts/oof_svm_nystroem_linsvc_g0.02_strat.npy`,
  `scripts/artifacts/test_svm_nystroem_linsvc_g0.02_strat.npy`.
- `scripts/artifacts/oof_svm_nystroem_klogreg_g0.02_strat.npy`,
  `scripts/artifacts/test_svm_nystroem_klogreg_g0.02_strat.npy`.
- `audit/decisions.jsonl` — BOTE prediction + −0.05 bp recorded outcome.
