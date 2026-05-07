# LR-diagnostics Arc C — new LR-population bases (2026-05-07)

Branch `claude/ensemble-logistic-regression-research-MbLKu`. Arc C
asks: can any LR-population construction add meta-utility to the
K=10 core (per Arc B / E9)? Two structurally distinct constructions
tested. Both null. **Cross-confirmation strong.**

Scripts: `scripts/lr_diag_a{2,4}_*.py`. Artifacts:
`scripts/artifacts/{oof,test}_a{2_vanilla,2_rich,4_per_compound}_strat.npy` +
`lr_diag_a{2_bagged_lr,2_gate,4_per_segment}.json`.

## A2 — Bagged-LR base, vanilla vs rich

| Variant | OOF AUC | features | ρ vs PRIMARY |
|---|---:|---:|---:|
| vanilla | 0.85588 | 47 | 0.6884 |
| **rich** (+9 E6 Stint-cross) | **0.86817** | 56 | 0.7099 |

**+122.9 bp** standalone from adding the 9 E6-identified Stint-cross
interactions. Validates E6's interaction-hub diagnosis.

Per-interaction stability in rich (sign-flip = 0; SNR > 5 = real):

| interaction | coef mean | SNR |
|---|---:|---:|
| Stint × TyreLife | +0.73 | 32.2 |
| Stint × LapTime | +1.63 | 17.8 |
| Stint × LapTime_Delta | −0.69 | 16.0 |
| Stint × Cumulative_Degradation | −0.26 | 9.5 |
| Stint × Year | −0.37 | 7.6 |
| Stint × Position | −0.28 | 6.5 |
| Stint × RaceProgress | −3.77 | 6.3 |
| Stint × LapNumber | +1.04 | 1.9 (collinear) |
| LapTime × LapTime_Delta | +1.26 | **0.4** (noise) |

Two of E6's flagged pairs (Stint × LapNumber collinear; LapTime ×
LapTime_Delta unstable across folds) failed to land cleanly.
**Reusable lesson:** cell-level residual magnitude is necessary but
not sufficient for usable LR signal — collinearity and fold stability
matter too. Filter E6's residual rank by SNR + sign-flip.

### A2 + K=10 gate

| Config | K | OOF AUC | Δ vs K=10 | ρ vs PRIMARY |
|---|---:|---:|---:|---:|
| K=10 baseline | 10 | 0.95381 | — | — |
| K=10 + A2_vanilla | 11 | 0.95381 | +0.00 bp | 0.99879 |
| K=10 + A2_rich | 11 | 0.95381 | +0.04 bp | 0.99882 |
| K=10 + both | 12 | 0.95381 | −0.01 bp | 0.99884 |
| K=10 swap FMA → A2_rich | 10 | 0.95379 | −0.15 bp | — |

**Result: meta-null** despite A2_rich having lowest standalone ρ to
PRIMARY of any base we've ever produced (0.71).

## A4 — per-Compound LR specialists

5 LRs, one per Compound; each fit on rows of that compound only;
predictions concatenated. Same rich features as A2_rich.

| Compound | n_train (fold 0) | va AUC |
|---|---:|---:|
| HARD | 136,330 | 0.8635 |
| INTERMEDIATE | 13,948 | 0.8783 ← strongest specialist |
| MEDIUM | 168,888 | 0.8694 |
| SOFT | 31,081 | 0.8148 |
| WET | 1,065 | 0.6766 ← small-data limit |

Combined OOF AUC: **0.87386** (+59 bp over A2_rich).
ρ(A4, PRIMARY) = 0.74777.

### A4 + K=10 gate

| Config | OOF AUC | Δ vs K=10 |
|---|---:|---:|
| K=10 baseline | 0.95381 | — |
| K=10 + A4 | 0.95381 | +0.03 bp |
| K=10 + A2_rich + A4 (K=12) | 0.95381 | +0.01 bp |

**Result: meta-null.** Cross-confirmed via structurally distinct
construction.

## Synthesis (Arc C, load-bearing)

**Two independent LR-population constructions both null on K=10:**

| Construction | Standalone AUC | ρ vs PRIM | Δ on K=10 |
|---|---:|---:|---:|
| A2_vanilla (global linear LR) | 0.856 | 0.688 | +0.00 bp |
| A2_rich (global LR + Stint-cross) | 0.868 | 0.710 | +0.04 bp |
| A4 (per-Compound specialists) | 0.874 | 0.748 | +0.03 bp |
| A2_rich + A4 combined | — | — | +0.01 bp |

**The durable lesson** (refines `rho-alone-insufficient-for-meta-utility`
from 4 cross-confirmations to 6):

> When a saturated GBDT pool has captured a DGP's interaction
> structure, no representation-only base addition (LR/FM/NN of any
> flavor on the same features) can lift the meta. Lift requires
> either (a) **new information** — external data we don't have —
> or (b) **new meta-architecture** — Path-B redesign on existing
> pool. Low ρ is necessary but not sufficient; the orthogonality
> must come from new structural information, not just from row-level
> prediction errors on the same information.

This is a clean, falsifiable, reusable lesson — applies to any
tabular comp where a GBDT pool has saturated the metric.

## Reusable artifacts (Day-1 of any tabular comp)

LR-diagnostic suite (Arcs A+B+C combined):

| script | answers | cost |
|---|---|---:|
| `lr_diag_e1_svd.py` | Pool effective rank; redundancy root cause | 1 min |
| `lr_diag_e2_calibration.py` | Per-base calibration; mis-calibrated bases | 30 min |
| `lr_diag_e4_per_segment.py` | Locally-linear cells | 45 min |
| `lr_diag_e5_bootstrap_coef.py` | Stable signal vs noise features | 10 min |
| `lr_diag_e6_residual_interactions.py` | Dominant interaction hub | 5 min |
| `lr_diag_e8_grid.py` | Meta hyperparameter ceiling | 5 min |
| `lr_diag_e9_forward_select.py` | True effective pool size | 15 min |
| `lr_diag_a2_bagged_lr.py` + `_gate.py` | Linear-with-interactions ceiling | 15 min |
| `lr_diag_a4_per_segment.py` | Per-segment specialization potential | 5 min |

Three durable findings to carry across comps:

1. **Pool eff_rank < nominal pool size**: SVD entropy diagnostic in
   <2 min CPU answers "is your pool redundant?" Forward-selection
   confirms empirically.
2. **The dominant interaction hub is found in <5 min** via residual
   binning. Apply that knowledge to GBDT FE on Day-1.
3. **Representation-only diversity has bounded meta-utility on a
   saturated pool**. If GBDT residuals at top-pair cells are < 1%,
   no LR/FM/NN-on-same-features will move the meta.

## What this means for s6e5 going forward (per Arc A handover)

The Arc-A/B/C trifecta confirms: **lift on s6e5 must come from new
information OR new meta-architecture.** The HANDOVER's
`meta_arch_redesign` axis (non-Gaussian shrinkage, Yao/Vehtari BMA,
alternative segmentation crosses) is the only in-pool path remaining;
external data was ruled out by PI. **No further base-add probes
inside the LR family will lift the meta.**

Friction-tag candidates:
- `representation-only-diversity-meta-null-on-saturated-info-space`
- `low-rho-necessary-but-not-sufficient-for-meta-utility-6-confirmations`
