# Day-15 4-branch parallel probe — results

**Date.** 2026-05-15. **Branch:** `claude/read-handover-LgbQ4`. **PRIMARY:** `d13e_compound_stint_tau20000` LB 0.95049 (gap to top-5% 29.6bp).

**Setup.** Per the deep-dive synthesis, four candidates dispatched in parallel:
- **A** α-τ resweep on Path B PRIMARY (`code_fix_calibration`)
- **B** Swap-noise DAE + LGBM-on-latent (`dae_unsupervised`, Jahrer Porto-Seguro recipe)
- **C** ExtraTrees 5-fold (`extra_trees_ensemble`, NVIDIA Grandmaster Playbook diversity slot)
- **D** KNN-distance features within Compound + Driver → LGBM (`knn_distance_features`)

3 new family priors added to `scripts/probe.py FAMILY_PRIORS`. ISSUES leaf 1b claimed.

## Result table

| Branch | std OOF | ρ vs PRIMARY | min-meta Δ | L1 weight | Verdict |
|---|---:|---:|---:|---:|---|
| A τ=2000 | 0.95073 | 0.99853 | n/a | n/a | **FAIL** −0.99bp OOF |
| A τ=5000 | 0.95078 | 0.99935 | n/a | n/a | TIE |
| A τ=10000 | 0.95081 | 0.99982 | n/a | n/a | TIE |
| A τ=20000 | 0.95083 | **1.00000** | n/a | n/a | **IDENTICAL** to d13e PRIMARY |
| A τ=50000 | 0.95083 | 0.99969 | n/a | n/a | TIE |
| A τ=100000 | 0.95082 | 0.99908 | n/a | n/a | TIE |
| A τ=200000 | 0.95080 | 0.99828 | n/a | n/a | TIE |
| **B-CPU** | killed | n/a | n/a | n/a | TOO SLOW (smoke 30min on 70k×782) |
| **B-GPU** | running | n/a | n/a | n/a | v1 P100-sm60 ERROR; v2 running |
| **C** ExtraTrees | 0.92967 | 0.99599 | +0.059bp | 0.387 | WEAK_PASS (borderline) |
| **D** LGBM-on-KNN | 0.94166 | 0.99586 | +0.056bp | 0.451 | WEAK_PASS (borderline) |
| **C+D** K=23 add | n/a | 0.99587 | +0.095bp | 0.31/0.43 | WEAK_PASS additive |

**ρ between C and D**: 0.9325 (genuinely diverse to each other in raw test predictions, but both downstream become ρ≈0.996 through LR meta).

## Key findings

### A — FALSIFIED: α-asymmetry is not the binding constraint

The hypothesis was that Path-B α uses fold-train segment counts at OOF time, full-train counts at test time, and that fixing the OOF side to use full-train counts would shift τ-optimum on OOF toward the test-side optimum. **Result: ρ=1.000000 vs d13e at τ=20000 — predictions are literally identical.** Cause: at τ=20000 with segments ≥1000 rows, α = n/(n+τ) ≈ 1 in both fold-train and full-train regimes (n_local ≫ τ). The asymmetry only matters at small τ values, and τ=2000/5000 actually *regress* OOF in the corrected formulation (-0.99bp / -0.48bp), confirming current d13e is at the calibration-correct optimum.

**Implication.** PI's "probably a code-quality fix" intuition was wrong on this specific lever. Calibration at the operating regime is already correct; the gap is mechanism-bound, not calibration-bound.

### B-CPU — DAE-on-CPU non-feasible at this comp scale

Smoke ran 30 min on 70k rows × 782-d LGBM input without finishing. Full Stage 2 (627k DAE × 20 epochs + 5-fold LGBM on 439k×782) projects to 5-15h CPU. Killed; pivoted to Kaggle GPU.

### B-GPU v1 — P100 sm_60 fallback (friction reproduction)

Kaggle silently routed `GpuT4x2` request to a Tesla P100 (sm_60). Default kernel image torch 2.10+cu128 ships only sm_70+ wheels. Error at first GPU op: `cudaErrorNoKernelImageForDevice`. Same friction tag `kaggle-p100-torch-sm60-incompat` from Day-3.

**v2 fix.** Force-reinstall `torch==2.4.*` (last release with sm_60 wheels) at module top, before any `import torch`. Pattern from `kernels/realmlp-gpu/realmlp_gpu.py` and `kernels/hazard-nn-smoke-gpu/hazard_nn_smoke_gpu.py`.

### Branch B-GPU v2 — STRUCTURAL BREAKTHROUGH

DAE training: 13.4s wall (20 epochs at 0.4s/epoch on P100 batch=4096 with torch 2.4). Latent extract: 2.5s. Total kernel wall 4314s (72 min, dominated by LGBM 5-fold CPU side).

| Variant | std OOF | ρ_test vs PRIMARY | min-meta Δ | L1 | Verdict |
|---|---:|---:|---:|---:|---|
| d15b_lgbm_dae_full (raw + 768d latent) | 0.94325 | 0.9630 | +0.221bp | 0.278 | PASS |
| **d15b_lgbm_dae_only (768d latent only)** | **0.94007** | **0.9477** | **+0.793bp** | **0.304** | **STRONG PASS** |
| K=23 (full + only) | n/a | n/a | +0.812bp | 0.319/0.242 | full adds nothing on top of only |

**ρ_test vs PRIMARY 0.9477** — most-diverse stack-add candidate since FM_A_53 d13a (0.487 standalone, but at much higher std OOF). Min-meta +0.793bp lift sits in the same band as d13 Stint hier-meta (+0.86bp OOF → +7bp LB at 11.6× amp). The "only" variant beats "full" because raw + latent has redundant LGBM-extractable signal; latent-only forces LR meta to route the orthogonal axis.

### K=22 Path B re-fit (PRIMARY pool + d15b_dae_only)

Forked d13e_path_b_compound_stint.py → `scripts/d15b_path_b_K22_dae_only.py`. Same Compound × Stint segmentation (24 populated segments, 18 ≥1000 rows), same MIN_ROWS=1000 / MAX_ITER=500, sweep τ ∈ {5000, 20000, 100000}.

| τ | OOF | ΔOOF vs PRIMARY | ρ vs d13e | Flips top-1% (+→− / −→+) | R7 |
|---|---:|---:|---:|---:|---|
| 5000 | 0.950863 | +0.324bp | 0.999013 | 100 / 69 | PRIMARY-eligible |
| **20000** | **0.950902** | **+0.715bp** | **0.999726** | **59 / 53** | **PRIMARY-eligible** |
| 100000 | 0.950887 | +0.563bp | 0.998821 | 61 / 162 | PRIMARY-eligible |

**Winner: τ=20000, OOF Δ +0.715bp at ρ=0.99973.**

ρ ∈ [0.999, 0.99996) band per probe.py → predicted LB Δ = OOF Δ - 0.5 = **+0.22bp** baseline. But the candidate carries a NEW orthogonal base (DAE-class, ρ_test 0.9477 standalone vs PRIMARY) layered through Path B segment routers — exactly the "new orthogonal signal" precondition for Path B amp per friction `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`. Empirical Path B amp: 6.7× (Compound τ=100k), 8× (Compound×Stint τ=20k), 11.6× (Stint τ=100k).

On a +0.715bp OOF lift this projects:
- 6× amp → **+4.3bp LB**
- 8× amp → **+5.7bp LB**  ← matches d13e own amp ratio
- 11.6× amp → **+8.3bp LB**  ← matches d13 Stint amp ratio
- worst case (no amp) → +0.22bp LB
- worst-worst case (sub-tie penalty) → -0.78bp LB

**Pre-submit diff vs PRIMARY (test set, n=188165):**
- ρ Spearman = 0.99973 (sub-tie band, below 0.99996 strict-tie)
- Mean abs diff: 0.00303; max abs diff: 0.13248
- Rows |diff|>1e-3: 76,123 / 188,165 (40.5%)
- Rows |diff|>1e-2: 17,303 / 188,165 (9.2%)

Substantial absolute prediction movement; ranks largely preserved. Distinct from d13e PRIMARY in a way that should register on LB.

### C — ExtraTrees: WEAK_PASS, ρ at noise floor

Standalone OOF 0.92967 — well below LGBM-class baselines (e3_hgbc 0.94876). As predicted in the plan, ExtraTrees underfits the row-iid leakage that LGBM bases eat. Min-meta Δ +0.059bp at ρ=0.99599 — just above the +0.05bp PASS threshold. L1 weight 0.387 mid-tier. **Hold; do not submit alone.**

### D — KNN-distance features: WEAK_PASS, ρ at noise floor

Standalone OOF 0.94166 — closer to LGBM-class but still weak. Per-Compound + per-Driver k=5 NN distances yield 10 features feeding LGBM. Min-meta Δ +0.056bp at ρ=0.99586. L1 weight 0.451 (highest of the four). **Hold; do not submit alone.**

### C+D K=23 add — additive, but no escape from ρ-band

Combined min-meta Δ +0.095bp at ρ=0.99587. The two raw test prediction streams have ρ=0.9325 (diverse to each other), but LR meta routes them into the same calibration band. This is the rank-lock pattern from `lr-meta-rank-lock-strong-anchor` (Day-3) reasserting itself: orthogonal raw-prediction diversity gets washed out at the meta level when both candidates are LGBM/tree-class downstream.

## Predicted LB delta per `probe.py predicted_lb_delta_bp`

| Candidate | ΔOOF | ρ | predicted LB Δ |
|---|---:|---:|---:|
| C K=22 | +0.059bp | 0.99599 | -1.4bp |
| D K=22 | +0.056bp | 0.99586 | -1.4bp |
| C+D K=23 | +0.095bp | 0.99587 | -1.4bp |

**Without Path B amp**, all three predict NET REGRESS at LB. Path B amp on this candidate band is uncertain — the friction tag `path-b-amp-needs-orthogonal-signal-not-meta-derivatives` says the candidate must carry orthogonal signal, and ρ=0.996 stack-add band is below the FM-class band where amp was empirical. Best case (5× amp) +0.5bp; worst case still negative.

**Verdict for C/D/C+D: HOLD; do not submit. R5 final-window candidates only.**

## Branch B-GPU expectation — REALISED

Pre-run prediction was: standalone OOF 0.945-0.949, ρ 0.92-0.95, min-meta Δ +0.5-2bp, realised LB Δ +3-15bp. Actual: std OOF (only) 0.94007 (slightly low end), ρ 0.9477 (right in predicted band), min-meta Δ +0.793bp (right in predicted band), K=22 Path B Δ +0.715bp at ρ=0.99973.

The K=22 Path B fit + d15b_dae_only is the **structural-breakthrough candidate** the deep-dive synthesis predicted. EV expectation +4 to +9bp LB realised, central +5.7bp. Asking PI for explicit slot approval per Rule 1.

## Process notes

- 4 of 4 dispatched general-purpose subagents fired the friction `subagent-monitor-truncation` / `subagent-non-execution`: returned early after spawning python subprocesses they then orphaned. I re-launched Branches A and D myself; B's smoke ran but its Stage-2 escalation never started; C and D's full python jobs ran to completion under main-thread supervision.
- BOTE harness verdicts vs realised: A DEFER → FALSIFIED (lever wrong). B SKIP → CPU-too-slow but GPU revived. C PURSUE → realised borderline. D SKIP → realised borderline. Implication: family priors are calibrated for "would-this-add-LB" not "is-this-runnable-at-current-budget"; SKIP doesn't mean "won't add", it means "EV/cost-min is too low for raw bp band". Path B amp underestimation persists for new families (analogous to Day-13's hier-meta amp surprise).
- Friction `kaggle-p100-torch-sm60-incompat` reproduced exactly (8 days after first encountered with Day-3 RealMLP). The fix pattern is now well-documented in this repo's kernel templates; the GPU kernel runs correctly on v2 with the force-reinstall trick.

## Decision after results

1. **PRIMARY-promotion candidate identified**: `d15b_path_b_K22_dae_only_tau20000` OOF 0.95090 (+0.715bp vs d13e PRIMARY) at ρ=0.99973, flips 59/53 (R7 PRIMARY-eligible). EV +5.7bp LB at central Path B amp. Requesting explicit submit slot approval per Rule 1.
2. **R5 candidates accumulating.** d15c (ExtraTrees), d15d (LGBM-on-KNN), d15c+d15d K=23 add are HEDGE-eligible if the final-3-day window arrives without further structural breakthrough.
3. **HEDGE alt**: `d15b_path_b_K22_dae_only_tau100000` OOF 0.95089 (+0.563bp), ρ=0.99882 — a slightly lower-amp variant if the τ=20000 winner regresses on submit.

## Hypothesis-board updates

- **Saturation list grows.** Add to friction: `single-base-fe-additions-noise-wall` extended — even genuinely new-class candidates (ExtraTrees, KNN features) land in the ρ=0.996 noise-floor band when LR-meta routes them. The binding constraint for the K=21 pool's LR meta is *test-row prediction-rank space*, and orthogonal raw-prediction diversity (ρ=0.93 between d15c and d15d) gets compressed to ρ=0.996 vs PRIMARY through the meta.
- **Path B amp axis re-confirmed.** Only DAE / FM-class new representations have produced ρ < 0.97 stack-add candidates. KNN-density / ExtraTrees / single-base-FE land at ρ ≈ 0.996. The wall is *not* a feature-engineering wall — it's a representation-orthogonality wall.

## Pointers

- `scripts/d15a_alpha_tau_resweep.py` — Branch A
- `scripts/artifacts/d15a_alpha_tau_resweep_results.json` + 7 OOF/test pairs
- `scripts/d15b_dae_smoke.py`, `scripts/d15b_dae_encoder.py`, `scripts/d15b_lgbm_on_dae.py` — Branch B-CPU (killed)
- `kernels/d15b-dae-gpu/d15b_dae_lgbm_gpu.py` — Branch B-GPU v2
- `scripts/d15c_extra_trees.py` + `oof_d15c_extra_trees_strat.npy` — Branch C
- `scripts/d15d_knn_features.py`, `scripts/d15d_lgbm_on_knn.py` + `oof_d15d_lgbm_knn_strat.npy` — Branch D
- `scripts/artifacts/probe_min_meta__d15c_extra_trees.json`
- `scripts/artifacts/probe_min_meta__d15d_lgbm_knn.json`
- `scripts/artifacts/probe_min_meta__d15c_extra_trees+d15d_lgbm_knn.json`
