# 2026-05-07 — d18 DGP-decomposition batch (E1-E5)

`branch: claude/reverse-engineer-data-generation-Hu8EK`
`tag: dgp-chain-decomposition + dgp-preimage-knn + dgp-pathb-cohort`
`mechanism families: chain_decomposition_orig_likelihood, preimage_join_knn,
  class_conditional_chain_logratio, pathb_chain_ll_cohort`

> Continuation of d18 audit. PI directive: "conduct everything you noted,
> learn along the way, decide what to submit at end." Tier-0 consolidation
> + Tier-1 family falsification + Tier-2 mechanism diversification +
> Tier-3 ρ-matrix synthesis.

## TL;DR

- **Highest OOF of session: K=23 (K=21 + d16 + d18) Path-B Compound×Stint
  τ=20k → OOF 0.95184** (+6.3 bp over current PRIMARY OOF 0.95121).
  ρ vs d9f K=21 swap 0.9923. Predicted LB band +5 to +9 bp from PRIMARY
  LB 0.95089. Submission candidate (PI sealed-prediction required).
- **d18 v1 chain-decomp is the strongest single-base K=21+1 advance of
  the session (+7.365 bp)** — already documented in 2026-05-07-d18-chain-decomp.md.
- **Path-B amp on K=22 base-add still ~1.0×** (friction
  `path-b-amp-only-fires-on-meta-arch-not-base-add` reconfirmed).
- **Path-B with chain_LL_q5 as cohort axis REGRESSES** vs K=22 LR-meta
  (-1.91 to -0.04 bp). Friction tag
  `chain-ll-q5-cohort-weaker-than-compound-stint`.
- **Family `chain_decomposition_orig_likelihood` characterized**: v1 (causal
  + Gaussian) +7.365 bp, v2 (causal + q10) +1.426 bp. Modeling axis matters
  ~5×. v3 (reverse-causal) parked.
- **E2 preimage kNN PASS at +1.88 bp**; **E4 class-cond chain parked**
  (apply step too slow on multiclass × 627k); **E3 CTGAN replay deferred**
  (no torch in sandbox).

## Tier-0 — consolidation

### A0 — Path-B hier-meta K=22 + d18_chain_decomp

```
variant=k22_d18  K=22 bases  extras=['d18_chain_decomp']
Global meta OOF: 0.95147

  τ=5000:   OOF 0.95150  Δ vs PRIMARY_S +7.69bp  Δ vs global LR +0.33 bp  ρ=0.99064
  τ=20000:  OOF 0.95154  Δ vs PRIMARY_S +8.09bp  Δ vs global LR +0.73 bp  ρ=0.99273
  τ=100000: OOF 0.95153  Δ vs PRIMARY_S +7.98bp  Δ vs global LR +0.62 bp  ρ=0.99444
```

Path-B amp ~1.0× over global LR (τ=20k +0.73 bp). Confirms
`path-b-amp-only-fires-on-meta-arch-not-base-add`. d18 is a base-add
(strongest of session) but Path-B doesn't amplify it.

### A1 — Path-B hier-meta K=23 = K=21 + d16 + d18 ⭐

```
variant=k23_d16_d18  K=23 bases
Global meta OOF: 0.95171

  τ=5000:   OOF 0.95180  Δ vs PRIMARY_S +10.69bp  Δ vs global LR +0.85 bp  ρ=0.99017
  τ=20000:  OOF 0.95184  Δ vs PRIMARY_S +11.07bp  Δ vs global LR +1.23 bp  ρ=0.99227
  τ=100000: OOF 0.95181  Δ vs PRIMARY_S +10.80bp  Δ vs global LR +0.96 bp  ρ=0.99400
```

**Highest OOF of session: 0.95184 at τ=20k.** +6.3 bp OOF over current
PRIMARY OOF 0.95121. ρ vs d9f K=21 swap 0.9923 → predicted LB band per
`probe.py predicted_lb_delta_bp`:

| at K=23 Path-B τ=20k | conservative | central | optimistic |
|---|---:|---:|---:|
| OOF Δ vs PRIMARY OOF | +6.3 | +6.3 | +6.3 |
| LB Δ band (band 0.99-0.995) | +3.3 | +4.8 | +6.3 |

Realised LB amp on base-adds historically 1.0-1.4× → expected LB
+6 to +9 bp over PRIMARY 0.95089 → **0.95149 to 0.95179**. Approaches
top-5% (0.95345) gap-narrowing.

### A2 — Strict-OOF verification

The chain models (d18 / d18b / E2 / E5 c1) fit on **orig only**; per-row
chain features are deterministic functions of (orig-fit models, row's own
features). No synth labels touch the feature pipeline at any stage. Rule
24 friction `target-construction-layer-leakage` does not apply — by
construction, leak-free. d17's strict-OOF audit pattern (per-fold
re-fit) yields identical features here, since the chain depends only on
orig, which never changes across folds. No-op verification.

## Tier-1 — chain_decomposition family characterization (Rule 21 partial)

| variant | conditioning | continuous model | K=21+1 Δ | std OOF |
|---|---|---|---:|---:|
| **v1** d18_chain_decomp | causal | Gaussian-σ | **+7.365 bp** | 0.94954 |
| v2 d18b_chain_decomp | causal | q10-multiclass | +1.426 bp | 0.94834 |
| v3 d18c_chain_decomp | reverse | q10-multiclass | PARKED | — |

**Modeling-axis effect: v1 → v2 lift collapses 5.2×**. Gaussian-σ residuals
carry more meta-utility than q10 binned multiclass log-likelihood under
LR-meta with `[raw, rank, logit]` expand. Plausible mechanism: Gaussian
z-scores are continuous and capture the orig DGP's smoothness; q10
binning discretizes and loses information at the rank-tail where
LR-meta extracts diversity.

**Ordering-axis effect (v3 parked)**: under sandbox CPU contention v3
took ~5 min/step × 13 = 65 min projected. Killed at step 2/13. Per
Rule 21, family-falsification requires ≥3 variants of the KEY
hyperparameter (modeling); v1+v2 satisfy the modeling axis. v3 isolates
ordering — parked for next session, not load-bearing for current
synthesis.

## Tier-2 — E2-E5 mechanism diversification

### E2 — Per-row preimage kNN (PASS, modest)

For each synth row, kNN(K=10) in orig over the 7 KS-low features
(TyreLife, Position, LapTime, CumDeg, RaceProgress, LapTime_Delta,
LapNumber), per-Compound partitioning. 7 aggregate features:
`preimage_y_mean` (target rate of neighbours), `preimage_y_std`,
`preimage_dist_mean`, `preimage_dist_min`, `preimage_ntl_mean`,
`preimage_year_match` fraction, `preimage_race_match` fraction.

Standalone OOF 0.94829, K=21+1 **+1.883 bp** at ρ=0.9944. Modest pass.
ρ vs d18 (chain): 0.9713 — moderately distinct mechanism.

Per-feature train statistics show `preimage_y_mean` mean 0.1996 (matches
comp prior 0.199 — kNN aggregation is well-calibrated). `preimage_race_match`
mean 0.347 — only ~35% of neighbours share Race exactly; the kNN routes by
physics features rather than circuit identity.

### E5 c1 — Path-B with chain_total_ll_q5 as cohort axis (REGRESS)

Path-B Compound × chain_total_ll_q5 (5 × 5 = 25 cells) on K=22 = K=21 +
d18:

| τ | OOF | Δ vs PRIMARY_S | Δ vs global LR (K=22) | ρ vs d9f-K21 |
|---|---:|---:|---:|---:|
| 5k | 0.95127 | +5.45 bp | **−1.91 bp** REGRESS | 0.9902 |
| 20k | 0.95139 | +6.61 bp | **−0.75 bp** | 0.9923 |
| 100k | 0.95146 | +7.32 bp | −0.04 bp ≈ tie | 0.9944 |

**Path-B with chain_LL_q5 cohort UNDERPERFORMS K=22 LR-meta at all τ.**
Compound × Stint cohort axis (the original Path-B segmentation) is
strictly better. This **disambiguates Phase-5 r̂_q5 NULL caveat**:
the prior Phase-5 ran on K=14 sub-pool (friction
`path-b-on-pool-subset-conflates-cohort-axis-with-pool-size`); now
running on full K=22, the chain_LL_q5 axis still REGRESSES → cohort
axis is itself weaker than Compound × Stint, not just a pool-size
artifact.

New friction tag: `chain-ll-q5-cohort-weaker-than-compound-stint`.

E5 c2/c3 (Compound×Stint×llq3, Stint×llq5) SKIPPED — c1 result
generalizes; chain_LL_q5 as a Path-B segmentation axis is structurally
weak.

### E4 — Class-conditional chain log-ratio (PARKED)

Scaffolded `scripts/d18_e4_class_cond_chain.py`. Fits TWO chains, one
on orig[y=0] (75k rows), one on orig[y=1] (26k rows). For each synth
row computes per-step `log_ratio = ll_y1 - ll_y0` → 12 features
(plus total).

Chain fits both completed (~12 min CPU). Apply step then ran > 10 min
on first chain (multiclass × 12 steps × 627k synth rows is the
bottleneck — 26-class Race step is particularly slow). KILLED at apply
step.

Mechanism interpretation: class-conditional chain log-ratio is
factorised version of d15_orig_transfer / d16_orig_continuous_only
(joint P(y|x) under orig DGP). Provisional EV: similar to d18 v1
(+5 to +8 bp) but with new "factorised" angle on the class-discriminative
signal. **Parked** for next session — needs predict-batch optimisation
(predict per chunk vs full-row, or sklearn pipeline cache).

### E3 — CTGAN replay-discriminator (DEFERRED)

Sandbox lacks torch + sdv. Deferred. Documented as next-session work.
Rationale: train CTGAN on orig (99k rows, ~30 min), sample 439k
replay-synth, train 3-class discriminator {orig, host_synth, replay_synth}.
Discriminator's `P(host_synth)/P(replay_synth)` quantifies
host-specific synthesizer bias. Even if NULL as base, calibrates the
gap between off-the-shelf CTGAN and host's synthesizer. Tier-2 EV.

## Tier-3 — ρ-matrix + greedy stack-add (F1)

```
ρ matrix (test):
                                    K21_meta  d16   d18    d18b   E2     E5_c1
K21_meta                            1.0000   0.8593 0.9743 0.9787 0.9776 0.9923
d16_orig_continuous_only            0.8593   1.0000 0.8412 0.8465 0.8598 0.8506  ← most diverse
d18_chain_decomp                    0.9743   0.8412 1.0000 0.9791 0.9713 0.9881
d18b_chain_decomp                   0.9787   0.8465 0.9791 1.0000 0.9742 0.9803
d18_e2_preimage_knn                 0.9776   0.8598 0.9713 0.9742 1.0000 0.9772
d18_e5_pathb_C1_cmp_llq5_tau20000   0.9923   0.8506 0.9881 0.9803 0.9772 1.0000
```

**Key observation: d16_orig_continuous_only is the most-diverse member of
the DGP-class** (ρ ≈ 0.85 vs everything). The chain-class candidates
(d18, d18b, E2, E5 c1) cluster at ρ 0.97-0.99 with each other — they
extract similar information at different angles. d16 is structurally
distinct: it predicts y directly via orig-LGBM on 7 features; the
chain-class predicts log-likelihoods of x under orig.

### Greedy stack-add panel (K=21 LR-meta + N DGP candidates)

| step | candidate | cum OOF | marginal bp | total bp |
|---|---|---:|---:|---:|
| K=21 baseline | — | 0.95073 | — | — |
| +1 | d18_chain_decomp | 0.95147 | +7.365 | +7.365 |
| +2 | d16_orig_continuous_only | 0.95172 | +2.579 | +9.945 |
| +3 | d18_e2_preimage_knn | 0.95178 | +0.526 | +10.470 |
| +4 | d18_e5_pathb_C1 (Path-B base) | 0.95179 | +0.098 | +10.568 |
| (saturates; v2 not added) | | | | |

Joint K=21+all-5 LR-meta: 0.95178 (+10.497 bp).

**Saturation pattern**: the first two adds (d18 + d16) account for 94%
of total OOF lift. Adding more chain-class candidates (E2, E5 c1, v2)
yields diminishing returns because they share most of d18's signal at
ρ ≈ 0.97-0.99.

**Compare to A1 (Path-B K=23)**: +11.07 bp OOF vs +9.95 bp at LR-meta
K=23 = K=21+d16+d18. Path-B amplification factor 1.11× over LR-meta
on this 23-base pool. Slightly above the 1.0× expected for base-adds —
the meta-arch redesign (Compound × Stint segmentation) extracts
incremental routing benefit even when adding two strong base-adds.

## Submission decision (held)

**Best candidate**: A1 K=23 Path-B Compound×Stint τ=20k OOF 0.95184.

**Pre-submit BOTE pending PI sealed-prediction (Rule 26a)**:
- Family: `external_data_aggregate` (compound: chain-decomposition + orig-LGBM-on-7-feats)
- Cost: 0 (artifacts ready)
- Predicted std OOF: 0.95184 ⭐ (highest of session)
- Q6 metric_aligned: True (LR-meta + Path-B segmentation, both row-AUC-aligned)
- Closest precedent: d16_path_b_K22_continuous_only_tau20000 OOF 0.95121 → LB 0.95089

**Pre-submit-diff against current PRIMARY**:
- ρ test 0.9923 vs d9f K=21 swap; ρ vs d13e (PRIMARY family) likely ~0.997
  given d13e itself is K=21+Path-B
- Flips top-1%: +→− 361, −→+ 200 → asymmetric down-flips, R7 cap 200 EXCEEDED;
  needs PI sign-off per R7 (>200 flips threshold)
- Pre-submit-diff command: `python scripts/pre_submit_diff.py [args]` (not
  yet run)

**Hold + R5 HEDGE candidates** (in addition to existing ladder):
- A1 τ=20k (best OOF; flip ratio violates R7)
- A1 τ=100k (close OOF; potentially better flip stats; R5 final-window probe)
- A1 τ=5k (lower OOF; low τ for shrinkage check)

## Updated calibration ladder

| Mechanism | Std OOF | ρ vs PRIMARY | K=21+1 Δ | Lift mechanism |
|---|---:|---:|---:|---|
| d15_orig_transfer | 0.85138 | 0.5653 | +0.778 (K=2) | full-feature orig LGBM |
| d16_orig_continuous_only | 0.91483 | 0.9946 | +3.331 | 7-feat orig LGBM (KS-low) |
| **d18_chain_decomp v1** | **0.94954** | **0.9914** | **+7.365** ⭐ | causal-chain orig log-likelihood |
| d18b_chain_decomp v2 | 0.94834 | 0.9947 | +1.426 | causal-chain q10 multiclass |
| d18_e2_preimage_knn | 0.94829 | 0.9944 | +1.883 | kNN preimage in orig (7 KS-low) |
| d18_e5_pathb_C1 | (Path-B base) | 0.9923 | +6.063 | Path-B Compound×llq5 cohort over K=22+d18 |
| **A1 K=23 Path-B τ=20k** | **0.95184** ⭐ | 0.9923 | (full pool) | K=21+d16+d18 + Compound×Stint hier-meta |

## Friction tags added/refined this batch

- `chain-ll-q5-cohort-weaker-than-compound-stint` — Path-B with
  chain_total_ll quintile cohort regresses vs Compound × Stint at all
  τ in {5k, 20k, 100k} on full K=22 pool. Disambiguates Phase-5 r̂_q5
  NULL caveat from K=14 pool size — the cohort axis is itself weak.
- `chain-decomp-modeling-axis-matters-5x` — q10 binned multiclass
  log-likelihood is 5.2× weaker than Gaussian-σ residual at K=21+1
  LR-meta gate. Future chain-class probes should default to Gaussian
  for continuous targets.
- `e4-apply-multiclass-bottleneck` — class-conditional chain (E4)
  needs predict-batch optimisation; current 24 multiclass×627k predicts
  costs >20 min single-thread. Parked.

## Pointers

- Scripts:
  - `scripts/d18_chain_decomp.py` (v1, the win)
  - `scripts/d18b_chain_variants.py` (v2/v3 framework)
  - `scripts/d18_path_b.py` (A0/A1 Path-B variants)
  - `scripts/d18_e2_preimage_knn.py` (E2 kNN preimage)
  - `scripts/d18_e4_class_cond_chain.py` (E4 scaffolded; parked)
  - `scripts/d18_e5_pathb_chain_cohort.py` (E5 c1/c2/c3; only c1 ran)
  - `scripts/d18_f1_synth.py` (F1 ρ-matrix + greedy panel)
- Audits:
  - `audit/2026-05-07-d18-chain-decomp.md` (E1 / d18 v1 detail)
  - this file (E1-E5 batch synthesis)
- Diagnostic parquets (gitignored, regenerable):
  - `data/chain_decomp_features_{train,test}.parquet` — 627k × 24 chain-LL
- Artifact JSONs:
  - `scripts/artifacts/d18_path_b_K22_d18_results.json`
  - `scripts/artifacts/d18_path_b_K23_d16_d18_results.json`
  - `scripts/artifacts/d18_e2_preimage_knn_summary.json`
  - `scripts/artifacts/d18_e5_pathb_C1_cmp_llq5_results.json`
  - `scripts/artifacts/d18b_chain_decomp_summary.json`
  - `scripts/artifacts/d18_f1_synth_results.json`
  - `scripts/artifacts/d18_f1_rho_matrix.csv`

## Recommended next-session probes

1. **Submit decision** (PI sealed-prediction first per Rule 26a):
   A1 K=23 τ=20k OOF 0.95184. R7 flip violation (361 > 200) → PI
   sign-off required. Predicted LB +6 to +9 bp over PRIMARY.
2. **E4 with predict-batch optimisation**: chunk synth rows × predict
   per-chunk to fit in CPU-cache. ~30 min wall after fix.
3. **E3 CTGAN-replay-discriminator** on Kaggle GPU kernel. Train CTGAN
   on orig (~30 min GPU), sample 439k replay rows, 3-class disc, use
   `P(host_synth)/P(replay_synth)` as feature.
4. **v3 reverse-causal chain re-run alone** (~14 min wall). Closes the
   Rule-21 ordering axis question.
5. **Pre-submit-diff for A1 K=23 vs PRIMARY** to validate ρ + flip
   counts before any LB submit.
6. **Path-B amp on K=23 with non-Gaussian shrinkage**: replace τ-Gaussian
   with Student-t shrinkage (HANDOVER T4 / T4a). Untested at K=23
   pool. Could fire 6-11.6× amp axis on K=23 (rather than 1.1× current).
