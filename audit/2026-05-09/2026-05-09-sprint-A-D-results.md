# 2026-05-09 — Sprint A through E results: stint_imputed + orig-cell + joint + d18_g unified ablation

`branch: claude/analyze-synthetic-data-generation-BtmFl`
`tag: dgp-decoded-features-K4-gate`
`session window: 2026-05-09 ~12:00 → ~13:00 UTC (autonomous loop)`

> Operationalised the second-wave research synthesis. Built qAA, qAB,
> qAC, ran unified gate against d18_g existing artifacts. All single-
> add candidates fall in (-0.005, +0.143) bp at K=4+1 LR-meta. Best
> combo K=7 = K=4 + qAA + qAC + d18g at +0.374 bp — still WEAK.
> Confirms `rank-lock-at-conditional-target-correlation-not-just-logit-direction`
> at the LR-meta layer for decoded features.

## Probes built

### qAA — stint_imputed-anchored LightGBM (M7 + M8 + M15 + M17)

11 new features anchored on `stint_imputed = LapNumber - TyreLife + 1`
(cardinality 105 vs synth `Stint`'s 8):

  CumulativeTimeStint, prev_lap_delta_stint, prev_lap_delta_drv,
  stint_lap_idx, stint_size, stint_lap_frac, compound_changes,
  position_at_stint_start, position_change_in_stint, prev_compound,
  stint_imputed itself.

LightGBM on raw 14 + 11 stint-anchored = 25 features. 5-fold StratifiedKF.

**Result:**
- Standalone OOF AUC: **0.94495** (fold std 0.00065 — very stable)
- ρ_oof vs PRIMARY: 0.95460
- ρ_test vs PRIMARY: 0.96471 (most-diverse positively-gating in K=4 era)
- K=4+1 plain LR-meta lift: **+0.143 bp** WEAK

Verified prev_lap_delta_stint correlation with our existing
LapTime_Delta is 0.16 — confirming `LapTime_Delta` is NOT the prev-lap
delta the Frontiers F1 paper uses. Genuinely new feature.

### qAB — orig-derived hierarchical TE + kNN + Gaussian density (M4 + M5)

15 features, all orig-derived:
- 6 hierarchical empirical-Bayes-shrunk PitNextLap rates (L1..L6, k_smooth=30)
- 6 hierarchical orig-cell counts (sparsity diagnostic)
- 1 orig-kNN PitNextLap vote (K=20 within (Y, C, PS) cell, hierarchical fallback)
- 1 orig-kNN median distance
- 1 per-cell Gaussian log-density on (LapTime, Δ, CumDeg, RP)

**Result:**
- Standalone OOF AUC: **0.92059** (fast: ~7s/fold)
- ρ_test vs PRIMARY: **0.89932** (most-diverse positively-gating ever observed)
- K=4+1 lift: **+0.017 bp** WEAK
- K=4+2 (qAA+qAB): **+0.213 bp** WEAK
- ρ(qAA, qAB) on OOF: 0.87281 — substantial structural orthogonality

### qAC — joint base: yekenot 14 raw + 11 stint + 15 orig = 40 features

LightGBM on the joint feature set. V4-pattern attempt: ingest decoded
features at the BASE level via tree splits.

**Result:**
- Standalone OOF AUC: **0.94605** (+1.1 bp over qAA)
- ρ_test vs PRIMARY: 0.96115
- K=4+1 lift: **−0.005 bp** DEAD null

Joint-base ingestion does not escape rank-lock at OOF. Surprising.
The V4 LB transfer +0.8 bp at ρ=0.99989 may not generalise to all
base+feature combos.

### qAE — unified ablation gate at K=4+1 .. K=4+4

Cross-product of {qAA, qAB, qAC, d18g} ∪ K=4. d18_g (Day-18 mode-id)
loaded from existing artifacts; standalone OOF 0.94877, ρ_test 0.97962.

| Combo | K | OOF | Δ bp | Verdict |
|---|---:|---:|---:|---|
| K=4 alone | 4 | 0.95399 | 0 | anchor |
| qAA | 5 | 0.95401 | +0.143 | WEAK |
| qAB | 5 | 0.95400 | +0.017 | WEAK |
| qAC | 5 | 0.95399 | -0.005 | NULL |
| d18g | 5 | 0.95400 | +0.028 | WEAK |
| qAA+qAB | 6 | 0.95402 | +0.213 | WEAK |
| qAA+qAC | 6 | 0.95402 | +0.275 | WEAK |
| qAA+d18g | 6 | 0.95402 | +0.258 | WEAK |
| qAB+qAC | 6 | 0.95400 | +0.021 | WEAK |
| qAB+d18g | 6 | 0.95400 | +0.044 | WEAK |
| qAC+d18g | 6 | 0.95400 | +0.051 | WEAK |
| qAA+qAB+qAC | 7 | 0.95402 | +0.290 | WEAK |
| qAA+qAB+d18g | 7 | 0.95402 | +0.272 | WEAK |
| **qAA+qAC+d18g** | 7 | 0.95403 | **+0.374** | best WEAK |
| qAB+qAC+d18g | 7 | 0.95400 | +0.058 | WEAK |
| qAA+qAB+qAC+d18g | 8 | 0.95403 | +0.357 | WEAK |

**Pattern:** qAA dominates. qAB carries minimal additional meta signal.
d18g + qAA combo nearly matches the K=8 quad. **No combination passes
the +0.5 bp strict gate.**

## What this confirms / refines

### Confirmed: rank-lock at conditional-target-correlation level holds for decoded features

Every decoded feature family (sequence-on-stint_imputed, orig-derived
TE+kNN+density, BGMM mode-id) absorbed at the LR-meta within ≤+0.4 bp.
ρ_test as low as 0.899 (qAB) does NOT translate to meta lift. The
rank-lock is conditional-target-correlation-level — qAB has new
partial correlation with y given K=4, but it's small enough that LR
absorbs it.

### Confirmed: V4 lesson is conditional, not universal

V4 lifted +0.8 bp on LB at ρ=0.99989 by ingesting kNN-target-mean as
a tree-split feature. qAC ingested decoded features as tree splits
(40 features) and got K=4+1 −0.005 bp. The V4 LB transfer is
specific to (a) the kNN-target-mean feature class and (b) the d16
base context, not all base+feature pairs.

### New observation: qAA carries the load

qAA is the only single-base candidate above +0.1 bp. Its content
(stint_imputed sequence + Frontiers F1 features) is the structurally
distinct contribution. qAB and d18_g are almost equally absorbed
despite very different mechanisms.

### qAA's `prev_lap_delta_stint` corr with existing `LapTime_Delta` = 0.16

Empirically confirmed: our existing `LapTime_Delta` is NOT the
prev-lap-delta that F1 academic literature uses. Distinct feature.

## What this does NOT close

- **V4 pattern with d16-style base.** A LightGBM trained on ORIG with
  qAA features added (as opposed to qAC's synth-trained version) is
  structurally different. orig has different label distribution
  (qX showed mean shift -2.81). Untested.
- **OOF→LB transfer for the K=7 +0.374 bp combo.** Per friction
  `rule-27-abort-threshold-empirically-too-strict-for-sub-bp-moves`,
  ρ in 0.999-0.9999 zone can produce sub-bp LB movement. The K=7
  combo's ρ_test would need direct measurement.
- **Path-B refit on K=5 = K=4 + qAA.** Per friction
  `path-b-cs-absorbs-single-base-orthogonal-additions-below-0.5bp`,
  predicted absorbed (qAA standalone +0.143 OOF is below the 0.5 bp
  threshold).

## Friction tags (proposed)

- `decoded-features-saturate-at-rank-lock-ceiling-+0.4-bp` — every
  decoded feature family absorbs into the LR-meta within +0.4 bp,
  regardless of whether decoded mechanism is sequence (qAA),
  orig-cell-aggregate (qAB), joint-base (qAC), or VGM mode-id (d18_g).
- `qAA-best-single-decoded-base-on-K4-stint-imputed-cardinality-13x` —
  the recovered `stint_imputed` carries +0.143 bp K=4+1 lift, the
  single largest decoded-feature signal.
- `qAB-most-diverse-rho-test-0.899-but-still-meta-absorbed` — the
  lowest ρ_test ever measured on a positively-gating base, yet K=4+1
  lift is only +0.017 bp.

## Pointers

- `scripts/dgp_v3/qAA_stint_imputed_base.py`
- `scripts/dgp_v3/qAB_orig_cell_label_vote.py`
- `scripts/dgp_v3/qAC_joint_stint_orig.py`
- `scripts/dgp_v3/qAE_unified_gate.py`
- `scripts/artifacts/dgp_v3_qAA_stint_imputed_{oof,test}.npy` + `.json`
- `scripts/artifacts/dgp_v3_qAB_orig_cell_{oof,test}.npy` + `.json`
- `scripts/artifacts/dgp_v3_qAC_joint_{oof,test}.npy` + `.json`
- `scripts/artifacts/dgp_v3_qAE_gate_table.json`
