# Round 8 — additional segmentation sweep on Path-B operator

PI: continuation of Day-18 work after R7 found DriverClass × Stint
+0.106 bp OOF / +0.02 bp LB winner. R8 mandate: test 4 more discrete-
axis segmentations to see whether R7 is a one-off or whether the
segmentation axis is a generic lift dimension.

## Results (4 new segmentations tested + R5.2 baseline)

R5.2 Path-B baseline (Compound × Stint τ=100k) OOF: **0.954460**.
R7.1 winner (DriverClass × Stint τ=100k): OOF 0.954466 (Δ +0.106 bp).

| Segmentation | n_seg used | OOF | Δ vs R5.2 | ρ_test vs R5.2 |
|---|---:|---:|---:|---:|
| Year × Stint (smoke) | 17 / 24 | 0.95445 | **−0.100** | 0.99974 |
| **DriverTier × Stint** | 20 / 24 | 0.95447 | **+0.050** | 0.99977 |
| **RaceCluster × Stint** | 10 / 12 | 0.95446 | **+0.042** | 0.99983 |
| Compound × FirstPitWindow | 16 / 20 | 0.95445 | **−0.091** | 0.99976 |

Survivor gate (Δ ≥ +0.10 bp): **0 / 4**. Marginal hits at +0.05
threshold: 2 / 4 (DriverTier, RaceCluster).

Total CPU wall: 122 s smoke + 362 s main = **484 s** (~ 8 min, vs
estimated 30 min — much faster than planned, segment count drove
runtime not row count).

## Cross-axis rank-blend sanity check (R7.1 + DriverTier + RaceCluster)

| Weights | OOF | Δ vs R7.1 | ρ vs R7.1 (rank space) |
|---|---:|---:|---:|
| 1/3 each | 0.954473 | +0.072 | 0.99993 |
| 50 / 25 / 25 | 0.954474 | +0.079 | 0.99996 |
| **60 / 20 / 20** | **0.954474** | **+0.079** | **0.99997** |
| 70 / 15 / 15 | 0.954474 | +0.075 | 0.99999 |
| 50 / 50 R7.1+DT | 0.954473 | +0.069 | 0.99994 |
| 50 / 50 R7.1+RC | 0.954472 | +0.065 | 0.99994 |

Pairwise Spearman of constituents (OOF):
- R7.1 vs DriverTier: 0.99973
- R7.1 vs RaceCluster: 0.99975
- DriverTier vs RaceCluster: 0.99970

All blends land at OOF ≥ +0.065 bp over R7.1 PRIMARY and at
ρ_test ≥ 0.9999 (TIE_ZONE per state/current.md band table). Pattern
matches the prior R6.1 / R7.2 fold-bag pairs: small OOF lifts at
near-1 ρ quantize at LB. 60/20/20 chosen as the most-balanced
blend (highest OOF among the tightest-ρ options).

Artifacts saved (not submitted):
- `scripts/artifacts/oof_R8_blend_60_20_20_r71_dt_rc.npy`
- `scripts/artifacts/test_R8_blend_60_20_20_r71_dt_rc.npy`
- `submissions/submission_R8_blend_60_20_20_r71_dt_rc.csv`

## PI decision

**No submit.** 3 daily slots preserved. Reasoning: blend at TIE_ZONE
ρ likely ties R7.1 at 0.95389 on public LB; if it's still the strongest
hedge candidate by final-window R7d, the saved CSV is ready to ship.

## Segmentation-axis assessment (R7 + R8 combined)

7 segmentations tested across R7 + R8:

| Segmentation | Δ vs C×S baseline | Verdict |
|---|---:|---|
| **DriverClass × Stint** (R7.1) | **+0.106** | WIN |
| Compound × Stint × LapBucket (R7) | +0.065 | marginal |
| DriverTier × Stint (R8) | +0.050 | marginal |
| RaceCluster × Stint (R8) | +0.042 | marginal |
| Compound × FirstPitWindow (R8) | −0.091 | NULL |
| Year × Stint (R8) | −0.100 | NULL |
| Year × Compound (R7) | −0.149 | NULL |

**1 clear WIN out of 7 tested.** Driver-axis (named-vs-D0XX) is
the unique +0.10 bp lift dimension; finer driver quartile splits
(DriverTier 4-class) capture the same signal at lower magnitude
(+0.05 bp) — same axis, no new info. Race-cluster axis (separate
axis from driver) is marginal. Year-axis and first-pit-window axis
NULL.

Friction `two-axis-operator-sweep-missed`: now has more
corroborating data — segmentation IS a real Path-B hyperparameter
axis (we did find a winner), but most variants are null. Promotion
gate (Rule 21 ≥ 3 variants of same key hyperparameter) is now
satisfied for the segmentation axis: 7 variants × 1 win × 3 marginals
× 3 nulls. Still awaiting PI promotion call.

## Operational findings

- **Kaggle CLI 401 on submissions list and kernels list**:
  `KaggleAPIToke=KGAT_a1858...` env var or stripped form both
  fail authentication. Blocks R22 public-notebook scan AND quota
  verification. Friction logged.
- **pre_submit_diff Spearman misleads on rank-blend vs probability
  outputs**: 0.998 reported vs true rank-divergence 0.99997. R7.1
  CSV had 15.6 % of rows floored at np.clip(0.001, 0.999);
  rank-uniform blend has 0 % floored rows; Spearman of the CSVs
  is degraded by tie-structure mismatch (not rank-order divergence).
  Action item: extend pre_submit_diff to rank-normalize inputs
  before correlation OR warn when input distributions differ.

## Files touched

- `audit/2026-05-18-round-8-multiseg.json` (new) — R8 main 3-seg results
- `audit/2026-05-18-round-8-multiseg.log` (new) — R8 main run log
- `audit/2026-05-18-round-8-smoke.json` (new) — Year × Stint smoke
- `audit/2026-05-18-round-8-smoke.log` (modified) — smoke run log
- `scripts/build_K13_pathb_multiseg.py` (unchanged from R7+R8 WIP) —
  ran with --segs flag for the 3-seg main + the smoke pre-stage
- `scripts/artifacts/{oof,test}_K13_pathb_{year_stint,driver_tier_stint,race_cluster_stint,compound_firstpit_window}_tau100000.npy`
- `scripts/artifacts/{oof,test}_R8_blend_60_20_20_r71_dt_rc.npy`
- `submissions/submission_R8_blend_60_20_20_r71_dt_rc.csv`

## Submission count

Daily: 7 / 10 used (unchanged from R7). Comp: 49 / 270.
Days remaining: 13.

## Strategic verdict

R8 closes the "more segmentations" leg of the post-R7 priority queue
with **1 win + 3 marginals + 3 nulls** total across R7+R8. The
DriverClass × Stint discovery is corroborated as the lone winner;
further single-segmentation hunting on this operator has diminishing
returns. Next priorities:

1. **Multi-segmentation rank-blend at TIE_ZONE**: artifact saved;
   PI declined this round but it's ready for final-window R7d
   private-LB hedge.
2. **C1 OpenF1 per-Race scalar join** (~45 min CPU) — still untried.
3. **DAE v2** — deeper bottleneck + masked-column / contrastive on
   Kaggle T4.
4. **Public-notebook scan** (Rule 22, 17+ days overdue) — blocked
   on kaggle CLI 401; needs creds refresh.

PRIMARY unchanged: **R7.1 LB 0.95389**. Top-5% gap −1.6 bp.
