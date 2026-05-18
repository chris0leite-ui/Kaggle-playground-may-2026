# 2026-05-18 — Round-7 execution audit

## Result headline

**NEW PRIMARY: R7.1 K=13 + Path-B DriverClass × Stint τ=100k →
LB 0.95389** (+0.02 bp over prior PRIMARY 0.95387). Top-5% gap
now 1.6 bp (was 1.8 bp). 7 submissions used today; 3 remaining.

## Phase results

### Phase A — Swap-noise DAE on Kaggle T4 (~50 min T4)

`kernels/r7-swapnoise-dae-gpu/r7_swapnoise_dae.py`. Porto Seguro
recipe: 3-layer MLP encoder [23→256→256→128] with 15% swap-noise
augmentation, 50 epochs MSE reconstruction on train+test combined,
downstream LightGBM on [raw 14 + 9 seg features + 128 DAE embeddings].

**Standalone OOF: 0.94665** (stronger than transformer v1 0.91974 and
v2 0.93330; comparable to HMM 0.94713 and segment-FE 0.94878).

**At K=14+Path-B (all combinations tested)**:
- K=14 + Path-B Compound × Stint: Δ −0.116 bp (REGRESS)
- K=14 + Path-B DriverClass × Stint: Δ −0.092 bp (REGRESS)
- K=14 LR-meta: Δ −0.089 bp (REGRESS)
- K=15 + DAE + TRFv2 + Path-B C×S: Δ −0.151 bp (REGRESS)

**DAE absorbs in EVERY configuration tested.** Standalone OOF
0.94665 is genuine signal but the meta layer extracts no marginal
value over the K=13 row+seq pool. Closes embedding-class as a
direct K=11-pool adder. Saved for next-session retry: different
DAE arch (deeper bottleneck, contrastive loss, deeper down-stream
MLP head).

### Phase B — Multi-segmentation Path-B sweep (~6 min CPU)

`scripts/build_K13_pathb_multiseg.py`. Reusable `run_pathb_segmented`
function with named segmentation schemes.

| segmentation | n_seg | used | OOF | Δ vs R5.2 | ρ vs R5.2 |
|---|---|---|---|---|---|
| **DriverClass × Stint** | 12 | 10 | 0.95447 | **+0.106 bp** | 0.999776 |
| Compound × Stint × LapBucket | 120 | 34 | 0.95447 | +0.065 bp | 0.999876 |
| Year × Compound | 20 | 17 | 0.95445 | -0.149 bp | 0.999678 |

**DriverClass × Stint wins.** The named-vs-anonymous driver split
captures pit-rate variance that the Compound × Stint default misses
(named drivers pit 32-43% vs D0XX 16-22%, per the R4 failure-
analysis agent). 12 segments, 10 above MIN_ROWS — manageable.

**τ sweep on winner**:
- τ=20k: Δ −0.129 bp (too tight)
- **τ=100k: Δ +0.106 bp** (LOCAL OPTIMUM)
- τ=500k: Δ +0.017 bp (too loose)

### Phase C — Multi-pool rank-blend hedge (held)

Built 50/50 R7.1 + R6.1 rank-blend; ρ=0.99995 (TIE_ZONE). Built
3-way R5.2 + R6.1 + R7.1 equal blend; OOF 0.95448, ρ to R7.1 TIE_ZONE.
**Both held to avoid TIE_ZONE slot waste.**

### Phase D — Cross-pollinated combo (R6 fold-fit bag × R7 segmentation)

Combined R6's fold-fit bagging technique with R7's DriverClass × Stint
segmentation: 5-seed fold-fit bag of K=13 + Path-B DC×S τ=100k.

**OOF 0.95450 (+0.264 bp over R7.1 single-seed)** — the strongest
OOF improvement this iteration. Per-seed OOFs 0.95447-0.95450.
ρ vs R7.1 = 0.999973 → TIE_ZONE per Rule 27.

**R7.2 LB submission: 0.95389 — ties R7.1 exactly** at 5-decimal
quantization. The +0.264 bp OOF is real variance reduction; public
LB sees it tied but private LB may show the lift.

## Submissions this iteration

| ref | LB | description |
|---|---|---|
| **R7.1 52778581** | **0.95389** | K=13+Path-B DriverClass×Stint τ=100k (new PRIMARY) |
| R7.2 52779240 | 0.95389 | R7.1 + 5-seed fold-fit bag (ties; hedge) |

Total daily: 7/10; total comp: 49/270.

## Strategic verdict

Round 7 delivered a +0.02 bp LB lift via segmentation choice on
the Path-B operator. The **DriverClass × Stint segmentation** is a
new finding: named-vs-anonymous driver clustering captures
pit-rate variance orthogonal to Compound × Stint. This is the
first segmentation alternative to beat the default in 6 weeks.

DAE absorbed at meta (consistent with NN-class absorption seen in
prior rounds at small K pools). The embedding-class diversity
hypothesis is not falsified per se — different architectures or
contrastive learning might still deliver — but cheap-EV swap-noise
DAE doesn't move PRIMARY.

Combo bag (R6 fold-fit × R7 segmentation) produced largest OOF
improvement of session (+0.264 bp) but LB-quantized away. Private
LB may show the lift.

**Top-5% gap closed from 1.8 bp → 1.6 bp.** Still need +1.6 bp to
break top-5%. Realistic next-session paths:
1. **More segmentations** — Driver-decade × Stint, Stint × first-pit-window
2. **C1 OpenF1 per-Race scalar** (deferred from R7)
3. **DAE v2** — deeper architecture, masked-column pretraining
4. **Multi-segmentation Path-B ensembling** — rank-blend across
   different segmentation outputs
5. **Public-notebook scan** (Rule 22; not done in 17 days)

## Files touched

New:
- `kernels/r7-swapnoise-dae-gpu/r7_swapnoise_dae.py` — Phase A
- `kernels/r7-swapnoise-dae-gpu/kernel-metadata.json` — Phase A
- `scripts/build_K13_pathb_multiseg.py` — Phase B
- `audit/2026-05-18-round-7-execution.md` — this file
- `audit/2026-05-18-round-7-phase-b.json` — Phase B results
- `submissions/submission_K13_pathb_driverclass_stint_tau100000.csv` — R7.1 (LB 0.95389)
- `submissions/submission_K13_dcs_pathb_foldbag.csv` — R7.2 (LB 0.95389)

Artifacts:
- `scripts/artifacts/oof_K13_pathb_{year_compound,driverclass_stint,compound_stint_lapbucket}_tau100000.npy` + test
- `scripts/artifacts/oof_K13_pathb_driverclass_stint_tau{20000,500000}.npy` + test
- `scripts/artifacts/oof_r7_swapnoise_dae_strat.npy` + test
- `scripts/artifacts/oof_K13_dcs_pathb_foldbag_strat.npy` + test
- `scripts/artifacts/K14_dae_pathb_*` + others (DAE held variants)
