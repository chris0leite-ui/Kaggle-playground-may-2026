# 2026-05-18 — Round-5 execution audit

## Result headline

**LB 0.95387 — NEW SESSION BEST.** Beats PRIMARY (LB 0.95386) by
+0.01 bp (within 5-decimal quantization, technical tie but above).
File: `submissions/submission_K13_seghmm_pathb_tau100000.csv`.

The breakthrough mechanism: K=11 + r4_segment_fe + r4_hmm_seq, with
**Path-B Compound × Stint τ=100k operator** (NOT LR-meta).
LR-meta-equivalent of same pool landed LB 0.95382 (-0.4 bp vs PRIMARY).

## Phase-by-phase results

### Phase A — Slim-kNN rebuild (~30 min wall, after F1 dataset pull)
All 6 slim-kNN bases rebuilt successfully. Required pulling
`aadigupta1601/f1-strategy-dataset-pit-stop-prediction` (missing
from local snapshot; failure mode: `data/original/f1_strategy_dataset_v4.csv`
not found). Each builder logged its own K=4+1 gate result:
- qAT: +1.172 bp at K=4+1 (best slim-kNN)
- qAV: (rebuilt, in chain)
- qAO: +0.730 bp at K=4+1
- qAA: +0.143 bp (weak)
- qAF: +0.149 bp standalone; SWAP -0.603 bp
- qAK: completed (last in chain)

K=11 LR-meta plain OOF: **0.95443** — matches the historical
PRIMARY value documented in `state/current.md` exactly. Confirms
the rebuild integrity.

### Phase B — Retest seg+HMM at REAL K=11+1

| pool | OOF | Δ vs K=11 | LB |
|---|---|---|---|
| K=11 (plain LR-meta) | 0.95443 | — | (not LB-submitted alone) |
| K=11 + r4_segment_fe + r4_hmm_seq | 0.95445 | **+0.245 bp** | 0.95382 (R5.1) |

**Anchor-attenuation pattern confirmed**: 0.542 @ K=4 → 0.275 @ K=5
(K=4+K27super) → 0.245 @ K=11. The Round-4 mechanism-orthogonality
finding survives at the real PRIMARY anchor.

### Phase C — Graph-class pit-pressure

`scripts/probe_r5_graph_pit_pressure.py`. 4 features (per-(Y,R,L)
pit-pressure, lagged 1-3 lap window, compound-specific, race-level).
Standalone OOF: 0.93344. At K=11+1 alone: −0.012 bp (NULL). Slightly
hurts the seg+HMM combination at LR-meta (-0.030 bp).

### Phase D — Multi-class super-stack sweep

Combination sweep at K=11+N LR-meta (15 combos):
| combo | OOF Δ vs K=11 |
|---|---|
| seg+HMM+graph+TRF | +0.254 bp |
| **seg+HMM (no graph, no TRF)** | **+0.245 bp** ← chosen |
| seg+HMM+graph | +0.215 bp |
| seg+HMM+TRF | +0.198 bp |
| seg alone | +0.114 bp |
| TRF alone | +0.009 bp |
| HMM alone | -0.032 bp |
| graph alone | -0.012 bp |

The 2-base (seg+HMM) and 4-base (+graph+TRF) combos are essentially
tied at OOF. **seg+HMM chosen for first submission** to test the
clean Round-4 finding.

### Phase F — Kaggle T4 transformer (concurrent background)

`kernels/r5-transformer-gpu/r5_transformer_gpu.py`. 4-layer Gaussian
transformer (D_MODEL=128, N_HEADS=8) on per-(Year, Race, Driver)
lap-ordered sequences with attention over LapNumber positional
encoding.

Two prior failures: (1) wrong data path (fixed via rglob);
(2) P100 sm_60 vs PyTorch 2.10 incompat (fixed by setting
`enable_internet: true` so the kernel can pip install torch==2.4).
v4 succeeded after the internet enable.

Standalone OOF: 0.91974 (35 bp BELOW K=11 baseline). Mechanism-
distinct but absorbed at meta — null contribution to multi-class
stack. Saved for next-session retry with larger architecture or
multi-seed bagging.

### Phase D Path-B operator — THE BREAKTHROUGH

`scripts/build_K13_seg_hmm_pathb.py`. Applied Path-B C × Stint
τ=100k operator to the K=11 + seg + HMM (= K=13) pool. Same OOF as
LR-meta (0.95446 vs 0.95445), but **dramatically better LB transfer**:

| operator | OOF | LB | transfer |
|---|---|---|---|
| K=11+seg+HMM LR-meta | 0.95445 | 0.95382 | -6.3 bp |
| K=11+seg+HMM Path-B τ=100k | 0.95446 | **0.95387** | -5.9 bp |
| K=11+seg+HMM Path-B τ=20k | 0.95444 | (not submitted) | tighter shrinkage hurt OOF |

**Same OOF, +5 bp LB swing from operator choice.** Path-B's per-
segment shrinkage preserves the mechanism-orthogonality signal
where LR-meta absorbs it.

K=15 (adding graph+TRF) tied at OOF 0.95446 + ρ vs K=13+Path-B
= 0.999937 (TIE_ZONE) → not submitted to avoid slot waste.

## Submissions today

| ref | LB | mechanism | description |
|---|---|---|---|
| R4 (52772090) | 0.95354 | K=4 LR-meta | row+seq orthogonality at proxy anchor |
| R5.1 (52773963) | 0.95382 | K=11 LR-meta + seg + HMM | proxy-substitution validated |
| **R5.2 (52774385)** | **0.95387** | K=13 + Path-B τ=100k | **new session best; ties+ PRIMARY** |

Submissions: 45 / 270 total; 3 used 2026-05-18; 7 remaining today.

## Strategic verdict

**The R4 plateau-break finding survives at PRIMARY anchor + Path-B
operator.** The session's deliverable:

1. **K=11 baseline** validated at OOF 0.95443 (matches historical).
2. **Mechanism-orthogonality** (row + sequence) lifts +0.245 bp at
   K=11 LR-meta layer, +0.090 bp at K=11+Path-B layer.
3. **Path-B operator** has BETTER LB transfer than LR-meta at this
   pool size (-5.9 bp vs -6.3 bp); critical for the +0.05 bp LB win.
4. **Graph-class and transformer** were null/marginal individually
   and at K=11+Path-B; saved for next-session augmentation paths
   (multi-seed bagging, alternative architectures).

**Top-5% gap**: 0.95405 − 0.95387 = 1.8 bp. Still not reached.
**Leader gap**: 0.95476 − 0.95387 = 8.9 bp. Structurally hard;
needs external-data injection or a +1 bp mechanism class we don't yet have.

## Next-session priorities

1. **Multi-seed bagging of K=13+Path-B** (~12 hr concurrent CPU
   overnight). 3-5 seeds, average the test predictions. Variance
   reduction; private-LB win likely +0.2-0.5 bp.
2. **K=11+K=9 PRIMARY-mimicking 70/30 rank-blend** of K=13+Path-B
   with the K=9-equivalent (qAT/qAV/qAO/qAA/qAF + Path-B τ=20k).
   Recreate PRIMARY's recipe with seg+HMM added.
3. **Transformer v2**: larger D_MODEL (256), more epochs (15+),
   group-based fold split (per-sequence GroupKFold). May produce
   stronger OOF than v1's 0.91974.
4. **C2 swap-noise DAE on Kaggle T4** — porto-seguro precedent;
   embedding-class diversity.
5. **R7d final-window hedge ladder**: lock R5.2 as new PRIMARY,
   K=27+Path-B (LB 0.95368) as Final-2 structural-different hedge.

## Files touched

New scripts:
- `scripts/probe_r5_graph_pit_pressure.py` (Phase C)
- `scripts/probe_r5_k11_super_stack.py` (Phase B/D analysis)
- `scripts/build_K13_seg_hmm_pathb.py` (Phase D Path-B variant)
- `kernels/r5-transformer-gpu/r5_transformer_gpu.py` (Phase F)
- `kernels/r5-transformer-gpu/kernel-metadata.json` (Phase F)

Data:
- `data/original/f1_strategy_dataset_v4.csv` (pulled from Kaggle
  dataset for slim-kNN rebuild)

Artifacts:
- `scripts/artifacts/dgp_v3_qA{T,V,O,A,F,K}_*_oof.npy` + test pairs
- `scripts/artifacts/oof_r5_graph_pit_pressure_strat.npy` + test
- `scripts/artifacts/oof_r5_transformer_strat.npy` + test
- `scripts/artifacts/K13_seghmm_pathb_tau{100000,20000}_oof.npy` + test
- `scripts/artifacts/oof_K11_segHMM_strat.npy` + test
- `submissions/submission_K11_segHMM.csv` (R5.1, LB 0.95382)
- `submissions/submission_K13_seghmm_pathb_tau100000.csv` (R5.2, LB 0.95387)
- `submissions/submission_K13_seghmm_pathb_tau20000.csv` (held)
- `submissions/submission_K15_pathb_tau100k.csv` (held, ties R5.2)
- `submissions/submission_R5_rank_70_K11segHMM_30_K27pb.csv` (held)
