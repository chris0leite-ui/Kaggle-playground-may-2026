# 2026-05-06 — synthetic-data-aware probe batch (8 probes parallel)

PI reframe: "data is synthetic, physical meaning removed or only
noisily present." Pivoted from physically-motivated transforms to
statistical-artifact-exploitation.

## Final tabulation

| # | Probe | Standalone OOF | Δ at K=21+1 | ρ vs PRIMARY | Verdict |
|---|---|---:|---:|---:|---|
| 1 | id-order audit (read-only) | n/a | n/a | n/a | **finding** below |
| 2 | FE combo (wide-TE/qbin/test-rank) | KILLED at fold 0 | — | — | abandoned (CPU) |
| 3 | KD-distilled LightGBM | 0.94212 | **+0.526 bp** | 0.9955 | marginal+; meta-derivative |
| 4 | NN with embedding layers | 0.92362 | −0.025 bp | 0.9180 | NULL (most-diverse measured, but no gain) |
| 5 | Lap-mod features (synth-artifact direct) | 0.94076 | +0.002 bp | 0.9959 | NULL |
| 6 | Pseudo cascade (top/bot 5%) | 0.94083 | +0.019 bp | 0.9956 | NULL |
| 7 | Driver-cluster Path B (cohort axis) | (cohort) | −0.44 to −0.92 bp | 0.996–0.997 | NULL |

**Net: 0 LB-eligible candidates, 6 NULL meta-gate, 1 killed.**

## The id-order audit finding (load-bearing for the synth-data thesis)

`LapNumber_mod_10` target rate spans **566 bp** (rates 22.98%, 17.31%,
18.19%, 18.00%, 20.07%, 19.38%, 21.09%, 19.31%, 21.10%, 22.88%). Other
modular patterns: `LapNumber_mod_3` 240 bp, `mod_5` 263 bp, `mod_7` 240
bp; `id_mod_1000` 568 bp. Test ids ENTIRELY OUTSIDE train range
(train [0, 439139]; test 439140+).

**Hypothesis (now falsified):** synthetic generator left modular patterns
that GBDTs can't exploit (split by threshold, not modulo).

**Result:** lap-mod LightGBM standalone OOF 0.94076 (+0.01 bp over
baseline 0.94075); min-meta gate +0.002 bp NULL. Feature importance
showed `id_mod_13` rank 14 by gain, `LapNumber_mod_10` rank 15 — the
LightGBM USES the mod features but gains essentially nothing over
baseline-feature-set capacity. **The 566 bp marginal pattern is fully
captured by existing GBDTs via interactions of LapNumber × other
features.** Marginal-bin span ≠ predictive lift in joint model.

## Cross-probe pattern (now triangulated)

The K=21 pool's outer LR meta has very high absorption capacity for
ANY single-base addition. The full ρ-vs-Δ relationship measured
this week:

| candidate | ρ vs PRIMARY | K=21+1 Δ | result |
|---|---:|---:|---|
| nn_embeddings | 0.918 | −0.025 bp | NULL |
| year_stint_sparse_lr | 0.844 | +0.05 bp | NULL |
| within_race_lt_q5 | 0.996 | +0.20 bp | NULL/marginal |
| lap_mod_features | 0.996 | +0.002 bp | NULL |
| pseudo_cascade | 0.996 | +0.019 bp | NULL |
| blend_rank_mean_K21 | 0.987 | +0.015 bp | NULL |
| d10d_leak_corrected_meta | 0.981 | −0.040 bp | NULL |
| d6_rule_compound_stint | 0.936 | −0.020 bp | NULL |
| **d12_lr_meta** | **0.996** | **+1.348 bp** | meta-derivative; LB **regress −4 bp** |
| **kd_lgbm** | **0.995** | **+0.526 bp** | meta-derivative; **LB regress predicted** |

Two patterns:
1. **High ρ-diversity ≠ high meta-utility.** ρ=0.918 (NN) and
   ρ=0.844 (sparse-LR) both produced NULL.
2. **Only meta-derivative-as-base candidates produce visible OOF lift,
   and that lift doesn't transfer to LB** (d12_lr_meta confirmed
   −4 bp; KD predicted same family).

The K=21 LR meta with `expand([raw, rank, logit])` reproduces high-
diversity bases as convex combinations of the existing pool when the
test-time-distinct signal is already linearly recoverable.

## What synthetic-data lens has clarified

The synth-data reframe was correct at one level (physical-feature
derivations are dead) but **the deeper constraint is architectural**:
the K=21 + outer-LR-meta + Path B hier-meta architecture has saturated
within "single-base addition" space. The path to additional lift is:

- **A genuinely orthogonal mechanism class** (e.g., NN with target
  reformulation; SCARF/contrastive; TabPFN — all not currently
  productive based on existing probes).
- **Pool replacement, not augmentation** — change the K_pool composition
  fundamentally (but d13c falsified naive drop-leak-eaters at −2.5 bp).
- **Meta architecture beyond Path B Compound×Stint** — multi-level
  hierarchy / multi-cohort blend (untested per d14 audit).
- **Target reformulation upstream of K=21** — `pit_window_in_lap` or
  `laps_until_pit_ratio` as auxiliary heads (d12 t12 4-of-4 failed,
  but with single-task formulations).

## Submission status

- PRIMARY: d13e_compound_stint_tau20000 LB **0.95049** (unchanged).
- Submissions used today: 1 (K=22 Path B τ=100k → LB 0.95045 regress).
- Submissions used total: 25/270.
- All 7 probe artifacts saved to `scripts/artifacts/`. Pre-submit-diff
  none triggered.

## Process learning

CPU-contention from running 7 LightGBM/NN trainings simultaneously
made each ~4× slower than alone. Net wall time still acceptable
(~3 hours for the batch) but next session should:
- Cap to 3-4 concurrent CPU-heavy probes max
- Schedule cheap probes (≤30s) ahead of slow ones
- Use `nohup` + log-file pattern (already used; works fine)

## Files
- Scripts: `scripts/probe_id_order_audit.py`, `probe_fe_combo.py`
  (killed), `probe_kd.py`, `probe_nn_embeddings.py`,
  `probe_lap_mod_features.py`, `probe_pseudo_cascade.py`,
  `probe_driver_cluster_path_b.py`
- Artifacts: `scripts/artifacts/oof_*_strat.npy` /
  `test_*_strat.npy` for kd_lgbm, nn_embeddings, lap_mod_features,
  pseudo_cascade, drv_cluster_path_b_tau{5k,20k,100k}
- Min-meta JSONs: `scripts/artifacts/probe_min_meta__<name>.json`
- Per-probe summaries: `scripts/artifacts/probe_<name>.json`
