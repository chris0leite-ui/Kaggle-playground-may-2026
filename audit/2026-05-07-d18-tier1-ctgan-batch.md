# 2026-05-07 — d18 Tier-1 CTGAN-aware batch (G/H/I/J/K) + F-series gates

`branch: claude/reverse-engineer-data-generation-Hu8EK`
`tag: dgp-ctgan-aware, mode-id-attribution, path-b-cohort-redesign`

> Continuation of d18 dig-deeper round. Triggered by F1 finding that
> host_synth = CTGAN-class GAN. Tests CTGAN-mechanism-aware exploits
> (mode-id attribution, mode-collapse, cond-vector lookup) plus the
> F-series probes (constraint violations, class-cond GMM) that landed
> earlier.

## TL;DR

- **Strongest PRIMARY-replacement candidate: K=28 Path-B Compound×Stint
  τ=20k → OOF 0.95201** (+1.7 bp over current PRIMARY OOF 0.95184).
  Predicted LB lift +1-3 bp over current PRIMARY 0.95149.
- Full-pool LR-meta K=21+7 (d16+d18+G+F2+F5+H+J): **+11.42 bp**, OOF 0.95187.
  Beats current PRIMARY OOF (K=23 Path-B 0.95184).
- 6 new DGP-class bases all pass K=21+1 gate (G/F2/F5/H/I/J: +0.53/+1.56/+3.56/+0.97/+0.80/+2.30 bp).
  F5 strongest single (+3.56 bp), H/G/I weakest (~+0.5-1 bp).
- **Path-B amp axis still hiding**: mode-id cohort variants (K1/K2/K3)
  all NULL vs Compound×Stint. Cohort-axis variation isn't the amp axis.
- **F1 verdict**: host = CTGAN-class GAN. Disc features as bases NULL,
  but the architecture knowledge unlocks G/H/I/J/K probe class.

## Standalone OOF + K=21+1 gate ladder (all DGP-class, sorted by gate Δ)

| Probe | Std OOF | K=21+1 Δ | ρ vs d13e | Mechanism |
|---|---:|---:|---:|---|
| **d18 v1 chain (causal+gauss)** | 0.94954 | **+7.365** ⭐ | 0.991 | per-step orig log-likelihood |
| **F5 class-cond GMM ratio** | 0.94895 | **+3.559** | 0.994 | GMM(8) per class on 5 KS-low feats; log p_y1/p_y0 |
| d16 orig cont_only (PRIMARY 22nd) | 0.91483 | +3.331 | 0.995 | orig-LGBM on 7 KS-low |
| **J cond-vector lookup** | 0.94929 | **+2.305** | 0.994 | EB(P×C×S×R×Y).y_mean (5-way) |
| E2 preimage kNN | 0.94829 | +1.883 | 0.994 | kNN(K=10) per-Compound, 7 KS-low |
| **F2 constraint violations** | 0.94945 | **+1.561** | 0.995 | 10 physical constraints (TyreLife mono, CumDeg=cumsum, etc.) |
| d18b v2 chain (causal+q10) | 0.94834 | +1.426 | 0.995 | q10-multiclass instead of Gaussian |
| H mode-lookup (G+EB) | 0.94876 | +0.970 | 0.996 | EB(C×S×mode_id_feat).y_mean |
| I mode-collapse (bias-factor) | 0.94924 | +0.797 | 0.995 | synth_freq[m]/orig_freq[m] per row |
| **G mode-id (CTGAN latent)** | 0.94877 | +0.532 | 0.995 | BGMM(10) on 7 KS-low feats; mode-id as cat |

## Joint K=21+N panels

```
K=21+6 (G+F2+F5+H+I+J — all NEW d18-class):
  OOF 0.95125  Δ +5.18 bp  ρ=0.9927
  |w| ranking: F5 (0.63) > J (0.40) > F2 (0.34) > H (0.14) > G (0.12) > I (0.07)

K=21+7 (d16+d18+G+F2+F5+H+J — drops weakest I):
  OOF 0.95187  Δ +11.42 bp  ρ=0.9904 ⭐
  |w| ranking: d18 (0.90) > d16 (0.83) > J (0.40) > F5 (0.36) > G (0.30) > F2 (0.28) > H (0.09)
  (BEATS current PRIMARY OOF K=23 Path-B 0.95184)
```

## Path-B sweep — cohort-axis comparison

All on K=28 = K=21+d16+d18+G+F2+F5+H+J pool. τ=20000.

| Cohort | n_seg | OOF | Δ vs global LR (K=28=0.95187) | Verdict |
|---|---:|---:|---:|---|
| **Compound × Stint** | 30 | **0.95201** | **+1.42** | ⭐ best (1.0× amp) |
| mode_TyreLife × Stint (K3) | 66 | 0.95198 | +1.11 | close 2nd |
| Compound × mode_TyreLife (K1) | 55 | 0.95189 | +0.19 | NULL |
| Compound × mode_LapTime_Delta (K2) | 55 | 0.95186 | -0.12 | NULL/regress |

**Cohort-axis variation is NOT the Path-B amp axis.** Even semantically-justified
mode-id cohorts (CTGAN's actual discrete latent) give same lift as plain
Compound × Stint. Mechanistic reading: the amp factor 6-11.6× we saw with
d13e Compound×Stint and d13 Stint was likely a **base-pool composition
effect**, not a cohort-segmentation effect. Once the pool is well-conditioned
(K=21+, with diverse DGP-class bases), Path-B amp degenerates to ~1.0×.

This adds a corollary to friction `path-b-amp-only-fires-on-meta-arch-not-base-add`:
**cohort-axis variation also doesn't fire amp**. The amp lives in
non-Gaussian shrinkage / BMA / non-linear meta — untested at K=28.

## CTGAN architecture verdict (F1)

| Arch | Mean KS vs host | Disc AUC | Synth P(replay-like) |
|---|---:|---:|---:|
| **CTGAN** | **0.1344** ⭐ | 0.9884 | 0.1286 |
| CopulaGAN | 0.1386 | 0.9877 | 0.1301 ⭐ |
| TVAE | 0.1646 | 0.9894 | 0.1163 |
| GaussianCopula | 0.1410 | 0.9967 | 0.0613 |

Host = CTGAN-class GAN. Three independent confirmations:
1. Lowest mean KS to CTGAN replay (0.1344).
2. Biggest jump in P(replay-like) at non-GAN→GAN boundary
   (0.061 GaussianCopula → 0.13 CTGAN/CopulaGAN).
3. Consistent with 97.55% LapTime literal-overlap (CTGAN mode-specific
   normalization signature).

All 4 disc AUCs ≥ 0.988 — host has its own signature (custom preprocessing,
conditioning, or wrapper). Per-row disc features as bases: NULL (-0.112 bp
K=21+4) — arch-bias is orthogonal to PitNextLap target signal.

## Diagnostic findings (mechanisms learned)

### Synthesizer broke physical constraints differentially by class
F2 surfaced strong class-conditional violation patterns:
- **`viol_C8_tl_within_stint`**: y=0=77.9% vs **y=1=97.5% (+19.6%)** —
  synthesizer breaks within-stint TyreLife=lap_idx consistency much
  more for PitNextLap=1 rows.
- `viol_C3_cumdeg_drift`: y=1 +8.7% (CumDeg ≠ cumsum(LapTime_Delta))
- `viol_C6_poschange`: y=1 +8.9% (Position_Change ≠ ΔPosition)
- `viol_C9_pit_stint_next`: y=1 +7.0% (PitStop=1 doesn't precede stint+1)
- `viol_count` total: y=0=296% vs y=1=342% (+46%, average y=1 row violates
  ~3.4 constraints vs ~3.0 for y=0)

This is direct evidence of CTGAN's class-conditional generator producing
less physically-consistent rows for the y=1 class. Likely because PitStop
(y proxy) was in the cond-vector and the generator over-adjusted on it.

### CTGAN modes are highly class-discriminative
G's BGMM on LapNumber found 10 modes with HUGE class-rate spread:
- mode 8: 5.4% pit rate
- mode 2: 6.7%
- mode 9: 36.0% (7× the lowest mode)
- mode 5: 34.8%, mode 0: 34.4%

But the class-discriminative signal is *largely already captured* by
LGBM's tree splits on LapNumber raw value. Mode-id-as-categorical adds
+0.53 bp K=21+1 — modest because raw LapNumber tree splits cover most.

### Mode-collapse exists but signal is weak
I (bias-factor) surfaced clear mode-collapse: TyreLife mode 8 has
synth_freq=0.124 vs orig_freq=0.067 → **bias_factor 1.84 (84% over-sampled)**.
But the bias-factor features have AUC < 0.51 individually — the
over-sampled modes don't correlate strongly with target. Adding them to
LGBM nets +0.80 bp K=21+1 (modest).

### Cond-vector tuple lookup is strong AT FEATURE LEVEL
J's `cv_pcsry` (5-way `(PitStop, Compound, Stint, Race, Year)` empirical
P(y=1)) has standalone AUC **0.8896**. But at K=21+1 LR-meta level, only
+2.30 bp because raw features already encode this via tree interactions.

## Updated PRIMARY-replacement decision matrix

Current PRIMARY: K=23 Path-B Compound×Stint τ=20k → **LB 0.95149**.

Candidates ranked by predicted LB:

| Candidate | OOF | Δ vs PRIMARY OOF | ρ vs PRIMARY (test) | Predicted LB Δ |
|---|---:|---:|---:|---:|
| **K=28 Path-B Compound×Stint τ=20k** ⭐ | **0.95201** | **+1.7 bp** | TBD (likely ~0.998) | +1 to +3 bp |
| K=28 LR-meta (no Path-B) | 0.95187 | +0.3 bp | TBD | -1 to +1 bp |
| K3 mode_TL×Stint Path-B τ=20k | 0.95198 | +1.4 bp | similar | +1 to +2 bp |
| K=23 Path-B (current PRIMARY) | 0.95184 | 0 | 1.0 | 0 |

**Recommendation**: submit K=28 Path-B Compound×Stint τ=20k as
PRIMARY-replacement. R7 flip count vs current PRIMARY needs computation
(flip count vs d9f K=21 swap was 344/261 — over R7 cap; but vs current
PRIMARY it should be lower since the pool is augmented).

## Top-level interpretation

The DGP reverse-engineering campaign (E1-E5 + F-series + G/H/I/J/K) is
**foundationally complete**. We have:

1. **Architecture identified**: CTGAN-class GAN with custom conditioning.
2. **Synthesizer mechanisms explained**:
   - Mode-specific normalization → 97.55% literal LapTime overlap.
   - Conditional generator with PitStop in cond-vector → sharper
     class-conditional KS in synth (0.43 vs orig 0.24).
   - Per-feature near-conditional-independence (d14) → joint corruption,
     marginal preservation.
3. **6 new DGP-class bases** built and gated (G/F2/F5/H/I/J), all pass.
4. **Joint stack reaches OOF 0.95187** at K=21+7 LR-meta — already beats
   the current PRIMARY OOF.
5. **Path-B amp axis confirmed elusive**: cohort-axis variation
   (Compound×Stint, mode-id×Stint, Compound×mode-id) all give ~1.0× amp.
   Real amp lives in shrinkage prior / BMA / non-linear meta.

**The remaining 19.6 bp gap to top-5% is most likely:**
- a meta-arch redesign (T4a-d in idea-board: Student-t shrinkage, BMA),
- external aggregate priors (Pirelli pit-window scrape), or
- a data-leak-class signal that we haven't yet found (membership inference,
  exact-row matching beyond the leak-lookup we did).

## Pointers (artifacts created this batch)

Scripts:
- `scripts/d18_g_mode_id_ctgan.py` — BGMM(10) per KS-low feature
- `scripts/d18_h_mode_lookup.py` — EB(C×S×mode_id_feat) lookup
- `scripts/d18_i_mode_collapse.py` — bias-factor synth_freq/orig_freq
- `scripts/d18_j_cond_vector_lookup.py` — EB cond-vector tuple lookup
- `scripts/d18_k_pathb_mode_cohort.py` — Path-B with mode-id cohorts
- `scripts/d18_path_b.py` — extended with `--variant k28_full_dgp`

Pre-existing F-series scripts (run this batch):
- `scripts/d18_f2_constraint.py`, `scripts/d18_f5_class_cond_gmm.py`,
  `scripts/d18_f6_kl_ceiling.py`

OOF/test artifacts:
- `oof_d18_{g_mode_id, f2_constraint, f5_class_cond_gmm, h_mode_lookup,
  i_mode_collapse, j_cond_vector}_strat.npy` (+test variants)
- `oof_d18_path_b_K28_full_dgp_tau{5k,20k,100k}_strat.npy` (+test) ⭐
- `oof_d18_k{1,2,3}_pathb_*_tau{5k,20k,100k}_strat.npy` (+test)

Gate JSONs:
- `probe_min_meta__d18_{g,f2,f5,h,i,j}_*.json` (6 K=21+1 gates)
- `probe_min_meta__d18_g+f2+f5+h+i+j.json` (K=21+6 joint)
- `probe_min_meta__d16+d18+g+f2+f5+h+j.json` (K=21+7 full pool) ⭐

Audits:
- `audit/2026-05-07-d18-ideaboard.md` — locked queue (still relevant for
  E4 fix, v3 reverse-causal, T4a-d, Pirelli, M residual disc, N CTGAN
  augmentation)
- This file — synthesis

Diagnostic parquets (gitignored, regenerable):
- `data/mode_id_features_{train,test}.parquet` (627k × 9 cols, BGMM mode-ids)
- `data/chain_decomp_features_{train,test}.parquet` (from d18 v1)

## Next-session priorities

1. **Submit K=28 Path-B Compound×Stint τ=20k** as PRIMARY-replacement
   (PI sealed-prediction first per Rule 26a; R7 flip check vs current
   PRIMARY).
2. **Run pre_submit_diff** vs current PRIMARY (LB 0.95149).
3. **Idea-board Tier-3 meta-arch redesign**: T4a Student-t shrinkage on
   K=28 (the real Path-B-amp axis test).
4. **Tier-3 T4b Yao/Vehtari covariance-modelled BMA** (Kaggle GPU PyMC-JAX).
5. **External Pirelli pit-window scrape** (Tier-5).
6. **E4 with predict-batch chunking** (parked); v3 reverse-causal alone.
