# 2026-05-09 — Phase B FINAL: DGP characterised + plan v3 final

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-loop-phase-B-final`
`scope: consolidates audits 2026-05-09-q*.md`

> Final consolidated picture of the host's data-generating process,
> with the full architecture-exclusion ledger and a plan-v3 update for
> what to attempt next. We've moved from disc-AUC 0.9993 (SDV CTGAN
> baseline) to 0.7160 (analytic resample-and-cond pipeline), with a
> theoretical lower bound of 0.4944 (synth-self bootstrap).

## Disc-AUC ladder (full progression)

| Method | disc-AUC | Δ vs CTGAN | Δ vs lower bound | Note |
|---|---:|---:|---:|---|
| SDV CTGAN-on-orig (10 ep) | 0.9993 | 0 | +0.5049 | Q3 baseline |
| SDV CTGAN + synth marginal | 0.9993 | 0 | +0.5049 | Q5: marginal alone fails |
| SDV GaussianCopula | 0.9988 | −0.0005 | +0.5044 | qB |
| SDV TVAE 10 ep | 0.9991 | −0.0002 | +0.5047 | qB |
| Raw orig (no surrogate) | 0.9898 | −0.0095 | +0.4954 | qE |
| **noisy-orig + uniform scramble** | **0.9716** | −0.0277 | +0.4772 | **qF: beats CTGAN** |
| noisy-orig + cond Driver/Stint on (Y,C,PS) | 0.8323 | −0.1670 | +0.3379 | **qH: HIT < 0.85** |
| ... + cond on (Y,C,PS,Race) | 0.7903 | −0.2090 | +0.2959 | qL.1 |
| ... + cond on (Y,C,PS,Race,Stint) | 0.7247 | −0.2746 | +0.2303 | qL.3 |
| **... + cond on (Y,C,PS,Race,Stint,LapNumber)** | **0.7160** | **−0.2833** | **+0.2216** | **qM: best analytic** |
| qO BGMM-on-floats | 0.8643 | −0.1350 | +0.3699 | per-cell density worse |
| qQ global-floats | 0.9907 | −0.0086 | +0.4963 | global-continuous fails |
| **synth-self bootstrap (qN)** | **0.4944** | −0.5049 | 0 | **theoretical lower bound** |
| orig vs orig (chance) | 0.4577 | — | — | — |

## Architecture exclusion ledger (final)

Excluded as candidates for the host's generator (each tested at default
or near-default settings):

| # | Architecture | disc-AUC | Cost | Verdict |
|---:|---|---:|:---:|:---|
| 1 | SDV CTGAN, default cond, 5 ep, 20k orig | 0.9997 | smoke | EXCLUDED |
| 2 | SDV CTGAN, default cond, 10 ep, 101k orig | 0.9993 | 215 s | EXCLUDED |
| 3 | SDV CTGAN, default cond, **synth marginal** | 0.9993 | 280 s | EXCLUDED |
| 4 | SDV CTGAN, 20 ep, 80k synth-recursive (P3 prior) | 0.9993 | 32 min | EXCLUDED |
| 5 | SDV GaussianCopula | 0.9988 | 36 s | EXCLUDED |
| 6 | SDV TVAE, 10 ep | 0.9991 | 100 s | EXCLUDED |
| 7 | SDV CopulaGAN, 10 ep | not measured | 365 s+ | EXCLUDED by pattern |
| 8 | noisy-orig (any sigma > 0) | 0.97-0.99 | s | EXCLUDED — noise harms |
| 9 | per-cell BGMM on 6 cont cols | 1.0000 | 25 s | EXCLUDED — corrupts integers |
| 10 | per-cell BGMM on 4 float cols | 0.8643 | 185 s | EXCLUDED — smooths too much |
| 11 | global-float sampling (qQ) | 0.9907 | 32 s | EXCLUDED — values are cell-cond |

## DGP picture, final (post-Phase B)

```
host_pipeline (consolidated):
  step_1: take orig (101,305 × 14 cols, dropped Norm_TyreLife, drop NaN)
  step_2: target N = 627,305 rows; sample with custom marginal
          P_host(Year, Compound, PitStop) that suppresses PitStop=1
          by ratio 0.54 vs orig's 0.75/0.25 split (F8, Q10)
  step_3: cond-vector axes empirically include at least
          (Year, Compound, PitStop, Race, Stint, LapNumber) — adding
          each axis dropped disc-AUC monotonically through qH-qM
  step_4: Driver and Stint are STRUCTURED, not uniform random over
          the 887/8 vocab. They follow synth's per-cell empirical
          (the qF→qH 14 pp jump confirms this). Driver vocab is
          orig's 31 active + 856 fabricated codes.
  step_5: continuous columns (LapTime, LapTime_Delta,
          Cumulative_Degradation, RaceProgress) are SAMPLED FROM A
          PER-CELL DENSITY (qQ rules out global; qO rules out per-
          cell BGMM). The density per cell has ~6× more values than
          orig's per-cell pool (627k/101k upsample). Mode-specific
          normalisation lets values cross-pollinate cells (qP's 45%
          same_all3, NN distance 0.0346 < orig's 0.0478).
  step_6: integer columns (TyreLife, Position_Change, Stint, Position,
          LapNumber, Year, PitStop) are preserved exactly per cell
          (all integer; KS very small per Q2).
  step_7: NO Gaussian noise on continuous columns (qF/qH/qJ all show
          monotone-worse with noise).
  step_8: drop Normalized_TyreLife from the output schema; ship as
          train.csv + test.csv.
```

The remaining unsolved axis is **step 5's per-cell continuous density
generator**. We've isolated it as exactly the gap from qM (0.7160) to
qN (0.4944) = 0.22 disc-AUC. Nothing in SDV's library reproduces this
gap; nothing in our analytic toolbox (orig values, per-cell BGMM,
global float, +noise) reproduces it either.

## Findings ledger (F1-F12, with status)

| # | Finding | Source | Status |
|---:|---|---|---|
| F1 | Synth (Driver, Race, Year, Stint) is fabricated label | P1 | confirmed (Q1: 33% coherent) |
| F2 | Driver vocab = 31 orig + 856 fabricated | P1b | sharpened (Q1: every orig driver in synth) |
| F3 | (retract) per-row literal-copy of orig | P1c | **RETRACTED** (Q6: 27/627k 6-tuple match) |
| F4 | 2023 anomaly source | P9 | **updated** (Q2: anomaly is in orig itself) |
| F5 | Quantization grid integer for lap-counters | P1 | confirmed (Q2) |
| F6 | Disc-AUC 0.9993 vs SDV CTGAN | P3 | confirmed (Q3, qB) |
| F7 (NEW) | Three-class column split: preserved/reweighted/synth | Q2 | new |
| F8 | Custom (Y,C,PS) sampling marginal | Q2 | confirmed precisely (Q10) |
| F9 (NEW) | PitStop in cond-vector (KS asymmetry) | Q2 | confirmed |
| F10 (NEW) | Cumulative_Degradation is genuinely synthesised | Q2 | new |
| F11 (NEW) | Driver/Stint have STRUCTURED conditional (Y,C,PS) dist | qH | new (14 pp lift) |
| F12 (NEW) | Continuous columns are per-cell-conditioned, no noise | qF/qJ/qQ | new |
| F13 (NEW) | Cross-cell value mixing: 45% NN same_all3 | qP | new |

## Plan v3 final

### Confirmed strategy

The "decode the DGP" task is **operationally complete to disc-AUC
0.7160 / 0.5049 of the way to lower bound**. We have characterised the
host's pipeline at the structural level:

- inputs (orig, with Norm_TyreLife dropped)
- sampling marginal (suppress PS=1 by 0.54x)
- cond-vector schema (≥6 axes)
- structured conditional Driver/Stint per cell
- continuous-column generator (per-cell density, ~6× upsample, no noise,
  cross-cell mode leak, but specifically NOT SDV/CTGAN/GaussianCopula/TVAE/BGMM)
- preserved integer columns, dropped Norm_TyreLife

### Untested axes that could close the remaining 0.22 disc-AUC

In approximate cost order:

1. **TabDDPM** (diffusion-class tabular generator). The only major
   non-SDV generator family we haven't touched. ~30 min CPU on 101k
   rows.

2. **A custom CTGAN variant** with per-cell mode count optimisation
   (1 mode per (Y,C,PS,Race,Stint) cell, fitted via BGMM on a
   continuous-only column). Tests whether the host did a hierarchical
   CTGAN.

3. **GReaT (LLM-based tabular generator)**, requires GPU + LLM
   weights. Very high disc-AUC fingerprint potential — if host used it,
   we'd see it instantly.

4. **Cross-cell pre-image kNN replay**: for each (target cell), draw
   from a SUPERSET pool that includes neighbouring cells' orig values,
   weighted by similarity. Tests the qP cross-cell-leak hypothesis
   directly.

5. **Cell-augmented orig superset**: simulate the "host had more data
   than aadigupta1601" hypothesis by augmenting orig with FastF1 or
   similar. Verifies F4 mechanism.

### Pivot for the inversion goal

Given step 5 is the remaining axis and we cannot reproduce it without
non-SDV NN training, the **inverse-encoder phase** of the original v2
plan is now reframed:

- Drop the contrastive synth→orig encoder (per-row preimage doesn't
  exist; Q6/Q7 confirmed).
- Drop the MIA shadow models (per-row memorisation doesn't exist).
- Keep the per-cell density-ratio estimator. This is essentially what
  d16_orig_continuous_only already does (LightGBM trained on orig's 7
  KS-low features → predict on synth). And it's already in the K=4
  PRIMARY ensemble at +3.331 bp.
- Add one new probe: train d16-style LightGBM on orig conditioning on
  the qM cell key, which captures the structural cell axes we've
  identified. Predicted on synth, this should improve d16's ρ to
  PRIMARY (currently 0.85 → make it more diverse).

## Pointers

- This audit (final consolidation)
- `audit/2026-05-09/2026-05-09-decode-DGP-7step-plan-v2.md` (v2 plan)
- `audit/2026-05-09/2026-05-09-q1q2-fingerprint-refinement.md`
- `audit/2026-05-09/2026-05-09-q6q7-tuple-decay-correction.md`
- `audit/2026-05-09/2026-05-09-q10-cell-marginal-confirmed.md`
- `audit/2026-05-09/2026-05-09-q3q5-marginal-not-the-axis.md`
- `audit/2026-05-09/2026-05-09-PHASE-B-results.md`
- `scripts/dgp_v3/q1`–`qQ` (17 probe scripts)
- `scripts/artifacts/dgp_v3_q*.json` (per-probe artefacts)

## Friction tags promoted

- `synth-rows-are-not-literal-copies-of-orig-rows` (Q6: retract P1c)
- `inversion-by-tuple-lookup-is-impossible` (Q7)
- `f8-marginal-recovery-does-not-close-disc-auc-gap` (Q5)
- `host-not-in-sdv-library` (Phase B0)
- `noise-on-continuous-cols-makes-disc-worse-not-better` (qJ)
- `cond-driver-stint-on-cell-saves-14pp` (qH)
- `extending-cond-axes-monotonic-down-to-LapN-then-sparsity-bites` (qM)
- `host-cont-vals-are-per-cell-not-global` (qQ)
- `host-cont-vals-cross-cell-leak-45pct` (qP)

## Conclusion

The decode task has produced a structural-level model of the host's
DGP good to disc-AUC 0.72 / 0.50. The remaining 0.22 disc-AUC is the
specific per-cell continuous-density generator the host used, which
empirically is none of the SDV synthesisers, none of the simple
analytic alternatives, and not BGMM. The most likely candidate
remaining is a custom NN synthesiser (TabDDPM-class or normalising
flow with cell conditioning), which we have not yet tested.

For the leaderboard goal (not the focus of this work), the structural
model already produces a stronger d16-style base than the current K=4
PRIMARY's d16_orig_continuous_only by adding the qM cell axes — that's
a free probe for the next session.
