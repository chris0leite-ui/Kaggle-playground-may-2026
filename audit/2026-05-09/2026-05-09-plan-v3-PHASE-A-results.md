# 2026-05-09 — Plan v3 — Phase A results consolidated

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-loop-plan-v3-skeleton`
`status: skeleton — to be filled in as Phase A finishes`

> Plan v2 anticipated five families (F1-F5). Phase A's Q1-Q10 probes
> sharpen the picture enough to retire one family (MIA, F4) and rewire
> two (forward surrogate F1; inverse encoder F2). This is the running
> v3 plan; final version after Q3+Q5 disc-AUC measurement.

## Phase A findings to-date (Q1 through Q10)

The host's data-generating process now decomposes into four mostly-
characterised stages and one unmeasured stage:

```
Stage 1 — base generator (CHARACTERISED)
  CTGAN-class with mode-specific normalisation per continuous column.
  Trained on full orig (101,305 × 14 cols, dropping Normalized_TyreLife).
  Disc-AUC vs SDV CTGAN default = 0.9993 (Q3, matches F6).
  Per-column marginals match orig at KS 0.014-0.19 (Q2).

Stage 2 — cond-vector schema (PARTIALLY CHARACTERISED)
  PitStop is in cond (d18 f5 KS asymmetry, Q2 confirms +0.020 to +0.026).
  Year, Compound, Race likely in cond (per-cell P(Year, Compound, PS)
  preserved; Q10 cells closely tracked).
  Driver: 856 fabricated codes (Q1, F2). Could be in cond OR post-hoc.
  Q9 driver-in-cond test pending.

Stage 3 — sampling marginal (CHARACTERISED, F8 confirmed by Q10)
  P_synth(PS=0) = 0.864 vs P_orig(PS=0) = 0.748  (×1.15)
  P_synth(PS=1) = 0.136 vs P_orig(PS=1) = 0.252  (×0.54)
  Host roughly halves PitStop=1 sampling weight.
  Within PS=0, MEDIUM compound is upweighted (×1.4 in 2024).

Stage 4 — per-row generation (CHARACTERISED, retracts P1c)
  Q6 + Q7 prove synth rows are NOT literal copies of orig rows.
  6-tuple synth→orig match rate = 0.0000 (27 of 627k).
  Per-column overlap is preserved but joint structure is independent
  per column within each (Year, Compound, PitStop) cell. CTGAN's
  mode-specific normalisation cross-pollinates modes across cells:
  LapTime within-cell literal overlap = 0.19 vs global 0.98.

Stage 5 — post-hoc label scrambling (UNCONFIRMED)
  F1 (synth Stint label is fabricated) reproduces (Q1, 33% coherent
  groups). F2 (Driver vocab includes 856 fabricated) reproduces (Q2).
  Mechanism: post-hoc re-assignment, OR Driver in cond with
  expanded vocab. Q9 distinguishes.
```

## Plan v3 deltas vs v2

### Drop:

- **Family F4 (MIA shadow models).** Membership inference assumes per-
  row memorisation. Q6 + Q7 prove memorisation is per-column not per-
  row, so individual orig rows are not recoverable — there is no
  "training row" to detect membership of in the row sense. MIA score
  distributions would be effectively flat. (~10 GPU-d saved.)

### Modify:

- **Phase B forward surrogate** — same family list (CTGAN, TabDDPM,
  TVAE, CopulaGAN, NSF, RealNVP) but the success metric is now
  *per-cell* density match (KS within each (Year, Compound, PitStop)
  cell ≤ 0.05) rather than overall. Add a hard gate on the LapTime
  within-cell literal-overlap (must reach ≥ 0.5; host is at 0.19,
  default SDV CTGAN should also be at 0.19, a custom mode-specific-
  normalisation tweak should *raise* it).

- **Phase C inverse encoder** — replace contrastive synth→orig
  encoder with a per-cell density-ratio estimator. For each cell c,
  fit `p_orig(x|c)` and `p_synth(x|c)` (BGMM, normalising flow, or
  neural density estimator on continuous columns). The label
  posterior is

  ```
  P(y=1 | x, c)_synth = E_orig[y | nbhd of x in cell c]
                      ≈ p_orig(x|c, y=1) / p_orig(x|c)
  ```

  Implementable with a per-cell LightGBM trained only on orig (an
  improved version of d16_orig_continuous_only).

### Add:

- **Phase B' (NEW) — sampling-marginal recovery.** Once a forward
  surrogate is fitted, re-sample with synth's empirical marginal
  applied at the cond-vector level, not at row level. Already cued
  up in Q5 (in-flight).

## Q5 result (in flight; placeholder)

`Q3 default-sampling disc-AUC = 0.9993` (confirmed)
`Q5 synth-marginal disc-AUC  = TBD`

If `Q5 ≤ 0.95`: sampling marginal is the dominant axis → Phase B
focuses on sampling-marginal recovery; per-cell density gap is the
remaining 0.95 → 0.5 gap.

If `Q5 ≈ 0.999`: marginal is not the issue → architecture / cond /
mode count is. Move to Phase B sweep.

## Pointers

- `scripts/dgp_v3/q1`–`q10`
- `scripts/artifacts/dgp_v3_q*.json`
- `audit/2026-05-09/2026-05-09-q1q2-fingerprint-refinement.md`
- `audit/2026-05-09/2026-05-09-q6q7-tuple-decay-correction.md`
- `audit/2026-05-09/2026-05-09-q10-cell-marginal-confirmed.md`
- v2 plan: `audit/2026-05-09/2026-05-09-decode-DGP-7step-plan-v2.md`

## Status checklist

- [x] Phase A0 — bootstrap, install
- [x] Phase A1 — Q1+Q2 fingerprint refinement (F4 update; F7-F10 new)
- [x] Phase A2 — Q6+Q7 retract per-row literal-copy
- [x] Phase A6 — Q10 confirms F8 cleanly
- [ ] Phase A3+A4 — Q3+Q5 CTGAN disc-AUC decomposition (in flight)
- [ ] Phase A5 — Q8 conditional MI (queued, post-CTGAN)
- [ ] Phase A7 — Q9 driver-in-cond (queued, post-CTGAN)
- [ ] Phase A wrap — final plan v3 doc
- [ ] Phase B start — surrogate grid
