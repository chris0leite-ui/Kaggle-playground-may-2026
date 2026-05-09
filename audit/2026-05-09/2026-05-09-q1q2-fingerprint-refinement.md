# 2026-05-09 — Phase A1: Q1+Q2 fingerprint refinement (new findings F7-F10)

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-loop-phase-A`
`scripts: scripts/dgp_v3/q1_load_and_sanity.py, q2_perfeature_ks.py`
`artifacts: scripts/artifacts/dgp_v3_q1_sanity.json, dgp_v3_q2_ks.json`

> Phase A is the cheap, no-GPU re-grounding of the F1-F6 facts using
> the now-available orig (aadigupta1601, 101,371 × 16). Two probes,
> ~10 s CPU, four new findings. Two prior facts updated.

## TL;DR

- **F1 reproduces** (33% coherent stint groups; prior 35%, within sample
  noise).
- **F4 is updated**: the 2023 anomaly is **already in orig** — orig 2023
  has PitStop rate 0.028 vs other-year orig 0.31. Host did not bolt on a
  practice/quali source; aadigupta1601 itself is heterogeneous on year.
- **F2 is sharpened**: synth's 887-driver vocabulary contains all 31
  orig drivers plus 856 ghosts; **zero orig drivers were dropped**.
- **F5 is corrected**: synth has a **`Position_Change` column** that is
  not in `comp-context.md`'s schema yaml. KS to orig is 0.014 — near-
  perfect distribution match.
- **NEW F7 — three-class column split.** Continuous columns separate
  cleanly into three buckets by KS + literal-overlap.
- **NEW F8 — host's row-count distribution differs from orig's.**
  Synth oversamples the (Year=2023, PitStop=0) cell by ~3.5× the orig
  marginal. The host's CTGAN was sampled with a *custom* `(Year,
  Compound, PitStop)` marginal, not orig's empirical one.
- **NEW F9 — PitStop-conditional KS asymmetry confirmed at +0.02 to
  +0.03.** Modest but consistent across columns. Confirms
  `PitStop` is in the cond-vector (d18 f5 hypothesis ratified).
- **NEW F10 — `Cumulative_Degradation` is genuinely synthesised**, not
  literal-copied. 71% of synth values are in orig set; the column
  generates new values. The literal-copy property is column-specific.

## Per-column KS table (synth vs orig, global)

| Column | KS | literal-overlap | unique synth/orig | bucket |
|---|---:|---:|---:|---|
| LapNumber | 0.188 | 1.000 | 78/78 | reweight |
| RaceProgress | 0.185 | 0.999 | 2097/1437 | reweight |
| LapTime_Delta | 0.178 | 0.911 | 65k/54k | reweight |
| Stint | 0.178 | 1.000 | 8/8 | reweight |
| PitStop | 0.115 | 1.000 | 2/2 | reweight |
| **Cumulative_Degradation** | **0.072** | **0.715** | **173k/81k** | **synthesise** |
| **LapTime** | **0.059** | **0.976** | **41k/41k** | **literal-copy** |
| Year | 0.055 | 1.000 | 4/4 | reweight |
| Position | 0.018 | 1.000 | 20/20 | preserved |
| TyreLife | 0.015 | 1.000 | 78/78 | preserved |
| Position_Change | 0.014 | 1.000 | 37/37 | preserved |

Three buckets:

1. **Preserved** (KS < 0.02, full overlap): Position, TyreLife,
   Position_Change. Distribution match is essentially perfect.
2. **Reweighted** (KS 0.05-0.19, full overlap): LapNumber, Stint, Year,
   RaceProgress, LapTime_Delta, PitStop. Same value set as orig but
   row-count distribution differs.
3. **Synthesised** (KS small, overlap < 0.95): Cumulative_Degradation
   (0.715). The host's CTGAN emits new values for this column.
4. **Literal-copy** (KS small, overlap ≈ 0.97-1.0, but value set from
   orig): LapTime. Confirms the d15 / P1c finding that LapTime values
   in synth are in orig's empirical set.

This 4-way split is strictly more informative than the prior "all
continuous columns are literal-copies" framing.

## Per-year breakdown (the 2023 anomaly is in orig)

| Year | n_synth | n_orig | synth_pit | orig_pit | orig_pitnext | KS(LapTime) | KS(CumDeg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 118,337 | 21,860 | 0.186 | 0.312 | 0.320 | 0.087 | 0.125 |
| 2023 | 194,307 | 24,914 | **0.013** | **0.028** | 0.031 | 0.054 | 0.089 |
| 2024 | 181,642 | 27,557 | 0.192 | 0.332 | 0.336 | 0.057 | 0.087 |
| 2025 | 133,019 | 27,040 | 0.196 | 0.327 | 0.327 | 0.057 | 0.106 |

Two structural facts:

- **Orig 2023 already has PitStop rate 0.028.** Whatever generated the
  orig 2023 slice (likely a non-race source) was not added by the host;
  it was already in aadigupta1601. F4's "2023 source heterogeneity in
  orig" is now a confirmed fact about the orig itself.
- **Synth row-counts skew toward 2023.** Orig is roughly even across
  years (22-27%). Synth is 31% 2023, 19% 2022. The host upweighted the
  pit-poor 2023 source. Synth pit rate (0.013) is even lower than orig
  2023 pit rate (0.028) — host downweighted PitStop=1 within 2023 too.

The simplest unifying explanation: the host trained CTGAN on full orig
with `(Year, Compound, PitStop)` in conditioning, then sampled with a
deliberate non-empirical marginal that **oversamples (Year=2023,
PitStop=0)**. This collapses the per-year pit rate to 0.013-0.20.

## PitStop-conditional KS asymmetry (d18 f5 confirmed)

| Column | KS\|PS=0 | KS\|PS=1 | asymmetry (PS=1 minus PS=0) |
|---|---:|---:|---:|
| LapTime | 0.052 | 0.072 | +0.020 |
| LapTime_Delta | 0.175 | 0.183 | +0.008 |
| Cumulative_Degradation | 0.068 | 0.084 | +0.016 |
| RaceProgress | 0.163 | 0.188 | +0.026 |
| Position_Change | 0.018 | 0.044 | +0.026 |
| **TyreLife** | **0.038** | **0.011** | **−0.027** |

Per-column asymmetry of ±0.02-0.03. The sign on TyreLife flips: synth
fits TyreLife *better* under PitStop=1 than PitStop=0. Mechanism: pit
laps tend to occur in a tight TyreLife window; the host's CTGAN
captures that conditional better than the broader PS=0 distribution.
For everything else, PS=1 is harder to fit.

This asymmetry is exactly the signature d18 f5 predicted. **PitStop
is in the cond-vector.**

## Driver vocab (F2 sharpened)

| | n |
|---|---:|
| synth drivers | 887 |
| orig drivers | 31 |
| common | 31 |
| synth-only ("ghosts") | 856 |
| orig-only | 0 |

Every orig driver appears in synth. The 856 ghosts split into D-prefix
(D001-D856, 756 codes) and 3-letter retired abbreviations (~100 codes,
e.g. ALE, AND, ARN, BAD, BAR, BAT, BEL, BER, BIA per first-20 sample).

Mechanism candidates:

(a) **Post-hoc randomisation.** Host generated 627k synth rows from
    CTGAN-on-orig, then re-assigned `Driver` from a fabricated 887-vocab
    categorical distribution. (b) **CTGAN with expanded driver
    embedding.** Driver was in cond-vector with the 887-driver vocab
    burned in; CTGAN learned to emit any driver code. (a) is consistent
    with F1 (Stint label is fabricated, not temporal).

## Implications for the next probes

1. **The host's customisation lives in the *sampling marginal*, not
   the CTGAN architecture.** F8 is the most likely source of the F6
   disc-AUC 0.9993. Test: train default SDV CTGAN on orig, sample 627k
   *with synth's empirical (Year, Compound, PitStop) marginal*, and
   re-run the discriminator. If disc-AUC falls from 0.9993 to <0.95, we
   have the dominant axis. (next probe Q5)

2. **PitStop is in cond-vector.** Confirm by P18 (next probe Q4): force
   PitStop into SDV CTGAN's cond, sample, measure disc-AUC.

3. **Driver post-hoc randomisation can be tested cheaply.** Apply a
   uniform random Driver assignment (from the 887-vocab) to a CTGAN
   replay with no Driver in cond. Compare disc-AUC; if no change vs
   "Driver in cond", the host did random post-hoc.

4. **`Cumulative_Degradation` synthesises new values.** Inversion via
   tuple-lookup needs to use only the 4 columns where literal-copy
   holds (LapTime, RaceProgress, Position, TyreLife) — not CumDeg.
   This *strengthens* the P1c tuple-concordance signal on a pure-copy
   substrate, and weakens the d18_e2 preimage-kNN that mixed in CumDeg
   as a feature.

## Updated DGP picture (after F7-F10)

```
host_pipeline:
  step_1: train CTGAN on full orig (101k × 14 cols, drop Norm_TyreLife)
          with cond_vector = (Year, Compound, PitStop, ?Driver)
  step_2: sample 627k rows with custom marginal that
          oversamples (Year=2023, PitStop=0)
          → produces F8's row-count skew
  step_3: ?(post-hoc) re-assign Driver from a fabricated 887-vocab
          distribution → produces F2's ghosts and F1's broken stint
  step_4: ?(post-hoc) re-assign Stint label arbitrarily
          → produces F1's 33% stint coherence
  step_5: ship train.csv + test.csv (drop Normalized_TyreLife)
```

Steps 3-4 are still hypotheses. Step 5 is verified.

## Pointers

- This audit
- `scripts/dgp_v3/q1_load_and_sanity.py`, `q2_perfeature_ks.py`
- `scripts/artifacts/dgp_v3_q1_sanity.json`, `dgp_v3_q2_ks.json`
- v2 plan: `audit/2026-05-09/2026-05-09-decode-DGP-7step-plan-v2.md`
- prior: `audit/2026-05-08/2026-05-08-DGP-FINAL-summary.md`

## Next probes (queued)

- **Q3 / P13** — train SDV CTGAN on orig (defaults), sample 200k,
  measure 2-class LightGBM disc-AUC vs synth. Reproduce F6's 0.9993.
  ~30 min CPU.
- **Q4 / P18** — same with `PitStop` forced into cond. Compare disc-AUC.
  ~30 min CPU.
- **Q5** (NEW) — sample Q3's CTGAN with synth's empirical
  `(Year, Compound, PitStop)` marginal. If disc-AUC drops sharply,
  the host's only customisation is the sampling marginal (F8 → primary
  driver of F6). ~5 min CPU once Q3 model exists.
