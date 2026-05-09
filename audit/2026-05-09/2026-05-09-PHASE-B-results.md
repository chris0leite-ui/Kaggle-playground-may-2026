# 2026-05-09 — Phase B results: noisy-orig BEATS CTGAN; closing toward 0.83

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-loop-phase-B`
`scripts: scripts/dgp_v3/qB,qC,qE,qE2,qF,qG,qH,qI,qJ,qK`
`artifacts: scripts/artifacts/dgp_v3_q[3..K]_*.json`

> The path to the host's generator does not pass through SDV's library.
> A dead-simple resample-and-scramble pipeline beats every CTGAN class
> we tested. The remaining 0.83 disc-AUC floor is the structural gap
> between orig's per-cell continuous distribution and synth's.

## Architecture exclusion ledger (final for SDV)

| # | Architecture | Config | disc-AUC | Verdict |
|---:|---|---|---:|---|
| 1 | SDV CTGAN | default cond + default sample, 5 ep / 20k orig | 0.9997 | EXCLUDED |
| 2 | SDV CTGAN | default cond + default sample, 10 ep / 101k orig | 0.9993 | EXCLUDED |
| 3 | SDV CTGAN | default cond + **synth marginal**, 10 ep | 0.9993 | EXCLUDED |
| 4 | SDV CTGAN | 20 ep / 80k synth-recursive (P3 prior session) | 0.9993 | EXCLUDED |
| 5 | SDV GaussianCopulaSynthesizer | default | 0.9988 | EXCLUDED |
| 6 | SDV TVAE | 10 ep / 101k orig | 0.9991 | EXCLUDED |
| 7 | SDV CopulaGAN | trained 364 s, fit succeeded but disc not measured | n/a | not measured (pattern conclusive) |

**Conclusion (intermediate):** every SDV synthesiser class converges
on disc-AUC 0.998–0.999 against host. The host's generator is **not in
SDV's library** (or, if it is, with hyperparameters that are not in
the default-near range we tested). All four meaningful tested SDV
variants are within a 0.5 pp band — that's the SDV ceiling, not the
host floor.

## Cheap baselines (without any surrogate)

| What | disc-AUC | Notes |
|---|---:|---|
| orig (sample 1) vs orig (sample 2) | **0.4577** | chance baseline |
| orig vs synth (all features) | 0.9898 | Driver dominates |
| orig vs synth (drop Driver) | 0.7711 | physics + Stint + Race |
| orig vs synth (only continuous: 6 cols) | 0.7436 | synth has different per-cell continuous distribution from orig |

So even *raw orig* differs from *raw synth* at 0.99 disc-AUC, but most
of that comes from the Driver-vocab mismatch. After removing Driver,
the gap drops to 0.77.

## Noisy-orig sweep (qF, qH, qJ): what beat CTGAN

| Variant | sigma | disc-AUC | Δ vs SDV CTGAN 0.9993 |
|---|---:|---:|---:|
| qF: orig + synth-marginal + uniform Driver/Stint scramble | 0.00 | 0.9716 | −2.8 pp |
| qF: + Gaussian noise on continuous | 0.01 | 0.9757 | worse |
| qF: ... | 0.10 | 0.9857 | worse |
| qF: ... | 1.00 | 0.9987 | back to CTGAN level |
| **qH: orig + synth-marginal + cond Driver/Stint** | **0.00** | **0.8323** | **−16.7 pp** ⭐ |
| qH: + Gaussian noise (global std) | 0.10 | 0.9392 | worse |
| qJ: + cell-specific Gaussian noise | 0.02 | 0.8683 | worse |
| qJ: + cell-specific noise | 0.20 | 0.9417 | worse |

Two strong findings:

1. **Adding ANY noise on continuous columns is monotonically worse.**
   sigma=0 (literal orig values, no perturbation) is the best variant
   in every configuration tested. The host does NOT add Gaussian (or
   cell-scaled Gaussian) noise on top of orig values.

2. **Conditional Driver/Stint sampling closes 14 pp of disc-AUC.**
   Going from uniform draw of synth driver vocab (qF) to per-cell
   empirical (qH) drops disc 0.97 → 0.83. The host's Driver/Stint
   distributions are structured by `(Year, Compound, PitStop)`, not
   uniform.

## qK BGMM-on-orig — falsifies "host is per-cell density estimator"

`disc-AUC = 1.0000`. Sampling 6 continuous columns from a per-cell
BGMM smooths integer columns (TyreLife is integer; BGMM emits
continuous values), giving the disc trivial features to detect. Two
takeaways: (a) the host preserves the integer/continuous nature of
each column exactly; (b) per-cell-density-estimator is not the right
abstraction at the column-joint level.

## qI — what's in the remaining 0.83 → 0.5 gap

Iterative-drop on top features of disc(qH-replay vs synth):

| Drop list | disc-AUC |
|---|---:|
| (none) | 0.8323 |
| LapTime | 0.8239 |
| LapTime + LapTime_Delta | 0.8066 |
| + Cumulative_Degradation | 0.7938 |
| + TyreLife | 0.7646 |
| + Stint | 0.6768 |
| + Driver | 0.6630 |
| + LapNumber | 0.6618 |
| + RaceProgress | 0.6091 |

Two structural facts:

- Continuous columns LapTime/LapTime_Delta/CumDeg/TyreLife together
  contribute ~7 pp. Their per-cell distribution differs slightly from
  orig's. The host does NOT add noise (qF/qH/qJ rule that out), so the
  difference is in the underlying value distribution itself.
- Stint and Driver still contribute 8-10 pp **even with conditional
  sampling matched to synth's marginals**. This means there is a
  **higher-order structure** (e.g., joint Driver × Stint, or cond on
  Race in addition to (Y, C, PS)) we're not yet matching.
- Even after dropping the 8 most-important features, disc stays at
  0.61 — the categorical structure (Race × Compound × Year × PS) has
  joint patterns we haven't fully captured.

## Updated DGP picture (post-Phase B)

```
host_pipeline (post-Phase B, intermediate confidence):
  step_1: take orig (101k × 14, drop Norm_TyreLife, drop NaN)
  step_2: sample N=627k rows with custom (Year, Compound, PitStop)
          marginal that suppresses PitStop=1 by 0.54x (F8)
  step_3: for each cell, draw Driver and Stint from a STRUCTURED
          conditional distribution (NOT uniform over the 887/8 vocab,
          but with synth-empirical structure)
  step_4: leave continuous columns AS IS from orig (no noise added)
  step_5: ?(unmodelled) something that affects continuous-column
          per-cell distribution shape
  step_6: ship train.csv + test.csv
```

Remaining uncertainty: whether step 5 is implemented by

- (a) **A per-cell value resample from a slightly-different empirical**
  (e.g., synth uses laps/races sampled with different weights, not
  equally), or
- (b) **A neural per-cell generator** that emits column-joint values
  near orig's per-cell distribution but not literal copies (CTGAN-style
  on a per-cell basis), or
- (c) **A hidden cell axis** the host conditions on (e.g., Race in
  cond-vector beyond (Year, Compound, PitStop)).

(a) is consistent with sigma=0 being best; (b) is not (CTGAN replays
of any kind ≈ 0.999); (c) is the most likely.

## What the disc-AUC ladder tells us

```
host vs SDV CTGAN replay        : 0.9993
host vs orig (raw)              : 0.9898
host vs noisy-orig (uniform)    : 0.9716
host vs orig (drop Driver)      : 0.7711
host vs cond-resample (qH)      : 0.8323
host vs cond-resample minus 4 cont: 0.7646
host vs cond-resample minus 8 features: 0.6091
host vs orig (chance)           : 0.4577 (orig vs orig)
```

We've climbed from 0.999 (CTGAN replay) to 0.83 (cond resample). Each
0.01 of progress represents identifying a specific axis of host
customisation. Five axes characterised:

1. F2 — Driver vocab includes 856 fabricated codes
2. F1 — Stint label is fabricated
3. F8 — PitStop=1 cells are halved in sampling marginal
4. F11 (NEW) — Driver and Stint distributions are structured by
   (Year, Compound, PitStop), not uniform over fabricated vocab
5. F12 (NEW, partial) — continuous columns are not noised; their
   per-cell distribution differs slightly from orig

## Plan v3 update

- **Phase B0 closed**: SDV's library does not contain the host
  generator. Move to non-SDV alternatives (TabDDPM, normalizing
  flow) or non-NN axes.
- **Phase B+ pivot**: keep optimising the analytic resample
  pipeline (qH-style) by adding more conditioning axes:
  - Race in addition to (Year, Compound, PitStop)
  - Joint Driver × Stint conditional matching
  - Per-(cell × Race) sub-sampling of orig rows
- **Phase B-final goal**: get disc-AUC < 0.7 with an analytic
  pipeline. If achievable, that *IS* the host's pipeline (or close
  enough).

## Pointers

- This audit
- `scripts/dgp_v3/qB`, `qE`, `qE2`, `qF`, `qG`, `qH`, `qI`, `qJ`, `qK`
- `scripts/artifacts/dgp_v3_q[B..K]_*.json`
- v3 plan skeleton: `audit/2026-05-09/2026-05-09-plan-v3-PHASE-A-results.md`
- prior corrections: `audit/2026-05-09/2026-05-09-q6q7-tuple-decay-correction.md`
- prior B0 q3q5: `audit/2026-05-09/2026-05-09-q3q5-marginal-not-the-axis.md`

## Friction tags promoted

- `host-not-in-sdv-library` — every SDV variant tested converges on
  disc-AUC 0.998-0.999. Don't waste cycles on more SDV variants.
- `noise-on-continuous-cols-makes-disc-worse-not-better` — host
  preserves orig values exactly. Don't search the noise axis.
- `cond-driver-stint-on-cell-saves-14pp` — promote to a default. Any
  future generative attempt should match synth's per-cell
  Driver/Stint distribution explicitly.
