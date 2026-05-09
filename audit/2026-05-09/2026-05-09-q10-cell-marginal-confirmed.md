# 2026-05-09 — Phase A6: Q10 confirms F8 cleanly

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-loop-phase-A`
`script: scripts/dgp_v3/q10_per_cell_marginal.py`
`artifact: scripts/artifacts/dgp_v3_q10_cell_ratio.json`

> One-second pandas groupby. Confirms the F8 hypothesis (host samples
> with custom marginal) precisely.

## Aggregate finding

| Marginal | Synth | Orig | synth/orig ratio |
|---|---:|---:|---:|
| `P(PitStop=0)` | 0.8638 | 0.7483 | 1.154 |
| `P(PitStop=1)` | 0.1362 | 0.2517 | **0.541** |

**Host roughly halves PitStop=1 sampling weight.** Not a soft tweak — a
2× suppression. This is the dominant mechanism producing the 19.9 vs
25.5 pit-rate difference between synth and orig.

## Per-cell ratio table (cropped to extremes)

Top-10 oversampled cells (synth >> orig), all `PitStop=0`:

| Year | Compound | PS | p_orig | p_synth | ratio |
|---:|---|---:|---:|---:|---:|
| 2024 | MEDIUM | 0 | 0.068 | 0.120 | 1.77 |
| 2023 | MEDIUM | 0 | 0.084 | 0.132 | 1.58 |
| 2022 | MEDIUM | 0 | 0.063 | 0.089 | 1.42 |
| 2025 | MEDIUM | 0 | 0.075 | 0.094 | 1.25 |
| 2023 | HARD | 0 | 0.113 | 0.136 | 1.20 |
| 2024 | INTERMEDIATE | 0 | 0.014 | 0.016 | 1.11 |
| 2022 | WET | 0 | 0.002 | 0.002 | 1.11 |

Bottom-10 undersampled cells, all `PitStop=1`:

| Year | Compound | PS | p_orig | p_synth | ratio |
|---:|---|---:|---:|---:|---:|
| 2023 | MEDIUM | 1 | 0.0018 | 0.0007 | 0.38 |
| 2022 | SOFT | 1 | 0.0118 | 0.0043 | 0.36 |
| 2025 | SOFT | 1 | 0.0132 | 0.0047 | 0.35 |
| 2024 | SOFT | 1 | 0.0055 | 0.0019 | 0.35 |
| 2024 | WET | 1 | 0.0001 | 0.0000 | 0.01 |

Cells dropped (p_orig > 0, p_synth = 0): **0**.
Cells invented (p_orig = 0, p_synth > 0): **1** ((2025, WET, PS=0),
mass 5e-6 — a CTGAN mode-leak across cells).

## Mechanism summary

The host's CTGAN sampling distribution is a custom mixture
approximately:

```
P_host(Year, Compound, PitStop) ≈
   0.86 × P_orig(Y, C | PS=0)   +   0.14 × P_orig(Y, C | PS=1)
```

vs. orig's empirical (0.75, 0.25). The cond-vector schema is
`(Year, Compound, PitStop)` (with PitStop *forced* into the cond),
and the sampling marginal uses suppressed PitStop=1 weight. Inside
PS=0, the host slightly upweights MEDIUM (especially in 2024).

Why this matters for the leaderboard goal:

- This shifts the decision-relevant region. Synth has fewer pit
  examples per row → P(PitNextLap=1|x) is computed in a
  PitStop=0-heavy regime. Models trained on orig directly transfer
  with a calibration mismatch (which `d16_orig_continuous_only` sees;
  ρ to PRIMARY is only 0.85).
- Inverting `f` requires inverting the *marginal* re-weighting too, not
  just the per-row generation.

## Next probe

Q3+Q5 (in flight) will test directly: SDV CTGAN-on-orig with the
*synth marginal* applied at sampling. If disc-AUC drops from 0.999
toward 0.5, the marginal is the dominant axis. If it stays high, we
also need architecture changes.
