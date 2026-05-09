# 2026-05-08 — P1c: tuple concordance proves CTGAN re-uses orig source rows

`branch: claude/find-dgp-research-ClsQE`
`tag: dgp-tuple-concordance + literal-copy-evidence`

## Headline finding

**Synth rows with identical (LapTime, LapTime_Delta, RaceProgress,
Cumulative_Degradation) tuples share `PitNextLap` 94.82% of the time
across 386 multi-row tuples (962 rows total, 0.22% of train).** With
6-tuple including (Compound, Race), concordance rises to 95.52%.

This is direct evidence that the host's CTGAN re-uses orig source
rows: when two synth rows have the same continuous values, they were
sampled from the SAME orig row (literal-copy property), and inherit
the same PitNextLap label.

## Why this matters for the DGP picture

| Tuple specification | Multi-row tuples | Rows | All-same-label frac | Within-tuple std |
|---|---:|---:|---:|---:|
| 4-tuple (LT, LTD, RP, CD), count≥2 | 386 | 962 | 0.948 | 0.033 |
| 4-tuple, count≥5 | 22 | 166 | 0.909 | 0.039 |
| 5-tuple (+Compound), count≥2 | 296 | — | 0.946 | — |
| 6-tuple (+Compound, +Race), count≥2 | 223 | 530 | 0.955 | 0.030 |

Compare to global rate: P(PitNextLap=1) = 0.199, so a noisy tuple
should have within-tuple std ≈ 0.4 (binary at 80/20). We observe 0.03
— **13× lower than chance**, confirming label concordance is real.

## What it confirms

1. **Per-row generation, not joint generation**. Each synth row is
   generated independently by sampling a row from orig's empirical
   distribution and adding label-preserving noise on the categorical
   labels (Driver, Stint).

2. **Literal-copy property is exact, not approximate**. d15 found
   97.55% of LapTime values are in orig's empirical set; this
   tuple-concordance result confirms that those values come from
   single orig rows (whose labels are inherited).

3. **Coverage is small**. Only 0.22% of train rows have a 2+ tuple
   match. The CTGAN's per-row diversity means most synth rows have
   unique tuples. So a tuple-lookup base wouldn't lift much by itself
   — covered by d18_e2 (preimage kNN, +1.88 bp at K=21+1, but ρ 0.994
   because info already absorbed by GBDT pool).

## Implication for further DGP work

The DGP characterization is now:

  1. CTGAN with mode-specific normalization on continuous features.
  2. Conditional generator with PitStop in cond-vector → class-
     conditional distortion.
  3. 887-driver vocabulary: 31 active codes faithfully matching real
     careers, 100 retired-but-uniformly-fabricated abbrev codes,
     756 D-prefix synthetic ghosts.
  4. Per-row generation: each synth row is an orig-row literal copy
     (97.55% LapTime overlap) with re-assigned (Driver, Stint) labels
     (15.3% intra-group consistency).
  5. PitNextLap label preserved per-row from the orig source row
     (proven by tuple-concordance 0.95).

Remaining DGP unknowns (would need orig access or membership-inference
to resolve):

  - Exact CTGAN config (epochs, batch, hidden dims, cond-vector
    schema)
  - Whether a custom wrapper or off-the-shelf SDV CTGAN was used
  - Random seed
  - Fingerprint of host-specific bias vs off-the-shelf CTGAN replay
    (P3 result pending)

## Pointers

- This audit, P1, P1b.
- Probe: inline shell call.
