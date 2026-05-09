# 2026-05-09 — Phase A2 parallel: Q6+Q7 retract per-row literal-copy

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-loop-phase-A + literal-copy-correction`
`scripts: scripts/dgp_v3/q6_preimage_extended.py, q7_tuple_size_decay.py`
`artifacts: scripts/artifacts/dgp_v3_q6_preimage.json, dgp_v3_q7_tuple_decay.json`

> Two cheap probes (~40 s CPU combined) overturn a load-bearing piece of
> the prior P1c interpretation. The host's CTGAN does NOT literally copy
> orig rows. Per-column marginals are preserved; per-row joints are not.

## Headline retraction

**Prior claim (P1c, 2026-05-08):** "Synth re-uses orig source rows.
Per-column literal overlap of 97.55% on LapTime; 4-tuple match within
synth shows 95% PitNextLap concordance. Each synth row is an
orig-row literal copy with re-assigned (Driver, Stint)."

**Updated claim (Q6+Q7, 2026-05-09):** Synth rows are NOT literal copies
of orig rows. Per-column overlap is real — synth values are drawn from
orig's value pools — but the joint tuple is independently composed.
Of 627,305 synth rows, **only 27 (0.004%)** match an orig row on the
6 best-overlapping columns simultaneously.

The 95% intra-synth tuple concordance from P1c is real but reflects
**CTGAN mode-collapse** (two synth rows sometimes collapsing to the same
generator-output), not orig-row inheritance. Mode-collapse-mate synth
rows share PitNextLap because they share latent + cond, not because
they share an orig source row.

## Q6 — synth→orig 6-tuple match rate

Build per-row fingerprint key = `LapTime|RaceProgress|LapTime_Delta|
Position|TyreLife|Position_Change`. All six columns are KS<0.06 vs
orig with high literal overlap (≥0.91 each).

| Metric | Value |
|---|---:|
| Synth rows | 627,305 |
| Orig rows | 101,305 |
| Synth keys matching orig keys | **27** |
| Match rate | 0.0000 (4.3 × 10⁻⁵) |
| Orig keys ever used in synth | 22 of 100,251 (0.022%) |
| Median uses per used orig key | 1 |
| Max uses per orig key | 3 |

When a match does happen (27 rows): Year inheritance 96%, Compound 56%,
PitStop 89% — sample too small for inference, but the partial Compound
agreement suggests the matches are coincidental rather than structural.

## Q7 — tuple-size match-rate decay

| K | Columns | Match rate | Decay vs K-1 |
|---:|---|---:|---:|
| 1 | LapTime | 0.9755 | — |
| 2 | + RaceProgress | 0.0260 | 37 × |
| 3 | + LapTime_Delta | 0.0012 | 22 × |
| 4 | + Position | 0.0002 | 6 × |
| 5 | + TyreLife | 0.0001 | 2 × |
| 6 | + Position_Change | 0.0000 | 10 × |

Independence ceiling (product of marginals) = 0.9755 × 0.999 × 0.911 ×
1.000 × 1.000 × 1.000 ≈ 0.886. Observed at K=6: 0.000043. Gap = 20,000×.
**Synth columns are not independent within a row, but their joint
distribution does NOT match orig's joint distribution either.**

## Q7.3 — within-cell single-column overlap (where the host actually
synthesises)

Per (Year, Compound, PitStop) cell, fraction of synth values whose
single-column value lies in the orig per-cell value set:

| Column | mean within-cell overlap | min | max | n_cells |
|---|---:|---:|---:|---:|
| Position | 0.997 | 0.932 | 1.000 | 33 |
| TyreLife | 0.985 | 0.859 | 1.000 | 33 |
| Position_Change | 0.979 | 0.836 | 1.000 | 33 |
| RaceProgress | 0.917 | 0.348 | 0.998 | 33 |
| LapTime_Delta | 0.292 | 0.051 | 0.931 | 33 |
| **LapTime** | **0.189** | **0.019** | **0.523** | **33** |

LapTime within-cell literal overlap is **19%** despite 97.55% global
overlap. Mechanism: CTGAN's mode-specific normalisation has modes
shared across cells; when sampling for cell A, a value from a mode
fitted on cell B can leak. Globally that value is "in the orig set,"
but within cell A it is not.

So the layered picture is:
- **Position, TyreLife, Position_Change**: discrete-ish, near-perfect
  per-cell match. CTGAN faithfully reproduces these.
- **RaceProgress**: continuous but bounded by n_laps × cell; mostly
  per-cell-faithful (92%) with a few cells that drift hard (35%).
- **LapTime_Delta**: 29% per-cell faithful; CTGAN actively synthesises.
- **LapTime**: 19% per-cell faithful; CTGAN actively synthesises and
  cross-pollinates modes across cells.

## Implications for inversion strategy

1. **Per-row tuple lookup is dead.** The d18_e2 preimage-kNN with K=10
   was *averaging* over orig neighbours; that's why it gave +1.88 bp
   even though no exact preimage exists. Any lookup-style inversion
   (d18 e2 family, P1c tuple-concordance leverage) is bounded by
   how good the local-density estimate is.

2. **Per-cell density estimation is the right abstraction.** For each
   (Year, Compound, PitStop) cell, fit a density on orig and on synth;
   the inverse encoder's job is to match those densities not row-wise.

3. **Mode-id latent (L1 in plan-v2) is the natural inversion variable.**
   Fit BGMM(K) per column per cell on orig; map each synth row to its
   most-likely mode in each column. Then within a mode, do per-row
   density-ratio inversion. That is the new Phase C plan.

4. **The 81% PitNextLap-concordance ceiling (A33) still applies**, but
   for a different reason. Even within a single (Year, Compound, PitStop)
   cell, PitNextLap is not deterministic given the row's features —
   that's a property of the orig data itself, not of the host's CTGAN.

5. **Cumulative_Degradation should be downweighted in the fingerprint.**
   The Q2 finding (71% global overlap, KS 0.07) plus the cross-cell
   mode-leak in LapTime suggests CumDeg is the most heavily synthesised
   numeric column; tuple keys built on CumDeg will miss everything.

## What this means for the v2 plan

The plan v2 anticipated this risk:

> **Failure mode**: Memorisation sparse (regularised host CTGAN) — 30% — pivot:
> re-aim at population-level statistics rather than per-row preimage.

Q6+Q7 confirm the failure mode at much higher confidence than 30%.
Per-row preimage is essentially impossible. The plan now collapses
toward the F3 (density-ratio) and F5 (optimal transport) families;
F1 (forward surrogate) remains useful as a teacher for F2 (inverse
encoder), but only in the population sense — not per-row.

**Phase C is rewritten:** instead of a contrastive synth→orig encoder
with per-row supervision, train a **per-cell density-matching head**
that estimates `p(x | cell)` separately on orig and on synth, and uses
the density ratio as the per-row label posterior. The 81% A33 ceiling
remains; we are now bounded by per-cell density quality.

## Updated DGP picture

```
host_pipeline (corrected):
  step_1: train CTGAN on full orig (101k × 14 cols, drop Norm_TyreLife)
          with cond_vector = (Year, Compound, PitStop, ?Race, ?Driver)
  step_2: sample 627k rows with custom marginal that
          oversamples (Year=2023, PitStop=0) [F8]
  step_3: per-row sampling: each column drawn near-independently from
          its conditional generator head, possibly drawing from
          mode-specific normalisation modes that cross-pollinate cells
          (especially for LapTime, LapTime_Delta, Cumulative_Degradation)
  step_4: ?(post-hoc) re-assign Driver from a fabricated 887-vocab
  step_5: ?(post-hoc) re-assign Stint label arbitrarily
  step_6: ship train.csv + test.csv (drop Normalized_TyreLife)
```

## Pointers

- This audit
- `scripts/dgp_v3/q6_preimage_extended.py` (6-tuple match rate)
- `scripts/dgp_v3/q7_tuple_size_decay.py` (decay + within-cell overlap)
- `scripts/artifacts/dgp_v3_q6_preimage.json`,
  `scripts/artifacts/dgp_v3_q7_tuple_decay.json`
- Prior incorrect interpretation: `audit/2026-05-08/2026-05-08-p1c-tuple-concordance.md`

## Friction tags

- `synth-rows-are-not-literal-copies-of-orig-rows` — per-column literal
  overlap of 97% does NOT imply per-row literal copy. The joint 6-tuple
  match rate is 0.004%. Promote to rules-history.
- `inversion-by-tuple-lookup-is-impossible` — d18_e2 preimage kNN was
  averaging, not looking up. Future inversion plans must work in the
  density-ratio regime, not the lookup regime.
- `lapTime-within-cell-overlap-19-percent` — CTGAN's mode-specific
  normalisation cross-pollinates LapTime values across cells. Use this
  as a sharp test for any forward surrogate (must reproduce it).
