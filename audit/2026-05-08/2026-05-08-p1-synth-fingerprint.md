# 2026-05-08 — Phase 1: pure-synth DGP fingerprint

`branch: claude/find-dgp-research-ClsQE`
`tag: dgp-no-public-csv`
`mechanism family: dgp_label_fabrication`

> PI directive: find the DGP without leaning on the aadigupta1601 public
> CSV. Phase 1 probes synth (train + test, 627,305 rows) directly to
> identify CTGAN-class signatures and structural fabrications.

## Headline finding

**The synth's `(Driver, Race, Year, Stint)` tuple is a fabricated
categorical label, not a true stint identifier.** Only **15.3% of
124,520 stint groups have all rows agreeing on the implied stint-start
lap** (`LapNumber − TyreLife + 1`). 35% of multi-row groups have every
row implying a *unique* stint start.

This means the host's CTGAN sampled rows independently from the
orig empirical marginal (preserving per-row physical relationships
within the row), then assigned synthetic `(Driver, Stint)` labels
conditioned on `(Race, Year, Compound)`. Two rows with the same
`(Driver, Race, Year, Stint)` synth label are NOT actually from the
same orig stint.

## What this changes

1. **Group-by aggregations on `(Driver, Race, Year, Stint)` tuples
   are noisy**: rows aren't actually from a coherent stint. Path-B's
   Compound × Stint cohort axis still works because it conditions on
   Compound (preserved exactly) and Stint (synthetic categorical that
   correlates with TyreLife range), but Stint itself has no temporal
   meaning.
2. **`stint_start_imputed = LapNumber − TyreLife + 1` is a stronger
   per-row identifier than the synth Stint label.** Every synth row
   has a coherent `(Race, Year, Compound, stint_start_imputed)`
   slice that maps to a real orig stint (or stint-start lap).
3. **Within-stint sequence FE that uses synth Stint as the time index
   is broken by construction.** This explains A4's "synth stint mean
   3.87 vs orig 19.80" finding — synth has 124k stints (24× more than
   orig's 5119) because the labels are fabricated, not because of
   downsampling within real stints.

## Probe methodology (synth-only, no orig data used)

For 30,000 sampled `(Driver, Race, Year, Stint)` groups with ≥2 rows:

- Compute `stint_start_imputed = LapNumber − TyreLife + 1` per row.
- For each group, compute std and n_unique of `stint_start_imputed`.
- A "true" stint group should have all rows agreeing (n_unique=1).

| Subset | n groups | std median | n_unique=1 frac | every-row-unique frac |
|---|---:|---:|---:|---:|
| All drivers | 30,000 | 2.43 | **0.153** | 0.355 |
| 3-letter abbrev (real names, e.g. MAS, RAI) | 5,000 | 2.55 | 0.129 | — |
| D-prefix ghosts (D001-D856) | 5,000 | 2.47 | 0.162 | — |

Median std is 2.43 laps within a single (Driver, Race, Year, Stint)
group; p90 is 6.6. This is far too large for a real stint (real
stints have all rows sharing exactly one stint_start by construction).

## Other Q1-Q6 findings

### Q1 — Quantization grid

Numeric columns are integer-valued except for the obvious continuous
ones (`LapTime`, `LapTime_Delta`, `Cumulative_Degradation`,
`RaceProgress`). Note an isolated TyreLife outlier value `60.5`
(1 row in 627k); the rest of TyreLife is integer 1-77. **Grid is
integer for all "lap-counter" columns.** No host-introduced fractional
grid found.

### Q2 — id-ordering is uninformative

`adjacent / random` distance ratio = 0.9988 on standardized 8-numeric
features. **CTGAN did not generate in batches with shared latent.**
Falsifies one mode of inversion-via-id-clusters.

### Q3 — Driver code structure

| Subset | n drivers | n rows | rows/driver median | races/driver median |
|---|---:|---:|---:|---:|
| All | 887 | 627,305 | — | — |
| D-prefix (D001-D856) | 756 | 373,277 | 99 | 21.5 (out of 26) |
| 3-letter abbrev | 131 | 254,028 | 1,939 | — |
| Other | 0 | 0 | — | — |

D-prefix ghosts each appear in median 21.5 of 26 races — fully
distributed across the schedule. Ghost driver IS NOT race-localized.

### Q4 — Within-stint coherence

124,520 synth stint groups (~24× orig's 5,119). Of 37,793 multi-row
groups sampled:

| Property | frac |
|---|---:|
| Compound constant within stint | 1.000 |
| LapNumber strictly increasing | 0.117 |
| TyreLife strictly increasing | 0.117 |
| LapNumber strictly consecutive (Δ=1) | 0.007 |

Compound is preserved; LapNumber/TyreLife monotonicity is broken.
This *is* consistent with the headline finding: the rows in a synth
"stint group" aren't a coherent sequence. They're independent draws
that happen to share a (Driver, Race, Year, Stint) label.

### Q5 — Class-conditional rate spread per 10-quantile bin

| Feature | y-rate spread |
|---|---:|
| TyreLife | 0.372 (0.017 → 0.389) |
| Stint | 0.331 |
| RaceProgress | 0.330 |
| LapNumber | 0.310 |
| LapTime_Delta | 0.279 |
| Cumulative_Degradation | 0.217 |
| LapTime | 0.147 |
| Position | 0.073 (weakest) |

TyreLife is the strongest single predictor. Position is weakest.

### Q6 — Top-5 mutual-information pairs (100k subsample)

| Pair | MI |
|---|---:|
| RaceProgress ↔ LapNumber | 3.71 |
| TyreLife ↔ LapNumber | 1.30 |
| TyreLife ↔ RaceProgress | 1.26 |
| LapTime ↔ Cumulative_Degradation | 0.43 |
| Cumulative_Degradation ↔ RaceProgress | 0.42 |

`RaceProgress = LapNumber / TotalLaps` is preserved at near-perfect
MI. Tyre wear physics partly preserved (LapTime ↔ CumDeg).

## Implications for next probes

1. **Phase 2** — Build a feature engineered from `stint_start_imputed`:
   target-encoded mean PitNextLap by `(Race, Year, Compound,
   stint_start_imputed)`. This is a per-row preimage feature that
   re-aggregates synth rows by their inferred orig-stint identity,
   not the fabricated synth Stint label. Per-fold refit per Rule 24.

2. **Phase 3** — Recursive CTGAN replay (no public CSV): train CTGAN
   on synth itself; sample 200k replay; build 2-class discriminator
   {synth, replay}. Disc output captures host-specific generator bias.

3. **Phase 4** — Density anomaly: fit a flow / KDE / GMM on synth
   continuous columns; flag low-density rows; class rate may differ
   in tails (CTGAN-extrapolated vs CTGAN-typical regions).

## Pointers

- `scripts/dgp_v2/p1_synth_only_fingerprint.py` — probe.
- `scripts/artifacts/p1_synth_fingerprint.json` — full numeric output.
- `audit/2026-05-08/2026-05-08-p1-synth-fingerprint.md` — this file.

## Friction tag

`synth-stint-label-is-fabricated-not-temporal` — key durable fact:
group-by `(Driver, Race, Year, Stint)` aggregates noise, not real
stints. Use `stint_start_imputed = LN − TL + 1` per row to recover
the orig-stint partition.
