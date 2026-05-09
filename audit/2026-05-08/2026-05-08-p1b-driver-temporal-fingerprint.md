# 2026-05-08 — P1b: Driver × Year temporal fingerprint

`branch: claude/find-dgp-research-ClsQE`
`tag: dgp-driver-vocabulary-leakage`

> Follow-up to P1. The synth has 131 3-letter abbreviation driver codes
> + 756 D-prefix ghosts = 887 total. P1 found that the (Driver, Race,
> Year, Stint) tuple is fabricated. P1b probes the abbrev codes
> specifically, finding a clear separation between "historical" codes
> (uniform-across-years) and "active" codes (year-dependent matching
> real careers).

## Headline finding

**The host's CTGAN faithfully reproduces real-world Driver × Year
career timelines for active drivers, but blindly fabricates rows for
long-retired drivers in modern years.** This is a strong fingerprint
of the orig dataset's Driver vocabulary: aadigupta1601 (or its
upstream source) contains a SUPERSET of historical 3-letter driver
codes, and CTGAN samples the joint without filtering by driver
activity.

## Examples (per-driver row count by year)

**Real active drivers** (year-dependent, matches careers):

| Driver | 2022 | 2023 | 2024 | 2025 | Notes |
|---|---:|---:|---:|---:|---|
| ZHO | 467 | 456 | 550 | 127 | Real: Zhou dropped in 2025 |
| ANT | 16 | 38 | 77 | 490 | Real: Antonelli debut 2025 |
| BEA | 12 | 41 | 216 | 510 | Real: Bearman debut 2024-2025 |
| PIA | 63 | 453 | 611 | 591 | Real: Piastri debut 2023 |
| MSC (Mick) | 451 | 12 | 13 | 14 | Real: Mick raced 2021-22 only |

**Long-retired drivers** (uniform across years — fabricated):

| Driver | 2022 | 2023 | 2024 | 2025 | Real activity |
|---|---:|---:|---:|---:|---|
| BAR | 521 | 602 | 624 | 577 | Barrichello, retired 2011 |
| MAS | 520 | 612 | 686 | 607 | Massa, retired 2017 |
| BUT | 510 | 581 | 682 | 612 | Button, retired 2017 |
| WEB | — | — | — | — | Webber, retired 2013 |

(BAR, MAS, BUT each have ~600 rows/year for 2022-2025 — they shouldn't
exist in any of these years.)

## Per-driver year-CV statistics

```
Per-driver CV (year): median=0.119, p25=0.096, p75=0.149
```

CV ≈ 0.12 means most drivers' year counts vary by only ~12% across
2022-2025. For genuinely active drivers (HAM, VER, ALO), this matches
real-world steady participation. For drivers like ANT/BEA (rookies),
year-CV is high. For retired drivers (BAR, MAS, BUT), year-CV is low
— they've been "uniformly fabricated" across all years.

## Per-year row totals by driver group

| Year | D### rows | abbrev rows |
|---|---:|---:|
| 2022 | 62,126 | 56,211 |
| 2023 | 133,707 | 60,600 |
| 2024 | 108,671 | 72,971 |
| 2025 | 68,773 | 64,246 |

D-prefix totals vary 2× across years (62k → 134k); abbrev varies 1.3×.
The D-prefix variance is roughly proportional to orig's per-year row
count (more 2023 races → more synth rows → more D-codes).

## Implications

1. **The orig (aadigupta1601 or upstream) has a Driver vocabulary that
   includes historical codes used uniformly across all years.**
   This is either intentional (data augmentation in orig) or a scrape
   artifact (e.g., the source listed driver codes from a master table
   without filtering by year).

2. **CTGAN's Driver-categorical sampling is conditional on (Race,
   Year), and it correctly marginalizes**: active-driver codes get
   active-year masses, retired-driver codes get whatever orig had
   (uniform).

3. **For prediction**: Driver as a raw categorical encodes both real
   (career-timeline) and fabricated (uniform) signal. The y-rates for
   "fabricated" rows (BAR-2024) may differ from "real" rows (HAM-2024)
   if the CTGAN inherited different y-distributions from each source.

4. **A driver-activity feature is potentially useful**: indicator of
   "is this row's driver actually active in this year?" plus
   `driver_year_count` and `driver_year_count_zscore`. Such features
   are derivable from synth alone and are fold-safe by construction
   (Rule 24 — they don't use the label).

## Pointers

- This audit and P1.
- `scripts/dgp_v2/p1_synth_only_fingerprint.py` — probe.
- `scripts/artifacts/p1_synth_fingerprint.json` — raw output.

## Friction tag

`driver-vocab-mixes-active-and-historical` — the orig dataset's Driver
column contains a superset of historical 3-letter codes. CTGAN samples
this faithfully. Use `driver_year_count` to distinguish active from
fabricated rows.
