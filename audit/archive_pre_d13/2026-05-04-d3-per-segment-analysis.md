# M5h per-segment OOF analysis — 2026-05-04

Triggered by Day-2 strategy critique (data-understanding gap #1):
"per-segment OOF AUC. A 38bp headroom could be 10 races at +5bp and
16 at −20bp; lift surface lives where the model is *bad*, and we
don't know where that is."

Aggregate Strat OOF: **0.95043** (M5h, current PRIMARY).
Reference table: `audit/2026-05-04-d3-per-segment-oof.md`.

## The model is wildly heterogeneous across segments

### Race — 600bp spread (worst → best)

| tier | races | pop. share | OOF AUC | Δ vs agg |
|---|---|---:|---:|---:|
| Worst 6 | Spanish, Emilia Romagna, Bahrain, French, Mexico City, São Paulo | 21.4% | 0.911–0.924 | −393 to −261bp |
| Mid 10 | Australian, Austrian, Hungarian, Japanese, Miami, Canadian, Qatar, Dutch, US, Azerbaijan | 41.7% | 0.936–0.948 | −149 to −20bp |
| Top 10 | Pre-Season, Italian, British, Belgian, Chinese, Singapore, Las Vegas, Monaco, Abu Dhabi, Saudi | 36.9% | 0.956–0.972 | +59 to +215bp |

**Spanish GP at 0.91117 (−393bp)** = single largest aggregate-AUC drag.
Lifting just the worst 6 races by +50bp each would deliver ~+10bp
aggregate. Lifting all to median would deliver ~+30bp aggregate.

### Year — 2023 is an outlier; the model is uniformly bad on the rest

| Year | n | pos_rate | OOF AUC | Δ vs agg |
|---|---:|---:|---:|---:|
| **2023** | 136k (31.0%) | **0.0096** | 0.94590 | −45bp |
| 2024 | 127k (28.9%) | 0.2953 | 0.92848 | −220bp |
| 2025 | 92k (21.2%) | 0.2844 | 0.92892 | −215bp |
| 2022 | 83k (18.9%) | 0.2665 | 0.91407 | −364bp |

**2023 has ~1% positive rate vs ~28-29% for other years** — completely
different label distribution. Most of M5h's aggregate AUC inflation
comes from being able to *rank* 2023 rows (low-prior class) above
the rest. Within-2023 AUC is 0.946, lower than aggregate. Within
the *active-pit-rate* years (2022, 2024, 2025): all ~0.91-0.93.

This means the **aggregate 0.95043 is misleading**: the model is
≈0.92 on the rows that actually matter (where pos_rate is non-trivial).

### Stint — the lift surface is Stint 2

| Stint | n | pos_rate | OOF AUC | Δ vs agg |
|---|---:|---:|---:|---:|
| 1 | 216k (49.2%) | 0.0598 | 0.93560 | −148bp |
| **2** | **130k (29.5%)** | **0.3911** | **0.91631** | **−341bp** |
| 3 | 69k (15.8%) | 0.2931 | 0.92806 | −224bp |
| 4 | 19k (4.3%) | 0.1717 | 0.93651 | −139bp |
| 5 | 4.3k (1.0%) | 0.0530 | 0.92634 | −241bp |
| 6+ | 0.9k (0.2%) | very mixed | n/a | n/a |

**Stint 2 is the strategic-decision zone** (post-first-pit, planning
second pit). 30% of data, AUC 0.916, the worst large segment. A
Stint=2-specific model could lift the in-segment AUC by even +50bp
→ ~+15bp aggregate.

### Compound — WET is a disaster but tiny; MEDIUM is the easy one

| Compound | n | pos_rate | OOF AUC | Δ vs agg |
|---|---:|---:|---:|---:|
| **WET** | 1.4k (0.3%) | 0.0251 | **0.84809** | **−1023bp** |
| SOFT | 39k (8.8%) | 0.1935 | 0.92947 | −210bp |
| HARD | 170k (38.8%) | 0.3275 | 0.93231 | −181bp |
| INTERMEDIATE | 17k (4.0%) | 0.1523 | 0.93593 | −145bp |
| MEDIUM | 211k (48.1%) | 0.1011 | 0.95300 | +26bp |

WET is a 1000bp drop but only 0.3% — negligible aggregate impact.
HARD (39%) at −181bp is the bigger drag.

### LapDecile (LapNumber bucket) — U-shape: early easy, mid-late hard

| Decile | n | OOF AUC | Δ vs agg |
|---:|---:|---:|---:|
| 0 (earliest) | 54k | 0.95595 | +55bp |
| 1 | 39k | 0.94534 | −51bp |
| 2 | 48k | 0.94184 | −86bp |
| 3 | 37k | 0.94196 | −85bp |
| 4 | 45k | 0.94128 | −92bp |
| 5 | 45k | 0.93928 | −112bp |
| 6 | 42k | 0.93403 | −164bp |
| **7** | **44k** | **0.92560** | **−248bp** |
| 8 | 46k | 0.93735 | −131bp |
| 9 (latest) | 39k | 0.92746 | −230bp |

Decile 7 (mid-late race) is worst — matches Stint 2 finding (Stint 2
correlates with mid-late laps).

### TyreDecile — fresh tyres are hardest

| Decile | n | OOF AUC | Δ vs agg |
|---:|---:|---:|---:|
| **0 (fresh)** | **52k** | **0.90379** | **−466bp** |
| 1 | 39k | 0.91173 | −387bp |
| 2 | 57k | 0.92504 | −254bp |
| 3 | 35k | 0.92940 | −210bp |
| 4 | 38k | 0.93651 | −139bp |
| 5 | 49k | 0.93822 | −122bp |
| 6 | 43k | 0.94471 | −57bp |
| 7 | 44k | 0.94185 | −86bp |
| 8 | 38k | 0.94834 | −21bp |
| 9 (oldest) | 44k | 0.95296 | +25bp |

**Monotone**: model is bad on fresh tyres (deciles 0-2, 34% of data,
≤−254bp) and good on aged tyres. Fresh-tyre rows correspond to the
laps right after a pitstop — also Stint-2-start territory.

## Synthesis

**Hard rows** cluster on:
- High-pos-rate races (Spanish, Emilia Romagna, Bahrain ≥ 27% pos)
- Active-pit years (2022, 2024, 2025; not 2023)
- Stint 2 (post-first-pit decision zone)
- Mid-late laps (decile 7 specifically)
- Fresh tyres (deciles 0-2, immediately post-pit)
- HARD compound

These overlap heavily — they all describe **the strategic-decision
moment in active-pit races**. The model is good at the trivial
"no-pit" zones (early laps, fresh stint, MEDIUM compound, 2023) and
bad at the "should-I-pit-now?" zones.

## Implications for slot-7 candidates beyond RealMLP

Three new high-EV levers surface:

### A. Per-Race calibration (~10 min, probably +3-5bp LB)

Post-hoc isotonic calibration per Race on M5h OOF. The bad races'
model probabilities are likely systematically miscalibrated. If we
fit isotonic per Race on OOF and apply at test time, we may close
some of the OOF→LB gap (currently −5.2bp on M5h). **Doesn't add
information; just reshapes probabilities.** Cheap, no new training.

### B. Stint-2 specialist model (~20-30 min CPU, potentially +10-20bp aggregate)

Stint 2 is 30% of data with worst large-segment AUC. Train a single
LGBM (or HGBC) using only Stint=2 train rows. At inference, use the
specialist for Stint=2 test rows, M5h for the rest. Strategy critique
called this out structurally; we now have evidence the segment matters.
Variant: train per-Stint specialists (Stint 1, 2, 3+).

### C. Active-year specialist (~20-30 min CPU)

Train on rows where Year ∈ {2022, 2024, 2025} (i.e. active-pit-rate
years; pos_rate ~28%). Drop 2023 from training. The motivation: 2023
is so easy it's diluting the gradient signal during training of the
existing pool. A model trained only on active years should be better
on the rows that matter. At inference, blend with M5h based on Year.

### D. (Lower priority) Year-conditional ensemble

Per-Year stacking — separate LR meta per Year. Fit 4 LR models on
the M5h pool, one per Year subset. Risky on 2023 (low pos_rate;
LR may struggle). Probably not the best lever.

## Recommended Day-3 sequencing (while RealMLP runs)

1. **Reliability + isotonic post-hoc on M5h** (#2 from earlier list,
   ~5 min). Even before A above, check whether GLOBAL miscalibration
   exists. If yes, apply globally first (free LB lift), then layer A
   on top.
2. **Per-Race isotonic** (lever A). ~10 min. Slot-7 candidate.
3. **Stint-2 specialist** (lever B). ~30 min. Slot-7-or-8 candidate.

If RealMLP delivers, stack it with whichever of {global isotonic,
per-Race isotonic, Stint-2 specialist} also delivers, and submit the
best as slot 7. If RealMLP fails, levers A+B alone make a 2-slot
program for slots 7-8.

## Pointers

- `audit/2026-05-04-d3-per-segment-oof.md` — raw segment tables.
- `scripts/diag_m5h_per_segment.py` — diagnostic script.
