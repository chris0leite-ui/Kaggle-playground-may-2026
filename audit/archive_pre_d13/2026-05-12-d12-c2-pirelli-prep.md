# Day-12 — C2 Pirelli pit-windows preparation [DEPRECATED]

> **DEPRECATED 2026-05-12** — synthetic-DGP incompatibility flagged
> by PI before execution. Three concrete reasons C2 is bounded near
> zero EV on this comp's data:
>
> 1. **Year=2023 mode-collapsed** in the CTGAN synth (24/26 races
>    <5% pit rate vs ~28% real-world). Pirelli's real-world windows
>    for 2023 races would predict normal pit-rates inside the window
>    but the synth shows almost no pits there at all → ~31% of train
>    rows yield flat global-mean fallback.
> 2. **CTGAN broke 4+-way joints** (P10). A Pirelli window's signal
>    lives in the 5-way (Race, Year, Compound, lap, what-actually-
>    happened) co-occurrence — exactly the structure CTGAN doesn't
>    preserve.
> 3. **Pool already absorbs the synth's empirical analog** through
>    `Compound × TyreLife` and `Race × Compound × Stint` rules.
>    Real-world priors that go beyond the synth's own joint structure
>    are noise.
>
> **Direct prior evidence**: C1 SC-prob (Day-9 K=19 TIE ρ=0.9999) is
> the same failure mode. External real-world signal does not transfer
> to synthetic data.
>
> **Falsified by Q5 of Rule 16 retroactively**: closest gate-PASS
> precedent at predicted ρ ≈ 0.998-0.999 from external-data lookup is
> C1, which TIE-locked. EV midpoint downgraded ×0.1.
>
> **Pivoted to**: 3-way multi-FM partition (cheap probe extending
> d9f's 2-way win) + G4 SCARF on aadigupta1601 (different unlabeled
> corpus → different inductive bias, robust to synth artifacts).
>
> Skeletons preserved for reference / repurposing if a future comp
> uses real F1 telemetry. Below is the original (now-stale) plan.

---

> Top-priority CPU move per Day-11 strategy critique + Day-12 HANDOVER
> Path A. Multi-FM partitions (d9c, d9f) added +5bp LB territory in
> 48h by introducing a new model class (low-rank pairwise FM). C2 is
> the parallel move on the **information** axis: bringing genuinely
> external pit-strategy signal that the 14 raw features cannot
> express. PI directive Day-12: "prepare 3" = prep without executing.

## 1. What is C2 Pirelli?

Before each F1 race, Pirelli (the sole tire supplier) publishes
**pre-race pit-window predictions** in their official press kits and
race-preview articles. F1.com strategy guides republish these in
plain-English form (e.g. "Optimal 1-stop on MEDIUM→HARD around lap
22-28; alternative 2-stop with SOFT/HARD windows at lap 14-18 +
lap 38-42").

These predictions encode three things our pool cannot derive from
the 14 raw features:

1. **Tire-degradation physics priors** — Pirelli simulates wear
   curves from pre-race testing, weather, track temperature, and
   compound nominations. The pool sees only LapTime, TyreLife,
   Position post-hoc; it can fit deg curves *empirically* but cannot
   know the *prescriptive* window without the simulation.
2. **Race-strategist consensus** — every team targets these windows
   when feasible, so pit decisions cluster around them. This is the
   pit-decision *prior*; the pool currently sees only the
   *posterior* (TyreLife at the time of the actual stop).
3. **Per-(Race, Year, Compound) regulatory context** — windows shift
   year-to-year as Pirelli adjusts compounds + as track surfaces
   age. A 3-way (Race, Year, Compound) interaction with EXTERNAL
   information; CTGAN preserves up to 3-way joints (P10) so the
   signal is recoverable in the synthetic test rows.

**Why this is structurally different from prior moves:**
- C1 SC-prob (Day-9 TIE) was a per-Race scalar — redundant with
  Race-as-categorical in the pool.
- F1.2 multi-rule (Day-6 +2.1bp PASS) was per-(Compound, TyreLife)
  *empirical* lookups — got 2/3 of the way there but missed the
  prescriptive window prior.
- C2 brings per-(Race, Year, Compound) PRESCRIPTIVE windows that
  are not in any feature, derivable or learned.

## 2. Data sources (24 races × 4 years = 96 race-events)

Primary sources, in order of access reliability:

1. **F1.com pre-race strategy guides** — `formula1.com/en/latest/article/`
   search "[race name] strategy guide [year]"; consistent format
   2022-2025. Plain-English window descriptions parseable with
   regex + a tire-name lookup. ~5 min per article.
2. **Pirelli Motorsport press kits** — `press.pirelli.com/motorsport/`
   per-race PDF kits include "Pirelli Strategy Predictions" tables.
   PDFs require text extraction (pdfplumber) but are highly
   structured. Pre-2024 archives may be incomplete.
3. **The Race / Autosport pre-race** — supplementary; cross-check
   only when sources 1 and 2 conflict.
4. **Sky Sports F1 race-preview transcripts** — fallback for races
   where F1.com strategy guide is missing (rare; ~2-3 races/year).

Year coverage:
- 2022, 2023, 2024: F1.com + Pirelli reliable
- 2025: F1.com reliable; Pirelli may lag for late-season races

## 3. Schema

`pirelli_windows.csv` with columns:
```
race, year, compound, n_stops_strategy,
window_start_lap, window_end_lap, window_center_lap,
window_width, source_url, scrape_confidence
```

Where `n_stops_strategy ∈ {1, 2, 3+}` and a single race-year may
have multiple rows (one per recommended strategy variant). Confidence
∈ {high, med, low} based on source agreement.

## 4. Bases to build (4-6, F1.2 template)

Per `scripts/d6_multi_rule.py` Bayesian-smoothed lookup → HGBC
residual on raw features + lookup feature. Strat-only (R1).

| # | Base | Lookup key | Rationale |
|---:|---|---|---|
| 1 | `in_window` | (Race, Year, Compound, lap_in_window) | binary: is current lap inside ANY recommended window? |
| 2 | `dist_to_window_center` | (Race, Year, Compound, signed_dist_decile) | how far from optimal stop, signed |
| 3 | `dist_to_window_edge` | (Race, Year, Compound, abs_dist_outside_decile) | distance OUTSIDE nearest window (0 if inside) |
| 4 | `stops_to_go` | (Race, Year, Driver, n_stops_remaining) | predicted total stops − stops_so_far |
| 5 | `window_progress` | (Race, Year, Compound, current_lap / window_center_decile) | fractional progress through optimal window |
| 6 | `multi_strategy_flex` | (Race, Year, n_window_options) | races with multiple viable strategies → pit decision noisier |

Each base goes through Q6 filter: ρ vs new PRIMARY (d9f K=21,
LB 0.95031) must be ≤ 0.997 standalone. K=N+m stack rebuild after
filtering; gate ρ vs PRIMARY ≤ 0.999.

## 5. Estimated cost & timeline

| Phase | Cost | Owner |
|---|---|---|
| Scrape 96 race-events (script + manual cleanup) | 6-8h CPU + manual review | Day-12 PM → Day-13 AM |
| Schema validation + missing-data audit | 1h | Day-13 |
| Build 6 rule_residual bases | 2h CPU | Day-13 |
| Q6 filter + K=N stack rebuild + min-meta gate | 30 min | Day-13 |
| Submit (single-shot per Rule 1) | — | Day-13 PM |

## 6. Risks

- **Some race-years lack public Pirelli predictions** (esp. pre-2024
  Pirelli archive gaps). Handle with `confidence='low'` flag and
  median-imputation; flag rows where ALL 6 bases fall back to global.
- **Window definitions drift between sources** — F1.com may give
  "lap 22-28", Pirelli "lap 20-30". Use F1.com when both available
  (it's downstream of Pirelli); record source disagreement for audit.
- **Late 2025 races may be missing scraping targets** entirely if
  Pirelli/F1.com haven't published. Fall back to year-1 transfer
  (use 2024 windows for 2025 if no 2025 data).
- **Scrape parser fragility** — F1.com strategy article HTML changed
  in 2024; need version-aware parser. Estimate +1-2h on parser edge
  cases.

## 7. Skeletons (DO NOT EXECUTE without PI sign-off)

- `scripts/d12_c2_pirelli_scrape.py` — scaffold for the scrape
  pipeline; manual review checkpoint after each year.
- `scripts/d12_c2_pirelli_build.py` — base-build per F1.2 template
  with Q6 filter wired in.

Both are skeletons only; execution requires explicit PI go for the
6-8h scrape window.

## 8. Why this is worth the time investment

Per Day-11 strategy critique §3 (FM-revised):
- C2 median EV: **+3-8bp**
- Tail-case: **+12bp**
- Sum-of-medians remaining moves ≈ 10bp; 50% transfer ≈ 5-7bp
- Top-5% (0.95345) requires +31.4bp from current PRIMARY

C2 alone won't reach top-5% but is the **single highest-EV move
remaining on the menu**, and the only one that adds genuinely new
information rather than re-projecting existing pool features.

PRIMARY: `d9f_K21_swap_partA_partB` LB 0.95031, gap −2.4bp.
17/270 submits used.
