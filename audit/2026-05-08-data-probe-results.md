# 2026-05-08 — Data probe results (P1..P10)

Structural probes on raw `data/{train,test}.csv` + `oof_m5q_strat.npy`.
Goal: empirically verify/falsify claims that would unlock new modeling
angles. Runtime: 31 s total CPU. Probe code in
`scripts/probes_d8/run_probes.py` + `p5_extended.py`. Numbers in
`scripts/probes_d8/out.json` + `p5_extended.json`.

## TL;DR

- **P1 — Sequence model:** TEST groups are SHORT FRAGMENTS, not full
  sequences. Mean group len 2.25, only 9.7% have ≥5 consecutive laps.
  46% of test groups have *contiguous* laps (rest have gaps). LSTM /
  attention can be trained on TRAIN (mean 3.9 laps/group, p95=11) but
  inference on test will be on tiny windows. **Sequence model is
  feasible but ROI is bounded by short test windows.**
- **P2 — kNN retrieval:** Mean nn-dist 0.73, p50=0.68, only 1.6% of
  test rows have a train neighbor at <0.1. **Synthetic-DGP-NN pattern
  FALSIFIED.** kNN unlikely to help.
- **P3 — Year×Race anomaly CONFIRMED.** 2023 pit rate 0.96% (vs
  ~28% other years). 24 of 26 (Year=2023, Race) cohorts <5% pit rate.
  **Train/test 2023 share IDENTICAL (31.0% vs 30.9%).** No
  downsampling — structural shift in the 2023 generative process.
- **P4 — Stint-2 mineable signal.** Stint-2 pit rate 39%; cleanly
  monotonic in laps-into-stint (22% at lap-0 → 51% by lap-30+).
  prev_compound × current_compound spreads pit rate from 18.9%
  (SOFT→HARD) to 75.4% (WET→HARD). **Stint-2 specialist via
  prev_compound + laps_into_stint is high-EV.**
- **P5 — CLAUDE's 97.4% successor claim FALSIFIED.** Only 12.4% of
  test rows have an in-test next-lap; 41.7% have a successor anywhere
  (train+test combined). lead_PitStop observable for 12.4%.
  next_compound observable for 68.2% (laps_until_eos for 46.6%).
- **P6 — Strat leakage MASSIVE.** 80.1% of consecutive-lap pairs
  (same Race×Driver×Year×Stint) land in DIFFERENT folds in our
  current StratifiedKFold(5). Within-group leakage is the rule, not
  the exception, for our OOF score.
- **P7 — Compound transitions.** Top sequence MEDIUM→HARD→HARD
  (7,631 races). MEDIUM→HARD covers 58% of all Stint-2 transitions.
  Conditional pit rate spans 18.9% (SOFT→HARD) to 75.4% (WET→HARD)
  — large feature signal.
- **P8 — Race lengths SANE.** Monaco median 77 (truth 78), Italian
  64 (Monza ~53 — slightly off but consistent intra-Year).
  RaceLength_Estimate per Race is a usable feature.
- **P9 — Driver dist.** 887 train, 801 test, **0 test-only**. 221
  drivers <10 rows (noise class). Top-50 only 17.3% of rows — long
  tail. low-count pit_mean 0.26% (noise) vs high-count 21.4% (real
  signal).
- **P10 — Anti-corr search MOSTLY NULL.** Only 3 (Race, Stint)
  cohorts with |residual|≥0.02 (all ≤0.03). 0 (Compound, TyreLife-dec)
  and 0 (Year, Position) cohorts. **Pool calibration is tight; no
  obvious orthogonal feature combo missed.**

---

## P1 — Sequence reconstructability

```python
g = df.groupby(["Race","Driver","Year","Stint"]); sizes = g.size()
contig = (g.LapNumber.max() - g.LapNumber.min() + 1 == sizes).mean()
```

| | n_groups | mean | p95 | max | contig | ≥5 | ≥10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 113,567 | 3.87 | 11 | 35 | 29.4% | 31.2% | 7.1% |
| TEST  |  83,812 | 2.25 | 6  | 15 | 46.3% |  9.7% | 0.30% |

72,859/83,812 test stints (86.9%) overlap a train stint with same
(Race, Driver, Year, Stint) — holdout is *within* stints, not
whole-stint. ~70% of train groups have NON-contiguous LapNumbers.

**Interp.** Sequence model trainable but test-side inference is on
2-lap windows on average; big LSTM gain unlikely. 1-step lookup
features (lag/lead pit, prev_compound) are the productive shape.
**Confidence:** high.

---

## P2 — Train/test row-level proximity

40k train × 20k test, 14 std features (11 num + 3 ordinal), L2.
mean nn-dist 0.729; p5 0.204; p50 0.675; p95 1.43. frac<0.1 1.6%,
frac<0.5 30.5%, frac<1.0 78.1%.

**Interp.** Typical 14-D dist; no synthetic-DGP-NN pattern. kNN /
retrieval unlikely to add fresh signal. **Confidence:** high.

---

## P3 — Year × Race anomaly verification

| Year | pit rate | rows | train share | test share |
|---|---:|---:|---:|---:|
| 2022 | 26.65% | 82,989  | 18.90% | 18.79% |
| **2023** | **0.96%** | 136,147 | 31.00% | 30.91% |
| 2024 | 29.53% | 127,110 | 28.95% | 28.98% |
| 2025 | 28.44% |  92,894 | 21.15% | 21.32% |

24/26 races in 2023 have pit rate <5%. Lowest: British (0.46%),
Singapore (0.47%), Pre-Season (0.51%). Highest valid: Qatar 2.31%.
Train/test 2023 shares match within 0.1pp → no test-side
downsampling. **Structural shift in the 2023 DGP.**

**Interp.** Year=2023 is near-zero base rate, almost-deterministic.
Don't treat Year as ordinal numeric only — Year×Race interaction or
2023 mask earns its slot. **Confidence:** high.

---

## P4 — Stint-2 deep dive

Stint-2: 129,536 rows (29.5%). Pit rate 39.1%.
- **Race×Compound spread (n≥100):** Monaco MEDIUM 76.5% → Pre-
  Season SOFT 0.85% (60-pp range).
- **laps_into_stint monotonic:** 22.4%(lap0) → 32.2%(1-2) →
  35.6%(3-5) → 41.1%(6-10) → 49.3%(16-20) → 51.5%(≥30).
- **prev_compound × s2_compound (n≥200):** WET→HARD 75.4%,
  INTER→MED 56.8%, SOFT→MED 47.2%, HARD→HARD 41.6%, MED→HARD
  39.2% (76,466, modal), SOFT→HARD 18.9%, MED→SOFT 19.0%.

**Interp.** Three high-leverage features for a Stint-2 specialist:
`prev_compound` (28-pp spread), `laps_into_stint` (29-pp), `Race ×
Compound` (60-pp). None first-class in M5q pool today.
**Confidence:** high.

---

## P5 — Test-only feature computability

**CLAUDE's "97.4% same-(Race,Driver) successor in test" claim
FALSIFIED at every reasonable interpretation:**

| Successor variant | frac test |
|---|---:|
| (Race,Driver,Year,Lap+1) in test | **12.38%** |
| same key in train | 29.34% |
| anywhere (train∪test) | 41.72% |
| (Race,Driver,Lap+1) ignoring Year, in test | 34.04% |
| predecessor (Lap−1) in train | 29.23% |
| lead_PitStop computable in test | 12.38% |
| **next_compound (next-stint compound)** | **68.15%** |
| laps_until_end_of_stint known | 46.59% |
| next-next lap in test | 1.94% |

**Interp.** lead-style features NOT broadly observable. Productive
features: `next_compound` (68%, high-EV with P4 Stint-2 spread),
`laps_until_end_of_stint` (47%, censored), `pred_in_train` (29%,
lag-OK). Depth-2 rolling NOT viable. **Confidence:** high.

---

## P6 — Lap-grouping leakage check

GroupKFold(Race, Driver) k=5: every fold has 87,828 rows (balanced).
n_unique (Race, Driver) = 14,942 — plenty for 5-fold groups.

**Within-group leakage of StratifiedKFold(5):** of 91,087
consecutive-lap pairs in same (Race,Driver,Year,Stint), **72,948
(80.1%) land in DIFFERENT folds.** Adjacent laps share LapTime,
TyreLife, Position state — direct leakage of held-out targets into
training neighbours.

**Interp.** Strat OOF is consistently optimistic vs true held-out —
explains OOF→LB gap structure (M5q −5.2bp, K=18 −3.9bp). R1 keeps
Strat as LB proxy, but GroupKFold by (Race,Driver,Year,Stint) is
the rigorous diagnostic. Could explain why minimal-meta gates are
tight (rule_residual passed because it doesn't use leaked neighbour
info). **Confidence:** high.

---

## P7 — Compound transition matrix

Stint-count dist per (Race,Driver,Year): 1:6,094  2:8,605
**3:16,710** (modal)  4:7,502  5:1,685  6+:273.

Top sequences: MED→HARD→HARD 7,631; MED→HARD 4,829; MED-only 3,052;
HARD-only 2,298; MED→HARD→MED 2,285; HARD→HARD 1,789;
HARD→HARD→HARD 1,188; MED→HARD→HARD→HARD 1,095; MED→HARD→SOFT
1,003.

Stint-2 conditional pit rate (P4): WET→HARD 75.4% vs SOFT→HARD
18.9% (56pp spread). MED→HARD modal (76,466) at 39.2% = population
mean.

**Interp.** s1→s2 mostly captured by current Compound, but
`prev_compound` adds 6-15pp signal in WET/INTER branches. Seq-FE
`stint_compound_sequence_so_far` (top-30 levels) plausibly earns a
slot. **Confidence:** high.

---

## P8 — LapNumber distribution per Race

Race-length medians (max LapNumber across Years): Monaco 77 (truth
78), Dutch 75, Mexico 72.5, São Paulo 71.5, Canadian 71, Italian
64.5 (Monza ~53 high), Belgian 56.5, Las Vegas 56. Within-Race std
mostly 2-4 laps; outliers Italian 5.3, Emilia 7.0.

**Interp.** `RaceLength_Estimate = median(max LapNumber)` is a
stable per-Race feature; pairs with normalised `RaceProgress` and
adds the absolute lap count (not in features today). **Confidence:**
med (small effect likely).

---

## P9 — Driver embedding plausibility

887 train drivers, 801 test, **0 test-only (cold-start)**. <10 rows:
221 drivers; <100: 402; <1000: 655. Top-50 row share 17.3%.
low-count pit_mean 0.26% (sd 1.6%) vs high-count 21.4% (sd 4.0%).

**Interp.** No cold-start fallback needed. Long tail (25% drivers
<10 rows) pit_mean 0.26% ≈ 2023 base rate → likely 2023-only
synthetic drivers (consistent with P3). Add `driver_is_low_count`
flag and/or shrinkage-TE prior tied to base rate. Top-50 only 17%
of rows → Driver-embedding (16-32 dim) well-conditioned.
**Confidence:** high.

---

## P10 — Anti-correlated cohort search on M5q residuals

residual = y_true − M5q_OOF[:,1]. Global mean −5.7e-5, std 0.264.
- (Race, Stint), n≥200, |bias|≥0.02: **3 cohorts**, all ≤0.03. Top:
  Abu Dhabi Stint-4 (−0.029), Emilia Stint-4 (−0.024), Qatar Stint-5
  (−0.021).
- (Compound, TyreLife-dec), n≥200, |bias|≥0.02: **0**.
- (Year, Position), n≥200, |bias|≥0.02: **0**.

**Interp.** Pool calibration is tight. No mass-cohort with ≥5pp
residual bias. The 3 Stint-4/5 outliers sit at noise floor
(~std/√count ≈ 0.018). **No obvious orthogonal feature combo the
K=18 pool is missing.** Future lift must come from new model
classes (RealMLP-bag, TabM, sequence), not tabular features.
Reinforces the Day-7 rank-lock thesis. **Confidence:** high.

---

## Action shortlist (what these probes unlock)

1. **prev_compound × laps_into_stint Stint-2 specialist** (P4, P7).
   New rule_residual base. Cheap CPU. EV +0.5-1.5bp K=19 stack.
2. **next_compound feature** (P5). Computable for 68% of test rows;
   train-side analog easy. High-EV for both M5q and rules.
3. **Year-2023 hard mask / interaction** (P3). Confirm `Year × Race`
   is a base-pool feature (it should be — verify in feature list).
4. **GroupKFold (Race, Driver, Year, Stint) diagnostic OOF** (P6).
   Only as a *diagnostic* — Strat remains LB proxy per R1 — but
   gives a leakage-free score to triangulate gap sources.
5. **DROP plans for kNN-retrieval and big-LSTM** (P1, P2). ROI
   bounded; the test windows are too short and the train/test NN
   distances are too large.
6. **Pool-residual search is exhausted (P10).** New base-classes
   only — focus the slot on RealMLP-bag (active) and TabM
   (Move D).
