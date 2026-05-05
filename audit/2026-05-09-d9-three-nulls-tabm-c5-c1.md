# 2026-05-09 — Day-9 closes with 3 NULLs: TabM, C5, C1

> Three independent probes, all NULL or FAIL today. PRIMARY unchanged at
> LB 0.95026 (gap −3.9bp). 0/10 submits used today. **Each null is a
> sharper falsification than yesterday's; this isn't saturation, it's
> stronger ceiling-confirmation.**

## Results table

| Probe | Class | Std OOF | ρ vs M5q test | min-meta Δ | K=N stack Δ | Verdict |
|---|---|---:|---:|---:|---:|---|
| **TabM-D smoke** (T4) | new model class | fold-0 0.94039 | n/a | n/a | n/a | **FAIL** (gate 0.945, −68bp vs RealMLP fold-0) |
| **C5 neighbor rules** | rule_residual + neighbor feat | 0.94475 / 0.94432 | 0.909 / 0.888 | +0.19/+0.37bp PASS | K=20 +0.04bp ρ=0.99985 | MARGINAL (TIE) |
| **C1 SC-prob rule** | rule_residual + external feat (cross-Race) | 0.94584 | 0.926 | +0.12bp PASS | K=19 −0.01bp ρ=0.99989 | DO_NOT_SLOT |

PRIMARY = `d6_k18_multi_rule` LB **0.95026**; K=18 OOF 0.95065. Gap −3.9bp.

## Mechanism analyses (load-bearing)

### TabM-D smoke FAIL
- pytabkit `TabM_D_Classifier` resolved cleanly (after v1 `use_ls` constructor
  fix, v2 ran). 11 cols continuous, 3 categorical auto-detected (Driver,
  Compound, Race). Same data prep as RealMLP.
- Training: 25 epochs at ~14s each; best val cross-entropy −0.4248 at epoch 5.
- Fold-0 AUC: **0.94039**. vs gate 0.945 = −46bp; vs RealMLP fold-0 ref
  0.94722 = **−68bp**; vs baseline_two_anchor 0.94075 = −3.6bp **(below
  baseline)**.
- 5-fold projection: 27 min — wall PASS but irrelevant.

**Likely root cause**: TabM-D's tuned-default config is calibrated for
small-to-mid tabular benchmarks; the combination of 350k rows + 887-unique
Driver column may need:
  - longer training (defaults early-stopped at epoch 5 with patience),
  - different cat-embedding sizing for high-cardinality Driver,
  - tuned LR / batch size for our dataset scale.

**Implication**: TabM is FALSIFIED with current pytabkit defaults. To
unfalsify, needed ~6h GPU of HPO sweep. EV downgraded substantially
because the existing RealMLP-TD already covers the MLP-with-embedding
mechanism class, and pytabkit's `TabM_HPO_Classifier` would mostly
re-learn the same architecture-class signal RealMLP captured on the
first try.

### C5 neighbor-rule extension MARGINAL
- prev_compound × Compound × stint_bucket: ρ=0.909 (most-diverse-since-
  RealMLP at the standalone level). std OOF 0.94475, min-meta +0.19bp.
- next_compound × Compound: ρ=**0.888** (NEW lowest). std OOF 0.94432,
  min-meta +0.37bp, K=20 L1=0.599 (5th rank).
- K=20 stack: OOF 0.95065 (= K=18 ±0.04bp), ρ=0.99985 (TIE).

**Mechanism finding**: orthogonality of feature is necessary BUT NOT
SUFFICIENT. The new signal must be missing from EVERY existing base
in the stack — not just M5q. Here, prev/next_compound is redundant with
d6 `rule_driver_compound` (drivers cluster strategy choices).

### C1 SC-probability NULL
- Per-Race SC-prob lookup (curated 2018-2024 from Lights Out Blog +
  Axiora + F1 domain priors, see SC_PROB dict in `scripts/d9_c1_sc_probability.py`).
- Bayesian-smoothed rule key `(sc_decile, stint_bucket, lap_quintile)` on
  Strat 5-fold; HGBC residual on raw features + sc_prob.
- std OOF 0.94584, ρ=0.926, min-meta +0.12bp PASS, K=19 OOF 0.95065
  (−0.01bp), ρ=0.99989, L1=**0.283** (low).

**Mechanism finding**: cross-Race generalization through SC-prob deciles
adds NOTHING the M5q pool's per-Race categorical hasn't already extracted.
The grouping promise (Singapore + Saudi share statistics through deciles)
turned out to be illusory: the residual GBDT routes around it because
each Race's per-row LapTime / RaceProgress / Stint already encodes the
within-Race signal. The continuous sc_prob feature added to X_enc carries
no extra information beyond Race-as-categorical for the residual GBDT.

## Codified update to Rule 16 (proposed)

Append a 6th question:

> **Q6: ρ-orthogonality with EVERY existing base in the stack, not just
> PRIMARY.** Today's C5 (ρ=0.888 vs M5q test → ρ=0.99985 vs K=18 stack)
> and C1 (ρ=0.926 vs M5q test → ρ=0.99989 vs K=18 stack) both passed Q4
> against PRIMARY but failed against the full stack. The K=18 LR-meta
> projects new bases onto the span of existing bases; only signal in
> the COMPLEMENT of that span survives.

This makes the 5-question check stricter and would have flagged C5/C1
both as predicted-NULL pre-flight.

## Updated calibration ladder

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | gap −3.9bp |
| d9_c5_prev_comp_stint (std) | 0.94475 | n/a | ρ=0.909 vs M5q test |
| d9_c5_next_comp_compound (std) | 0.94432 | n/a | ρ=**0.888** vs M5q test |
| d9_k20_neighbor (M5q+d6+c5) | 0.95065 | n/a | TIE; ρ=0.99985 vs K=18 (HELD) |
| d9_c1_sc_prob (std) | 0.94584 | n/a | ρ=0.926 vs M5q test |
| d9_k19_sc_prob (M5q+d6+c1) | 0.95065 | n/a | TIE; ρ=0.99989 vs K=18 (HELD) |
| TabM-D fold-0 smoke | 0.94039 | n/a | FAIL gate; held |

## What to do tomorrow (Day 10)

The d3-endgame ceiling has now been confirmed across **eight nulls** (Day-7
RealMLP bag B+C, Day-8 Deotte L2/L3 + Q12 + Poisson, Day-9 C5 + C1 + TabM).
Patterns:
- Adding bases that share inductive bias with existing pool → TIE.
- Adding meta complexity (LGBM, Ridge, blends) → TIE.
- Tuning a single GBDT-class component → TIE.

**Survivable mechanism classes (tomorrow's queue)**:
1. **T1.4 Hazard-rate NN** (different problem structure: K=20 hazard
   buckets with NLL loss over the full curve; not binary classification).
   Per HANDOVER GPU queue G3. Sole unfalsified Tier-1 after TabM HOLD.
2. **TabM HPO sweep** — would need a 6h GPU budget for proper tune.
   Down-weighted since RealMLP-TD is the direct comp arch and was
   cheaper.
3. **SCARF/VIME pretrain on aadigupta1601 unlabeled corpus** (G4) —
   different unlabeled dataset avoids the d5 pseudo-label amp. Unblocked
   by today's NULLs.
4. **F2 multi-rule REBUILD** with rules selected by orthogonality vs
   K=18 projected residual (not just vs M5q). Cheap CPU experiment that
   might yield a real K=N+1 gain by enforcing Q6 directly. (Not yet
   confirmed worth the slot, but cheap to probe.)
5. **External-data Q1 Pirelli pit-windows** (HANDOVER C2) — heaviest
   eng cost (6-8h scrape) but the highest-EV CPU candidate after C1's
   approach (curated lookup) was shown to NULL. Q1 carries different
   shape of external info (per-(Race,Year,Compound) windows, not per-
   Race scalar).

## Artifacts

- `scripts/d9_c1_sc_probability.py` — C1 builder (with curated SC_PROB)
- `scripts/d9_c5_neighbor_rules.py` — C5 builder
- `scripts/artifacts/d9_c1_sc_prob_results.json`
- `scripts/artifacts/d9_c5_neighbor_results.json`
- `scripts/artifacts/d9_tabm_smoke_results.json`
- `scripts/artifacts/d9_tabm_smoke_kernel.log` (full Kaggle log)
- `scripts/artifacts/oof_d9_c1_sc_prob_strat.npy`,
  `oof_d9_k19_sc_prob_strat.npy`,
  `oof_d9_c5_*_strat.npy`,
  `oof_d9_k20_neighbor_strat.npy`
- `submissions/submission_d9_k19_sc_prob.csv` — **HELD, do not submit**
- `submissions/submission_d9_k20_neighbor.csv` — **HELD, do not submit**
- `kernels/tabm-smoke-gpu/` — kernel v2 (committed)
