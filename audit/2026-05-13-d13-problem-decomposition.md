# 2026-05-13 — Day-13 problem decomposition (Conn-McLean re-entry)

> Trigger: PI directive "another branch is doing A and C1-C4; tackle
> from a different angle". Re-entered step 1 of the 7-step framework
> rather than continuing step 5. Branches A (TabPFN-2.5 fine-tune) and
> C1-C4 (4 new FM-input features) reserved for the other branch.

## Reframe (step 1 — Define)

Current frame (Days 1-12): every base in `mechanism_families_explored`
predicts `P(PitNextLap | this row's features)` independently. Pool has
20+ bases (GBDTs, FMs, sparse-LR, rules, MLPs); LR-meta combines via
`[raw, rank, logit]`. Selection on Strat-OOF, GroupKF as secondary
gate.

Day-12 finding (Option 1, load-bearing): rank-lock at ρ≈0.9999 is
*substantially a Strat-leakage artifact*. FM 23–37× more robust than
every GBDT under GroupKFold(Race,Driver,Year,Stint). FM survives by
*lacking the capacity* to memorise within-stint patterns — not by
*using* the structural information.

The structural blind spot: `PitNextLap` is the END-OF-STINT signal.
With train stint mean = 3.87 laps and class prior 0.199, each stint
has ≈0.78 positive rows → roughly 1 positive per stint at the last
lap (verified via probe Q1). **No base in the pool consumes the
"stints have an end" structure as input.**

L1 question unchanged: 31.1bp gap to top-5% (0.95345) by 2026-05-31
(14 days remaining, 8 sub slots today). Blocker reframed: pool
diversification has to come from a *different prediction-unit / loss
/ output normalisation*, not yet-another per-row-classifier base.

## Disaggregation (step 2 — modeling-axes view)

```
ROOT: predict P(PitNextLap) under AUC, leakage-bounded
α. PREDICTION UNIT  (currently per-row independent)
   α1. per-stint listwise (softmax over N laps; one-hot at end-of-stint)
   α2. per-stint pairwise (RankNet within stint; stint-grouped)
   α3. per-stint two-stage (stint length E[T] → soft assignment)
β. LOSS FUNCTION (currently BCE per row + AUC objective in some)
   β1. listwise CE over softmax (pairs with α1)
   β2. pairwise lambdarank (pairs with α2 — stint-grouped, not Race-grouped)
   β3. parametric Cox PH / Weibull-by-compound (NOT NN, different from dead T1.4)
γ. FEATURE VIEW (currently per-row absolute features)
   γ1. within-stint relative (feat - stint_mean) / stint_std
   γ2. within-stint rank (rank of TyreLife / Pos within stint)
   γ3. stint-aggregate (slope, variance, end-vs-start delta)
   γ4. cross-driver intra-race (probe Q5)
δ. INFERENCE OUTPUT (currently unconstrained per-row)
   δ1. stint-normalised probabilities (Σ_stint = 1 by construction)
```

MECE check passes. Every entry in `mechanism_families_explored` lives
on (per-row, BCE, absolute, unconstrained). The α/β/γ/δ axes are
essentially virgin.

## Prioritisation (step 3 — 2×2 impact × effort)

|              | Cheap (≤4h CPU)                 | Expensive (≥6h or GPU)   |
|--------------|---------------------------------|--------------------------|
| **High EV**  | data probe; G1 γ1+γ2+γ3 FE; G3 stint-grouped LambdaMART | G2 listwise softmax base |
| **Low EV**   | F1 hedge (already held)         | reg-FFM; meta tweaks     |

PRUNE: meta architecture (E1) — LambdaRank-meta dead, GBDT-meta dead,
LR-meta-on-`[raw,rank,logit]` is the local optimum.

## Workplan (step 4)

Day-13 sequence (this branch):
1. **Data probe** — `scripts/d13_data_probe.py`. 6 questions; ~30 min CPU.
2. **G1 γ-pack** — `scripts/d13_g1_within_stint_fe.py`. Closed-form
   features into a fresh FM and a fresh LGBM; OOF + 4-gate vs PRIMARY.
3. **G3 stint-grouped LambdaMART** — `scripts/d13_g3_lambdamart.py`.
   `objective=lambdarank, group=stint_id`; one config; gate.
4. **G2 listwise softmax** — `scripts/d13_g2_listwise_softmax.py`.
   Tiny 2-layer MLP; smoke 1-fold/50k before any 5-fold.

Stop conditions: each base must pass ρ-vs-PRIMARY < 0.999 AND
min-meta lift ≥ +0.05bp on Strat. If G1/G3 land, queue K=22/K=23
swap probe for tomorrow (single sub slot, calibration probe).

## Analyse (step 5 — heuristics first per Rule 6)

- G1: pure pandas groupby. No NN, no Optuna, no bagging.
- G3: single LightGBM call, default params except objective/group.
- G2: smoke gate before 5-fold; if fold-0 AUC < 0.93 single-fold, kill.
- Probe: ~30 min total wall, no model training.

## Probe results (`scripts/artifacts/d13_probe.json`)

Q1 — **PitNextLap NOT 1-per-stint.** 113,567 train stints: 63.1% have
0 pos, 18.2% have 1, 18.7% have 2+ (max 30). Only 35% of positives sit
at the last observed lap of their stint; only 22% of positives align
with the next observed PitStop=1. **Listwise-softmax (G2 original) is
structurally dead — drop.**

Q2 — every compound change → stint change (100%); 42% of stint changes
have NO compound change (sampling artifact). Stint range 1..8.

Q3 — **Year=2023 anomaly = synthetic-driver injection.** Y2023 has 887
unique drivers (vs ~547 in 2022/2024/2025) at pos rate **0.96%** (vs
~28%). This is the long-tail synthetic-driver cohort P9 flagged. A
hard-mask on `(Year=2023 ∩ low-count Driver)` is post-processing free.

Q4 — driver persistence weak: within-driver pit-TyreLife std 9.57 vs
across-driver 10.37 (8% reduction); within (Driver, Compound) 8.42
(19%). Driver embedding low-EV unless conditioned on Compound.

Q5 — **CROSS-DRIVER INTRA-RACE SIGNAL IS LARGE.** Per (Race, Year,
LapNumber) block (n=5,698 blocks, mean 77 drivers/block):
- `block_tyrelife_std`: row-corr with target = **+0.29**
- `block_hard_frac`: row-corr = **+0.25**
- `block_soft_frac`: row-corr = +0.01 (null)

For comparison TyreLife row-corr = +0.27, Position +0.02. **No base in
the pool consumes these. New highest-EV branch.**

Q6 — TyreLife corr +0.27 (strongest single); Cumulative_Degradation
−0.17; LapTime_Delta ≈0 (non-linear or noisy); Position ≈0;
Position_Change ≈0.05.

## Revised G plan

- **G1** (γ1+γ2+γ3) within-stint relative FE — keep; small stints cap
  EV but the FE is cheap and feeds existing FM/LGBM.
- ~~G2 listwise-softmax~~ — **dropped** (probe Q1).
- **G2′ cross-driver intra-race pack** (γ4) — NEW; built from Q5
  finding. Block-level features per (Race, Year, LapNumber) into a
  fresh LGBM and a fresh FM. **Highest-EV branch.**
- **G3** stint-grouped LambdaMART — keep.
- **G4 (queued)** Year=2023 hard mask post-processing — cheap, defer
  until G1/G2′/G3 land or fail.

## Pointers

- `scripts/d13_data_probe.py` — probe (Q1-Q6)
- `scripts/artifacts/d13_probe.json` — probe results
- `scripts/d13_g1_within_stint_fe.py` — G1
- `scripts/d13_g2_cross_driver_fe.py` — G2′ (replaces listwise-softmax)
- `scripts/d13_g3_lambdamart.py` — G3
- `audit/2026-05-12-d12-master-synthesis.md` — Day-12 unifying frame
- `audit/2026-05-08-data-probe-results.md` — P1-P10 priors
