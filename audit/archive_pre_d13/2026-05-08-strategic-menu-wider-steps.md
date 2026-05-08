# 2026-05-08 — Strategic menu: wider-step bets to climb the LB

> PI directive: "Look carefully at the problem, disaggregate it, do
> research, propose options that give new perspectives. We want to
> climb a lot — not small optimizations, wider steps. Many will fail
> but we have time. Find angles."
>
> Source: 3 parallel research streams (a) tabular-SOTA white-space,
> (b) F1 domain + synthetic-DGP, (c) local data probes P1–P10
> (`audit/2026-05-08-data-probe-results.md`). All claims are
> calibrated against probe results.

---

## 1. Problem, disaggregated (4 axes)

### Axis A — what we predict
Binary AUC on `PitNextLap`. Prior 0.199. 188k test rows. **Single
prediction per (Driver, Race, Year, Lap).**

### Axis B — the data structure (CTGAN-style synthetic)
- 14 raw features; categorical Driver(887), Compound(5), Race(26).
- **Train/test is i.i.d. row-level shuffle within (Race, Driver,
  Year, Stint) groups** — 86.9% of test stints overlap a train
  stint with same key. Holdout is *within-stint*, not whole-stint.
- **Pairwise and 3-way interactions preserved** (Race×Compound,
  Compound×TyreLife, Driver×Race) — that's why F1.2 multi-rule
  worked at +2.1bp / 2.7× OOF→LB amplification.
- **4+-way joints broken**, including time-series within-group.
- **Year=2023 is a different DGP regime** — 0.96% pit rate vs ~28%
  elsewhere; train/test 2023 share matches → not test-side
  downsampling, structural mode collapse in the host's generator.

### Axis C — the current model family
- 14 GBDT bases (LGBM/XGB/HGBC/CatBoost variants) + 1 RealMLP +
  4 rule_residuals; LR-meta on `[raw, rank, logit]` features.
- **Pool diversity is illusory**: 3 sources of true diversity
  (a_horizon, b_lapsuntilpit, cb_slow-wide-bag) + ~10 GBDT
  consensus clones + RealMLP (ρ=0.972) + 4 rules.
- **LR-meta is rank-locked**: 4 separate confirmations
  (M5h2/M5j/d4-yetirank/d5-recursive) all collapsed to ρ≥0.999.
- Strat OOF 0.95065 → LB 0.95026 (gap −3.9bp).

### Axis D — where it fails (segments)
Per-segment OOF on M5h (Day-3 audit) + verified by P4/P10:
| segment | share | OOF AUC | Δ aggregate |
|---|---:|---:|---:|
| Stint 2 | 29.5% | 0.9163 | **−341bp** |
| Year=2023 | 31.0% | 0.9459 | −45bp (but 0.92 within active years) |
| Fresh tyres (TyreLife dec 0-2) | 34.0% | 0.9038-0.9250 | up to −466bp |
| Mid-late laps (LapNumber dec 7) | 10.0% | 0.9256 | −248bp |
| HARD compound | 38.8% | 0.9323 | −181bp |

**P10 (anti-corr probe) finds NO residual cohort with |bias|≥0.02.**
The pool is well-calibrated *within its hypothesis class*. Lift
must come from outside the class — different inductive bias, new
features that encode signals the pool doesn't see at all, or
problem reformulation.

---

## 2. Binding constraints (load-bearing for menu ranking)

### C1 — Mechanism-class-only changes lift LB
3× rank-lock at LR-meta layer. Pool tweaks via LR-meta are dead.
New slots must:
- change L1 problem formulation (hazard, multi-target, reformulation)
- add structurally orthogonal model class (TabM ICL-style, sequence
  model, EmbMLP), OR
- add HIGH-CONSTRAINT features that GBDTs can't naturally find
  (regulatory rules, physics-aware soft features).

### C2 — Sequence models are bounded by short test windows
P1 falsifies the "97.4% successor" claim. Test groups average 2.25
laps; only 9.7% have ≥5 consecutive laps. **Big LSTM/Transformer
on (Race, Driver) sequences is bounded** — train-side fine, but
test inference is on tiny windows.

What survives: **1-step lookup features** (`next_compound`,
`prev_compound`, `lead_PitStop`-style for the 12.4% where it's
computable, `laps_into_stint`).

### C3 — kNN / retrieval is dead
P2 — train/test mean nn-distance 0.73; only 1.6% of test rows have
a train neighbor at <0.1. No synthetic-DGP-NN pattern. **TabR,
Hopular, retrieval-augmented inference all bounded EV.**

### C4 — StratifiedKFold OOF has 80% within-group leakage
P6 — 80.1% of consecutive-lap pairs land in different folds; OOF
optimistic by ~5bp (matches gap structure). **Use Strat as LB proxy
(R1) but add GroupKFold(Race, Driver, Year, Stint) as diagnostic.**
Rules+gates that survive both are robust.

### C5 — Tabular-feature lift is bounded
P10 — no residual cohort with ≥2pp bias. The pool already extracts
what's extractable from the 14 raw features. F1 domain features add
value only if they encode HARD CONSTRAINTS (regulatory rules) or
EXTERNAL signals (track-specific priors), not subtle calibration
shifts.

### C6 — Compute is NOT the binding constraint
Kaggle GPU access (T4×2) plus 8-core local CPU plus 20 days × 10
slots = effectively unlimited compute. **The constraint is finding
mechanism-class changes, not running them.**

---

## 3. White-space menu (ranked)

Notation: **EV** = honest LB delta range; **Risk** = probability of
0bp or regression; **Stage** = (NN/CPU/MIX). Ranking by
`(orthogonality × EV) / cost`. Emphasis: **wider steps**, many will
fail, but we have time and slots.

### Tier 1 — highest expected value, mechanism-class changes

#### T1.1 — TabM (ICLR 2025) as new NN base
- **What**: Parameter-efficient ensemble (BatchEnsemble adapters
  → K=32 internal heads from one forward pass). Beats RealMLP on
  Yandex/Grinsztajn benchmark.
- **Why this works here**: Different inductive prior than RealMLP-
  TD; the K=32 internal heads give FREE per-row std-dev as a meta
  feature (Deotte L2 trick — see T1.5). Doesn't depend on long
  sequences (which P1 killed).
- **Cost**: T4×2, 6-10h GPU for 5-fold × 3 seeds.
- **EV**: +2-8bp LB. Risk MEDIUM (could collapse to ρ=0.99 like
  RealMLP did against M5h).
- **Pointer**: `pip install tabm`; arXiv 2410.24210. Use the
  K=32-head std as a meta-feature regardless of whether it earns
  a slot.

#### T1.2 — Multi-formulation L1 (Deotte's 4-reformulation pattern)
- **What**: Train 4 distinct problem formulations as L1 bases:
  (a) direct binary P(pit_next_lap) — current; (b) Poisson regression
  on `cum_pit_count_remaining_in_race`; (c) censored regression on
  `laps_until_next_pit`; (d) ratio target `PitStop_total / Stint`.
  Each contributes 3 base variants (LGBM/CB/RealMLP).
- **Why this works here**: Each formulation produces a base whose
  prediction is NOT a monotone transform of the binary classifier's
  level set. **F3/F4 in particular don't collapse to the rank-lock
  set.** Most-replicated winning pattern in 2024-2025 PSes (NVIDIA
  cuML blog, s5e8 1st).
- **Cost**: 4 reformulations × 3 archs × 5 folds ≈ 60 base fits,
  CPU 8-core, ~6-10h.
- **EV**: +2-10bp. Risk LOW (probably some lift from at least one
  formulation surviving the gate).
- **Pointer**: `scikit-survival` for (c); standard LGBM for others.
  See `audit/2026-05-04-research-loop-day3.md` mechanism #1.

#### T1.3 — Q12 Mandatory-2-compound rule_residual base
- **What**: F1 regulation requires ≥2 distinct dry compounds per
  driver per race. Encode `compounds_used_so_far[Driver, Race,
  Year]` as a within-group feature; derive
  `must_change_compound = (n_distinct == 1) AND
  (LapNumber > race_total_laps × 0.6)`,
  `forced_pit_pressure = must_change_compound × Stint`.
- **Why this works here**: Pool has zero regulatory-constraint
  features. **Directly attacks Stint-2 blind spot**: a driver
  late-race on single compound in Stint-2 MUST pit. The constraint
  is a 3-way (Driver × Race × Compound-history) which CTGAN
  partially preserves at within-group level if (Driver, Race,
  Year) groups are atomic in the synth.
- **Cost**: 3-5h CPU. Build as rule_residual base à la F1.2 → adds
  K=19 stack base.
- **EV**: +5-10bp standalone, +1.5-3bp K=19 stacked. Risk MEDIUM
  (depends on group atomicity — verify via
  `train.groupby([Driver,Race,Year])[Compound].nunique()`
  distribution before committing).
- **Pointer**: 51gt3.com FIA tyre rules; FastF1 for verification.

#### T1.4 — Hazard-rate / discrete-time survival reformulation
- **What**: Reformulate as discrete-time hazard. For each row,
  predict K=20 hazard-bucket outputs h(t) = P(pit on lap t |
  survived to t). NLL of hazard parametrization equals BCE on
  same target (Brown 1975 / Gensheimer-Narasimhan), but the
  decomposition is structurally different: the model learns the
  *shape* of the pit-decision curve over remaining-stint-laps.
- **Why this works here**: Stint-2 specifically benefits because
  hazard naturally encodes the (fresh-tyre low-hazard → old-tyre
  rising-hazard) curve through the K=20 head structure. The h(0)
  output is ρ≈0.95-0.97 vs M5q (genuine orthogonality).
- **Cost**: PyTorch MLP + nnet-survival loss; CPU or T4. 5-fold
  ≈ 3-4h.
- **EV**: +1-7bp. Risk MEDIUM (could end up correlated with M5q if
  loss landscape converges).
- **Pointer**: github.com/MGensheimer/nnet-survival; arXiv 2410.01086.

#### T1.5 — Deotte L2 stacking with std/mean/range meta-features
- **What**: Replace single LR-meta with **L2 = LGBM/Ridge** over
  `[base preds, std-across-bases, mean, range, max-min-gap]`.
  Then **L3 = Ridge weighted average** of {LR-meta, LGBM-L2,
  Ridge-L2}.
- **Why this works here**: LR can't encode std/mean across bases —
  those are not in the row's prediction span. **Adding std as
  feature is a strictly larger hypothesis class.** Up-weights rows
  where bases disagree (high-info rows). Cheap, safe, near-zero
  regression risk.
- **Cost**: 30 min CPU; reuses existing OOF arrays.
- **EV**: +0.5-3bp. Risk LOW.
- **Pointer**: NVIDIA Grandmaster cuML stacking blog (Deotte April
  2025); cuML for L2 if GPU.

### Tier 2 — strong second-week candidates

#### T2.1 — `next_compound` as feature (P5: 68% test computable)
- **What**: For 68% of test rows, the next-stint compound is
  observable in test. Train-side analog is trivial. Encode as
  one-hot Compound × `next_compound_known` flag.
- **Why this works here**: Combined with P4's prev_compound spread
  (18.9% SOFT→HARD vs 75.4% WET→HARD), this is a 56pp signal
  the pool isn't using. The 32% missing test rows get a
  `next_compound_known=0` fallback flag.
- **Cost**: 1-2h CPU. New base or feature add to existing rules.
- **EV**: +1-4bp. Risk LOW (signal is observed in train + 68% of
  test).
- **Pointer**: P5 in data probe results.

#### T2.2 — `prev_compound × laps_into_stint` Stint-2 rule_residual
- **What**: New rule_residual base (F1.2 pattern). Lookup
  Bayesian-smoothed pit rate over (prev_compound, Stint,
  laps_into_stint_decile). Train HGBC residual.
- **Why this works here**: P4 mapped exact Stint-2 signal —
  laps_into_stint monotone 22%→52%, prev_compound spread 56pp.
  None of these are first-class in M5q. **Targeted attack on the
  −341bp blind spot.**
- **Cost**: 1-2h CPU. Reuse `scripts/d6_multi_rule.py`.
- **EV**: +1-3bp. Risk LOW (proven F1.2 mechanism class).

#### T2.3 — Q7 Expected-stops + pit-pressure ratio
- **What**: `expected_stops[Race, Year]` from train empirical or
  Pirelli predictions; `pit_pressure = Stint / expected_stops`.
- **Why this works here**: When pit_pressure>0.7, pit imminent.
  Race-level feature; safe under CTGAN (Race-marginal preserved).
  Direct attack on Stint-2 (decision depends on expected stop count).
- **Cost**: 2-3h CPU + 1-2h Pirelli scrape (optional; empirical
  alone works).
- **EV**: +2-5bp. Risk LOW.

#### T2.4 — Q3 Safety-car probability per track + interactions
- **What**: `sc_prob[Race]` from public 2018-2024 SC stats;
  interactions `sc_prob × Stint`, `sc_prob × LapNumber_quintile`.
- **Why this works here**: At high-SC tracks (Singapore, Monaco,
  Baku, Vegas), opportunistic pits drive Stint-2 decisions outside
  the Pirelli-predicted windows. Pool doesn't encode this.
- **Cost**: 2-3h CPU + 1h public-data scrape.
- **EV**: +2-5bp. Risk LOW (Race-marginal preserved).

#### T2.5 — Bayesian hierarchical stacking (Yao 2021)
- **What**: Stacking weights `w_k(x)` vary across discrete strata
  (Compound, Stint, Year×Compound) under Gaussian partial-pooling
  prior. Inferred via VI in PyMC+JAX, ~10-30 min/fold.
- **Why this works here**: Per-Race isotonic over-fit (in-sample
  +24.6bp / inner-CV −10.9bp). Hierarchical Bayes shrinks toward
  global, so noisy segments collapse, signal-rich segments
  (Stint-2, Year=2023 mixture) pulled toward local optimum WITHOUT
  overfit.
- **Cost**: CPU 2-3h.
- **EV**: +1-6bp. Risk MEDIUM (could degenerate to LR-meta if all
  segments share the same optimum).
- **Pointer**: Yao/Pirš/Vehtari/Gelman 2021 PDF; PyMC+JAX backend.

#### T2.6 — SCARF/VIME pretraining on aadigupta1601 unlabeled
- **What**: aadigupta1601 has 101k rows the host did NOT include
  in train. Pretrain TabM/RealMLP backbone with SCARF (random
  feature corruption + InfoNCE) on those unlabeled rows. Fine-tune
  on labeled comp data.
- **Why this works here**: All previous pseudo-label attempts used
  the comp test set. Domain-shifted unlabeled is a **different
  statistical regime** → encoder initialization doesn't over-fit
  to test. Mathematically distinct from the d5 partial-pseudo
  failure mode.
- **Cost**: T4×2, ~14h end-to-end (pretrain 4-6h + fine-tune
  5-fold 8h).
- **EV**: +1-6bp on top of underlying NN. Risk MEDIUM (domain
  shift may be too severe — host explicitly mutated rows).
- **Pointer**: arXiv 2106.15147; Tschalzev arXiv 2302.14013.

### Tier 3 — cheap, low-EV-but-positive insurance moves

#### T3.1 — Year-2023 explicit mask + Year × Race interaction
P3 confirms 2023 is a structural mode collapse. Verify Year×Race
is in pool features; if not, add as one-hot in next K=N rebuild.
EV +1-3bp. Cost <1h.

#### T3.2 — `RaceLength_Estimate` per Race
P8 — Monaco 77, Italian 64.5, etc. Pairs with normalised
RaceProgress; adds absolute-lap-count signal not currently in pool.
EV +0-2bp. Cost <1h.

#### T3.3 — Adversarial-validation instance weights
Train classifier on `(train_or_test_label)`; use `p(test|x)`
density ratio as instance weight. Year=2023 mixture rescue.
EV +0-2bp; HIGH FLOOR (no harm). Cost 30 min.

#### T3.4 — Snapshot ensembling on RealMLP
Cosine annealing 5 cycles, save 5 snapshots, average. Cheap
multi-seed substitute. Expected bump alongside RealMLP-bag (in
flight on parallel branch).
EV +0.5-3bp. Cost ~3h T4 (one extra training).

#### T3.5 — GroupKFold(Race, Driver, Year, Stint) DIAGNOSTIC
P6 — Strat OOF leaks 80% of consecutive-lap pairs across folds.
Add GroupKFold as a *second OOF measure* (NOT as the LB proxy —
R1 keeps Strat). Use it to:
- triangulate which bases / rules survive without leakage,
- detect over-amplified pseudo-rebuilds early.
EV indirect (better gate calibration → fewer wasted slots).
Cost ~6h CPU for one full pool re-run.

#### T3.6 — `driver_is_low_count × Year=2023` interaction
P9 — 221 drivers with <10 rows; their pit_mean (0.26%) matches
2023 base rate, suggesting they're 2023-only synthetics. Adding
this interaction may sharpen the Year=2023 mode without leaking.
EV +0-2bp. Cost <1h.

### Tier 4 — falsified by probes; do NOT pursue

- **Big Bi-LSTM/Transformer over (Race, Driver) sequences** — P1.
  Test groups are 2.25-lap fragments; Frontiers AI 2025's 0.988
  AUC requires telemetry-rich 10-step sequences we don't have.
- **TabR / Hopular / retrieval-augmented MLP** — P2. NN distances
  too large for retrieval to add fresh signal.
- **TabPFN-2.5 / Mitra ICL ensemble** — same regime issue. Marginal
  EV at our 440k-row scale; sub-sampling to 50k context loses
  statistical power. Held but de-prioritized.
- **Broad pseudo-labeling** — d5 partial-pseudo confirmed gap-
  widening (−4.2bp LB). DEAD unless gate is structurally tighter.
- **Single-model Stint-2 specialist** — d3c −124bp on segment.
  Only useful as a rule_residual base (T2.2 above).
- **Per-Race / per-Stint isotonic** — d3 calibration in-CV regress.
  Bayesian hierarchical (T2.5) replaces this lever.
- **Reintroduce `Normalized_TyreLife`** — host explicitly removed.
  Coarse-bucket Pirelli reconstruction (Q4) is risky → NOT in
  Tier 1; only consider after T1+T2 land.

---

## 4. Suggested 14-day execution sequence

Three workstreams in parallel: GPU (Kaggle), CPU (local), Meta
(stacking-only, near-zero compute).

| Day | GPU stream | CPU stream | Meta stream |
|---:|---|---|---|
| 1 | (RealMLP-bag pulls finish on parallel branch) | T1.3 Q12 forced-pit rule_residual | T1.5 Deotte L2 std/mean/range |
| 2 | T1.1 TabM 1-fold smoke | T1.3 K=19 stack rebuild + LB submit | T3.1 Year-2023 mask audit |
| 3 | T1.1 TabM 5-fold | T2.1 next_compound feature → rule | T3.5 GroupKFold diagnostic |
| 4 | T1.1 TabM stack-add + LB submit | T2.2 prev_compound × laps_into_stint rule | T3.4 RealMLP snapshot ensemble |
| 5 | T2.6 SCARF pretrain on aadigupta | T1.2 Multi-formulation #1 (Poisson cum_pit) | — |
| 6 | T2.6 fine-tune | T1.2 Multi-formulation #2 (laps-until-next) | T1.5 polish; LB submit |
| 7 | T2.6 stack-add + LB submit | T1.2 Multi-formulation #3 (ratio target) | T2.5 Bayesian hierarchical smoke |
| 8 | T1.4 hazard-rate model | T1.2 multi-formulation K=N stack + LB | T2.5 hierarchical 5-fold |
| 9 | T1.4 hazard 5-fold | T2.3 expected-stops feature | T2.5 LB submit |
| 10 | T1.4 stack-add + LB submit | T2.4 SC probability features | — |
| 11 | TabM bag (3 seeds) | Q1 pit-window scrape (optional) | LB submit consolidation |
| 12 | TabM bag stack-add | Q1 rule_residual variants | — |
| 13 | Hold; check submission gate | Q1 rules → K=N+ stack | LB submit |
| 14+ | Final lock-in window per R5 | — | hedge selection |

Each LB submit is single-shot per R1, with **predicted-gap gate**
(<−7bp needs sign-off) + **minimal-input-meta sanity check** + new
**GroupKFold diagnostic** (T3.5).

---

## 5. EV math + what "winning" looks like

### Aggregate EV math
- Tier 1 sum-of-medians: T1.1 (5) + T1.2 (6) + T1.3 (7.5) + T1.4
  (4) + T1.5 (1.75) ≈ **24bp expected** if all land.
- Tier 2 sum-of-medians: ~10bp.
- Realistic transfer ≈ 50% (additivity is fictional, base
  redundancy compounds): **~17bp aggregate from Tiers 1+2.**

Current LB 0.95026 + 17bp = **0.95196**. That's 15bp short of
top-5% (0.95345), 24bp short of leader (0.95435).

**To get on top**, we need either:
1. **One of T1.1-T1.4 surprises us with double-digit lift** —
   e.g., TabM ρ=0.94 (vs RealMLP's 0.972) and adds 6-12bp; or
   the multi-formulation L1 hits a 4-formulation grid that
   re-shapes the rank entirely.
2. **A long-tail synth-leak surprise** — e.g., the Q12 forced-pit
   rule perfectly identifies the 1% of rows that determine the LB
   tail (the 80%-of-AUC-gain-from-20%-of-rows pattern).
3. **The leader is also stuck near 0.95435 ceiling** — they may
   not be using Tier-1 mechanism-class moves (most public
   notebooks for s6e5 are ≤0.946). Top-1% may be more reachable
   than the gap math suggests.

### Variance-conscious framing
The PI explicitly authorized wider steps and acceptance of
high-failure rate. **Tier 1 contains 5 candidates with median
1-7bp each but tail outcomes 8-15bp on T1.1, T1.2, T1.4.**
Running all 5 means at least one tail-case lift is plausible.

### What we do NOT lose by trying
- Compute is free (Kaggle GPU + CPU).
- Slot consumption: ~10 LB submits across 14 days = budget-safe
  (we have ~140 slots remaining).
- Each Tier-1 mechanism is a real-positive-information
  experiment regardless of outcome — the negative result moves
  the calibration ladder, the positive result is the breakthrough.

### What kills us
- **Spending another week on LR-meta pool tweaks.** 4× rank-lock
  confirmed. Stop.
- **Trying T4 sequence model anyway** — P1 is decisive.
- **Re-running broad pseudo-label** — d5 confirmed dead.
- **Optuna sweep on existing GBDTs** — single-model ceiling at
  E3 0.94870 reached Day-2. Diminishing returns.

---

## 6. Pointers

- `audit/2026-05-08-data-probe-results.md` — load-bearing for C2-C5
- `audit/2026-05-07-d6-critic-loop.md` — Rule 14 audit
- `audit/2026-05-07-d6-f1-2-multi-rule.md` — F1.2 mechanism template
- `audit/2026-05-04-research-loop-day3.md` — cross-comp deltas
- `audit/2026-05-04-d3-per-segment-analysis.md` — segment blind spots
- `scripts/d6_multi_rule.py` — F1.2 builder; reuse for T1.3, T2.1, T2.2
- `scripts/probes_d8/` — probe code
