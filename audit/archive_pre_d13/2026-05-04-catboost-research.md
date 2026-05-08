# CatBoost research note — Day-3 (2026-05-04)

Stage-A artifact for branch `claude/explore-catboost-options-jWhDD`. Maps the
CatBoost-specific levers we have NOT yet exploited against M3/E1 evidence,
with citations for each. ≤150 lines.

## Where M3/E1 stand (evidence anchor)

- M3 shrunk (depth=6, iter=800, lr=0.08, l2=3.0, default CTR): Strat
  **0.94612** (+53.7bp), GroupKF(Race) **0.91645** (-41.4bp **G1 FAIL**).
- E1 row-subsample (Bernoulli subsample=0.8 over M3): Strat 0.94596,
  GroupKF 0.91638 — dominated by M3 on both anchors.
- Strat best_iters: `[799, 799, 799, 799, 797]` → 4/5 hit cap → AUC is a
  floor.
- GroupKF best_iters: `[292, 133, 343, 375, 421]` → all early-stopped;
  fold variance 0.89758-0.93603 — Race is the binding constraint.
- Top-feature importance: **Year=48.7**, Stint=13.1, TyreLife=9.8 — Year
  dominates *as a numeric*, despite having only 4 unique values
  (2022-2025). Currently NOT in `CAT_COLS=[Driver, Race, Compound]`.
- Cardinalities: Driver=887 (test 801, all overlap), Race=26, Compound=5
  (`HARD/INTERMEDIATE/MEDIUM/SOFT/WET`), Year=4. Driver-Race pairs=14942.
- GPU: **unavailable** (CUDA driver mismatch). All variants are CPU-only.

## Untried mechanisms (≥5, with citations and predicted-lift bands)

| # | Mechanism | Predicted lift (bp) | Cost | Citation |
|---|---|---|---|---|
| **1** | **Year → `CAT_COLS`** (4 values, top importance, currently treated as numeric ordinal) | +5-15 Strat / +0-15 GroupKF | <5min edit | M3 audit; CatBoost CTR enables order-free encoding of low-cardinality ints |
| **2** | **`one_hot_max_size=10`** so Compound(5) + Year(4) become one-hot; CTR capacity concentrates on Driver(887) + Race(26) | +5-15 both anchors | 1-fold probe ~12min | [Garkavenko, "Categorical features parameters in CatBoost"](https://medium.com/data-science/categorical-features-parameters-in-catboost-4ebd1326bee5) |
| **3** | **`max_ctr_complexity=6`** (default 4) + explicit `combinations_ctr=['Borders:CtrBorderCount=15:Prior=0/1']` to force 3-way Driver×Race×Compound combinations | +5-10 Strat / ambiguous GroupKF | ~2× slower → probe at iter=400 first | [CTR settings](https://catboost.ai/docs/en/references/training-parameters/ctr); [ApxML: feature combinations](https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/catboost-feature-combinations) |
| **4** | **Counter-only CTR** (`simple_ctr=['Counter:Prior=0/1']`) — pure frequency encoding, zero target leakage | -10 to +5 Strat / +5-10 GroupKF | 1-fold probe | CatBoost docs; isolates leakage-vs-signal share of the GroupKF gap |
| **5** | **Slow + wide: `lr=0.03, iter=2500, l2=8, od_wait=100`** to let early-stop fire on Strat (M3 hit iter cap on 4/5 folds) | +5-10 Strat | 2× wall vs M3 | M3 audit; standard "ES properly fires" hygiene |
| **6** | **MVS bootstrap** (`bootstrap_type=MVS, subsample=0.7, mvs_reg=0.1`) — importance-weighted sampling, distinct from E1's Bernoulli | +5-15 GroupKF (regularization), -5 to +5 Strat | 1-fold probe | [Bootstrap options](https://catboost.ai/docs/en/concepts/algorithm-main-stages_bootstrap-options); MVS is "speed + reg" per CatBoost docs |
| **7** | **`grow_policy=Lossguide`** (leaf-wise, LightGBM-style) with `num_leaves=64, max_depth=8` | ambiguous +0-10 | Tree predict 10× slower per docs but training similar | [`grow_policy` issue #1348](https://github.com/catboost/catboost/issues/1348); SymmetricTree default is unique to CatBoost — Lossguide drops that inductive bias |
| **8** | **`boosting_type=Ordered`** (forced; auto-switches to Plain at >50k rows) — slower but reduces CTR target leakage | -10 Strat / +5-15 GroupKF | 3-5× slower; smoke first | CatBoost docs `common parameters`; "Ordered" is the original ordered-TS algorithm |
| **9** | **CB-RESID** — train CB on residuals from a closed-form rule (LapNumber == TyreLife → pit threshold) | +5-15 (stack-only diversifier) | 1-fold probe | Day-3 research-loop M#1 problem-reformulation; A/B already done in LGBM, CB adds a different base on the same reformulation |
| **10** | **Multi-seed CB bag** (seeds 42/123/456 on the winning variant, rank-average) | +10-30 | 3× wall of single variant | [Grandmaster Playbook #7](https://github.com/catboost/tutorials); typical GBDT seed-bag yield |

Not tried but lower priority (rejected for this branch):
- GPU CatBoost (driver unavailable).
- `score_function=L2` (default Cosine is well-tuned for AUC).
- Pseudo-labelling CB-PL (M5c already covers reformulation diversity at LB
  proxy 0.95000; PL adds train-time leakage risk without clear ROI vs #1-#3).

## Why CTR tuning is the highest-ROI single CatBoost lever here

Driver=887 and Race=26 are exactly what CatBoost's ordered-target-statistics
(CTRs) are designed for. M3 used **defaults** on every CTR knob:
- `simple_ctr` defaults to `Borders:CtrBorderCount=15:CtrBorderType=Uniform`
- `combinations_ctr` defaults to a similar config
- `max_ctr_complexity=4` (auto-built combinations)
- `one_hot_max_size=2` (so Compound and Year are NOT one-hot, even though
  card=5 and 4)

The single most important untouched knob is `one_hot_max_size`: with it at
2 (default), Compound (5 levels) and Year (which would be 4 if added to
CAT_COLS) get unnecessarily encoded as CTRs, *competing with Driver/Race
for CTR capacity*. Setting `one_hot_max_size=10` is free (no risk of
overfit on a 5-level cat) and frees CTR for the high-card features where
ordered-TS actually helps.

## Sequencing recommendation

Do mechanisms in order of expected ROI / risk:
1. **#1 Year → CAT_COLS** (5min, high prior — Year is already the single
   most-important feature). If lift, retain for all subsequent variants.
2. **#2 `one_hot_max_size=10`** (1-fold probe, ~12min). Often the cleanest
   single-knob lift on small-cat datasets.
3. **#5 slow+wide** in parallel with #2 (separate 1-fold probe). Tests the
   "iter cap is a floor" hypothesis from the M3 audit.
4. **#3 `max_ctr_complexity=6`** + #4 Counter-only as parallel CTR probes.
   Probe at iter=400 to bound wall.
5. **#6 MVS, #7 Lossguide, #8 Ordered** as the 3 anti-overfit variants
   targeted at the GroupKF Race-overfit constraint.
6. Best variant from 1-5 → **#10 multi-seed bag** (only on the winner; do
   not bag every variant).

## Risk register

- **Year leakage check.** If `Year ∈ CAT_COLS` lifts >20bp Strat but
  GroupKF gap widens, the CTR is exploiting the same Year-time-series leak
  numeric Year already exploits. Run G1 + G3 (rare-class flips) before
  promoting to stack.
- **CTR + GroupKF interaction.** CatBoost's CTRs are leakage-safe *within
  a fold*, but if `Race` is in `CAT_COLS` AND we evaluate on `GroupKFold(Race)`,
  the CTR for Race in val rows is built from a held-out value the CTR has
  never seen — this is essentially "CTR cold-start". Counter-only (#4) is
  the natural baseline that isolates this effect.
- **Wall budget on 1-fold probes.** CPU-only at ~70s/fold for M3 baseline
  → expect 100-150s/fold for #3 (more CTRs) and 200-400s/fold for #7-8
  (ordered/lossguide). Hard-cap each probe at 25min; abort if not done.

## Checklist (Stage B0 / B / C / D will reference this)

- [ ] Probe #1 (Year-cat) — single-knob delta — REQUIRED first.
- [ ] Probe #2 (one-hot size).
- [ ] Probe #5 (slow+wide).
- [ ] Probe #3 + #4 (CTR variants).
- [ ] Probe #6 + #7 + #8 (anti-overfit variants).
- [ ] Bag winner with 3 seeds.
- [ ] Refit M5c stack with new CB OOF columns.

End — 144 lines.
