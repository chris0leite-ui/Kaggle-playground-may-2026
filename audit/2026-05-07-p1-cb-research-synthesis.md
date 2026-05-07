# P1 single-CB research synthesis — Day-17 PM (2026-05-07)

Goal: push past v3 single-LGBM honest ceiling (OOF 0.94563) toward
PI-asked LB 0.955 single-model, using CatBoost-GPU with target-encoded
features. This note synthesises CatBoost best-practice research from
(a) our local audit/2026-05-04-catboost-research.md, (b) the
chris0leite-ui/Kaggle-irrigation-water postmortem + LEARNINGS.md, and
(c) CatBoost official docs + Garkavenko's "Categorical features
parameters in CatBoost" Medium article.

## Findings that changed the recipe

### 1. Defaults are right for binary class — DROP custom CTR

- Garkavenko: "For **binary classification**, parameter tuning is very
  similar to regression task, except that it is usually useless to
  increase the `TargetBorderCount` parameter value." Default is 1 for
  classification — leave it.
- `simple_ctr` and `combinations_ctr` defaults (`Borders` +
  optional `Counter` with `CtrBorderCount=15`) are well-tuned for
  binary class. Custom configs we tried trip "0 target borders" on
  fold-1 smoke (Median/MinEntropy qualifiers are regression-only).
- `max_ctr_complexity=6` produces **6.4× larger model** for negligible
  accuracy lift (Garkavenko). Default 4 is sweet spot. **Reverted
  from `=6` to default 4.**

### 2. Bernoulli + rsm=0.8 beat MVS in our smoke

- Recipe v1 (MVS, custom CTR, no rsm): smoke fold-1 AUC **0.93355**,
  ES at iter 106, wall 40s, projected 5-fold wall 30 min.
- Recipe v2 (Bernoulli + subsample=0.8 + rsm=0.8 + default CTR):
  smoke fold-1 AUC **0.93489 (+13 bp)**, ES at iter 336, wall 60s,
  projected 5-fold wall 44 min. Cleaner recipe trains longer before
  ES (more headroom), produces +13 bp on identical 50k subsample.
- Bernoulli is GPU/CPU symmetric (no `mvs_reg` strip needed). MVS
  is theoretically faster + more regularised but didn't pay off here.

### 3. Column subsampling (rsm) was untested on s6e5

- All M3 / E1 / cb_lossguide / cb_slow-wide variants used default
  `rsm=1.0`. Adding `rsm=0.8` (~80% feature sample per tree) is a
  free regulariser — irrigation-water shipped this in their
  PRIMARY recipe.

### 4. min_data_in_leaf — irrigation's 2 is too aggressive at 350k rows

- Irrigation-water used `min_data_in_leaf=2` (with `l2_leaf_reg=0`)
  on a 600k-row 3-class problem. Worked for them.
- s6e5 has 351k train rows (per-fold ti ≈ 280k), pos rate ~2-3%.
  With l2=8.0 + rsm=0.8 + subsample=0.8 we already have substantial
  regularisation. Set `min_data_in_leaf=20` — modest constraint that
  prevents leaf overfit on the rare (PitNextLap=1) class without
  starving the majority class.

### 5. border_count=254 on GPU per CatBoost docs

- "128 splits are enough for many datasets. However, try to set the
  value to **254 when training on GPU if the best possible quality
  is required**" (CatBoost parameter-tuning docs). CPU keeps default
  128 — minimal lift on CPU at much higher mem cost.

### 6. CatBoost native CTR > hand-rolled post-hoc TE

- Irrigation LEARNINGS.md verbatim: "CatBoost native ordered TE ≠
  mean TE. CatBoost's built-in `cat_features` uses a permutation-
  ordered row-wise TE with noise injection... When OOF mean TE
  yields no lift but CatBoost-native produces gains, the noise-
  injection and row-permutation structure is performing genuine work
  that simple encoding misses."
- Implication for our recipe: pass `[Driver, Race, Compound, Year,
  Stint]` as `cat_features` and TRUST CatBoost's CTR. Our 6
  hand-rolled `cv_target_encode` columns (Rozen recipe verbatim) are
  retained as low-cost additions but not load-bearing — they may be
  redundant with CatBoost's native ordered-TS at high CTR
  cardinality (Driver=887, Race=26).

### 7. Original-data row-augmentation — irrigation's synthetic-DGP trick

- Irrigation `recipe_catboost_skte.py`: combined synthetic train fold
  with original archive data, weighted at 0.5, in the same fit call.
- s6e5 d15 work: 97.55% of synth `LapTime` values exist in the
  `aadigupta1601` original. Synth corrupted joint structure but
  preserved marginals. **Original rows give CB clean per-row signal
  about the underlying DGP**; never tried in any of our 22 K=22 base
  models. New `--with-orig-data` flag.

## Recipe diff summary

| Knob | Before | After (research-backed) | Rationale |
|---|---|---|---|
| `bootstrap_type` | MVS | **Bernoulli** | irrigation proven; GPU/CPU symmetric; +13 bp smoke |
| `subsample` | 0.7 | **0.8** | irrigation proven |
| `rsm` | unset (1.0) | **0.8** | column subsampling = free regulariser |
| `mvs_reg` | 0.1 | **dropped** | not needed with Bernoulli |
| `simple_ctr` | custom Borders+Counter+BTMV | **default** | binary class needs no tuning |
| `combinations_ctr` | custom | **default** | same |
| `max_ctr_complexity` | 6 | **default 4** | Garkavenko: 6.4× model size for marginal lift |
| `min_data_in_leaf` | unset (1) | **20** | leaf-level rare-class regulariser |
| `border_count` (GPU) | unset (32) | **254** | CB docs explicit max-quality |
| (new) `--depth` knob | hardcoded 10 | **arg** | docs recommend 6-10 |
| (new) `--with-orig-data` | n/a | **arg** | irrigation synthetic-DGP trick |

## Predicted OOF (BOTE on 50k smoke → full data)

Smoke v2 fold-1 AUC = 0.93489 on 50k subsample, 1500 round cap. The v3
LGBM at the same scale produces fold-1 AUC ~0.945 (from p1-results.md
fold-table mean). So 50k smoke is a noisy lower bound; full 5-fold on
350k rows + ES off the cap should produce:

- Bare recipe (no extras, no orig-aug): expected OOF 0.948 ± 0.003
  (vs LGBM v3 honest 0.94563 — small lift from CB native CTR + Year
  cat + Bernoulli+rsm+min_data_in_leaf)
- + base-OOF + KNN extras: expected OOF 0.952 ± 0.003
- + 3-seed bag: +1-3 bp
- + `--with-orig-data`: ambiguous +5 to -5 bp (test/private LB
  distribution may regress; original rows could pull predictions
  off the synth-test marginal)

**Most likely outcomes** (5-fold CV, full data):
- **Realistic**: OOF 0.949-0.952, LB 0.945-0.948
- **Bull**: OOF 0.953-0.955, LB 0.948-0.951 (if orig-aug + extras
  combine multiplicatively rather than additively)
- **Bear**: OOF 0.946-0.948, LB 0.941-0.944 (extras don't transfer
  as features to a single-CB the way they did as stacked OOFs in
  K=22 LR-meta)

LB 0.955 single-model remains structurally implausible without
external Pirelli/FastF1 hard-join features (HANDOVER A4) — see
research synthesis above.

## Next steps

1. Build self-contained Kaggle GPU kernel `kernels/p1-single-cb-v3-gpu/`
   (CPU 5-fold projected ~4 h is over the 1-h gate; GPU T4×2 should
   be ~30-60 min including FE).
2. Submit with --with-base-oofs --with-knn --n-seeds 3 as PRIMARY-
   candidate single-CB attempt.
3. **MANDATORY 80/20 honest holdout** before LB submit (Rule 24); the
   `with-orig-data` axis is a new FE family and must be holdout-
   validated to catch any FS_A-style leak.
4. Compare outcomes vs the BOTE bands above; update friction tags
   and improvements.md.

## TODO at end-of-comp — promote findings to cross-comp playbook

PI ask 2026-05-07: collate these CatBoost best-practice findings into
a *durable* cross-comp document for future Kaggle tabular comps.
Promote the lessons that empirically held on s6e5 (e.g. "Bernoulli
beats MVS at 350k rows binary class", "default `simple_ctr` is right
for binary class", "max_ctr_complexity=6 is cosmetic per Garkavenko,
costs 6.4× model size", "rsm=0.8 is a free regulariser",
"`with-orig-data` row-augmentation works on synthetic-DGP comps when
AV-AUC ≈ 0.5") into:

- `~/.claude/skills/kaggle-comp/improvements.md` (R8; the
  framework-level postmortem destination)
- A new section in the eventual `s6e5-postmortem/03-what-worked.md`
  if the recipe lands an LB lift

Findings that need empirical validation on s6e5 BEFORE promotion:

- [ ] Bernoulli vs MVS at full 350k 5-fold (smoke shows +13 bp on 50k)
- [ ] rsm=0.8 vs rsm=1.0 isolated A/B
- [ ] `--with-orig-data` weight-0.5 lift vs ablation
- [ ] depth=10 vs depth=8 vs depth=6 sweep
- [ ] Native CTR vs hand-rolled cv_target_encode columns isolated
      (drop TE, keep raw cats; measure delta)

Leave the consolidated playbook write-up for end-of-comp wrap (R8).

## Sources

- audit/2026-05-04-catboost-research.md
- audit/2026-05-04-irrigation-water-postmortem.md
- chris0leite-ui/Kaggle-irrigation-water `LEARNINGS.md` (raw)
- chris0leite-ui/Kaggle-irrigation-water `scripts/recipe_catboost_skte.py` (raw)
- catboost.ai/docs/en/concepts/parameter-tuning
- catboost.ai/docs/en/references/training-parameters/ctr
- medium.com/data-science/categorical-features-parameters-in-catboost-4ebd1326bee5 (Garkavenko)
- forecastegy.com/posts/catboost-binary-classification-python/

— 138 lines.
