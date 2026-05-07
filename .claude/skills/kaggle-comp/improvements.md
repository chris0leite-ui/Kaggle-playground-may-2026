# kaggle-comp skill — cross-comp improvements log

Edits promoted here when a friction pattern appears in 2+ comps, costs > 1 LB slot,
or required a human nag. See self-improvement.md for the full distillation protocol.

---

## Pending (not yet applied to skill files)

### [ ] kickoff-runbook.md / day-1: simple-LR baseline as Day-1 ceiling probe

`tag: lr-recipe-portable`. Day-1 of any new tabular comp, run the
30-second LR baseline (`KBins(20, quantile, onehot)` on every numeric +
`OneHot` on every cat → `LogisticRegression(C=1, solver='liblinear')`).
On s6e5: AUC 0.92038 in 22 s, closing 88% of the GBDT-vs-`lr_raw` gap.
Then run the mega LR (~8 min CPU, all FE families concatenated) — its
gap to single-GBDT tells you if stacking is necessary (>100 bp gap →
yes). Recipe + per-fold mechanics + mechanism map (LR vs GBDT vs NN
FE preferences) at `examples/fe-recipe-simple-lr.md`. **Origin:** s6e5
LR-bank experiment; `lr_kbins20_ohe` 0.92038 / `lr_mega` 0.92776 /
GBDT pool 0.95385. Anti-patterns codified: tree-engineered FE *hurts*
LR (Rozen FE: 0.857 vs raw+OHE 0.854 baseline); class_weight/L1/L2/
C-sweep are AUC rank-no-ops (skip the variants).

### [ ] kickoff-runbook.md Q5b — data + task description (≤10 sentences)

`tag: settled-once`. After Q5 EDA. (1) Each feature in domain terms,
(2) prediction task in real-world terms, (3) class balance →
metric/threshold implication, (4) top-3 features by F-score and why
they make domain sense. Write to `audit/<date>-day-1-kickoff.md`.

### [ ] guardrails.md G13 — single-model-first / kitchen-sink FE before stacking

`tag: recipe-over-judgment`. Before adding 2nd base or LR-meta in
first 3 days, build kitchen-sink FE (≥30 engineered features + CV TE
on every high-card combo) and the BEST single model. That OOF is the
floor; stacking adds on top, does NOT replace it. **Origin:** s6e5
ran K=22 + Path B for 13 days; a single LGBM with FE matched it on
Day-16 (after FS_A leak fix, OOF 0.946 — still −5bp under stack).

### [ ] guardrails.md G14 — family falsification requires ≥3 variants

`tag: family-falsification-too-quick`. A mechanism family (TE, FM,
lag, target-reform, pseudo, calibration) is only "dead" after ≥3
distinct configs of its key hyperparameter. Single-variant nulls
update the prior on that variant, not on the family. **Origin:** s6e5
TE family closed Day-3 on one 2-way × one smoothing variant; the 3-way
(Driver, Race, Year) was the load-bearing trick.

### [ ] guardrails.md G15 — framework is scaffolding, not authorship

`tag: recipe-over-judgment`. Reserve ≥1 slot per 3-day cycle for FE
creativity uncoupled from existing pool. Triggered when 3+ days
without a probe whose source idea is NOT a 1-step variant of an
existing experiment.

### [ ] guardrails.md G16 — fold-safe label-conditional aggregates

`tag: target-construction-layer-leakage`. Any feature derived from
labels via groupby aggregation (target encoding, mean-of-positives-
per-group, target-conditional ratios) MUST be re-fit per CV fold
using ti rows only. For test prediction either refit on full train
+ apply to test, or 5-fold-average models each with their own
ti-fitted aggregate. **Origin:** s6e5 Day-17 — `compound_avg_life`,
`race_avg_pit_lap`, `dc_avg_stint_life` fit on full train inflated
OOF +490 bp (0.95128 vs holdout 0.94637); v1 single LB 0.94107
(−863 bp gap); K=2 LR-meta LB −63 bp. **Diagnostic:** strict 80/20
holdout test (independent seed, FE state on 80% only, eval on 20%)
detects this in <10 min CPU without burning a slot.

### [ ] guardrails.md G17 — transductive features need AV check

`tag: transductive-features-need-AV-check`. Any FE that fits on
combined train+test (frequency encoding, quantile binning, factorize
maps, PCA/AE) requires adversarial-validation: train-vs-test
classifier AUC. If AV-AUC ≈ 0.5, combined is safe. If AV-AUC > ~0.55,
fit on train only. Even feature VALUES (not labels) can encode
distributional structure differing between train/test (or
public/private LB). **Origin:** PI s6e5 Day-17 lesson; companion to
G16. (s6e5 AV-AUC = 0.502 so combined-FE was safe here.)

### [ ] pre-baseline-gate.md items 8-11

`tag: eda-thin` + `public-notebook-scan-missing`.

```markdown
8. Public-notebook scan. `kaggle kernels list -s "<comp>" --sort-by voteCount`;
   pull top 5; list OOF AUCs, FE tricks, model classes. Re-scan at every plateau.
9. High-card TE inventory. List every cat × cat (and cat³) combo with
   unique-key count in (50, n_train/4). Flag the 3-way combo with largest
   unique count as load-bearing.
10. Domain-physics feature list. 5-10 features a domain expert would compute,
    each with one-line physics rationale. Implement ALL.
11. Single-model OOF target. Predict what kitchen-sink single LGBM should
    hit, calibrated against top public-notebook OOFs (step 8).
```

### [ ] day-loop.md — public-notebook re-scan + 80/20 holdout diagnostic

```markdown
### Auto-trigger: public-notebook re-scan
On 3 nulls / 5 saturations / 50% checkpoint / "redecompose": pull top
5 notebooks (≥10 votes); ask which features are NOT in our pool.

### 80/20 holdout (mandatory before any new-FE-family LB submit)
StratifiedKFold with INDEPENDENT seed; fold 0 as 20% holdout; fit FE
+ inner-CV TE on 80% only; train + eval on 20%. If holdout ≪ OOF by
> 10 bp, leak present — debug before submit.
```

### [ ] kickoff-runbook.md / day-loop.md — keep top public notebooks as repo reference

`tag: recipe-over-judgment`. Keep top 3-5 public Kaggle notebooks
under `external/kernels/` as reference examples (not copy-pasted
code). Use them to (1) reverse-engineer FE at every plateau,
(2) sanity-check our feature factory vs published recipes, (3) build
a cross-comp recipe library. End-of-comp: review and promote durable
patterns to skill `examples/` or `recipes/`.

**Recipe library (seed entries):**
- `s6e5/romanrozen/f1-pit-driver-race-year-encoding-0-95354.ipynb`
  CV TE on 6 high-card combos (incl. 3-way), ~50 engineered FE,
  Rozen-LGBM hparams (lr=0.025, leaves=255, max_depth=10, ff=0.65).
  CAVEAT: Rozen's reported OOF 0.95241 likely inflated by FS_A leak
  per s6e5 Day-17 audit; honest single-LGBM ceiling ~0.946.
- `s6e5/yekenot/ps-s6-e5-realmlp-pytabkit.ipynb` —
  6 load-bearing FE items (arithmetic ratios + floor-cat + count-
  encoding + KBins + 2-way combo cats + CV TE inside fold loop) +
  per-fold orig-aug stratified 4/5. Verified 5-fold OOF 0.95257
  standalone on s6e5 (matches yekenot pub 0.95273 within 1.6 bp).
  Full audit at `.claude/skills/kaggle-comp/examples/fe-recipe-
  yekenot-realmlp-kitchen-sink.md`.

### [ ] examples/ — yekenot FE transfers to GBDT (CatBoost) too

`tag: recipe-over-judgment`. Research-branch audit caveats yekenot
items 2 (floor-cat), 3 (count-encoding), 4 (KBinsDiscretizer) as
"NN-specific (RealMLP can't derive these; CatBoost CAN via CTR +
split-finding; expected lift smaller for CB)." **Empirically false
on s6e5 Day-17 PM**: applying items 2/3/4 + item 7 (orig-aug) to a
research-recipe CatBoost-GPU (Bernoulli + min_data_in_leaf=20 +
Year/Stint cat + default CTR; "v3" → "v4") lifted standalone 5-fold
OOF by **+20.7 bp** (0.94993 → 0.95200) and DOUBLED the K=21+1
LR-meta contribution (+12.06 bp → +24.21 bp). Stacked K=23 + h1d
landed **LB 0.95354** (s6e5 PRIMARY). Mechanism hypothesis: explicit
floor/count/KBins as direct numeric/cat inputs interact with
CatBoost's CTR + split-finding in ways pure native CTR doesn't
capture — the GBDT split-finder benefits from pre-discretized
columns at the same fineness yekenot tuned for the NN.

**Apply yekenot's full FE recipe to both NN AND GBDT bases.**

Origin: s6e5 Day-17 PM, commit 7d179d6 on
`claude/optimize-model-performance-rruC2`. Friction tag candidate:
`yekenot-floor-count-kbins-fires-on-gbdt-too`. Promote to
`examples/cb-yekenot-transfer.md` when seen in a 2nd comp.

### [ ] kickoff-runbook.md / day-loop.md — original-data row-augmentation default

`tag: recipe-over-judgment`. For synthetic-tabular Playground comps
with AV-classifier AUC < 0.55 (train/test ≈ i.i.d. with original):
default to per-fold concat of the original (real-DGP) data,
stratified 4/5 split, weight 1.0 (or downweighted if synthesizer
has heavy label distribution shift). On s6e5 Day-17 PM, the v3 → v4
single-CB lift was driven by the combination of yekenot FE items +
this orig-aug item; never documented as a default kickoff move.
Cross-comp: irrigation-water used the same trick at weight 0.5.

**Pre-condition:** AV-classifier AUC < 0.55. Skip if AV > 0.55
(distribution shift risk; orig rows pull predictions off synth-test
marginal).

---

## Applied
<!-- log completed edits here: date · file · one-line description -->
