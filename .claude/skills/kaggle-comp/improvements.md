# kaggle-comp skill — cross-comp improvements log

Edits promoted here when a friction pattern appears in 2+ comps, costs > 1 LB slot,
or required a human nag. See self-improvement.md for the full distillation protocol.

---

## Pending (not yet applied to skill files)

### [ ] kickoff-runbook.md — add data + task description step

**Tag:** `settled-once`. **Origin:** s6e5 kickoff PI manual nag.
**Where:** new Q5b after Q5 (EDA summary).

```markdown
### Q5b — data + task description (≤10 sentences)
1. Each feature in domain terms (not column name).
2. Prediction task in real-world terms.
3. Class balance → metric/threshold implication.
4. Top-3 features by F-score and why they make domain sense.
Write to `audit/<date>-day-1-kickoff.md ## Domain context`.
```

**Why:** Anchors experiments to real DGP. s6e5 TyreLife/Stint/Cumulative_Degradation
are physical tyre-wear proxies — surfacing this Day-1 would have seeded better
hypotheses earlier.

### [ ] guardrails.md — Guardrail 13: single-model-first

**Tag:** `recipe-over-judgment`. **Origin:** s6e5 Day-16 PI question.

```markdown
## 13. Single-model-first / kitchen-sink FE before stacking
**Trigger**: any decision to add a 2nd base or LR-meta in the first 3 days.
**Rule**: build kitchen-sink FE (≥30 engineered features + CV target encoding
on every high-card combo) and the BEST single model possible BEFORE stacking.
That OOF is the floor; stacking adds on top, it does not replace it.
**Prevents**: spending Day-3+ on stack-mechanism work while a single LGBM
with proper FE could close 70-80% of the gap to top-5%. In s6e5 we ran
K=22 + Path B for 13 days; a single LGBM with FE matched it on Day-16
(OOF ~0.951 vs PRIMARY 0.95090).
```

### [ ] pre-baseline-gate.md — items 8-11 (public-notebook scan, TE inventory, physics features, single-model target)

**Tag:** `eda-thin` + `public-notebook-scan-missing`. **Origin:** s6e5 Day-16.

```markdown
8. **Public-notebook scan.** `kaggle kernels list -s "<comp-slug>"
   --sort-by voteCount`; pull top 5 by vote count. List their published
   OOF AUCs, FE tricks, model classes. Re-scan at every plateau.
9. **High-card TE inventory.** List every cat × cat (and cat³) combo
   with unique-key count in (50, n_train/4). Flag the 3-way combo with
   largest unique count as a load-bearing TE candidate.
10. **Domain-physics feature list.** Write 5-10 features a domain expert
    would compute, each with a one-line physics rationale. Implement ALL
    in the kitchen-sink FE; let LGBM choose.
11. **Single-model OOF target.** Predict what a kitchen-sink single LGBM
    should hit, calibrated against top public-notebook OOFs (step 8).
    Single-model OOF is the floor; if stack OOF < single-model + 5 bp,
    you are leaving signal on the table.
```

### [ ] day-loop.md — public-notebook re-scan auto-trigger

**Tag:** `recipe-over-judgment`. **Where:** day-loop step 3 (pick experiment).

```markdown
### Auto-trigger: public-notebook re-scan
Fire on: 3 consecutive nulls / 5 saturations / 50% checkpoint / "redecompose".
Pull top 5 notebooks (≥10 votes), list features + OOF mentioned, ask which
features are NOT in our pool, build the gap as next experiment.
```

**Why:** Rule 7 covers domain web search but not explicitly current-comp
public notebooks. In s6e5 the leader-tier recipe sat at 72 votes the entire time.

### [ ] guardrails.md — Guardrail 14: family falsification needs ≥3 variants

**Tag:** `family-falsification-too-quick`. **Origin:** s6e5 TE-family
closed on Day-3 single 2-way variant; 3-way was the +200 bp magic.

```markdown
## 14. Family falsification requires ≥3 variants
**Trigger**: any "X family is dead" claim after <3 configs tested.
**Rule**: a mechanism family (TE, FM, lag, target-reform, pseudo, calibration)
is only "dead" after ≥3 distinct configs of its key hyperparameter (smoothing,
polynomial order, field count, key cardinality, regularization). Single-variant
nulls update the prior on that variant, not on the family.
**Prevents**: closing TE family on one 2-way × one smoothing variant when the
3-way (Driver, Race, Year) at smoothing 20 was the comp's load-bearing trick.
```

### [ ] kickoff-runbook.md / day-loop.md — keep top public notebooks as repo reference

**Tag:** `recipe-over-judgment`. **Origin:** s6e5 Day-16 PI suggestion.

Keep top 3-5 public Kaggle notebooks (those with the highest published
OOF / LB scores or vote counts) checked into the comp repo under
`external/kernels/` as **reference examples**. Don't copy their code
into our pipeline; keep them as a separate library to:

1. Reverse-engineer feature engineering tricks at every plateau (Rule 22).
2. Sanity-check our `make_features_*` against published recipes — if
   they have a feature we don't, justify the absence or add it.
3. Build a small library of "known-good single-model recipes" across
   comps (FE patterns + hparam regimes that empirically work) — the
   long-run version of `improvements.md` for tactics rather than process.

**Periodic review.** At end-of-comp wrap-up: review the saved kernels,
extract reusable FE patterns, promote 2-3 to a cross-comp `examples/`
or `recipes/` folder under the skill. **Track candidate recipes here**
under a "Recipe library" section as an indexed list.

**Recipe library (seed entries):**
- `s6e5/romanrozen/f1-pit-driver-race-year-encoding-0-95354.ipynb` —
  CV TE on 6 high-card combos (incl. 3-way), ~50 engineered FE,
  Rozen-LGBM hparams (lr=0.025, leaves=255, max_depth=10, ff=0.65).
  Single LGBM OOF 0.95241, blend LB 0.95354. Pull-and-keep pattern
  validated on s6e5 Day-16.

### [ ] guardrails.md — Guardrail 15: framework is scaffolding, not authorship

**Tag:** `recipe-over-judgment`. **Origin:** s6e5 Day-16 16-day plateau.

```markdown
## 15. Framework is scaffolding, not authorship
**Trigger**: 3+ days without a feature-creativity probe (defined as a probe
whose source idea is NOT a 1-step variant of an existing experiment).
**Rule**: the framework optimises HOW to evaluate; it does not generate WHAT.
Creativity must come from EDA, domain physics, public-notebook intelligence,
free-form ideation. Reserve ≥1 slot per 3-day cycle for FE creativity
uncoupled from existing pool.
**Prevents**: 16 days of disciplined loops never asking "what's the BEST
single model?". Discipline is necessary but not sufficient.
```

---

## Applied
<!-- log completed edits here: date · file · one-line description -->
