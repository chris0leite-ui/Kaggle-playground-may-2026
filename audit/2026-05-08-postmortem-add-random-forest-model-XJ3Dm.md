# Postmortem — 2026-05-08 add-random-forest-model-XJ3Dm

## What went wrong

- **Initial Angle B RF probe was over-sized** (n_estimators=1500,
  min_samples_leaf=50, no max_samples). 9-11 min per fold; killed
  at fold 3/5. ~30 min wasted. Misread d15c's n_estimators=4000 as
  a probe template — that was a single-base production fit, not a
  multi-angle sweep substrate. **Fix forward:** default probe
  scripts to fast-smoke settings (n_est ≤ 500, min_samples_leaf ≥
  100, max_samples ≤ 0.5) unless explicitly producing a final
  artifact. Rule 19 mentions "smoke at 1 fold / 50k rows" — extend
  the spirit to RF sweeps.
- **Initial Optuna search space included `max_depth=None` and
  `max_samples=None`.** Trial 2 grew past 3 min before I tightened
  and restarted. ~3 min wasted, low blast radius. Pre-flight
  question Q1 (already-explored?) would have helped — unbounded-
  depth RF on 440k rows × 30 features is empirically slow on 4
  cores.

## PI overrides this session

Three PI directions, all calibration-positive:

- "go kitchen sink" → produced first reproducibility check on the
  +0.25 bp K=4+1 forest-base lift; right call.
- "opting + kitchen sink" → confirmed +0.25 bp ceiling is
  hyperparameter-insensitive (cross-seed |Δ|=0.030 bp); right call.
- "Hold per Rule 27" on the K=5 + Path-B τ=100k submission
  (ρ=0.999917 vs PRIMARY; tie-band); correct protocol application.

No PI corrections of agent priors or rule-applications this
session. Calibration check (probe.py calibration): 14 paired
records, 1 PI override (handover open-axes count from prior
session). 0 PI overrides of agent decisions in this session →
within the 2-postmortem-without-overrides watch window
(`pi-stamp-risk` flag candidate but not yet active).

## Frictions logged this session

Cross-links to `audit/friction.md` 2026-05-08 entries (added by
this branch):

- `non-lr-meta-falsified-across-bagged-and-boosted-tree-classes`
  — RF-meta + RF-combined-input both lost to LR-meta on K=4
  expansion. Bagged-tree variant of Day-20 PCA-meta finding.
  4 of 4 non-LR-meta variants now falsified. See
  `audit/2026-05-08-rf-forest-sweep.md` § Angle B/C.
- `forest-base-on-yekenot-recipe-most-diverse-positive-on-K4`
  — RF-yekenot standalone OOF 0.94178, ρ=0.9595 vs PRIMARY,
  K=4+1 LR-meta +0.26 bp. Lowest ρ on a positively-gating base
  in the K=4 era. See § Angle A.
- `rf-feature-breadth-does-not-scale-on-s6e5` — Kitchen-sink RF
  (yekenot + 12 constraint + 7 inter-stint = 57 feat) standalone
  OOF dropped 1.24 bp vs yekenot-only; K=4+1 lift unchanged within
  fold noise. See § Kitchen-sink follow-up.
- `rf-optuna-cant-tune-past-natural-+0.25bp-ceiling` — 15-trial
  TPE search yielded cross-seed |Δ|=0.030 bp; the +0.25 bp ceiling
  is set by the meta architecture, not by RF. See § Optuna
  follow-up.
- `path-b-cs-absorbs-single-base-orthogonal-additions-below-0.5bp`
  — K=5 = K=4 + RF Path-B C×S τ=100k OOF +0.02 bp vs PRIMARY,
  ρ=0.999917 → tie-band, PI held submission per Rule 27. See
  § Path-B K=5 refit.

## Promotion candidates (PI ratified: PENDING; not yet promoted)

PI was asked for additions and ratification (postmortem step 4)
but issued "wrap up" without engaging the questions; promotion
candidates remain UNRATIFIED in `.claude/skills/kaggle-comp/improvements.md`.
Re-ask next session.

### Candidate 1 — non-LR meta closed across tree classes

**Tag:** `non-lr-meta-falsified-across-bagged-and-boosted-tree-classes`

**Where to insert:** `do-and-dont.md` § "Model selection / meta
architecture"

**What to add:**
> If the pool's logit effective rank is at or below `log2(K)+1`,
> do not test tree-class metas regardless of input augmentation
> (combined-input variants don't help). The bottleneck is
> logit-direction coverage, not non-linear meta interactions.
> LR meta with `[P, rank, logit]` expansion is the binding
> ceiling. Falsified across (LightGBM-meta Day-20, RF-meta today,
> RF-combined-input today) = 4 of 4 variants.

**Why:** Same pattern in two probes ~5 days apart; cost ~30 min
compute combined; predictable from priors but the rule didn't
exist until today.

### Candidate 2 — Path-B retention threshold

**Tag:** `path-b-cs-absorbs-single-base-orthogonal-additions-below-0.5bp`

**Where to insert:** `do-and-dont.md` § "Per-segment stacker"
(generalizes the Day-15 entry).

**What to add:**
> Per-segment Compound × Stint shrinkage at τ=100k retains the
> OOF lift of bases with ≥+0.5 bp standalone-OOF orthogonality
> (d15b DAE: +0.715 bp OOF on K=22, ρ=0.948 → 1.4× amp at LB).
> Below that threshold, shrinkage averages the contribution to
> noise: today's RF-yekenot at +0.25 bp K=4+1 LR-meta (ρ=0.96)
> → +0.02 bp through Path-B, ρ=0.999917 vs PRIMARY. **Rule:**
> for new-base candidates with ρ≈0.95 vs PRIMARY, if standalone-
> OOF lift < +0.3 bp at the K-pool's LR-meta gate, skip the
> Path-B refit — it will absorb.

**Why:** Generalizes the Day-15 friction with a quantitative
threshold derived from two data points (d15b, today). Saves a
predictable Path-B refit slot when standalone OOF is too small.

### Candidate 3 — skip Optuna on diversity-only bases

**Tag:** `rf-optuna-cant-tune-past-natural-+0.25bp-ceiling`

**Where to insert:** `do-and-dont.md` § "Heavy compute" (extends
Rule 6).

**What to add:**
> When a base's meta-utility is bounded by the meta architecture's
> logit-direction ceiling (rank-lock), hyperparameter tuning can
> only redistribute the diversity bonus across folds, not amplify
> it. **Rule:** skip Optuna on diversity-only bases (low standalone
> OOF, low ρ to PRIMARY, used only for rank-information at the
> meta). Reserve Optuna for strong contributors (CB-yekenot
> v3→v4 +12 bp OOF lift via FE recipe transfer was the canonical
> positive case). Today's RF Optuna over 6 hyperparameters × 15
> trials produced cross-seed |Δ|=0.030 bp.

**Why:** Today's run produced the data point. One comp's worth
of evidence; ratify only if PI sees the pattern.

## PI additions (from step 4)

PI issued "wrap up" without engaging the postmortem questions.
No additions captured this session. Re-ask next session: anything
to add, frictions missed, rules to extract, decisions to flag.
PI ratification of the three promotion candidates above is also
pending.

## Framework version at session-end

- Commit SHA: `52cc952a61e651d1ecb2a8063be4da11703b5258` (head
  before this postmortem commit)
- Active rules: 1..36 per `CLAUDE.md`
- Loaded skills this session: kaggle-comp, postmortem
- Submission count: 41 of 270 used (no submission this session
  per Rule 27 hold)
- Today's session-leaf in `state/current.md` updated; hedge
  ladder has 4 new R5 entries (RF-yekenot stack-add + 3 τ
  variants of K=5 Path-B refit).
