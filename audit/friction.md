# Friction log

One-sentence-per-item, weekly summaries. **For the full historical
detail, including code snippets, mechanism explanations, and the
running-diary entries from Days 1-19, see `audit/friction-archive.md`
(1,450 lines, kept for back-reference; not read by default).**

How to add an entry:

1. One sentence describing what went wrong.
2. One sentence describing the fix or what was learned.
3. If durable, promote to a CLAUDE.md rule. If session-specific, leave here.

If a fix is already a rule, reference the rule number rather than
restating it.

## Week of 2026-05-08

- `research-scan-duplicate-mechanism-claim` (2026-05-08 PM
  research-feature-engineering-7oCmj): proposed Frontiers AI 2025's
  `DriverAheadPit`/`DriverBehindPit` peer-effect features as a
  research-backed untried mechanism, but A3-1 RankSortedGaps already
  implements both `_ahead_pitted_lag1`/`_behind_pitted_lag1` and the
  full gap-based peer family — and A3-1 NULLED in Phase 1 smoke.
  Self-corrected before launching the probe. **Fix:** before
  proposing any "research-backed untried mechanism," grep the
  existing FE pick registry (`fe_picks_*.py`) for class-overlap;
  read the smoke results table to confirm not-yet-tested. Cost of
  catch was nil (caught in PI question construction); cost of miss
  would have been ~10 min CPU spent reproducing a known null.
- `tree-stack-meta-overfits-small-K-pool` (2026-05-08 PM): A2-8
  LightGBM stack-meta on K=4 with 43 meta features (P + ranks +
  logits + pairwise products + abs-diffs + logit-diffs + raw side
  info) lost −1.30 bp vs Path-B PRIMARY and even −0.96 bp vs plain
  LR-meta. Fold-std 0.00080 (vs typical ~0.00050). Tree depth-4
  splits absorb interaction noise faster than they extract signal
  on a 4-base pool. **Lesson:** convex LR + Path-B partial-pooling
  regularize better than gradient boosting at small K. Add to
  `audit/friction-archive.md` under axis-class falsifications.
- `day-counter-drift` (PI-flagged 2026-05-08 PM): prose across
  `state/`, `HANDOVER.md`, `audit/`, `glossary.md` referred to "Day-17
  PM", "Day-18 PM", "Day-19" as if calendar-aligned. They were not.
  The `d13`..`d19` labels are an experiment-iteration counter that
  ran ~10 days ahead of the calendar. Today is **2026-05-08 = comp
  day 8 of 31**, with 23 days remaining; "Day-19" prose was
  hallucinated. **Fix forward:** all prose now uses ISO dates or
  comp-day-N anchored to 2026-05-01. The `dN` short-codes remain as
  frozen file/code prefixes (per `glossary.md`) and explicitly NOT
  calendar days.
- `pool-rank-lock-at-logit-direction-not-rank-correlation`
  (2026-05-08 PM): three structurally-different inductive biases
  (LambdaRank per-stint, inter-stint memory features, stint-completion
  dual-head) all NULL at K=10+1 within ±0.05 bp despite low rank-
  correlation (ρ 0.41–0.73). Pinpoints the rank-lock mechanism: the
  K=10 [P, rank, logit] = 30-feature expansion can reconstruct any
  new base's logit prediction as a linear combination. Different
  rank info ≠ logit-direction contribution. See `ASSUMPTIONS.md` A29,
  A30.
- `K4-sparse-pool-promoted-to-PRIMARY` (2026-05-08 PM): K=4 forward-
  greedy + Path-B Compound × Stint τ=100k landed at LB 0.95351 vs the
  prior K=27 PRIMARY at 0.95368 (Δ −1.7 bp). The 17 extra bases were
  buying us 1.7 bp on LB. Promoted to PRIMARY at this deliberate cost
  for cleaner reference; old K=27 artefact retained as hedge per Rule
  R7.
- `kernel-class-fails-when-standalone-AUC-gap-to-gbdt-exceeds-300bp`
  (explore-svm-kernels-TRcuo 2026-05-08 PM): kernel-SVM family
  (Nyström-RBF + LinearSVC, kernel-logistic, 5 SVM specialists)
  standalone OOF 0.91-0.92, all null at K=27+1 / K=10+1 (Δ −0.09 to
  +0.05 bp). 8th-9th rank-lock confirmation. Even kernel-class
  structural diversity insufficient when AUC gap to GBDT-class
  exceeds 300 bp.
- `non-parametric-meta-on-K=4-cant-beat-LR-meta-without-new-input`
  (same branch): kernel-SVM-meta over K=4 ties Path-B PRIMARY
  exactly (0.95403 OOF). NCA-kNN on K=4 / K=10 ensemble nulls
  (±0.07 bp). Combined-input meta (K=4 preds + top-5 numerics)
  +0.03 bp LR / −1.64 bp kernel — bases already absorb raw features.
  K=4 saturated for meta-routing; logit effective rank ~3. Need a
  fresh base, not a fresh router.
- `nca-loss-matrix-O(n2)-OOM-at-50k`: NCA pairwise-distance loss
  matrix is O(n²) regardless of input dim; 50k subsample tries to
  allocate 18.6 GB. Fix: cap NCA fit subsample at 8-10k for 15 GB
  RAM, accept the metric-fit-on-subsample tradeoff. Apply learned
  projection to full 350k for kNN classify.
- `lightgbm-pandas-2-string-dtype`: LightGBM's
  `_check_for_bad_pandas_dtypes` rejects pandas StringDtype columns
  (typed as `str` not `object`); detection via `dtype == object`
  silently misses them. Fix: detect non-numeric cols via
  `not pd.api.types.is_numeric_dtype(...)` instead.
- `pandas-merge-many-to-many-row-explosion`: sequence-feature builder
  used `sorted_df.merge(prev_seg, ...)` where prev_seg was not
  deduplicated by merge key; produced 1.4M rows from 627k input. Fix:
  build a single per-segment summary via `groupby(...).agg()` and
  merge once with `validate="many_to_one"` to catch row explosions
  immediately.
- The audit-ml-repo branch's history rewrite removed binary blobs from
  git (3.9 GB → 31 MB on origin) but leaves a `.git` of the same
  size locally until `git gc --prune=now --aggressive` runs (slow,
  ~20-30 min on this size).
- `handover-open-axes-overstated`: `HANDOVER.md` and
  `state/{current,hypothesis-board}.md` listed sequence-level
  fingerprinting and Driver×Race×Year interaction-TE as "open" or
  "untouched" axes, but both had been falsified or had leaked under
  earlier strict audits (d16 GRU −0.043 bp NULL; field-state −0.015 bp
  NULL; combined-frame lead/lag −0.36 bp; `make_features_A` interaction
  TE LB 0.94107 vs OOF 0.94970). Errata in `HANDOVER-ERRATA.md`. Fix:
  every "open axis" line in the handover must cite the specific
  variant that was NOT tried, distinguished from the tried-and-failed
  version.
- `oof-lb-gap-misread-as-overfit`: the team has been tracking a
  consistent −5 to −6 bp OOF→LB gap as a structural overfit signal.
  Probe B (Day-19) shows a bootstrapped 95% CI of [0.95309, 0.95550]
  for a random 20% public draw on the PRIMARY OOF; the observed LB
  0.95368 is well inside that band. The "gap" is sampling noise.
  Fix: re-run the 1000-bootstrap CI before treating any OOF→LB
  divergence as structural.
- `synth-coherence-misframed`: the assumption that the synthesiser
  "broke within-stint sequence coherence" is wrong. Probe C shows
  physical constraints (Compound, TyreLife, LapNumber) are preserved
  at ≥99.99%. Mechanism is temporal downsampling: synthetic stints
  mean 3.87 laps vs original 19.80; gap=1 frac 27.98% vs 99.60%.
- `assumption-vs-evidence-tracking`: introduced `ASSUMPTIONS.md` to
  separate MEASURED / INFERRED / ASSUMED / FALSIFIED claims. Re-check
  on every postmortem and at handover prep.
- `residual-concentrated-on-rain-rows`: PRIMARY's worst (Compound × Stint
  × position) cells are all INTERMEDIATE / WET. Per-cell AUC 0.68-0.86
  vs global 0.954. Suggests a rain-condition specialist as a candidate
  axis NOT currently in any open-axes list.
- `pool-collapse-K4-effective-rank-1.33` (2026-05-08 PM,
  research-model-extensions-Ibwvn): SVD on the K=4 forward-greedy pool
  shows logit effective rank = **1.33** (entropy on singular values),
  far below K=27's 3.23 (A25). Component 1 alone captures 93.6% of
  variance and correlates with TyreLife (−0.33), LapNumber (−0.30),
  Compound dummies — the dominant direction is "tyre-degradation
  pressure × compound." **Forward-greedy reduces effective rank faster
  than base count.** Implication: the "3-D ceiling" framing in A25 was
  K=27-specific; K=4 is much tighter. Audit:
  `audit/2026-05-08-four-lane-research-extension.md`.
- `predictive-eff-rank-not-variance-eff-rank` (2026-05-08 PM, pca-k25-
  ensemble branch): A25 recorded the K=27 logit pool's eff-rank as 3.23
  and we treated that as the **predictive** ceiling under LR-meta (per
  A30 wording). The PCA-meta probe shows top-3 PCA-LR scores 0.95061
  (−35.64 bp vs the K=10 anchor 0.95417), while top-15 PCA-LR scores
  0.95401 (≈anchor). The 3.23 is **variance**-eff-rank; predictive
  eff-rank ≈ 15. Original A25 claim was correct; the A25→A30 inference
  is where the slip happened. **Fix:** `ASSUMPTIONS.md` A25 now reads
  "variance-eff-rank=3.23"; A30 cites A30b/A30c for refinements.
  Complementary to `pool-collapse-K4-effective-rank-1.33` and
  `non-LR-meta-on-K4-regresses` — three independent confirmations that
  the rank-lock framing needed precision.
- `non-LR-meta-on-K4-regresses` (2026-05-08 PM): direct test of A30
  (the only architecturally-untested avenue per
  `state/hypothesis-board.md`). Gradient-boosted meta on K=4
  [P, rank, logit] = **−1.20 bp** vs LR; 2-hidden-layer MLP meta =
  **−7.77 bp**. Augmented LR with raw row features = −0.04 bp (flat).
  **A30 dropped from `live` to `FALSIFIED`.** LR is the right model
  class for combining 4 collinear bases — non-linearity overfits the
  30-feature meta projection. Cross-confirmation: pca-k25-ensemble
  branch tested LightGBM-meta and Path-B-on-PCs at K=27 — both
  underperform LR by 1-7+ bp. **Two independent K-pool sizes, two
  non-LR meta classes, all negative.**
- `gap-feature-absorbed-by-tyrelife-stint-lap-compound` (2026-05-08 PM):
  W3 (downsampling) marginal is strong — P(pit | gap=1) = 8.5% vs
  P(pit | gap≥11) = 30%, a 3.5× gradient — but K=4 LR meta calibration
  per gap-bucket has ECE 0.0001-0.0015 (near-perfect). Gap as meta
  feature +0.02 bp; gap-augmented LGBM as base-level feature K=4+1
  gate +0.001 bp; per-gap isotonic recalibration −2.18 bp. Conclusion:
  TyreLife + Stint + LapNumber + Compound implicitly carry all gap
  information. Closes W3 as an actionable axis.
- `synth-divergence-from-F1-realism-on-last-lap` (2026-05-08 PM):
  in this synth, P(pit | is_last_lap_of_race) = **0.38**, NOT ~0 as F1
  reality dictates. n=21 so noisy, but the direction is opposite.
  Senior-lens "race-end no-pit" rule clamp consequently misfires
  (−9.48 bp under deterministic clamp). **Lesson:** F1-domain priors
  must be empirically verified against the synth's labelling before
  use; do not apply real-F1-strategy rules without checking.
- `pitnextlap-not-deterministic-from-observed-row-structure`
  (2026-05-08 PM): observation-time check showed only **29% of train
  rows have lap L+1 (real, observed) present** in the data. When L+1
  is observed, PitStop[L+1] matches PitNextLap[L] only 81% of the time.
  PitNextLap is therefore a probabilistic forward-looking label whose
  construction includes synth-introduced noise; this bounds achievable
  AUC near where K=4 already sits (~0.954). Implication: a
  discrete-time-hazard reformulation against `(stint ends in next k
  laps)` doesn't cleanly map to PitNextLap — abandoned.
- `isotonic-overfits-when-base-already-calibrated` (2026-05-08 PM,
  2nd confirmation): per-gap isotonic (P1.3) and per-Compound isotonic
  (P3.2) BOTH regressed (−2.18, −1.78 bp) despite ECE diagnostics
  showing well-calibrated input. Pattern: when the base meta is
  already near-zero ECE per stratum, fold-restricted per-stratum
  isotonic wastes parameters on noise. **Promotion candidate to
  CLAUDE.md Rule:** "Isotonic per-stratum recalibration requires ECE
  > 1% in the stratum to be worth attempting."
- `anchor-cited-from-memory-not-measurement` (2026-05-08 PM, pca-k25-
  ensemble branch): I cited the K=10+1 plain LR-meta OOF as ~0.94850 in
  a PI-facing AskUserQuestion when designing the PCA-meta probe. Actual
  anchor (re-measured by the probe itself) was **0.95417** — 5.7 bp off
  the cited value. Didn't change the strategic framing (relative deltas
  drive verdicts, not absolute level), but it's a discipline failure.
  Root cause: confused the K=10 plain LR-meta OOF with a different
  number elsewhere in the docs and didn't verify before pasting.
  **Fix:** when citing OOF anchors in PI-facing prose, grep
  `state/calibration-ladder.md` and `audit/decisions.jsonl` first; if
  the anchor isn't there as a clean datapoint, say "from memory, will
  re-measure in probe" instead of pasting a number. Promotion
  candidate: extension to Rule 26 (i)/(ii) — "(iii) every numerical
  anchor in a BOTE / question is grep-cited or labelled 'from memory'".

## Week of 2026-05-07

- The `cross-row-aggregates-fire-where-own-row-sequence-doesnt` finding
  (per-(Race, Year, LapNumber) aggregates lift a single LightGBM by
  +15.58 bp standalone) survived strict fold-safe audit but evaporated
  to zero at the meta-stacker — 7th rank-lock confirmation; promoted
  to Rule 32 family.
- Yao/Vehtari covariance-modulated Path-B regressed across three τ
  values vs plain shrinkage; the per-segment-stacker family is now
  empirically exhausted on K=27 (9 variants tested across Days 14-19).
- D1 external-data join (debashish historical priors) closed null-by-
  pre-flight when PI's sealed prediction (−2 to 0 bp) and the harness's
  BOTE (0.20 bp) converged; treat PI + harness convergence as a
  load-bearing close signal.
- LR-bank effective rank ceiling is 2.0-2.19 even with 5 distinct LR
  variants; LR-class is structurally low-rank for this comp; LR-only
  candidates are bounded at +0.3-1 bp K=21+1 lift.
- Per-segment LR with rich FE (Compound × Year, mega features)
  delivered +60.8 bp standalone but routes through cb_year-cat at the
  meta-stacker — friction
  `per-segment-mega-LR-fires-only-at-LR-class-not-meta-class`.
- The s6e4 "three axes must all be true" recipe (logits / class_weight
  / multinomial) doesn't transfer; it applied to balanced-accuracy on
  multinomial, not to binary AUC; check Q6 (Rule 16) before adopting
  cross-comp recipes.

## Week of 2026-05-06 (load-bearing leakage discoveries)

- Target-construction-layer leakage discovered: per-group computations
  on the label (`reverse_cum`, `inv_laps_until_pit`, `pit_horizon`)
  inflated OOF by 88-100% when they used full-train labels per group
  instead of fold-restricted ones; **this is the origin of Rule 24**.
- Held submissions built on those targets must not be submitted
  (`path_b_K22_invlaps_*`, `path_b_K23_dae_invlaps_*`,
  `path_b_K25_megapool_*`).
- The synthetic data-generating process is conditionally near-
  independent: trained 4 LightGBM regressors to predict each of 4
  features from the others; OOF RMSE matched marginal σ within 3 sig
  figs across all four. Per-row feature engineering is dead.
- ρ alone is not sufficient for meta-utility: 4+ confirmations of
  bases with very low ρ (extreme diversity) that landed null at
  meta-add. Codified in `scripts/probe.py` family priors.
- LR-meta multi-add gives no more than the max of the individual
  candidate gains, not the sum (Rule K).
- The denoising-autoencoder base became PRIMARY at LB 0.95059 (+1 bp);
  realised amp 1.4× — well below the 6-11.6× per-segment-stacker
  precedent; the amp pattern is conditional on a meta-architecture
  redesign, not a base addition.
- The kitchen-sink recipe transfer from a public Kaggle notebook
  (`yekenot`) lifted the stack by 24 bp at K=21+1 — the project's
  largest single base-add. Public-notebook scan should have happened
  16 days earlier; **this is the origin of Rule 22**.
- Day-17 P1 single-model thesis falsified: best honest single LightGBM
  with kitchen-sink Rozen recipe is OOF 0.94563; PRIMARY hier-meta is
  0.95090. Stacking is +52 bp ahead.
- Two-level stacking with the meta-OOF as a base produced LB regress
  −63 bp on +30.79 bp OOF; meta-derivative bases don't fire Path-B amp
  and DON'T transfer to the LB.

## Process frictions (recurring across weeks)

- Subagents dispatched for long-running Python jobs SIGTERM their child
  processes when they time out or exit; 4-of-4 recurrence. **Rule 28**
  forbids this.
- The bootstrap script gated on `KAGGLE_API_TOKEN` while the sandbox
  exposed `KAGGLE_KEY`; agent surfaced a false "missing token" blocker
  twice. The bootstrap fallback is now wired in; agents must
  `env | grep -i <service>` before asking PI for a credential.
- Pre-submit-diff was MISSING multiple times; landed three identical
  LB 0.94991 submissions in one day. **Rule 27** is now mandatory.
- Same-session friction not applied within the same session multiple
  times (`lesson-not-applied`). **Rule 29.**
- P100 vs T4 GPU compatibility friction reproduced 12 days apart
  (Day 3 → Day 15). **Rule 30.**
- 7 parallel LightGBM probes ran 4× slower than 1; 3-parallel hit OOM.
  **Rule 31.**
- Multi-agent HANDOVER collisions: 5+ rewrites in one session,
  4 merge conflicts, parallel branches submitting same probe.
  **Rule 32** (session-start fetch).
- Premature day-close on PI signal ambiguity; "go" interpreted as
  "go submit." Both lessons now in `do-and-dont.md`.
- 25+ acronyms in CLAUDE.md without inline expansion; PI couldn't audit
  the file on first read. **Rule 0** (this session) addresses it.

## Killed-and-do-not-retry summary

For the deduplicated list, see `state/hypothesis-board.md`. For the
full enumeration, see `state/mechanism-ledger.md`. Highlights:

- Target reformulation single-add (all variants leaky).
- Multi-level 4-tier per-segment stacker.
- Day-16 virgin-axes round (11 of 11 null/falsified).
- TabPFN v2.5/v2.6.
- 16+ field factorisation machines.
- Drop-GBDT pool refactor.
- Simple K=21 blends.
- α-calibrated τ resweep.
- Multi-target NN.
- Masked-column self-prediction.
- Twin-pool 2-meta blending.
- Yao/Vehtari covariance-modulated per-segment stacker.
