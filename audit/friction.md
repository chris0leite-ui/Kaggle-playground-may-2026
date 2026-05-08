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

- Submission `kaggle datasets version` requires `~/.kaggle/kaggle.json`,
  but the harness only exposes `KAGGLE_API_TOKEN` in the environment;
  fix is to materialise `kaggle.json` from the env token before
  running dataset commands.
- The audit-ml-repo branch's history rewrite removed binary blobs from
  git (3.9 GB → 31 MB on origin) but leaves a `.git` of the same
  size locally until `git gc --prune=now --aggressive` runs (slow,
  ~20-30 min on this size).

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
