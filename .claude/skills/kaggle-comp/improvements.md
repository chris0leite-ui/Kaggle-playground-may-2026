# kaggle-comp skill — cross-comp improvements log

Edits promoted here when a friction pattern appears in 2+ comps, costs
> 1 LB slot, or required a human nag. See `self-improvement.md` for
the full distillation protocol.

Status markers: `[x]` applied · `[~]` superseded · `[ ]` open
(genuinely future work, not yet actionable).

---

## Applied 2026-05-14 (this arrears distillation)

- `[x]` **kickoff-runbook.md D3.1 — Simple-LR ceiling probe on Day 1.**
  `tag: lr-recipe-portable`. 30-second LR baseline (KBins-20 + OHE +
  C=1) closes 80-90% of GBDT-vs-`lr_raw` gap on Playground; mega-LR
  follow-up tells you whether stacking is necessary. s6e5 evidence:
  `lr_kbins20_ohe` 0.92038 / `lr_mega` 0.92776 / GBDT pool 0.95385.
  Recipe: `examples/fe-recipe-simple-lr.md`.

- `[x]` **kickoff-runbook.md D3.2 — Pool eff-rank diagnostic.**
  `tag: pool-rank-lock-at-logit-direction`. SVD eff-rank on the
  base-prediction matrix as soon as ≥4 bases exist. If logit
  eff-rank stalls below `log2(K) + 1`, pool is rank-collapsed
  regardless of nominal K. Low ρ is necessary but NOT sufficient
  for amp-eligibility (s6e5 ρ=0.41 still absorbed at K=10+1).
  Template: `scripts/lr_diag_e1_svd.py`.

- `[x]` **kickoff-runbook.md D3.3 — Strict 80/20 holdout for new FE.**
  `tag: cross-row-aggregates-survive-strict-fold-safe-audit`.
  Independent seed; fit FE + inner-CV TE on 80% only; eval on 20%.
  If holdout < 5-fold OOF by ≥10 bp, leak present — debug before LB
  submit. Catches the Day-17 `make_features_A` 88-100% collapse
  class in <10 min CPU.

- `[x]` **pre-baseline-gate.md items 8-11 promoted from pending.**
  `tag: eda-thin + public-notebook-scan-missing`. Item 8: top-5
  public-notebook scan Day-1 (s6e5 Rozen recipe sat 19-72 votes for
  16 days; copying its features = +24 bp = origin of R22). Item 9:
  high-card TE inventory (3-way Driver×Race×Year was the load-bearing
  trick). Item 10: domain-physics feature list. Item 11: single-model
  OOF target vs achieved gap.

- `[x]` **day-loop.md auto-triggers — public-notebook re-scan,
  pool eff-rank refresh, 80/20 holdout gate, friction-log scan,
  day-counter sanity.** Each is mechanical and has a single output
  artifact. Closes the gap where the loops fired only on PI prompting.

- `[x]` **agent-ops.md created — subagent / monitor / pgrep / CPU
  contention / GPU template consolidation.** Folds in
  `Monitor-discipline rule`, `bash-watcher-pgrep-self-match`,
  `concurrent-CPU-heavy-job cap`, `tail-pipe-buffering`,
  `bootstrap-token-name-mismatch`. Rules 28/30/31 now reference
  this file for operational detail.

- `[x]` **loops.md Research-loop fleshed with grep-ledger dedup and
  candidate template.** `tag: research-scan-duplicate-mechanism-claim`.
  Mandatory grep of `state/mechanism-ledger.md`,
  `state/hypothesis-board.md`, `audit/friction*.md`,
  `scripts/fe_picks_*.py` before any "untried" claim. Candidate
  template includes `ledger_grep:` field, Q6 metric alignment, and
  kill criterion.

- `[x]` **CLAUDE.md collapsed to 6 rule families + 2 checklists.**
  149 lines (under R9 cap). Rule numbers preserved for audit
  back-references. Pre-submit and Day-end checklists pulled out
  separately. State files now rewrite-on-change (not append).

- `[x]` **state/current.md + HANDOVER.md rewritten to single-current.**
  Prior versions in `audit/archive-2026-05-14-*.md`. Removed the
  duplicate K=4 and K=11+K=9 PRIMARY blocks that sat side-by-side.

- `[x]` **friction.md distilled from 544 → ≤150 lines.** This-week
  + recurring-process frictions retained; prior weeks moved to
  `audit/friction-archive.md`. Weekly distillation checklist added
  inline (Mondays).

- `[x]` **Submission-cap reconciled to `comp-context.md:
  submission_budget`.** Removed literal `5/day` from guardrails.md
  and do-and-dont.md; the actual cap is the comp-context source.

- `[x]` **`.claude/settings.json` created** with read-only Bash
  allowlist for git / kaggle status / `python -c`. `kaggle
  competitions submit` and `git push` routed through `ask` to
  enforce R1 + branch discipline.

## Applied 2026-05-18 (round-4+5 wrap; PI-ratified)

- `[x]` **guardrails.md §13 — multi-seed bag harness must verify
  test-prediction path is seed-variant.** `tag: pathb-bag-seed-invariant-test`.
  `run_pathb` in `scripts/build_K11_full_pathb.py:116-128` fits on
  FULL training data for test predictions; LR convex fit is
  seed-invariant given the same training set, so multi-seed bag
  test predictions are identical (ρ=1.0 confirmed empirically).
  Only the fold-OOF path is seed-variant. For true bagging:
  average per-fold-fit test predictions across seeds, vary base
  OOFs, or sub-sample-bootstrap the training rows. Cost evidence:
  2026-05-18 R5 5-seed Path-B bag wasted ~17 min CPU.

- `[x]` **experiment-loop.md — operator class is distinct from
  mechanism class.** `tag: operator-vs-mechanism-axis`. When
  retesting a previously-null mechanism, sweep at least two
  operator classes (LR-meta with alternate C, plus Path-B at one τ)
  before declaring the mechanism dead. Same OOF can show wildly
  different LB transfer under different operators. Cost evidence:
  2026-05-18 R5 K=11 + r4_segment_fe + r4_hmm_seq pool produced
  OOF 0.95446 under both LR-meta and Path-B; LB differed by +5 bp
  (Path-B 0.95387 vs LR-meta 0.95382). Closed mechanism-class
  results are operator-conditional unless verified across operators.

- `[x]` **bootstrap.sh — auto-pull `data/original/f1_strategy_dataset_v4.csv`
  if absent.** `tag: snapshot-missing-orig-dataset`. The slim-kNN
  builders (`scripts/dgp_v3/qA*.py`) reference this file via
  `DATA / "original/f1_strategy_dataset_v4.csv"` but the 2026-05-08
  artifact snapshot didn't include it. Without it, 5 of 6 builders
  in Phase A fail silently with FileNotFoundError. `bootstrap.sh`
  now checks and `kaggle datasets download -d
  aadigupta1601/f1-strategy-dataset-pit-stop-prediction` if absent.
  Cost evidence: 2026-05-18 R5 Phase A first-run failed for 5/6
  builders before manual pull.

- `[x]` **guardrails.md §14 — vectorise probe aggregators before
  full-data run.** `tag: cpu-contention-phase-c-starved`. Probe
  scripts with O(N) Python row-loops over training rows must be
  replaced with pandas/numpy vectorized equivalents before the
  full-data smoke. Smoke at 50k rows first; if smoke wall × (full
  N / 50k) > 5 min on contended CPU, vectorize before full. Cost
  evidence: 2026-05-18 R5 Phase C `probe_r5_graph_pit_pressure.py`
  smoke took 8+ min CPU on row-loop aggregator (vectorized variant
  ran in 0.04 s on same data).

## Applied 2026-05-18 (round-1+2+3 wrap; PI-ratified)

- `[x]` **bootstrap.sh — auto-isolate `KAGGLE_API_TOKEN` when it
  starts with `KGAT_`.** `tag: kggt-token-needs-isolated-auth`.
  Harness-issued KGAT_ token is incompatible with simultaneous
  KAGGLE_USERNAME + KAGGLE_KEY; the CLI tries basic-auth and 403s
  on private datasets. `bootstrap.sh` now detects the KGAT_ prefix
  and unsets the username/key pair. Cost evidence: ~10 min lost on
  2026-05-18 before the workaround was found by trial. Applies to
  every future harness-cloned session.

- `[x]` **experiment-loop.md — step 0 mandatory ledger-grep gate.**
  `tag: pre-probe-ledger-grep`. Before any ≥10-min probe, grep the
  candidate's mechanism name in `state/mechanism-ledger.md`,
  `state/hypothesis-board.md`, `audit/friction*.md`, and
  `scripts/fe_picks_*.py`. If a prior result with OOF Δ < +0.5 bp
  is recorded AND the anchor pool is unchanged, SKIP. Cost evidence
  (s6e5 2026-05-18): re-ran a2_2 mandatory-compound (60 min CPU)
  and a3_1 rank-sorted-gaps (55 min CPU) full 5-fold when
  hypothesis-board.md already recorded "K=4+1 +0.302 bp WEAK" and
  predicted ~+0.337 bp. 115 min CPU for zero new information.

- `[x]` **strategy-critic.md — Section 5 (headroom math) FIRST at
  plateau, not last.** `tag: headroom-math-decisive-not-final`.
  Headroom math is the cheapest decisive strategic input (~5 min)
  — it tells you whether the queue can mathematically reach the
  goal BEFORE compute is spent on it. If queue-midpoint-discounted
  lift < headroom, the strategic posture (lift-seeking vs
  variance-reduction vs hedge-prep) is already decided. Cost
  evidence (s6e5 2026-05-18): 15 mechanism probes over 3 rounds;
  Section 5 ran in Round 3 only. The queue-midpoint result
  (1.4 bp discounted vs 1.9 bp gap) would have pivoted Round-1
  compute to infrastructure (kNN-base rebuild) if it had fired
  first.

- `[x]` **personas.md — Senior ML Engineer (review mode) FIRST
  on initial "structural ceiling" claim.** `tag:
  senior-ml-first-on-ceiling-claim`. Senior persona surfaces
  methodological flaws (proxy substitution, anchor bias, missing
  C-sweep, ρ-band thresholds) cheaply; running it BEFORE
  brainstorm-class personas (10 Wild Options, Junior ML) prevents
  accumulating evidence against a claim that may already be wrong.
  Cost evidence (s6e5 2026-05-18): Round-2 ran 9 fresh probes
  against the K=4 proxy gate; Round-3 Senior ML in 5 min surfaced
  the proxy-substitution concern AND yielded the killer Pearson
  ρ=0.998 K=4↔K=27 residual-correlation datum. Round-2 work would
  have been re-prioritised if the Senior persona had fired first.

- `[x]` **day-loop.md — snapshot-freshness audit at session
  start.** `tag: artifact-snapshot-blocks-k11-gating`. After the
  Kaggle artifact dataset is pulled, verify the PRIMARY's
  underlying base OOFs are on disk via a `grep`-the-build-script
  shell snippet. If any are missing, the snapshot is stale
  relative to PRIMARY (1.8-3.5 bp gating gap in s6e5's case).
  Either rebuild the missing bases (~30-60 min each) or downgrade
  the candidate-screening threshold. Cost evidence (s6e5
  2026-05-18): 6 slim-kNN bases missing from 2026-05-08 snapshot;
  every Round-1+2 probe ran against K=4 proxy (3.5 bp behind
  PRIMARY) without the agent flagging.

## Applied previously (kept from earlier sessions)

- `[~]` **PI-protocol — Sealed-prediction protocol REMOVED (Day-19).**
  `tag: rule-26a-removed-by-pi-directive`. CLAUDE.md Rule 26a
  removed 2026-05-07 PM per PI verbatim "remove asking for the
  sealed prediction." Calibration loop continues with agent BOTE
  only (`pi_predicted_lb_bp` optional).

- `[x]` **comp-context.md — meta-arch redesign 9-variant tally
  (Day-19).** `tag: meta-arch-redesign-family-empirically-exhausted`.
  On K=27 pool, 9 Path-B variants exhausted across Days 14-19;
  Compound × Stint τ=100k is local optimum. Future variants need a
  fundamentally different segmentation axis or a different
  meta-objective.

- `[x]` **operational-tip — pgrep self-match.** `tag:
  bash-watcher-pgrep-self-match-zombie-loops`. Folded into
  `agent-ops.md`; preferred polling order is file-sentinel >
  anchored-pgrep > Monitor tool.

- `[x]` **Rule 0 — No-unexplained-abbreviations.** Already in
  CLAUDE.md. `tag: pi-comm-no-unexplained-abbreviations`. Load-bearing
  communication rule per PI verbatim 2026-05-07.

## Open (not yet actionable — needs cross-comp data or recipe work)

- `[ ]` **kickoff-runbook.md Q5b — data + task description in ≤10
  sentences.** `tag: settled-once`. After EDA, summarise each
  feature in domain terms, the prediction task in real-world terms,
  class balance → metric/threshold implication, top-3 features by
  F-score with domain reasoning. Will fold into Q5 chat once the
  template is finalised.

- `[ ]` **`recipes/path-b-prerun-base-routing-audit.md`.** `tag:
  pathb-amp-dead-when-pool-already-routes-segmentation-variable`.
  Pre-Path-B audit: enumerate K-pool bases; for each, check whether
  it natively routes by the candidate segmentation axis. If 1+
  routes, predict NULL ≥90% and require +0.5 bp gate. Saves
  estimated 30-60 min CPU per future Path-B candidate. Needs a
  separate recipe file.

- `[ ]` **`examples/cb-yekenot-transfer.md`.** `tag:
  yekenot-floor-count-kbins-fires-on-gbdt-too`. Document that
  yekenot items 2/3/4 (floor-cat, count-encoding, KBins) lift CB
  +20.7 bp on s6e5 despite being "NN-specific" in original framing.
  Promote when seen in a 2nd comp.

- `[ ]` **kickoff-runbook.md / day-loop.md — original-data
  row-augmentation default for synth-tabular comps with AV-AUC <
  0.55.** `tag: recipe-over-judgment`. Cross-comp evidence (s6e5 +
  irrigation-water); needs a 3rd comp before promoting to
  guardrail.

- `[ ]` **`kickoff-runbook.md` probe-template — persist OOF arrays
  for every gate probe.** `tag: gate-probe-oof-not-persisted`. Add
  unconditional `np.save oof_<slug>_strat.npy` and `test_<slug>_strat.npy`
  in script success path. Saves 30 min × multiple re-runs. Needs an
  edit to `scripts/probe.py` template, not just docs.

- `[ ]` **examples/ — keep top 3-5 public Kaggle notebooks under
  `external/kernels/` as reference.** `tag: recipe-over-judgment`.
  Build cross-comp recipe library. Promote durable patterns to
  `examples/` or `recipes/` at end-of-comp (R8d trigger).

## How the skill knows it's getting better

- **Friction-tag entropy decreasing.** Same tags comp-after-comp =
  not learning. New tags = new exposure (good).
- **Time-to-first-LB decreasing.** Day-1 setup time should shrink.
- **Submission-slot wastage → 0.** `retry-loop`, `re-recommend`,
  `gate-skip` should fall off once automated.
- **Plateau-break time decreasing.** Faster Research-loop via
  better persona prompts and pre-staged citation lists.
