# Rules history — origin notes per rule

The rules in `CLAUDE.md` are stated in concise form for readability. The
war stories that produced them, the friction tags, and the audit
pointers live here. Read this when you need to know **why** a rule
exists, not when you just need to **follow** it.

## Rule 0 — Plain-English communication with the PI

Origin: PI directive 2026-05-08, this session. Repeated friction-file
entry `tag: jargon-drift-without-glossary` (2026-05-06): "PI read
CLAUDE.md for the first time and did not know what BOTE stood for
despite it being load-bearing in Rule 19. ~25 acronyms appear in
CLAUDE.md without inline expansion."

## Rule 1 — Ask-first / no-loop on submissions

Origin: PI burned slots in early session when an agent interpreted "go"
on a multi-step plan as transitive to the submit step. Friction tag
`submit-without-confirmation`. Rule states: every `kaggle competitions
submit` is single-shot, explicitly approved, no retry/until/while/for.

## Rule 2 — Smoke + 1-fold time-probe + 1h GPU cap

Origin: Day-3 RealMLP kernel ran 175 minutes total without smoke gate;
TabPFN fold-0 ran 85 min on 150k rows. Friction tags
`rule2-smoke-skip-realmlp-day3`, `probe-extrapolation-drift`. Rule
states: smoke at 1 fold / 50k rows; if 5-fold projection ≥1h, shrink;
kill any kernel that pre-processes ≥30 min without fold output.

## Rule 3 — 4-gate leakage filter pre-LB-probe

Origin: `kaggle-comp` framework defaults. G1 standalone OOF clears
anchor; G2 blend lift; G3 net rare-class flip ratio ≥0.5; G4 direction
asymmetry; plus minimal-input-meta sanity check.

## Rule 4 — NEVER-GIVE-UP / saturation-is-bounded

Origin: framework default. After every null, brainstorm 3 untried
mechanisms. Locking is for the final 3-day window only.

## Rule 5 — Keep CLAUDE.md fresh / archive-on-bloat

Origin: CLAUDE.md exceeded 50k tokens twice; second compression
2026-05-06. The 2026-05-08 cleanup (this session) split state and
ladder out into `state/`.

## Rule 6 — Heuristics before heavy compute

Origin: framework default. Closed-form rule / threshold / hand-coded
baseline before Optuna / GPU / 5-fold-bagging.

## Rule 7 — Research before saturation

Origin: framework default. At 3 nulls / 5 saturations at the same LB
/ 2 days no lift: web search + 2 prior-comp writeups + 5 untried
mechanisms with citations, all before declaring ceiling.

## Rule 8 — Settled-once facts in `comp-context.md`

Origin: framework default; surfaces during kickoff. Never re-ask.

## Rule 9 — File-size cap ≤150 lines

Origin: framework default. `HANDOVER.md` repeatedly tried to embed
day-by-day commentary; cap forced compression.

## Rule 10 — Pull-style updates

Origin: PI directive on chat-spam. No proactive minute-level updates;
on PI pull, 1-2 sentences with the latest fact.

## Rule 11 — Model routing

Origin: framework default. Haiku read-only; Sonnet default; Opus hard.
10/day budget.

## Rule 12 — Spend the full submission budget

Origin: PI directive on calibration probes. Submissions are the
load-bearing OOF→LB calibration data.

## Rule 13 — Kaggle GPU is part of compute budget

Origin: framework default. Local CPU-only; Kaggle notebooks are the GPU
path. Port NN / deep-CB-depth≥8 / any 5-fold > 1h-CPU before declaring
"not cost-justified."

## Rule 14 — Strategy-critic-loop

Origin: framework default. End-of-day audit, on OOF→LB drift ≥2 bp on
consecutive submits, before adding a new mechanism family, at 50%
checkpoint, or at any plateau.

## Rule 15 — Handover protocol

Origin: framework default. PI says "handover" → read `HANDOVER.md`.
PI says "prepare handover" → follow `WRAPUP.md` section B.

## Rule 16 — New-candidate pre-flight (6-question check)

Origin (Q1-Q5): framework default. **Origin (Q6, metric alignment):**
Day-12 LambdaRank meta regressed −86 bp; AUC-pairwise XGB regressed
−451 bp on fold 0; Mast FM-2.6 reasoning-action-mismatch (arXiv
2503.13657). The five existing questions had passed all three pre-flight,
yet they all turned out to be metric-misaligned (group-rank objectives
on a row-AUC metric). Q6 forces explicit confirmation before BOTE.

Friction tag: `menu-overcrediting-redundant-mechanism` — Day-8
falsified T1.5/T1.3/T1.2 all of which passed research-agent EV ranking
but failed the 5-question check retroactively. The 6th question is now
asked first.

## Rule 17 — Wrap-up + handover triggers

Origin: framework default. PI says "wrap up" → `WRAPUP.md` section A.
PI says "prepare handover" → both sections.

## Rule 18 — Issue-tree claim before compute

Origin: framework default. Before any new probe consuming >10 min
CPU/GPU, claim an unclaimed `open` leaf in `ISSUES.md`. One open leaf
per branch.

## Rule 19 — Experimentation harness (BOTE-first / gate-after)

Origin: framework default + iterative refinement.

- Sub-rule (a) — `scripts/probe.py bote` before any candidate ≥10 min.
- Sub-rule (b) — `scripts/probe.py gate` for the uniform structured report.
- Sub-rule (c) — `scripts/probe_min_meta.py` for K=K_pool+N stack-add.
- Sub-rule (d) — Rule-out is a valid result; document nulls too.
- Sub-rule (e) — "Many small things" beats "one big bet."
- Sub-rule (f) — Calibration loop via `audit/decisions.jsonl`. Added
  2026-05-06; revised 2026-05-07 PM (sealed-prediction protocol
  deprecated, see Rule 26).
- Sub-rule (g) — Family kickoff seed via `scripts/research_seed.py`.
  Origin: MLE-STAR's web-retrieval seed accounts for most of its +47
  bp over AIDE on MLE-bench-Lite (arXiv 2506.15692).

## Rule 20 — Single-model-first / kitchen-sink FE before stacking

Origin: PI question Day-16 PM: "we ran 16 days of disciplined experiment
loops and never asked WHAT'S THE BEST SINGLE MODEL? We jumped to
stacking on Day 2." The diagnosis: agents treated FE additions as
"+1 feature to existing base" probes, never built a kitchen-sink
factory. Single LightGBM with Rozen's recipe matched the K=22+Path-B
PRIMARY OOF on Fold-1 alone.

Friction tag: `recipe-over-judgment`.

## Rule 21 — Family falsification requires ≥3 variants

Origin: Day-3 d3a `unified_te_2way_keys` tested ONE smoothing × ONE
2-way key, scored +0.1 bp NULL at meta-add, closed the entire
target-encoding family. The 3-way (Driver, Race, Year) at smoothing 20
was the comp's load-bearing single trick (~+200 bp standalone for any
single LightGBM). One null does not falsify a family.

Friction tag: `family-falsification-too-quick`.

## Rule 22 — Public-notebook scan at every plateau

Origin: Day 16 — Rozen's 0.95354 recipe sat at 19-72 votes the entire
comp without us pulling it. When we finally did, copying its features
produced the project's biggest single lift (+24 bp). Strengthens
Rule 7 with comp-specific recipe intelligence.

## Rule 23 — Framework is scaffolding, not authorship

Origin: 16-day plateau where every probe was a rank-locked stack-add
variant; the discipline was correct but insufficient. Discipline
optimises HOW to evaluate; it doesn't generate WHAT to evaluate.
Reserve ≥1 slot per 3-day cycle for free-form FE creativity uncoupled
from the existing pool.

## Rule 24 — Fold-safe label-conditional aggregates

Origin: Day-17 P1 v2 — `make_features_A` computed `compound_avg_life`,
`race_avg_pit_lap`, `dc_avg_stint_life` from
`df[df['PitNextLap']==1].groupby(...).mean()` on full train, then
merged the same lookup into both train and test. In CV-OOF, val rows
had their own labels included in the FS_A aggregates → OOF inflated
by ~500 bp (0.95128 vs honest holdout 0.94637). LB submitted at
LB 0.94107 (v1 standalone) and 0.94996 (K=2 LR with v2).

Friction tag: `target-construction-layer-leakage`. Audit pointer:
`audit/2026-05-06-target-reform-leakage-audit.md`.

## Rule 25 — Transductive features need AV check

Origin: PI Day-17 lesson on cross-comp generalisation discipline.
Frequency encoding, quantile binning, factorize maps, PCA/AE fit on
combined train+test can encode distributional structure that differs
between train/test or public/private LB. The standard check: train
a binary classifier to tell train rows from test rows; if AV-AUC ≈ 0.5,
combined-set FE is safe; AV-AUC > ~0.55, fit on train only. (s6e5
AV-AUC = 0.502.)

## Rule 26 — PI interaction protocol

Origin: `knowledge-base/concepts/agentic-kaggle-systems-comparison.md`
HITL section + non-coding-PI reframe (2026-05-06 chat).

- Sub-rule (a) — Sealed-prediction order **REMOVED 2026-05-07 PM**
  (Day-19 wrap-up postmortem). PI directive: "remove asking for the
  sealed prediction." Removal rationale: protocol added cognitive
  overhead for a non-coding PI without proportional calibration gain.
- Sub-rule (b) — Two required questions on every BOTE.
- Sub-rule (c) — Devil's-advocate ritual once per session.
- Sub-rule (d) — Daily deep-read.
- Sub-rule (e) — Override-rate captured by the postmortem skill.
  Origin: rubber-stamp anti-pattern (Sethserver; MLE-bench HITL
  literature).

## Rule 27 — Pre-submit prediction diff (mandatory)

Origin: Day-3 burned 3 slots (m5h, m5h2, m5j) all landing at the same
LB 0.94991. Post-hoc Spearman ≥0.9997 across all pairs; AUC depends
only on rank, so near-identical rank → identical LB. Friction tag:
`pre-submit-rank-diff-check`.

## Rule 28 — Subagent dispatch limits

Origin: Repeatedly hit through the project. Friction tags:
`subagent-non-execution`, `subagent-monitor-truncation`,
`subagent-friction-4-of-4-recurrence`, `subagent-shell-children-die-on-subagent-exit`.
Long-running compute via subagents loses artifacts when the subagent
SIGTERMs. Permanent fix: launch from main thread.

## Rule 29 — Same-session friction must apply same-session

Origin: `tag: lesson-not-applied` — logged tail-pipe-buffering friction
early in Day 3, then made the same mistake immediately after on the
next script. Logging is a learning artifact; applying is the rule.

## Rule 30 — GPU kernel template gate

Origin: Day-3 `kaggle-p100-torch-sm60-incompat` — RealMLP kernel
crashed on P100 with "no kernel image is available for this device"
because PyPI torch only supports sm_70+. Day-15
`kaggle-p100-fallback-reproduced-day15` — same friction hit again 12
days later despite the lesson being logged. Permanent fix: GpuT4x2 is
the default for any torch-based kernel; CatBoost-GPU and LightGBM-GPU
have their own CUDA kernels and can use P100.

## Rule 31 — Concurrent compute cap (≤2 CPU-heavy jobs)

Origin: Day-15 `cpu-contention-multi-probe-batch` — 7 LightGBM/NN probes
run simultaneously made each 4× slower; KD probe never finished. Day-18
`parallel-lgbm-3way-contention-oom` — 3 parallel LightGBM trains hit
OOM under sandbox CPU contention.

## Rule 32 — Session-start git fetch

Origin: Day-13 PM `cross-branch-converging-same-conclusion-redundant-submit`
— main and a parallel branch independently submitted the same
12-field FM-partition probe. Both landed LB 0.95032 TIE; the parallel
session merged origin/main only after experiments completed (when
preparing handover), so neither agent had visibility into the other's
work. Permanent fix: session-start ritual `git fetch origin && git log
HEAD..origin/main && git diff HANDOVER.md` before any base build.

## Rule 33 — Inner-CV any post-hoc OOF transformation

Origin: Day-3 `posthoc-isotonic-overfits-OOF` — per-(Year, Race) isotonic
fit on m5h OOF showed +24.6 bp Strat OOF lift in-sample; inner-CV gave
−10.9 bp. Per-Race alone: +11.8 in-sample, −5.3 inner-CV. Fitting
per-group isotonic on the same OOF rows you evaluate on is fitting
noise.

## Rule 34 — Experiments use descriptive names in docs

Origin: PI directive 2026-05-08 (this session). Letter-number codes
(E1, F2, G/H/I, K=27, h1d, v4) are load-bearing for back-references in
old audits and may stay in artifact filenames, but new prose must say
what each thing **is**. The glossary holds the mapping.

## Defaults from prior comp (R1-R8)

These are inherited from a prior playground postmortem; they live in
CLAUDE.md but their origin is the cross-comp `improvements.md` skill
file, not the s6e5 friction log.

- **R1** — Two-anchor OOF. (s6e5: GroupKF dropped Day-3+; U3 confirms
  i.i.d. test → Strat is LB proxy, gap +3.8 bp.)
- **R2** — Final selection along public-LB axis. PRIMARY = best public.
  HEDGE = best OOF that regressed ≤30 bp on public.
- **R5** — Final OOF-best regression probe in the final 3-day window.
- **R7** — Override-mechanism rules. Flip count <200 → HEDGE only;
  >200 needs explicit PI sign-off.
- **R8** — End-of-comp: log final percentile to
  `~/.claude/skills/kaggle-comp/improvements.md`.
