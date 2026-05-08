# CLAUDE.md — Predicting F1 Pit Stops (playground-series-s6e5)

This file is the rules + pointers index. Detailed state lives in
`state/`, `HANDOVER.md`, and the references at the bottom of this file.

Truth lives on `origin/main`. Session-start git fetch is Rule 32.

## Rule 0 — How to communicate with the PI

**Plain English, every time. No abbreviations the PI hasn't already used.**
No letter-number experiment codes (E1, F2, h1d, K=27, τ=100k) in chat —
describe what each thing does. The glossary is for agent-to-agent
reference and for reading older audit notes; it is **not** something the
PI should ever have to look up mid-conversation.

If you have to introduce a technical term, define it inline on first use.
If you find yourself reaching for an acronym, that's a smell.

## Operating rules — concise (full origin notes in `rules-history.md`)

1. **Submission discipline.** Every `kaggle competitions submit` is
   single-shot, explicitly approved by the PI. No retry/until/while loops.
2. **Smoke + 1-fold time-probe + 1h GPU cap.** Smoke at 1 fold / 50k rows;
   if 5-fold projection ≥ 1 hour, shrink. Kill any kernel that
   pre-processes ≥30 min without fold output.
3. **4-gate leakage filter pre-LB-probe** (G1 OOF clears anchor; G2 blend
   lift; G3 net rare-class flip ratio ≥0.5; G4 direction asymmetry).
4. **Never give up; saturation is bounded.** After every null, brainstorm
   3 untried mechanisms. Locking is for the final 3-day window only.
5. **Keep CLAUDE.md lean.** Hits the 150-line cap (Rule 9); anything
   that grows beyond rules + pointers goes in a dedicated file.
6. **Heuristics before heavy compute.** Closed-form rule before Optuna /
   GPU / 5-fold-bagging.
7. **Research before saturation.** At 3 nulls / 5 same-LB saturations /
   2 days no lift: web search, 2 prior-comp writeups, 5 untried
   mechanisms with citations.
8. **Settled-once facts** in `comp-context.md`. Never re-ask.
9. **~160-line guideline on session-read docs** (`CLAUDE.md`,
   `HANDOVER.md`, `state/current.md`, `audit/friction.md`). Reference
   docs (glossary, rules-history, mechanism-ledger, audit archive,
   postmortems) are exempt — pulled on demand, not read by default.
10. **Pull-style updates.** No proactive minute-level chatter; on PI pull,
    1-2 sentences with the latest fact.
11. **Model routing.** Haiku read-only; Sonnet default; Opus hard. 10/day.
12. **Spend the full 10/day submission budget.** Submissions are
    calibration probes; the OOF→LB gap per mechanism family is data.
13. **Kaggle GPU is part of compute budget.** Local CPU-only; Kaggle
    notebooks are the GPU path. Port any 5-fold > 1h CPU before declaring
    "not cost-justified."
14. **Strategy-critic-loop fires automatically** at end-of-day, on OOF→LB
    drift ≥2 bp on consecutive submits, before any new mechanism family,
    at 50% checkpoint, and at any plateau.
15. **Handover protocol.** PI says "handover" → read `HANDOVER.md`.
    PI says "prepare handover" → follow `WRAPUP.md` section B.
16. **6-question pre-flight check** before any candidate ≥10 min CPU/GPU.
    Q1 already-explored? Q2 rank-lock-vulnerable family? Q3-Q5 predict
    standalone OOF + ρ + cite precedent. **Q6 does training objective
    match the row-AUC metric?** Q6 unanswered = forced SKIP.
17. **Wrap-up triggers.** "Wrap up" → `WRAPUP.md` section A; "prepare
    handover" → both sections.
18. **Issue-tree claim before compute.** Before any probe ≥10 min, claim
    an unclaimed `open` leaf in `ISSUES.md`.
19. **Experimentation harness.** `scripts/probe.py bote` before compute;
    `gate` after artifacts; `probe_min_meta.py` for stack-add gates;
    `research_seed.py` for new families. Calibration log:
    `audit/decisions.jsonl`. Document nulls too.
20. **Single-model-first / kitchen-sink FE before stacking.** Build the
    feature factory and a single strong model first; stacking adds, it
    doesn't replace.
21. **Family falsification requires ≥3 variants** of the key
    hyperparameter.
22. **Public-notebook scan at every plateau.** Pull top 5 Kaggle notebooks
    (≥10 votes); list features + OOF + model class; build the gap.
23. **Framework is scaffolding, not authorship.** Discipline optimises
    HOW to evaluate, not WHAT. ≥1 slot per 3-day cycle for free-form FE.
24. **Fold-safe label-conditional aggregates.** ANY feature derived from
    labels via groupby aggregation MUST be re-fit per CV fold using
    training rows only. Diagnostic: 80/20 holdout test in <10 min CPU.
25. **Transductive features need adversarial-validation check.** Before
    any combined train+test transform, run AV; if AV-AUC > 0.55, fit on
    train only. (s6e5 AV-AUC = 0.502, so combined is safe.)
26. **PI interaction protocol.** PI is read+strategy, not keyboard.
    Every BOTE asks PI: (i) Q6 metric alignment, (ii) which precedent
    are we pricing this against? Once-per-session devil's-advocate
    ritual on the strongest current recommendation.
27. **Pre-submit prediction diff is mandatory.** `scripts/pre_submit_diff.py`
    against the previous submit. If Spearman > 0.999, abort — LB will tie.
28. **Subagent dispatch limits.** Don't dispatch general-purpose subagents
    for Python jobs > 5 min. Long-running compute launches from the main
    thread.
29. **Same-session friction must apply same-session.** When a friction is
    logged, immediately apply the fix to all in-flight invocations.
30. **GPU kernel template gate.** Any new torch / pytabkit kernel uses
    `"machine_shape": "GpuT4x2"`. P100 is for CatBoost-GPU and LightGBM-GPU
    only.
31. **Concurrent compute cap ≤2 CPU-heavy jobs.** Schedule cheap probes
    (<30 s) ahead of slow ones.
32. **Session-start git fetch.** `git fetch origin && git log
    HEAD..origin/main && git diff HEAD..origin/main HANDOVER.md` BEFORE
    any new compute.
33. **Inner-CV-validate post-hoc OOF transformations.** Isotonic / Platt /
    per-group rescaling; never trust the in-sample lift.
34. **Experiments use descriptive names in docs.** Letter-number codes
    may stay in artifact filenames and old audits, but new prose says
    what each thing IS. Glossary maps codes ↔ descriptions.
35. **PI thoughts are append-only.** Transcribe PI voice-dumps to
    `knowledge-base/thoughts/YYYY-MM-DD-slug.md`. Never overwrite,
    delete, or archive on cleanup. Folder is permanent.
36. **Session-end second-brain update.** Before wrap-up, add at least
    one entry to `knowledge-base/thoughts/`; log open questions in
    `questions/`; surface persistent flags in `flags/`.

## Defaults from prior-comp postmortem

- **R1** Two-anchor OOF. (s6e5: GroupKF dropped Day-3+; Strat is LB proxy.)
- **R2** Final selection along public-LB axis. PRIMARY = best public.
  HEDGE = best OOF that regressed ≤30 bp on public.
- **R5** Final-window OOF-best regression probe (mandatory in last 3 days).
- **R7** Override-mechanism rules. Flip count <200 → HEDGE only;
  >200 needs explicit PI sign-off.
- **R8** End-of-comp: log final percentile to
  `~/.claude/skills/kaggle-comp/improvements.md`.

## Pointers

State (mutable, refresh each session):
- `state/current.md` — current PRIMARY, LB ladder, submission count, axes status.
- `state/calibration-ladder.md` — OOF / LB anchor table for new-candidate sizing.
- `state/hypothesis-board.md` — open ideas, killed list, hedge ladder.
- `state/mechanism-ledger.md` — every mechanism family tried, with results.

Process docs (read once / on trigger):
- `SETUP.md` — onboarding checklist for a new comp (read on day 1 of a fresh repo).
- `HANDOVER.md` — next-session brief (Rule 15).
- `WRAPUP.md` — wrap-up + prepare-handover procedure (Rule 17).
- `ISSUES.md` — live problem decomposition / claim board (Rule 18).
- `comp-context.md` — settled-once facts (kickoff, schema, gate clearance).
- `glossary.md` — agent-to-agent reference for abbreviations and short-codes.
- `rules-history.md` — origin notes / war stories / friction tags per rule.
- `audit/INDEX.md` — map of audit/ subdirs.
- `audit/friction.md` — current friction summary (concise).
- `audit/friction-archive.md` — full historical friction (1,450 lines; do
  not read by default).

- `knowledge-base/` — PI second-brain (permanent; Rules 35-36).
  Subdirs: `thoughts/`, `concepts/`, `friction/`, `flags/`, `questions/`.
- `templates/` — copy-paste starters for a new competition repo.

Skills + scripts:
- `.claude/skills/postmortem/SKILL.md`, `.claude/skills/kaggle-comp/`.
- `scripts/`: `probe.py` (Rule 19), `probe_min_meta.py`,
  `pre_submit_diff.py` (Rule 27), `research_seed.py`, smoke + setup.
