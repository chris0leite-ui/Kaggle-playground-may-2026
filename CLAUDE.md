# CLAUDE.md — Predicting F1 Pit Stops (playground-series-s6e5)

Rules + pointers index. Detailed state lives in `state/`,
`HANDOVER.md`, and the references at the bottom. War stories /
origin notes per rule live in `rules-history.md`. Truth lives on
`origin/main`; session-start git fetch is Rule 32.

## Rule 0 — Communicate with the PI in plain English

No abbreviations the PI hasn't used. No letter-number experiment
codes in chat (E1, K=27, τ=100k) — describe what each thing does.
Glossary is agent-to-agent reference, not something PI should
have to look up mid-conversation. Define any technical term inline
on first use.

## Rules — families

### Submission discipline

- **R1.** Every `kaggle competitions submit` is single-shot, PI-approved,
  never wrapped in retry / `until` / `while` / `for`.
- **R12.** Spend the full daily quota (`comp-context.md:
  submission_budget`). Slots are calibration probes; unused slots
  by Kaggle UTC midnight are forfeit.
- **R27.** Pre-submit `scripts/pre_submit_diff.py` against the
  previous submit. Spearman > 0.999 → REGRESSION_RISK or TIE_ZONE
  (band table in `state/current.md`); abort or PI-authorise override.

### Leakage & validation

- **R24.** ANY feature derived from labels via groupby aggregation
  MUST be re-fit per CV fold using training rows only. Diagnostic:
  80/20 holdout in <10 min CPU.
- **R25.** Transductive features (fit on combined train+test) need
  adversarial-validation. If AV-AUC > 0.55, fit on train only.
  (s6e5 AV-AUC = 0.502.)
- **R33.** Inner-CV-validate any post-hoc OOF transformation
  (isotonic / Platt / per-group rescale). Never trust in-sample lift.
- **R3.** Pre-LB-probe 4-gate filter: G1 standalone OOF / G2 blend
  lift / G3 net rare-class-flip ≥0.5 / G4 direction asymmetry.

### Compute discipline (operational detail: `agent-ops.md`)

- **R2.** Smoke at 1 fold / 50k rows. 1-hour cap is on single-fold
  wall time, not extrapolated 5-fold. Kill kernels that preprocess
  ≥30 min without fold output.
- **R9.** ≤150 lines per agent-loaded file. Subagents load slices.
- **R11.** Routing — Haiku read-only; Sonnet default; Opus hard.
- **R13.** Kaggle GPU part of compute budget. Port any 5-fold >1h
  CPU before declaring "not cost-justified."
- **R28/R30/R31.** Subagents must not run Python >5 min (use main
  thread). Torch kernels use GpuT4x2; P100 for CatBoost-GPU /
  LightGBM-GPU only. ≤2 concurrent CPU-heavy jobs. Full detail:
  `agent-ops.md`.

### Experimentation harness

- **R6.** Heuristic before heavy compute. Closed-form rule before
  Optuna / GPU / 5-fold-bagging.
- **R16.** 6-question pre-flight before any candidate ≥10 min.
  **Q6 (forced): does training objective match the row-AUC metric?**
  Q6 unanswered = SKIP.
- **R18.** Claim an unclaimed `open` leaf in `ISSUES.md` before any
  ≥10-min probe.
- **R19.** `probe.py bote` before compute; `gate` after; `probe_min_meta.py`
  for stack-adds; calibration log `audit/decisions.jsonl`. Log nulls.
- **R20.** Single-model-first / kitchen-sink FE before stacking.
- **R21.** Family falsification needs ≥3 variants of the key hyperparameter.
- **R22.** Public-notebook scan on Day 1 (pre-baseline-gate item 8)
  and at every plateau.
- **R23.** Framework is scaffolding, not authorship. ≥1 slot per
  3-day cycle for free-form FE.
- **R14.** Strategy-critic-loop auto-fires at EOD, on OOF→LB drift
  ≥2 bp, before any new mechanism family, at 50% checkpoint, at plateau.
- **R32.** Session-start `git fetch origin && git log HEAD..origin/main
  && git diff HEAD..origin/main HANDOVER.md` BEFORE any compute.

### Strategic posture

- **R4.** Never give up; saturation is bounded. After every null,
  brainstorm 3 untried mechanisms. Locking is final-3-day-window only.
- **R5.** Keep CLAUDE.md lean. State in `state/`; friction in
  `audit/friction.md`.
- **R7.** Research before saturation. At 3 nulls / 5 same-LB sats /
  2 days no lift: trigger Research-loop (`loops.md`).
- **R29.** Same-session friction applies same-session.

### Process & communication

- **R8.** Settled-once facts in `comp-context.md`. Never re-ask.
- **R10.** Pull-style updates. No minute-level chatter; on PI pull,
  1-2 sentences.
- **R15/R17.** "Handover" → read `HANDOVER.md`. "Prepare handover"
  → `WRAPUP.md` section B. "Wrap up" → `WRAPUP.md` section A.
- **R26.** PI is read+strategy. Every BOTE asks (i) Q6 metric
  alignment, (ii) precedent we're pricing against. Once-per-session
  devil's-advocate ritual.
- **R34.** Descriptive names in prose. Letter-number codes (d13,
  qBA, K=27) stay in artifact filenames; new prose says what each
  thing IS. Glossary maps codes ↔ descriptions.
- **R35/R36.** PI thoughts append-only at
  `knowledge-base/thoughts/`. Session-end: add ≥1 thought, log open
  questions in `questions/`, surface persistent flags in `flags/`.

## Defaults from prior-comp postmortem

- **R1d** Two-anchor OOF. (s6e5: GroupKF dropped Day-3+; Strat is LB proxy.)
- **R2d** Final selection: PRIMARY = best public; HEDGE = best OOF
  regressed ≤30 bp on public.
- **R5d** Final-window OOF-best regression probe (last 3 days).
- **R7d** Override-mechanism rules. Flip count <200 → HEDGE only;
  >200 needs explicit PI sign-off.
- **R8d** End-of-comp: log final percentile to
  `~/.claude/skills/kaggle-comp/improvements.md`.

## Pre-submit checklist (tick all before `kaggle competitions submit`)

R3 4-gate PASS · minimal-input meta beats anchor · R27 ρ-band check
· R24 fold-safe verified · R25 AV check · R1 PI-approved single-shot
· not already in `kaggle competitions submissions`.

## Day-end checklist (tick before EOD wrap)

R7 Research-loop run if plateau · R26 devil's-advocate rotated
· `state/current.md` rewritten if PRIMARY changed · `audit/friction.md`
< 150 lines · `audit/YYYY-MM-DD-day-N-wrap.md` written · HANDOVER.md
rewritten with single-current state.

## Pointers

State (mutable; rewrite each session, don't append): `state/current.md`,
`state/calibration-ladder.md`, `state/hypothesis-board.md`,
`state/mechanism-ledger.md`.

Process (read on trigger): `SETUP.md`, `HANDOVER.md`, `WRAPUP.md`,
`ISSUES.md`, `comp-context.md`, `glossary.md`, `rules-history.md`,
`audit/INDEX.md`, `audit/friction.md`,
`audit/friction-archive.md` (1,450 lines; not read by default),
`knowledge-base/` (PI second-brain; permanent), `templates/`.

Skill: `.claude/skills/kaggle-comp/` — SKILL.md, guardrails.md,
loops.md, day-loop.md, experiment-loop.md, agent-ops.md,
pre-baseline-gate.md, strategy-critic.md, self-improvement.md,
kickoff-runbook.md, problem-solving.md, examples/, templates/.
Also `.claude/skills/postmortem/`.

Scripts: `probe.py` (R19), `probe_min_meta.py`, `pre_submit_diff.py`
(R27), `research_seed.py`, `lr_diag_e1_svd.py` (eff-rank), smoke +
setup.
