# Kickoff runbook — agent step-by-step

User invokes by saying "let's do the kickoff" / "start a new comp"
(see SKILL.md trigger). **Conversational free-form**: each numbered
step is one chat turn (≤4 sentences) followed by a wait for PI
reply, OR one Bash batch from [kickoff-bash.md](kickoff-bash.md).

**Hard rule**: never invoke `kaggle competitions submit` without the
explicit Q6 ask-PI gate. Single-shot only.

## Sequence

### Pre-flight (silent unless missing)

Run `pre-flight` from kickoff-bash.md. If `MISSING_TOKEN`: ask PI
in chat for the token (single-shot, redact in logs). If network
fails: ask PI for fix path; do NOT loop.

### Q1 — chat

> "Which Kaggle competition slug? (e.g., `playground-series-s7e1`,
> visible in the comp URL after `/competitions/`)"

Validate with `kaggle competitions view <slug>`; if errors, re-ask.

### Q2 — chat

> "Where should I create the comp directory? Default
> `~/projects/<slug>/`."

### Bash batch A — scaffold

Run `batch-A-scaffold` from kickoff-bash.md. Copies templates,
substitutes `{{COMP_SLUG}}`, `git init`.

### Bash batch B — populate comp-context.md from API

Run `batch-B-context` from kickoff-bash.md. Pulls from
`kaggle competitions view` + `leaderboard --download`.

### Q3 — chat (batch confirm)

> "comp-context.md auto-populated from the Kaggle API. Here it is —
> anything to fix or add? [paste populated YAML]"

PI replies any format. Apply edits, write final `comp-context.md`.

### Q3.5 — chat (problem statement, 1-turn)

PI fills the worksheet from [problem-solving.md](problem-solving.md):
L1/L2/L3 questions, criteria, constraints, boundary. Write the
answer to `audit/<date>-problem-statement.md`. Step 1 of the
7-step framework. Re-read at every plateau.

### Q4 — chat (4 strategic questions in one batch)

> "Strategic questions — answer in any format, one paragraph fine:
>
> (a) **LB stability** — stable / per-row-seeded / unknown?
> (b) **External data** — use / skip / depends_on_rules?
> (c) **Time budget** — hours/day × days?
> (d) **Compute** — CPU only / Kaggle GPU / local GPU?"

Extract free-form answers into structured fields; write to
`comp-context.md`. Show PI a 2-line interpretation summary for
one-line confirm.

### Bash batch C — download data + EDA

Run `batch-C-data` from kickoff-bash.md. Downloads data, extracts
target_col + id_col from sample_submission.csv header, runs EDA.

If multi-output: ask PI which target is primary; default to first
non-id column.

### Q5 — chat (EDA summary)

> "EDA done. Top features: `<top-3 from eda-summary.md>`.
> Anything to flag for hand engineering before the baseline?
> (Default: skip FE for the baseline; we add FE only after the DGP
> is understood — guardrail #6.)"

Capture any FE notes in `audit/<date>-day-1-kickoff.md`.

### Pre-baseline understanding gate (mandatory)

Full spec: [pre-baseline-gate.md](pre-baseline-gate.md). 7 items;
PI signs off before Bash batch D. Spawn 3 `general-purpose`
research agents in parallel (web / domain / metric); local Bash
for schema / target-rate / group-keys.

### Q5b — chat (gate sign-off)

> "Gate done: brief / prior_art / domain_notes / metric_notes /
> schema / target-rate / group-keys. Cleared? [yes / fix what]"

Wait for explicit "cleared". Friction-log thin output
(`audit/friction.md`, `tag: research-thin`).

### Bash batch D — baseline LGBM

Run `batch-D-baseline` from kickoff-bash.md. Reads OOF + fold-std
from `scripts/artifacts/baseline_lgbm_results.json`.

If baseline fails: 1-sentence summary, ask PI for fix path. Do NOT
loop.

### Q6 — chat (ASK PI BEFORE SUBMIT — critical)

> "Baseline OOF=`$OOF`, fold-std=`$STD`. Submitting calibrates the
> OOF→public-LB gap (Day-1's whole point). 1 of `$DAILY` slots
> today. Submit `submission_baseline_lgbm.csv`?
> [yes / no / show me first]"

**Wait for explicit PI yes.** If no: skip submit, log in audit. If
"show me first": print first 5 + last 5 rows of the CSV; re-ask.

### Bash D2 (only on PI yes) — single-shot submit

Run `batch-D2-submit` from kickoff-bash.md. **Never wrap in a loop.**
If kaggle command fails, surface the error verbatim, ask whether to
retry. PI decides.

### Bash batch D3 — Day-1 learning probes (mandatory)

After the baseline LB result lands, run THREE cheap probes that
have retroactively been the highest-EV moves in past comps. Each
is <30 min total. The Day-1 audit MUST report all three.

**D3.1 — Simple-LR ceiling probe (~30 s CPU)**

```python
KBins(20, quantile, onehot) on every numeric +
OneHotEncoder on every cat → LogisticRegression(C=1, solver='liblinear')
```

Closes 80-90% of the GBDT-vs-`lr_raw` gap on Playground tabular
comps. Then build the **mega LR** (~8 min CPU): all FE families
concatenated, same LR head. Gap from mega-LR to single-GBDT tells
you whether stacking is necessary (>100 bp gap → yes; <30 bp →
probably not). See `examples/fe-recipe-simple-lr.md`.

**D3.2 — Pool eff-rank diagnostic (~2 min CPU after ≥4 bases exist)**

SVD on the base-prediction matrix; report `entropy(singular_values)`
as effective rank. **If logit eff-rank stalls below `log2(K) + 1`,
the pool is rank-collapsed regardless of nominal K.** Low Spearman
ρ to PRIMARY is necessary but not sufficient for amp-eligibility
(s6e5 evidence: ρ=0.41 still absorbed at K=10+1). Code template:
`scripts/lr_diag_e1_svd.py`. If you don't have 4 bases yet on Day
1, run this on Day 2 after the first 2-3 stack-adds. Skipping it
cost s6e5 ~half a session of dead-axis exploration.

**D3.3 — Strict 80/20 holdout for any new FE family (~10 min CPU)**

StratifiedKFold with an INDEPENDENT seed; fold-0 as 20% holdout;
fit FE + any inner-CV target encoding on the 80% only; train + eval
on the 20%. If `holdout_AUC` < `5-fold OOF` by ≥ 10 bp, leak
present — debug before any LB submit. Rules 24/25 origin lesson;
s6e5 Day-17 caught the `make_features_A` 88-100% leakage trio.

### Bash batch E — Day-1 audit + Day-2 queue

Run `batch-E-audit` from kickoff-bash.md. Computes calibration
verdict, writes `audit/<date>-day-1-kickoff.md`, updates CLAUDE.md
current-state, commits.

**Day-1 audit MUST include** (in addition to OOF→LB gap):

- D3.1 simple-LR + mega-LR OOF, gap to GBDT baseline.
- D3.2 pool eff-rank (or "deferred to Day 2 — too few bases").
- D3.3 80/20 holdout for any FE family used in the baseline.
- Item 8 public-notebook scan: list of top-5 OOFs, our gap to each.
- Item 11 single-model OOF target vs achieved.

These five gates collectively prevent the four recurring Day-1
miscalibrations: (a) jumping to stacking too early, (b) building
on a leaky FE, (c) over-investing in a rank-collapsed pool,
(d) ignoring a public-notebook recipe for 2+ weeks.

### Final chat turn — handoff

> "Day 1 done. OOF→LB gap = `$GAP` (`$VERDICT`). Day 2 queue
> in `audit/...-day-1-kickoff.md`; CLAUDE.md current-state
> updated. Next session, say 'continue' or pick from the queue.
> Friction logged: 0 events."

End the session. Do NOT auto-start Day 2.

## Style rules

- Every chat turn ≤4 sentences. Wait for PI reply.
- Bash batches are silent unless they fail. On failure: 1-sentence
  summary + ask PI.
- Never loop `kaggle competitions submit`.
- Never auto-confirm Q6. PI must explicitly say yes.
- Model routing: pre-flight + batches A/C/D2/E = Haiku
  (deterministic). Q&A + Q4 extraction + Day-2 queue = Sonnet.
- Log every avoidable friction (broken CLI, ambiguous Q4 reply,
  failed sed substitution) to `audit/friction.md` per
  `self-improvement.md`.
