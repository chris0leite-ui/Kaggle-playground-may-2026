# Loops — router

Six loops total. Day-loop wraps Experiment-loop. Calibration /
Strategy-critic / Research / Weekly trigger by event. The two largest
specs (Day, Experiment) live in their own slice files; the small
event-triggered loops are inline below.

## Day-loop

**Trigger**: every session start.
**Stop**: 10/10 submissions used, OR PI declares EOD.
**Summary**: load state → pick experiment (re-rank by learning-per-slot)
→ run Experiment-loop → on LB result update calibration ladder + loop
→ at EOD auto-fire steps 5-7 (audit + Strategy-critic + friction +
HANDOVER.md rewrite + commit/push/merge).

→ Full spec: `day-loop.md`. Includes day-boundary definition, EOD
auto-trigger recognition, and Day-loop-specific anti-patterns.

## Experiment-loop

**Trigger**: an experiment is selected (Day-loop step 2).
**Stop**: gate fails, OR PI declines, OR LB result lands.
**Summary**: heuristic baseline → smoke (50k, 5min cap) → 1-fold
time-probe → 5-fold production → 4-gate filter (G1-G4) →
minimal-input meta sanity check → reviewer audit → ask PI to submit.
1-hour cap applies to single-fold wall time, not 5-fold projection;
heavy mechanisms go to Kaggle GPU per Rule 13.

→ Full spec: `experiment-loop.md`. Includes the 4-gate filter detail,
the L1-coef prune addendum (Day-3 m5h finding), and the common
Experiment-loop failure modes distilled from `audit/friction.md`.

## Calibration-loop

**Trigger**: every 5 LB submissions, OR after any negative-gap entry
> 5bp (LB above OOF), OR after any leakage incident.

```
1. Refresh calibration ladder (Haiku): parse all (OOF, LB) pairs,
   compute per-mechanism-family gap.
2. Drift check: if any family's gap moves > 5bp from its trailing
   average, flag in CLAUDE.md and pause new submissions in that
   family.
3. Refit blend weights if applicable (Sonnet) + re-run minimal-input
   meta on the refit.
4. Commit calibration_ladder.md.
```

## Strategy-critic-loop

**Trigger** (auto-fire): end-of-day audit, OOF→LB gap drift ≥2bp on
consecutive submits in the same family, before adding a new mechanism
family, mid-comp 50%-checkpoint, OR plateau (in which case it runs
BEFORE Research).

```
1. Per-segment OOF-AUC failure map (Race / Stint / TyreLife / Year)
2. Probability calibration (Brier, ECE, reliability diagram)
3. Model-disagreement localization (residual-difficulty rows)
4. Unexploited structural-finding scout (sequence FE, etc.)
5. Headroom math vs realistic-discount H-list lift
→ emit audit/YYYY-MM-DD-strategy-critique.md and re-rank H-list
```

→ Full spec: `strategy-critic.md`. Strategy-critic interrogates OUR
data; Research-loop scouts EXTERNAL writeups. Orthogonal — at
plateau, both fire, critic first.

## Research-loop

**Trigger** (mandatory): 3 consecutive nulls, OR 5 saturation events
at the same LB, OR 2 days without LB lift.

The agent will want to skip this. The human PI should enforce it.

### Step 1 — Scout external sources (parallel, ≤200 words each)

Dispatch three `general-purpose` research agents in one message,
each writes its artifact to disk and returns a ≤200-word summary:

- **Notebooks agent** — `kaggle kernels list -c <slug> --sort-by
  voteCount` + WebFetch the top 5; for each, extract OOF / LB,
  FE list, model class, any leakage / group-CV warnings. Write
  to `audit/research/YYYY-MM-DD-notebooks.md`.
- **Prior-comp agent** — WebSearch 2 prior tabular Playground
  postmortems with the same metric AND similar class imbalance.
  Pull mechanism families that worked. Write to
  `audit/research/YYYY-MM-DD-prior-comp.md`.
- **Domain agent** — WebSearch for the comp's real-world
  decision drivers, citing ≥3 sources (academic / industry /
  prior writeups). Write to
  `audit/research/YYYY-MM-DD-domain.md`.

### Step 2 — Persona rotation (Opus)

Invoke the ML Researcher persona (see `personas.md`) on the three
agent artifacts + current `state/hypothesis-board.md`. Return 5
candidate mechanisms.

### Step 3 — Dedup against ledger (MANDATORY before queueing)

For EACH candidate, before adding to the experiment queue:

```
grep -l "<mechanism keyword>" state/mechanism-ledger.md \
                              state/hypothesis-board.md \
                              audit/friction.md \
                              audit/friction-archive.md \
                              scripts/fe_picks_*.py 2>/dev/null
```

If any grep hits, the candidate is NOT untried. Either reframe it
(narrower scope, different variant per Rule 21) or drop it. This
closes the recurring `research-scan-duplicate-mechanism-claim`
friction (2026-05-08 PM: Frontiers AI peer-effect features were
proposed as "untried" when `RankSortedGaps` already implements them
and they were nulled in Phase 1 smoke).

### Step 4 — Candidate template (one per surviving mechanism)

Append to `audit/research/YYYY-MM-DD-research.md`:

```yaml
- name: <descriptive>             # Rule 34 — no letter-number codes
  source: <citation URL or DOI>
  dataset_analogy: <prior comp / paper> — <metric, class balance, row count>
  ledger_grep: NO MATCH            # paste the actual grep output
  mechanism_class: <FE | model | meta | calibration | external | sequence>
  predicted_standalone_oof_lift: <bp range>
  predicted_meta_add_lift: <bp range>     # given current K-pool eff-rank
  cost_to_test:
    cpu_min: <int>
    gpu_min: <int>
    submission_slots: 0..1
  q6_metric_alignment: <one sentence — why training objective matches the row-AUC metric>
  kill_criterion: <e.g., "smoke OOF below baseline at 50k rows">
```

### Step 5 — Rank and emit

Rank by `predicted_meta_add_lift × probability_real / cost_to_test`.
Top 3 to experiment queue. Emit
`audit/research/YYYY-MM-DD-research.md` with citations and the
filled candidate templates.

### Anti-patterns (this loop)

- ❌ Citing a "research-backed" mechanism without grepping the
  ledger first.
- ❌ Returning candidates that don't pass Q6 metric-alignment.
- ❌ Letting a research-thin agent finding count as cleared.
  Re-spawn with a sharper prompt or surface the gap.
- ❌ Stacking the loop's output candidates without first running
  the heuristic baseline (Experiment-loop step 1).

## Weekly-loop

**Trigger**: every 7 days, plus the start of the final 3-day window.

```
1. Re-read CLAUDE.md ⚠️ rules in full.
2. Audit ceiling thesis: if any session claimed "structural ceiling",
   trigger Research-loop now.
3. Persona rotation: rotate at least one on a stuck problem.
4. Update REPORT.md with the week's results.
5. Submission-budget audit: did we use 10/day? if not, why?
6. Friction distillation: scan audit/friction.md, find tags with
   ≥3 entries, edit guardrails / personas / examples in the skill
   itself. Reset friction.md (archive prior week).
   See self-improvement.md.
7. Commit a 5-line weekly summary to audit/.
```

## Loop interaction

Day-loop wraps Experiment-loop. Calibration-loop is triggered by
submission count or leakage drift. Strategy-critic-loop fires at EOD
+ four other event triggers (Rule 14). Research-loop triggers on
plateau detection (mandatory). Weekly-loop runs across day boundaries.

The agent's default failure mode is to stay in Experiment-loop,
ignoring Strategy-critic and Research-loop triggers. This is what
the PI watches for.
