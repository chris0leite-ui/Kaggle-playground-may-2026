# 2026-05-06 — What transfers: the research loop, not the models

> PI's answer to [Q4](../questions/2026-05-06-grilling-round-1.md). Synthesis,
> not full transcript. Original phrasing preserved where it carries weight.

## The headline

**The research loop is what transfers — not the models, not the features.**
Across Kaggle tasks (regression / classification / time-series), the model
class changes; the loop doesn't. The same loop also transfers to the day
job.

## What "the loop" contains

- Get a benchmark (LB, scorecard, paper trial, whatever).
- Understand the problem before touching it.
- Find key components, identify the most important ones.
- Pick the experiments that *clarify* — not just the experiments that
  produce a number.
- Articulate hypotheses; specify how each will be evaluated.
- Run, log, document — including assumptions made along the way.
- Hand off context so others (humans or agents) can build on it.

> [Claude note] This is essentially the 7-step McKinsey "bulletproof"
> process applied at experiment granularity. Mapping to existing
> framework: step 5/6 ↔ Rule 19 BOTE/gate harness; step 3 ↔ Rule 18
> ISSUES.md; step 7 ↔ `audit/*.md` discipline. The KB-relevant gap is
> step 1-2 (understand/frame) — least operationalized today.

## Four secondary claims layered on top

### 1. Documentation & context-handoff is the bottleneck

Per-agent context is narrow; you can spawn many agents. So the limit on
the system isn't compute — it's **how cleanly findings move between
agents (and back to PI)**. This reframes the project: the Kaggle comp is
partly an excuse to stress-test handoff mechanisms.

### 2. PI as focus-setter

Even with unlimited agent capacity, focus matters. PI's role is not to
direct each experiment — it's to **encode** instructions / scaffolding /
priorities such that agents land on the right focus themselves and
don't waste a week of compute on probes with negligible expected value.

> [Claude note] The texture of those instructions is exactly what's
> under-specified. CLAUDE.md rules 1-19 are one attempt; the BOTE
> harness with `FAMILY_PRIORS` is another; the issue tree is a third.
> We don't yet have a clean mental model of *what kind of instruction*
> works for *what kind of agent decision*. Candidate `concepts/` entry.

### 3. Incidental learning beats short-term LB

Don't optimize for tomorrow's leaderboard. Optimize so that **week 3-4
of the comp** finds you in a strong position. This implies:

- Some experiments today should be foundation-builders, not score-chasers.
- "Failed" experiments that *teach* (rule out a family, calibrate a
  prior, expose a leakage signature) are first-class outputs.
- The framework should reward and surface those, not bury them.

> [Claude note] CLAUDE.md Rule 19(d) ("rule-out is a valid result") and
> Rule 14 (strategy-critic-loop) gesture at this, but neither
> *positively incentivises* foundation work. Worth pushing on.

### 4. Same principle transfers to the day job

In the electricity-price work, the same explore-and-invest discipline
applies — except the feedback loop is much longer (see [F3 follow-up](../questions/2026-05-06-grilling-round-2.md)).

## What PI's answer commits them to

- The KB's primary product is **distilled loop discipline**, not model
  recipes.
- Bad probes are bad even if they "work" by accident, when they don't
  build the foundation. → Need to log *why we chose* a probe, not only
  *what it returned*.
- Kaggle competitions are **deliberately a sequence**, with the second
  competition supposed to be cheaper/sharper because of the first.
  Implies a meta-loop above the per-comp loop. Not yet articulated.

## Where this answer is still soft

(See [grilling round 2](../questions/2026-05-06-grilling-round-2.md) for
the next probes.)

- Concrete instances where instructions actually focused an agent —
  good and bad.
- A concrete week-3 foundation experiment for *this* comp.
- The day-job feedback-loop length, which determines whether the same
  loop runs unmodified.
