# 2026-05-06 — Process: from problem to framework

> Continuation of [2026-05-06-kickoff.md](./2026-05-06-kickoff.md). PI
> describes (a) the nature of the Kaggle problem, (b) how it differs from
> day-job work, (c) the desired research process, and (d) experimentation
> culture. Lightly cleaned; Claude annotations marked.

## The Kaggle task: synthetic data, low physical interpretability

- Kaggle Playground problems use **synthetically generated** data.
- The data usually originates from a real dataset, sometimes with
  physical observations and physical laws underneath.
- But the **data-generating process (DGP) layered on top destroys some
  of that meaning**. Some signal is recoverable, some isn't.
- Consequence: **little physical intuition** to lean on. Feature
  engineering driven by domain understanding is **hindered**.

## Contrast with the day job (electricity-price forecasting)

- Heavy in **physical laws** that can be respected and exploited.
- Feature engineering is **the** most important part of the pipeline.
- Complicated models (e.g. deep nets) have **not** outperformed
  simpler approaches like regressions [Claude: "counter regressions" —
  did you mean *kernel regressions*? *quantile regressions*? — flagging
  in `questions/`].
- Key features + clean data > model complexity. Intuition matters a lot.

> So Kaggle and the day job sit at **opposite ends of a spectrum**:
> intuition-rich physical → intuition-poor synthetic. The methods that
> transfer between them are not the feature-engineering tricks but
> the **research process itself**.

## Why agentic coding matters here

Hope: with coding agents we can
- **explore more** — test more models, more features, more variants;
- do it **systematically** — log all experiments so knowledge compounds;
- shift the PI's time off trivial execution and onto high-level
  framing and judgement.

The hardest part is **not** running experiments. It's:
- having **systematic research**,
- **communicating** results faithfully,
- **aligning** on what was tested, how, with what assumptions,
- so that one person can **trust** another's finding without re-checking
  every step.

Today this is loose: lots of assumptions, lots of effort to verify,
documentation-light. Agentic coding may finally make rigorous logging
and reproduction cheap enough to be the default.

## The framework PI is building

A **skeleton** of instructions / documentation / project structure /
heuristics that the agent follows. The agent then handles the
trivial-but-laborious parts; PI focuses on high-level direction and
keeps **learning**.

> Learning targets the PI cares about: ML, data engineering, software
> engineering, software architecture and design.

## Preferred research process: 7-step "bulletproof problem solving"

McKinsey-origin (book by ex-McKinsey authors). Roughly:

1. **Understand** the problem.
2. **Frame** it correctly — multiple angles.
3. **Disentangle** — find key sub-problems.
4. **Prioritize** — order them by importance.
5. **Plan** based on the prioritization.
6. **Back-of-the-envelope** / heuristics / small probes to check whether
   an approach is even worth pursuing.
7. **Synthesize and communicate** — so others can build on it.

> [Claude note] This 7-step process maps surprisingly cleanly onto
> what's already operational in `CLAUDE.md`:
> - step 6 ↔ Rule 19 BOTE-first / gate-after harness;
> - step 3 ↔ Rule 18 issue-tree / `ISSUES.md`;
> - step 7 ↔ the `audit/*.md` discipline.
> Worth a `concepts/` entry: **mapping bulletproof-7 onto the existing
> Kaggle-comp framework**, and finding the steps that are *under*-served.

## Experimentation culture

- **Explore aggressively.** Test many options even when individual
  success probability is low — *some* will pay off.
- **But: multiple-testing problem.** Running many probes guarantees
  some look interesting **by chance**. Need guards.

> [Claude note] PI said "Trulikki" — almost certainly a transcription
> artifact. Candidates: **Tukey** (John Tukey, exploratory data
> analysis culture), "true likelihood", or just an unclear word.
> Flagging in `questions/`.

## Open follow-ups

- Concept: bulletproof-7 ↔ existing framework mapping.
- Concept: false-discovery / multiple-testing controls in this
  agent-driven setting (what's the playground equivalent of pre-
  registration?).
- Concept: trust + verification primitives that make
  agent-to-agent and agent-to-PI handoffs auditable.
- Friction: the "loose communication" pain — needs its own dedicated
  entry once PI elaborates with concrete examples.
