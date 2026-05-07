# Decision quality vs outcome quality

> Concept distilled from PI's [round-5 answer](../questions/2026-05-06-grilling-round-5.md)
> on 2026-05-06. PI explicitly rejected the success-rate framing in
> favour of evaluating *decisions* rather than *outcomes*.

## The two ways to grade a research move

- **Outcome quality**: did the probe pay off? (Did we get LB lift?)
- **Decision quality**: was running the probe a reasonable thing to do
  given the information available *before* the probe ran?

In a noisy exploration regime, outcomes are dominated by luck on small
samples. Decisions are what we control. The trust mechanism the project
needs at scale is therefore **decision quality**, not outcome quality.

## Why outcome-only evaluation breaks

### Per-instance

- A bad decision can produce a good outcome (luck).
- A good decision can produce a bad outcome (variance).
- Grading the decision by the outcome rewards the wrong thing.
- TabPFN (Day-14) is concrete: the *outcome* (DEAD) was right, but
  the *decision* (queue 5-fold without smoke) was bad — fixed by PI
  override, not by the rule machinery.

### At scale

- A 5% success rate could be excellent if you ran 1,000 probes.
- A 95% success rate could be terrible if you only ran obvious wins.
- Aggregate success rates conflate edge with luck.

PI quote (2026-05-06): *"the success rate might be the wrong framing
because we might have a success rate of 5% that would be incredible if
we could test a thousand things."*

## What "reasonable decision" means (PI's two halves)

- **Positive**: positive EV given pre-run information × effort budget.
- **Negative**: not an obvious mistake the framework could have
  ruled out.

> "If there's a reasonable chance in comparison to the effort we can
> put in, that's good." — PI

## Hindsight is the central trap

PI flagged this directly:
> *"afterwards you always want to [see things you couldn't see at the
> time]."*

The decision-quality framing is the antidote — it forces post-mortems
to ask *"what could we have known then?"* rather than *"what do we know
now?"* But the antidote only works if:

1. **Pre-run information is logged in a fixed schema.** Predictions,
   EV estimates, family priors, ρ predictions, predicted OOF Δ — all
   recorded **before** the probe runs. Currently narrated in audits;
   not yet structured. (See [F1.6.1](../questions/2026-05-06-grilling-round-6.md#f161).)
2. **The framework version is stamped at decision-time.** Which rules
   existed, which thresholds were active. Otherwise "obvious mistake"
   silently means *"obvious now"*. (See refinement below.)

## "Obvious mistake" is framework-time-relative

The negative criterion in PI's definition depends on which rules
existed when the decision was made. So the post-mortem question is:

> Given the rules and priors **as they stood at the time of the
> decision**, should this probe have been skipped?

- **Yes** → framework failed despite having the rule. Why wasn't it
  applied? Audit the gap.
- **No** → framework didn't have the rule yet. Opportunity to extract
  Rule N+1.

This split is the framework's growth loop in disguise.

## The growth loop: bad decisions → rules

PI: *"doing those is a good opportunity to learn... extract rules like
that could avoid leakage."*

Empirical example: **Rule 16 (5-question pre-flight)** was extracted
on Day-8 from the falsification of T1.5/T1.3/T1.2 — three candidates
that passed the research-agent EV ranking but failed the 5-question
check retroactively.

Pattern:

```
bad decision →
  postmortem identifies an extractable rule →
    rule added to framework →
      future decisions of same kind become "obvious mistake"
```

This converts *"we don't want to make the same mistake again"* from
a slogan into the framework's actual update mechanism.

## Operational requirements (load-bearing) — status as of 2026-05-06

For the decision-quality framing to function as a trust mechanism:

1. **Decision-time logging** — schema, not prose.
   *PI committed*: "really really need to lock the decision-time
   capture." No infrastructure yet. Pending postmortem-skill design
   ([F1.7.1.1](../questions/2026-05-06-grilling-round-7.md#f171)).
2. **Framework versioning** — which rule was active when.
   *PI committed*: "lock our decisions together with the status of
   the framework at the time that we took the decision." Currently
   inferable only from git log.
3. **Deliberate rule extraction** — explicit post-mortem step, not
   accidental.
   *Current state*: PI manually adds to `audit/friction.md` (580
   lines, active) and `.claude/skills/kaggle-comp/improvements.md`
   (40 lines, 1 pending, 0 applied). Periodically prompts agents
   to discuss + update rules. PI accepts the loose loop and does
   not want to over-engineer it.
   *Proposal on the table*: a **post-session postmortem skill** that
   would capture decisions + framework state automatically. Scope
   pending [F1.7.1.1](../questions/2026-05-06-grilling-round-7.md#f171).
4. **Cross-comp persistence** — rules survive comp boundaries.
   *Verified empirically*: cross-comp store exists but is essentially
   unpopulated (0 applied entries). See [flag F-3](../flags/2026-05-06.md#f-3).
   The loop runs single-comp-only today. Cause hypothesised but not
   confirmed ([F1.8.1](../questions/2026-05-06-grilling-round-7.md#f181)).

If any of these is missing, the framing degrades back to
outcome-quality with extra prose. Today: 2 are committed-but-unbuilt,
1 is loose-but-running, 1 is broken.

## Adjacent concepts

- *Process versus results* (Annie Duke, "Thinking in Bets") — same
  distinction, decision-theoretic vocabulary.
- *Pre-registration* (scientific replication crisis) — solves the
  same hindsight problem by fixing predictions before data.
- *Bayesian sequential design* — formalises EV-vs-effort across
  branching probes; close relative of BOTE.
