# Grilling round 2 — drilling into the transfer answer

After PI's [Q4 answer](./2026-05-06-grilling-round-1.md#q4-—-transfer-what-crosses-the-labproduction-gap)
and [synthesis](../thoughts/2026-05-06-transfer-answer.md). Three sharp
follow-ups that test whether the high-level claims hold up under
specifics.

---

## F1 — Focus-setting, concrete (good case + bad case)

PI claim: PI's job is to scaffold instructions so agents land on the
right focus.

Need:

1. **One thing in the current `CLAUDE.md` (or `ISSUES.md`, or audit
   discipline) that *demonstrably focused* an agent well** — i.e. a
   specific rule that caused an agent to skip a bad probe or pick a
   good one. Cite the rule and the incident.
2. **One thing that *failed* to focus** — an agent went down a path
   that wasted compute despite the rules being in place.

**Why I'm asking.** Without both halves, "focus-setting via
instructions" is a slogan. With both, we can see the actual mechanism
and where it breaks. Likely candidates from `CLAUDE.md`:
- Rule 16 (5-question pre-flight) — was this *born* from a failure?
- Rule 19(a) BOTE SKIP/DEFER/PURSUE verdicts — when have they been
  ignored? When have they correctly killed something?

> PI answer:
> _to fill_

---

## F2 — Long-term vs short-term, concrete

PI claim: don't optimize next-day LB; invest in foundation that pays
off at the month-out final evaluation.

Need:

- **Name one experiment you'd run *this week* whose payoff is *only* in
  week 3-4** of the comp.
- It must be defensible even if it produces zero LB lift on Day-15.

**Why I'm asking.** If you can't name one, that's strong evidence the
foundation-investment principle is aspirational, not operational. Not
a gotcha — it tells us this is a thing the framework doesn't yet
*cause to happen*.

> PI answer:
> _to fill_

---

## F3 — Feedback-loop asymmetry (day-job vs Kaggle)

In the day job, between proposing a model change and getting
**trustworthy** evidence about it, how long does it take?

- Hours? Days? Weeks? Months?
- What's the trustworthy signal — paper trial PnL? Out-of-sample
  backtest with proper walk-forward? Live trading?

Kaggle gives OOF in minutes and LB in an hour.

**Why I'm asking.** If day-job evaluation is 100× slower than Kaggle's,
the same research loop *cannot* run the same way:

- Cost of a wrong experiment is much higher in $ terms.
- Prior-tightening before running matters more.
- BOTE/gate/audit discipline must be much stricter pre-run.
- "Many cheap probes" doesn't work; "few expensive bets" does.

This is probably the biggest single thing that *doesn't* transfer
cleanly from Kaggle. I want to know what you actually face.

> PI answer:
> _to fill_
