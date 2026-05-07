# Grilling round 6 — operationalising the decision-quality framing

PI's [round-5 answers](./2026-05-06-grilling-round-5.md) replaced the
outcome-rate framing with **decision quality** (concept distilled in
[`concepts/decision-quality-vs-outcome-quality.md`](../concepts/decision-quality-vs-outcome-quality.md)).

Three follow-ups, each tied to a load-bearing operational requirement
of that framing.

---

## F1.6.1 — Decision-time information capture

If decisions are graded by what was knowable beforehand, **the priors
must be logged in a fixed schema** before the probe runs. Otherwise
post-mortems re-narrate from outcomes and hindsight bias enters
silently.

What needs logging at decision-time:

- Predicted OOF Δ (BOTE point + band).
- Predicted ρ vs. PRIMARY (and which precedent base it borrows from).
- EV midpoint and band.
- Effort estimate (CPU/GPU minutes).
- Family prior used.
- Active framework version (rules / thresholds in force).

Concrete check (PI to do or delegate, ≤5 min):

- Open a recent audit such as `audit/2026-05-07-d6-f5-aux-meta-result.md`
  or `audit/2026-05-09-d9c-fm.md`.
- Does it record predictions in **machine-readable form** (yaml /
  table), or only as prose paragraphs?
- If only prose: decision-quality eval 4 weeks later will be
  *unreliable* — there'll be nothing to compare actuals against
  except memory.

**Why I'm asking.** This is the single most concrete piece of
infrastructure required to make F1.4's calibration actually work.
Without it the rest is performative.

> PI answer:
> _to fill_

---

## F1.7.1 — Rule extraction as a deliberate post-mortem step

You named the pattern: bad decisions → extracted rule → don't repeat.
Concrete instance: Rule 16 (5-question pre-flight) born from Day-8.

Question: is this currently **deliberate or accidental**?

- **Deliberate**: post-mortem template includes an explicit "is there
  a generalizable rule?" step. Every audit considers it.
- **Accidental**: someone happens to notice a pattern, drafts a rule,
  it lands. Most patterns leak silently.

Quick way to tell: pick five recent audit files. Of the five, how many
explicitly considered rule extraction (not just "we won't repeat this"
prose)? Count: \_\_\_ / 5.

**Why I'm asking.** If accidental, the framework's growth rate is
governed by how often someone notices, not by how many failures
occur. That's a slow growth function. Deliberate extraction makes
the growth rate proportional to the failure rate — which is what the
decision-quality framing implicitly assumes.

> PI answer:
> _to fill_

---

## F1.8 — Cross-comp memory

PI: *"we don't want to make the same mistake again and again."* But
this comp's CLAUDE.md is per-comp. Rule 8 references
`~/.claude/skills/kaggle-comp/improvements.md` as the cross-comp store.

Is it actually being maintained? Sample questions:

- Has any rule extracted in *this* comp been transcribed to
  `improvements.md` already, or only at end-of-comp (Rule 8 wording)?
- Has any rule from a *previous* comp been imported into this comp's
  CLAUDE.md visibly?
- Concretely: where did Rule 16's pattern come from — purely Day-8
  here, or was there a precedent in a prior comp's `improvements.md`?

**Why I'm asking.** If the cross-comp store doesn't actually exist or
isn't maintained, the rule-extraction loop runs single-comp-only and
resets. PI's day-job framing (transferable research loop) implies
the same risk: if patterns observed in Kaggle don't get formalised
into the day-job toolkit, the transfer is performative too.

> PI answer:
> _to fill_

---

## Parked

- **Framework version stamping.** Currently inferable only from git
  log. If decision-quality eval becomes routine, a lightweight
  "framework_version: rule-1..rule-19" field on each probe / audit
  would make post-mortems trivial. Not asked yet.
- **The day-job parallel.** PI explicitly said two-stage prediction
  (backtest → live) is the typical problem in their work. A future
  concept entry should articulate what does and doesn't transfer
  between Kaggle's OOF→LB and the day-job's backtest→live. Parked.
