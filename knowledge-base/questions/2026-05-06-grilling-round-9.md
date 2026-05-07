# Grilling round 9 — promotion trigger + postmortem scope

After [round 8](./2026-05-06-grilling-round-8.md): PI confirmed
session-cadence promotion (loose) and asked Claude to recommend on
skill-vs-doc (recommendation: skill, single SKILL.md, auto-discovered).
Two follow-ups remain about *who does what* and *what the postmortem
covers*.

---

## F1.8.2.1 — Who triggers promotion, and what counts as "clear"?

PI's [F1.8.2 answer](./2026-05-06-grilling-round-8.md#f182) sets
session-cadence promotion (after one or several sessions if the
improvement is "very clear"). Two pieces left to nail:

### Trigger

Who actually does the promotion (friction.md → improvements.md)?

- **PI manually**, when reviewing what changed since last review.
- **Agent at session-end**, as a step in the postmortem skill.
- **Hybrid**: agent drafts candidates, PI ratifies before commit.

The hybrid keeps PI in the loop (preserves the bipartite-CLAUDE.md
ownership shape from F1.5) while letting the agent do the bulk
work. But PI may want pure-manual to stay safe from drift.

### "Clear improvement" — judgement or criteria

- **PI judgement only**: "I see this and decide." Highest bar; lowest
  formalism. Bottleneck on PI attention.
- **Stated criteria**: e.g. "(i) names a generalisable rule, not a
  one-off observation; (ii) cost ≥ X bp or ≥ Y CPU-hours wasted;
  (iii) the friction has been observed twice in the same comp."
  Lets the agent draft most promotions and you only adjudicate
  disputed cases.

**Why I'm asking.** The combination of (trigger × criteria)
determines how much of the loop runs autonomously. With strict
criteria + agent-drafted promotions, the loop self-runs and PI
audits. With pure judgement + manual trigger, the loop is fully
PI-bottlenecked. Both are valid; just need to know which you want.

> **PI answer (2026-05-06, implicit in F1.7.2 spec).** **Hybrid**:
> agent (postmortem skill) drafts promotion candidates at session-end,
> PI ratifies before any commit to `improvements.md`. *"suggest to me
> what could be promoted as rules and ask me if there's anything else
> I would like to add."*
>
> Criteria operationalised in
> [`.claude/skills/postmortem/SKILL.md`](../../.claude/skills/postmortem/SKILL.md)
> step 3 (any one of: generalisable rule named, ≥1 LB slot or ≥1h
> compute waste cost, required PI override, or repeated pattern this
> comp). PI judgement is final.

---

## F1.7.2 — Should the postmortem skill include friction→improvements promotion as a step?

The postmortem already runs at session-end and reviews what changed.
Adding a "promotion candidates?" pass would close the friction →
improvements loop you wanted (regular session-cadence, not waiting
for end-of-comp).

If yes, the skill effectively replaces *"once in a while I tell the
agents to read both files and discuss"*. Tighter loop, less PI
context-switching.

If no, the postmortem stays focused on decision-quality capture only,
and friction→improvements remains your manual prompt-to-agent task.

(Hybrid: postmortem flags candidates, you promote. Same shape as the
hybrid in F1.8.2.1.)

**Why I'm asking.** This decision determines whether the postmortem
skill is one thing (decision capture) or two (decision capture +
loop closure). One-thing is simpler and ships faster; two-thing is
more valuable per session.

> **PI answer (2026-05-06).** **Yes** — postmortem owns the
> promotion step. *"suggest to me what could be promoted as rules
> and ask me if there's anything else I would like to add."*
>
> Postmortem augments WRAPUP (does not replace). Built as
> [`.claude/skills/postmortem/SKILL.md`](../../.claude/skills/postmortem/SKILL.md)
> + WRAPUP.md step 4b. Auto-discovered by the available-skills system
> reminder — verified working in this session.
>
> **Open**: decision-time mid-session capture (F1.6.1's *"lock
> decisions together with framework state at the time"*) is **not**
> handled by this skill. Postmortem is retrospective only. Drilled
> in [F1.6.2](./2026-05-06-grilling-round-10.md#f162).

---

## Parked

- **F1.6.1.1 — minimum viable schema for decision-time logging.**
  Will open once F1.7.2 resolves (skill scope). Fields likely:
  predictions made (OOF Δ band, ρ, EV midpoint), framework SHA at
  decision-time, override events, frictions noticed, promotion
  candidates.
- **CLAUDE.md cleanup (F1.5).** Still PI-deferred. Will re-open
  empirically as PI's edits arrive.
- **The day-job parallel (transfer concept).** Backtest→live
  asymmetry. Parked since round 5.
