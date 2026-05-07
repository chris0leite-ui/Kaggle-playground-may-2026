# Grilling round 8 — file boundary + postmortem-skill design

After [round 7](./2026-05-06-grilling-round-7.md): session = ephemeral
cloud container ([operational-environment](../concepts/operational-environment.md)),
and the empty-improvements.md problem is a role-clarity gap
([friction entry](../friction/2026-05-06-friction-vs-improvements.md)).
Two follow-ups, both close to design — say so if too far from
understand-mode.

---

## F1.8.2 — Define the file boundary, or collapse to one file

`improvements.md` header criteria: *"appears in 2+ comps, costs > 1
LB slot, or required a human nag."* Three operational readings PI
could pick:

- **Strict**: only after 2+ comps. → improvements.md stays empty
  until comp #2. Slow but unambiguous.
- **Loose**: any pattern costing >1 LB slot or needing human nag is
  promoted within-comp now. → faster cross-comp seed; higher
  false-promotion risk.
- **Tag-in-place**: collapse to one file (`friction.md`), tag entries
  `[promotion-candidate]` when they meet criteria, harvest at
  end-of-comp. → no boundary problem, single-source-of-truth.

(Hybrid: tag-in-place + monthly harvest cron — not necessary on
single-machine, but trivial in this ephemeral-container setup.)

**Why I'm asking.** Today's empirical state is a frozen middle: two
files exist with implicit different scopes, neither populating
correctly. Picking one rule unblocks the cross-comp loop and lets
flag F-3 close.

> PI answer:
> _to fill_

---

## F1.7.1.2 — Postmortem skill: replace or augment Rule 17 wrap-up?

Sessions are ephemeral; framework already has a wrap-up flow
(Rule 17 / `WRAPUP.md`) that runs at session-end and pushes to git.
The postmortem skill needs git-commit-before-/exit anyway.

- **Augment**: the postmortem hooks into the existing wrap-up flow
  and adds decision-quality capture as one extra artefact (e.g.
  `audit/YYYY-MM-DD-postmortem.md` produced alongside the wrap-up
  notes). One trigger, two outputs.
- **Replace**: a dedicated postmortem skill subsumes wrap-up and
  unifies them. More work; risk of regressing wrap-up's existing
  behaviour.

**Why I'm asking.** The augment path is cheaper and avoids parallel
mechanisms. If you'd push back on it, I want to know why before any
design starts.

If this is too design-y for understand-mode, say so and I'll park
both questions.

> PI answer:
> _to fill_

---

## Parked

- **F1.6.1.1 — minimum viable schema for decision-time logging.**
  Once cadence (per-session, confirmed) and replace-vs-augment
  (F1.7.1.2) are settled, the schema becomes specifiable:
  predictions, EV midpoint, ρ, family prior used, framework SHA at
  decision-time, override events. Not opening yet.
- **Cold-start cost.** As CLAUDE.md grows, every session pays for
  re-reading it. PI flagged length earlier. Will surface again
  when PI returns to CLAUDE.md cleanup (F1.5).
- **Inter-agent timing.** Multi-branch coordination via merge-only
  is its own friction class. Not yet drilled.
