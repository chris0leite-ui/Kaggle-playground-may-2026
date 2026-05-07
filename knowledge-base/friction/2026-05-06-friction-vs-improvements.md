# Friction: friction.md vs improvements.md role boundary

**Status**: open. Named explicitly by PI on 2026-05-06.

## Symptom

- `audit/friction.md`: 580 lines, actively maintained.
- `.claude/skills/kaggle-comp/improvements.md`: 1 pending entry,
  0 applied entries.

PI quote: *"most of improvement ideas land in the friction part.
There is maybe missing clarity about what to communicate where. What
to lock in which file — the friction file or the improvements file —
that needs more clarity."*

## Diagnosis

Not a discipline gap (PI is logging frictions). A **role-clarity gap**:
the boundary between the two files is undefined in practice. Default
behaviour is "log everything to friction.md," so improvements.md
starves.

The improvements.md header states criteria — *"appears in 2+ comps,
costs > 1 LB slot, or required a human nag"* — but they're
implicitly used as a strict-AND ("comp 2+") rather than the OR they
appear to be. With one comp running, the strict reading evaluates
to permanently empty.

## Why this matters

The cross-comp learning loop depends on improvements.md being
populated. Empty improvements.md → rule extraction is single-comp-
only → next comp re-learns the same lessons. PI explicitly flagged
this risk in [decision-quality framing](../concepts/decision-quality-vs-outcome-quality.md):
"we don't want to make the same mistake again and again." Empirically
right now, the next comp would.

## Hypotheses for resolution (not actioned)

PI to choose, in [F1.8.2](../questions/2026-05-06-grilling-round-8.md#f182):
- Strict: improvements.md fills only at end-of-comp.
- Loose: promote within-comp on the *or* criteria.
- Tag-in-place: collapse to one file with a `[promotion-candidate]`
  tag — no boundary problem.

## PI's resolution direction (2026-05-06)

**Loose / session-cadence**, two-file structure retained.

- Comp duration is too long to wait for end-of-comp distillation.
- Promotion considered after each session, or after a couple of
  sessions, governed by *clarity* of the improvement.
- "Tag-in-place" not chosen; PI keeps friction.md and improvements.md
  as separate files.

Open: who triggers the promotion (PI / agent / postmortem skill),
and what defines "clear" — drilled in
[F1.8.2.1](../questions/2026-05-06-grilling-round-9.md#f1821).

## Adjacent

- [Flag F-3](../flags/2026-05-06.md#f-3) — empirical evidence
  underlying this friction.
- [Operational environment](../concepts/operational-environment.md)
  — the file-as-mailbox constraint that makes file boundaries this
  important.
