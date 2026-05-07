# Grilling round 7 — scoping the postmortem skill + diagnosing the cross-comp gap

After [round 6](./2026-05-06-grilling-round-6.md): PI committed to
locking decision-time capture (F1.6.1), proposed a post-session
postmortem skill, and the cross-comp store was found
[empty](../flags/2026-05-06.md#f-3). Two follow-ups.

---

## F1.7.1.1 — What is a "session" for the postmortem skill?

PI proposed a post-session postmortem skill. Cadence is the first
design choice, because it determines how often "framework state at
decision-time" gets stamped:

- **Per Claude Code session** (start → /exit). Natural unit; aligns
  with current `~/.claude/sessions/` structure. Cadence: minutes-to-
  hours. May be too noisy.
- **Per day**. Aligns with `audit/YYYY-MM-DD-*.md` discipline. Cadence:
  ~daily. Stable, but can miss within-day pivots.
- **Per Rule-17 wrap-up trigger**. Already a defined event in the
  framework. PI says "wrap up" → `WRAPUP.md` runs. Postmortem could
  ride alongside.
- **Per merge to `main`**. Aligns with shared-state changes; cleanly
  attaches a postmortem to each rule-affecting commit.

(Hybrid plausible: lightweight per-session log + heavier per-wrap-up
synthesis.)

**Why I'm asking.** The cadence picks the trade-off between capture
fidelity (more often = less lost context) and noise (more often = more
near-empty postmortems). It also determines whether the skill is
something the agent runs autonomously or PI triggers manually.

> PI answer:
> _to fill_

---

## F1.8.1 — Why is `improvements.md` empty?

Per [flag F-3](../flags/2026-05-06.md#f-3): the cross-comp store has
1 pending, 0 applied entries despite `audit/friction.md` being heavily
populated. Three causes hypothesised; need PI's read:

- **(a) Distillation isn't running.** The weekly distillation step in
  `self-improvement.md` doesn't get triggered. No one runs it.
- **(b) Running, not ratifying.** Distillation runs, produces pending
  entries, PI hasn't been promoting them to applied.
- **(c) Genuinely early.** This is the first comp using this skill;
  Rule 8 says log at *end-of-comp*; comp not over yet.

Quick check that would disambiguate:
- Was there ever a Day-N audit explicitly producing a candidate
  improvement entry that landed in `improvements.md`? If yes → (b).
  If no entries at all → (a) or (c).
- Does the kaggle-comp skill have a prior comp's `applied:` entries
  somewhere PI doesn't see? Would distinguish (c) from (a/b).

**Why I'm asking.** If (a) or (b), the cross-comp loop is broken —
each comp re-learns the same lessons. PI explicitly said "we don't
want to make the same mistake again" — but the empirical state today
is that we *would* make it again. If (c), the loop is fine but we
should verify it actually fires at end-of-comp, before declaring it
operational.

> PI answer:
> _to fill_

---

## Parked

- **F1.6.1.1 — minimum viable schema.** Once cadence (F1.7.1.1) is
  picked, the postmortem schema becomes specifiable: which fields
  must be captured per probe / per session / per merge to support
  decision-quality eval. Not opening yet — too design-heavy without
  the cadence answer.
- **The day-job parallel.** PI noted backtest→live is "the typical
  problem in my work." Worth a `concepts/` entry on transfer-vs-
  reset across the lab/production gap. Parked.
- **F-3 resolution.** If PI confirms (a) or (b) above, a one-line
  change in CLAUDE.md (e.g. moving the distillation trigger from
  end-of-comp to weekly) might fix it. PI said don't over-optimize;
  not pushing.
