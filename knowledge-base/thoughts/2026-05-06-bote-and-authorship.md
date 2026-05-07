# 2026-05-06 — BOTE, jargon, and the authorship question

> PI's [F1 answer](../questions/2026-05-06-grilling-round-2.md#f1-—-focus-setting-concrete-good-case--bad-case)
> on focus-setting, plus the structural finding it surfaced.

## What PI actually said

- **Good focus-instance: BOTE.** PI identified `BOTE` (Rule 19) as the
  rule that helps focus — "a check to see if an approach really looks
  promising or not." PI did not know what the acronym stood for.
- **Bad instance: pre-BOTE fine-tuning.** Agents spent significant
  effort on tiny LB-step improvements before BOTE existed. PI did not
  give specifics (deferred to F1.2).
- **CLAUDE.md observation.** PI read it "for the first time" on this
  date. Found it "really, really long." Suggested moving the
  hypothesis board / calibration ladder elsewhere.

## Synthesis: three findings, ordered by depth

### 1. Surface — BOTE works as a focus-setter (likely)

BOTE = Back-Of-The-Envelope. Rule 19 mandates a `probe.py bote NAME`
call before any ≥10-min CPU/GPU probe; it returns SKIP / DEFER /
PURSUE. PI's intuition that this focuses agents is consistent with
the rule's design.

**Open**: do we have evidence of an *actual SKIP-obeyed incident*?
Without one, BOTE is decoration. (See [F1.1](../questions/2026-05-06-grilling-round-3.md#f11).)

### 2. Middle — jargon-drift is real friction

PI did not know what BOTE meant despite it being central. They
explicitly disliked the discovery. CLAUDE.md uses ~25+ acronyms (G1,
G2, G3, ρ, OOF, GKF, FM, K=N, τ, R5, R7, P10, …) without inline
expansion. See [`friction/2026-05-06-jargon-drift.md`](../friction/2026-05-06-jargon-drift.md).

This is **not** cosmetic. The whole "PI as focus-setter / strategist"
frame requires PI to read agent output and audit it. Acronym drift
silently breaks audit ability. Same disease as the trust problem PI
named for human colleagues (unstated assumptions), different host.

### 3. Deep — who actually authors the framework?

The structural finding. **PI read CLAUDE.md for the first time today.**
Implications:

- The framework is at minimum **co-authored** with the agent. It is
  unlikely PI personally wrote rules using terms PI didn't know.
- That's not a problem in itself — most successful framework docs are
  co-authored. But it changes what "PI as strategist" means:
  - **Strict reading**: PI sets goals, agent encodes them as rules,
    PI ratifies. PI is a *strategist + editor*, not a sole author.
  - **Loose reading**: agent accreted rules from prior-comp memories
    and ongoing work, PI never read most of it, framework drifted.
- The KB we are building right now becomes the **PI's** authoritative
  scratchpad — distinct from CLAUDE.md, which is the **agent's**
  operational manual. That distinction may be the most useful design
  output of this whole exercise.

> [Claude note] If F1.3 confirms the loose reading, the KB ↔ CLAUDE.md
> separation is even more important: the KB is where PI thinks; CLAUDE.md
> is the agent's executable contract. Right now they're tangled.

## Implicit commitments

- BOTE-SKIP discipline is the named focus-setting mechanism PI trusts.
  If F1.1 fails to find a SKIP-obeyed incident, that trust needs
  re-grounding.
- Jargon-drift is now logged as friction; it will re-surface every
  time PI reads agent output.
- "PI as strategist" remains the working frame, but it may need to
  be rewritten to "PI as strategist + editor of an agent-drafted
  framework" depending on F1.3.

## Side-note (PI raised, parked)

CLAUDE.md length / hypothesis-board placement is its own design
question — not the same as the authorship question, but they
interact: if PI is mostly a reader, PI's read budget matters more,
which is an argument for moving high-volume / fast-changing material
out of the always-loaded prompt.
