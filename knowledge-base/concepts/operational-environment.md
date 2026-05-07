# Operational environment: Anthropic cloud sessions, git-only persistence

> Captured 2026-05-06 from PI's [F1.7.1.1 answer](../questions/2026-05-06-grilling-round-7.md#f171).
> This is the missing context that explains why so much of the framework
> looks the way it does.

## Definitions

- **Session** = one Claude Code invocation, start → /exit.
- **Container** = the ephemeral environment the session runs in.
  Anthropic cloud. State not committed to git is gone at /exit.
- **Inter-session communication** = git only (GitHub or GitLab).
  No shared filesystem, no message bus, no live RPC, no shared memory.
- **Multi-machine** = PI runs several containers on several
  branches in parallel. They see each other's work only after merge.

## Why this shapes everything

The framework's heavy reliance on long markdown files is not a
stylistic choice — it's a **hard operational constraint**:

| Artifact         | Why git-tracked                                      |
|------------------|-------------------------------------------------------|
| `CLAUDE.md`      | Persistent operational context across sessions       |
| `HANDOVER.md`    | Cross-session handoff (no live handoff possible)     |
| `ISSUES.md`      | Claim board for parallel agents (no shared lock)     |
| `audit/*.md`     | Per-decision durable record                          |
| `audit/friction.md` | Learning-loop input that must outlive any session  |
| `.claude/skills/.../improvements.md` | Cross-comp memory                |
| this KB          | PI's scratchpad, must persist                        |

> If a fact isn't committed, it's lost at /exit.

## Implications for the postmortem skill (F1.6.1 / F1.7.1)

1. **Capture must commit to git before /exit.** No in-process buffer.
   The skill writes a markdown artifact and stages it.
2. **Framework-version stamp = commit SHA at decision-time.** Already
   automatic — the skill just records the active SHA when each
   prediction is made.
3. **Decision-time freshness matters.** Git records when a *commit*
   happened, not when a *decision* was taken. If predictions are
   only logged at /exit, they get post-hoc rationalised. Mitigation:
   write predictions immediately into the postmortem file as
   decisions occur, not as a single end-of-session dump.

## Frictions inherent to this environment (residual after git)

Even when git is functioning correctly, three classes of friction remain:

- **Cold-start cost.** Every session begins with no memory; CLAUDE.md
  must do the work of every prior session. As it grows, this gets
  expensive. PI already flagged length.
- **Inter-agent timing.** Parallel agents on different branches only
  see each other after merge. Race conditions and stale-state hazards
  on shared files (CLAUDE.md, ISSUES.md, calibration ladder) are the
  failure mode, not lost messages.
- **Async PI review.** PI can't approve mid-session changes; merges
  arrive in batches. Rule 19/BOTE's agent-authored origin
  ([flag F-2](../flags/2026-05-06.md#f-2)) is exactly this — agent
  drafts on a feature branch, PI ratifies via merge.

## Adjacent KB entries

- [Jargon-drift friction](../friction/2026-05-06-jargon-drift.md) — partly
  an artefact of cold-start: agents accrete vocabulary into shared
  markdown that PI hasn't yet read.
- [Friction-vs-improvements role ambiguity](../friction/2026-05-06-friction-vs-improvements.md) — file-as-mailbox failure mode in action.
- [Decision quality vs outcome quality](./decision-quality-vs-outcome-quality.md) — the
  trust framework whose operational requirements depend on this
  environment.
