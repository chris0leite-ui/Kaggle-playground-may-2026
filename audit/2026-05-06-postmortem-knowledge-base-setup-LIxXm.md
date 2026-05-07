# Postmortem — 2026-05-06 knowledge-base-setup-LIxXm

> Per `.claude/skills/postmortem/SKILL.md`, run at WRAPUP step 4b.
> Decision-quality framing (see
> `knowledge-base/concepts/decision-quality-vs-outcome-quality.md`):
> sessions scored on whether decisions were reasonable given priors at
> decision-time, not on outcomes alone.

## Session scope

PI opened the branch to capture thoughts and ask Claude to grill them
into shared understanding. Started in pure understand-mode; PI later
opened the solution lane on two specific items (postmortem skill,
decision-time logging) and approved shipping both.

Kaggle-comp work: none. No experiments, no submissions, no
mechanism-family probes. Meta-process work only.

## What went wrong

**Nothing severe flagged.** Three lower-severity items worth recording
under decision-quality framing:

- **`probe.py` MVP smoke-test was non-rigorous.** Container lacked
  `numpy` so I could not run `python scripts/probe.py bote …` end-to-end.
  Mitigation: re-implemented the logging code path inline in pure
  Python and verified schema. Decision was reasonable (proceeding
  was higher EV than waiting for `pip install`); flagged so future
  postmortems on this MVP cite a real BOTE call as confirmation,
  not just a hand-rolled smoke test.
- **Rule-2-vs-BOTE crediting confusion in PI's F1.2 answer was not
  caught until I git-log-verified.** Standing-duty fired correctly;
  surfaced as `flags/2026-05-06.md` F-1. Indicates the standing duty
  works, but only because I happened to verify; a less skeptical
  agent would have absorbed the misframing into KB synthesis.
  Lesson already encoded in shipped infrastructure: decision-time
  logging stamps `framework_sha`, making future "did rule X exist?"
  questions trivially answerable.
- **CLAUDE.md was not loaded in full at session start.** PI's
  reading of CLAUDE.md mid-session surfaced jargon-drift; agents
  routinely operate on CLAUDE.md while PI doesn't read it. Not a
  decision error, but worth noting as a recurring environmental
  asymmetry.

## Frictions logged this session

Four entries appended to `audit/friction.md` under `## 2026-05-06`:

- `tag: jargon-drift-without-glossary`
- `tag: friction-vs-improvements-role-ambiguity`
- `tag: framework-credit-vs-pi-override-conflation`
- `tag: bote-skipped-under-no-rule-yet`

KB versions with extended context:
- `knowledge-base/friction/2026-05-06-jargon-drift.md`
- `knowledge-base/friction/2026-05-06-friction-vs-improvements.md`

## Promotion candidates (pending PI sign-off pre-merge)

### [ ] expand-on-first-use convention for new acronyms

**Tag:** `jargon-drift-without-glossary` (PI did not know BOTE meant
Back-Of-The-Envelope despite it being load-bearing — s6e5 Day-14 KB
session)

**Where to insert:** `kaggle-comp/SKILL.md` "What never to do" or
`do-and-dont.md`.

**What to add:**

```markdown
- Introduce a new acronym in CLAUDE.md / audits / skill files
  without expanding it on first use, OR without adding it to a
  cross-linked glossary. PI cannot audit vocabulary they don't
  share with the agent.
```

**Why:** PI explicitly disliked discovering shared-markdown vocabulary
they hadn't ratified; breaks the audit-ability that the whole
PI-as-strategist frame depends on.

### [ ] postmortem skill at WRAPUP step 4b (cross-comp portable)

**Tag:** `friction-vs-improvements-role-ambiguity` (cross-comp store
sat at 0 applied entries — s6e5 Day-14 KB session)

**Where to insert:** `kaggle-comp/SKILL.md` "What to do, in priority
order" — add reference to postmortem skill if not already present.

**What to add:**

```markdown
9b. **Postmortem at session-end** — invoke postmortem skill as part
    of WRAPUP section A step 4b. Drafts promotion candidates from
    today's friction.md entries; PI ratifies before commit to
    improvements.md.
```

**Why:** closes the friction → improvements promotion loop that was
empirically broken (1 pending / 0 applied across 580 lines of
friction). Operational at session-cadence rather than end-of-comp.

### [ ] decision-time logging via probe.py JSONL append

**Tag:** `bote-skipped-under-no-rule-yet` + decision-quality framing

**Where to insert:** `kaggle-comp/SKILL.md` reading order; reference
the new `audit/decisions.jsonl` substrate.

**What to add:**

```markdown
- `audit/decisions.jsonl` — append-only JSONL written by every BOTE
  call. Locks predictions + framework SHA at decision-time. Use for
  retrospective decision-quality eval.
```

**Why:** without a structured decision-time log, decision-quality
postmortems re-narrate from outcomes (hindsight bias). Now shipped
in `scripts/probe.py`.

## PI additions

> _Filled in interactively before commit. PI invoked wrap-up + merge
> in one instruction; presenting the candidates above for ratification
> in chat. If PI says "merge as-is", treat candidates as queued for
> the next comp's skill review rather than committed to
> improvements.md immediately on merge._

## Framework version at session-end

- Commit SHA: `c7c86c6` (`Ship decision-time logging MVP in scripts/probe.py`).
- Active rules: 1..19 per CLAUDE.md `## Top-level rules`.
- Loaded skills this session: `kaggle-comp`, `postmortem` (created
  this session), `session-start-hook`, `update-config`,
  `keybindings-help`, `simplify`, `fewer-permission-prompts`, `loop`,
  `claude-api`, `init`, `review`, `security-review`.
- Branch: `claude/knowledge-base-setup-LIxXm`, 14 ahead of fork
  point, 103 commits behind `origin/main` (lots of parallel agents
  merged in the meantime).

## Standing-duty flags raised this session

- `knowledge-base/flags/2026-05-06.md` F-1 — TabPFN documentation
  drift (CLAUDE.md credits framework, mechanism was PI override).
- `knowledge-base/flags/2026-05-06.md` F-2 — Rule 19/BOTE itself
  agent-authored on a feature branch.
- `knowledge-base/flags/2026-05-06.md` F-3 — cross-comp improvements
  store unpopulated.

## Open questions still on the table

- F1.4 calibration approach — committed in principle but two-stage
  (BOTE→OOF→LB) and SKIP-recall problem unresolved.
- F1.5 bipartite CLAUDE.md ownership — PI deferred.
- F1.6.2 extensions (PI override events, realized outcome events,
  bypass detection) — deferred until painful.
- Day-job parallel concept entry — parked.

## Adjacent

- `knowledge-base/README.md` — KB structure + Claude's standing duties.
- `knowledge-base/concepts/decision-quality-vs-outcome-quality.md` —
  framing this postmortem operationalises.
- `knowledge-base/concepts/decision-time-logging.md` — substrate
  shipped this session.
- `knowledge-base/concepts/operational-environment.md` — why git is
  the only persistence mechanism.
