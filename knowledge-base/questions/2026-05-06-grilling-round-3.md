# Grilling round 3 — drilling into focus-setting (F1)

After PI's [F1 answer](./2026-05-06-grilling-round-2.md#f1-—-focus-setting-concrete-good-case--bad-case)
identified BOTE as the good case and pre-BOTE fine-tuning as the bad
case, plus the meta-finding that PI was reading CLAUDE.md for the
first time. Three sharper follow-ups.

---

## F1.1 — Cite a specific BOTE-SKIP-OBEYED incident

PI claim: BOTE checks whether an approach is promising before running.

Need: **one concrete incident** where `python scripts/probe.py bote
NAME` returned `SKIP` or `DEFER` and an agent obeyed (i.e. the probe
was not run).

Sources to check:
- recent `audit/*.md` files,
- the hypothesis board in `CLAUDE.md`,
- friction logs.

**Why I'm asking.** Without one, BOTE may be post-hoc decoration —
i.e. it produces a verdict on probes the agent was going to run anyway,
and never actually causes a stop. That's a very different (weaker)
mechanism than "focus-setting".

> PI answer:
> _to fill_

---

## F1.2 — The pre-BOTE fine-tuning waste, concretely

PI said: time was spent moving the public LB by tiny steps before BOTE
existed; this was the original focus failure.

Need:

- **What** were the experiments? Optuna sweeps? Hparam grids?
  Bag-size sweeps? CB depth sweeps?
- **When** in the comp timeline?
- **Roughly how much compute** was spent (CPU-hours / GPU-hours)?
- **What lift was actually achieved** vs. what BOTE would have
  predicted in advance?

**Why I'm asking.** "Tiny steps" is hand-wavy. To know whether BOTE
*solved* the problem we need to know what the problem actually was.
This also gives us a calibration anchor: how much waste BOTE prevents
per comp.

> PI answer:
> _to fill_

---

## F1.3 — Authorship of CLAUDE.md

PI just read `CLAUDE.md` for the first time today, did not recognise
core terminology (BOTE), and suggested moving the hypothesis board /
calibration ladder elsewhere — i.e. PI is treating it as a *reader*,
not an *author*.

Need honest answer:

- Did **you** write the rules in `CLAUDE.md`?
- Or did the agent draft most of them based on prior-comp memories
  (`~/.claude/skills/kaggle-comp/`) and your directional input?
- Or some mix — and what's the rough split?

**Why I'm asking.** This is structural for the whole project frame:

- If PI is the author, "PI as strategist / focus-setter" is literal.
- If the agent drafts and PI approves, the model is closer to
  **agent-as-author + PI-as-editor**. That's not bad — most successful
  framework docs are co-authored — but it changes:
  - what the KB needs to be (a PI scratchpad alone, vs. shared workspace);
  - how new rules should be ratified (who can write Rule 20?);
  - whether jargon-drift will keep recurring unless governed.

It also touches the [trust question (Q3)](./2026-05-06-grilling-round-1.md#q3-—-trust-what-is-structurally-different-about-an-agents-audit):
if you don't fully own the framework, can you trust audits that
appeal to it?

> PI answer:
> _to fill_

---

## Side-note (PI raised, not yet drilled)

CLAUDE.md is "really, really long." PI suggested moving the hypothesis
board (and possibly the calibration ladder) elsewhere. Not actioned.
If we drill into it later it becomes a separate concept entry on
**framework-doc shape**: what belongs in the always-loaded prompt vs.
what should be retrieved on demand. Out of scope for round 3.
