# 2026-05-06 — BOTE applied, scale-trust, and the authorship answer

> PI's answers to [F1.1, F1.2, F1.3](../questions/2026-05-06-grilling-round-3.md).
> Synthesis. The deepest finding is **trust must shift from per-instance
> to structural at scale** because PI cannot audit every BOTE verdict.

## Substantive answers

### F1.1 — BOTE has been applied; some probes rejected

PI confirms BOTE works *some* of the time. **No specific incident
cited.** PI explicitly admits they will not audit each verdict — and
this gets more extreme as the system scales. Followed up in
[F1.1.1](../questions/2026-05-06-grilling-round-4.md#f111).

### F1.2 — TabPFN fine-tuning (Day-14)

The pre-BOTE-style waste PI cited is actually a **post-BOTE incident**:

- TabPFN-2.5/2.6 fine-tuning queued.
- Several GPU hours already spent.
- Without intervention: full 5-fold = >10h GPU.
- PI personally insisted on **smoke / fold-0 first** (Rule 2).
- Fold-0 returned ~0.94439 → AUC ceiling 0.944 → -64bp vs PRIMARY → DEAD.

> **The wrinkle that matters.** PI presented this as evidence BOTE
> works. It is actually evidence that **PI override** was the operative
> focus mechanism on this run, not BOTE/Rule-2 firing on its own.
> Important to keep separated.

### F1.3 — Authorship: confirmed agent-drafted

- PI did **not** write CLAUDE.md.
- PI plans to keep a section that is theirs.
- PI accepts the agent authoring the parts the agent is better at —
  PI won't manually log those.
- PI explicitly asks Claude to flag points warranting careful PI
  review. → Filed as [standing duty in the KB README](../README.md#claudes-standing-duties).

## The structural finding

Across the three answers, one claim does the work:

> **At scale, the per-probe audit budget shrinks toward zero.**

PI cannot read every BOTE verdict. PI cannot personally intervene on
every TabPFN-style run. PI cannot author every rule. So trust must
shift from per-instance to **statistical / structural**.

What changes structurally if we accept this:

1. **Audit unit becomes the population**, not the individual decision.
   The right metric is e.g. "of the last 50 BOTE PURSUE verdicts, how
   many landed within ±0.5bp of predicted lift?" Not "did BOTE call
   this one right?"
2. **PI's intervention becomes a sampling strategy.** When PI overrides
   (TabPFN smoke-first), the framework should *learn* from the override
   — not just absorb the outcome. Currently CLAUDE.md absorbs the
   outcome ("TabPFN DEAD") but loses the fact that PI overrode.
3. **Documentation drift becomes a first-class risk.** When agent-
   written CLAUDE.md credits the rules for outcomes that actually
   required PI override, the framework will *appear* to be working
   when it isn't. See first concrete flag below.

## First flag (kicking off the standing duty)

CLAUDE.md hypothesis-board entry on TabPFN reads:

> "DEAD. AUC ceiling 0.944 = -64bp vs PRIMARY... ρ=0.960 (diverse) but
> gap too large for pool."

This narrates the *outcome* but not the *mechanism*. From PI's F1.2
the kill required PI's "test first" instruction. The audit / hypothesis
board doesn't say so. **Result**: future readers (humans or agents)
will infer the rules were sufficient. They weren't. This is the kind
of drift Claude should flag — and PI should consider whether
audit/hypothesis entries need a `pi_override: yes/no` field.

(Not proposing the field yet — flagging the gap. PI requested
understand-first.)

## Second-order question PI raised

PI said: *"maybe I will collect some useful examples which the agent
could then use leverage to guide the decision making. They will need
to know how to do these calculations, they will need to know how to
do heuristics."*

This is PI proposing a **heuristics library** for the agent — a body
of worked BOTE examples / decision priors. Adjacent to the existing
`FAMILY_PRIORS` dict in `scripts/probe.py`, but more general.

Parking as concept-entry candidate; not actioned today. Filed as
[F2 follow-up parked](../questions/2026-05-06-grilling-round-4.md#parked).

## Where this sits w.r.t. the original Q4 answer

PI's transfer claim was: the **research loop** is what transfers,
underwritten by the framework. Today's answers add:

- The framework is not a closed system PI fully owns. It is co-authored,
  and the PI's override is part of how it works. Both are facts the
  framework should make explicit, not hide.
- "Trust the loop" cashes out at scale as "trust the loop's *aggregate
  behavior*", not "trust each verdict". This is the same shift in
  granularity that distinguishes good from sloppy backtest discipline
  in the day job.

That parallel — paper-trial-aggregate vs. per-trade audit — is
worth a `concepts/` entry at some later point.
