# Grilling round 10 — what the postmortem skill does NOT solve

After [round 9](./2026-05-06-grilling-round-9.md): the postmortem
skill is built and integrated with WRAPUP step 4b. It captures
*retrospective* reflection at session-end. It does **not** capture
the decision-time information you committed to in F1.6.1.

One follow-up.

---

## F1.6.2 — Decision-time mid-session capture (not yet built)

The postmortem skill captures things at **session-end**:
- What went wrong (in retrospect).
- Frictions logged (already written by WRAPUP step 4).
- Promotion candidates.
- PI additions.
- Framework SHA at session-end.

It does **not** capture, for each probe / decision **as it happens**:

- BOTE prediction (predicted OOF Δ + LB Δ + EV midpoint + band).
- Family prior used.
- Predicted ρ vs PRIMARY.
- Effort estimate (CPU/GPU minutes).
- Active framework SHA at decision-time.
- Override events (when PI says "test first" or similar).

Without that data, decision-quality eval at session-end (or 4 weeks
later) reduces to "what do I remember thinking?" — exactly the
hindsight trap the
[concept](../concepts/decision-quality-vs-outcome-quality.md) warns
about.

**Three plausible ways to add decision-time capture** (PI to pick;
none of these are tiny):

- **(a) Schema in `audit/YYYY-MM-DD-<probe>.md`** — the existing
  per-probe audit file gets a YAML frontmatter block written
  *before* the probe runs, with the predictions. This piggybacks
  on existing discipline (audits are already written) but requires
  agents to commit a stub at decision-time, not at write-up time.
- **(b) `probe.py bote` writes a record.** The BOTE harness already
  produces a SKIP / DEFER / PURSUE verdict. Extend it to append
  the verdict + predictions to a structured log
  (`audit/decisions.jsonl` or similar). Naturally tied to
  decision-time (BOTE runs before the probe). Requires modifying
  `scripts/probe.py`.
- **(c) Manual: PI's call.** Predictions captured only when PI
  explicitly asks for them in chat, narrated in the audit. Same
  as today; not a new mechanism. Acknowledges the gap; doesn't
  close it.

Each has different setup cost and different coverage. (a) and (b)
are complementary, not exclusive.

**Why I'm asking.** Without one of (a) or (b), the postmortem skill
is half the F1.6.1 commitment — retrospective half built,
prospective half not. Outcome quality bias creeps back in via
memory.

If you want this v1.5 right now, pick one. If you want to live with
the gap until you've used the postmortem v1 a few times and seen
where it's actually painful, say so — that's a defensible choice.

> **PI follow-up (2026-05-06).** *"How would we best implement
> decision-time logging?"* — solution-mode opened.
>
> **Claude recommendation**: substrate (b) — extend
> `scripts/probe.py bote` to append a JSONL record to
> `audit/decisions.jsonl` on every call. ~30 lines in probe.py. Full
> design + schema + rationale: [`concepts/decision-time-logging.md`](../concepts/decision-time-logging.md).
>
> **PI: "go" (2026-05-06).** **Shipped.** MVP merged into
> `scripts/probe.py` (this branch); smoke-test verified the JSONL
> schema parses cleanly with all 13 fields. First real BOTE call
> will create `audit/decisions.jsonl`. Extensions (PI overrides,
> realized outcomes, bypass detection) deferred per the design.

---

## Parked

- **F1.6.1.1 — minimum viable schema for decision-time logging.**
  Becomes specifiable once F1.6.2 picks a substrate.
- **CLAUDE.md cleanup (F1.5).** Still PI-deferred.
- **Day-job parallel.** Backtest→live transfer concept. Parked since
  round 5.
- **Cold-start cost.** As skills accrete (kaggle-comp, postmortem,
  …), CLAUDE.md + skill list grow. PI flagged early. Will surface
  again at length.
