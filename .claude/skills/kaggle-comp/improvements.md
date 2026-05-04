# kaggle-comp skill — cross-comp improvements log

Edits promoted here when a friction pattern appears in 2+ comps, costs > 1 LB slot,
or required a human nag. See self-improvement.md for the full distillation protocol.

---

## Pending (not yet applied to skill files)

### [ ] kickoff-runbook.md — add data + task description step

**Tag:** `settled-once` (PI had to ask for domain explanation manually — s6e5 kickoff)

**Where to insert:** After Q5 (EDA summary), before batch-D (baseline).

**What to add:**

```markdown
### Q5b — data + task description

Produce a ≤10-sentence plain-English description covering:
1. What each feature means in domain terms (not just the column name).
2. What the prediction task is asking in real-world terms.
3. Class balance interpretation and what it implies for metric + threshold strategy.
4. Top-3 features by F-score and why they make domain sense.

Write under `## Domain context` in `audit/<date>-day-1-kickoff.md`.
Ask PI: "Does this match your understanding? Anything to correct?"
```

**Why:** Anchors every subsequent experiment to the real DGP. Prevents treating
features as opaque floats. In s6e5, TyreLife/Stint/Cumulative_Degradation are
physical tyre-wear proxies — surfacing this on Day 1 would have seeded better
hypotheses earlier.

---

## Applied
<!-- log completed edits here: date · file · one-line description -->
