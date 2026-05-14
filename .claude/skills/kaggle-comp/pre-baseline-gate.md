# Pre-baseline understanding gate

Mandatory before the first `baseline_lgbm` smoke run. PI must
sign off on all seven items below.

**Use dedicated research agents in parallel** for the web /
research items (1, 5, 6, 7) — single message, foreground, three
`general-purpose` subagents. Doing this in the main thread bloats
context and is the documented failure mode (kickoff-#2 friction,
2026-05-04, `playground-series-s6e5`). Local Bash for the schema
items (2, 3, 4).

## Why this gate exists

Without it the agent autopilots into a baseline on default columns
with zero domain understanding, no group-CV check, no published
prior art, and no metric-specific tactics. A "smoke run" framed as
a time probe quietly becomes the first calibration anchor and
locks in bad assumptions.

## The seven items

### Item 1 — host brief verbatim *(web-research agent)*

WebFetch `https://www.kaggle.com/competitions/<slug>`. Extract
**verbatim** (exact host wording — DO NOT summarise) the
Description / Overview, Evaluation, Data, Rules tabs and any
pinned host posts from Discussion. Write to `brief.md` using the
section headers from `templates/brief-template.md`. Cap at 150
lines.

Fallback if WebFetch is gated: `kaggle competitions view <slug>`
and per-tab scraping with the kaggle CLI.

### Item 2 — full schema dump *(local Bash)*

```python
df = pd.read_csv("data/train.csv")
schema = pd.DataFrame({
  "dtype": df.dtypes,
  "n_unique": df.nunique(),
  "n_null": df.isna().sum(),
  "top3": [df[c].value_counts().head(3).to_dict() for c in df.columns],
})
schema.to_markdown("audit/schema-<date>.md")
```

Same for `data/test.csv`. Diff the two — flag any column missing
from test (potential leakage source).

### Item 3 — per-feature target-rate *(local Bash)*

For each top-10 numeric: bin into 20 quantiles, compute target
rate per bin, plot. For each categorical: target rate per level.
Embed in `plots/eda/report.html`. Extends `scripts/eda.py`.

### Item 4 — GroupKFold candidate keys *(local Bash)*

Identify columns that look like a group key (race-id, driver-id,
sequence-id, anything with cardinality ≪ N_rows but ≫ 10).
Document at least one candidate; row-count distribution per
group; whether the candidate appears in test (if not, can't fold
on it without aggregating).

### Item 5 — top-3 public notebooks *(web-research agent)*

Same agent as item 1. WebFetch top-3 most-upvoted notebooks for
the slug. For each capture: title, author, votes, CV scheme
(stratified? group? group key?), feature engineering, model
class, reported OOF / LB, any leakage / group-CV warning. Output a
`prior_art:` YAML block for `comp-context.md`.

Fallback: `kaggle kernels list -c <slug> --sort-by voteCount`
+ `kaggle kernels pull <ref>`.

### Item 6 — domain-knowledge paragraph *(domain-research agent)*

WebSearch real-world decision drivers behind the target. Cite
≥3 sources (Wikipedia, domain press, academic papers, prior-comp
writeups). Output an 8-line `domain_notes:` paragraph plus a
column-to-driver mapping table for `comp-context.md`.

### Item 7 — metric-specific notes *(metric-research agent)*

WebSearch best practices for the comp's metric at this imbalance
/ row count / group structure. Six bullets:

1. Imbalance handling — does class-weighting help this metric?
2. Calibration plan — does the metric require it? When?
3. CV scheme — stratified vs group, with citation.
4. Blend topology — what historically wins this metric on
   Playground.
5. Metric-specific gotchas (e.g. AUC: probability monotonicity,
   public-private split sensitivity).
6. Best public-LB tactics at this scale (early stopping, depth,
   learning rate).

Cite ≥4 sources. Output as a `metric_notes:` YAML block.

### Item 8 — Public-notebook scan *(web-research agent)*

`kaggle kernels list -c <slug> --sort-by voteCount` and pull the
top 5 with ≥10 votes. For each: title, author, votes, CV scheme,
FE tricks, model class, reported OOF / LB. Write to
`external/kernels/notebooks-scan.md`.

**Why mandatory Day-1**: s6e5 Rozen's 0.95354 recipe sat at 19-72
votes the entire comp without being pulled; copying its features
produced the project's biggest single lift (+24 bp) — discovered
on Day-17. Origin of Rule 22. Re-scan at every plateau (day-loop
auto-trigger).

### Item 9 — High-card target-encoding inventory *(local Bash)*

List every cat × cat (and cat³) combo with unique-key count in
`(50, n_train/4)`. Flag the 3-way combo with the largest unique
count as load-bearing. Origin of Rule 21 (the 3-way `(Driver,
Race, Year)` was s6e5's load-bearing single trick at ~+200 bp
standalone). Output: a table appended to
`audit/<date>-pre-baseline-gate.md`.

### Item 10 — Domain-physics feature list *(domain agent or PI)*

5-10 features a domain expert would compute, each with a one-line
physics rationale. Implement ALL before stacking. Forces
single-model-first discipline (Rule 20).

### Item 11 — Single-model OOF target *(synthesis)*

Predict what a kitchen-sink single LGBM should hit, calibrated
against the top public-notebook OOFs from item 8. Use as the
floor against which stacking lift gets measured. If achieved
single-model OOF is more than ~50 bp below the target, debug FE
before any stacking probe.

## Agent dispatch pattern

One message, three parallel `Agent` calls (subagent_type:
`general-purpose`). Each agent writes its detailed artifact to
disk and returns a ≤200-word summary so the main thread doesn't
inhale 10k tokens of research output.

```
Agent(web-research)    → writes brief.md, returns prior_art YAML
Agent(domain-research) → returns domain_notes YAML
Agent(metric-research) → returns metric_notes YAML
```

The main thread merges the three YAML blocks into `comp-context.md`
under new top-level keys. Schema / target-rate / group-keys are
local Bash, written to `audit/schema-<date>.md` and
`audit/<date>-pre-baseline-gate.md`.

## Gate-exit chat (Q5b)

> "Pre-baseline gate done: brief / prior_art / domain_notes /
> metric_notes / schema / per-feature target-rate / group-keys.
> Posting agent summaries. Cleared? [yes / fix what / show me X]"

PI must explicitly say "cleared". Then proceed to Bash batch D.

## Friction handling

Any agent that returns thin / unverifiable findings: log to
`audit/friction.md` with `tag: research-thin`. Re-spawn with a
sharper prompt or surface the gap to PI. Do not let a stub
finding count as a cleared item.
