# Cross-comp research — first agent's summary (preserved from completion notification)

The first similar-comps agent (a79d96741c3cff312) wrote a 271-line
report to `audit/2026-05-04-similar-comps-research.md`, but a second
re-spawned agent is now overwriting that path. The full 271-line
content was lost from disk before I could read it. The summary
below is preserved verbatim from the completion-notification
`<result>` block.

## First agent's summary (≤200 words)

Kaggle WebFetch on writeup/discussion URLs returns only `<title>`
(JS-rendered), so I extracted bodies from third-party sources
(NVIDIA developer blog, Medium, dev.to, GitHub READMEs) and search
snippets. Six comps were evaluated; closest matches: **S6E3
Customer Churn** (binary AUC, similar shape), **S5E11 Loan Payback**
(binary AUC, prior 0.202 ≈ s6e5's 0.199), **S5E8 Bank** (binary
AUC), with secondary signal from S4E7 Insurance Cross-Sell, S4E1
Bank Churn, S6E2 Heart Disease, plus the NVIDIA Grandmasters
Playbook. S3E23, S5E1/S5E5/S4E5/S4E12 skipped per criteria.
Convergent themes: GBDT-trio bases + **linear (Ridge/Logistic) meta**
beats tree meta in 4/5 disclosed; **hill-climbing for blend
weights**; FE-at-scale via groupby × stat search; pseudo-labeling
and multi-seed bagging in deepest stacks; trust CV when public LB
diverges. F1's per-(driver, race) sequence structure is unique to
s6e5 — no analog comp tested it.

## First agent's top-5 candidates for s6e5

- **Sequence FE within (Driver, Race)** — `laps_since_last_pit`,
  lag/lead Compound, tire-life proxy, gap-to-leader Δ. F1-unique
  edge. **+30–60 bp.**
- **Original-dataset (aadigupta1601) join** — replicate S4E1's
  `Exited_Orig` lookup move. **+50–200 bp if a (Driver, Race, Lap)
  join exists; 0 bp otherwise — probe first.**
- **GBDT-trio + Ridge/Logistic meta, 5–10 fold OOF** — switch stacker
  from tree to linear; matches S6E3 1st and S5E11 stack pattern.
  **+10–25 bp over single-LGBM.**
- **FE-at-scale: `groupby(KEY)[NUM].agg(STAT)`** sweep over
  {Driver, Race, Compound, Driver×Race} × numerics × {mean, std,
  count, nunique, min/max, p10/p90}; keep top 30–50 by gain.
  **+15–40 bp.**
- **Hill-climbing blend selector + multi-seed (5×) base bag** —
  cheap at 439k rows, on top of #3. **+5–15 bp.**

## Lessons (already covered by irrigation-water postmortem)

- Trust CV when public LB diverges (R1)
- GBDT trio dominates NN at this scale (matches irrigation finding)

## Lessons NOT in irrigation-water (added value)

- **Ridge/Logistic meta beats tree meta** in 4/5 disclosed
  Playgrounds. Irrigation used RF meta (worked there because of
  rule structure); for s6e5 default-to-linear-meta is the
  cross-comp norm.
- **Hill-climbing for blend weights** is the standard cross-comp
  blend selector (vs. equal-weight or arithmetic mean).
- **FE-at-scale `groupby × stat` sweep** is the routine pattern;
  irrigation hand-crafted FE regressed when applied on top of the
  DGP rule, but the groupby×stat pattern is mechanical and
  data-driven, not hand-crafted.
- **F1 sequence structure is unique** — no analog comp tested
  per-(driver, race) lap sequences. Sequence FE has high upside
  but is unprecedented; treat as exploration, not replication.
