# Postmortem — 2026-05-22 research-improvements-jjI84

Session arc: R22 pivot wrap, comp-day 22 AM. Branch
`claude/research-improvements-jjI84`. No compute run; no leaf claimed;
one BOTE SKIP recorded.

## What went wrong

- **Rule-gap (load-bearing):** R16 6-Q pre-flight Q1 ("family in
  `mechanism_families_explored`?") was answered NO in yesterday's R22
  audit (`audit/2026-05-22-r22-public-notebook-scan.md:84-86`) via
  free-text introspection. But `scripts/probe_r16_real_f1_specialist.py`
  + `audit/2026-05-20-round-16-real-f1-specialist.{log,json}` from
  2026-05-20 named the same axis with verdict -0.025 bp at LR-meta
  (TIE_ZONE). A literal grep would have caught it. Near-miss: 1h CPU
  on a 2-axis variation of a dead axis; only avoided because this
  session's session-start audit found R16 manually.

- **Process bug:** `WRAPUP.md` step 3 directs editing
  `CLAUDE.md ## Current state` YAML — that section was removed in
  the lean rewrite; state moved to `state/current.md`. Step 3
  silently skipped this session.

- **Calibration data point (R26):** sealed PI +0.30 vs agent +0.04
  on `ext_aug_lgbm_K19` BOTE; 7× optimism gap. Pattern: PI consistently
  prices external-data higher than family priors suggest
  (h2_fastf1: PI +5.00 vs agent +3.60; d19_historical_priors: PI
  -1.00 vs agent +0.20; today PI +0.30 vs agent +0.04). 3-of-3 in this
  direction across paired ext-data rows.

- **Soft friction:** post-BOTE-SKIP-after-PI-pivot re-ask created a
  circular decision flow. PI returned "no preference" → agent picked
  cheapest diagnostic, but PI wrapped before it ran.

**Bad decisions:** none I'd retake given decision-time priors.

**PI overrides:** 0 this session ("no preference" is discretion grant).
Last session also 0. If next session also 0, postmortem flags
`pi-stamp-risk` in HANDOVER per WRAPUP step 5b.

## Frictions logged this session

Appended to `audit/friction.md` under `## 2026-05-22`:

- `r22-finding-overlooked-r16-precedent` — see What-went-wrong rule-gap.
- `pi-vs-agent-bote-7x-gap-on-ext-aug` — see R26 calibration data point.
- `wrapup-step-3-CLAUDE-md-current-state-stale` — see process bug.

## Promotion candidates (PI ratify pending)

PI was asked "yes / no / edit each" but had not replied before the stop
hook fired. Candidates documented here; **none committed to
`.claude/skills/kaggle-comp/improvements.md`** this commit. Next session
must surface these for PI ratify before any other work.

### A. Pre-baseline-gate Q1 grep-binding

**Target:** `.claude/skills/kaggle-comp/pre-baseline-gate.md`

**What to add:** Q1 must include the literal shell check
`grep -lrn "<mechanism keyword>" scripts/probe_*.py audit/*round*
ISSUES.md` where `<mechanism keyword>` ∈ {`specialist`, `cohort`,
`ext_aug`, `subset`, the exact axis name}. Free-text introspection ≠
search.

**Why:** today's near-miss; agent-time check missed R16 in the original
R22 audit answer.

### B. WRAPUP.md step 3 fix

**Target:** `WRAPUP.md`

**What to add:** replace step 3 entirely with: "Update `state/current.md`
if PRIMARY, hedge-ladder, submission count, or active-axes table changed
today. Do not edit CLAUDE.md (rules + pointers only post-lean-rewrite)."

**Why:** stale instructions; CLAUDE.md has no `## Current state` YAML.

### C. BOTE-SKIP-after-pivot guard

**Target:** `.claude/skills/kaggle-comp/experiment-loop.md`

**What to add:** "When `probe.py bote` returns SKIP **after** PI just
pivoted INTO that path (sealed PI > 2× agent expected_lb_bp), the agent
must NOT re-ask 'do you want to override?'. Default: cheapest
information-bearing diagnostic (AV-AUC / 1-fold smoke / heuristic prior)
at ≤15 min and report with re-priced BOTE. Treat PI 'no preference' as
agent-discretion grant."

**Why:** today's circular flow; pattern recurs on external-data family.

## PI additions

PI did not reply before stop-hook commit window. Carry forward to
next session.

## Framework version at session-end

- Commit SHA at session-start: `2cd106bf5fda04db5be285ce4aaeb6980138806d`
- Branch: `claude/research-improvements-jjI84`
- Active rules: R0-R36 + R1d-R8d per CLAUDE.md
- Loaded skills this session: postmortem (this run)
