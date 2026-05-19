# Strategy critique — Day 19 PM after cb_horizon win (2026-05-19)

Triggered by: **guardrails.md §14 auto-trigger #3** ("before adding
a new mechanism family"). cb_horizon (R12-2) is a fresh loss-class
mechanism (RMSE regression on count target vs the K=13 pool's
binary Logloss). Section 5 only — sections 1-4 SKIPPED per skill
plateau-mode ordering (strategy-critic.md §37-47): the +0.046 bp
OOF shift from R7.1 → R12-2 leaves the per-segment failure map
essentially unchanged below the ±0.5 bp resolution of segment-mean
shifts; this is a delta on `audit/2026-05-18-strategy-critique.md`.

## 5. Headroom math vs plan — DAY 19 PM (post-R12-2 win)

Inputs:
- New PRIMARY R12-2 K=14 (K=13 + cb_horizon) + Path-B DCS τ=100k
  → **LB 0.95392** (OOF 0.954475).
- Top-5% boundary 0.95405 → **headroom = +1.3 bp** (was +1.6 bp
  under R7.1; the cb_horizon win closed +0.3 bp of the gap).
- Leader 0.95476 → leader gap −8.4 bp.

Remaining swing queue (Phases C/D/E in today's plan):

| Phase | Mechanism | Midpoint Δ (bp) | P(submission-clear) | Wall |
|---|---|---:|---:|---:|
| C | S4 xendcg-meta on K=14 | +0.04 | 0.35 | 25 min |
| D | cb_stint_completion as 15th base | +0.02 | 0.20 | 40 min |
| E | C4 UID-aggregates LGBM base (fallback) | +0.01 | 0.10 | 30 min |

- Sum optimistic Σ = **+0.07 bp** midpoint EV (assumes all clear).
- 50 % additivity discount = **+0.035 bp** realistic.
- Discounted Σ < headroom (1.3 bp). **Swing cannot close top-5%
  on its own.** Same structural shortfall as 2026-05-18 critique
  (line 23-26: "Σ = 0.115 bp / 0.058 discounted vs 1.6 bp gap"),
  refined to today's numbers.

### Verdict

**Continue the swing** — but for slot-spending discipline and
calibration-ladder data, NOT lift-to-top-5%. Same EOD strategic
posture as 2026-05-18: top-5% requires a +1 bp mechanism class not
yet on the queue. The cb_horizon win was a structural surprise
(R7.1 was thought rank-locked); the swing's remaining items don't
have similar surprise potential individually but each new LB
datapoint refines the band calibration (today's cb_horizon
established that OK-band lower-edge OOF Δ +0.046 bp gives LB
+0.03 bp — a useful new datapoint vs 2026-05-09 K=5 V4 +0.20 bp
OOF → +0.8 bp LB).

### Contingency for top-5% — beyond today

- **TabM (pytabkit Tabular Mixture)** — notebook-rescan flagged as
  the only structurally novel mechanism on the public LB
  (`audit/research/2026-05-19-notebooks.md`); kernel scaffold at
  `kernels/tabm-smoke-v3-gpu/`. Predicted +0.3 to +0.8 bp standalone
  diversity per 2026-05-18 plateau-brainstorm line 81-83. Queued
  for next session, NOT this one.
- **Multi-step cb_horizon family extensions** beyond cb_stint_
  completion: per-(Compound, Stint) horizon predictors, multi-target
  cb_horizon with (laps-until-pit + stint-completion + position-
  change) jointly trained. These extend the winning loss-class
  axis; if today's Phase C+D suggest the axis is generative,
  next session pushes 2-3 more variants.
- **Hedge-ladder maturation** — R12-2 PRIMARY is sub-G2-OOF +
  OK-band; private-LB risk is non-negligible. Final-window R7d
  ladder should now contain: R12-2 (PRIMARY), R7.1 (HEDGE 1 same-
  pool minus cb_horizon), R10 HEDGE 3 (HEDGE 2 cross-mechanism),
  TBD HEDGE 3. Defer ladder finalization to last 3 days.

## Concrete plan re-rank (Phase 5 ordering of skill day-loop step 5d)

Re-ranked queue (was: S4 / C3 / C4 in priority order from plan-agent):

1. **Phase A** (this file) — DONE.
2. **Phase B operator-class diagnostic on cb_horizon** — 10 min;
   informational; confirms whether Path-B or base-diversity earned
   the +0.03 bp LB. Read drives Phase C/D scaling decisions.
3. **Phase C S4 xendcg-meta** — highest base-rate, untouched axis.
4. **Phase D cb_stint_completion** — winning-family extension.
5. **Phase E C4 UID-aggregates** — only if C+D both null.

Slot budget: ≤3 submits today (8 unspent / 12 days remaining
natural cadence = 0.67 slots/day; final-window reserves 3-5).
Submission gate: OOF Δ ≥ +0.02 bp AND ρ_test OK band, OR strict G2.
