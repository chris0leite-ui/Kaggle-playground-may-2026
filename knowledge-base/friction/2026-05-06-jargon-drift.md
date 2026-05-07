# Friction: jargon-drift in CLAUDE.md

**Status:** open. Named explicitly by PI on 2026-05-06.

## Symptom

PI read `CLAUDE.md` for the first time on 2026-05-06 and discovered they
did not know what **BOTE** stood for, despite it appearing dozens of
times as the load-bearing focus-setting mechanism (Rule 19).

PI quote: *"agents drifted into some sort of slang or shortcuts that
I'm not aware of, which I don't like."*

BOTE is "Back-Of-The-Envelope." It's a standard term in physics /
consulting / engineering — but not universal, and **never expanded
inline in CLAUDE.md**.

## Why this matters

If PI does not share vocabulary with the agent, PI cannot audit the
agent. The whole "PI as focus-setter / strategist" frame depends on
PI being able to read what the agent produced and verify it. Acronym
drift silently breaks that.

This is structurally similar to the trust problem PI raised w.r.t.
human colleagues (unstated assumptions). Same disease, different host.

## Inventory of likely jargon in CLAUDE.md

Rough scan of `CLAUDE.md` (not exhaustive):

- **BOTE** — Back-Of-The-Envelope.
- **OOF** — Out-Of-Fold (cross-validated prediction on held-out fold).
- **LB** — (Public) Leaderboard.
- **GKF / GroupKF** — Group K-Fold.
- **Strat** — StratifiedKFold.
- **FM / FFM** — Factorization Machine / Field-aware FM.
- **GBDT** — Gradient-Boosted Decision Tree.
- **HGBC / LGBM / CB / XGB** — sklearn HistGradientBoostingClassifier /
  LightGBM / CatBoost / XGBoost.
- **ρ (rho)** — Pearson correlation between two prediction vectors.
- **L1** — typically the L1 coefficient / L1 norm in this doc.
- **K=N** — pool size (number of base models in the stacking pool).
- **τ (tau)** — shrinkage strength in empirical-Bayes hier-meta.
- **G1/G2/G3/G4** — leakage-filter gates (Rule 3).
- **R1/R5/R7/R8** — rules from the prior-comp postmortem.
- **P3/P5/P6/P10** — patterns / priors from prior comps (need source).
- **TE** — Target Encoding.
- **EV** — Expected Value (in BOTE prior).
- **PI** — Principal Investigator (you).
- **bp** — basis points (0.0001 of AUC).

## Hypotheses about cause

1. Agent absorbed terminology from prior-comp postmortems and the
   `kaggle-comp` skill without expanding for the human reader.
2. CLAUDE.md grew by accretion; no rule says "expand acronyms on
   first use."
3. PI didn't request expansion; agent optimised for terseness (≤150
   line cap, ≤50k token cap).

## Status

- Not fixed. Documenting only at this stage per PI request to
  understand-first / propose-later.
- Linked to deeper [authorship question](../questions/2026-05-06-grilling-round-3.md#f13)
  — if PI is a reader of CLAUDE.md, jargon-drift is also an
  agent-to-PI handoff failure, not just a documentation hygiene issue.
