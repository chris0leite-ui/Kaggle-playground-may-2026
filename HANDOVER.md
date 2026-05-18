# HANDOVER

Next-session brief. **PI says "handover"** → agent reads this file
and proceeds. **PI says "prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

This file is rewritten (not tail-appended) every wrap-up. Prior
versions: `audit/archive-YYYY-MM-DD-handover-*.md`.

---

## Where we are

**PRIMARY: R7.1 K=13 + Path-B DriverClass × Stint τ=100k.
LB 0.95389.** (Unchanged through R8 + R9.) Top-5% gap −1.6 bp;
leader gap −8.7 bp. File:
`submissions/submission_K13_pathb_driverclass_stint_tau100000.csv`.

Submissions: **49 / 270** total; **7 used 2026-05-18**; **3 daily
slots still available** at session end. Comp-day **18 of 31**; days
remaining **13**. PI: hold all slots.

## Round 9 — Dual-track NB4 + C1 (LATEST)

**Mandate**: After R8 EOD strategy-critic verdict "Σ × P(real) ≈
0.058 bp vs 1.6 bp gap = structural shortfall", run last 2 viable
research-loop candidates with R8 PM addendum priority. PI chose
dual-track: NB4 (cheap TE-as-base diagnostic) + C1 (external-info
structural lever from EOD-critic Section 5).

### Session-start dedup (Phase 1 Explore reconnaissance)

Discovery that **PM research-loop missed 2 of 3** candidates already
tested today:
- **C4 UID magic-features** smoke-FAILED at −16.2 bp on 50k rows
  (`audit/2026-05-18-tier-a-batch.md:92-102`).
- **Competitor pit cascade** tested in 3 variants (A3-1 K=4+1
  +0.337 bp WEAK ρ=0.983 REGRESSION_RISK; F3 K=24 −0.015 bp NULL;
  pit-pressure K=11 −0.012 bp NULL). Lagged PitStop already
  implemented in `scripts/fe_picks_a2a3.py:218-332`.
- **NB4** confirmed novel (no Compound × Stint TE in any of
  yekenot's 6 `TE_CONFIGS`, `scripts/p1_features.py:336-342`).

Friction logged: `research-loop-dedup-miss-vs-ledger`. PM research
synthesis must include same-session prior-run results, not just
historical mechanism-ledger.

### R9 numerical results

| Probe | Standalone OOF | Wall | K=14 OOF | Δ vs R7.1 | ρ vs R7.1 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **NB4** Compound×Stint TE base | 0.94850 G1✓ | 150 s | 0.954469 | **−0.022 bp** | 0.99977 | NULL |
| **C1** Aadigupta per-Race scalars | 0.94902 G1✓ | 162 s | 0.954466 | **−0.045 bp** | **0.99998** | NULL |

Reference: R7.1 PRIMARY K=13 OOF 0.954471 (recomputed in-session).
PRIMARY unchanged.

### R9 strategic conclusion

**Rank-lock at K=13+Path-B is structurally confirmed across three
axes:**

1. **Operator family** — R6 v2 transformer absorbed, R7 swap-noise
   DAE absorbed at every Path-B variant tested.
2. **Mechanism class** — R4 segment-FE G2-fail, R4/R5 HMM and
   pit-cascade null (3 variants), R9 NB4 TE-base absorbed.
3. **Data class** — R9 C1 external Aadigupta scalars absorbed
   (regressed MORE than NB4; yekenot's 6 TE_CONFIGS touch Race in
   5 of 6 configs, absorbing per-Race signal density).

The K=13+Path-B pool has reached its **structural ceiling for
row-level features**. PI directive at R9 end-of-session: **hold all
3 daily slots, pivot to mechanism expansion for R10.**

### R9 deliverables on disk

- Scripts: `scripts/probe_r9_compound_stint_te_base.py`,
  `scripts/probe_r9_race_external_scalars.py`.
- Artifacts: `scripts/artifacts/oof_NB4_compound_stint_te_strat.npy`,
  `test_NB4_compound_stint_te_strat.npy`,
  `oof_C1_race_external_strat.npy`, `test_C1_race_external_strat.npy`.
- Path-B K=14 outputs (last write = C1 run):
  `scripts/artifacts/oof_K14_pathb_driverclass_stint_tau100000.npy`,
  `submissions/submission_K14_pathb_driverclass_stint_tau100000.csv`
  (NOT submitted; collision friction logged).
- Audit: `audit/2026-05-18-round-9-execution.md`,
  `audit/2026-05-18-round-9-K14-nb4.{log,json}`,
  `audit/2026-05-18-round-9-K14-c1.{log,json}`.
- Code: `scripts/build_K13_pathb_multiseg.py` modified — added
  `--extra-bases` CLI flag.

## R10 priority queue (mechanism-expansion candidates)

All three are structurally orthogonal to row features and would
inject signal NOT derivable from the 14-column s6e5 schema or the
existing K=13 pool.

### A. Sequence-to-sequence transformer on lap sequences
- Per-(Driver, Race) lap sequences with full attention; predict
  the next-N-laps' PitNextLap probability. K=13 HMM is a 4-state
  Baum-Welch one-shot — full seq2seq orthogonal.
- ~2 hr Kaggle T4. Highest novelty + highest variance.

### B. Graph mechanism (competitor edges)
- Per-(Race, Lap) graph with cross-driver edges; LightGCN or GAT
  2-layer. Models true competitor interactions (the F1 undercut
  game) which row-level pit-cascade aggregates failed to capture.
- ~3 hr local CPU or Kaggle T4. Medium novelty.

### C. Survival / hazard model on stint life
- Cox PH or DeepSurv on (Compound, TyreLife, Driver, Race) with
  pit event as event indicator. Hazard rate at lap L as base.
- ~30 min CPU. Fastest, lowest novelty (semi-parametric reduces
  to logit at single-time-point evaluation), but orthogonal
  inductive bias to row-LGBM.

### D. (Hedge-prep track) Final-window R7d ladder
- Held-back hedge candidates: R7.2 fold-bag (LB 0.95389 tied),
  R8 60/20/20 multi-seg blend (OOF +0.079 bp, ρ TIE_ZONE, NOT
  submitted). Final-3-day window starts ~2026-05-28; ladder
  finalisation can happen any time in R10-R13.

## What's confirmed CLOSED

- All single-base additions to K=13+Path-B (R9 confirms structural
  rank-lock).
- Path-B segmentation hunt (R7+R8 sweep, 7 segs total: 1 winner,
  3 marginal, 3 null; DriverClass × Stint is unique > +0.10 bp dim).
- External-data injection via per-Race FEATURE scalars (R9 C1).
- TE-as-base for Compound × Stint (R9 NB4).
- UID magic-features (R8 C4 smoke −16.2 bp).
- Competitor pit cascade (3 prior variants all null/absorbed).
- DAE swap-noise v1 (R7).
- Transformer v1 / v2 (R6 / R7).

## R9 frictions (newly logged this session)

See `audit/friction.md`:
- `research-loop-dedup-miss-vs-ledger` — PM addendum's top-3 had
  2 already-tested items.
- `rank-lock-confirmed-three-axes` — three structurally distinct
  mechanisms fell in <1 week.
- `k14-output-collision-extra-bases` — `--extra-bases` artifacts
  collide between runs.
- `pathb-ref-baseline-hardcoded-r52` — multiseg script Δ-comparison
  is anchored on R5.2 not current PRIMARY.

## How to resume

1. `git fetch origin && git log HEAD..origin/main` (Rule 32).
2. PI typical opens: "next iteration" → start R10 plan mode with
   A/B/C candidates above; ask which to attempt.
3. PI: "wrap up" → run `WRAPUP.md` section A + postmortem skill.
4. PI: "submit hedge" → submit R7.2 fold-bag (LB 0.95389 tied, on
   disk) OR R8 60/20/20 blend (saved as
   `submissions/submission_R8_blend_60_20_20_r71_dt_rc.csv`).
5. Daily quota resets at Kaggle UTC midnight (currently 17:17 UTC
   on 2026-05-18, so 7 hours until reset).

## Files to scan first next session

- `state/current.md` — single-current state, R9 update appended.
- `state/mechanism-ledger.md` — R9 closure entries (~lines 478-517).
- `audit/2026-05-18-round-9-execution.md` — full R9 narrative.
- `audit/2026-05-18-strategy-critique.md` — R8 EOD critique
  (still the operating thesis after R9 confirmation).
- `audit/research/2026-05-18-research-pm-addendum.md` — PM
  addendum's 5 novel candidates (only NB4 testable post-dedup;
  R10 candidates DO NOT come from here).
