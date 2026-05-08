# Strategy critique — Day 2 close (2026-05-04)

Triggered by PI ask "is our strategy sound? do we understand the data
/ where models struggle / what is hard vs easy?" Findings here drive
the new Strategy-critic-loop (`.claude/skills/kaggle-comp/strategy-critic.md`,
CLAUDE.md Rule 14) that fires automatically going forward.

## What's solid

- Calibration ladder discipline: every base has Strat-OOF + LB; gap
  is tracked; gap-widening pattern was *noticed* (M5 −4.4 → M5b −3.5
  → M5d −6.0 bp).
- Pre-baseline gate cleared with real probes: U2 (`lead_PitStop`
  AUC=0.512 — not a leak), U3 (test = i.i.d. row split), DGP probe
  (32% rows in low-entropy leaves at depth 7).
- Diagnostic discipline: GroupKF dropped on U3 evidence; M3/E1 flagged
  as Race-overfits; β HGBC clones diagnosed at 99% correlation.
- Research-loop ran Day 3 with citations — 5 ranked mechanisms.

## Data-understanding gaps (load-bearing)

We have aggregate AUCs and feature importances. We do NOT have:

1. **Per-segment OOF AUC.** Per-Race (26), per-Stint (8), per-TyreLife-
   bucket (10 deciles), per-Year (4), per-Compound (5). A 38bp headroom
   could be 10 races at +5bp and 16 at −20bp; lift surface lives where
   the model is *bad*, and we don't know where that is.
2. **Probability calibration.** Brier score, reliability diagram —
   never computed. Critical for H1 (pseudo-labeling): thresholding at
   0.95 from an uncalibrated stacker is unsafe.
3. **Model-disagreement map.** Where do E3 and M5d disagree? Pairwise
   ρ measured (β = 99% correlation with E3) but the 1% diversity
   region was never localized.
4. **Sequence-structure FE.** `test_lead_pitstop_computable_pct = 0.974`
   says 97.4% of test rows have a same-(Race, Driver) next-lap row in
   test. The lap-sequence structure (laps_since_last_pitstop,
   cumulative_pitstops_this_race, rolling_target_rate(window=5)) is
   unexploited. M4 added 2 features (Recent_Degradation,
   Traffic_Pressure_Proxy); the rest of the sequence space is unscouted.

## Method exploration thin-spots

| Lever | Status | Citation |
|---|---|---|
| 2-way TE (Driver×Race, Driver×Compound, Race×Lap-bin, α=80) | Identified Day-1 by research. **Never executed.** | analyticaobscura Source 1 #2 |
| Compound-fastness ordinal (SOFT/MEDIUM/HARD numeric hardness) | Noted, not added. | analyticaobscura |
| EmbMLP (PyTorch driver embeddings, 887 levels) | **Not tried.** | analyticaobscura Source 1 #2 |
| RealMLP / PyTabKit on Kaggle GPU | Tried fold-0 local CPU only (39.5min). 56-vote public notebook for *this exact comp* uses RealMLP. | yekenot Source 1 #1 |
| Sequence model (LSTM/GRU over (Race, Driver) lap sequence) | **Not discussed.** | new |
| Optuna on Cat / HGBC / XGB | Only LGBM (E5) tuned. | E5 audit |
| Soft pseudo-labels with multi-base agreement guard | H1 currently uses M5d-confidence — but M5d is the over-fit pool. | new |
| Bayes-error / irreducible-floor estimate | Not done. We don't know if 0.95435 is reachable or a noise ceiling. | new |

## Strategic risks

1. **Process error on Day-2 submit selection.** We submitted M5d (gap
   −6.0bp) AFTER seeing gap widen vs M5b (gap −3.5bp). M5b LB 0.94891
   vs M5d 0.94963 = +7bp lift, while gap widened 2.5bp. Rule 12 says
   spend the budget; rule 4 / R2 say select on LB; the gap-widening
   signal said *the smaller pool would win on LB.* No meta-decision
   rule in CLAUDE.md tells the agent to swap meta or prune when gap
   widens — H3 fixes this post hoc only.
2. **Headroom math is optimistic.** H1+H2+H3+H4 midpoints sum to 43bp,
   but additivity is fictional. Realistic 50% effective ≈ 22bp lifts
   us to **0.95183 — 16bp short of top-5%.** No contingency for that.
3. **H1 pseudo-label risk under-specified.** Pseudo-labels from a
   −6bp-gap stacker propagate the overfit. H1 needs (a) multi-base
   agreement guard (≥10 of 12 bases agree), not single-stacker
   confidence, AND (b) a regression test (does pseudo-labeling LIFT
   or REGRESS the *single-base* OOF, not just the stack?).
4. **Kaggle-GPU pipeline not documented.** Rule 13 added but the
   roundtrip (push notebook → schedule → pull artifacts → integrate)
   isn't speced. First GPU experiment will eat 1–2h friction.

## Easy vs hard about this problem

**Easy** (well-handled):
- Raw GBDT signal; single-model ceiling 0.94870 reached fast.
- Native categorical handling on Driver=887 (LGBM/Cat/HGBC).
- Class imbalance (0.199/0.801) — mild; no resampling needed.
- Train/test drift — top z=0.006, negligible.

**Hard** (under-handled):
- Sequence structure within (Race, Driver) groups (97.4% recoverable).
- High-cardinality Driver embeddings (887 train, 801 test, 86 train-only).
- Synthetic-DGP penalty for physics-faithful FE (per pilkwang +
  irrigation-water PM-02). Relative-state FE is the right replacement
  but only 2 features deep.
- Stacker overfit on correlated pool — M5d −6bp IS the diagnostic.

## Concrete additions to Day-3 plan (30 min, before H1-H5)

These four cheap diagnostics RE-RANK the H-list:

1. **Per-Race OOF AUC table** (5 min, M5d_strat OOF). Identifies which
   Races drag the mean. Output → `audit/d3-per-race-oof.md`.
2. **Reliability diagram on M5d OOF** (5 min). If miscalibrated, H1
   uses isotonic-calibrated probs, not raw.
3. **Multi-base agreement matrix** (10 min). For each test row, count
   how many of 12 bases predict >0.5. The 2-tail subset (count ∈
   {0,1,2} ∪ {10,11,12}) is the safe pseudo-label pool — H1 guard.
4. **Sequence-FE candidate scout** (10 min). Compute
   laps_since_last_pitstop, cumulative_pitstops_this_race,
   rolling_target_rate(window=5) on (Race, Driver) groups. Single
   LGBM probe on baseline + these 3 features → does Strat-OOF lift?

After these: H3 → H1 (with guard) → 2-way TE (the missed Day-1
lever) → Kaggle-GPU port of RealMLP (Rule 13 first execution).

## Bottom line

Strategy is *coherent* but *evidence-thin*: we don't know where the
model fails (per-segment) and we haven't scouted the sequence-FE
space. Day-3 plan's 43bp midpoint is optimistic by ~2× under
realistic additivity — single-track plan won't reach top-5%. The
four 30-min diagnostics above rerank the levers and give H1 a real
safety guard. The structural workflow fix is the new
**Strategy-critic-loop** (CLAUDE.md Rule 14, skill slice file).
