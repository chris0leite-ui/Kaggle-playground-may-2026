# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are

- **PRIMARY** = `d13e_compound_stint_tau20000` LB **0.95049** (Day-13 PM advance).
- **TOP SUBMISSION CANDIDATE (HELD):** `path_b_K22_invlaps_tau20000.csv`
  - OOF **0.95110** (+2.75 bp); largest non-meta-derivative single-add OOF advance ever.
  - **Target-derived** signal (LGBM regression on `1/(1+laps_until_pit)`); orthogonal-signal criterion satisfied.
  - ρ vs PRIMARY 0.99753; flips 57/96 (ratio 0.594).
  - Path B family amp prior (8-11.6×) → predicted LB band **+1.25 to +32 bp**.
- **HEDGE candidates held:** `path_b_K22_invlaps_tau100000.csv` (asymmetric flips 45/189),
  `path_b_K22_d12meta_tau100000.csv` (LB 0.95045 R7-eligible),
  `d12_lr_meta` (+1.348 bp OOF meta-derivative).
- **Gap to top-5%** (0.95345): −29.6 bp from PRIMARY. Bull projection on the held
  τ=20k candidate (11.6× amp) → 0.95370 (top-1% range).
- **Submissions used total:** 25/270; 1 LB submission today.

## Read order on session start (skip default; this is the synthesis)

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `scripts/probe.py` — entry point. `bote()` for BOTE, `gate()` for uniform gate report
3. `scripts/probe_min_meta.py` — K=K_pool+N stack-add gate
4. `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-pred NULL + load-bearing DGP diagnostic (joint-explains all per-row-FE NULLs)
5. `audit/2026-05-06-synthetic-data-batch.md` — 7-probe batch (today's session-1 NULLs)
6. `audit/2026-05-06-do-all-4-probes.md` — TE-audit CLEAN, α-resweep NULL, sparse-LR/lt-q5 NULL
7. `audit/2026-05-06-blend-and-rho-probes.md` — K=21 blends ruled out + ρ inventory
8. `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB-failure record
9. `audit/2026-05-06-alpha-asymmetry-verification.md` — α verification, harness intro
10. `audit/friction.md` — 5 new tags from this session at top
11. `scripts/pre_submit_diff.py` — MANDATORY before submit

**Harness usage cheatsheet (Rule 19):**
```bash
python scripts/probe.py bote NAME --family X --cost_min N \
    [--std_oof_lift_bp Y] [--prob_useful U]
python scripts/probe.py gate NAME --oof PATH --test PATH
python scripts/probe_min_meta.py --candidates NAME1 NAME2 ...
```

## Today's progress (2026-05-06, both sessions)

**Per-row feature-engineering family CLOSED.** Across both sessions today,
13+ single-base candidates tested at K=21+1 meta gate. All NULL when not
target-derived: NN-with-embeddings (ρ=0.918 most-diverse, NULL), Year×Stint
sparse-LR (ρ=0.844, NULL), within-Race LapTime_Δ q5, lap-mod/id-mod features
(566 bp marginal pattern absorbed), confidence-extreme pseudo-cascade,
multi-target NN with shared trunk, blend aggregators (mean/gmean/rank/trimmed).
**The 5th independent NULL** — `d14_dgp_residuals` (masked-column self-prediction)
— produced a load-bearing diagnostic: across all 4 self-pred targets, OOF RMSE
≈ marginal σ within 3 sig figs. **The synthetic NN-DGP is conditionally
near-independent within rows.** Per-row FE / SSL pretraining cannot break
the K=21 + Path-B ceiling — joint-explains FM-aug12 saturation, Move D NULL,
Day-13/14 alt-axis 4-of-4, TabPFN's 0.944 ceiling.

**The breakthrough.** `inv_laps_until_pit` = LGBM regression on
`1/(1+laps_until_pit)` (computed from PitNextLap labels per Driver-Race-Year
group). K=21+1 OOF +1.899 bp (largest non-meta-derivative single-add measured).
**Path B Compound×Stint over K=22+inv_laps τ=20k → OOF 0.95110** (+2.75 bp
vs PRIMARY). Pre-submit-diff PASS (ρ=0.99753; 53% rows shifted >1e-3).
**HELD pending submission decision.** Mechanistic distinction from
d12_lr_meta failure (LB −4 bp): inv_laps is target-derived (orthogonal),
not convex-combo of pool predictions.

**Day-14 also confirmed:** TabPFN v2.5/v2.6 DEAD (AUC ceiling 0.944);
FM-field-augmentation saturated at 12 fields (Move D / aug16 -0.07 bp NULL).

**Harness installed:** `scripts/probe.py` (bote + gate), `probe_min_meta.py`,
`probe_path_b_K22_invlaps.py`, `probe_target_reform.py` etc. CLAUDE.md
Rule 19 codifies the workflow.

## Falsified or dead — do NOT retry

See `ISSUES.md ## Falsified or dead` (full list, 20+ entries). Highlights:
- Per-row feature-engineering of any kind (5 NULLs jointly explained by `tag: synthetic-dgp-conditionally-near-independent`).
- Meta-derivative-as-base 2-level stacking (d12_lr_meta LB −4 bp confirmed; KD same family).
- TabPFN fine-tuning v2.5 / v2.6.
- FM-field-augmentation beyond 12 fields (Move D aug16, d14 H1 aug15).
- Drop-GBDT pool refactor (d13c −2.5 to −2.6 bp Strat).
- α-calibrated τ-resweep (τ=20k empirically optimal).
- TE fold-leak audit (d2a/d3a CLEAN).
- ρ-alone diversity heuristic (NULL across 3 probes today).

## Next-session first-action — RANKED by EV/cost

### A1 — SUBMISSION DECISION (PI-gated; Rule 1 single-shot approval)

`submission_path_b_K22_invlaps_tau20000.csv`. Top OOF candidate (0.95110,
+2.75 bp). Pre-submit-diff already passed. Target-derived → orthogonal-signal
criterion satisfied → Path B amp expected to fire. **Cost:** 1 slot.
**Action on PI "submit":** `kaggle competitions submit -c playground-series-s6e5
-f submissions/submission_path_b_K22_invlaps_tau20000.csv -m "K=22 Path B
Compound×Stint τ=20k = K=21 + inv_laps_until_pit (target-derived) | OOF +2.75bp"`.

### A2 — More target reformulations (family looks alive; ≤10 min CPU each)

`scripts/probe_target_reform.py` has the scaffold. Add new targets:
- `pit_horizon_multiclass` (4-class softmax: this / 1-2 / 3-5 / >5 laps)
- `reverse_cumcount_pits_in_race`
- `stint_index_within_race`
- `next_pit_lap_number` (absolute lap regression)

EV +0.5 to +2 bp K=21+1 OOF; +1 to +6 bp under Path B amp.

### A3 — Pool composition: STRUCTURED replace, not naive drop

d13c falsified naive drop. Structured swap (drop 2 leak-eaters AND add 2
target-derived bases) untested. ~30 min CPU. EV +1 to +5 bp.

### A4 — External data revisit (Pirelli scrape)

ISSUES leaf 4a still open. Pirelli pit-window per (Compound, Race) historical.
Tier-2 highest absolute EV per Day-8 research. Cost: scrape time + integration.
Aggregate-prior (not row-join) integration pattern. EV +0.5 to +3 bp.

### Research-loop trigger (Rule 7) IF A1 misses on LB

If A1 LB ≤ 0.95049: target-reformulation may be a 2nd false-positive after
meta-derivative. Pause submits, redecompose ISSUES, web-search top finisher
writeups for synthetic-tabular Playground.

## Operating rules (load-bearing)

1. Pre-submit-diff before EVERY submit; ρ < 0.999 mandatory.
2. **Per-row feature engineering is dead** (`synthetic-dgp-conditionally-near-independent`). Don't propose new ones.
3. **ρ alone NOT sufficient for meta-utility** (3 NULLs at low ρ today triangulate this).
4. **Target-derived single-bases pass orthogonality**; meta-derivatives FAIL Path B amp.
5. Strat-only Day-3+ (R1) for primary OOF; public LB row-iid per U3.
6. Cap ≤3 concurrent CPU-heavy probes; schedule cheap probes first.

## Pointers (audit notes added today)

- `audit/2026-05-06-alpha-asymmetry-verification.md` — α verification + harness intro
- `audit/2026-05-06-blend-and-rho-probes.md` — K=21 blends ruled out + ρ inventory
- `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB-failure record
- `audit/2026-05-06-do-all-4-probes.md` — TE-audit / α-resweep / sparse-LR / lt-q5
- `audit/2026-05-06-synthetic-data-batch.md` — 7-probe batch
- `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-pred NULL + DGP diagnostic (load-bearing)
- `scripts/probe.py` / `probe_min_meta.py` — harness
- `scripts/probe_path_b_K22_invlaps.py` — THE breakthrough probe
- `scripts/probe_target_reform.py` — target reformulation scaffold (extensible)
- `scripts/d14_dgp_residuals.py` — DGP-residual probe
