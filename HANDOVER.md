# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are

- **PRIMARY** = `d13e_compound_stint_tau20000` LB **0.95049** (unchanged across two
  sessions; Day-13 PM advance still standing).
- **HEDGE candidates held** (no submission yet):
  - `path_b_K22_invlaps_tau20000.csv` — OOF **0.95110** (+2.75 bp);
    largest OOF advance ever measured; **target-derived** signal source
    (NOT meta-derivative); ρ=0.99753 vs PRIMARY; flips 57/96 (ratio 0.594).
    **Top submission candidate.**
  - `path_b_K22_invlaps_tau100000.csv` — OOF 0.95106 (+2.32 bp); flips
    45/189 (asymmetric, echoes d13 Stint Path B which lifted +7 bp).
  - `path_b_K22_d12meta_tau100000.csv` — LB 0.95045 (submitted, regressed
    −4 bp); R7-eligible HEDGE since flip count 188 < 200.
  - `d12_lr_meta` — OOF 0.95073, ρ=0.996; near-tie hedge.
- **Gap to top-5%** (0.95345): −29.6 bp from current PRIMARY. If the
  τ=20k candidate hits even median Path B amp (8×), LB → ~0.95271
  (top-5% range). Bull case (11.6×) → 0.95370 (top-1% range).
- **Submissions used total:** 25/270. Today (2026-05-06): 1 submit (failed
  K=22+d12_lr_meta path).

## Read order on session start (skip default; this is the synthesis)

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `scripts/probe.py` — entry point. `bote()` for BOTE, `gate()` for uniform gate report
3. `scripts/probe_min_meta.py` — K=21+N stack-add gate
4. `audit/2026-05-06-synthetic-data-batch.md` — most recent batch tabulation
5. `audit/2026-05-06-do-all-4-probes.md` — preceding 4-probes batch (TE-audit clean, α-resweep NULL)
6. `audit/2026-05-06-blend-and-rho-probes.md` — blend rule-out + ρ inventory
7. `audit/2026-05-06-alpha-asymmetry-verification.md` — Path B α verified
8. `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB-failure record
9. `audit/2026-05-13-d13-path-b-hier-meta.md` — Path B mechanism (load-bearing)
10. `audit/friction.md` — load-bearing tags from this session at top
11. `scripts/pre_submit_diff.py` — MANDATORY before submit

**Harness usage cheatsheet (Rule 19):**
```bash
python scripts/probe.py bote NAME --family X --cost_min N \
    [--std_oof_lift_bp Y] [--prob_useful U]
python scripts/probe.py gate NAME --oof PATH --test PATH
python scripts/probe_min_meta.py --candidates NAME1 NAME2 ...
```

## Today's progress (2026-05-06)

**Mechanism boundary clarified.** Sessions today and yesterday tested 13+
single-base candidates against the K=21 LR-meta. All NULL or negative
EXCEPT two: meta-derivative-as-base (+1.348 bp d12_lr_meta, +0.526 bp KD)
and **target-derived single-task LGBMs** (+1.899 bp inv_laps_until_pit).
The d12_lr_meta candidate was submitted and **regressed −4 bp on LB**,
falsifying meta-derivative additions and producing the friction tag
`path-b-amp-needs-orthogonal-signal-not-meta-derivatives`. Three more
friction tags codified today; CLAUDE.md Rule 19 + harness in place.

**The breakthrough.** `inv_laps_until_pit` = LGBM regression on
`1/(1+laps_until_pit)` (computed from PitNextLap labels). K=21+1
yields +1.899 bp OOF (largest non-meta-derivative single-add measured).
**Path B Compound×Stint over K=22+inv_laps τ=20k → OOF 0.95110**
(+2.75 bp vs PRIMARY). Pre-submit-diff PASS (ρ=0.99753; 53% rows
shifted >1e-3). HELD pending PI submission decision.

**Synth-data lens consequence.** Mod-K patterns (LapNumber_mod_10
marginal span 566 bp) DO NOT translate to predictive lift —
absorbed by GBDT feature interactions. Physical-feature normalisations
(within-Race LapTime_Delta) similarly ruled out. **The lever is
target reformulation, not feature engineering.**

## Falsified or dead — do NOT retry

- All 12+ NULL candidates from this session are listed in
  `ISSUES.md ## Falsified or dead`. Highlights:
- Meta-derivative-as-base via 2-level stacking (d12_lr_meta LB
  −4 bp confirmed; KD same family).
- Single-base FE additions in any class (LGBM, FM, sparse-LR,
  rule-residual, NN-with-embeddings, multi-target NN). 13/13 NULL
  at meta gate when not target-derived.
- ρ-alone diversity argument (NN ρ=0.918, year_stint sparse-LR
  ρ=0.844, stint_progress ρ=0.252 — all NULL at meta gate).
- α-calibrated τ-resweep (PRIMARY's τ=20k empirically optimal).
- TE fold-leak in d2a/d3a (CLEAN — no leakage, OOF discipline correct).
- Driver-cluster Path B cohort axis (-0.4 to -0.9 bp).
- Lap-mod / id-mod features in LightGBM (566 bp marginal span absorbed).
- K=21 simple aggregators (mean/gmean/rank_mean/trimmed -19 to -32 bp).

## Next-session first-action — RANKED by EV/cost

### A1 — SUBMISSION DECISION (PI-gated; Rule 1 single-shot approval needed)

`submission_path_b_K22_invlaps_tau20000.csv`. Top OOF candidate
(0.95110, +2.75 bp). Pre-submit-diff already passed. Path B
family-conditional amp predicts +1.25 to +32 bp LB. The
target-derived signal source mechanistically separates this from
the d12_lr_meta failure (orthogonal-signal criterion satisfied).
**Cost:** 1 slot. **Action:** PI says "submit" → run
`kaggle competitions submit -c playground-series-s6e5 -f
submissions/submission_path_b_K22_invlaps_tau20000.csv -m "..."`.

### A2 — Other target reformulations (untested, family looks alive)

`scripts/probe_target_reform.py` already has the LGBM scaffolding.
Add NEW targets (one per probe) and re-gate:
- `pit_horizon_multiclass`: 4-class softmax {this lap / 1-2 / 3-5 / >5}.
- `reverse_cumcount_pits_in_race`: # of pits remaining for this driver.
- `stint_index_within_race`: # of stints completed so far.
- `next_pit_lap_number`: regression on absolute lap number of next pit.
EV per probe: +0.5 to +2 bp single-add at meta; +1 to +6 bp under
Path B amp (target-derived qualifies). ≤10 min CPU each.

### A3 — Pool composition: REPLACE not augment (untouched lever)

d13c falsified naive drop-leak-eaters (-2.5 bp Strat). But a STRUCTURED
swap (drop 2 of {e5_optuna_lgbm, cb_slow-wide-bag, e1_cb_sub} AND
add 2 target-derived bases like inv_laps_until_pit + a hypothetical
new one) is untested. ~30 min CPU. EV +1 to +5 bp.

### A4 — Driver-cluster k-means CAVEAT (cohort axis exhausted; FE axis untouched)

Driver-cluster Path B cohort failed; **but** Driver-cluster as a
ONE-HOT FEATURE in a sparse-LR or LGBM base hasn't been tested.
~10 min. Low-EV per ρ-alone-not-sufficient pattern.

### Research-loop trigger (Rule 7) IF A1 misses on LB

If A1 LB ≤ 0.95049 (no advance): the target-reformulation lever is
the second meta-derivative-failure candidate. Pause submits; web-search
top-finishers' approaches on PS6 series; re-decompose ISSUES.md.

## Operating rules (load-bearing)

1. Pre-submit-diff before EVERY submit; ρ < 0.999 mandatory.
2. ρ alone NOT sufficient for meta-utility (3 probes today triangulate this).
3. Target-derived candidates are the only single-base addition family
   producing visible OOF lift this week. Treat them as PASS for
   Path-B-amp-eligibility; meta-derivatives FAIL.
4. Strat-only Day-3+ (R1) for primary OOF; public LB row-iid per U3.
5. Cap ≤3 concurrent CPU-heavy probes; schedule cheap probes first.
6. Submit budget 25/270; 17 days × 10 = 170 remaining (PI calendar).

## Pointers

- `audit/2026-05-06-alpha-asymmetry-verification.md` — α verification + harness intro
- `audit/2026-05-06-blend-and-rho-probes.md` — K=21 blends ruled out + ρ inventory
- `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB failure
- `audit/2026-05-06-do-all-4-probes.md` — TE-audit clean, α-resweep NULL, sparse-LR/lt-q5 NULL
- `audit/2026-05-06-synthetic-data-batch.md` — 7-probe synth-data batch
- `scripts/probe.py` — bote+gate harness
- `scripts/probe_min_meta.py` — K=K_pool+N stack-add gate
- `scripts/probe_path_b_K22_invlaps.py` — Path B over K=22 with inv_laps (THE breakthrough probe)
- `scripts/probe_target_reform.py` — target reformulation scaffold (extensible)
