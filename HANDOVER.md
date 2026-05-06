# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are

- **PRIMARY** = `d15b_path_b_K22_dae_only_tau20000` LB **0.95059**
  (Day-15 PM advance via parallel branch; DAE swap-noise → LGBM-on-latent
  as 22nd base, Path B Compound×Stint τ=20k). DAE-derived signal is
  legitimate (no target-label leakage).
- **HEDGE held**: `path_b_K22_d12meta_tau100000.csv` (LB 0.95045, R7 eligible).
- **🔴 Held candidates INVALIDATED** by 2026-05-06 strict-OOF audit:
  - `path_b_K22_invlaps_tau20000.csv` — 88% of OOF lift was leakage
  - `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv` — partially leaky (inv_laps component)
  - `path_b_K25_megapool_tau{5k,20k,100k}.csv` — 96% leakage mirage
  - **Do NOT submit any of these.** See `audit/2026-05-06-target-reform-leakage-audit.md`.
- **Gap to top-5%** (0.95345): −28.6 bp from PRIMARY.
- **Top of LB ~0.955** (PI observation): leaders likely use FEW or a
  SINGLE model with a structural mechanism we haven't found yet.
  This INVALIDATES our stacking-with-target-derived-bases approach.
- **Submissions used total**: 25/270 cumulative; 10/10 used today
  (2026-05-06); resume tomorrow.

## Read order on session start

1. `CLAUDE.md` — state block + Rules 1-19
2. `audit/2026-05-06-target-reform-leakage-audit.md` — **load-bearing** —
   88-100% collapse of target-reform OOF lifts under strict-OOF construction
3. `audit/friction.md` — top friction tags `target-construction-layer-leakage`,
   `path-b-amp-only-fires-on-meta-arch-not-base-add`, `path-b-amp-needs-
   orthogonal-signal-not-meta-derivatives`
4. `scripts/probe.py` — bote+gate harness (Rule 19)
5. `scripts/probe_min_meta.py` — K=21+N stack-add gate
6. `scripts/probe_target_reform_strict_oof.py` — strict-OOF audit pattern
7. `scripts/pre_submit_diff.py` — MANDATORY before submit

## Today's progress (2026-05-06)

**Three independent breakthroughs** (per branch parallel; main merged):
- **DAE-LGBM (parallel branch)**: K=22 + Jahrer swap-noise DAE → +1.0 bp
  LB → **NEW PRIMARY 0.95059**. Realised amp 1.4× (per friction `path-b-
  amp-only-fires-on-meta-arch-not-base-add`).
- **Experimentation harness installed** (this branch): `scripts/probe.py`
  (bote + gate), `probe_min_meta.py`, 18+ probe scripts. CLAUDE.md
  Rule 19 codifies BOTE-first / gate-after / many-cheap-probes workflow.
- **Target reformulation thesis tested + FALSIFIED**: 13+ probes
  this session converged on inv_laps_until_pit / pit_horizon /
  reverse_cum as "biggest single-add wins" (+1.9 to +7.7 bp K=21+1).
  PI flagged inconsistency with leader-ranks-with-few-models; strict-OOF
  audit confirmed 88-100% collapse. **All target-reform-based held
  submissions are leakage mirages.**

**Per-row feature engineering family CLOSED** (5 NULLs jointly explained
by `tag: synthetic-dgp-conditionally-near-independent`).

**4-tier multi-level Path B (T4a)**: 5 (τ_0, τ_1, τ_2) configs all NULL.
Meta-arch redesign axis still alive in principle (Student-t shrinkage,
Yao/Vehtari BMA, multi-cohort blend untested) but the simple
multi-tier-shrinkage variant doesn't fire Path-B amp.

## Falsified or dead — do NOT retry

See `ISSUES.md ## Falsified or dead` (full list). Highlights from today:
- **target_reformulation_invlaps / pit_horizon / reverse_cum / stintprog**
  — all leaky; strict-OOF audit 88-100% collapse.
- **path_b_K22_invlaps_*, path_b_K23_dae_invlaps_*, path_b_K25_megapool_***
  — all built on leaky targets; do not submit.
- **multi_level_path_b_4tier** — 5 configs NULL.
- **multi-target NN with shared trunk** — +0.086 bp NULL.
- (See HANDOVER section below for prior-day list.)

## Next-session first-action — RANKED by EV/cost

### A1 — SINGLE-MODEL HYPOTHESIS TEST (PI-directed; **next session**)

PI hypothesis: leader at LB ~0.955 likely uses ONE strong model
with a structural mechanism we missed. Our 25-base stacking
chases inflated OOF that doesn't transfer.

**Procedure**: train ONE LightGBM (or CatBoost) with a wide feature
set including raw + strict-OOF target reformulations as FEATURES
(not separate bases). Measure standalone OOF AUC. If it beats ~0.945
(baseline+, similar to e3_hgbc 0.94876) by significant margin → the
"single model" path is alive. If standalone is unchanged → the +50bp
gap to leader requires something we haven't found (data trick,
different model class, or structural insight).

Inputs available:
- Raw features (11 numeric + 3 cat)
- `oof_target_reform_reverse_cum_strict_strat.npy`,
  `pit_horizon_strict`, `inv_laps_strict` (strict-OOF, leak-free
  per probe_target_reform_strict_oof.py)
- DAE 768d latent (via the parallel branch's d15b_dae_encoder.py)
- Available targets to predict: PitNextLap, plus the strict-OOF
  derived target reformulations as ADDITIONAL features

Cost: ~10-30 min for one wide-feature LightGBM 5-fold + standalone
AUC + ρ vs PRIMARY. Decisive on whether "single model" can lift.

### A2 — Pirelli external data scrape (ISSUES leaf 4a; untouched)

Aggregate-prior pattern (per (Compound, Race, Year) historical),
NOT row-join (d2 row-join failed at 5.6% match rate). Tier-2 highest
absolute EV per Day-8 research. Untested. ~scrape + integration time.
EV +0.5 to +3 bp.

### A3 — Examine raw data structure for missed leak

`id_mod_1000` had 568 bp marginal span (Day-14 audit) — but that
was absorbed by GBDT interactions when added as a feature. The
question worth re-asking: is there a **non-feature** structure
(row order, group ordering, time-stamp embedded somewhere) that
encodes pit_next_lap directly and the leaders found?

Check: pit_next_lap target rate by `id // N` for various N. By
position within (Driver, Race, Year) group. By exact order in the
CSV file. By Driver_Race_Year × LapNumber-mod patterns the GBDT
can't capture.

### A4 — Web search top-finisher Playground writeups

If A1 doesn't reveal a single-model path, web-search PS6 series
1st-place writeups for synthetic-tabular pattern matches. We did
this in session 1 today (SOSTA paper found, but its features
require raw LapTime which the host removed).

### Research-loop trigger (Rule 7)

If A1 + A2 + A3 all NULL in next session: pause submits, re-decompose
ISSUES.md (5 plateau-days now confirmed; per-row FE family closed;
target-reform family closed via leakage-audit; meta-arch redesign
still untouched at structural level).

## Operating rules (load-bearing)

1. **Pre-submit-diff before EVERY submit**; ρ < 0.999 mandatory.
2. **Strict-OOF audit any per-group y-derived target before submission**.
   New friction: `target-construction-layer-leakage`.
3. **Per-row feature engineering is dead** (`synthetic-dgp-conditionally-
   near-independent`).
4. **ρ alone NOT sufficient for meta-utility** (3 NULLs at low ρ today
   triangulate this; rho-alone-not-sufficient-for-meta-utility).
5. **Path B amp does NOT fire on base-adds** (1.4× realised, not 6-11.6×;
   `path-b-amp-only-fires-on-meta-arch-not-base-add`).
6. **Path B amp REQUIRES orthogonal signal** (`path-b-amp-needs-orthogonal-
   signal-not-meta-derivatives`); meta-derivatives FAIL.
7. Strat-only Day-3+ (R1) for primary OOF; public LB row-iid per U3.
8. Cap ≤3 concurrent CPU-heavy probes; schedule cheap probes first.

## Pointers (audit notes from this session)

- `audit/2026-05-06-target-reform-leakage-audit.md` — **load-bearing**
- `audit/2026-05-06-alpha-asymmetry-verification.md` — α verification + harness intro
- `audit/2026-05-06-blend-and-rho-probes.md` — K=21 blends ruled out + ρ inventory
- `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB-failure record
- `audit/2026-05-06-do-all-4-probes.md` — TE-audit / α-resweep / sparse-LR / lt-q5
- `audit/2026-05-06-synthetic-data-batch.md` — 7-probe batch
- `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-pred + DGP diagnostic
- `scripts/probe.py` / `probe_min_meta.py` — harness
- `scripts/probe_target_reform_strict_oof.py` — strict-OOF target audit
