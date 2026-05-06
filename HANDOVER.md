# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are (Day-16 PM, 2026-05-06 evening)

- **PRIMARY** = `d15b_path_b_K22_dae_only_tau20000` LB **0.95059**
  (Day-15 PM advance via parallel branch; DAE swap-noise → LGBM-on-latent
  as 22nd base, Path B Compound×Stint τ=20k). DAE-derived signal is
  legitimate (no target-label leakage).
- **Gap to top-5%** (0.95345): −28.6 bp.
- **Top of LB ~0.955** (PI observation, end of session): leaders likely
  use FEW or a SINGLE model with a structural mechanism we haven't found
  yet. Our stacking-with-target-derived-bases approach was chasing
  inflated OOF (see leakage section below).
- **Submissions used total:** 28/270; 10/10 used today (2026-05-06).
- **Today's branches:**
  - `claude/read-handover-lA8Nr` (Day-16 virgin-axes, 11 probes /
    4 NULL / 1 falsified / 3 KILLED / 2 parked / 1 marginal — no advance)
  - `claude/ml-handover-alignment-xvUN0` (this branch — harness +
    target-reformulation thesis tested + **falsified via strict-OOF
    audit**)

## 🔴 CRITICAL — held candidates INVALIDATED

End-of-day strict-OOF audit on this branch **collapsed all
target-reformulation single-add results 88-100%**:

| candidate | original Δ at K=21+1 | strict-OOF Δ | collapse |
|---|---:|---:|---:|
| reverse_cum | +4.867 bp | −0.005 bp | 100% |
| pit_horizon | +3.191 bp | +0.302 bp | 90% |
| inv_laps_until_pit | +1.899 bp | +0.234 bp | 88% |
| Joint K=21+3 | +7.667 bp | +0.275 bp | 96% |

**Bug:** `compute_targets()` in `scripts/probe_target_reform.py` and
`_v2.py` aggregates per (Driver, Race, Year) group using ALL train
labels — leaking val-row labels into tr-row regression targets via
`total_pits` + `cumsum`. New friction tag
`target-construction-layer-leakage`. Same family as
`d12_lr_meta` 2-level stacking (LB regress on +1.348 bp inflated OOF).

**Held candidates DO NOT submit:**
- `path_b_K22_invlaps_tau{5k,20k,100k}.csv` — 88% leaky
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv` — partially leaky (inv_laps component)
- `path_b_K25_megapool_tau{5k,20k,100k}.csv` — 96% leakage mirage
- `path_b_multilevel_τ_*.csv` — 5 configs NULL anyway

**Held candidates safe (no target-leakage):**
- `d15b_path_b_K22_dae_only_tau{20k,100k}.csv` (PRIMARY + close-second)
- `path_b_K22_d12meta_tau100000.csv` (LB 0.95045, R7-eligible HEDGE)
- `d15c` (ExtraTrees), `d15d` (LGBM-on-KNN) — R5 HEDGE only

Audit: `audit/2026-05-06-target-reform-leakage-audit.md`.

## Read order on session start

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `audit/2026-05-06-target-reform-leakage-audit.md` — **load-bearing** —
   strict-OOF audit collapse table
3. `audit/2026-05-16-d16-virgin-axes-results.md` — Day-16 virgin-axes
   (11 probes, all NULL / falsified / parked)
4. `audit/friction.md` — top tags `target-construction-layer-leakage`,
   `path-b-amp-only-fires-on-meta-arch-not-base-add`,
   `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`,
   `lr-meta-rank-lock-strong-anchor`
5. `scripts/probe.py` — `bote()` + `gate()` harness (Rule 19)
6. `scripts/probe_min_meta.py` — K=21+N stack-add gate
7. `scripts/probe_target_reform_strict_oof.py` — strict-OOF audit pattern
8. `scripts/pre_submit_diff.py` — MANDATORY before submit

## Day-15 PM (read-handover-LgbQ4): NEW PRIMARY LB 0.95059 via DAE

Submission `d15b_path_b_K22_dae_only_tau20000` (52394353 COMPLETE
2026-05-06 15:38) — DAE 768d latent → LGBM-on-latent → K=22 + Path B
Compound×Stint τ=20k. Realised LB amp **1.4×** on +0.715 bp OOF —
load-bearing for the new friction tag
`path-b-amp-only-fires-on-meta-arch-not-base-add`.

DAE artifacts re-usable for any future K_pool+N probe (no need to
retrain): `oof_d15b_lgbm_dae_{full,only}_strat.npy` + test variants.

## Day-16 (read-handover-lA8Nr): virgin-axes complement, all NULL

11 probes covering α/β/δ/ε/ζ/η axes from the d13 problem decomposition
tree. Highlights:
- **α4 GRU sequence on (Driver, Race) lap windows**: std OOF 0.93066,
  ρ=0.919 (most-diverse base of session). K=22+1 LR-meta Δ=−0.043 bp NULL.
  **5th cross-confirmation of `lr-meta-rank-lock-strong-anchor`.**
- **ε2 twin parallel-pool 2-meta blend**: ρ(metaA, metaB)=0.967 real
  disagreement; top-level LR vs single LR-meta(K=11): FALSIFIED Δ=−1.79 bp.
  Friction `twin-pool-2-meta-collapses-rank-info`.
- **δ2/3 conformal isotonic 4 schemes**: All regress −2.5 to −9.6 bp NULL.
  Friction `primary-hier-meta-globally-calibrated`.
- **ζ6 transductive pseudo (full-test soft labels)**: marginal +0.63 bp
  at LR-meta-K22 but −0.30 vs PRIMARY hier. R5 HEDGE only.
- 2 parked, 3 killed (DeepGBM ε4 over-engineered, etc.)

Full audit: `audit/2026-05-16-d16-virgin-axes-results.md`.

## This branch (ml-handover-alignment-xvUN0): harness + target-reform leakage

**Three significant deliverables:**
1. **Harness installed** (`scripts/probe.py`, `probe_min_meta.py`,
   18+ probe scripts). CLAUDE.md Rule 19 codifies BOTE-first / gate-after.
2. **Target reformulation thesis FALSIFIED via strict-OOF audit** (above).
3. **4-tier multi-level Path B (T4a)**: 5 (τ_0, τ_1, τ_2) configs all NULL.
   Simple multi-tier-shrinkage variant doesn't fire Path-B amp.

**Per-row feature engineering family CLOSED** (5 NULLs jointly explained
by `tag: synthetic-dgp-conditionally-near-independent`).

## Falsified or dead — do NOT retry

See `ISSUES.md ## Falsified or dead` (full list). Highlights:
- **target_reformulation_invlaps / pit_horizon / reverse_cum / stintprog**
  — all leaky; strict-OOF audit 88-100% collapse
- **path_b_K22_invlaps_*, path_b_K23_dae_invlaps_*, path_b_K25_megapool_***
  — all built on leaky targets
- **multi_level_path_b_4tier** — 5 configs NULL
- **Day-16 virgin-axes** — 11 of 11 NULL/falsified/killed
- TabPFN v2.5/v2.6, FM-aug16+, drop-GBDT pool refactor, simple K=21
  blends, α-calibrated τ-resweep, multi-target NN, masked-column
  self-prediction (DGP-residual)

## Next-session first-action — RANKED by EV/cost

### A1 — SINGLE-MODEL HYPOTHESIS TEST (PI-directed for next session)

PI hypothesis: leader at LB ~0.955 likely uses ONE strong model with
a structural mechanism we missed. Our 25-base stacking chases inflated
OOF that doesn't transfer.

**Procedure:** train ONE LightGBM (or CatBoost) with a wide feature
set including raw + strict-OOF target reformulations as FEATURES (not
separate bases). Measure standalone OOF AUC. If it beats ~0.945
(baseline+) significantly → "single model" path alive. If standalone
is unchanged → the +50 bp gap requires a structural insight we
haven't found.

Inputs available:
- Raw features (11 numeric + 3 cat)
- `oof_target_reform_{reverse_cum,pit_horizon,inv_laps}_strict_strat.npy`
  (strict-OOF, leak-free per `probe_target_reform_strict_oof.py`)
- DAE 768d latent (`oof_d15b_lgbm_dae_only_strat.npy`)

Cost: ~10-30 min for one wide-feature LightGBM 5-fold + standalone AUC.

### A2 — Pirelli external data scrape (ISSUES leaf 4a; untouched)

Aggregate-prior pattern (per (Compound, Race, Year) historical),
NOT row-join (d2 row-join failed at 5.6% match rate). Tier-2 EV per
Day-8 research. EV +0.5 to +3 bp.

### A3 — Examine raw data structure for missed leak

`id_mod_1000` 568 bp marginal span absorbed by GBDT interactions when
added as feature. Question: is there a **non-feature** structure (row
order, group ordering) that encodes pit_next_lap directly?

### A4 — Web search top-finisher Playground writeups

Pattern-match leader's "single model at 0.955" against published
synthetic-tabular Playground writeups.

### Meta-arch redesign (still alive at structural level)

Untested in T4a's simple multi-tier:
- Non-Gaussian shrinkage prior (Beta-Binomial / Student-t)
- Yao/Vehtari covariance-Σ BMA (LKJ + GP prior)
- Alternative segmentation cross (Year×Compound, Compound×TyreLife_q5,
  Driver-cluster × Stint)

### Research-loop trigger (Rule 7)

If A1 + A2 + A3 all NULL: pause submits, re-decompose ISSUES.md (3+
plateau-days now confirmed; per-row FE family closed; target-reform
family closed via leakage-audit; meta-arch redesign still untouched
at proper Bayesian level).

## Operating rules (load-bearing)

1. **Pre-submit-diff before EVERY submit**; ρ < 0.999 mandatory.
2. **Strict-OOF audit any per-group y-derived target before submission**
   (`tag: target-construction-layer-leakage`).
3. **Per-row feature engineering is dead**
   (`tag: synthetic-dgp-conditionally-near-independent`).
4. **ρ alone NOT sufficient for meta-utility** (5 cross-confirmations).
5. **Path B amp does NOT fire on base-adds** (1.4× realised, not 6-11.6×;
   `tag: path-b-amp-only-fires-on-meta-arch-not-base-add`).
6. **Path B amp REQUIRES orthogonal signal** (meta-derivatives FAIL;
   `tag: path-b-amp-needs-orthogonal-signal-not-meta-derivatives`).
7. Strat-only Day-3+ (R1) for primary OOF; public LB row-iid per U3.
8. Cap ≤3 concurrent CPU-heavy probes; schedule cheap probes first.

## Pointers (audit notes added today)

- `audit/2026-05-06-target-reform-leakage-audit.md` — **load-bearing**
- `audit/2026-05-16-d16-virgin-axes-results.md` — Day-16 11-probe NULL audit
- `audit/2026-05-06-alpha-asymmetry-verification.md` — α verification + harness intro
- `audit/2026-05-06-blend-and-rho-probes.md` — K=21 blends ruled out + ρ inventory
- `audit/2026-05-06-path-b-K22-d12meta.md` — meta-derivative LB-failure record
- `audit/2026-05-06-do-all-4-probes.md` — TE-audit / α-resweep / sparse-LR / lt-q5
- `audit/2026-05-06-synthetic-data-batch.md` — 7-probe synth-data batch
- `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-pred + DGP diagnostic
- `audit/2026-05-15-d15-4branch-results.md` — 4-branch + B-GPU + DAE submit audit
- `scripts/probe.py` + `probe_min_meta.py` + `probe_target_reform_strict_oof.py`
