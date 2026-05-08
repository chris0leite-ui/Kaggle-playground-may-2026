# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are (2026-05-08; 23 days to deadline)

- **PRIMARY** = `d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000`, LB **0.95368**
  (Day-18 PM-late). K=27 = K=21 + v4 (CB yekenot transfer) + h1d (RealMLP
  yekenot full) + d16 (orig cont_only) + d18 (chain decomp v1) + E2
  (preimage kNN) + F2 (constraint violations). Path-B Compound×Stint
  hier-meta τ=100k. Realised OOF→LB gap −6.4 bp.
- **Rank #98 of 893 = top 11%.** Top-5% boundary 0.95405 (gap **−3.7 bp**).
  Leader MILANFX 0.95476 (gap −10.8 bp).
- **Submissions:** 39/270 used.
- **PRIMARY trail (LB ladder):**
  - 0.95354 `d17_path_b_K23_v4_h1d_tau100000` (Day-17 PM)
  - 0.95345 `d17_K24_d18pool_h1d` (Day-17 mid)
  - 0.95149 `d18_path_b_K23_d16_d18_tau20000` (Day-17 AM)
  - 0.95089 `d16_path_b_K22_continuous_only_tau20000` (Day-16)
  - 0.95059 `d15b_path_b_K22_dae_only_tau20000` (Day-15)

## State of axes (post-Day-19 overnight)

In-pool axes empirically exhausted on K=27 with v4+h1d anchors:

- **A axis (orthogonal-base-add):** 7× rank-lock confirms. Only structurally
  distinct sub-axis remaining is **A1 sequence-level** (HMM Compound transitions
  + AR(1) within-stint TyreLife). ~2-3 h CPU.
- **B axis (anchor-swap on shared FE):** B2 xgb_v4 closed REDUNDANT at K=27
  (+0.143 bp NULL). B1 RealMLP n_ens=24 untested (~3.5 h, +1-3 bp single-model
  predicted). B4 per-Year v4-specialists untested (30 min, ±2 bp).
- **C axis (meta-arch redesign):** **CLOSED.** 9 variants tested across
  Days 14-19; Compound × Stint with plain τ shrinkage τ=100k IS the local
  optimum. Most recently: C1 Yao/Vehtari covariance-modulated Path-B (V3
  τ ∈ {10k, 50k, 200k}) all regress −0.47 to −0.59 bp.
- **D axis (external data):** **CLOSED.** D1 debashish historical-priors
  closed null-by-pre-flight (PI + harness convergence). D2 capped by H2
  audit (1.4 % match). D3 (aadigupta) already in K=27 pool.

## 🎯 Next-step priority list

Ordered by EV/cost given empirical exhaustion above.

| # | Move | Cost | Notes |
|---|---|---|---|
| 1 | **A1 sequence-level fingerprinting** | 2-3 h CPU | Only mechanism axis still structurally orthogonal on K=27. Predicted +1-3 bp. ISSUES leaf 7h. |
| 2 | B1 RealMLP n_ens=24 (h1d used 4) | 3.5 h GPU | Yekenot's published recipe. +1-3 bp standalone. |
| 3 | B4 per-Year v4 specialists | 30 min GPU | d12 found 2023 easiest. ±2 bp. |
| 4 | **R5 HEDGE prep (final-window)** | 30 min CPU | List OOF-best rejected for public regression; HEDGE ladder already populated. |
| 5 | Wrap-up posture | n/a | Top-11 % achieved; reserve compute for next Featured comp. Durable artifacts already shipped (LR-diag suite, BOTE harness, decisions.jsonl). |
| 6 | FastF1 lap-by-lap pit-call hard-join | 1-2 days | Only single-mechanism path to top-5 %. Predicted +10-30 bp. Cost prohibitive vs days remaining. |

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



---

## Archived per-branch PM detail

Per Rule 5 file-size guard (≤150 lines), per-branch PM sections live in
`audit/archive-*-handover-*-sections.md`. Load-bearing summaries remain
in `## Where we are` and `## 🔴 CRITICAL` above. Read archives only for
detailed reproductions.

- `audit/archive-2026-05-07-handover-day15-17-pm-sections.md` — Day-15/16/17
- `audit/archive-2026-05-07-handover-pm-sections.md` — Day-17/18 sibling branches
- `audit/archive-2026-05-08-handover-day19-pm.md` — Day-19 overnight (B2/A5/C1)

