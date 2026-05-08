# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are (Day-17 PM, 2026-05-07 evening)

- **PRIMARY** = `d17_path_b_K23_v4_h1d_tau100000` LB **0.95354** (Day-17 PM
  advance via `claude/optimize-model-performance-rruC2`, scored 2026-05-07
  13:27 UTC). K=23 = K=21 + `p1_single_cb_v4_gpu` (yekenot transfer
  recipe; see CB-axis arc below) + `d17_h1d_yekenot_full` (RealMLP-
  pytabkit replication imported from `claude/read-handover-62BCt`),
  Path-B Compound×Stint hier-meta τ=100k. **Realised OOF→LB gap −6.1 bp.**
- _Previous PRIMARIES (Day-17 trail):_
  - `d17_K24_d18pool_h1d` LB 0.95345 (research-branch midday;
    K=21 + d16_cont_only + p1_single_cb_v3 + h1d).
  - `d18_path_b_K23_d16_d18_tau20000` LB 0.95149 (research-branch AM;
    K=21 + d16_cont_only + d18_chain_decomp).
  - `d16_path_b_K22_continuous_only_tau20000` LB 0.95089 (yesterday).
  - `d15b_path_b_K22_dae_only_tau20000` LB 0.95059 (Day-15 PM).
- **Rank: #98 of 893 = top 11%.** Tied with [QWERTY] Roman Rozen at
  0.95354. Top-5% boundary moved up 60 bp today: was 0.95345 (stale),
  now **0.95405** (rank 44). **Gap to top-5%: −51 bp.** Leader MILANFX
  0.95476 (gap −122 bp).
- **Submissions used total:** 38/270; today 7/10.
- **Branches active recently (Day-17):**
  - `claude/optimize-model-performance-rruC2` — **NEW PRIMARY LB 0.95354**
    via CB-axis closure (v3→v4 yekenot transfer, +20.7 bp standalone OOF).
  - `claude/read-handover-62BCt` — h1d (yekenot RealMLP replication,
    OOF 0.95257) + K=24+h1d LB 0.95345 (intermediate Day-17 PM PRIMARY).
  - `claude/reverse-engineer-data-generation-Hu8EK` — d18 chain_decomp
    DGP probe (K=21+1 +7.37 bp) + d18 K=23 LB 0.95149 (Day-17 AM PRIMARY).
  - `claude/research-agentic-kaggle-W6IAP` — yekenot recipe extraction
    + h1d full-recipe replication.
  - `claude/autoencoder-synthetic-data-pEMB6` — d16 cont_only LB 0.95089.

## ⭐ Day-17 PM CB-axis arc + new PRIMARY

This branch (`claude/optimize-model-performance-rruC2`) closed the CB
recipe axis with a +20.7 bp single-model lift via yekenot FE transfer.
Full audit: `audit/2026-05-07-d17-cb-v4-yekenot-transfer.md`.

| Stage | OOF | LB | Key fact |
|---|---:|---:|---|
| LGBM v3 honest (Day-17 AM, P1) | 0.94563 | n/a | single-model honest ceiling |
| **CB v3 (research-recipe)** | 0.94993 | **0.95143** | +43 bp over LGBM v3; K=21+1 +12.06 bp |
| **CB v4 (yekenot transfer + orig-aug)** | **0.95200** | (held) | +20.7 bp over v3; K=21+1 **+24.21 bp** |
| K=22 v4 Path-B C×S τ=20k | 0.95319 | (held) | Path-B amp +0.39 bp (friction re-confirmed) |
| **K=23 v4+h1d Path-B τ=100k = NEW PRIMARY** | **0.95415** | **0.95354** | submitted; gap to LB top −122 bp; rank 98/893 |

**Recipe transfer items 2/3/4 (floor-cat, count-encoding, KBins) fired
on CatBoost despite research-branch audit caveat that they are
"NN-specific."** New friction tag candidate
`yekenot-floor-count-kbins-fires-on-gbdt-too`.

## 🎯 Next-step priority list (Day-18+)

To close the −51 bp gap to top-5% (boundary 0.95405), ranked by EV/cost.

| # | Move | Cost | Predicted LB Δ | Notes |
|---|---|---|---:|---|
| 1 | **K=25 = K=21 + v4 + h1d + d16 + d18** | 10 min CPU | +3 to +8 bp | Cheapest near-term; combines all today's bases |
| 2 | 3-seed bag of v4 (seeds 42/13/71) | 75 min Kaggle GPU | +1 to +3 bp | Variance reduction on load-bearing base |
| 3 | XGB with v4-recipe FE | 30 min Kaggle GPU | +5 to +15 bp | New model class on same FE; ρ < 0.97 likely |
| 4 | RealMLP n_ens=24 (h1d ran 4) | 3.5 h CPU/GPU | +2 to +5 bp standalone | Yekenot's published; +1-3 bp at K-meta |
| 5 | **FastF1 lap-by-lap pit-call hard-join** | 1-2 days | +10 to +30 bp | HANDOVER A4; only single-mechanism path to top-5% |
| 6 | Pirelli tyre-curve scrape | 1-2 days | +10 to +30 bp | Same axis as #5 |
| 7 | Per-Year specialists with v4 recipe | 30 min Kaggle GPU | ±5 bp | d12 found 2023 is easiest segment |
| 8 | Cross-segmentation Path-B (Y×S, R×C) on K=23 v4+h1d | 20 min CPU | +0 to +3 bp | d14 falsified Y-axis without v4 |
| **9** | **Path-B Compound × Year on K=24 (NEW; spec ready)** | **15 min CPU** | **+2 to +10 bp** | **Spec at `audit/2026-05-07-pathb-compound-year-probe-plan.md`. Motivated by LR-leverage Probe 5: per-Year LR specialists give +1081 bp on MEDIUM_2023, +865 on SOFT_2023, +725 on HARD_2023. Cohort-conditional DGP signal that Compound×Stint segmentation missed for 17 days. Path-B-amp-eligible (meta-arch redesign).** |

Items 1-3 sum to ~+10-25 bp predicted; still below the 51 bp gap.
Items 5-6 are the only single-mechanism path. **Item 9 is the highest-EV
new probe to come out of the LR-leverage session and is fastest among
non-external-data items at 15 min CPU; full spec + 6-Q + sealed-prediction
template are in the audit doc.**

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

Per Rule 5 file-size guard (HANDOVER cap ≤ 150 lines), the detailed
per-branch PM sections from Day-15/16/17 have been moved to:
`audit/archive-2026-05-07-handover-day15-17-pm-sections.md`

The detailed per-branch PM sections from Day-17/18 (multiple sibling
branches: ensemble-logistic-regression-research-MbLKu, autoencoder-
synthetic-data-pEMB6, read-kaggle-handover-rsi2Q, read-handover-62BCt,
read-handover-62BCt phase-A, review-handover-solutions-oE78b, reverse-
engineer-data-generation-Hu8EK) have been moved to:
`audit/archive-2026-05-07-handover-pm-sections.md` (~580 lines).

Load-bearing summaries remain in `## Where we are` and `## 🔴 CRITICAL`
above. Read the archives only for archaeology / detailed reproductions.

## Day-19 PM ml-competition-analysis-rwD3f (overnight research, 0 LB submits)

PI directive 2026-05-07 evening: "do all (B2 XGB-v4 K=27 verify, A5
LGBM-v4-fs anchor-modify proxy, C1 Yao/Vehtari covariance-modulated
Path-B BMA), in sequence, skip my predictions, you have all night."

PRIMARY unchanged: `d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000`
LB **0.95368**. 0 submits this branch.

### What ran + verdicts

| probe | OOF / lift | gate verdict |
|---|---:|---|
| **D1** debashish historical-priors join | n/a (didn't run) | **SKIP pre-flight**: harness 0.20 bp / 0.004 bp/min; PI gut −2 to 0 NULL agreed |
| **B2** xgb_v4 add at K=27 (re-verify) | K=21+7 LR-meta 0.95430 vs K=21+6 0.95428 | **+0.143 bp NULL**, ρ=0.987, |w|=0.39 — `gbdt-class-redundant-on-shared-FE` confirmed at K=27 |
| **A5** LGBM-v4-fs (anchor-modify proxy) | std OOF **0.94510**; K=21+7 LR-meta 0.95427 vs K=21+6 0.95428 | **−0.106 bp NULL**, **7th cross-confirmation `lr-meta-rank-lock-strong-anchor`**; fs aggregates ADD NOISE not signal at LGBM class without orig-aug |
| **C1** Yao/Vehtari covariance-modulated Path-B | V0 LR 0.95428; V1 Path-B τ=100k 0.95432 (matches PRIMARY ✓); V2 plain BMA 0.95390 (−4.2 bp); V3 covariance-Σ τ ∈ {10k, 50k, 200k} → 0.95426 / 0.95427 / 0.95427 | **V3 FALSIFIED (−0.47 to −0.59 bp regress)**. Meta-arch redesign family closed (9th variant tested over Days 14-19) |

### Three honest takeaways

1. **External-data axis empirically closed** for our budget. D1 SKIP'd
   on harness + PI agreement. D2 capped by H2 audit (1.4% match,
   0.55 TyreLife correlation ceiling). D3 (aadigupta) already in K=27
   pool. No public top-LB kernel demonstrably uses real-world external
   data beyond aadigupta and a debashish historical priors join (deployed
   only by Rozen at LB 0.95354).
2. **Anchor-modify pathway closes at LGBM class.** A5-light proxy nulls
   at K=27+1 stack-add. fs_cum_pits's +13.73 bp standalone-vs-raw-14
   lift (Day-17 audit) does not survive integration into v4 yekenot
   recipe. CB-GPU A5-full deferred (probability ~10-15% of lift; kernel
   scaffold at `kernels/a5-cb-v4-fs-gpu/` if PI wants to test CTR
   mechanism).
3. **B2 redundancy holds at K=27.** xgb_v4 was REDUNDANT at K=24 per
   prior sibling-branch audit; this session re-verified at K=27 (+0.143
   bp NULL). Rules out re-adding xgb_v4 to PRIMARY.

### Calibration loop entries (`audit/decisions.jsonl`)

| name | PI sealed | agent expected | actual | note |
|---|---:|---:|---:|---|
| d19_historical_priors_debashish | −1 bp | +0.20 bp | NULL pre-flight | SKIP'd; D-axis closure |
| (B2/A5/C1 — no PI seal per directive; Rule 26b omission flagged) | — | family default | varies | calibration column blank for PI-vs-agent |

### Next-session priorities (post-overnight)

In-pool axes empirically exhausted on K=27 with v4+h1d anchors:
- **A axis (orthogonal-base-add)**: 7× rank-lock confirmations; only
  structurally-distinct axis remaining is sequence-level (A1: HMM
  Compound transitions + AR(1) within-stint TyreLife). ~2-3h CPU.
  Family `single_base_fe_addition` priors (P=0.05, midpoint 0.5 bp);
  override to `meta_arch_redesign-adjacent` (P=0.30, midpoint 4 bp)
  given structural orthogonality argument.
- **B axis (anchor-swap on shared FE)**: B2 closed REDUNDANT.
  B1 RealMLP n_ens=24 (~3.5h, +1-3 bp single-model OOF predicted) and
  B4 per-Year v4-specialists (30 min, ±2 bp predicted) untested.
- **C axis (meta-arch redesign)**: C1 in flight; if NULL, family closes.
- **D axis**: closed pre-flight + audit.

If C1 NULL (likely per priors): **strategy-critic-loop fires** (Rule 14)
— "is the criterion still private LB rank, or is it now 'understand
ceiling on this pool'?" Re-decompose ISSUES.md per Rule 18. Honest
options:
1. **A1 sequence-level fingerprinting** (~2-3h CPU; only mechanism axis
   not yet attempted on K=27).
2. **Wrap-up posture**: top-11% achieved (LB 0.95368, rank #98/893).
   Reserve compute for next Featured comp; ship durable artifacts
   (LR-diag suite, BOTE harness, decisions.jsonl calibration loop
   already promoted to skill).
3. **Final-window R5 HEDGE preparation** (Day-25-26): list OOF-best
   candidates rejected for public regression. Existing HEDGE ladder:
   d13e τ=100k, d15c ExtraTrees, d15d KNN-LGBM, path_b_K22_d12meta
   (LB 0.95045 −4 bp), d18_path_b_K23_d16_d18 (LB 0.95149 PRIMARY-trail).

### Files added/modified this session

- `scripts/d19_lgbm_v4_fs.py` — A5-light LGBM proxy
- `scripts/c1_yao_vehtari_bma.py` — C1 four-variant comparison
- `kernels/a5-cb-v4-fs-gpu/a5_cb_v4_fs_gpu.py` — A5-full kernel scaffold
- `audit/2026-05-07-d19-overnight-research.md` — full audit (this file
  cross-links there)
- `scripts/artifacts/oof_d19_lgbm_v4_fs_strat.npy` + test + JSON
- `scripts/artifacts/probe_min_meta__*+d19_lgbm_v4_fs.json` (gate JSON)
- `scripts/artifacts/probe_min_meta__*+p1_xgb_v4_gpu.json` (B2 K=27)
- `scripts/artifacts/c1_yao_vehtari_bma_results.json` (C1, on completion)
- `audit/decisions.jsonl` — D1/B2/A5/C1 BOTE entries
- ISSUES.md leaves 4b (D1 closed null-pre-flight) + 2d (C1 wip)
