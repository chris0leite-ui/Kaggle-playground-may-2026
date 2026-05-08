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
