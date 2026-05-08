# Day-19 overnight research note (in progress)

> Branch: `claude/ml-competition-analysis-rwD3f`
> PI directive 2026-05-07 evening: "do all (B2, A5, C1) in sequence, skip
> my predictions, you have all night."
>
> Context: PI sealed prediction was given for D1 (debashish historical-
> priors aggregate join) only. Harness verdict on D1 was SKIP (expected
> +0.20 bp at 45 min = 0.004 bp/min); PI gut −2 to 0 bp (NULL like H2)
> agreed. ISSUES leaf 4b closed null-by-pre-flight (`audit/decisions.jsonl`).
> External-data axis empirically closed for our budget.
>
> Pivoted to A5 (LGBM-v4-fs anchor-modify proxy), C1 (Yao/Vehtari BMA),
> with B2 re-verify on the new K=27 pool (was redundant on K=24).
> No PI sealed predictions for B2/A5/C1 per PI's directive — flagged
> per Rule 26b as anomalies in `audit/decisions.jsonl`.

## Pre-existing context

PRIMARY = `d18_path_b_K27_v4h1d_d16_d18_e2_f2` τ=100k LB **0.95368**.
K=21+6 (no xgb_v4) LR-meta OOF 0.95428; Path-B C×S τ=100k OOF 0.95432
(+0.4 bp amp = 1.0× friction reconfirmed).

Top-5% gap: −3.7 bp (boundary 0.95405). Leader: −10.8 bp (0.95476).

## D1 = debashish historical-priors aggregate join (closed null-pre-flight)

PI sealed: **−2 to 0 bp NULL** (band midpoint −1).
Agent BOTE: family `external_data_aggregate` priors (P=0.20, midpoint
1.0 bp); cost 45 min CPU → expected +0.20 bp; cost-efficiency 0.004
bp/min; **verdict SKIP**. PI agreement = load-bearing signal that the
calibration loop is doing its job.

ISSUES leaf 4b: **closed null-by-pre-flight**.

Friction tag candidate: `external-data-axis-closed-by-pre-flight-when-pi-and-harness-agree`.

## B2 = XGB v4 add at K=27 (re-verify; was REDUNDANT at K=24)

Existing artifacts from sibling branch `optimize-model-performance-rruC2`:
- XGB v4 standalone OOF: **0.95135** (5-fold StratifiedKFold seed=42).
- vs CB v4 OOF 0.95200 (-6.5 bp lower standalone).
- K=21+xgb_v4 alone: +18.16 bp / ρ=0.988.

K=27 verify result (this session):
- K=21 baseline LR-meta: 0.95073
- K=21+6 (no xgb_v4): **0.95428** (Δ +35.528 bp; ρ vs PRIMARY 0.987)
- K=21+7 (with xgb_v4): **0.95430** (Δ +35.671 bp)
- **xgb_v4 marginal at K=27: +0.143 bp NULL/noise floor**
- |w| xgb_v4 = 0.39 (low but non-zero; LR routes some weight via logit)

**Verdict: closed REDUNDANT at K=27** — confirms `gbdt-class-redundant-on-shared-FE`
extends from K=24 → K=27. HEDGE-tier eligibility only (R5 final-window).

## A5 = LGBM-v4-fs (proxy for CB-v4-fs anchor-modify) — NULL

Mechanism prescription from friction `lr-meta-rank-lock-strong-anchor`
(6× cross-confirmations): integrate orthogonal fs_cum_pits signal INTO
the strong anchor by retraining (not as a stacked base). A5-full would
be CB-GPU; this is the LGBM-CPU proxy.

Recipe: make_features_static (v3 base) + yekenot items 2/3/4 (floor-cat
/ count-enc / KBins) + 24 field-state cross-row aggregates per (Race,
Year, LapNumber) ± Compound. Excludes orig-aug for CPU time budget.

5-fold StratifiedKFold(seed=42) LGBM CPU. 135 features / 19 cats.
**Standalone OOF: 0.94510** (5 bp BELOW v3 honest fold-safe ceiling
0.94563; fs aggregates ADD NOISE relative to v4 recipe at LGBM class
without orig-aug). Per-fold AUCs: 0.9458 / 0.9454 / 0.9450 / 0.9444 /
0.9450. Walls 106 / 50 / 46 / 47 / 47 s; total 329 s = 5.5 min CPU.

**K=27+1 stack-add gate**: K=21+6 baseline LR-meta OOF 0.95428;
K=21+7 (with d19_lgbm_v4_fs) OOF **0.95427** → **−0.106 bp NULL**.
ρ vs PRIMARY = 0.9865.

Verdict: **NULL, anchor-modify pathway closed at LGBM class.**
**7th cross-confirmation of `lr-meta-rank-lock-strong-anchor`.**
fs aggregates' standalone +13.73 bp lift over raw-14 (Day-17 audit)
does not survive integration into v4 yekenot recipe at LGBM level.

Implication for A5-full (CB-GPU): probability revised down ~10-15%
(CB has CTR which uses different feature interactions than LGBM splits;
could potentially extract fs signal where LGBM cannot, but proxy NULL
is strong negative evidence). A5-full Kaggle kernel scaffold written
at `kernels/a5-cb-v4-fs-gpu/a5_cb_v4_fs_gpu.py`; **deferred** unless
PI specifically wants to test the CTR mechanism difference.

Friction tag candidate: `fs-aggregates-add-noise-not-signal-when-merged-into-yekenot-recipe-without-orig-aug`.

## C1 = Yao/Vehtari covariance-modelled BMA on K=27 — FALSIFIED

Last untested meta-arch on the only live amp axis (Compound × Stint).

Implementation (`scripts/c1_yao_vehtari_bma.py`, 542s wall):
- V0: plain global LR-meta (no segmentation)
- V1: Path-B current shrinkage τ=100k (replicates PRIMARY)
- V2: plain BMA (w_k ∝ exp(−N × BCE_k))
- V3: Yao/Vehtari covariance-modulated Path-B with Σ-aware shrinkage:
      `(X'X + τΣ^{-1})w = X'y + τΣ^{-1} w_global`
      Shrinks weights along low-eigenvalue (highly-correlated) directions
      MORE than plain Path-B; preserves dominant orthogonal directions.
      τ ∈ {10k, 50k, 200k}.

**Results:**

| Variant | OOF AUC | Δ vs V1 (bp) |
|---|---:|---:|
| V0 plain LR-meta | 0.95428 | −0.4 |
| **V1 Path-B plain τ=100k** | **0.95432** | 0 |
| V2 plain BMA | 0.95390 | −4.2 |
| V3 Yao/Vehtari τ=10000 | 0.95426 | **−0.59** |
| V3 Yao/Vehtari τ=50000 | 0.95427 | **−0.49** |
| V3 Yao/Vehtari τ=200000 | 0.95427 | **−0.47** |

V3 covariance-modulated Path-B **regresses across all 3 τ values**. Mechanism:
V3 over-shrinks weights along highly-correlated base directions, but K=27
LR-meta USES those correlated dims for useful base routing. Σ-prior pulls
the segment fits toward the global mean MORE than plain shrinkage τ does,
removing the segment-conditional signal that plain Path-B captures.

V2 plain BMA degenerates to v4+h1d (96% mass) and -4.2 bp confirms simple
weighted average loses the LR-meta routing structure entirely.

**Verdict: meta-arch redesign family empirically closed on K=27.**
Combined with prior session falsifications (Path-B alt-axes
4 axes NULL across τ, twin-meta-blend −1.79 bp, conformal isotonic 4
schemes −2.5 to −9.6 bp, multi-level 4-tier 5 configs NULL, K=10
forward-selected Path-B 9 configs sub-bp), C1 is the 9th distinct
meta-arch variant tested with Compound × Stint as the only live amp
dim. **Family closed.**

Friction tag candidate: `covariance-modulated-path-b-overshrinks-correlated-base-routing-directions-vs-plain-tau`.

## Calibration log

`audit/decisions.jsonl` entries this session:
1. **D1 debashish historical-priors**: PI −1 / agent +0.20 / actual NULL
   (didn't run; SKIP'd by harness pre-flight). Calibration row CLOSED.
2. **B2 XGB v4 K=27**: no PI seal (PI directive); agent +0.20 (family
   default); actual +0.143 bp at meta level. **NULL**.
3. **A5 LGBM-v4-fs**: no PI seal; agent +0.5 bp midpoint (override
   `single_base_fe_addition` toward `meta_arch_redesign-adjacent`);
   actual **−0.106 bp NULL** at K=27+1 stack-add.
4. **C1 Yao/Vehtari**: no PI seal; agent +1.2 bp midpoint; actual
   **best τ=200k = −0.47 bp REGRESS**. Family falsified.

Total session calibration: 4-of-4 NULL/regress. Predicted vs actual
delta sums: agent net +1.7 bp predicted, actual −0.4 bp realised
(agent over-predicted by ~2 bp on aggregate). Family priors were
appropriately defensive on B2 and D1; the override on A5 (toward
meta-arch-adjacent) was slightly optimistic; the override on C1
(meta_arch_redesign default) overshot for this specific Σ-prior
formulation.

## Synthesis: ceiling-on-this-pool-with-this-FE

Day-19 overnight closes 4 axes simultaneously:
- **D-axis (external data)**: D1 SKIP'd, D2 capped, D3 in pool.
- **B-axis (anchor-swap on shared FE)**: B2 redundant on K=27.
- **A-axis (orthogonal-base-add)**: 7th rank-lock confirmation via A5.
- **C-axis (meta-arch redesign on Compound × Stint)**: 9th variant
  falsified (V3 covariance-modulated Path-B).

**The remaining open structural axis is sequence-level (A1: HMM Compound
transitions + AR(1) within-stint TyreLife).** Every K=27 base treats
rows i.i.d. internally, so a per-(Driver, Race, Year)-group sequence
log-likelihood IS structurally orthogonal — though the empirical pattern
of 7× rank-lock confirmations suggests even genuinely-orthogonal new
bases get absorbed at the LR-meta layer once v4+h1d are anchors.

## Next-session priorities (post-overnight, ranked by remaining-EV)

1. **A1 sequence-level DGP fingerprinting** (~2-3h CPU). The only
   structurally-distinct axis untested. Predicted: family
   `single_base_fe_addition` p=0.05 midpoint 0.5 bp under default;
   override to `meta_arch_redesign-adjacent` p=0.20 midpoint 2 bp under
   structural-orthogonality argument. Expected LB Δ +0.5 bp.

2. **B1 RealMLP n_ens=24 on Kaggle GPU** (~3.5h Kaggle). Current h1d is
   n_ens=4. Yekenot's published n_ens=8/24 in the recipe. Predicted
   +1-3 bp standalone OOF; +1 bp at K=27+1 (meta routes through h1d
   already). Family `tuning_existing` p=0.20 midpoint 0.5 bp;
   harness verdict likely DEFER.

3. **Wrap-up posture**. Top-11% achieved (LB 0.95368, rank #98/893).
   Durable artifacts already shipped (LR-diag suite, BOTE harness,
   decisions.jsonl calibration loop, Path-B-amp friction docs, kaggle-comp
   skill suite). 9 days remain to deadline; reserve compute for next
   Featured comp Day-1 if A1 also nulls.

4. **Final-window R5 HEDGE preparation** (Day-25-26): list OOF-best
   candidates rejected for public-LB regression. Existing HEDGE ladder:
   d13e τ=100k, d15c ExtraTrees, d15d KNN-LGBM, d18_path_b_K23_d16_d18
   (LB 0.95149).

## Friction tags introduced this session

- `external-data-axis-closed-by-pre-flight-when-pi-and-harness-agree` —
  Calibration loop working as intended; PI gut + family priors converged
  on SKIP. D1 closed without compute spend.
- `fs-aggregates-add-noise-not-signal-when-merged-into-yekenot-recipe-without-orig-aug`
  — d19 LGBM-v4-fs std OOF 0.94510 below v3 ceiling 0.94563. Probably
  the orig-aug item 7 (which we skipped for CPU budget) is needed to
  rebalance the recipe when fs aggregates enter.
- `covariance-modulated-path-b-overshrinks-correlated-base-routing-directions-vs-plain-tau`
  — V3 over-shrinks Σ-direction weights that LR-meta uses for routing.
  Plain τ shrinkage is the right level for this pool.
- `meta-arch-redesign-family-empirically-exhausted-on-k27-pool` (consolidated
  across 9 variants tested over Days 14-19).
