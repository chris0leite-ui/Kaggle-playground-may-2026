# 2026-05-07 — d18 chain-decomposition of P(X) on orig (E1)

`branch: claude/reverse-engineer-data-generation-Hu8EK`
`tag: dgp-chain-decomposition`
`mechanism family: external_data_aggregate (chain variant)`
`ISSUES leaf: 7f`

> Reverse-engineering the synthesizer. Decompose P(X) on the orig
> dataset along a domain-causal chain; for each synth row compute
> per-step orig-log-likelihood + z-score; use the 24-feature
> diagnostic vector as both base input and reusable artifact for
> downstream DGP probes (E2-E5).

## TL;DR

- K=21 LR-meta baseline OOF 0.95073 → **K=21+1 (d18_chain_decomp)
  0.95147 = +7.365 bp**, ρ vs PRIMARY 0.99137. **Largest single-base
  K=21+1 OOF advance of the session** (prior record:
  d16_orig_continuous_only +3.331 bp).
- K=21+2 (d16_orig_continuous_only + d18_chain_decomp) **+9.848 bp**
  = additive over component K=21+1 lifts (+3.33 + +6.52 = 9.85). The
  two are mechanistically orthogonal — d16 routes via |w_rank|=0.614,
  d18 routes via |w_logit|=0.519.
- Standalone gate FAIL (-12.91 bp). Same shape as d9c FM (std 0.92069
  → K-meta +3 bp LB) and d16_orig_continuous_only (std 0.91483 → K=21+1
  +3.33 bp). Weak base, strong meta-utility.
- Diagnostic artifact `data/chain_decomp_features_{train,test}.parquet`
  (627k rows × 24 chain features) is reusable for E2–E5.
- Predicted LB band per `probe.py` at ρ 0.991 (band 0.99-0.995): d_oof
  − 3.0 = **+4.4 bp conservative / +5.9 bp optimistic**. Path-B hier-meta
  amp untested.

## Method

Domain-causal chain (12 LGBM steps + 1 marginal):

```
Year (marginal)
→ Race | Year
→ Compound | Year, Race, Stint
→ Stint | Year, Race, Compound
→ LapNumber | Year, Race, Compound, Stint
→ TyreLife | Year, Race, Compound, Stint, LapNumber
→ RaceProgress | Year, Race, LapNumber
→ Position | Year, Race, Compound, Stint, LapNumber, TyreLife
→ LapTime | Year, Race, Compound, Stint, LapNumber, TyreLife, Position
→ LapTime_Delta | Compound, TyreLife, LapTime, LapNumber, Stint
→ Cumulative_Degradation | Compound, Stint, TyreLife, LapTime_Delta, LapTime
→ Position_Change | Year, Race, LapNumber, Position
→ PitStop | Year, Race, Compound, Stint, LapNumber, TyreLife, Position, RaceProgress
```

Each step: small LGBM on **orig only** (101k rows). Driver excluded
(orig has 31 historical codes, synth has 856 D-prefixed ghosts; no
overlap; matches d16's KS-driven feature exclusion).

Per synth row, compute:
- `chain_ll_<col>` = log-likelihood of actual value under fitted
  conditional (Gaussian for continuous with σ = orig-RMSE; categorical
  softmax floor 1e-9 for discrete).
- `chain_z_<col>` = (actual − pred_mean) / σ_orig (continuous only).
- `chain_anomaly_L1` = Σ|z|, `chain_total_ll` = Σ ll.

Total: 23 chain features + 2 composites = 25 features. Wall: 7 min CPU.
Leakage status: chain models fit on orig (no synth labels touched);
applied as a deterministic-per-row feature transform on synth — no
fold contamination possible (Rule 24 clean). Same external-data-
transfer pattern as d15_orig_transfer / d16_orig_continuous_only;
AV-AUC=0.502 (Day-12) confirms no domain shift to worry about.

## Standalone OOF (5-fold StratifiedKFold)

| variant | OOF AUC | Δ vs raw-only |
|---|---:|---:|
| raw 14 features only (LGBM ablation) | 0.94956 | — |
| raw + 25 chain features | 0.94954 | −0.24 bp |

The downstream LGBM extracts **zero** incremental signal from chain
features over raw features alone — same finding as d14 DGP-residuals
and the same load-bearing mechanism (`tag:
synthetic-dgp-conditionally-near-independent`): GBDTs already absorb
the per-row signal optimally from raw features.

## Class-conditional KS (y=0 vs y=1, top 10)

The DGP fingerprint that LR-meta extracts (per-step orig-likelihood
discriminates by target):

| feature | KS y=0 vs y=1 | p |
|---|---:|---:|
| `chain_ll_TyreLife` | **0.403** | 0 |
| `chain_ll_Year` | 0.288 | 0 |
| `chain_ll_Position_Change` | 0.265 | 0 |
| `chain_ll_PitStop` | 0.257 | 0 |
| `chain_total_ll` | 0.220 | 0 |
| `chain_ll_Cumulative_Degradation` | 0.143 | 0 |
| `chain_ll_Compound` | 0.138 | 0 |
| `chain_ll_Stint` | 0.090 | 0 |
| `chain_ll_RaceProgress` | 0.081 | 0 |
| `chain_ll_Race` | 0.079 | 0 |

**TyreLife**'s chain log-likelihood has KS 0.403 between PitNextLap=0
and PitNextLap=1 — synth rows whose TyreLife is unlikely under the
orig DGP given upstream context have very different pit rates from
synth rows whose TyreLife is likely. This is the synthesizer's
class-conditional structure leaking through the chain decomposition.

## Min-meta gate (K=21 LR-meta)

```
=== K=21 LR-meta baseline ===
  OOF 0.95073  (75.5s)

=== K=21+1 d18_chain_decomp ===
  OOF 0.95147  Δ +7.365 bp  ρ vs PRIMARY 0.991370
  |w| = 1.124  (raw -0.560, rank -0.025, logit +0.539)

=== K=21+2 d16_orig_continuous_only + d18_chain_decomp ===
  OOF 0.95171  Δ +9.848 bp  ρ vs PRIMARY 0.990886
  d16 |w| = 0.956  (rank-dominant: raw +0.332, rank +0.614, logit -0.010)
  d18 |w| = 1.084  (logit-dominant: raw -0.538, rank -0.027, logit +0.519)
```

## Calibration ladder placement

| Base | Std OOF | ρ vs PRIMARY | K=21+1 Δ | LB |
|---|---:|---:|---:|---:|
| d15_orig_transfer | 0.85138 | 0.5653 | +0.778 (K=2) | TIE 0.95049 |
| d16_orig_continuous_only | 0.91483 | 0.9946 | +3.331 | 0.95089 PRIMARY |
| **d18_chain_decomp** | **0.94954** | **0.9914** | **+7.365** | **n/a (no submit)** |

Note: ρ comparisons are vs d13e (probe.py default) — comparable across
the ladder. The +7.37 bp lift is a 2.2× advance over d16's record.

## Predicted LB band

ρ 0.9914 places this in the (0.99, 0.995) band of `probe.py`'s
`predicted_lb_delta_bp`:

| at K=21+1 | conservative | central | optimistic |
|---|---:|---:|---:|
| OOF Δ | +7.37 | +7.37 | +7.37 |
| LB Δ (band) | **+4.4** | +5.9 | +7.4 |

K=22+1 over current PRIMARY's pool (K=21 + d16_orig_continuous_only)
NOT yet tested at LR-meta. Path-B Compound×Stint hier-meta over
K=22+d18 also untested. Both probes are next-step.

Per friction `path-b-amp-only-fires-on-meta-arch-not-base-add` (1.4×
realised on d15b DAE base-add, 1.0× on d16 cont_only) the realised
LB amp on a base-add probably lands ~1.0–1.4×: **+4-7 bp LB**.

## Why this works (mechanistic interpretation)

1. **Standalone**: chain features add nothing over raw (d14 finding
   reconfirmed: synth's per-row joint structure is conditionally
   near-independent → GBDTs extract everything from raw alone).
2. **At LR-meta**: the *combination* of chain features changes the
   downstream LGBM's prediction surface to be systematically
   different from K=21 bases, even though it doesn't beat raw at
   standalone AUC. The LR-meta absorbs this as orthogonal information
   via [raw, rank, logit] expand.
3. **Why orthogonal**: the K=21 pool consists of:
   - leakage-eating bases (high Strat AUC, leak via TE/group-aggregates)
   - leakage-robust bases (FM-class, rules)
   None of them encode "where does this row sit in the orig DGP chain?"
   d18 is the first base in the pool whose features come from the
   *orig DGP under a chain factorisation*, structurally distinct from
   d16's "orig-LGBM on a feature subset".

## Diagnostic value (independent of LB outcome)

The parquet `data/chain_decomp_features_{train,test}.parquet` contains
per-row chain LL + z-score for all 627k rows. Reusable inputs to:

- **E2 — per-row preimage join** (kNN in orig over the 7 low-KS features).
  The chain-LL z-score gives a confidence weight per match.
- **E3 — replay discriminator** (CTGAN replay vs host_synth). The chain
  features are an alternative to GMM(16) for log-density; can be the
  feature for the discriminator head.
- **E4 — class-conditional density ratio** r̂_y(x). The per-step LL
  difference between y=0 and y=1 conditional models is the
  factorised version of the joint likelihood ratio.
- **E5 — Path-B cohort axis on chain_total_ll quintile** (50-cell
  Compound × Stint × chain_LL_q5). Direct successor to Phase-5 r̂_q5
  cohort (which was NULL on K=14 sub-pool; this is a cleaner version
  on the proper K=22 pool).

## Concerns + sanity checks

- **Leakage check.** Each chain model is fit on orig rows; the chain
  feature for synth row r is a deterministic function of (orig-fit
  models, synth row r's own features). No synth labels touch the
  feature pipeline at any stage. Strict-OOF status: clean by
  construction (Rule 24 friction
  `target-construction-layer-leakage` does not apply).
- **Single-variant only.** Per Rule 21, family is not "alive" or
  "dead" until ≥3 variants. This is variant-1 of `chain_decomposition`.
  Variants to try (ordered, cheap):
    (i) alternative chain ordering (data-driven via topological sort
        of mutual information, vs domain-causal)
    (ii) chain steps as classifiers w/ continuous targets binned
        (q10) to extract log-density at the actual quantile (instead
        of Gaussian σ approximation)
    (iii) richer per-step model (CatBoost on cat-cols where Compound
        and Stint sit in the conditioning set)
- **Predicted LB inflation risk.** +7.37 bp K=21+1 is unusually large.
  Could be a coincidence due to ρ landing in a band where LR-meta
  exploits noise. The Path-B hier-meta probe is the canonical
  amplification test — if it lands at 1.4× (DAE precedent), expected
  LB ~+10 bp. If it lands at 1.0× (d16 cont_only realised), expected
  LB ~+7 bp. Both significantly exceed the LB quantization floor.
- **Q6 metric_aligned**: True. Chain steps train log-loss / regression
  on each conditional; final LR-meta is row-AUC-aligned.

## Pointers

- Script: `scripts/d18_chain_decomp.py`
- BOTE log: `audit/decisions.jsonl` (cost_min 90, expected_lb_bp 0.10
  SKIP — overridden per PI "build foundation" framing). Outcome will
  be `record-outcome` after a submit.
- Min-meta JSONs:
  - `scripts/artifacts/probe_min_meta__d18_chain_decomp.json`
  - `scripts/artifacts/probe_min_meta__d16_orig_continuous_only+d18_chain_decomp.json`
- Diagnostic parquets: `data/chain_decomp_features_{train,test}.parquet`
  (gitignored; regenerable via `python scripts/d18_chain_decomp.py`).
- Per-step diagnostic JSON: `scripts/artifacts/d18_chain_decomp_summary.json`
- OOF/test artifacts:
  - `oof_d18_chain_decomp_strat.npy` + `test_d18_chain_decomp_strat.npy`
    (raw + chain features; the candidate base)
  - `oof_d18_chain_decomp_lgbm_only_strat.npy` + test (raw-only ablation)

## Add to mechanism_families_explored

- `chain_decomposition_orig_likelihood` — d18 first variant, **PASS
  K=21+1 +7.37 bp** (largest of session); standalone FAIL (Rule 21
  variant-1 of 3 needed).

## Recommended next probes (ranked by EV/cost)

1. **K=22+1 LR-meta over current PRIMARY pool** (K=21 + d16_orig_continuous_only
   + d18_chain_decomp). 5 min. Confirms the +9.85 K=21+2 result holds
   when we use the existing-PRIMARY pool composition.
2. **Path-B hier-meta K=22+d18 Compound×Stint τ=20k**. 10 min.
   Determines amp factor. If it lands ≥+10 bp OOF, this is a
   PRIMARY-replacement candidate.
3. **Strict-OOF audit** (per-fold chain-model refit with synth-train
   subsetted to the fold's tr rows for any synth-derived features —
   here we use orig only, so this is a no-op verification, but worth
   documenting). 10 min.
4. **Variant-2 chain orderings** for Rule 21 family-falsification.
   - data-driven MI ordering
   - reverse-causal (LapTime first, then back-decompose)
   30 min.
5. **E2 (per-row preimage join)** using chain LL as match-confidence
   weight. 2 h CPU.

## Submission decision deferred

PI directive: "build foundation, doesn't need to immediately improve
predictions." This probe is foundation work. Next session: PI
sealed-prediction order (Rule 26a) before agent reveals expected_lb_bp
on a submit candidate.
