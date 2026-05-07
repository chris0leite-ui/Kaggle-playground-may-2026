# Day-17 H2 — FastF1 external-data join (audit)

**Date:** 2026-05-07
**Branch:** `claude/read-handover-62BCt`
**Reference paper:** Frontiers AI 2025 Bi-LSTM (DriverAheadPit +1.1 logit shift; TrackStatus; CumulativeTimeStint).
**Verdict:** **NULL → DGP-leaky-on-matched-rows but match rate too thin to move K=21 LR-meta.**

## Wall-time

| stage | seconds |
|---|---:|
| FastF1 install | ~5 |
| FastF1 pull (104 combos, 30 successful + cached) | 238 |
| Holdout 80/20 train | 27 |
| 5-fold OOF train | 235 |
| K=21+1 gate | 188 |
| **total** | **~510s + 188s gate ≈ 11.6 min** |

## Year coverage and match rate

- Dataset spans 2022–2025 (439140 train rows, 188165 test).
- Driver column structure: 887 unique values, 131 are real F1 3-letter
  TLAs (e.g. VER, HAM, LEC) covering ~40% of rows; 756 are synthetic
  D### codes (e.g. D109, D086) covering the other ~60%.
- **Synthetic data confirmed**: per-(Year,Race) driver count up to 856
  (real F1 grid is 20). Cancelled races (Chinese GP 2022) have rows.
  This is augmented from a real F1 seed.
- **FastF1 livetiming endpoint (livetiming.formula1.com) returns 403
  from the sandbox IP and rate-limits at ~500 calls/h.** Successful
  pulls: 30 races (all of 2022 + a few cached 2023/2024). 64 races
  skipped due to rate-limit / 403 / cancelled-race.
- FastF1 feature rows pulled: 28846. Even on the matched subset
  (real-driver, lap N exists in FastF1), Compound match was only 74%
  and TyreLife correlation only 0.55 between dataset and FastF1 —
  i.e. dataset is synthetic-perturbed even on real-driver real-lap rows.
- **Final merge match rate: train 1.42%, test 1.40%** (6249/439140
  train, 2633/188165 test). The vast majority of the dataset cannot
  be joined to FastF1 because: (a) synthetic D### drivers don't exist;
  (b) cancelled races have no FastF1 data; (c) sandbox rate-limit
  blocked 2025+ races entirely.

## DGP-leak AV check

- AV LGBM (matched=1 vs matched=0 classifier) **AV-AUC = 0.9602.**
  Matched and unmatched rows are massively distinguishable by base
  features alone — i.e. matching is highly correlated with raw row
  features (likely because matched rows tend to come from particular
  driver/race/lap segments).
- **PitNextLap rate**: matched 0.4173, unmatched 0.1958 — matched
  subset has 2.13× the pit rate. The matched subset is a structurally
  biased slice (perhaps clustered around real-data laps preserved
  near pit-stop moments).
- Conclusion: there IS DGP signal on the matched 1.4% but the bias
  is so confounded with base-features that the lift is unrecoverable
  via simple add-on features at corpus level.

## Standalone OOF and 80/20 holdout

- Standalone LGBM (5-fold StratifiedKFold seed=42) **OOF AUC 0.94800**.
- Honest 80/20 holdout (seed=99) **AUC 0.94823**. Holdout − OOF =
  +0.23bp — within noise, no leakage signature (consistent with
  Rule 24 holdout diagnostic; FastF1 features are merge-derived
  not label-derived, so no fold-leak risk).
- Standalone OOF is **−290bp below PRIMARY OOF 0.95090** — single-LGBM
  ceiling, not a stack candidate by itself.

## Min-meta K=22-add gate vs PRIMARY pool (d13e proxy)

| metric | value |
|---|---:|
| K=21 baseline OOF | 0.95073 |
| K=22 + d17_h2_fastf1 OOF | 0.95077 |
| **Δ bp** | **+0.428 bp** |
| ρ vs PRIMARY (d13e Compound×Stint τ=20k) | **0.99555** |
| |w| (raw + rank + logit) | 0.6488 |
| weights | raw -0.442, rank -0.016, logit +0.191 |

ρ = 0.9956 puts this candidate firmly in the rank-lock band
(precedent: d15c ExtraTrees ρ=0.99599 at +0.059bp; d15d KNN-LGBM
+0.056bp at routed ρ=0.996). +0.428bp is in the lower noise-floor
range, consistent with friction tag `lr-meta-rank-lock-strong-anchor`.

## Per-feature LGBM importance (top 10)

| feature | gain |
|---|---:|
| Year | 3278447 |
| Stint | 2624087 |
| TyreLife | 1517768 |
| RaceProgress | 594654 |
| LapTime_Delta | 550001 |
| Race_le | 499837 |
| Compound_le | 447016 |
| LapTime (s) | 376547 |
| Cumulative_Degradation | 325648 |
| LapNumber | 224131 |

**Notably absent from top 10**: all five FastF1-derived features
(DriverAheadPitLastLap, TrackStatusCode, CumulativeTimeStint,
delta_laptime, LapTimeSec). With 1.4% match rate, 98.6% of rows
have fill values for these columns, so they carry essentially no
signal at corpus scale.

## Verdict

**NULL** — rank-locked at ρ=0.9956 with +0.428bp Δ; well within
the K=21+1 noise floor. **HEDGE-tier eligibility**: yes if PI wants
the diversification-tax (cheap to keep, may marginally help final
HEDGE blend), but no PRIMARY-replace signal. **Do not submit**.

### What killed it
1. **Sandbox network blocks livetiming.formula1.com** — 403 from this
   IP, rate-limited at ~500 calls/h. Even with retries, 2025 data
   was completely unreachable. To get full 2022–2025 coverage we'd
   need either a different network egress, FastF1 cache from another
   source, or pre-downloaded data files.
2. **Synthetic dataset structure** — 60% of rows have D### codes
   that don't map to real F1 drivers (no telemetry exists). Even on
   the real-TLA 40% subset, per-race driver counts (up to 856)
   confirm massive augmentation; matched Compound only 74%, TyreLife
   correlation only 0.55. The dataset is **synthetic-derived from
   FastF1 seed but heavily perturbed**, not raw FastF1.
3. **DGP-leak signature on the matched subset** (AV-AUC 0.96, pit
   rate 2.1×) confirms the join is biased toward a particular slice
   of the data — but at 1.4% coverage, this bias cannot be exploited
   to lift corpus-level OOF.

### Companion findings (Rule 11 / lessons-to-skill candidates)
- **External-data integration on synthetic-augmented datasets is
  bounded by augmentation fidelity, not match rate.** Even if we
  had 100% match rate, the 0.55 TyreLife correlation says raw
  telemetry features are corrupted — gain ceiling on these features
  is small.
- **AV-AUC on matched-vs-unmatched is a useful pre-flight diagnostic**
  even when DGP-leak is partial: high AV-AUC + base-rate skew
  signals "biased subset" risk before any 5-fold compute.
- This validates Rule 25 (transductive features need AV check) — we
  ran the AV check up-front and it correctly flagged the merge as
  biased, saving us from over-interpreting any positive lift.

### What would unblock this hypothesis
- Different network egress (Kaggle notebook GPU env, where FastF1
  has been used in past comp public notebooks without 403).
- Pre-bundled FastF1 cache via Kaggle dataset.
- Trying Jolpica/Ergast (laps available but reformatted) and patching
  it in.
- Accepting that the synthetic-augmentation perturbation puts a hard
  ceiling on this approach regardless of network.

## Artifacts

- `scripts/d17_h2_fastf1_join.py`
- `scripts/artifacts/oof_d17_h2_fastf1_strat.npy` (n=439140, 2-col)
- `scripts/artifacts/test_d17_h2_fastf1_strat.npy` (n=188165, 2-col)
- `scripts/artifacts/d17_h2_fastf1_results.json`
- `scripts/artifacts/probe_min_meta__d17_h2_fastf1.json`

## Mechanism family update (for CLAUDE.md when scribe consolidates)

```
- d17_h2_fastf1_external_join  # 2026-05-07 NULL +0.428bp at ρ=0.9956;
  match rate 1.42% (sandbox IP blocked livetiming.formula1.com after
  ~30 cached races; synthetic D### drivers don't exist in FastF1);
  AV-AUC matched-vs-unmatched = 0.96 + pit-rate 2.13× (DGP-leak on
  matched subset but unrecoverable at 1.4% coverage); HEDGE-tier
  eligible only on diversification-tax grounds; do not submit.
```
