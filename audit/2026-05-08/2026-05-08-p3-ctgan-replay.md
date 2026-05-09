# 2026-05-09 — P3: CTGAN-replay-discriminator (deferred E3 unblocked)

`branch: claude/find-dgp-research-ClsQE`
`tag: dgp-ctgan-replay + host-signature`

## TL;DR

- Trained off-the-shelf SDV CTGAN on 80k synth-train rows for 20
  epochs (~32 min CPU).
- Sampled 200k replay rows.
- Trained 5-fold LightGBM 2-class discriminator: synth (label=1) vs
  replay (label=0).
- **Overall disc AUC: 0.9993** — host's CTGAN is near-perfectly
  distinguishable from off-the-shelf CTGAN. Strong host-specific
  signature.
- disc-pred mean 0.9894, std 0.0500 (heavily saturated near 1).
- K=4+1 LR-meta gate: **−0.02 bp** (disc-as-feature collinear with
  K=4 PitNextLap signal). NULL.

## Disc-pred structure across (Year, Compound) cells

| Year | mean disc | std disc | pos_rate |
|---|---:|---:|---:|
| 2022 | 0.9894 | 0.0472 | 0.2665 |
| **2023** | **0.9868** | 0.0611 | 0.0096 |
| 2024 | 0.9918 | 0.0391 | 0.2953 |
| 2025 | 0.9900 | 0.0472 | 0.2844 |

| Compound | mean disc | pos_rate |
|---|---:|---:|
| HARD | 0.9861 | 0.3275 |
| INTERMEDIATE | 0.9856 | 0.1523 |
| MEDIUM | 0.9930 | 0.1011 |
| SOFT | 0.9863 | 0.1935 |
| WET | 0.9781 | 0.0251 |

2023 has slightly lower disc (more replicable by off-the-shelf CTGAN)
— consistent with its anomalous "practice-source" hypothesis (P9).

## Monotonic y-rate vs disc-pred quintile

| disc-pred quintile | n | mean disc | y_rate |
|---|---:|---:|---:|
| 1 (lowest, <0.994) | 87,828 | 0.9525 | **0.2793** |
| 2 | 87,827 | 0.9965 | 0.2549 |
| 3 | 87,822 | 0.9987 | 0.2065 |
| 4 | 87,824 | 0.9995 | 0.1561 |
| 5 (highest, >0.9997) | 87,839 | 0.9999 | **0.0981** |

**Pit rate decreases 3× from low-disc to high-disc.** Rows that are
HOST-SPECIFIC (off-the-shelf CTGAN can't replicate) have LOWER pit
rate. Rows that look "vanilla CTGAN" (low disc) have higher pit rate.

Interpretation: pit-stop laps have characteristic feature patterns
(high TyreLife, late-stint LapTime spikes) that off-the-shelf CTGAN
reproduces well. Non-pit laps have subtler patterns that the host's
CTGAN learned via its custom conditioning, which the off-the-shelf
CTGAN misses.

## Why K=4+1 NULL despite real signal

The disc-pred's monotonic relationship with PitNextLap means the
**signal is collinear with K=4's existing tyre-degradation × compound
axis**. The 12-feat [P, rank, logit] expansion on K=4 already extracts
this information via tree splits in the base models. Adding disc-pred
as a 13th feature gives no orthogonal lift.

This confirms d18_f1's similar finding (disc-as-feature −0.112 bp at
K=21+4) and extends to K=4 PRIMARY.

## DGP characterization output

P3 contributes one durable DGP fact:

  **F6. Host's CTGAN is heavily host-specific.** Disc AUC 0.9993
  vs off-the-shelf CTGAN means the host used:
   - A custom conditioning vector (not just default SDV CTGAN cond)
   - Possibly a custom preprocessing layer (mode-specific
     normalization with custom mode counts)
   - Possibly different training duration / batch size / seed

The host's CTGAN cannot be replicated by off-the-shelf SDV CTGAN.
The signature is consistent across all 627k synth rows (disc-pred
mean 0.989, std 0.05).

## Pointers

- `scripts/dgp_v2/p3_ctgan_replay.py` — training + sampling
- `scripts/dgp_v2/gate_p3_k4plus1.py` — gate
- `scripts/artifacts/p3_ctgan_replay_disc_results.json`
- `oof_p3_ctgan_replay_disc_strat.npy`, `test_*`

## Friction tag

`ctgan-replay-disc-saturated-and-collinear-with-K4` — host-CTGAN
disc AUC 0.999 is durable DGP fact but disc-as-feature is NULL at
K=4+1 (collinear with tyre-degradation × compound logit axis). Skip
recursive replay-disc as a base candidate; reuse for DGP
characterization only.
