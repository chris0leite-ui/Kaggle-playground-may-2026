# 2026-05-09 — Decode-NN session WRAPUP

`branch: claude/find-dgp-research-ClsQE`
`tag: wrapup-decode-nn-session`
`session window: 2026-05-08 evening through 2026-05-09 ~07:50`

> Hard wrap. Stating only what was measured. Open questions and
> untested variants enumerated, not adjudicated.

## What was measured this session

### DGP characterization (synth-only, no public CSV used originally)

P1, P1b, P1c, P3, P9, P9b, P10, P11 — committed in earlier audits.
Six durable findings F1-F6 (committed in `2026-05-08-DGP-FINAL-summary.md`).
Five base candidates P2/P5/P7/P8 + P3 (disc-as-feature) all gated at
K=4+1; lift range −0.02 to +0.17 bp. Path-B with stint_start_imputed
cohort (P10) and per-cell calibration (P11) also NULL.

### Mid-session pivot

PI relaxed the "no public CSV" constraint and pointed out that
recursive synth-on-synth surrogates don't make sense. P12 (recursive
on synth, 40 ep, dim 256) was killed mid-training; aadigupta1601 was
downloaded.

### Critic agent review

Spawned a critic agent on the decode-NN plan. Critic recommended
PIVOT from base candidates to meta-architecture, citing 5 prior
NULLs in the +0.09 to +0.17 bp band as empirical evidence the
binding constraint is at the meta level on K=4 [P, rank, logit].

### Meta-architecture probes

| Probe | Description | OOF | Δ vs plain LR-meta (0.95399) |
|---|---|---:|---:|
| P14 Poly2 LR L2 C=0.01 | 90 feat (12 + 78 pairwise) | 0.95402 | +0.15 bp |
| P14 Poly2 LR L2 C=0.10 | same | 0.95402 | +0.13 bp |
| P14 Poly2 LR L2 C=1.00 | same | 0.95402 | +0.16 bp |
| P14 Poly2 LR L2 C=10.0 | same | 0.95402 | +0.15 bp |
| P15 Bagged LR-meta N=30 | 12 feat, bootstrap | 0.95399 | +0.00 bp |
| P15 Bagged LR-meta N=100 | 12 feat, bootstrap | 0.95399 | +0.00 bp |
| (Reference) Plain LR-meta | 12 feat | 0.95399 | — |
| (Reference) K=4 + Path-B C×S τ=100k | 12 feat | 0.95403 | +0.04 bp (PRIMARY) |

P14 + P15 measurements: across L2 regularization sweeps and bagging
bootstrap counts on K=4 [P, rank, logit], OOF AUC moved within
[+0.00, +0.16] bp. P14's polynomial expansion did not produce more
lift than the plain 12-feat LR fit. P15's bagging produced exactly
zero delta vs single LR.

### Surrogate runs (incomplete)

| Probe | Scope | Status | disc AUC vs host |
|---|---|---|---:|
| P3 (earlier session) | recursive on synth, 20 ep, defaults | done | 0.99926 |
| P12 | recursive on synth, 40 ep, dim 256 | KILLED at 17/40 (CPU contention) | — |
| P13 v1 | CTGAN on orig, 80 ep, defaults | KILLED at 37/80 (CPU contention from P14) | — |
| P13 v2 | CTGAN on orig, 80 ep, defaults | KILLED at ~9/80 (session wrap) | — |
| P16 | CTGAN on orig, pac=1, dim 256, 120 ep | NOT RUN | — |
| P17 | per-feature KS dissection | NOT RUN (depends on P13/P16) | — |

P13 v2 reached iter 9/80 of CTGAN training before the wrap signal.
The corresponding disc-AUC measurement was NOT obtained.

## What is NOT concluded from this session

The following statements are *hypotheses supported by the data
collected*, but not proven within this session's scope:

- "Rank-lock at K=4 is structural to the predictive subspace."
  Supported by P14 + P15 observations, but the structurally-untested
  meta variant Student-t shrinkage Path-B (d18 idea-board T4a) was
  not implemented. A heavy-tailed-prior LR meta on K=4 has not been
  measured.

- "Host CTGAN cannot be replicated from CTGAN-on-orig with default
  config." P13 v2 did not finish; the disc-AUC measurement for the
  P13 setup was not obtained. d18 f1 reported 0.988 for a similar
  protocol but on a different test set (orig-vs-replay, not
  host-synth-vs-replay).

- "Hyperparameter axes don't matter for disc AUC." P16 (pac=1, big
  CTGAN config) was not run. The within-CTGAN hyperparameter sweep
  is unmeasured this session.

- "+0.17 bp is the K=4+1 ceiling across all base/meta variants." It
  is the empirical ceiling across the variants tested (5 base + 7
  meta). Other candidate probes — Student-t Path-B, Yao/Vehtari BMA
  on K=4, GAN-inversion-trained encoder, MIA-shadow-model preimage,
  different cond-vector CTGAN config — were not tested.

- "DGP recovery cannot improve LB past the K=4 ceiling." This
  requires a definition of "DGP recovery" the session did not
  formalize past the synthetic-vs-orig distance metric.

## Open questions for next session

1. P13 disc-AUC measurement (CTGAN-on-orig, defaults, no contention).
   Estimated ~25 min CPU. The data point d18 f1 implied (~0.988) but
   never measured under this exact protocol.
2. P16 disc-AUC (CTGAN-on-orig, pac=1 + bigger config). Tests within-
   CTGAN hyperparameter axis. ~30 min CPU.
3. P17 per-feature KS dissection. Identifies which feature
   distributions each surrogate matches host on and which it
   diverges on. Pure analysis; few minutes once replays exist.
4. Student-t shrinkage Path-B (d18 T4a). The single documented
   meta-architecture variant never implemented. Requires
   variational inference or EM — non-trivial code.
5. Per-conditional-schema CTGAN sweep. SDV CTGAN's default cond
   samples one discrete column per row. Forcing cond on PitStop
   alone (per d18 f5 KS asymmetry inference) has not been tested.

## Files committed this session

Audits in `audit/2026-05-08/`:
- `2026-05-08-p1-synth-fingerprint.md`
- `2026-05-08-p1b-driver-temporal-fingerprint.md`
- `2026-05-08-p1c-tuple-concordance.md`
- `2026-05-08-p2p5-stint-recovery-bases.md`
- `2026-05-08-p3-ctgan-replay.md`
- `2026-05-08-p7-driver-atypicality.md`
- `2026-05-08-p9-2023-anomaly.md`
- `2026-05-08-p9b-race-year-anomalies.md`
- `2026-05-08-p10-pathb-stint-start-cohort.md`
- `2026-05-08-p11-cell-calibration.md`
- `2026-05-08-DGP-FINAL-summary.md`
- `2026-05-09-decode-NN-7step-plan.md`
- `2026-05-09-p14-poly2-lr-meta.md`
- `2026-05-09-p15-bagged-lr-meta.md`
- this file

Scripts in `scripts/dgp_v2/`:
- `p1_synth_only_fingerprint.py`
- `p2_orig_stint_recovery.py`, `gate_p2_k4plus1.py`
- `p3_ctgan_replay.py`, `gate_p3_k4plus1.py`
- `p4_anomaly_scan.py` (not run)
- `p5_pure_orig_stint.py`, `gate_p5_k4plus1.py`
- `p6_memorization_signature.py` (not run)
- `p7_driver_atypicality.py`, `gate_p7_k4plus1.py`
- `p8_kitchen_sink_dgp.py`, `gate_p8_k4plus1.py`
- `p10_pathb_stint_start.py`
- `p11_cell_calibration.py`
- `p12_ctgan_replay_big.py` (killed)
- `p13_orig_surrogate_v1.py` (run twice; both killed before completion)
- `p13b_lgbm_on_replay.py` (queued, not run)
- `p14_poly2_lr_meta.py`
- `p15_bagged_lr_meta.py`
- `p16_ctgan_config_variant.py` (queued, not run)
- `p16_tvae_on_orig.py` (queued, then deprecated — TVAE was already
  closed by d18 f1, kept as a record of the dead-end)
- `p17_perfeature_ks.py` (queued, not run)

Artifacts: OOF/test `.npy` files for P2, P3, P5, P7, P8, P14, P15;
parquet replay for none (P3 surrogate object was not pickled; P13's
replay parquets were never written because P13 didn't complete).

## Friction tags created this session

(To promote/discard at next postmortem; not auto-promoted.)

- `synth-stint-label-is-fabricated-not-temporal` (F1)
- `driver-vocab-mixes-active-and-historical` (F2)
- `dgp-aware-fe-rank-lock-saturates-at-0.2bp` (P7 closure)
- `stint-recovery-fe-orthogonal-but-rank-locked` (P2/P5 closure)
- `2023-year-portion-has-different-source-distribution` (F4)
- `ctgan-replay-disc-saturated-and-collinear-with-K4` (P3 closure)
- `pathb-cohort-recovered-orig-stint-equal-to-fabricated-stint` (P10)
- `per-cell-residual-calibration-regresses-K4-primary` (P11)
- `poly2-lr-meta-on-K4-equals-plain-lr-meta` (P14)
- `bagged-lr-meta-on-K4-equals-plain-lr-meta` (P15)
- `recursive-synth-on-synth-was-wrong-direction` (P12 / mid-session
  PI correction; per-iteration recursive-on-synth doesn't measure
  what we wanted)

## Final state

K=4 PRIMARY (LB 0.95351) unchanged. No submissions made this session.
Branch `claude/find-dgp-research-ClsQE` pushed up through P15.
Untracked queued scripts (P16 variants, P17) staged in this commit.
P13/P16/P17 results unmeasured.
