# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 14 (2026-05-14)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `scripts/probe.py` — entry point. `bote()` for BOTE, `gate()` for uniform gate report
3. `scripts/probe_min_meta.py` — K=21+N stack-add gate
4. `audit/2026-05-06-blend-and-rho-probes.md` — most recent rule-out + ρ inventory
5. `audit/2026-05-06-alpha-asymmetry-verification.md` — Path B α-asymmetry verified
6. `audit/2026-05-13-d13-path-b-hier-meta.md` — Path B mechanism (load-bearing)
7. `audit/2026-05-13-d13d-path-b-gkf-probe.md` — GKF amplification confirms private-robust
8. `audit/2026-05-12-d12-master-synthesis.md` — leakage-robust thesis
9. `scripts/pre_submit_diff.py` — MANDATORY before submit

Open with a 3-bullet read-back of state + first action.

**Harness usage cheatsheet (Rule 19):**
```bash
# BEFORE writing code for a candidate ≥10 min CPU:
python scripts/probe.py bote NAME --family X --cost_min N \
    [--std_oof_lift_bp Y] [--prob_useful U] [--note "rationale"]

# AFTER artifacts exist (under scripts/artifacts/oof_<NAME>_strat.npy):
python scripts/probe.py gate NAME \
    --oof scripts/artifacts/oof_NAME_strat.npy \
    --test scripts/artifacts/test_NAME_strat.npy

# Stack-add probe (K=21 + candidate(s)):
python scripts/probe_min_meta.py --candidates NAME1 NAME2 ...
```
Family priors are in `scripts/probe.py FAMILY_PRIORS`. Rule-out is
a valid result; cheap NULL findings get audit notes too.

## Where we are (Day 14 evening)

- **PRIMARY** = `d13e_compound_stint_tau20000` LB **0.95049** (+8bp Day-13 PM).
- **HEDGE** = `d13_path_b_stint_tau100000` LB 0.95041 (R5 candidate).
- **Gap to top-5%** (0.95345): **29.6bp**. 13 days remaining.
- **Submits used**: 24/270 total (Day-13 used 6/9, Day-14 used 0).

## Day-14 session — TabPFN + Move D results

### Move A — TabPFN fine-tune: DEAD

- **v2.5 @ 150k rows** (kernel v10): fold-0 AUC **0.94446** — identical to 50k-row result
  (0.94439). Training loss flat from epoch 1; fine-tuning not learning. Wall 6829s, no gain.
- **v2.6 @ any row count**: OOM at epoch 1 (model weights ≈15.37GB, P100 = 16GB). Dead.
- **Verdict**: TabPFN ceiling ~0.944 (-64bp vs PRIMARY). ρ=0.960 diverse but gap too large.
  **Both versions dead-listed.**

### Move D — FM new inputs (F1-F4): DEAD

`scripts/d13_move_f_fm_aug16.py` 16-field FM (12 d9h + 4 new: PitWindow/HazardDecay/
CompoundPressure/RaceStage). Standalone +20.1bp (0.92741 vs aug12 0.92540), ρ=0.919.
Min-meta: **-0.07bp FAIL**. Confirms FM-field-augmentation saturated at 12 fields.

## Remaining live moves (Day 15)

### PURSUE: α-calibrated τ-resweep (~30 min CPU)
τ chosen on OOF may not be τ-optimal for test (fold-train vs full-train counts differ).
Re-sweep τ ∈ {5k, 10k, 20k, 50k, 100k, 200k} on d13e and d13b Stint under corrected
α formula. EV +0-3bp. Cheap — run via harness first.

### PURSUE: d12_lr_meta single-candidate stack-add
K=21 + d12_lr_meta alone. If Δ > 0bp OOF → HEDGE candidate. Cost ~5 min.

### Move B — Pseudo-label cascade at K=21+hier-meta level (~3-4h CPU)
EV +5-10bp. Use d13e PRIMARY preds, top-30% confidence filter, retrain 5 fastest
bases, re-stack K=21+hier-meta. Risk: d5 widened gap on m5q (-4.2bp LB).

### Move C — DeepFM-lite (~3-4h CPU)
FM pairwise + 2-layer MLP head. New model class. EV +3-8bp standalone, +1-3bp stacked.
Risk: overfit (d9e FFM precedent). Mitigation: dropout + batch-norm + depth=2.

### Research loop trigger (Rule 7)
If no ≥+5bp structural move found: pause submits, web-search top-5 finisher writeups
from comparable playground comps, identify untried mechanism families.

## Falsified / dead — do NOT retry

All prior entries remain. Additional dead from Day-14:
- **TabPFN v2.5 fine-tune** — AUC ceiling 0.9444 regardless of row count
- **TabPFN v2.6 fine-tune** — OOM on P100 at any row count
- **FM-field-augmentation** (Move D / d14 aug13 / aug16) — saturated at 12 fields
- d14 Path B cohort sweep (Year/Year×Stint/Race × τ) — all NULL vs Compound×Stint PRIMARY

## Critical operating rules

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **BOTE before code** (Rule 19). Cost-gate: expected OOF lift × prob_useful > 0.1bp.
3. **NEW Day-13: ρ/G3/R7 heuristics DO NOT apply to new mechanism families.**
4. **Rule 16 Q6** (ρ vs FULL pool, not just PRIMARY): binding gate.
5. **GroupKF as secondary gate**. Strat AND not-regress-GKF, or pass GKF directly.
6. **Submit budget** 24/270; ~115 remaining. 40bp gap → structural moves only.
7. **Model-class diversification > tuning** (FM + hier-meta both confirmed).

## Pointers

- `audit/2026-05-13-d13-{path-b-hier-meta,d13d-path-b-gkf-probe}.md` — load-bearing
- `audit/2026-05-06-{blend-and-rho-probes,alpha-asymmetry-verification}.md` — Day-14 probes
- `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-prediction NULL + load-bearing DGP diagnostic
- `audit/archive-2026-05-06-handover-xvUN0-addendum.md` — archived per-branch addendum (consolidated at merge)
- `scripts/d13_move_f_features.py` + `scripts/d13_move_f_fm_aug16.py` — Move D
- `scripts/d14_dgp_residuals.py` — DGP-residual probe
- `scripts/artifacts/d13_move_f_fm_aug16_results.json` + `d12_tabpfn_finetune_150k_results.json`
- `scripts/artifacts/d14_dgp_residuals_results.json` + `probe_min_meta__d14_dgp_residuals.json`
- `kernels/d12-tabpfn-finetune-gpu/` (v2.5) + `kernels/d13-tabpfn-v26-strat/` (v2.6) — archived

## Day-14 PM `assess-synthetic-data-features-NYZuK`

PI thesis: predict every column from the rest, exploit emergent DGP
structure. Implemented in `scripts/d14_dgp_residuals.py` (4 LGBM
regressors → z-residuals + L1 anomaly as 5 new LGBM features; 11 min
wall). **Family CLOSED.** Std OOF 0.94200 (Δ −88 bp), K=2 min-meta
−0.025 bp NULL, K=22 add +0.17 bp at ρ=0.9958 → pred LB **−1.3 bp**.

Load-bearing diagnostic: across all 4 targets OOF RMSE ≈ marginal σ
within 3 sig figs — synthetic NN-DGP added near-independent per-
feature noise within rows. **Jointly explains** FM-aug12 saturation,
Move D NULL, Day-13/14 alt-axis 4-of-4 NULL, TabPFN 0.944 ceiling.
Per-row FE / self-supervised pretraining cannot break the ceiling;
single-base FE additions across LGBM/FM/DGP-residual now dead-listed.
Path forward: meta-layer (Path B variants), new model class (EmbMLP,
DeepFM-lite), or external data only. Detail: `audit/2026-05-06-d14-dgp-residuals.md`.
