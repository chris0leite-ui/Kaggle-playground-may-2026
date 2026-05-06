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

## 2026-05-06 PM addendum (branch `claude/ml-handover-alignment-xvUN0`)

PI redirect: experimentation culture; many small probes; BOTE-first;
"the solution is probably simple, maybe a code-quality fix".

**Built:** experimentation harness — `scripts/probe.py` (`bote` +
`gate`), `scripts/probe_min_meta.py` (K=21+N stack-add gate),
`scripts/probe_blends_K21.py`, `scripts/probe_rho_inventory.py`.
Rule 19 added to CLAUDE.md codifying BOTE-first / gate-after.

**Cheap probes (all via harness):**

1. **α-asymmetry verification.** OOF uses fold-train counts in α=n/(n+τ);
   test uses full-train counts. Bayesian-correct shrinkage, NOT a fixable LB
   cap. **PURSUE**: α-calibrated τ-resweep (~30 min).

2. **K=21 simple-blend probe.** mean/gmean/rank_mean/trimmed all regress
   19–32bp standalone vs PRIMARY. LR-meta-stays-best CONFIRMED.

3. **ρ inventory of 22 held candidates.** Best near-tie HEDGE: **`d12_lr_meta`**
   (OOF 0.95073, ρ=0.996, flip ratio 0.297).

4. **K=21 + d6_rule_compound_stint min-meta.** Δ −0.020bp NULL (already absorbed).

5. **K=21 + 3 (`d12_lr_meta` + `d10d_leak_corrected_meta` + `blend_rank_mean_K21`).**
   **Δ +1.298bp OOF** (0.95073 → 0.95086). `d12_lr_meta` dominates. **First non-NULL.**

**Open candidates (NOT YET RUN, BOTE-graded):**
- α-calibrated τ-resweep on PRIMARY hier-meta (PURSUE; ~30 min).
- `d12_lr_meta` single-candidate ablation (was in flight at session-end).
- Within-Race quantile-rank of LapTime_Delta as FM input (DEFER; H5 z-score leak fix needed).
- Per-Driver historical pit rate smoothed EB (DEFER; ~10 min).
- Year×Stint sparse-LR / FM partition (DEFER; ~30 min).

## 2026-05-06 PM addendum (branch `claude/decode-synthetic-data-uoPIn`)

PI redirect: "we have synthetic data; how could we decode this and get
the real DGP — give 3 options". Then "do it" (execute) → "submit"
→ "do 1" (leak_lookup) — full thread.

**Source confirmed**: `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`
is the verified parent of synth. 31/887 synth Drivers exact-match;
all Races/Compounds/Years overlap. Host removed `Normalized_TyreLife`
and added 856 ghost Driver codes.

**Synthesizer fingerprint (CTGAN/CopulaGAN-style)**: 97.55% of synth
`LapTime` values, 95% of `LapTime_Delta`, 99.95% of `RaceProgress`,
87% of `Cumulative_Degradation` are drawn from the original's
empirical marginals. Joint structure broken (only 5.8% of
`(LapTime, TyreLife, Compound)` triples survive). NTL formula
recovered: `NTL = TyreLife / D(Driver, Race, Year, Stint)`.

**Probes run (5 base mechanisms × 3 hier-meta configs = 8 gate calls):**

| Mechanism | Std OOF | min-meta Δ | hier-meta(K=22) Δ | LB |
|---|---:|---:|---:|---:|
| d15_orig_transfer (LGBM on orig) | 0.85138 | **+0.778** | **+1.127** | 0.95049 TIE |
| d15_orig_cb / xgb / lgbm_t | 0.84-0.87 | — | K=23 +0.005 / K=24 +0.33 | not subm. |
| d15_decode_normalized_tyrelife | 0.94162 | −0.008 NULL | — | — |
| d15_physics_residual | 0.94228 | −0.036 NULL | — | — |
| d15_leak_lookup | 0.94203 | +0.270 soft | K=22 −0.90 / K=23(+orig) +0.19 | not subm. |

**Submits this branch (3 today)**:
1. K=22 LR-meta + d15_orig_transfer → LB 0.95039 (−10 bp; meta-arch confound, NOT mechanism falsification)
2. K=22 hier-meta + d15_orig_transfer → LB 0.95049 (TIE with old PRIMARY — HEDGE-tier, ρ=0.998 flips 180<200)
3. None — slot saved (K=24 had R7 violation; K=23 leak+orig predicted tie)

**Best held-not-submitted**: `submission_d15_path_b_K23_leak_orig.csv`
(OOF 0.95096, ρ=0.9986, flips 198 R7 ✓; predicted LB tie at OLD
PRIMARY 0.95049 → not submitted but ready for R5 final-window).

**Net for the comp**: decoded-data thesis is **REAL but
QUANTIZATION-BOUNDED** on this comp. orig-trained transfer base
contributes +1.13 bp OOF at hier-meta with ρ=0.565 vs OLD PRIMARY
single-row (most-diverse base since d9f FM_A) but lands within
Kaggle's ~5 bp LB resolution. Combining with main's NEW PRIMARY
(LB 0.95059, +10 bp via inv_laps + B-GPU dae) is the load-bearing
next move — orthogonal mechanism families.

**Audit**: `audit/2026-05-06-d15-decode-synthesizer.md` (fully
detailed — fingerprint, gates, hier-meta probes, 4 ranked next-steps,
3 friction tags).

**Friction tags added**:
- `external-data-arch-bag-redundant-when-shared-training-data`
- `meta-arch-required-for-orthogonal-base-eval`
- `lb-quantization-floor-defeats-decoded-data`

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
- `scripts/d13_move_f_features.py` + `scripts/d13_move_f_fm_aug16.py` — Move D
- `scripts/artifacts/d13_move_f_fm_aug16_results.json` + `d12_tabpfn_finetune_150k_results.json`
- `kernels/d12-tabpfn-finetune-gpu/` (v2.5) + `kernels/d13-tabpfn-v26-strat/` (v2.6) — archived
