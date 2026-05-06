# 2026-05-13 — EDA hypothesis execution results

Day-13 evening: PI approved "go as you recommend. do it all".  Built and
gated the four EDA hypotheses (H3, H4, H5, H1).  Three falsified at the
GBDT/post-hoc class.  H1 in the leakage-robust class produces a candidate
with the d9h-style fingerprint that LB-amplified +3 bp on a +0.013 bp
OOF tie.

PRIMARY K=22 reconstructed at OOF 0.950739 (matches d9h target 0.95073
exactly).  Ranks: `oof_PRIMARY_K22_strat.npy`, `test_PRIMARY_K22_strat.npy`.

## Results table

| H | Mechanism | Std OOF | ρ PRIMARY | min-meta Δbp | K=23 add Δbp | K=22 swap Δbp | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| H3 | per-Stint isotonic (in-sample) | n/a | n/a | +3.89 | n/a | n/a | misleading (in-sample overfit) |
| H3 | per-Stint isotonic (nested CV)  | n/a | n/a | **-1.41** | n/a | n/a | **FALSIFIED** |
| H3 | Stint × Year nested CV          | n/a | n/a | -3.73 | n/a | n/a | FALSIFIED |
| H5a | LGBM, LapTime_Delta_zr replace  | 0.94020 | 0.9495 | -0.09 | -0.01 | n/a | FAIL gate |
| H5b | LGBM, raw + zr augment          | 0.94168 | 0.9560 | -0.10 | +0.02 | n/a | FAIL gate |
| H4a | LGBM, +cumdeg_per_lap           | 0.94101 | 0.9565 | -0.08 | +0.18 | n/a | FAIL gate |
| H4c | sparse-LR, hashed q5 keys       | 0.88957 | 0.8389 | -0.07 | +0.06 | n/a | FAIL gate |
| **H1** | **FM_aug15 (12 fields + CRT, Cdpl, Ldz)** | **0.92711** | **0.9091** | **-0.09** | **-0.02** | **-0.04** | **CANDIDATE** |

PRIMARY anchor: K=22 OOF 0.95074 / LB 0.95034.

## Why H3, H4, H5 falsified at GBDT-class

The K=22 LR-meta already absorbs every drop of LapTime_Delta and Cum_Deg
signal via the 22 GBDTs/CatBoosts/HGBCs.  Adding another GBDT base with
the same raw features (even with z-score normalization or ratio
combinations) produces a base with ρ ≈ 0.95 vs PRIMARY → no orthogonal
information.  Strat OOF sees the redundancy and assigns near-zero weight
in min-meta.

H3 isotonic was "free" but nested-CV showed it fits per-fold noise; the
+4 bp in-sample upper bound was an artifact of the same OOF being used
for fit and evaluation.

## Why H1 is the surviving candidate

FM-class is the only model class with leakage-robust amplification on the
public LB (Day-12 GroupKF rebuild: FM −9 bp under GKF vs GBDT −200 to
−343 bp; Day-9 d9c-d9h-d9i all delivered +2 to +3 bp LB on +0.01 to
+0.18 bp OOF predictions).  The H1 FM_aug15 has the same fingerprint:

- Standalone std OOF 0.92711 vs d9h FM_aug12 0.92540 = **+1.7 bp stronger**
- ρ vs PRIMARY 0.9091 vs d9h FM_aug12 0.9171 = **more diverse**
- K=23 add OOF Δ = −0.02 bp (within ±0.05 bp noise band)
- New input fields are: Compound × TL_q5 × RP_q5 (the 3.35× lift
  triplet from Phase C), cumdeg_per_lap_q5 (the independent-of-TyreLife
  signal from Phase C), LapTime_Delta_zr_q5 (Race-Year-Compound z-score
  from Phase F)

5-question pre-flight (Rule 16):
1. Mechanism family: factorization_machine_aug12 (existing) but **3 new
   field types** (CRT, Cdpl, Ldz) — passes "new mechanism" check
2. Class check: NOT meta-only, NOT rule-residual-on-raw, NOT GBDT,
   NOT formulation-already-in-pool — passes
3. Std OOF prediction: predicted ~0.926 by analogy to aug12 + 3 new
   fields adding ~+1-2 bp. Actual 0.92711.  ✓
4. ρ prediction: predicted ~0.91 (slightly more diverse than aug12's
   0.917 due to new orthogonal fields).  Actual 0.909.  ✓
5. Gate precedent: d9h S2 K=22 add OOF +0.013 bp PASS (TIE_EXPECTED)
   → LB +3 bp.  H1 K=23 add OOF -0.02 bp is even tighter; precedent
   permits LB lift.

5/5 coherent — EV midpoint stands.

## Submission candidates ready

| File | Composition | OOF | EV bp (P/M/O) |
|---|---|---:|---:|
| `submissions/submission_K23_add_H1_FM_aug15.csv` | K=22 + FM_aug15 (preserves d9h FM_aug12) | 0.95074 | -1 / +1 / +4 |
| `submissions/submission_K22_swap_H1_FM_aug15.csv` | K=21 + FM_aug15 (drops d9h FM_aug12; aug15 strictly contains aug12 fields + 3 new) | 0.95073 | -1 / 0 / +3 |

Recommended: **K=23 add** (matches d9h S2 pattern, OOF 1bp higher than swap).

## Status — H1 SUBMITTED, FALSIFIED

PI approved.  Submitted `submission_K23_add_H1_FM_aug15.csv` at 06:58 UTC.

**LB 0.95032** = -2 bp vs PRIMARY-of-record anchor 0.95034.  Lands in
the pre-submit "LB -1 to -3 bp" outcome zone.

### What this falsifies

The d9h FM-field-augmentation lever does NOT generalise to arbitrary new
field types.  3 new fields (CRT = Compound × TL_q5 × RP_q5, Cdpl =
cumdeg_per_lap_q5, Ldz = LapTime_Delta_zr_q5) added noise that the LR-meta
could not downweight on Strat OOF.  The 4 fields d9h added (next_compound,
prev_compound, Cum_Deg_q5, LapTime_Delta_q5) were a specific lucky pull,
not a structural property of "FM with more fields".

### Updated FM-class precedent table

| Submit | Std OOF | ρ vs PRIMARY | K-stack OOF Δ bp | LB Δ vs prior | Outcome |
|---|---:|---:|---:|---:|---|
| d9c FM_solo | 0.9207 | 0.899 | min-meta +0.18 | +3 | LB-amplify |
| d9f FM partition | 0.825 / 0.884 | 0.487 / 0.861 | K=21 swap +0.29 | +2 | LB-amplify |
| d9h FM_aug12 add | 0.9254 | 0.917 | K=22 add +0.013 | +3 | LB-amplify (300×) |
| d9i FM partition aug | 0.881 / 0.886 | 0.720 / 0.863 | K=21 swap -0.19 | +3 | LB-amplify (direction-flip) |
| d13 V1 5/3 multi-FM | (similar) | sim | +0.06 | 0 | TIE |
| **H1 FM_aug15 add** | **0.9271** | **0.909** | **K=23 add -0.02** | **-2** | **REGRESS** |

H1 had the strongest standalone OOF and the most-diverse ρ of any FM
attempt to date — and still regressed.  Lever is saturated.

### Strategic implication

Pivot away from FM-field augmentation.  The TabPFN-2.5 fine-tuned kernel
prepared on Day-12 (`kernels/d12-tabpfn-finetune-gpu/`) is the only live
"+10 bp shot" remaining and now the highest-priority Day-14 move.

### Side discovery from polling submissions

Current actual LB-best on this account is **0.95041**
(`submission_d13_path_b_stint_tau100000.csv`, posted today 05:34 UTC) —
+7 bp over the d9h/d9i 0.95034 PRIMARY-of-record.  Despite failing the
G3 leakage flip-ratio gate (0.211 FAIL), the per-Stint EB-shrunk
segmented LR-meta lifted LB +7 bp.  CLAUDE.md current_state and the
EDA synthesis treated 0.95034 as PRIMARY; it is now 0.95041.  Gap to
top-5% (0.95345) closes 31 bp → 24 bp.

Implication: G3 may be over-conservative.  Worth a Day-14 re-audit of
the 4-gate leakage filter for per-segment LR-meta candidates.  Not
the EDA branch's scope.

### Submission token spend

1 / 9 remaining used today.  8 tokens left.

## Pointers

- `scripts/eda_deep/10_primary_rebuild.py` — PRIMARY K=22 reconstruction
- `scripts/eda_deep/11_H3_calibrate.py` — H3 isotonic, falsified
- `scripts/eda_deep/12_H5_laptimedelta_z.py` — H5 GBDT, falsified
- `scripts/eda_deep/13_H4_cumdeg_per_lap.py` — H4 GBDT, falsified
- `scripts/eda_deep/14_H1_aug15_fm.py` — H1 FM_aug15, candidate
- `scripts/artifacts/oof_H1_FM_aug15_strat.npy` (439,140) and
  `test_H1_FM_aug15_strat.npy` (188,165)
- `scripts/artifacts/oof_K23_add_H1_FM_aug15_strat.npy` and
  `test_K23_add_H1_FM_aug15_strat.npy`
