
> Day-8 PM (`research-feature-engineering-7oCmj`) section archived to
> `audit/archive-2026-05-09-handover-day8-pm-section.md`.

---

## Day-9 PM decode-data-process-5uLq3

DGP-decode session (no LB submissions). Twenty-three probes
(Q1-Q10 + qB-qZ), twenty-five commits, twelve audit docs in
`audit/2026-05-09/`. Pushed disc-AUC gap host-vs-candidate from
0.999 (off-the-shelf SDV CTGAN baseline) to **0.7160** (analytic
resample-and-cond pipeline) — half of the way to the perfect-mimicry
lower bound of 0.4944.

**Read first**: `audit/2026-05-09/2026-05-09-EXEC-SUMMARY.md`
(plain-English) and `audit/2026-05-09/2026-05-09-PHASE-B-FINAL-and-plan-v3.md`
(full ledger).

**DGP picture (final):** input aadigupta1601 → custom marginal that
suppresses PitStop=1 by 0.54× → per-cell NN generator (the unsolved
residual; produces 73% novel values per cell, no noise, NOT BGMM /
KDE / affine / global / cross-cell-mixed) → structured per-cell
Driver/Stint sampling → drop Norm_TyreLife → ship.

**Architecture exclusion ledger** (host generator NOT any of):

| Architecture | disc-AUC |
|---|---:|
| SDV CTGAN (5/10/20 ep + synth-marginal cond) | 0.9993-0.9997 |
| SDV GaussianCopula | 0.9988 |
| SDV TVAE 10 ep | 0.9991 |
| SDV CopulaGAN | conclusive by pattern |
| noisy-orig + Gaussian sigma > 0 | monotone-worse |
| per-cell BGMM (4 floats) | 0.8643 |
| per-cell KDE bw 0.05-0.5 | 0.7448-0.7657 |
| global float sampling | 0.9907 |
| cross-cell mixing fraction > 0 | monotone-worse |
| affine moment-matching | 0.9883-0.9979 |

**New findings F11-F15:** Driver/Stint structured per-cell (qH +14 pp);
continuous columns strictly per-cell (qU); 73% novel `(Y, C, LapTime)`
keys per cell (qR); per-cell mean shifts -2.81 with std ratio 0.87 but
non-affine (qX/qY); d16++ standalone synth AUC 0.940 (+2.5 pp over
d16) but only +0.149 bp at K=4+1 (rank-lock saturates).

**qZ d16++ artifacts saved** at
`scripts/artifacts/dgp_v3_qZ_{oof_strat, test, train_synth}.npy`.
Stack-add gate measured at +0.149 bp (below +0.5 strict threshold).

**Next-session first actions** (in EV / cost order):
1. **TabDDPM-on-orig** if the install-debug session can land it. The
   single most likely candidate to close the 0.22 disc-AUC residual.
   ~30 min GPU.
2. **Normalising flow (RealNVP / NSF) per-cell** with cell-key
   conditioning. Skew-sensitive; matches the qX skewness diffs.
3. **Re-decompose K=4 PRIMARY** to swap d16 for qZ and re-measure;
   small expected lift but free.
4. **Accept structural decode as the answer** and wrap the comp;
   the rank-lock cap on K=4+1 means decode-derived features are
   bounded ≤1 bp at the LB.

**Friction tags promoted (this session, in audit/friction.md):**
`synth-rows-are-not-literal-copies-of-orig-rows` (retract P1c),
`host-not-in-sdv-library`, `noise-on-continuous-cols-makes-disc-worse-not-better`,
`cond-driver-stint-on-cell-saves-14pp`,
`extending-cond-axes-monotonic-down-to-LapN-then-sparsity-bites`,
`affine-moment-matching-fails-skewness-non-trivial`,
`host-cont-vals-strictly-per-cell-no-cross-cell-mixing`,
`rank-lock-saturation-puts-cap-on-K4plus1-with-decode-features`.

Postmortem: `audit/2026-05-09-postmortem-decode-data-process-5uLq3.md`.

---

## Day-9 evening analyze-synthetic-data-generation-BtmFl (autonomous loop)

**Twenty-one probes (qAA → qAY) translating decode insights into K=4+1 lift.**
Headline: K=9 = K=4 + qAT + qAV + qAO + qAA + qAF + Path-B C×S τ=20k →
**OOF +2.017 bp vs PRIMARY 0.95403** (PRIMARY 0.95421). Six commits
pushed; audit at `audit/2026-05-09/2026-05-09-final-results-summary.md`
+ `audit/2026-05-09/2026-05-09-qAK-breakthrough.md`.

**Breakthrough mechanism**: orig-kNN with **K=1 strictest match** (3
features only: label/distance/level_used) inside the 6-axis cell key
(Y, C, PS, R, S, LapN) with hierarchical fallback (L6→L5→L4→L3). Each
slim base captures **per-row identity attribution** against orig rather
than smooth aggregate. Two complementary distance spaces:
- qAT: 4-feat (LapTime/Δ/CumDeg/RP) standardized euclidean
- qAV: 7-feat (TyreLife/Position/LapTime/CumDeg/RP/Δ/LapNumber) standardized euclidean

The 6-axis cell key matches the host's decoded cond-vector schema (qH-qM
prior session). 79% of train rows fit at L6.

**Key probe results**:
- qAT alone: standalone OOF 0.821, ρ_test 0.644, K=4+1 plain LR-meta **+1.172 bp** PASS
- qAV alone: standalone OOF 0.841, ρ_test 0.618 (lowest ever), K=4+1 +1.162 bp
- qAK (K=3, 6 features): K=4+1 +0.717 bp PASS — first probe to break gate
- qAR (yekenot recipe + kNN combined): K=4+1 +0.044 bp WEAK (kNN signal absorbed)
- qAY (K=1 cosine): redundant with qAT (ρ 0.914)

**Path-B C×S τ=20k amp ladder** vs PRIMARY 0.95403:
| Pool | Δ_oof bp |
|---|---:|
| K=4 + qAT + qAV (qAX) | +1.800 |
| K=7 qAT+qAV+qAF | +1.929 |
| K=8 qAT+qAV+qAA+qAF | +1.974 |
| **K=9 qAT+qAV+qAO+qAA+qAF** | **+2.017** ← BEST |
| K=10 +qAK | +1.950 (qAK redundant) |

**Submissions ready** (NOT submitted; awaits PI approval per Rule 1):
- `submissions/submission_qAT_qAV_qAT_qAV_qAO_qAA_qAF_pathb_cs_tau20000.csv` (best)
- 4 other qAT_qAV variants
- 7 K=7 qAK_qAA_qAF / qAK_qAO_qAF variants

**Friction tags proposed**:
- `tight-K1-with-6-axis-cell-hierarchical-fallback-breaks-rank-lock`
- `slim-feature-base-design-required-for-per-row-attribution-signal`
  (qAR confirmed: yekenot+kNN absorbed; slim 3-feat qAT escaped)
- `cell-granularity-not-distance-metric-determines-attribution-quality`
  (qAY cosine ≈ qAT euclidean at 4-feat scale)

**Updated rank-lock framing**: LR-meta absorbs base predictions whose
conditional-target-correlation is parallel to K=4's logit, BUT does NOT
absorb features that match each test row to specific orig rows via
cell-conditional kNN at sufficient cell granularity. The MIDST 2025
"loss-features-across-noises" pattern is the formal analog.

---
