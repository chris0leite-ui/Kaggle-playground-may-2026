# 2026-05-08 — kernel-SVM-meta + kNN + NCA-kNN on K=4/K=10 ensembles

**Result: every non-linear meta-stacker tested ties or underperforms the
linear LR-meta on the K=4 forward-greedy pool. The K=4 stack is
saturated for meta-routing.**

## Context

After kernel-SVM and SVM specialists on raw features both nulled
(`audit/2026-05-08-svm-kernel-probe.md` + `audit/2026-05-08-svm-specialists.md`),
the PI redirected to (a) SVM-as-meta-stacker over the new K=4 PRIMARY
ensemble (set on `origin/claude/review-ml-handover-IMUNP`, LB 0.95351)
and (b) a kNN family with feature subsets and learned manifold distance.

Three new mechanisms tested as meta-stackers over the K=4 pool:

1. **Kernel-SVM-meta** — Nyström-RBF + LinearSVC (squared hinge) on the
   K=4 base predictions expanded raw + rank + logit (12 features).
2. **kNN with feature subsets** — 10 hand-picked subsets of ≤5 raw
   features, each fed to a distance-weighted kNN K=50; results pooled
   via a thin LR meta-stacker.
3. **NCA-kNN on the ensemble** — Neighbourhood Components Analysis
   learns a 3-D linear projection of the K=4 (12 features) and K=10
   (30 features) ensemble predictions; kNN K=50 in that learned space
   is the final classifier.

## Reference baselines (K=4 pool, OOF, all on identical folds)

| meta-stacker | OOF |
|---|---:|
| K=4 plain LR-meta | 0.95399 |
| K=4 + Path-B C×S τ=100k (PRIMARY rebuild) | **0.95403** |
| K=4 kernel-SVM-meta linsvc γ=0.02 | 0.95403 (ties) |
| K=4 kernel-SVM-meta linsvc γ=0.05 | 0.95401 |
| K=4 kernel-SVM-meta linsvc γ=0.10 | 0.95397 |
| K=4 kernel-SVM-meta klogreg γ=0.02 | 0.95396 |

Path-B amp at K=4 is +0.04 bp over plain LR — confirms the IMUNP
branch's finding that the per-segment shrinkage stacker is "a myth at
sparse pools" (the +6–11× amplification was K=21-era-specific).

Kernel-SVM-meta with γ=0.02 ties Path-B at OOF level. The flips
diagnostic shows it makes asymmetric rare-class calls vs Path-B
(linsvc: 1882 PRIMARY positives → negative, 0 added; klogreg: 0 lost,
11608 added). Structurally different, aggregate-AUC matched.

## kNN with feature subsets

10 subsets of ≤5 raw features; each subset → kNN K=50 distance-weighted
on full 350k rows. LR-pool over the 10 OOF columns.

| best subset | OOF |
|---|---:|
| top-5 numeric (TyreLife, LapNumber, Stint, RaceProgress, Cumulative_Degradation) | **0.89426** |
| multi-categorical (Year, Compound_LE, TyreLife, Stint, RaceProgress) | 0.87871 |
| recipe-5 (TyreLife, Stint, RaceProgress, Compound_LE, LapNumber) | 0.86582 |
| ... 7 more subsets ... | 0.75–0.85 |
| **10-subset LR-pool** | **0.92285** |

Pool reaches the LR-bank ceiling (0.92776). Subsets that label-encode
high-cardinality categoricals into the distance metric (Compound_LE) hurt
kNN — known kNN failure mode.

## NCA-kNN on ensemble base predictions

NCA learns a linear projection W : R^d → R^k that maximises leave-
one-out kNN classification accuracy on the training set. Standard
"learned manifold distance" for kNN. NCA's pairwise-distance loss is
O(n²) per L-BFGS iteration → fit on a stratified 8k subsample per
fold (50k OOM-killed, 20k took 22 min/fold — too slow). Apply the
learned projection to all 350k training rows + 88k val + 188k test
for kNN classify.

| variant | input | NCA dim | standalone OOF | ρ vs PRIMARY | K=4+1 gate | K=10+1 gate |
|---|---|---:|---:|---:|---:|---:|
| NCA-K4 ensemble | 4 bases × 3 = 12 feat | 3 | 0.94683 | 0.93 | **−0.07 bp** | **−0.02 bp** |
| NCA-K10 ensemble | 10 bases × 3 = 30 feat | 3 | 0.94673 | 0.90 | **+0.01 bp** | **+0.02 bp** |

Both gates within ±0.1 bp = full absorption.

## Why nothing fires

Three reinforcing reasons:

1. **K=4 is already saturated for meta-routing.** Logit effective rank
   on the K=27 pool was 3.23 (per IMUNP audit); K=4 is selected to
   span that 3-D subspace already. Any non-linear router that
   maintains AUC ranking on the 4 base predictions can at best tie
   the LR-meta because the answers it can express are the same.
2. **kNN voting is high-variance** when the inputs are already
   calibrated probabilities — the LR-meta's smooth boundary captures
   the right linear combination cleanly; kNN's local voting adds
   noise without new signal.
3. **The 8k NCA subsample is ~2.3% of full train.** The learned metric
   is fit on a small sample; even if it captures the right structure
   on that sample, its variance dominates the systematic gain
   compared to a closed-form LR fit on all 350k rows.

## Friction (load-bearing for future sessions)

`non-parametric-meta-on-K=4-cant-beat-LR-meta-without-new-input`. A
non-parametric or non-linear meta-stacker over the K=4 pool can at
best tie the LR-meta. To exceed it, the new mechanism must either
(a) bring fresh input features (not just the K=4 predictions, but
e.g. the original 14 raw-data columns alongside), or (b) introduce
predictions from a model that genuinely produces different rankings
on hard cases — i.e., a new base, not a new router.

## What would change the verdict

If the K=4 pool's effective rank really is 3, no router can extract
more than 3 dimensions of signal. Two follow-up tests this session
closed two of the three escape paths:

1. ~~**Combined raw-features + base-predictions input.**~~ TESTED.
   `concat(K=4 bases, top-5 numeric)` → LR-meta = +0.03 bp (null);
   kernel-SVM-meta = −1.64 bp (regress). The K=4 bases already absorb
   the top-5 numerics — adding raw features to the meta input does
   nothing because every base is already a model trained on those
   features. Friction confirmed.
2. **Sequence-level fingerprint LightGBM.** TESTED. Within-stint
   structural features (stint_lap_idx, prev_stint_length, compound
   history one-hot, position_change_in_stint, stint_lap_frac) added
   to the standard 14-column feature set and trained as a single
   LightGBM base. Standalone OOF 0.94202; gate K=4+1 **+0.15 bp**,
   K=10+1 −0.08, K=27+1 −0.08. The +0.15 on K=4+1 is the only
   positive meta-add in this entire branch's arc, but it's within
   fold noise and absorbed at K=10 / K=27. Indication that the
   sequence-feature axis is real but small at this scale; richer
   features (HMM transition probabilities, AR(1) TyreLife, neural
   sequence model) might extract more.
3. **Add a base that's structurally different from all 4** (yekenot
   recipe to a non-tree class, FastF1 hard-join). UNTESTED here;
   remains the strongest open axis from HANDOVER.

## Artifacts

Saved (pushed to `chrisleitescha/s6e5-artifacts` Kaggle Dataset):

- `oof_K4_fwd_pathb.npy` + `test_K4_fwd_pathb.npy` — K=4 PRIMARY
  rebuild (OOF 0.95403).
- `oof_svm_kmeta_lr_meta_k4.npy` + test — K=4 plain LR-meta (0.95399).
- `oof_svm_kmeta_linsvc_g{0.02,0.05,0.1}.npy` + test — kernel-SVM-meta.
- `oof_svm_kmeta_klogreg_g{0.02,0.05,0.1}.npy` + test — kernel-logistic.
- `oof_knn_s{1..10}.npy` + test — 10 kNN feature-subset heads.
- `oof_knn_pool_lrmeta.npy` + test — LR-pool over the 10 kNN heads
  (0.92285).
- `oof_knn_nca_K4_ensemble.npy` + test — NCA on K=4 ensemble (0.94683).
- `oof_knn_nca_K10_ensemble.npy` + test — NCA on K=10 ensemble (0.94673).

Scripts:

- `scripts/svm_kernel_meta.py` — kernel-SVM-meta probe + K=4 PRIMARY
  rebuild.
- `scripts/knn_feature_subsets.py` — 10 kNN heads + LR-pool.
- `scripts/nca_knn.py` — NCA-kNN on raw features and on K=4/K=10
  ensembles.
