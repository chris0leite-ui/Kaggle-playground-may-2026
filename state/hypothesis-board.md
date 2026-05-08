# Hypothesis board

Live working notes on what's still open, what's killed, and what's
parked. For history, read `audit/research/` and the postmortems.

## Insights that are still load-bearing

- **The per-segment shrinkage trick fires on a redundant pool, not on a
  saturated one.** It worked when the stack had 21-22 bases of similar
  GBDT/FM material (high redundancy, low effective rank). It stopped
  firing once the stack was small (effective rank ≈ pool size) or once
  the pool already routed by the segmenting variable at the base level.
- **The synthetic data-generating process is conditionally near-independent
  per row.** Self-supervised feature engineering — predict one column
  from the others — produces residual variance ≈ marginal variance.
  Five separate probes confirmed this. Per-row feature engineering on
  these 14 raw columns is a dead axis.
- **The stack of 21-22 bases is rank-locked at the meta-stacker.** Even
  bases with 0.7-0.9 correlation to PRIMARY add zero. The escape was a
  feature-recipe transfer (yekenot), not another base.
- **Target reformulation is a leakage trap on this comp.** Any per-group
  computation that uses the label has to be redone per cross-validation
  fold; otherwise OOF inflates 88-100%. Three reformulations (inv-laps,
  pit-horizon, reverse-cumulative) all collapsed under the strict audit.
- **Public leaderboard is row-i.i.d. with train.** Adversarial-validation
  classifier scores AUC = 0.502; Stratified-Kfold is the LB proxy, not
  GroupKFold. Path-B amp does not transfer to private LB if the test
  partition turns out to be different (unknowable until comp ends).

## Killed — do not retry

- Target-reformulation single-add (inv-laps, pit-horizon, reverse-cumulative,
  stint-progress) — all leaky.
- Anything stacked on top of those leaky targets (path-B with inv-laps
  injected; the "megapool" 25-base variant; the 23-base DAE + inv-laps).
- Multi-level 4-tier per-segment stacker — 5 configurations all null.
- Day-16 virgin axes — 11 of 11 null, falsified, or killed.
- TabPFN v2.5 / v2.6 — AUC ceiling 0.944; v2.6 ran out of memory on
  Kaggle P100.
- 16+ field factorisation machines — saturated at 12 fields.
- Drop-GBDT pool refactor — leak-eaters carry signal that survives at
  the meta on this row-i.i.d. test set.
- Simple K=21 blending (mean / geometric-mean / rank / trimmed) — −19
  to −32 bp; the LR-meta does real work.
- α-calibrated τ resweep — already at the local optimum.
- Multi-target neural network with auxiliary heads — null.
- Masked-column self-prediction (DGP-residuals) — null per the
  conditional-near-independence finding.
- Twin-pool 2-meta blending — collapses rank info.
- Conformal isotonic recalibration of PRIMARY — already globally
  calibrated.
- Adversarial-validation sample weighting — bounded by AV-AUC = 0.502
  (i.e., none).
- Yao/Vehtari covariance-modulated per-segment stacker — overshrinks
  along the highly-correlated base directions the stacker uses for
  routing.
- **Non-LR meta architecture (LightGBM on PCA / raw expansion).**
  PCA-meta probe 2026-05-08 PM: LightGBM meta is *worse* than LR meta
  by 1-2 bp at every input representation tested (PCA top-K for K
  in 3..27, K=10 / K=27 raw [P, rank, logit] expansion). EXP-NEW
  closes FALSIFIED. The "non-LR meta" clause of A30 is empirically
  refuted. See `scripts/probe_pca_meta.py`,
  `audit/2026-05-08-pca-meta-probe.md`, A30b.
- **Random forest as meta-stacker over K=4** (12 feat, today). OOF
  −1.54 bp vs LR-meta. Bagged-tree variant of the same finding;
  closes non-LR meta family across boosted *and* bagged tree
  classes. See `audit/2026-05-08-rf-forest-sweep.md`,
  `scripts/probe_forest_sweep.py` Angle B.
- **Random forest as meta-stacker on combined input** (K=4 expansion
  + 6 raw numerics = 18 feat, today). OOF −0.70 bp vs LR-on-same.
  Adding raw signals to the meta does not rescue tree-class meta.
  Same audit as Angle B; Angle C.
- **Path-B segmentation in PC space.** PCA on K=27 logit pool
  decorrelates the routing variables; Path-B C×S on top-K PCs scores
  −28 to −34 bp vs K=10 plain LR. Path-B fires on redundant pools,
  not orthogonal ones. See A30c.
- Kernel SVM family (Nyström-RBF + LinearSVC and + kernel-logistic):
  standalone OOF 0.912-0.914 (+56 bp over matched-feature linear LR,
  but −400 bp from PRIMARY); both variants null on K=10 sparse and
  K=27 dense pools (Δ −0.09 to +0.00 bp). 8th rank-lock confirmation;
  kernel-class disagreement isn't enough when standalone-AUC gap to
  GBDT-class exceeds 300 bp.
- SVM specialists (5 variants: linear-global / linear-per-Year /
  linear-per-Compound / linear-per-Stint / gaussian-kernel-per-Year)
  on the same 45-feature recipe. Every variant nulls at K=27 PRIMARY
  (Δ −0.09 to +0.00 bp). Strongest was kernel-per-Year (+0.05 bp on
  K=10 sparse, +0.00 on K=27). linear-per-Year produced the lowest
  ρ_test in project history (0.548, vs prior low 0.71 of bagged LR)
  yet still nulled — 9th rank-lock confirmation that low correlation
  alone is not sufficient meta-utility. SVM family closed across both
  global and specialist axes.
- **Kernel-SVM-meta on K=4 ensemble.** Nyström-RBF + LinearSVC over
  the K=4 base predictions (12 feat: raw + rank + logit) with γ-sweep.
  γ=0.02 *ties* Path-B PRIMARY at OOF 0.95403; γ=0.05/0.10 within
  noise. Asymmetric flip diagnostics: linsvc drops 1882 PRIMARY
  positives without adding any; klogreg adds 11608 without dropping.
  Same OOF, structurally different rare-class operating points —
  candidates for blend on a future submission slot.
- **kNN with feature subsets** (10 subsets ≤5 features, distance-
  weighted K=50). LR-pool over 10 heads → OOF 0.92285 (≈ LR-bank
  ceiling 0.928). Best single subset: top-5 numeric at 0.89426.
  Label-encoded categoricals (Compound_LE) hurt kNN distance.
- **NCA-kNN on K=4 / K=10 ensemble** (Neighbourhood Components
  Analysis = learned manifold distance). Standalone OOF 0.946–0.947
  (~70 bp below LR-meta 0.954). K=4+1 gate ±0.07 bp; K=10+1 gate
  ±0.02 bp — null on both pools. Friction
  `non-parametric-meta-on-K=4-cant-beat-LR-meta-without-new-input`.
  K=4 is already saturated for meta-routing; logit effective rank
  ~3 ⇒ any router can at best tie LR-meta until a new base is added.
- **Combined-input meta-stacker** (K=4 base predictions + top-5 raw
  numerics standardised → 17-feature meta input). LR-meta: +0.03 bp
  vs K=4 LR-meta (null). Kernel-SVM-meta γ=0.02: −1.64 bp (regress).
  Confirms friction: the K=4 bases already absorb the top-5 numerics.
- **Sequence-level fingerprint LightGBM** (HANDOVER #1 axis;
  within-stint structure: prev_compound, compound_changes,
  stint_lap_idx, prev_stint_length, position/tyre_life at stint
  start, position_change_in_stint, stint_lap_frac, compound history
  one-hot). Standalone OOF 0.94202. K=4+1 **+0.15 bp** (only
  positive meta-add in this branch's arc), K=10+1 −0.08, K=27+1
  −0.08. Magnitude within fold noise but structurally in the right
  direction on the sparse pool. Richer sequence features (HMM
  transitions, AR(1) TyreLife, RNN-class sequence model) could
  potentially scale this up — open path for next session.

## Open priorities (best EV / cost first)

(Reordered 2026-05-08 PM after EXP-NEW falsification: non-LR meta is
closed, so the "structurally untested architecture" priority drops out.
Updated 2026-05-08 evening after forest sweep: forest base on yekenot
recipe is the first non-null new-base lift on K=4.)

0. ~~Path-B Compound × Stint τ=100k refit on K=5 = K=4 + RF-yekenot.~~
   **Closed 2026-05-08 evening.** Refit produced K=5 + Path-B C×S
   τ=100k OOF 0.95405 (vs K=4 + Path-B PRIMARY 0.95403, Δ +0.02 bp).
   ρ vs PRIMARY 0.999917 → tie-band at LB per Rule 27. PI held
   submission. **Path-B absorbs the +0.25 bp K=4+1 forest lift to
   +0.02 bp** — confirms Day-15 friction
   `path-b-amp-only-fires-on-meta-arch-not-base-add`. Three τ
   variants saved as R5 hedge. Forest family characterized
   end-to-end. See `audit/2026-05-08-rf-forest-sweep.md`.
1. **R5 hedge preparation for the final-window probe.** List the
   OOF-best candidates that were rejected for public-LB regression.
   Hedge ladder already populated. Cost 30 minutes. **Highest-value
   next move now that all single-axis lift candidates are NULL.**
2. **RealMLP with 24 ensembles** (instead of the current 4). Yekenot's
   published recipe. Predicted +1 to +3 bp standalone; cost 3.5 hours
   GPU on Kaggle. Low confidence — sqrt(n_ens) law gives ≤1 bp.
3. **Per-Year CatBoost-yekenot specialists.** Day-12 found 2023 was the
   easiest year. Predicted ±2 bp; cost 30 minutes GPU. Low confidence.
4. **Wrap-up posture.** Top-11% achieved. Reserve compute for the next
   competition. Durable artifacts already shipped (LR-diagnostic suite,
   BOTE harness, decisions.jsonl, PCA-meta probe).
5. **FastF1 lap-by-lap pit-call hard-join.** Only single-mechanism path
   to top-5. Predicted +10 to +30 bp. Cost: 1-2 days of work, which is
   prohibitive given days remaining and the 1.4% match-rate cap from
   synthetic driver codes.

(Dropped: "Sequence-level DGP fingerprinting" was already closed by
A28 / EXP-1 — the GRU at K=10+1 is NULL; rank-lock is pool-size-
independent.)

## Hedge ladder (final-window candidates)

These don't beat PRIMARY but are eligible for the final-window R7
override probe:

- per-segment Compound × Stint, τ=100k (held)
- ExtraTrees stack-add
- LightGBM-on-kNN stack-add
- **RF-yekenot stack-add** (today; ρ=0.959 most-diverse positively-
  gating base on K=4; +0.26 bp OOF at K=4+1 LR-meta)
- **Path-B K=5 = K=4 + RF-yekenot, τ=100k** (today; OOF 0.95405,
  ρ=0.999917 vs PRIMARY → tie-band; balanced flips 37/31)
- **Path-B K=5, τ=20k** (today; OOF 0.95405, ρ=0.999448; asymmetric
  86/23 flips — R7-style)
- **Path-B K=5, τ=5k** (today; OOF 0.95403, ρ=0.998734; asymmetric
  121/29 flips — R7-style override territory, risky)
- 22-base + d12 LR-meta + per-segment, τ=100k
- DAE-only PRIMARY from Day 15 (the τ=20k variant)
