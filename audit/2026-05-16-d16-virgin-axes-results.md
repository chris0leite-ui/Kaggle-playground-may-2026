# 2026-05-16 — Day-16 virgin-axes complement results

> Trigger: PI re-entry of Conn-McLean step 1 on `claude/read-handover-lA8Nr`
> after Day-15 PM advance to LB 0.95059. HANDOVER T1-T4 owned by other
> branches; this branch tackles the orthogonal axes from the d13
> problem-decomposition tree (α/β/δ/ε/ζ/η) that no branch executed.

## Plan recap

Issue-tree leaf 7 (in `ISSUES.md`) covers nine candidates. Tonight's
execution prioritised cheap probes first, GPU long-tail kicked off in
parallel.

## Results table (Strat OOF, vs new PRIMARY 0.95090; Day-16 final)

| Probe | Axis | Std OOF | ρ vs PR | min-meta Δ vs LR-meta(K=22) | Verdict | Wall |
|---|---|---:|---:|---:|---|---:|
| H4 Year=2023 ∩ rare-Driver mask | η1 | n/a (post-process) | n/a | n/a | **NULL** +0.004 bp ceiling | 5s |
| H7 Conformal isotonic 4 schemes | δ2/3 | n/a (post-process) | n/a | n/a | **NULL** -2.5 to -9.6 bp | 8s |
| H10 Two-stage stint (logistic) | α5 | 0.625 | 0.196 | (skipped, weak base) | **NULL** stage-2 too restrictive | 107s |
| H2 Twin parallel-pool 2-meta | ε2 | 0.95010 | 0.991 | (vs LR-meta-K=11: -1.79 bp) | **FALSIFIED** | 170s |
| ε4 cat-LGBM stage-2 (627 cats) | ε4 | — | — | — | **KILLED** at 16 min (over-engineered) | 16m |
| ε4b sparse-LR head | ε4 | 0.92507 fold-0 only | — | — | **KILLED** weak fold-0 (sparse-LR ~20 min/fold) | 30m |
| H11 AV-sample-weight | ε | — | — | — | **KILLED** AV stuck under contention 12 min | 12m |
| H9 Transductive full-test pseudo | ζ6 | 0.93433 | 0.872 | **+0.631 bp PASS** at LR-meta-K22 | **MARGINAL** Δ -0.30 vs PRIMARY hier | 30m |
| H1 GRU sequence (Kaggle T4×2) | α4 | 0.93066 | **0.919** | **-0.043 bp NULL** | **NULL** at meta gate | 58m kernel |
| H9 + H2 multi-add | ε2+ζ6 | — | 0.9953 | **+0.671 bp** (≈ H9 alone) | NULL marginal | 4m |
| H9 + GRU multi-add | α4+ζ6 | — | 0.9955 | **+0.629 bp** (≈ H9 alone) | NULL marginal | 4m |

**No candidate beats PRIMARY hier-meta (0.95090).** H9 is the only +signal at the LR-meta(K=22) baseline level, but PRIMARY's Path-B Compound×Stint hier-meta sits +0.93 bp above LR-meta(K=22) — the +0.63 bp H9 gain at LR-meta gets erased once you account for PRIMARY's hier-meta amp.

## Load-bearing findings

### F1. PRIMARY hier-meta is globally well-calibrated

H4 (Year=2023 hard-mask) and H7 (per-bin isotonic, inner-CV-validated,
4 schemes) BOTH land at noise-floor or regress. PRIMARY has fully
absorbed the (Year, Compound, Stint, RaceProgress) cohort calibration
through its hier-meta segmentation. Post-processing recalibration is
NOT the binding axis. Reconfirms `posthoc-isotonic-overfits-OOF`
extension: even when inner-CV-corrected against posthoc overfit, the
isotonic recalibration provides no lift if PRIMARY's predictions are
already calibrated within the bin.

### F2. Twin parallel-pool 2-meta hierarchy LOSES information vs single LR-meta

H2 built two LR metas (Pool A = 6 GBDTs / Pool B = 5 model-class diverse
including FM/rule/DAE), ρ(metaA, metaB) = 0.967 (real disagreement),
top-level LR over [metaA, metaB] OOF = 0.95010, single LR-meta
[A∪B, K=11] OOF = 0.95028. **Δ = -1.79 bp.** Top-level LR over
2-feature [metaA, metaB] collapses the rank info that the 11-dim LR
captures with [raw, rank, logit] expand. Reconfirms friction
`lr-meta-rank-lock-strong-anchor` from a different angle:
rank-lock isn't broken by hierarchical meta-stacking either.

New friction tag: `twin-pool-2-meta-collapses-rank-info`.

### F3. Two-stage stint reformulation needs richer stage-2 (H10 methodological miss)

α5 axis (per-stint two-stage) was implemented with stage-1 LGBM
regression on E[T_stint] + stage-2 1-D logistic on remaining-laps.
Std AUC 0.625 — far below the bar. The 1-D logistic threw away
joint structure (laps_so_far, current TyreLife, Compound). A future
retry should use Stage-2 LGBM with [E[T_stint], laps_so_far, Compound,
TyreLife, etc.] features.

## Friction tags to log

- `twin-pool-2-meta-collapses-rank-info` — H2 falsified.
- `primary-hier-meta-globally-calibrated` — H4/H7 NULL on post-process axes.
- `two-stage-stint-needs-richer-stage-2` — H10 methodological retry candidate.

## Final synthesis (after all probes complete)

### Load-bearing finding F4 — α4 temporal axis is rank-locked at K=22

GRU sequence model on (Driver, Race) lap windows achieved std OOF
0.93066 with ρ=0.919 vs PRIMARY (most-diverse single base of session,
genuinely orthogonal at the prediction-unit level α4). At K=22+1
LR-meta gate: **Δ = -0.043 bp NULL.** Even the prediction-unit
reframing (causal GRU consuming sequence context — the unique virgin
axis from d13 problem-decomposition tree) is fully absorbed by the
K=22 LR-meta with [raw, rank, logit] expand. The temporal axis
signal that the GRU captures from the lap-sequence is recoverable as
a convex combination of K=22 base predictions.

This is the 5th cross-confirmation of friction
`lr-meta-rank-lock-strong-anchor`: no single-base addition that's
NOT a direct meta-derivative (which fails for orthogonality reasons)
nor a meta-arch redesign (Path-B segmentation) survives the K=22
rank-lock at meta gate.

### Load-bearing finding F5 — H9 transductive pseudo lifts LR-meta(K=22) but NOT PRIMARY hier-meta

H9 std OOF 0.93433, ρ=0.872. K=22+1 LR-meta gate: **Δ +0.631 bp
PASS** vs LR-meta(K=22) — small but real positive signal. Meta gives
H9 a NEGATIVE-direction routing weight (raw -0.295, logit -0.222),
i.e., H9 is used as "what NOT to predict" in subset of feature space.

BUT vs PRIMARY hier-meta: **Δ -0.30 bp regress.** PRIMARY's hier-meta
with Compound×Stint segmentation (τ=20k) sits +0.93 bp above
LR-meta(K=22). The H9 +0.63 bp gain at LR-meta level is erased once
you account for PRIMARY's amp. This is the empirical confirmation
of friction `path-b-amp-only-fires-on-meta-arch-not-base-add` —
even a marginal LR-meta lift doesn't transfer to LB if the base-add
mechanism doesn't carry the per-segment routing structure that
hier-meta exploits.

### Multi-add does NOT add over single-add

H9 + H2: +0.671 bp (vs +0.631 bp H9 alone) → +0.04 bp marginal.
H9 + GRU: +0.629 bp (vs +0.631 bp H9 alone) → effectively 0.
H2 and GRU contribute the same routing direction as H9 — orthogonal
diversity at standalone level (ρ 0.872, 0.991, 0.919) collapses to
the same single direction at meta level.

### Strategic implication

The K=22 + Path-B-hier-meta architecture is rank-saturated against
every base-add axis (per-row FE, calibration, prediction-unit/α,
loss-axis/β, output-norm/δ, pool-architecture/ε) reachable via
standalone OOF computation. The only remaining amp-eligible axis is
**meta-architecture redesign** (HANDOVER T4: non-Gaussian shrinkage,
nested hierarchy, Yao/Vehtari covariance-modelled BMA, alternative
segmentation crosses) — owned by other branches.

## Pointers

- `scripts/d16_h4_year_mask.py` + results
- `scripts/d16_h7_conformal_calibrate.py` + results
- `scripts/d16_h10_two_stage_stint.py` + results
- `scripts/d16_h2_twin_pool_meta.py` + results
- `scripts/d16_epsilon4_deepgbm.py` (running)
- `scripts/d16_h9_transductive_pseudo.py` (running)
- `scripts/d16_h11_adv_weight_lgbm.py` (running)
- `scripts/d16_multi_min_meta.py` — K=22+N stack-add gate
- `kernels/d16-gru-sequence-gpu/` — Kaggle T4x2 GRU kernel
- `ISSUES.md` leaf 7 (a-i) — claimed by `read-handover-lA8Nr`
