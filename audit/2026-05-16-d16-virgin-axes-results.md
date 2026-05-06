# 2026-05-16 — Day-16 virgin-axes complement results

> Trigger: PI re-entry of Conn-McLean step 1 on `claude/read-handover-lA8Nr`
> after Day-15 PM advance to LB 0.95059. HANDOVER T1-T4 owned by other
> branches; this branch tackles the orthogonal axes from the d13
> problem-decomposition tree (α/β/δ/ε/ζ/η) that no branch executed.

## Plan recap

Issue-tree leaf 7 (in `ISSUES.md`) covers nine candidates. Tonight's
execution prioritised cheap probes first, GPU long-tail kicked off in
parallel.

## Results table (Strat OOF, vs new PRIMARY 0.95090)

| Probe | Axis | Std OOF | ρ vs PRIMARY | Verdict | Wall |
|---|---|---:|---:|---|---:|
| H4 Year=2023 ∩ rare-Driver mask | η1 | n/a (post-process) | n/a | **NULL** +0.004 bp ceiling at K=5 | 5s |
| H7 Conformal isotonic 4 schemes | δ2/3 | n/a (post-process) | n/a | **NULL** all 4 schemes -2.5 to -9.6 bp | 8s |
| H10 Two-stage stint (E[T] → soft) | α5 | 0.625 | 0.196 | **NULL** stage-2 logistic too restrictive | 107s |
| H2 Twin parallel-pool 2-meta blend | ε2 | 0.95010 | 0.991 | **FALSIFIED** Δ -1.79 bp vs single LR-meta(K=11) | 170s |
| ε4 DeepGBM (leaf-encoding 2-stage) | ε4 | TBD | TBD | TBD | TBD |
| H9 Transductive full-test pseudo | ζ6 | TBD | TBD | TBD | TBD |
| H11 AV-sample-weight LGBM | ε  | TBD | TBD | TBD | TBD |
| H1 GRU sequence model (Kaggle T4x2) | α4 | TBD | TBD | TBD | TBD (running) |

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

## In-flight probes (continue overnight)

ε4 (DeepGBM leaf-encoding) — Stage-1 done in 80s, Stage-2 LGBM on 627
leaf-categorical features running.
H9 (transductive pseudo) — fold 0/5 of LGBM on 627k rows
(synth_train + half-weighted PRIMARY-pseudo-test).
H11 (AV-sample-weight) — AV classifier + base LGBM with weights; just started.
H1 GRU sequence — Kaggle T4x2 RUNNING; ~3-4h projected wall.

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
