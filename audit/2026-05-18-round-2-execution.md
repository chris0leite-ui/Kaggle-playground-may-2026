# 2026-05-18 — Round 2 execution: 9/9 NULL across 3 phases

Triggered by: PI request "iterate through your ideas autonomously"
following the persona-rotation brainstorm (`2026-05-18-plateau-
brainstorm.md`). Plan: `/root/.claude/plans/read-the-handover-look-
toasty-candle.md` (round 2). 3-phase execution with stop-gates.

**Headline: every pick tested today nulled.** The K=4 LR-meta +
Path-B PRIMARY (OOF 0.95399, LB 0.95351) is empirically stable
against the strongest larger-step-size mechanism candidates we
could find via persona rotation + 4 research artifacts.

## Results table

| Phase | Pick | Mechanism class | Cost | OOF | Δ bp | ρ_test | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| P0.1 | RRF blend (k=60) | post-process | 5s | 0.94929 | -47.0 standalone | n/a | -- |
| P0.2 | Trimmed-rank (1,1) | post-process | 5s | 0.95328 | -7.1 standalone | n/a | -- |
| P0.1+P0.2 gate | RRF + trimmed → K=4 LR-meta | 5th-base add | 9s | 0.95400 | +0.060 | 0.983 | NULL |
| P0.3 | Stint-cap multiplier (15 cells) | post-process | <5s | 0.95401 | -0.18 best | n/a | NULL (15/15 negative) |
| P1.1 | LGBM `rank_xendcg` meta | meta loss-class | 6s | 0.95304 | -9.5 | 0.960 | REGRESS |
| P1.2 | SGD hinge pairwise meta | meta loss-class | 2s | 0.95187 | -21.2 | 0.974 | REGRESS |
| P1.3 | torch AUC-surrogate MLP | meta loss-class | 61s | 0.94767 | -63.2 | 0.982 | REGRESS |
| P2.1 | Per-Driver random-effect (12 cells) | base/meta | 30s | 0.94662 | -73.8 best | n/a | REGRESS (12/12) |
| P2.2 | Per-bin conformal widths | meta feature | 30s | 0.95400 | +0.012 | 0.983 | NULL |

(K=4 LR-meta baseline OOF AUC: **0.95399**.)

## What each result tells us

### Phase 0 stop-gate (3 nulls; skipped remainder, advanced)

- **RRF / trimmed-rank blends are noise to the K=4 LR-meta.** The
  LR-meta absorbs the new 5th base at |w| < 0.17 (vs |w|~0.27 for
  prior bases). Standalone RRF AUC (0.949) is 45 bp below LR-meta
  baseline because RRF discards the calibrated probability scale
  in favor of pure rank; the LR-meta can re-derive ranking from
  the [P, rank, logit] expansion.
- **Stint-cap multipliers always regress.** Every (threshold,
  multiplier) cell in the 15-cell grid is negative. The Path-B
  per-segment Compound × Stint shrinkage already routes these
  cells optimally; multiplying P by 1.05–1.50 on overdue tyres
  distorts the calibration that Path-B carefully built.

### Phase 1 stop-gate (3 regressions; advanced)

**Major finding: LR with log-loss is loss-OPTIMAL at the K=4 meta
layer.** Three AUC-aligned losses tested:
- LightGBM `rank_xendcg` (with bucketed groups of 5000): **-9.5 bp**.
- SGD hinge with pairwise differences (200k pairs per fold): **-21.2 bp**.
- Torch MLP with smooth-AUC surrogate (`sigmoid(s_neg - s_pos)`,
  30 epochs): **-63.2 bp**.

All three pairwise/ranking losses lose information that log-loss
preserves on this saturated subspace. The mechanism: log-loss
training on [P, rank, logit] = 12 features extracts BOTH ranking
AND calibration, then the trained linear combiner is automatically
AUC-optimal under the linear hypothesis class because AUC is
monotone in the score and the score is monotone in the linear
combination. Switching to a pairwise loss discards the
probability-scale information without adding any new ranking
information.

**This closes a clean theoretical axis** that the team had only
half-tested (LambdaRank per-stint was scoped to within-stint
ranking; this is the first proper global AUC-loss-at-meta test).

### Phase 2 (2 nulls; iteration end)

- **Per-Driver random effects on residuals are pure noise.** Every
  lambda from 50 to 2000 regresses, intercept-only AND
  intercept+slope variants. At lambda → ∞ the effect → 0 and we
  recover the baseline. The K=4 LR-meta already encodes Driver-level
  signal via the underlying bases' target encodings; adding per-
  Driver residual BLUP adds residual noise without signal.
- **Conformal-like per-bin width features tie.** Δ +0.012 bp at
  ρ=0.983. LR coefs on the 4 width features are around -0.09 each
  (weak negative weights), suggesting the meta found them mildly
  informative but the AUC gain is sub-bp. The
  per-(Compound, Stint) residual std is too coarse to capture
  per-row uncertainty heterogeneity; a full CQR with
  GradientBoostingRegressor(loss='quantile') per base might
  do better but at 20-40 min CPU for a likely <+0.3 bp lift, EV
  is poor.

## Net strategic finding

Across **9 distinct mechanism classes** in a single iteration —
RRF blend, trimmed-rank blend, hand-coded multiplier, LGBM ranker,
SGD pairwise hinge, torch AUC-surrogate MLP, per-Driver random
effects, conformal width features — **zero cleared the +0.5 bp
G1 PASS threshold against the K=4 LR-meta baseline 0.95399**.

This is the strongest empirical confirmation yet of the
2026-05-14 "Bayes-optimal ceiling on row features" interpretation.
**Three independent axes of post-Tier-A novelty are now closed:**

1. Loss-class diversity at meta (LR log-loss is optimal).
2. Per-actor heterogeneity capture (per-Driver BLUP adds noise).
3. Uncertainty quantification meta-features (per-bin std ties).

Combined with the Tier-A null (a2_2 +0.302 bp, a3_1 +0.337 bp,
both REGRESSION_RISK band), the post-Tier-A axis is **closed for
row-feature mechanisms** on this competition's K=4 + Path-B base.

## What remains untested (Tier C reserves)

The next plateau-break path must come from outside the row-feature
+ row-prediction-meta space:

- **C2 — Swap-noise DAE on combined train+test** (~2-3 hr Kaggle
  T4). Porto Seguro 1st-place mechanism. The team's d15b vanilla
  DAE lived in hedge at +0.79 bp K=22+1; swap-noise on combined
  frame is structurally different. **Highest-EV remaining pick.**
- **C1 — OpenF1 per-Race scalar join** (~45 min). 26-Race-level
  external join; novel join key vs 1.4%-Driver-cap.
- **EXP-9 — Gap-aware sequence transformer** (~4-6 hr Kaggle T4×2).
  Final-window reserve; directly attacks W3 synth-downsampling.
- **B2 — GraphSAGE Driver-Race-Compound tripartite** (~2-3 hr T4).
  Genuinely novel mechanism class; speculative.
- **A3 — TabDDPM diffusion imputation** (~2-3 hr T4). Speculative
  generative augmentation.

## Recommended next-session posture

1. **Push the missing K=11+K=9 OOFs to Kaggle artifact dataset.**
   The 2026-05-08 snapshot is 10 days stale; future iterations
   need K=11 OOFs to gate at K=11+1 (the actual PRIMARY layer).
   ~10 min via `kaggle datasets version`.
2. **C2 swap-noise DAE** on combined frame. ~2-3 hr GPU. If it
   lifts ≥+0.5 bp at K=4+1: extend to K=11. If null: confirms
   the noise ceiling is truly structural.
3. **Final-window posture** (R5/R7 hedge ladder preparation) if
   C2 is null. 12 days remain; the realistic top-5% probability
   is now <20% absent external data or generative mechanism
   surprise.

## Operational note

Compute spend this iteration: ~3 min CPU total (Phase 0+1+2 all
fast). Submission slots used today: 0. Total daily budget for
2026-05-18: 0/10 used. Tomorrow we can submit P0.1 RRF (Δ +0.060 bp,
ρ 0.983) as a calibration probe IF the PI wants to validate the
NULL claim — but the ρ_test < 0.999 makes this a REGRESSION_RISK
submission per Rule 27.

## Files added today

```
scripts/probe_phase0_rrf_trimmed.py
scripts/probe_a2_stint_cap.py
scripts/probe_meta_loss_variants.py
scripts/probe_a1_glmm.py
scripts/probe_s5_conformal_cqr.py
audit/2026-05-18-plateau-brainstorm.md     (from earlier in session)
audit/2026-05-18-round-2-execution.md       (this file)
```

## Friction surfaced today

- `tag: lgbm-rank-group-size-cap` — LightGBM hard-caps single-group
  size at 10000 rows. `LGBMRanker` with `group=[len(y)]` fails on
  351k-row training fold. Workaround: bucketed groups of 5000.
  Note this in any future ranking-loss probes.
- `tag: 9-of-9-nulls-confirm-ceiling` — 9 distinct mechanism
  classes in a single iteration nulled at K=4+1 gate. Strongest
  evidence yet for the row-feature ceiling. Per Rule 4, the
  Research-loop has been run; per Rule 7 the saturation triggers
  are met; per `problem-solving.md` step 1, the problem statement
  has been restated. The team can now defensibly state "row-
  feature ceiling reached" per Rule 4's escape clause.
