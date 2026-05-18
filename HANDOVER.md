# HANDOVER

Next-session brief. **PI says "handover"** → agent reads this file
and proceeds. **PI says "prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

This file is rewritten (not tail-appended) every wrap-up. Prior
versions: `audit/archive-YYYY-MM-DD-handover-*.md`.

---

## Where we are

**NEW PRIMARY: R7.1 K=13 + Path-B DriverClass × Stint τ=100k.
LB 0.95389.** (Round 7 result.) +0.02 bp over prior PRIMARY (R5.2
LB 0.95387). Top-5% gap −1.6 bp; leader gap −8.7 bp. File:
`submissions/submission_K13_pathb_driverclass_stint_tau100000.csv`.

Submissions: **49 / 270** total; **7 used 2026-05-18**. Comp-day
**18 of 31**; days remaining **13**.

**Round 7 finding**: DriverClass × Stint segmentation (named-vs-D0XX
× Stint = 12 segments) on Path-B beats default Compound × Stint by
+0.106 bp at OOF, +0.02 bp at LB. The named-driver pit-rate
differential (32-43% vs 16-22% for D0XX) is captured by this
segmentation; Compound × Stint missed it.

**Round 7 negative finding**: swap-noise DAE absorbs at meta for
EVERY Path-B segmentation tested (-0.09 to -0.15 bp). Standalone
OOF 0.94665 was decent but extracted no marginal value over K=13.

**Round 5 plateau break**: K=11 (rebuilt slim-kNN) + r4_segment_fe
(row-class) + r4_hmm_seq (sequence-class) under Path-B Compound×Stint
τ=100k operator. The R4 mechanism-orthogonality finding survives at
the REAL K=11 anchor (+0.245 bp at LR-meta OOF), and the **Path-B
operator preserved +5 bp of LB transfer** vs LR-meta at the same OOF.

Today's 7 submissions:
| ref | LB | mechanism |
|---|---|---|
| 52772090 (R4) | 0.95354 | K=4 + seg + HMM LR-meta (R4 probe) |
| 52773963 (R5.1) | 0.95382 | K=11 + seg + HMM LR-meta |
| 52774385 (R5.2) | 0.95387 | K=13 + Path-B Compound×Stint τ=100k |
| 52774692 (R5.3) | 0.95385 | 70/30 rank-blend R5.2 + K=27+Path-B |
| 52776849 (R6.1) | 0.95387 | K=13+Path-B 5-seed fold-fit bag (ties R5.2) |
| **52778581 (R7.1)** | **0.95389** | **K=13+Path-B DriverClass×Stint τ=100k** ← PRIMARY |
| 52779240 (R7.2) | 0.95389 | R7.1 + 5-seed fold-fit bag (ties R7.1; hedge) |

Submissions: 49 / 270; 7 used 2026-05-18; 3 remaining today.

## Round 7 — multi-segmentation Path-B + swap-noise DAE (LATEST)

PI: "go". Round 7 plan executed Phases A-D in parallel where possible.

**Phase A — Swap-noise DAE (Kaggle T4, ~50 min)**: Porto Seguro
3-layer MLP encoder [23→256→256→128] + 15% swap-noise + MSE
reconstruction on train+test. Standalone OOF **0.94665** (stronger
than transformer v1 0.91974 and v2 0.93330). At K=14+Path-B for
every segmentation tested: Δ −0.09 to −0.15 bp (REGRESS). DAE
absorbs at meta. Embedding-class diversity didn't help K=11 pool.

**Phase B — Multi-segmentation Path-B sweep (~6 min CPU)**: 3
segmentations tested:
- Year × Compound (20 seg): Δ −0.149 bp NULL
- **DriverClass × Stint (12 seg)**: Δ **+0.106 bp** WIN
- Compound × Stint × LapBucket (120 seg): Δ +0.065 bp marginal
τ sweep on winner: τ=100k optimal (+0.106), τ=20k regresses, τ=500k
marginal.

**Phase D — Cross-pollination (R6 fold-bag × R7 segmentation)**:
5-seed fold-fit bag of K=13+Path-B DriverClass×Stint. OOF **0.95450**
(+0.264 bp over R7.1 single-seed; largest OOF improvement of session).
ρ vs R7.1 = 0.999973 → TIE_ZONE. R7.2 LB tied R7.1 at 0.95389;
private LB may show the lift.

**Submissions** (2 of today's 4 remaining slots used in R7):
- R7.1: K=13+Path-B DriverClass×Stint τ=100k → **LB 0.95389** (PRIMARY)
- R7.2: R7.1 + 5-seed fold-fit bag → LB 0.95389 (ties; hedge)

## Round 6 — operator-axis retest + fold-fit bagging + transformer v2

PI: "go" / "iterate, follow what skill says, think deeper."

**Phase A — Operator-axis retest at K=11+Path-B**: re-gated 5 prior
LR-meta nulls (conformal_widths, rrf_k60, meta_lgbm_rank, trimmed_rank,
seg_fe_v2). **5/5 NULL under Path-B too** (Δ −0.026 to −0.090 bp).
The R5 +5 bp Path-B-vs-LR-meta swing was **pool-composition-specific**
(seg+HMM × Path-B segment-shrinkage interaction), NOT a general
operator advantage.

**Phase B — Proper fold-fit bagging**: rewrote `run_pathb` test-prediction
path to bag per-fold per-seed (5 seeds × 5 folds = 25 fits averaged).
Bag predictions DIFFER from single-seed (ρ=0.999988 vs ρ=1.0 in R5's
broken full-train bag). OOF 0.95448 (+0.212 bp). **LB 0.95387 — ties
R5.2 within 5-decimal quantization (TIE_ZONE prediction confirmed).**
Variance reduction is real but quantized away at LB precision.

**Phase C — Transformer v2 (Kaggle T4)**: D=256, 6 layers, 15 epochs,
GroupKFold by sequence (was Stratified per row in v1). Standalone
OOF **0.93330 (+13.5 bp vs v1's 0.91974)** despite the harder split.
At K=14+Path-B: Δ −0.014 bp (absorbed at meta). Still 21 bp below
K=11 baseline; doesn't reach meta-utility threshold.

**Phase D — Multi-class combos**: K=14 fold-fit bag (R5.2 pool +
TRFv2 + bagged) = OOF 0.95448, same as Phase B alone. Transformer
adds nothing on top under Path-B.

## Round 5 — multi-class + Path-B operator + slim-kNN rebuild

## Round 5 — multi-class + Path-B operator + slim-kNN rebuild (LATEST)

PI directed "iterate, get on top of leaderboard." Round 5 plan
(`/root/.claude/plans/read-the-handover-look-toasty-candle.md`)
executed Phases A through F.

**Phase A** — Rebuilt 6 missing slim-kNN bases (qAT, qAV, qAO, qAA,
qAF, qAK). Required pulling `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`
from Kaggle (the slim-kNN builders reference `data/original/f1_strategy_dataset_v4.csv`
which was missing from local snapshot). K=11 LR-meta plain OOF = 0.95443
— matches historical PRIMARY exactly. Rebuild verified.

**Phase B** — Re-gated R4 mechanism-orthogonality at REAL K=11+1:
- K=11+seg+HMM: Δ +0.245 bp OOF, R5.1 submission **LB 0.95382**
- Anchor-attenuation pattern confirmed: 0.542 @ K=4 → 0.275 @ K=5 → 0.245 @ K=11.

**Phase C** — `probe_r5_graph_pit_pressure.py`: 4 per-(Race, Lap)
pit-pressure features (graph-class). Standalone OOF 0.93344;
at K=11 LR-meta: -0.012 bp alone, marginally regresses the seg+HMM
combination.

**Phase D super-stack sweep** — 15 combinations of {seg, segv2, HMM,
graph, TRF} at K=11+N LR-meta. Best: seg+HMM at Δ +0.245 bp; the
4-way (seg+HMM+graph+TRF) at +0.254 bp (negligible diff).

**Phase D + Path-B operator — THE BREAKTHROUGH**: applied Path-B
Compound × Stint τ=100k to K=13 = K=11+seg+HMM pool. **Same OOF
(0.95446 vs 0.95445 LR-meta), but LB transferred 5 bp better**:
- K=11+seg+HMM LR-meta: LB 0.95382 (-6.3 bp from OOF)
- **K=13+Path-B τ=100k: LB 0.95387** (-5.9 bp from OOF) ← new PRIMARY

**Phase F** — Kaggle T4 gap-aware transformer (4-layer, D=128).
Failed twice (data path resolution + P100/sm_60 PyTorch incompat);
fixed via rglob + `enable_internet: true` for torch 2.4 reinstall.
Standalone OOF 0.91974 (weak; absorbed at K=11+Path-B). Saved for
next-session retry with larger architecture + GroupKFold split.

**5-seed bagging attempt**: Path-B's full-train fit is seed-invariant
for test predictions (ρ=1.0 vs single-seed). The "bag" produces
identical test predictions — bag doesn't help unless we change to
fold-fit averaging. Skipped submission.

**Top-5% gap**: 0.95405 − 0.95387 = -1.8 bp. **Leader gap**:
0.95476 − 0.95387 = -8.9 bp (structurally hard).

## 2026-05-18 session — FIVE rounds; PRIMARY beaten

### Round 4 — Rule 23 free-form-FE + mechanism-orthogonal stacking (LATEST)

PI re-entered `problem-solving.md` step 1 after Round 3 closed.
Three Explore agents (state map, skill prescription, OOF failure
analysis) produced a concrete target list:
- Weak segments: WET+S1 (AUC 0.81), INTER+S2 (AUC 0.86), VET-driver
- Unused interactions: Cumulative_Degradation × Compound (gap 14.5),
  Position_Change × Driver-class (gap 1.86 for named drivers)

**Phase A** — `probe_r4_segment_fe.py` (9 interaction features added
to LightGBM): standalone OOF 0.94878. K=4+1: Δ +0.263 bp (strongest
in 15+ rounds, but G2-fails 0.30). Variant v2 (drop WET, add tire-life)
+0.211 bp. Combined v1+v2 +0.282 bp. ρ vs K=4+Path-B = 0.999183.

**Phase D — super-model attempt: HMM sequence model** —
`probe_r4_hmm_seq.py` (Gaussian HMM with K=8 hidden states over
per-(Year, Race, Driver) sequences of Compound+TyreLife+RaceProgress
+Stint+Position_Change+Cumulative_Degradation; 8-dim posterior + entropy
as features for downstream LightGBM). HMM standalone alone: K=4+1
Δ −0.005 bp (null). **But the 2-base combination (seg_fe + HMM) at
K=4+1: Δ +0.542 bp** — first G2 PASS of Round 4 after 18 nulls.

**Mechanism-orthogonality finding**: the row-class FE and the
sequence-class HMM provide CORRECTIONS IN OPPOSITE DIRECTIONS at the
LR-meta layer (logit-column coefs: seg_fe +0.200, HMM −0.127). This
is true cross-mechanism diversity, not redundant base addition.

**Anchor progression** (Round-3 absorption pattern confirms attenuation
not absorption to zero):
- K=4+1: Δ +0.542 bp ← G2 PASS
- K=5 (K=4 + K=27 super-base)+1: Δ +0.275 bp ← marginal
- K=21+1: Δ +1.449 bp (inflated, weak-anchor)

**LB calibration probe submitted: LB 0.95354** (OOF→LB drop 5.1 bp).

**Implication for next session**: at REAL K=11+K=9 PRIMARY (richer than
K=27 super-base), expected lift is +0.0 to +0.3 bp → LB 0.9539-0.9542,
potentially within top-5% boundary 0.95405.

### Round 1-3 — original ceiling-validation work (UNCHANGED)

Three execution rounds today, all in compliance with `loops.md`
plateau triggers and `problem-solving.md` step-1 re-entry.

### Round 1 — Original Tier-A FE batch + Research-loop

Per the 2026-05-14 plateau brief. Result: 4 picks tested, all
nulled at K=4+1 gate (a2_2 +0.302 bp, a3_1 +0.337 bp, C3
falsified, C4 smoke-fail). Research-loop synthesis at
`audit/research/2026-05-18-research.md`.

### Round 2 — Persona rotation + larger-step-size brainstorm

10 Wild Options + Junior ML Engineer Opus personas. 9 fresh
picks across 3 phases. All NULL/regress at K=4+1. Major finding:
**LR with log-loss is loss-OPTIMAL at K=4 meta layer.** Three
AUC-aligned loss variants (LightGBM rank_xendcg, SGD hinge,
torch surrogate) all worse than LR. Closes a clean axis.

### Round 3 — Senior ML Engineer pressure-test + Caruana

Senior ML reviewer surfaced a load-bearing concern: 11/11 nulls
were gated against K=4 proxy (LB 0.95351), not actual PRIMARY
K=11+K=9 (LB 0.95386). The 3.5 bp gap could mean anchor-conditional
nulls. **Phase 0 multi-anchor retest REFUTED this hypothesis:**

- K=4 vs K=27 residual Pearson correlation: **0.998**.
- The 3 most-plausible Round-2 nulls retested at K=4+1, K=21+1,
  and K=4+K27super+1 anchors. K=21 shows artificial +14 to +34
  bp because its base AUC is 0.95073 (far below K=4's 0.95399);
  K=4+K27super (the strongest available anchor at 0.95429 OOF)
  is null on all 3.
- LR-meta C-sweep across 0.01-100 confirms LGBM-rank null is
  C-robust (Rule 21 satisfied).

**Verdict: row-feature ceiling claim VALIDATED.** Both K=4 and
K=27 anchors agree at the residual level.

Caruana hill-climb on 11 available LB/hedge OOFs plateaus at
OOF 0.95433 with 88% weight on K=27+Path-B alone. +0.11 bp tie
vs best-single. The snapshot has no diversity beyond K=27 to
exploit.

## Cumulative tally for 2026-05-18

**Round 1-3: 15 distinct mechanism classes tested; all NULL/regress.**
Round 4: 4 single-mechanism nulls + 1 **G2-PASSING combination**.

Round 1 (4): a2_2 mandatory-compound, a3_1 rank-sorted-gaps,
C3 per-(R,L) shrinkage, C4 UID magic.

Round 2 (9): RRF blend, trimmed-rank blend, stint-cap multiplier,
LGBM rank_xendcg meta, SGD hinge pairwise meta, torch AUC
surrogate, per-Driver random-intercept BLUP, per-Driver random-
slope BLUP, per-bin conformal widths.

Round 3 (2): multi-anchor retest of 3 candidates (null at K=4
AND K=4+K27super); Caruana hill-climb (degenerates to K=27).

Round 4 single-mechanism (4): r4_segment_fe v1 +0.263 bp G2-fail,
v2 +0.211 bp G2-fail, v1+v2 stack +0.282 bp G2-fail, HMM alone
−0.005 bp null.

**Round 4 plateau-break (1)**: r4_segment_fe + r4_hmm_seq 2-base add
at K=4+1 = **Δ +0.542 bp G2 PASS**. LB-submitted, scored 0.95354.

## Critical operational gap

**The 2026-05-08 artifact snapshot is missing the 6 slim-kNN
bases that constitute the K=11 diversity layer.** Without these:

- We cannot reconstruct K=11 LR-meta locally.
- New mechanisms are gated at K=4+1 (3.5 bp behind PRIMARY) or
  K=27+1 (1.8 bp behind), not at the actual PRIMARY K=11+1.
- Caruana blending caps at K=27 LB 0.95368, not PRIMARY LB
  0.95386.

The 6 missing bases (per `scripts/build_K11_full_pathb.py:151-156`):
- dgp_v3_qAT_K1
- dgp_v3_qAV_K1_7feat
- dgp_v3_qAO_knn_multi
- dgp_v3_qAA_stint_imputed
- dgp_v3_qAF_d16plus
- dgp_v3_qAK_knn3

Each is built by an upstream dgp_v3 family script (likely a
fixed-feature-subset kNN; ~30-60 min CPU per base; total ~3-6 hr).

## Strategic posture for remaining 13 days

**(Per Round-3 Headroom-Math agent and confirmed by 15-of-15 null
result):**

- Remaining queue midpoint lift sum: 2.80 bp; 50% additivity
  discount: 1.40 bp. Top-5% gap 1.9 bp.
- **P(reach top-5%) ≈ 20-25% via queue alone**, conditional on
  proper K=11 gating becoming available.
- **P(catch leader 9 bp) ≈ 1-3%.** Structurally hard.

Recommended posture: **(a) Aggressive lift-seeking with discipline**
(per headroom-math agent), but operationally gated on rebuilding
kNN diversity first.

## Next-session first actions (priority order)

PRIMARY is R7.1 (LB 0.95389) — top-5% boundary still 1.6 bp away.
R7 closed two more axes (DAE absorbs; multi-tau on winner already
optimal at 100k). Remaining cheap-EV is segmentation variants.

1. **More Path-B segmentations** (~30 min/segmentation CPU). The
   DriverClass × Stint win (+0.106 bp OOF, +0.02 bp LB) opens
   the door for more discrete-axis segmentations:
   - Driver-tier × Stint (top-quartile / middle-half / bottom-quartile
     by pit-rate × 6 stints = 18 segments)
   - Race-cluster × Stint (high-pit-rate races vs low × stint)
   - Year × Stint (4 × 6 = 24 segments)
   - Compound × first-pit-window (5 × 4 buckets)
   **P ≈ 25% one segmentation lifts ≥ +0.05 bp at LB.**
2. **Multi-segmentation Path-B ensembling**: rank-blend output of
   3+ Path-B variants (Compound×Stint + DriverClass×Stint + new).
   Each captures different sub-population variance; blending should
   stack. **P ≈ 30% at +0.05-0.10 bp.**
3. **C1 OpenF1 per-Race scalar join** (~45 min CPU). 1.4% match
   cap; not yet tried. **P ≈ 15% at +0.1-0.2 bp.**
4. **DAE v2 architecture**: deeper bottleneck (64 dim), masked-
   column pretraining (BERT-style), contrastive loss. ~3 hr Kaggle T4.
   v1 absorbed at meta; v2 with stronger embedding signal might
   cross the threshold. **P ≈ 20%.**
5. **Public-notebook scan** (Rule 22; not done in 17 days). Check
   if a top-kernel insight has emerged.
6. **Submit R7.2 combo bag** during final-window R7d period — the
   +0.264 bp OOF improvement may register on private LB.

## Round 8 — hedge ladder for final-window R7d (Days 28-31)

- **Final-1**: R7.1 K=13+Path-B DriverClass×Stint (LB 0.95389) — PRIMARY
- **Final-2**: R7.2 K=13+Path-B DC×S 5-seed fold-fit bag (LB 0.95389,
  ties; structurally distinct = 5-seed averaged; private-LB hedge)
- **Final-3 backup**: K=27+Path-B τ=100k (LB 0.95368) — different
  operator-pool composition

## Round 7 — hedge ladder for final-window R7d (Days 28-31)

- **Final-1**: R5.2 K=13+Path-B (LB 0.95387) — current PRIMARY.
- **Final-2**: R6.1 K=13+Path-B fold-fit bag (LB 0.95387) —
  structurally-distinct (5-seed averaged) tied LB; private-LB
  variance hedge.
- **Final-3 backup**: K=27+Path-B τ=100k (LB 0.95368) — different
  operator-pool composition, no seg+HMM augmentation.

## 🔴 Critical: held submissions — DO NOT submit

Day-17 strict fold-safe audit collapsed all target-reformulation
single-add results 88-100% (Rule 24 origin). Files still on disk
but invalidated:

- `path_b_K22_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K25_megapool_tau{5k,20k,100k}.csv`
- `path_b_multilevel_τ_*.csv`

Origin: `audit/2026-05-06-target-reform-leakage-audit.md`.

## Read order on session start

1. `CLAUDE.md` — rules + pointers.
2. `state/current.md` — current PRIMARY, LB ladder.
3. **THIS FILE** — held-submissions warning + today's 15-of-15
   null + snapshot-gap operational fix.
4. `audit/2026-05-18-round-3-execution.md` — Round 3 ceiling
   validation + Caruana null + kNN-snapshot diagnosis.
5. `audit/2026-05-18-round-2-execution.md` — Round 2 9-of-9
   null breakdown (loss-class diversity closed).
6. `audit/2026-05-18-plateau-brainstorm.md` — persona output
   from rounds 1+2.
7. `audit/2026-05-18-tier-a-batch.md` — Round 1 Tier-A null.
8. `audit/research/2026-05-18-research.md` — Research-loop
   synthesis (C1/C2 still pending).

## Operational fixes (highest-priority)

- **Bootstrap KGAT_ token handling.** `bootstrap.sh` should detect
  a KGAT_-prefixed `KAGGLE_API_TOKEN` and explicitly UNSET
  `KAGGLE_USERNAME` and `KAGGLE_KEY` before invoking the Kaggle
  CLI. Setting all three triggers 403 on private datasets.
- **Push K=11+K=9 OOFs to Kaggle artifact dataset.** Snapshot is
  2026-05-08; PRIMARY built 2026-05-09 to 2026-05-12. Stale by
  10 days; blocks all K=11-gating today.
- **LightGBM `LGBMRanker` group-size cap.** Hard limit 10000 rows
  per group; future ranking-loss probes need bucketed groups
  (5000 is safe).

## Empirical findings to preserve

- **K=4 vs K=27 residual Pearson correlation: 0.998.** Confirms
  the row-feature subspace is locked at OOF ~0.95430 *for single-
  mechanism row-features*. Future challenger mechanisms must be
  tested against BOTH anchors.
- **LR with log-loss is C-robust loss-optimal at K=4 meta.**
  Sweep across 4 orders of magnitude (C ∈ {0.01, 100}) shows
  the LGBM-rank candidate is null at every C. Closes the loss-
  class diversity axis at the K=4 meta layer.
- **Per-Driver random-effect BLUP adds noise on synth.** Every
  lambda from 50-2000 regresses; closes the per-actor heterogeneity
  axis.
- **2026-05-18 (Round 4) — mechanism-orthogonal stacking breaks
  the single-class ceiling.** Row-class FE (segment-targeted
  interactions) + sequence-class HMM (state-posteriors over
  per-Driver lap trajectories) combined as 2-base K=4 stack-add
  gives Δ +0.542 bp at K=4+1 with G2 PASS, where each alone is
  G2-fail (segment_fe +0.263, HMM −0.005). LR-meta logit-column
  coefficients point in OPPOSITE directions (seg +0.200, HMM
  −0.127): genuine mechanism orthogonality, not redundant base
  addition. Implication: future plateau breaks should test CROSS-
  CLASS combinations, not single-class refinement.
- **2026-05-18 (Round 5) — Path-B operator vs LR-meta operator
  has +5 bp LB transfer advantage at K=11 pool.** Same OOF
  (0.95446 vs 0.95445), but LB 0.95387 (Path-B) vs 0.95382
  (LR-meta). The per-segment shrinkage operator preserves the
  mechanism-orthogonality signal where the global LR meta absorbs
  it. **Implication: every new mechanism class should be tested
  under Path-B operator, not just LR-meta, when comparing to
  PRIMARY**.
- **2026-05-18 (Round 5) — Path-B's full-train fit is seed-invariant
  for test predictions.** Multi-seed bagging via the existing
  `run_pathb` function only changes OOF (which uses fold-dependent
  Stratified splits); the test predictions are identical across
  seeds because they come from a single FULL-TRAIN fit. For true
  bagging, switch to fold-fit averaging (skip the full-train fit
  for test) or vary the base OOFs themselves.
- **2026-05-18 (Round 6) — The R5 +5 bp Path-B-vs-LR-meta swing
  is pool-composition-specific, NOT a general operator advantage.**
  5 prior LR-meta nulls (conformal_widths, rrf_k60, meta_lgbm_rank,
  trimmed_rank, seg_fe_v2) ALL nulled at K=11+Path-B too (Δ −0.026
  to −0.090 bp). The R5 lift was specific to the seg+HMM mechanism-
  orthogonality × Path-B segment-shrinkage interaction. Generalizing
  "Path-B always preserves more signal than LR-meta" is FALSE.
- **2026-05-18 (Round 6) — Fold-fit bagging works mechanically but
  is LB-quantized at this precision.** `scripts/build_K13_seghmm_pathb_foldbag.py`
  averages per-fold per-seed test predictions across 5 seeds × 5
  folds = 25 fits. Bag predictions DIFFER from single-seed (ρ=0.999988
  vs ρ=1.0 in R5's broken full-train bag). OOF +0.212 bp; LB tied
  at 0.95387 (5-decimal quantization). May register on private LB.
- **2026-05-18 (Round 6) — Transformer v2 with GroupKFold +
  larger arch (D=256, 6 layers) improved standalone OOF by +13.5
  bp over v1 (0.91974 → 0.93330) despite structurally harder
  fold split.** Still absorbs at K=14+Path-B meta (Δ −0.014 bp);
  not enough to overcome the 21 bp standalone gap to K=11 baseline.
  v3 with even larger arch + pretraining might cross the threshold.
- **2026-05-18 (Round 7) — Path-B SEGMENTATION choice is a new
  lift axis.** DriverClass × Stint (named-vs-D0XX × 6 = 12 segments)
  beats default Compound × Stint by +0.106 bp OOF / +0.02 bp LB.
  The named-driver pit-rate differential (32-43% vs 16-22%) is
  captured by driver-class segmentation; Compound × Stint missed
  it. Other untried segmentations may yield more lift.
- **2026-05-18 (Round 7) — Swap-noise DAE (Porto Seguro recipe)
  absorbs at K=14+Path-B for every segmentation tested.**
  Standalone OOF 0.94665 (decent, better than HMM 0.94713
  surprisingly close) but Δ −0.09 to −0.15 bp at meta. Embedding-
  class diversity didn't help K=11 pool. DAE v2 (deeper, contrastive)
  may cross.
- **K=4 LR-meta operator OOF→LB transfer = ~5 bp drop.** Today's
  submission OOF 0.95405 → LB 0.95354. Matches K=4+Path-B's transfer
  (OOF 0.95403 → LB 0.95351). Future K=4-class submissions can be
  LB-predicted within ±1 bp from OOF.

## Hedge ladder (R5 / R7 final-window candidates) — unchanged

- K=4 + Path-B C×S τ=100k (LB 0.95351) — clean reference base.
- K=9 qAX + Path-B τ=20k (LB 0.95375) — slim-kNN solo.
- K=27 + Path-B τ=100k (LB 0.95368) — pre-sparse-pool PRIMARY;
  the structurally-different Final-2 candidate.
- `d15b_path_b_K22_dae_only_tau{20k,100k}` — Day-15 PRIMARY
  runner-up.
- `path_b_K5_rf_yekenot_tau{5k,20k,100k}` — Forest-base τ-sweep.
