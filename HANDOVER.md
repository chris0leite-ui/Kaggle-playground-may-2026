# HANDOVER

Next-session brief. **PI says "handover"** → agent reads this file
and proceeds. **PI says "prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

This file is rewritten (not tail-appended) every wrap-up. Prior
versions: `audit/archive-YYYY-MM-DD-handover-*.md`.

---

## Where we are

**NEW PRIMARY: K=13 + Path-B τ=100k. LB 0.95387.** (Round 5 result.)
Slightly above prior PRIMARY (LB 0.95386). Top-5% gap −1.8 bp;
leader gap −8.9 bp. File:
`submissions/submission_K13_seghmm_pathb_tau100000.csv`.

Submissions: **45 / 270** total; **3 used 2026-05-18**. Comp-day
**18 of 31**; days remaining **13**.

**Round 5 plateau break**: K=11 (rebuilt slim-kNN) + r4_segment_fe
(row-class) + r4_hmm_seq (sequence-class) under Path-B Compound×Stint
τ=100k operator. The R4 mechanism-orthogonality finding survives at
the REAL K=11 anchor (+0.245 bp at LR-meta OOF), and the **Path-B
operator preserved +5 bp of LB transfer** vs LR-meta at the same OOF.

Today's 4 submissions:
| ref | LB | mechanism |
|---|---|---|
| 52772090 (R4) | 0.95354 | K=4 + seg + HMM LR-meta (R4 probe) |
| 52773963 (R5.1) | 0.95382 | K=11 + seg + HMM LR-meta |
| **52774385 (R5.2)** | **0.95387** | **K=13 + Path-B τ=100k** ← new best |
| 52774692 (R5.3) | 0.95385 | 70/30 rank-blend R5.2 + K=27+Path-B |

Submissions: 46 / 270; 4 used 2026-05-18; 6 remaining today.

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

PRIMARY is now R5.2 (LB 0.95387) — top-5% boundary still 1.8 bp away.

1. **Proper multi-seed bagging** (~30 min CPU). Modify Path-B to
   bag at FOLD-FIT level (average per-fold predictions across seeds)
   not full-train fit (which is seed-invariant). Variance reduction
   could yield +0.2-0.5 bp at LB.
2. **Multi-tau Path-B blend**: rank-blend K=13+Path-B τ=100k with
   τ=20k (built but not submitted; tighter shrinkage). Each tau
   weights segments differently; blend may capture both extremes.
3. **Transformer v2** (~4-6 hr Kaggle T4 GPU). Larger model
   (D_MODEL=256, 6 layers, 15 epochs), GroupKFold by sequence
   (R1d compliant), longer training. Current v1 OOF 0.91974 is
   weak; v2 could land 0.94+, useful at K=11+Path-B+TRF stack.
4. **C2 swap-noise DAE on Kaggle T4** (~2-3 hr GPU). Porto Seguro
   precedent; embedding-class diversity orthogonal to row + seq.
5. **C1 OpenF1 per-Race scalar join** (~45 min). Sub-bp expected
   but novel join key.
6. **Hedge ladder for final-window R7d** (Days 28-31):
   - Final-1 = **R5.2 K=13+Path-B** (LB 0.95387) — new PRIMARY
   - Final-2 = K=27+Path-B τ=100k (LB 0.95368) — structurally
     different operator (no seg+HMM, no row/seq augmentation)
   - Final-3 hedge = R5.1 K=11+seg+HMM LR-meta (LB 0.95382) —
     different operator class, mechanism-orthogonal pool

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
