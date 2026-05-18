# HANDOVER

Next-session brief. **PI says "handover"** → agent reads this file
and proceeds. **PI says "prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

This file is rewritten (not tail-appended) every wrap-up. Prior
versions: `audit/archive-YYYY-MM-DD-handover-*.md`.

---

## Where we are

**Active PRIMARY: rank-blend 70/30 K=11 + K=9. LB 0.95386.**
Unchanged from 2026-05-14. Top-5% gap −1.9 bp; leader gap −9.0 bp.

Submissions: **42 / 270** total; **0 used 2026-05-18**. Comp-day
**18 of 31**; days remaining **13**.

File: `submissions/submission_blend_K11_K9_w_70_30.csv`.

## 2026-05-18 session — three rounds; ceiling validated

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

## Cumulative null tally for 2026-05-18

**15 distinct mechanism classes tested today; all NULL/regress
at the gates available** (K=4+1, K=21+1, K=4+K27super+1):

Round 1 (4): a2_2 mandatory-compound, a3_1 rank-sorted-gaps,
C3 per-(R,L) shrinkage, C4 UID magic.

Round 2 (9): RRF blend, trimmed-rank blend, stint-cap multiplier,
LGBM rank_xendcg meta, SGD hinge pairwise meta, torch AUC
surrogate, per-Driver random-intercept BLUP, per-Driver random-
slope BLUP, per-bin conformal widths.

Round 3 (2): multi-anchor retest of 3 candidates (null at K=4
AND K=4+K27super); Caruana hill-climb (degenerates to K=27).

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

1. **Rebuild slim-kNN bases** (~3-6 hr CPU). Highest-priority
   operational fix. Reuses `scripts/build_K11_full_pathb.py`
   and the upstream dgp_v3 builders.
2. **Push K=11+K=9 OOFs to Kaggle artifact dataset** via
   `kaggle datasets version` once kNN bases are rebuilt
   (~10 min wall).
3. **Re-test all 11 Round-2 nulls + 3 Round-3 retests at K=11+1**
   with the proper anchor. ~30 min CPU. Confirms or refutes the
   ceiling at the real PRIMARY level.
4. **C2 swap-noise DAE on Kaggle T4** (~2-3 hr GPU). Highest-EV
   remaining mechanism class; Porto Seguro 1st-place precedent.
5. **C1 OpenF1 per-Race scalar join** (~45 min). Sub-bp expected
   but novel join key.
6. **Final-window hedge prep** (Days 10-13). Final-1 = PRIMARY
   K=11+K=9 (LB 0.95386). Final-2 = K=27 + Path-B (LB 0.95368)
   — structurally different; wins ~30-40% of private-LB
   realizations per headroom-math agent.

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
  the row-feature subspace is locked at OOF ~0.95430. Future
  challenger mechanisms must be tested against BOTH anchors;
  a positive result at K=21+1 alone is misleading (K=21 base is
  weak).
- **LR with log-loss is C-robust loss-optimal at K=4 meta.**
  Sweep across 4 orders of magnitude (C ∈ {0.01, 100}) shows
  the LGBM-rank candidate is null at every C. Closes the loss-
  class diversity axis at the K=4 meta layer.
- **Per-Driver random-effect BLUP adds noise on synth.** Every
  lambda from 50-2000 regresses; closes the per-actor heterogeneity
  axis.

## Hedge ladder (R5 / R7 final-window candidates) — unchanged

- K=4 + Path-B C×S τ=100k (LB 0.95351) — clean reference base.
- K=9 qAX + Path-B τ=20k (LB 0.95375) — slim-kNN solo.
- K=27 + Path-B τ=100k (LB 0.95368) — pre-sparse-pool PRIMARY;
  the structurally-different Final-2 candidate.
- `d15b_path_b_K22_dae_only_tau{20k,100k}` — Day-15 PRIMARY
  runner-up.
- `path_b_K5_rf_yekenot_tau{5k,20k,100k}` — Forest-base τ-sweep.
