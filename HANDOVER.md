# HANDOVER

Next-session brief. **PI says "handover"** → agent reads this file
and proceeds. **PI says "prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

This file is rewritten (not tail-appended) every wrap-up. Prior
versions: `audit/archive-YYYY-MM-DD-handover-*.md`.

---

## Where we are

**PRIMARY: R25 rank-mean blend (K=18 × 0.89 + R21 CB YetiRank × 0.11).
LB 0.95402.** Set 2026-05-21 PM. Top-5% gap −0.7 bp; leader gap −7.4
bp. File: `submissions/submission_R25_K18_R21_rankmean_w89.csv`.

Submissions: **52 / 270** total; **2 used 2026-05-21** (R25, R27);
**8 daily slots still available**. Comp-day **21 of 31**; days
remaining **10**. Final-3-day lock window: **2026-05-29 → 2026-05-31**.

## Today's session arc (2026-05-21)

1. **R20 LGB lambdarank** on (Y,R,L) cohort: K=19 add +0.011 bp vs
   K=18 — sub-G2, TIE_ZONE. Standalone OOF 0.95165.
2. **R21 CB YetiRank** on (Y,R,L) cohort (Kaggle T4×2 GPU, 3 min):
   standalone OOF 0.95396 (just 0.5 bp below R15), ρ vs R17 = 0.746
   (cross-family orthogonal). K=19 add REGRESSED -0.061 bp (Path-B
   LR-meta absorbed as same direction as R17). **But its standalone
   prediction became the load-bearing R25 blend ingredient**.
3. **R23 cohort-aggregate features** (5 derived): K=24 add regressed
   -0.054 bp. Feature mechanism on cohort axis also closed.
4. **K=20 combo (R17+R20+R21)**: regression -0.086 bp vs K=18.
5. **R26 XGB / LGB stacker swap** on K=20: XGB -6.79 / LGB -2.78 bp
   vs Path-B. Stacker is NOT the bottleneck.
6. **R25 K=18 + R21 rank-mean blend** at w=0.89: OOF 0.954508 (+0.179
   bp vs R15 OOF; +0.087 bp vs K=18 OOF). **LB 0.95402** (+0.05 bp vs
   R15 LB; +0.04 bp vs K=18 LB). 351 flips vs K=18 (R7d-eligible).
7. **R27 4-way blend** (K=18 + R21 + K=27 + R7.2): OOF 0.954519
   (+0.111 bp vs R25 OOF) but ρ_R25=0.9999 TIE_ZONE → LB 0.95402
   tied R25. Used as HEDGE 0 for structural diversity.
8. **R28 XGB rank:ndcg** Kaggle GPU: standalone OOF 0.95242 — weaker
   than R21; blends with it didn't break LB ceiling.
9. **R29 CB QueryCrossEntropy / YetiRankPairwise** Kaggle GPU: result
   pending at handover time.

## Key finding

**Path-B LR-meta absorbs cohort-listwise as one direction.** Adding
multiple listwise bases (R17/R20/R21) to the K=N+1 pool doesn't lift —
the meta only learns one cohort-listwise coefficient. BUT
**submission-level rank-blend bypasses this**: combining pre-LR-meta
R21 standalone (which had high OOF + orthogonal ρ) with post-LR-meta
K=18 extracted +0.179 bp OOF / +0.05 bp LB.

The blend ceiling at LB 0.95402 was confirmed via:
- R21 weight sweep — optimal w=0.11; w≥0.30 breaks ρ but drops OOF
- Different operators (arith/gmean/logit_mean) — rank_mean dominates
- 3-way and 4-way blends — OOF marginally higher (+0.01-0.02 bp) but
  ρ_R25 stays at 0.9999 → LB-tie
- New cross-family standalones (R28 XGB) — weaker standalone OOF, no
  blend lift

To break past LB 0.95402 requires a NEW STRONG ORTHOGONAL STANDALONE
(R21-quality at OOF 0.954+ with ρ_R21 < 0.95). R29 pending; if null
the listwise-cohort axis is fully closed for this comp.

## Active axes

| Axis | Status |
|------|--------|
| Submission-level rank-blend | **OPEN, lifted** (R25 LB 0.95402) |
| Path-B K-add (cohort-listwise) | CLOSED — R20/R21/R23/K=20 combo all sub-G2 or regression |
| Non-linear stacker swap | CLOSED — R26 XGB/LGB metas worse than Path-B |
| Cross-family standalone bases | OPEN — R21 worked; R28 weaker; R29 pending |
| Wide-pool blending (K=27) | OPEN — included in R27 4-way hedge |
| Inventor track NN family | 1 lift (TabM) / 2 nulls (R18 multitask, R19 aleatoric) — frozen |
| Per-cohort calibration | UNTRIED — would need inner-CV per R33 |
| Pseudo-labels from test | UNTRIED |
| Sequence base (GRU/LSTM) | UNTRIED |

## Next-session priorities

1. **Read R29 result** if not yet pulled. If standalone OOF ≥ 0.95390
   AND ρ vs R21 < 0.95 → new blend ingredient, test K=18+R21+R29
   3-way rank-blend → LB submit.
2. **Untried axes** (in cost order):
   - Bag 5-seed R21 — variance-reduction of the load-bearing ingredient
     (3 min × 5 on Kaggle GPU). If R21-bag OOF > 0.95400, blend may
     break to LB 0.95403.
   - Per-cohort isotonic calibration of R25 with inner-CV (R33). ~2 hr
     local. Expected lift: tiny but legitimate.
   - Sequence base (GRU on per-driver lap history with K=17 logit
     context). Kaggle GPU. ~3 hr.
3. **Final-window prep (8 days away)**: HANDOVER.md + state/current.md
   already R25-anchored. R27 is HEDGE 0. Need to validate selection
   pair (PRIMARY=R25, HEDGE=R27 or HEDGE=K=18) by Day 30 with PI sign-off.

## Critical files

- `state/current.md` — R25 PRIMARY (current)
- `state/hedge-ladder.md` — R25 PRIMARY, R27 HEDGE 0, K=18 HEDGE 1
- `state/mechanism-ledger.md` — needs R20-R29 entries appended
- `submissions/submission_R25_K18_R21_rankmean_w89.csv` — PRIMARY CSV
- `submissions/submission_R27_4way_K18_R21_K27_R72_70_12_10_08.csv` — HEDGE 0
- `scripts/probe_r20_listwise_variants.py` — R20 (LGB lambdarank cohort)
- `scripts/probe_r23_cohort_aggregate_features.py` — R23 features
- `scripts/probe_r25_blend_today.py` — R25 blend harness
- `scripts/probe_r26_xgb_meta_swap.py` — R26 stacker swap
- `kernels/r21-yetirank-cohort-gpu/` — R21 CB YetiRank Kaggle (LIFT INGREDIENT)
- `kernels/r28-xgb-ndcg-cohort-gpu/` — R28 XGB ndcg Kaggle (weak null)
- `kernels/r29-cb-qce-cohort-gpu/` — R29 CB YetiRankPairwise (pending)

## Frictions surfaced today

- **k20-output-collision-pathb-k19**: building two different K=19 pools
  (R20 vs R21) writes to same `oof_K19_pathb_*.npy` filename — must
  rename between runs. Use `--out` param + post-rename, or add tag.
  Same friction class as `k14-output-collision-extra-bases`.
- **cb-gpu-query-cross-entropy-size-limit**: CatBoost GPU
  QueryCrossEntropy errors at max query size > 256. (Y,R,L) cohorts
  max=373. Workaround: switch to YetiRank/YetiRankPairwise (no limit)
  or use CPU.
- **cwd-drift-mid-bash-kernel-output**: `kaggle kernels output` run
  inside a kernel dir creates `kernels/<name>/kernels/<name>/output/`
  nested structure. Use absolute paths or explicit cd back to repo root.
- **blend-sweep-grid-too-dense**: `probe_r25_blend_today.py` at
  step=0.025 (k=2), 0.05 (k=3), 0.10 (k=4) gives ~10k blends × 300ms
  = 50 min wall — killed at 34 min. Use coarser step (≥0.05) or skip
  rank_mean operator for grid sweep (it's the slow one).
