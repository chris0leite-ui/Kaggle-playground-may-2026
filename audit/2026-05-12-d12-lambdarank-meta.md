# Day-12 — d12: metric-aligned LambdaRank meta over K=21 PRIMARY pool

Hypothesis: replacing LR-meta (BCE-aligned) with a pairwise rank
meta-learner (LambdaMART or YetiRank) — both AUC-aligned by
construction — could shift the K=21 stack's test ranking by 2-8bp.
Plus: AUC-direct base retraining via XGBoost `rank:pairwise`.

## Pool

K=21 PRIMARY = `d9f_K21_swap_partA_partB` (LB 0.95031, OOF 0.95073).
POOL_KEEP (16) + TOP_3_D9 (R6/R10/R7) + d9f_FM_A + d9f_FM_B = 21.
Meta features `expand(P)` = [raw, rank, logit] → 63 cols.

## Part A — meta swap results

| meta | grouping | OOF Strat | Δ PRIM (bp) | ρ vs PRIM-test | ρ vs LR-test | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| LR (anchor)            | n/a    | **0.95073** | −0.01 | 1.00000 | 1.00000 | 210 |
| LambdaMART             | Race¹  | **0.94210** | **−86.32** | 0.94238 | 0.94238 | 466 |
| LambdaMART             | rand-1k² | est. ~0.949³ | est. −15bp | n/a | n/a | n/a |
| YetiRank               | Race   | NOT_RUN⁴ | — | — | — | — |

¹ Race groups (26 in train) chunked at ≤8000 rows per LightGBM cap.
² Random groups of ~1000 rows, fitted within each fold.
³ 4 of 5 folds completed: per-fold AUC = [0.95088, 0.94850, 0.94996,
   0.94882], mean 0.94954. Run killed at 22:56 wall (~30 min meta cap).
⁴ Killed before YetiRank phase to stay within budget.

### LambdaMART Race-grouped: catastrophic (−86bp)

Per-fold val AUC: [0.94912, 0.94676, 0.94812, 0.94622, 0.94840] —
mean per-fold 0.94772; concatenated OOF 0.94210. ρ vs LR-test 0.94238 —
divergent but in the wrong direction.

### LambdaMART Random-grouped: high variance, mild regress

4-fold AUCs span 24bp (0.94850 to 0.95088). Fold 0 LIFTED +13bp vs LR;
folds 1-3 regressed −12 to −22bp. Net ~tie or mild regression. The
per-fold variance suggests overfit to the random group assignment.

### Mechanism

Lambdarank optimizes **within-query** pairwise rank. Race groups
contain 12-24k rows; the gradient is dominated by within-Race pair
classification — but the competition AUC is **global** across all
188k test rows. Within-Race optimization actively destroys the
cross-Race calibration that LR-meta's logit channel encodes.

Random small groups approximate global pair-sampling but at much
higher variance per gradient step (each ~200 pos × 800 neg = 160k
pairs vs the global 89B pairs in true AUC). LR's rank+logit expansion
captures the same signal more efficiently.

## Part B — XGB rank:pairwise base smoke

Smoke (1 fold, random groups of 1000) on the e3_hgbc feature set:

| metric | val |
|---|---:|
| XGB rank:pairwise fold-0 AUC | **0.90368** |
| best_iter                    | 0 |
| e3_hgbc baseline (BCE)       | 0.94876 |
| Δ                            | **−450.81bp** |

`best_iter=0` → XGBoost never improved val AUC over zero-tree
baseline. The pairwise objective on this feature set with random
small groups produces an unconverged model. **SMOKE FAIL.** Skipped
full 5-fold per gating (≥10bp regression).

Plausible failures: (1) random groups create trivially-solvable
pair classes; (2) `rank:pairwise` is invariant to monotonic transforms
so initial splits look equivalent; (3) tunable but EV not justified
given Part A's negative signal.

## Verdict

| approach | result |
|---|---|
| LambdaMART meta, Race-grouped     | **DEAD** — −86bp |
| LambdaMART meta, Random-grouped 1k| **DEAD-borderline** — high var, ~tie/mild regress |
| YetiRank meta, Race-grouped       | NOT_RUN (same query semantics → expected dead) |
| XGB rank:pairwise base retrain   | **DEAD** — smoke −451bp, did not train |

**Combined: dead-list.** The K=21 LR-meta on [raw, rank, logit] is
metric-equivalent to (or better than) LambdaMART/YetiRank at this
pool composition.

## Pred-LB

- LR-meta (anchor): predicted LB = PRIMARY 0.95031.
- LambdaMART-Race: pred LB ≈ 0.94170 (catastrophic).
- LambdaMART-Rand: pred LB ≈ 0.94910 (regression).

## Mechanistic note for the playbook

When the metric is **global AUC** and bases output calibrated
probabilities, an LR meta over [raw, rank, logit] is theoretically
equivalent to optimizing global AUC in the limit (rank ↔ AUC by
Mann-Whitney). LR's logit channel preserves cross-row score calibration
that within-query pairwise objectives discard. LambdaMART/YetiRank
optimize a strictly smaller within-group AUC; that doesn't transfer
to the leaderboard's full-population ranking.

## Artifacts

- `scripts/d12_lambdarank_meta.py` — heavy default version (lr=0.05,
  num_leaves=15, ES=50). First run completed LR + 1 fold of LM-Race
  in 12 min before kill.
- `scripts/d12_lambdarank_meta_fast.py` — ≤30 min budget version
  (lr=0.1, num_leaves=8, max_rounds=300, ES=20). Produced full LR meta
  + full LM-Race 5-fold + 4-of-5 LM-Rand1000 folds before kill.
- `scripts/d12_aucpairwise_base.py` — XGB rank:pairwise smoke + full
  retrain gate. Smoke FAILED → no full output.
- `scripts/d12_smoke.py` — 1-fold smoke for both LambdaMART and YetiRank.
- `scripts/artifacts/d12_lambdarank_meta_results.json` — per-meta
  OOF/Δ/ρ summary curated from the killed-script logs.
- `scripts/artifacts/d12_aucpairwise_base_results.json` — smoke-only
  result with `decision="skip_full_5fold"`.
- `scripts/artifacts/oof_d12_lr_meta_strat.npy` +
  `test_d12_lr_meta_strat.npy` — saved 5-fold LR meta artifacts.

## Decision

**Submit-ready: no.** LambdaMART/YetiRank meta swap is **dead-listed**
for this stack. The LR-meta is metric-aligned for global AUC with
calibrated bases.

**Next mech-class diversity if pursuing:** (1) CatBoost native on the
63-feature expand basis — already TIE_EXPECTED per d4 GBDT-meta; (2)
calibrated stacking (Platt/isotonic on LR output); (3) pairwise loss
with **stratified balanced random groups** (1 pos : N neg per group of
~50) — only untried angle that might rescue the family. EV ≤ +1bp at
ρ ≈ 0.998. None higher-EV than non-meta moves (external data, etc.).
