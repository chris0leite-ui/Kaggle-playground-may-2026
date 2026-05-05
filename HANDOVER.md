# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 7 (2026-05-08)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R14)
2. `audit/2026-05-07-d6-f1-2-LB-result.md` — F1.2 LB win (+2.1bp PRIMARY)
3. `audit/2026-05-07-d6-f1-2-multi-rule.md` — K=18 build details
4. `audit/2026-05-07-d6-critic-loop.md` — Rule 14 audit + 5 untried mechs
5. `scripts/pre_submit_diff.py` — MANDATORY before every submit.
   **NOTE**: tighten ρ threshold to 0.9995 OR diff on `.npy` arrays.
   CSV precision loss flagged TIE_EXPECTED on the Day-6 +2.1bp lift.

Open with a 3-bullet read-back of state + first action.

## Where we are

- **Day 7, 0/10 used today.** Day-6 closed at 1/10 used → **LB
  0.95026 +2.1bp**.
- **PRIMARY** = `d6_k18_multi_rule` LB 0.95026 (M5q + 4
  rule-residual bases). Strat OOF 0.95065. Gap NARROWED from
  −5.2bp (M5q) to −3.9bp (K=18).
- **Headroom to top-5%** (0.95345): **31.9bp**.
- **20 days remaining** (deadline 2026-05-31). 9 slots/day.
- **Active parallel branch**: PI is exploring a different angle on
  another branch. This branch's continuation is the RealMLP-bag
  thread; coordinate at merge points.

## ⚠️ ACTIVE THREAD: RealMLP bag (primary continuation)

`kernels/realmlp-bag-gpu/` was pushed to Kaggle on Day-6 (v1).
RealMLP-TD seeds 123 + 456 in 5-fold Strat with the SAME split
seed (42) as the seed-42 run, so OOF/test arrays are directly
rank-averageable. Per-fold checkpointing.

**ETA**: ~6h after Day-6 evening start; should be COMPLETE by
Day-7 morning.

```bash
kaggle kernels status chrisleitescha/realmlp-bag-gpu
kaggle kernels output chrisleitescha/realmlp-bag-gpu -p scripts/artifacts/
```

Expected files:
- `oof_realmlp_seed{123,456}_strat.npy`
- `test_realmlp_seed{123,456}_strat.npy`
- `realmlp_bag_results.json`

### Day-7 first action — RealMLP bag → K=18 rebuild

1. **Pull artifacts** (above).
2. **Rank-average seeds {42, 123, 456}** for OOF and test:
   ```python
   from scipy.stats import rankdata
   oof_bag = sum(rankdata(o) for o in [oof42, oof123, oof456]) / (3 * N)
   ```
   Save as `oof_realmlp_bag_strat.npy`, `test_realmlp_bag_strat.npy`.
3. **Rebuild K=18 stack**: M5q pool with `realmlp_bag` swapped for
   `realmlp` + 4 rule-residuals. Reuse `scripts/d6_multi_rule.py`
   (just change the realmlp pool entry).
4. **Pre-submit-diff vs `submission_d6_k18_multi_rule.csv`** (NEW
   PRIMARY ref, not M5q). **Threshold ρ < 0.9995** for slot-worthy
   (NOT 0.999 — see friction below).
5. **If OOF > 0.95065 AND ρ < 0.9995 → slot Day-7 slot-1.** HANDOVER
   A.1 prior: expected +1–3bp on top of +2.1bp baseline.

## Re-rankable next moves (post-RealMLP-bag)

| # | Move | Cost | EV (bp) | Notes |
|---|---|---|---:|---|
| A | RealMLP bag K=18 rebuild | 30min CPU | 1–3 | top priority Day-7 slot-1 |
| B | F1.3 classifier-residual (sample_weight) | 2h CPU | 1–2 | inverse rule_proba weighting |
| C | F1.4 rule_proba as meta-feature | 30s CPU | 0–1 | append to LR-meta input |
| D | TabM smoke (Kaggle GPU, 1-fold first) | 6h GPU | 2–6 | Rule 2 |
| E | Sequence-FE LGBM probe | 2h CPU | 1–4 | unmined since Day-2 |
| F | More rules: Compound × Position-bin | 30min CPU | 0–1 | diminishing returns |

## Strategic notes for parallel branch

The PI is working a different angle on another branch. Likely
candidates given Day-6 falsifications:

- **Pseudo-label retry** (Tschalzev 2023 regularized): Day-5 broad
  gate falsified; tighter gate / sample-weighted addresses
  over-amp. Audit §3 side-quest.
- **Sequence model** (Bi-LSTM on Race × Driver): strategy critique
  flagged Day-2 as the largest unmined class; Frontiers AI 2025
  documents 0.988 ROC-AUC. F4 in the audit.
- **TabPFN-v2 / TabICL-v2**: foundation-model inductive bias.

If parallel-branch work lands lift, merge via the meta layer:
stack new bases onto the K=18 pool. Pre-submit-diff on every
combination.

## Falsified Day-6 (do not retry without new evidence)

- **F5 aux-feature GBDT-meta** — +0.12bp over no-aux LGBM, OOF
  −0.78bp vs M5q. 3rd rank-lock confirmation.
- **Move B 2-base [M5q, recursive]** — V1 ρ=0.99996 tie-lock;
  V2-V4 OOF regression. Recursive trained on `m5q_oof_proba` →
  structurally redundant with M5q.

## Critical operating rules (freshly used Day-6)

1. **Pre-submit-diff before EVERY submit.** **Tighten ρ to 0.9995**
   or diff on `.npy` — CSV precision loss flipped verdict on Day-6
   K=18.
2. **Mechanism-class-only**: pool-tweaks via LR-meta are dead
   (3× rank-lock). New slots must change L1 formulation, meta
   family, OR add orthogonal model class.
3. **Predicted-gap gate**: pred-gap <−7bp needs PI sign-off.
4. **Minimal-input-meta sanity check**: for every base-add, train
   2-comp meta on `anchor + new` only. If 2-comp OOF < anchor,
   K-comp lift was memorization. Reject.
5. **Strat-only Day-3+** (R1; U3 confirmed i.i.d.).
6. **Track gap direction** — Day-6 K=18 narrowed gap (opposite of
   d5 over-amp). Real positive transfer.

## Calibration ladder snapshot (Day 7 morning)

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | gap −5.2bp |
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | **gap −3.9bp NARROWED** |

## Held submissions (do not blindly submit)

- (carry-forward) `m5x_yetirank.csv`, `m5z_yetirank_nb.csv` — TIE
- `m5_meta_lgbm_medium.csv`, `m5_meta_hgbc.csv` — meta variants
- `d5_meta_k15_*.csv`, `m5_k15a/b/c.csv` — K=15 NULLs
- `d5_partial_pseudo_m5q.csv` — burned LB 0.94963
- `d6_aux_meta_with_aux.csv` — F5 falsified
- `d6_2base_v[1-4]_*.csv` — Move B falsified
- `d6_k15_rule_residual.csv`, `d6_k16_two_diverse.csv` — superseded

## Pointers

- `audit/2026-05-07-d6-f1-2-LB-result.md` — Day-6 LB win
- `audit/2026-05-07-d6-f1-2-multi-rule.md` — K=18 build
- `audit/2026-05-07-d6-critic-loop.md` — Rule 14 audit + 5 untried
- `audit/2026-05-07-d6-move-c-rule-residual.md` — F1.1 single rule
- `scripts/d6_multi_rule.py` — F1.2 builder (reuse for K=18 rebuild)
- `scripts/d6_rule_residual.py` — F1.1 builder
- `scripts/pre_submit_diff.py` — MANDATORY (tighten ρ to 0.9995)
- `kernels/realmlp-bag-gpu/` — Move F kernel (running)
