# HANDOVER

Next-session brief. **PI says "handover"** → agent reads this file
and proceeds. **PI says "prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

This file is rewritten (not tail-appended) every wrap-up. Prior
versions: `audit/archive-YYYY-MM-DD-handover-*.md`.

---

## Where we are

**PRIMARY: R25 rank-mean blend (K=18 × 0.89 + R21 CB YetiRank × 0.11).
LB 0.95402.** Set 2026-05-21 PM. Day-22 confirmed LB ceiling on own bases.

**Realistic position: rank ~297 / 2063 (top-14.4%).** HANDOVER prior
("top-5% gap -0.7 bp") was stale; current downloaded LB shows:
- Top-1% boundary: 0.95453 (+5.1 bp gap)
- Top-5% boundary: 0.95449 (+4.7 bp gap)
- Top-10% boundary: 0.9542 (+1.8 bp gap)

Strategy-Critic Section 5 (headroom math) ran at plateau + 65% checkpoint:
**own-bases discounted lift ~0.30 bp << +1.8 bp gap to top-10%**.
Top-5% mathematically unreachable on own bases. PI rejected public-kernel
CSV blending; R22 IDEA-scan deferred pending PI approval.

Submissions: **53 / 270** total; **3 used 2026-05-21** (R25, R27, R31);
**7 daily slots still available**. Comp-day **21 of 31**; days remaining
**10**. Final-3-day lock window: **2026-05-29 → 2026-05-31**.

## Today's session arc (2026-05-21 PM, comp-day 21)

1. **R30 5-seed bag of R21** (Kaggle GPU): OOF 0.954027 (+0.07 bp vs R21).
   ρ vs K18=0.9795, ρ vs R21=0.9988. Useful low-weight blend ingredient.
2. **R31 (Driver,Y,R) cohort YetiRank** (Kaggle GPU): OOF 0.95178 — WEAK
   NULL. Cohort definition too fine (median 11 laps); listwise loss
   loses signal-to-noise.
3. **R32 (Y,R,Stint) cohort YetiRank**: ERROR — CB GPU YetiRank max query
   size = 1023, (Y,R,Stint) max = 5,621. New friction
   `cb-gpu-yetirank-max-query-1023`.
4. **R31-submission (3-way K18+R30+R72 rank_mean 0.50/0.15/0.35)**: nested-CV
   OOF 0.954558 (+0.010 bp vs R25), ρ_R25=0.99993 (TIE_ZONE-extended).
   **LB 0.95402 — TIE confirmed**. Closes rank-blend extra-weight axis.
5. **Per-cohort isotonic of R25 (R33 inner-CV)**: REGRESSED -13 to -44 bp
   across all 3 cohort defs tested. CLOSED — confirms 2026-05-08 friction.
6. **R34 pseudo-label CB YetiRank**: OOF 0.953709 — NULL. Pseudo-labels
   from K18 (>=0.85 / <=0.02 thresholds, ~112k rows) contaminated cohort
   ranking; standalone 25 bp below R21.
7. **R35 FT-Transformer** (feature-axis self-attention, 3L×96D×8H, 9 num
   + 4 cat + 17 K17-logit tokens, 8 epochs): OOF **0.954086** —
   STRONGEST NN-class base ever (TabM was 0.93856; GRU 0.93066). ρ vs
   K18=0.9832 (moderately orthogonal). But best K18+R30+R35 3-way nested
   OOF 0.954547 (BELOW R31's K18+R30+R72 = 0.954558). NULL for lift but
   useful as PRIMARY hedge ingredient (most-orthogonal NN base).
8. **R37 CB PairLogit cohort YetiRank**: OOF 0.951919 — WEAK NULL. ρ vs
   R21=0.9499 (most-orthogonal cohort base ever) but standalone gap kills
   blend utility. Closes CB ranking-loss-variant axis.
9. **R36 10-seed R21 bag**: OOF **0.954039** (+0.012 bp vs R30, +0.083 bp
   vs R21). ρ vs R30 = 0.99979 — indistinguishable at LB resolution. Best
   R36-blend nested OOF 0.954557 vs R31's 0.954558 — no improvement.
   No additional submit; PI directive met (nothing above TIE_ZONE).

## Key finding

**LB ceiling at 0.95402 confirmed**. Every rank-blend variant tested
today produces OOF in [0.95453, 0.95456] but ρ vs R25 in [0.99986,
0.99996] — TIE_ZONE-extended. Empirically, this band ties at LB in this
comp (friction `blend-op-axis-closed-at-r15`).

To break ceiling requires NEW STRONG ORTHOGONAL STANDALONE with
**OOF >0.954 AND ρ_K18 < 0.96**. Today's NN/cohort/pseudo-label attempts
all failed to meet this:
- R35 FT-Transformer: OOF 0.954086 ✓, ρ_K18 0.9832 ✗
- R37 PairLogit: OOF 0.951919 ✗, ρ_K18 0.9695 ✓
- R30 5-seed bag: OOF 0.954027 ✓, ρ_K18 0.9795 ✓ — but redundant with R21

R17 (already in K=18 pool) has ρ_K18 = 0.7450 — the most orthogonal
high-OOF base by far — but as standalone OOF 0.943264 has too-large
gap. Path-B meta absorbs R17's signal optimally; using as rank-blend
ingredient REGRESSES.

## Active axes

| Axis | Status |
|------|--------|
| Submission-level rank-blend | **SATURATED** at 0.95402 (R31, R25, R27 all tied) |
| Path-B K-add (cohort-listwise) | CLOSED |
| Per-cohort isotonic of PRIMARY | CLOSED (-13 to -44 bp) |
| Pseudo-label cohort YetiRank | CLOSED (-0.025 bp standalone) |
| CB ranking-loss variants (PairLogit) | CLOSED (-2.0 bp standalone) |
| Cohort definition variants ((Y,R,L) unique) | CLOSED ((Driver,Y,R) -2.2 bp; (Y,R,Stint) GPU-blocked; (Y,R) GPU-blocked) |
| NN-class structural diversity | OPEN — R35 FT-Transformer strongest yet, but blend regression at K-add layer |
| Final-3-day hedge ladder prep | OPEN — needed by 2026-05-29 |
| R22 public-notebook IDEA-scan | DEFERRED — PI authorization pending |

## Next-session priorities

1. **R36 result check**: if standalone OOF > 0.95405 AND ρ vs R30 < 0.95
   → new blend ingredient (probably tiny lift). Likely NULL.
2. **Final-window hedge-ladder build** (high priority; lock window opens
   in 8 days):
   - PRIMARY: R25 (LB 0.95402)
   - HEDGE 0: R31 (3-way K18+R30+R72; LB 0.95402, structurally different blend)
   - HEDGE 1: K=18 unblended (LB 0.95398; clean reversion test)
   - HEDGE 2 candidate: 4-way blend including R35 (different orthogonal ingredient)
   - Need: at Day 28, run private-LB regression-risk probe per R5d.
3. **R22 public-notebook IDEA-scan** if PI authorizes (cheap; READ-ONLY mechanism research, no CSV blending).
4. **Strategic accept**: top-5% / top-10% unreachable; aim for top-15%
   private-LB stability via well-hedged final pair.

## Critical files

- `submissions/submission_R25_K18_R21_rankmean_w89.csv` — PRIMARY (LB 0.95402)
- `submissions/submission_R31_K18_R30_R72_rankmean_50_15_35.csv` — HEDGE 0 (LB 0.95402)
- `submissions/submission_R27_4way_K18_R21_K27_R72_70_12_10_08.csv` — HEDGE 1 (LB 0.95402)
- `submissions/submission_K18_pathb_driverclass_stint_tau100000.csv` — HEDGE 2 (LB 0.95398)
- `scripts/artifacts/oof_R30_r21_bag_yrl_strat.npy` — R30 5-seed bag
- `scripts/artifacts/oof_R35_ft_transformer_strat.npy` — R35 FT-Transformer
- `scripts/artifacts/oof_R37_pairlogit_yrl_strat.npy` — R37 PairLogit (NULL)
- `scripts/artifacts/oof_R34_pseudolabel_yrl_strat.npy` — R34 pseudo-label (NULL)
- `audit/2026-05-21-day-22-pivot.md` — today's full session arc + Strategy-critic
- `kernels/r36-r21-10seed-bag-cohort-gpu/` — pending at session end

## Frictions surfaced today

- **cb-gpu-yetirank-max-query-1023**: CB GPU YetiRank fails at max group >
  1023. Cap cohort definitions accordingly. Supersedes prior friction
  `cb-gpu-query-cross-entropy-size-limit` which claimed YetiRank had no limit.
- **public-LB-gap-stale-in-handover**: HANDOVER's top-5%/top-10% gaps from
  earlier in comp were stale. Promotion: refresh LB-percentile gaps at every
  session start via `kaggle competitions leaderboard --download`.
- **per-cohort-isotonic-needs-explicit-block**: 30 min spent prototyping
  per-cohort isotonic despite 2026-05-08 friction
  `isotonic-overfits-when-base-calibrated`. Add to permanent-closed list
  with cohort-agnostic note.
- **monitor-until-loop-empty-status-false-positive**: until-loop condition
  becomes true when CLI returns empty. Fix: require non-empty status string.
- **rank-blend-ceiling-at-LB**: 3 submits with OOF +0.013-0.021 bp vs R25
  all LB-tied at 0.95402. Promotion: rank-blend axis closed when ρ_PRIMARY
  > 0.9998; further blend search waste of slots.
- **pseudo-label-breaks-cohort-listwise-signal**: R34 OOF -0.025 bp vs R21;
  K18 pseudo-labels at extreme thresholds inject biased rows into cohorts,
  break within-cohort label distribution that listwise loss needs.
