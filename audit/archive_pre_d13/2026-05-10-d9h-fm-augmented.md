# Day-9h — Feature-augmented FM (12 fields)

> Hypothesis: richer features in a unified FM beat d9f's 2-way
> partition. Falsified at the stack level.
> Builder `scripts/d9h_fm_augmented.py` (241s wall total).

## Feature design

Original d9c FM had 8 fields. d9h adds 4 new field types:

  Original 8: D, C, R, Y, S, T_q5, Rp_q5, P_q5
  + Nx (next_compound, 93.3% test coverage from d9 P5)
  + Pv (prev_compound, 93.2% test coverage)
  + Cd (Cumulative_Degradation quintile)
  + Ld (LapTime_Delta quintile)

12 fields per row → **66 pairwise interactions** per row (vs unified
8-field FM's 28). FM model unchanged: k=8 embeddings, 6 epochs,
SparseAdam, batch 8192, lr=0.05.

## Standalone — strongest single FM yet

| Model | Std OOF | Δ vs d9c FM | ρ vs d9f PRIMARY |
|---|---:|---:|---:|
| d9c FM (8 feat) | 0.92069 | (anchor) | 0.899 (vs d6_k18) |
| **d9h FM_aug12 (12 feat)** | **0.92540** | **+4.7bp** | 0.917 |

d9h FM_aug12 is the **strongest single FM in the entire project**.
The 4 new features genuinely add information — particularly
next_compound (P5 flagged it as the largest unused signal).

But ρ vs d9f PRIMARY is **higher** (0.917 > d9c's 0.899 vs d6). The
new features overlap with what GBDT pool bases extract, so adding
them to an already-saturated FM increases its correlation with
PRIMARY rather than orthogonality.

Min-meta vs PRIMARY (using d9c K=20 OOF as anchor since we don't
have d9f K=21 OOF saved): **−0.36bp FAIL**. The unified FM at this
strength fails the gate vs PRIMARY which already has FM_A+FM_B.

## K=N stack experiments

PRIMARY = d9f K=21 swap+multi-FM (Strat OOF 0.95073, LB 0.95031).

| Stack | K | OOF | Δ PRIMARY | ρ |
|---|---:|---:|---:|---:|
| S1 K=20 swap (replace d9f 2-way with FM_aug12) | 20 | 0.95070 | **−0.31bp** | 0.99939 |
| S2 K=22 add (keep d9f 2-way, add FM_aug12) | 22 | 0.95073 | **+0.01bp** | 0.99978 |

S1 (single unified strong FM) regresses by 0.31bp. The single FM is
"more powerful" but provides only one ρ-axis; the LR meta needs two
diverse FMs to extract more total information.

S2 (add FM_aug12 on top of d9f 2-way) is essentially flat.
Critical L1 ranking finding: in the K=22 add stack, **FM_A_d9f and
FM_B_d9f are demoted out of L1 top-15** (L1<0.4) while FM_aug12 sits
at L1=0.512. **FM_aug12 effectively replaces the partitioned pair**
in the meta's weighting — getting almost the same total contribution
as the d9f 2-way did alone. Adding it is redundant with what was
already there.

## Why feature augmentation didn't lift the stack

The d9c→d9f progression worked because partition diversity captured
information the LR meta could NOT have extracted from a single FM
alone (different ρ signatures vs PRIMARY).

The d9h augmentation gives a single FM more raw predictive power,
but that power is *correlated* with what the partitioned d9f pair
already provides. The LR meta sees:
- d9f path: 2 FMs at ρ=0.49 / 0.86 vs the GBDT consensus, total
  routing weight ≈ 0.46 + 0.41 = 0.87.
- d9h path: 1 FM at ρ=0.92 vs the GBDT consensus, total routing
  weight ≈ 0.51.

The d9f pair's lower individual correlations let the meta extract
information from BOTH directions of feature-set partition. d9h's
single stronger FM is pulled toward the GBDT consensus, despite
having more features.

**Information-theoretic reading**: standalone strength is bounded by
what features express, but stack contribution is bounded by what the
META can route. Two correlated-but-distinct ρ-axes (d9f pair) >>
one stronger but more-correlated axis (d9h unified).

## Implication: try augmented partition

The 4 new features ARE useful — they just need to feed the
*partitioned* FMs, not a unified one. Next-step candidates:

1. **d9i: Augmented 2-way partition**
   - FM_A_aug: D, C, S, T_q5, **Cd, Ld** (driver-dynamics + degradation)
   - FM_B_aug: R, Y, Rp_q5, P_q5, **Nx, Pv** (race-context + neighbor compound)
   - 6 features each → 15 pairwise interactions each. Tests whether
     partition + augmentation > partition alone.
   - ~3 min CPU.
2. **d9j: Asymmetric augmented partition**
   - Put all 4 new features on one side: e.g., FM_B' = R+Y+Rp+P+Nx+Pv+Cd+Ld
     while FM_A = D+C+S+T_q5 unchanged.
   - Tests whether the new info has a "natural side".

Both candidates are ~5 min CPU, safe to try.

## Triage decision

**HOLD all d9h candidates.** d9f K=21 swap remains PRIMARY (LB 0.95031).

The strongest single FM ever (FM_aug12, std OOF 0.92540) provides
no slot-worthy lift over the d9f 2-way partition.

The next step is to **combine** feature-augmentation with the 2-way
partition (d9i), capturing both diversity benefits at once.

## Pointers

- `scripts/d9h_fm_augmented.py` — 12-field FM builder.
- `scripts/artifacts/d9h_aug12_results.json` — full metrics.
- `scripts/artifacts/oof_d9h_FM_aug12_strat.npy` — base predictions.
- `submissions/submission_d9h_S1_K20_swap_aug12.csv` — HELD (−0.31bp).
- `submissions/submission_d9h_S2_K22_add_aug12.csv` — HELD (+0.01bp,
  ρ in tie band).
