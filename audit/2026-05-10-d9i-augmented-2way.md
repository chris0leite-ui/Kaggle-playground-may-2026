# Day-9i — Augmented 2-way multi-FM (FM family closed)

> Hypothesis: combine d9f's partition-diversity win with d9h's
> feature-augmentation win — two FMs each on 6 features instead of
> 4. Falsified.
> Builder `scripts/d9i_multi_fm_aug.py` (~150s wall).

## Augmented partition design

  FM_A_aug "driver+degradation": D, C, S, T_q5, **Cd, Ld**  (6 feat, 15 pairs)
  FM_B_aug "race+neighbor":      R, Y, Rp_q5, P_q5, **Nx, Pv**  (6 feat, 15 pairs)

Same FM hyperparameters as d9c/d9f (k=8, 6 epochs, batch 8192,
lr=0.05).

## Standalone — strength up, diversity down

| Side | d9f (4 feat) | d9i_aug (6 feat) | Δ std | ρ vs PRIMARY |
|---|---:|---:|---:|---:|
| FM_A | 0.82505 | **0.88123** | **+5.6bp** | 0.487 → 0.720 (worse) |
| FM_B | 0.88438 | **0.88561** | +0.1bp | 0.861 → 0.863 (flat) |
| ρ A vs B | 0.406 | **0.663** | **+0.26** (worse) | — |

The augmented partition gains standalone strength on FM_A
(degradation features Cd+Ld add ~6bp). But the ρ-diversity collapses
in two directions simultaneously:
- ρ between the two FMs jumps from 0.406 → 0.663.
- ρ vs PRIMARY for FM_A jumps from 0.487 → 0.720.

The richer features pull both FMs toward the GBDT consensus and
toward each other.

## K=N stack experiments

PRIMARY = d9f K=21 swap+multi-FM (Strat OOF 0.95073, LB 0.95031).

| Stack | K | OOF | Δ PRIMARY | ρ |
|---|---:|---:|---:|---:|
| S1 K=21 swap (replace d9f 2-way with d9i aug 2-way) | 21 | 0.95071 | **−0.19bp** | 0.99973 |
| S2 K=23 add (keep d9f + add d9i aug) | 23 | 0.95073 | **−0.01bp** | 0.99987 |

In S1, the augmented FMs **fail to enter L1 top-15** despite being
+5.6bp stronger standalone than the d9f pair. In S2, where both
pairs coexist, the augmented FMs also fail to add measurable lift.

## FM family is now exhausted — d9f is local optimum

Eight FM-family experiments since d9c FM landed at LB 0.95029:

| Variant | Δ vs prior PRIMARY OOF | LB Δ | Verdict |
|---|---:|---:|---|
| d9c FM unified 8-feat | +0.13bp | **+3.0bp** | ✓ banked (hedge) |
| d9d FM sweep + bag | flat | (held) | TIE |
| d9e FFM (richer params) | regressed | (held) | dead |
| **d9f 2-way 4+4** | **+0.30bp** | **+2.0bp** | **✓ NEW PRIMARY** |
| d9g 3-way 3+2+3 | regressed | (held) | dead |
| d9h aug-12 unified | flat | (held) | dead |
| **d9i aug 2-way 6+6** | **regressed** | (held) | **dead** |

## Mathematical structure of the FM sweet spot

The d9f 2-way partition simultaneously satisfies three constraints
that all other FM variants violate at least one of:

1. **Per-FM strength threshold**: each FM has ≥ 4 features → ≥ 6
   pairwise interactions. Below this threshold (d9g's 2-feature
   FM_β, 3-way's per-FM averages) the FM degenerates to near-LR.
2. **Inter-FM diversity**: ρ between partitioned FMs ≤ 0.5. Above
   this (d9i aug 2-way at 0.663), the LR meta routes through one
   FM only and demotes the other.
3. **Per-FM diversity vs PRIMARY**: at least one FM has ρ vs the
   GBDT pool < 0.6. d9f's FM_A (ρ=0.487) carries this; adding
   features moves it past the threshold (d9i FM_A_aug at 0.720).

Adding features to a partition member helps constraint (1) but
hurts (2) and (3). Splitting into more partitions helps (2) and
(3) but hurts (1). The Pareto-optimal point is **2 partitions ×
4 features each** — no other configuration beats it on the meta-
weighted product.

## What's next

The FM family at the single-feature-set level is **closed**. Any
further FM-class improvement requires either:

1. **External features** that genuinely shift the per-row prediction
   direction (Pirelli pit-windows, SC probability per track) — only
   the parallel agent's d12 Pirelli scrape addresses this.
2. **A different model class** (DeepFM-lite, EmbMLP, hazard-NN done
   correctly, TabM v3-extended) that adds non-FM predictions.
3. **A different meta** (Bayesian hierarchical stacker, LGBM-meta
   with rank+logit features) that routes information differently.

The remaining cheap-CPU experiments within FM family are dead.
d9f K=21 swap (LB 0.95031) is held as PRIMARY.

## Pointers

- `scripts/d9i_multi_fm_aug.py` — builder + S1/S2 stacks.
- `scripts/artifacts/d9i_aug2way_results.json` — full metrics.
- `scripts/artifacts/oof_d9i_FM_A_aug_strat.npy`,
  `scripts/artifacts/oof_d9i_FM_B_aug_strat.npy` — augmented bases.
- `submissions/submission_d9i_S1_K21_swap_aug2way.csv`,
  `submissions/submission_d9i_S2_K23_add_aug2way_to_d9f.csv` — HELD.
