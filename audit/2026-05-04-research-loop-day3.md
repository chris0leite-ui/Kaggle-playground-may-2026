# Day-3 Research-loop — what the leader is doing that we are NOT

Mission: 0.94891 → 0.95345 top-5% (45bp) / 0.95435 leader (54bp). Day-1
already covered F1-dataset notebooks (yekenot, analyticaobscura,
kospintr, pilkwang) + S4E1/S3E23. This pass = updated cross-comp
delta + community signal triangulation.

## 1. Top public notebooks for s6e5

**Access constraint**: Kaggle CLI returned `401 Unauthorized` (token
likely scope-limited to the active comp); WebFetch on Kaggle pages
returns title-only (JS gating) for *all* `kaggle.com/competitions/.../code`
and `/discussion` URLs. Top-by-votes ranking therefore could NOT be
retrieved fresh today. The Day-1 list is still the best evidence:
1. **`yekenot/ps-s6-e5-realmlp-pytabkit`** — 56 votes, OOF≈0.946
   single RealMLP. Already integrated as M5b base (HGBC stand-in).
2. **`analyticaobscura/pit-or-stay-f1-strategy-1`** — 34 votes, target
   0.9550+. 5-model OOF stack + Dirichlet + LR-meta. Fully ingested.
3. **`kospintr/pitstop-catb-hgbc-xgb-lgbm-realmlp-baseline`** — 36
   votes, mean blend over 5 bases. Subset of M5b.
4. **`sarcasmos/pit-stop-prodigy`** — 37 votes, EDA only.
5. **`pilkwang/Driver's High`** — 41 votes, DGP-faithful FE warning.

No fresh `>0.95 LB` public notebook was discoverable via web search
on s6e5 today. The top-5% has not yet leaked methodology publicly.

## 2. Recent discussion threads

Kaggle discussion-list endpoints returned title-only too. `kaggle
competitions discussions` is not a CLI verb in 2.1.0. **Discussion
content (not accessible)** for s6e5 specifically. Web search surfaced
zero indexed forum threads with content for this comp — likely
because forum index lag is >2 days and the comp opened ~Day 0–1 ago
relative to indexing.

## 3. Cross-comp deltas — five mechanisms ranked

Sources: NVIDIA Grandmaster cuML stacking blog (s5e8 1st),
NVIDIA Grandmaster Playbook (7 techniques), s6e4 "Error Diversity
Matters: 200-model" writeup (title-only, name = mechanism), s5e12
1st "Hill Climbing + Ridge", s5e5 "GPU Hill Climbing" (Chris Deotte),
s6e1 2nd "NNs sometimes work better than GBMs".

Predicted lift = midpoint of cited gain band; cost on 2-fold full
data probe; rank = lift × CPU-feasibility.

| # | Mechanism | Citation | Predicted lift | Cost | Why we missed it |
|---|---|---:|---:|---|---|
| 1 | **Problem-reformulation diversity** — train base models on (a) target, (b) target/race-length ratio, (c) residuals from a linear baseline, (d) impute-then-predict, (e) pseudo-labels on test. Stack the 5×base-arch grid. | NVIDIA cuML blog (s5e8 1st, 75-of-500 model selection); s6e4 "Error Diversity Matters" writeup title | 25–60bp | 4–6h CPU (5×3 GBDTs) | We diversified over **architecture** (LGBM/XGB/Cat/HGBC/RealMLP) but not over **objective**. All our 7 bases learn the same target. |
| 2 | **Pseudo-labeling on high-confidence test rows** | Grandmaster Playbook #6; cdeotte `Pseudo Labeling QDA 0.969` | 10–30bp | 2h CPU (LGBM round-2) | Never tried on s6e5 — 188k test rows × 0.94891 prob → ~80k high-confidence. Soft-label (probabilities) variant only, k-fold-safe. |
| 3 | **Hill-climbing + Ridge ensemble** (replace LR-meta) | s5e12 1st place title; Matt-OP `hillclimbers` lib; s5e5 Chris Deotte | 5–20bp | 30min CPU | We use LR-meta on `[raw,rank,logit]`. Hill-climb selects a *sparse* subset (Lasso-like) and Ridge adds shrinkage; less prone to OOF-noise overfit than LR over 7 correlated bases. Drop-in. |
| 4 | **Multi-seed bagging + 100% retrain** | Grandmaster Playbook #7 (charts ~30bp gain on s5e8) | 10–30bp | 2h CPU (5 seeds × 5 folds for our top-2 bases only) | We run seed=42 only. Day-1 didn't surface this; Playbook says NN gains more from seeds, GBDT gains less, but charted ~0.3% on s5e8. CPU-feasible if scoped to top-2 bases. |
| 5 | **OOF target encoding on 2-way interactions** (Driver×Race, Driver×Compound, Race×Lap-bin) with smoothing α=80 | analyticaobscura Source 1 #2 (Day-1 notes); recurring in s5e11 1st | 10–25bp | 1h CPU | We have D2-A single-col TE (NULL on G1) but never tried 2-way. Day-1 queued it as "TE-Driver-Race-only"; never executed. |

## 4. Gap analysis — what's the leader probably doing?

Leader at 0.95435, us at 0.94891, gap 54bp. Triangulating from
NVIDIA cuML (75-of-500 stack), s6e4 "200-model" title, s5e12 (hill
climb + ridge), Grandmaster Playbook (pseudo-labeling, multi-seed),
and the F1-dataset notebook ladder (RealMLP/EmbMLP critical):
**most likely composition** = 15–30 base GBDTs spanning 5 problem
reformulations (target / ratio / residual / imputation / pseudo-label)
× 3 architectures (LGBM/XGB/Cat) × 2 seeds, **plus 2–3 RealMLP/EmbMLP
variants**, blended via **hill-climb + Ridge** (not LR-meta) on OOF
`[raw,rank,logit]`. Likely also runs **soft pseudo-labels on the
~30% high-confidence test rows** — this is the cheapest 15–25bp lift
that we've never probed and that matches the gap-budget arithmetic
(items 1+2 alone span the 54bp gap at midpoints).

## Actionable handoff

D3 PRIMARY: probe Mechanism #1 (problem-reformulation, 1 reformulation
at a time) + Mechanism #2 (pseudo-label) under standard 4-gate filter.
D3 HEDGE: Mechanism #3 hill-climb-Ridge as a stacker swap on existing
M5b OOF (zero new-base cost). All four are CPU-feasible.
