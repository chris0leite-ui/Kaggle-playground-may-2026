# Day-9d — FM hyperparameter sweep + 3-seed bag

> Day-9c FM landed at LB 0.95029 (+3bp lift, NEW PRIMARY). Two cheap
> follow-ups from the d9c "what next" plan: (1) hyperparameter sweep
> over embed_dim / weight decay / epochs; (2) 3-seed bag of the
> winner. Builder `scripts/d9d_fm_sweep_bag.py` (749s wall total).

## Sweep — single-FM variants

| Variant | Δ vs k=8 | Std OOF | ρ vs new PRIMARY | Min-meta Δ |
|---|---:|---:|---:|---:|
| v0_k4 (lower rank) | embed=4 | 0.91926 | 0.891 | −0.05bp |
| **v1_k8_baseline** | (= d9c) | **0.92069** | 0.894 | −0.06bp |
| v2_k16 (higher rank) | embed=16 | 0.91929 | 0.891 | −0.07bp |
| v3_k8_wd1e5 | weight_decay=1e-5 | 0.92069 | 0.894 | −0.06bp |
| v4_k8_ep10 | epochs=10 | 0.92066 | 0.891 | −0.06bp |
| **bag3_seeds** | k=8 × 3 seeds | **0.92253** | 0.895 | −0.07bp |

Findings:
- **k=8 is the local sweet spot.** k=4 underfits (−0.14bp), k=16
  overfits (−0.14bp). The interaction surface is well-fit at rank 8.
- **Weight decay on the dense bias has no effect** (v3 ≡ v1). Sparse
  embeddings receive no L2; the bias is dominated by data terms.
- **Longer training plateaus.** v4 (10 epochs) ≈ v1 (6 epochs).
  Loss curves earlier indicated convergence by epoch 4.
- **3-seed bag adds +1.84bp std OOF** (0.92069 → 0.92253) — meaningful
  variance reduction.
- **All variants FAIL min-meta vs new PRIMARY** because the new PRIMARY
  already contains a k=8 FM in its 20-base pool; adding another FM
  variant is double-counting.

## K=20 swap stacks — replace PRIMARY's FM with each variant

| Stack | K | OOF | Δ vs PRIMARY | ρ vs PRIMARY |
|---|---:|---:|---:|---:|
| stack_v0_k4 | 20 | 0.95071 | **+0.08bp** | 0.99994 |
| stack_v1_k8 (replicates PRIMARY) | 20 | 0.95070 | +0.03bp | 1.00000 |
| stack_v2_k16 | 20 | 0.95068 | −0.17bp | 0.99995 |
| stack_v3_k8_wd1e5 | 20 | 0.95069 | −0.10bp | 0.99998 |
| stack_v4_k8_ep10 | 20 | 0.95069 | −0.08bp | 0.99996 |
| stack_bag3_seeds | 20 | 0.95069 | −0.13bp | 0.99988 |

**Surprising finding**: v0_k4 (lower rank, weaker as a single base)
gives the **best K=20 OOF** (+0.08bp). The bag (strongest single base
at 0.92253) gives a *worse* K=20 OOF (−0.13bp) than the single k=8.

### Mechanism

Stacking optimizes for *uncorrelated error*, not raw strength.
Bagging across 3 seeds of the *same FM architecture* smooths the
predictions toward the bias surface they share, which is *closer* to
the LR meta's other smooth bases. Lower variance per row → less
diversity for the meta to route around. The bag's std OOF gain comes
from variance reduction in the regions where all 3 seeds agree, but
the meta already had an FM with similar predictions.

v0_k4's slightly different inductive bias (rank-4 manifold vs rank-8)
gives the LR meta a different *direction* of error — more useful to
the stack than v1_k8's smaller residual variance.

## Submission analysis

All d9d K=20 stacks have **ρ > 0.99988 vs new PRIMARY**:
- 5-decimal LB quantization noise floor ≈ 0.5bp / 1bp band.
- All d9d stacks fall well within the TIE band; predicted LB lift
  zero or negative for all but v0_k4.
- v0_k4 stack: pred-LB Δ +0.08bp (within quantization noise).

**No d9d variant offers a slot-worthy improvement over PRIMARY.**

## Triage decision

**HOLD** all d9d candidates. The hyperparameter neighborhood of the
d9c FM is flat — there's no slot-worthy lift available by tweaking
k / wd / epochs / seeds within the same feature set.

To meaningfully extend the FM mechanism, the next-step probes need to
add **structurally different FMs**, not the same one tuned:
1. **FM with different feature partition**: e.g., split categoricals
   into 2 fields with separate embedding tables (proto-FFM); or use
   different binned numerics (8-quantile, 12-quantile).
2. **Field-aware FM (FFM)**: per-field-pair embedding tables. ~2h
   CPU implementation. Predicted +0.5–1.5bp std OOF, distinct error
   structure.
3. **DeepFM-lite**: FM + a 1-hidden-layer MLP head sharing the same
   embeddings. Adds non-linearity in the same low-rank space.
4. **Multi-FM ensemble**: 2 FMs with disjoint feature subsets, both
   added to K=22 stack. Tests whether the LR meta can route 2
   independent FMs.

Each of these takes 1–4 hours CPU; only worth doing once #1 (the
cheapest) is shown to add ρ-orthogonality vs PRIMARY's FM.

## Pointers

- `scripts/d9d_fm_sweep_bag.py` — combined sweep + bag + K=20 stack
  experiments.
- `scripts/artifacts/d9d_fm_sweep_results.json` — all metrics.
- `scripts/artifacts/oof_d9d_v*_strat.npy` — per-variant FM bases.
- `scripts/artifacts/oof_d9d_bag3_strat.npy` — 3-seed bag.
- `submissions/submission_d9d_K20_swap_v0_k4.csv` — best d9d stack
  (HELD; pred-LB Δ +0.08bp, in TIE band).
- `submissions/submission_d9d_K20_swap_bag3_seeds.csv` — bag stack
  (HELD; pred-LB Δ −0.13bp).
