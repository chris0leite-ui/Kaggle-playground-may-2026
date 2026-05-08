# 2026-05-11 — Day-11: TabM v3 extended training confirmed DEAD

> Path A from HANDOVER Day-10: extended training (n_epochs ≥200,
> patience ≥50) to test whether v2's epoch-5 stop was under-training.
> Result: extending training made fold-0 AUC **WORSE**. TabM is now
> falsified at both default (v2) and extended-training (v3) configs.
> Pivot to Path B (G4 SCARF) or Path C (F2 multi-rule rebuild).

## Result table

| Run | n_epochs | patience | lr | Best val epoch | Fold-0 AUC | vs gate (0.945) | vs v2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 (Day-9) | default ~25 | default | default ~1e-3 | 5 | 0.94039 | −46bp | — |
| **v3 (Day-11)** | **200** | **50** | **3e-4** | **11** | **0.93926** | **−52bp** | **−11.3bp** |

`applied_kwargs` confirms all three v3 knobs took effect (pytabkit
TabM_D_Classifier accepts `n_epochs`, `patience`, `lr` directly).
Source = pytabkit. Wall = 881s fold-0 → 73 min 5-fold projection
(also breaks the wall_gate for any 5-fold; v2 was 27 min PASS).

## Training trajectory diagnostic

From `tabm-smoke-v3-strat.log`:

```
epoch 11: val -0.4254  ← BEST
... 189 epochs of slow drift, NEVER improving on epoch 11
epoch 200: val -0.4282  (drift ~+0.003 in negative-CE, monotone worse)
```

Patience=50 never tripped because pytabkit's "no improvement in N
epochs" logic only flags moves outside a tolerance band; the slow
drift stayed inside the band. Effectively, the model trained 189
useless epochs.

**Implication**: TabM-D converges in ~10 epochs on this DGP and
DEGRADES MONOTONICALLY thereafter. The architecture is the bound,
not the training duration.

## Mechanism finding (load-bearing for dead-list)

Two independent failure modes confirmed within the same arch:

1. **v2 (default config)**: epoch-5 stop, AUC 0.94039 — gate FAIL.
2. **v3 (extended training, lower lr)**: epoch-11 best, then drift.
   Final AUC 0.93926 — worse than v2 by 11.3bp.

Combined: **no plausible training schedule rescues TabM-D on this
problem.** The only un-tested lever is HPO sweep over architecture
hyperparameters (depth, width, k_heads, embedding sizing for the
887-cardinality Driver). Per Day-9 audit, that's a 6h GPU sweep
whose EV is downgraded because RealMLP-TD already covers the
MLP-with-embedding mechanism class. **Not worth the slot.**

## What this means for the strategic menu

Eight prior nulls + Day-9 TabM v2 + Day-9/10 hazard NN + Day-11
TabM v3 = **eleven nulls in eleven days** since K=18 PRIMARY landed
on Day-6.

The pattern: every base-pool addition collapses to ρ ≥ 0.99 vs K=18
unless its target/feature construction is structurally outside the
pool's hypothesis class. TabM is in the pool's hypothesis class
(MLP + embeddings = same class as RealMLP). **Mechanism-class-only
filter sharpens further.**

Survivable mechanism classes (post-Day-11):
1. **G4 SCARF/VIME pretrain on aadigupta1601 unlabeled** (different
   unlabeled corpus avoids d5 partial-pseudo failure mode; different
   inductive bias = contrastive pretraining). 6-10h T4.
2. **F2 multi-rule rebuild with Q6 enforcement** (rules selected by
   ρ-orthogonality vs the FULL K=18 pool, not just M5q). Cheap CPU.
   Predicted-NULL by analogy with C5/C1 but cheap to falsify.
3. **External-data Q1 Pirelli pit-windows** (HANDOVER C2). 6-8h
   scrape + 2h CPU. Different shape of external info than C1.

DEAD-list additions:
- **TabM-D extended training (200 epochs, lr=3e-4)** — Day-11 v3.

## Artifacts

- `kernels/tabm-smoke-v3-gpu/tabm_smoke_v3_gpu.py` — kernel
- `kernels/tabm-smoke-v3-gpu/kernel-metadata.json`
- `scripts/artifacts/tabm_smoke_v3_results.json` — result JSON
- `scripts/artifacts/tabm-smoke-v3-strat.log` — training log
- Kaggle: <https://www.kaggle.com/code/chrisleitescha/tabm-smoke-v3-strat>

## Recommended next move

**Path C (F2 multi-rule rebuild) first** — cheap CPU, ~2-3h, would
either (a) confirm Q6 is the binding constraint (predicted) or (b)
yield a +1-3bp K=N base, in which case it's the easiest win on
the menu. Then **Path B (G4 SCARF)** as the next GPU bet given
TabM is now closed.

PRIMARY unchanged: `d6_k18_multi_rule` LB **0.95026**, gap −3.9bp.
0/10 submits used today.
