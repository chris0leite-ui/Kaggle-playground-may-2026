# Day-5 — TabNet 1-fold smoke FAILS gate

Path A.2 from HANDOVER. 1-fold smoke per Rule 2 before any 5-fold
GPU commit. Result: gate failed by ~12bp.

## Result

| Quantity | Value | Threshold | Verdict |
|---|---:|---:|---|
| Fold-0 val AUC | 0.93532 | ≥ 0.945 | **FAIL by 12bp** |
| Δ baseline (0.94075) | **−54.3bp** | ≥ 0 | **FAIL** |
| Δ RealMLP fold-0 (0.94722) | **−119.0bp** | ~ neutral | weak |
| Fold-0 wall | 711s (11.9min) | — | — |
| 5-fold projection | **59.3min** | < 60 min | borderline (Rule 2) |

## Why it failed (training trajectory from kernel log)

```
epoch  0 : val_auc 0.879
epoch 50 : val_auc 0.924
epoch 90 : val_auc 0.932
epoch118 : val_auc 0.935  ← BEST, stopped at max_epochs=120
```

Model was **still climbing** when max_epochs hit. Patience=15 never
triggered. TabNet under-trained at the smoke config — would need
~200-300 epochs to find its plateau. With 5-fold projection already
at 59min on 120 epochs, doubling epochs blows the wall budget at
~2h on Kaggle GPU.

## Falsified

- **TabNet at default pytorch-tabnet config (n_d=32, n_a=32, n_steps=5,
  cat_emb_dim=4, lr=2e-2, max_epochs=120) is competitive with RealMLP
  on this data** → null. Plateau is ~0.935; RealMLP plateau is ~0.947.

## Likely root causes (not pursued — see "decision")

1. **Driver cat (cardinality 887) under-embedded.** `cat_emb_dim=4` is
   too small for 887 unique drivers; RealMLP-TD uses adaptive embedding
   sizes that grow with cardinality.
2. **Sparsemax masking + small batches.** TabNet often needs
   self-supervised pretraining to find good masks; we skipped that.
3. **No early-stopping signal triggered.** Patience over a slowly
   climbing curve never fires; learning rate schedule was too aggressive
   (StepLR step=20, gamma=0.9 → lr decays before plateau is reached).

## Decision: do NOT pursue 5-fold TabNet

Per HANDOVER falsified list: "Hand-crafted FE specifically for the NN
branch" — RealMLP's internal embeddings re-derive most signals. Same
likely holds for tuned TabNet. EV calculus per Rule 12 + R8 PI directive
("bigger moves only, multi-bp, stop sub-1bp tuning"):

- Best-case tuned TabNet: maybe ρ=0.97 vs RealMLP, +0.5-1.5bp on M5q
  pool (the rank-lock confirmed today says even that's optimistic).
- Cost: 2-4h GPU, 2-3 retries to find a config that converges.
- vs Path B (pseudo-label, 30bp-class ceiling) or A.1 (RealMLP seed
  bag, +1-3bp prior at known cost): both better EV.

Park TabNet behind Path A.1 + A.3 (FT-Transformer); revisit only if
those land and the "multiple NN families" thesis is validated.

## Held artifacts

- `/tmp/tabnet-smoke-out/tabnet_smoke_results.json` — fold-0 metrics
- `/tmp/tabnet-smoke-out/tabnet-smoke-strat.log` — full training log

End — 55 lines.
