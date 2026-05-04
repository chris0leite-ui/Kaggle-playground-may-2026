# D2-C — concat-original-data baseline (2026-05-04)

External: aadigupta1601 dataset (101,371 rows, `Normalized_TyreLife` dropped per host rule). Concatenated to s6e5 train; OOF computed on s6e5 train rows only.

## Results

| anchor | OOF AUC | fold_std | per-fold | Δ vs baseline_two_anchor |
|---|---:|---:|---|---:|
| A — StratKFold | **0.94301** | 0.00076 | ['0.9440', '0.9420', '0.9432', '0.9423', '0.9437'] | +22.6bp |
| B — GroupKFold(Race) | 0.93915 | 0.00869 | ['0.9353', '0.9289', '0.9285', '0.9475', '0.9483'] | +185.6bp |

