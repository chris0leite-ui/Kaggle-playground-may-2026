# Day-2 probe #1 — external-dataset join (2026-05-04)

Original dataset: `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`
Original shape: (101371, 16);  s6e5 train: (439140, 16);  s6e5 test: (188165, 15)

## Match rate by key

| key | n_cols | train_match | train_target_agree | test_match |
|---|---:|---:|---:|---:|
| `Driver+Race+Year+LapNumber` | 4 | 0.0557 | 0.7429 | 0.0556 |
| `Driver+Race+Year+LapNumber+Compound` | 5 | 0.0427 | 0.7696 | 0.0426 |
| `Driver+Race+Year+LapNumber+Compound+Stint+TyreLife` | 7 | 0.0059 | 0.9486 | 0.0055 |
| `Driver+Race+Year+LapNumber+Stint+TyreLife+Position` | 7 | 0.0018 | 0.9558 | 0.0017 |

## Verdict

**JOIN MISSES** — best test match rate is 0.0556 (< 10% threshold). The host shuffled or synthesized rows beyond the original. Move on to probe #2 (DGP-rule probe).

`Normalized_TyreLife` (host-removed) recoverable on 5.56% of test rows via the same key. **Host explicitly forbade reintroducing this column** (brief.md). Do NOT use it.
