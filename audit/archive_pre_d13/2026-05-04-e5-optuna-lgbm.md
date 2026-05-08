# E5 — Optuna-tuned LGBM (D3 prep)

30 trials in 2005s. Best 1-fold AUC: 0.94867.
Best params: {'lr': 0.017739639246790825, 'num_leaves': 229, 'min_data_leaf': 103, 'feature_frac': 0.6575935831140413, 'bagging_frac': 0.7647859924409988, 'bagging_freq': 4, 'l1': 3.819433130462022e-05, 'l2': 0.00019363249015436128, 'max_depth': 10}

## 5-fold both-anchor

| anchor | OOF AUC | Δ baseline |
|---|---:|---:|
| Strat | **0.94736** | +66.1bp |
| GroupKF | **0.92585** | +52.6bp |
