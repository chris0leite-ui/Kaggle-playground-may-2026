"""K=4 / K=5 / K=6 gates for V6 base.

Mirrors gate_v5.py but for V6 (embedding-kNN base).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ART = Path('scripts/artifacts')
SEED, N_FOLDS = 42, 5


def pos(p):
    a = np.load(p)
    return a[:, 1] if a.ndim == 2 else a


def expand(P_dict):
    parts = []
    for v in P_dict.values():
        parts.append(v)
        parts.append(rankdata(v) / len(v))
        parts.append(np.log(np.clip(v, 1e-7, 1-1e-7) /
                            np.clip(1-v, 1e-7, 1-1e-7)))
    return np.column_stack(parts).astype(np.float32)


def fit_lr(X_tr, y_tr, X_va):
    sc = StandardScaler().fit(X_tr)
    Xt, Xv = sc.transform(X_tr), sc.transform(X_va)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    lr.fit(Xt, y_tr)
    return lr.predict_proba(Xv)[:, 1]


def cv_oof(X, y, splits):
    n = len(y)
    oof = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        oof[vai] = fit_lr(X[tri], y[tri], X[vai])
    return oof


def main():
    tr = pd.read_csv('data/train.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)

    K4 = {
        'h1d':  pos(ART / 'oof_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'oof_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'oof_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'oof_d16_orig_continuous_only_strat.npy'),
    }
    v4 = pos(ART / 'oof_knn_aug_base.npy')
    v6 = pos(ART / 'oof_v6_base.npy')

    print(f'V6 standalone OOF AUC: {roc_auc_score(y, v6):.5f}')
    print(f'V4 standalone OOF AUC: {roc_auc_score(y, v4):.5f}')
    print(f'ρ_spearman V6 vs V4: {spearmanr(v6, v4).statistic:.5f}')
    for k_, v in K4.items():
        print(f'ρ_spearman V6 vs {k_}: {spearmanr(v6, v).statistic:.5f}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    X_K4    = expand(K4)
    X_K4_v6 = expand({**K4, 'v6': v6})
    X_K4_v4 = expand({**K4, 'v4': v4})
    X_K6    = expand({**K4, 'v4': v4, 'v6': v6})

    print('\n=== A. K=4 only LR meta ===')
    oof_A = cv_oof(X_K4, y, splits)
    auc_A = float(roc_auc_score(y, oof_A))
    print(f'  OOF = {auc_A:.5f}')

    print('\n=== B. K=4 + V6 (V6 replaces V4) ===')
    oof_B = cv_oof(X_K4_v6, y, splits)
    auc_B = float(roc_auc_score(y, oof_B))
    delta_B = (auc_B - auc_A) * 1e4
    print(f'  OOF = {auc_B:.5f}  Δ vs K=4 = {delta_B:+.2f} bp')

    print('\n=== C. K=5 (K=4 + V4) ===')
    oof_C = cv_oof(X_K4_v4, y, splits)
    auc_C = float(roc_auc_score(y, oof_C))
    delta_C = (auc_C - auc_A) * 1e4
    print(f'  OOF = {auc_C:.5f}  Δ vs K=4 = {delta_C:+.2f} bp')

    print('\n=== D. K=6 (K=4 + V4 + V6) ===')
    oof_D = cv_oof(X_K6, y, splits)
    auc_D = float(roc_auc_score(y, oof_D))
    delta_D_vs_C = (auc_D - auc_C) * 1e4
    print(f'  OOF = {auc_D:.5f}  Δ vs K=5 = {delta_D_vs_C:+.2f} bp')

    out = {
        'A_K4_only':                auc_A,
        'B_K4_plus_V6':             auc_B,
        'C_K5_K4_plus_V4':          auc_C,
        'D_K6_K4_plus_V4_plus_V6':  auc_D,
        'delta_B_vs_A_bp':          delta_B,
        'delta_C_vs_A_bp':          delta_C,
        'delta_D_vs_C_bp':          delta_D_vs_C,
        'rho_V6_vs_V4':             float(spearmanr(v6, v4).statistic),
        'v6_standalone':            float(roc_auc_score(y, v6)),
        'v4_standalone':            float(roc_auc_score(y, v4)),
    }
    (ART / 'probe_v6_gate.json').write_text(json.dumps(out, indent=2))
    print('\n=== V6 GATE ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
