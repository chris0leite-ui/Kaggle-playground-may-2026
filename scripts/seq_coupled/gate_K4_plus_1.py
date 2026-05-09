"""K=4+1 plain LR-meta gate for the V4 kNN-augmented base.

Compares two meta variants:
  1. K=4 only [P, rank, logit] = 12 features → LR meta (anchor)
  2. K=4 + V4 kNN-aug base [P, rank, logit] = 15 features → LR meta

Gate verdict:
  PASS: Δ ≥ +0.5 bp on OOF
  WEAK: 0 ≤ Δ < +0.5 bp
  FAIL: Δ < 0 bp

Output: scripts/artifacts/probe_k4_plus_v4_gate.json
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


def main():
    tr = pd.read_csv('data/train.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)
    bases_K4 = {
        'h1d':  pos(ART / 'oof_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'oof_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'oof_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'oof_d16_orig_continuous_only_strat.npy'),
    }
    v4 = pos(ART / 'oof_knn_aug_base.npy')
    print(f'V4 standalone AUC: {roc_auc_score(y, v4):.5f}')
    bases_K5 = {**bases_K4, 'v4_knn_aug': v4}
    pearson_v4_vs_K4 = {}
    for k, v in bases_K4.items():
        rho = float(spearmanr(v4, v).statistic)
        pearson_v4_vs_K4[k] = rho
    print('V4 vs K=4 base ρ_spearman:')
    for k, r in pearson_v4_vs_K4.items():
        print(f'  {k}: {r:.5f}')

    X_K4 = expand(bases_K4)              # 12 feat
    X_K5 = expand(bases_K5)              # 15 feat
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    oof_K4 = np.zeros(n, dtype=np.float32)
    oof_K5 = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        oof_K4[vai] = fit_lr(X_K4[tri], y[tri], X_K4[vai])
        oof_K5[vai] = fit_lr(X_K5[tri], y[tri], X_K5[vai])
        print(f'  fold {k}: K=4 {roc_auc_score(y[vai], oof_K4[vai]):.5f}; '
              f'K=5 {roc_auc_score(y[vai], oof_K5[vai]):.5f}; '
              f'Δ={(roc_auc_score(y[vai], oof_K5[vai]) - roc_auc_score(y[vai], oof_K4[vai]))*1e4:+.2f} bp')
    auc_K4 = float(roc_auc_score(y, oof_K4))
    auc_K5 = float(roc_auc_score(y, oof_K5))
    delta_bp = (auc_K5 - auc_K4) * 1e4
    rho_oof = float(spearmanr(oof_K5, oof_K4).statistic)
    verdict = ('PASS' if delta_bp >= 0.5 else
               'WEAK' if delta_bp >= 0 else 'FAIL')
    out = {
        'oof_K4_only_lr_meta':       auc_K4,
        'oof_K5_with_v4_lr_meta':    auc_K5,
        'delta_bp':                  delta_bp,
        'verdict':                   verdict,
        'rho_oofK5_vs_oofK4':        rho_oof,
        'v4_standalone_oof_auc':     float(roc_auc_score(y, v4)),
        'v4_vs_K4_base_spearman':    pearson_v4_vs_K4,
    }
    (ART / 'probe_k4_plus_v4_gate.json').write_text(json.dumps(out, indent=2))
    print('\n=== K=4+1 GATE SUMMARY ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
