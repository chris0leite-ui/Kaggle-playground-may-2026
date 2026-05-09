"""K=4 / K=5 / K=6 gates for V5 base.

Tests three configurations:
  A. K=4 only LR meta                              (anchor)
  B. K=4 + V5 LR meta (V5 replaces V4)             — does V5 alone beat V4?
  C. K=5 (K=4 + V4) LR meta                        (current PRIMARY plain LR)
  D. K=6 (K=4 + V4 + V5) LR meta                   — does V5 stack-add?

Decision rules:
  - If B beats A by ≥+0.5 bp AND ≥+0.3 bp over C, V5 SWAPS V4 in PRIMARY
  - If D beats C by ≥+0.5 bp, V5 ADDS to current K=5 → K=6 PRIMARY
  - Otherwise V5 has no production lift.

Output: scripts/artifacts/probe_v5_gate.json
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

    K4_oof = {
        'h1d':  pos(ART / 'oof_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'oof_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'oof_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'oof_d16_orig_continuous_only_strat.npy'),
    }
    v4 = pos(ART / 'oof_knn_aug_base.npy')
    v5 = pos(ART / 'oof_v5_base.npy')

    # Diversity: ρ_spearman vs K=4 + V4
    rho_v5_v4 = float(spearmanr(v5, v4).statistic)
    rho_v5_K4 = {k: float(spearmanr(v5, v).statistic) for k, v in K4_oof.items()}
    print(f'V5 standalone OOF AUC: {roc_auc_score(y, v5):.5f}')
    print(f'V4 standalone OOF AUC: {roc_auc_score(y, v4):.5f}')
    print(f'ρ_spearman V5 vs V4: {rho_v5_v4:.5f}')
    for k, r in rho_v5_K4.items():
        print(f'ρ_spearman V5 vs {k}: {r:.5f}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    X_K4 = expand(K4_oof)                                          # 12
    X_K4_v5 = expand({**K4_oof, 'v5': v5})                         # 15
    X_K4_v4 = expand({**K4_oof, 'v4': v4})                         # 15  (= current K=5)
    X_K6 = expand({**K4_oof, 'v4': v4, 'v5': v5})                  # 18

    print('\n=== A. K=4 only LR meta (anchor) ===')
    oof_A = cv_oof(X_K4, y, splits)
    auc_A = float(roc_auc_score(y, oof_A))
    print(f'  OOF AUC = {auc_A:.5f}')

    print('\n=== B. K=4 + V5 LR meta (V5 replaces V4) ===')
    oof_B = cv_oof(X_K4_v5, y, splits)
    auc_B = float(roc_auc_score(y, oof_B))
    delta_B_vs_A = (auc_B - auc_A) * 1e4
    print(f'  OOF AUC = {auc_B:.5f}  Δ vs K=4 = {delta_B_vs_A:+.2f} bp')

    print('\n=== C. K=5 (K=4 + V4) LR meta (current PRIMARY plain LR) ===')
    oof_C = cv_oof(X_K4_v4, y, splits)
    auc_C = float(roc_auc_score(y, oof_C))
    delta_C_vs_A = (auc_C - auc_A) * 1e4
    print(f'  OOF AUC = {auc_C:.5f}  Δ vs K=4 = {delta_C_vs_A:+.2f} bp')

    print('\n=== D. K=6 (K=4 + V4 + V5) LR meta ===')
    oof_D = cv_oof(X_K6, y, splits)
    auc_D = float(roc_auc_score(y, oof_D))
    delta_D_vs_A = (auc_D - auc_A) * 1e4
    delta_D_vs_C = (auc_D - auc_C) * 1e4
    print(f'  OOF AUC = {auc_D:.5f}  Δ vs K=4 = {delta_D_vs_A:+.2f} bp  '
          f'Δ vs K=5 = {delta_D_vs_C:+.2f} bp')

    # Decisions
    swap_v4_v5 = (delta_B_vs_A >= 0.5 and (auc_B - auc_C) * 1e4 >= 0.3)
    add_v5_to_K5 = delta_D_vs_C >= 0.5

    out = {
        'A_K4_only_oof_auc':           auc_A,
        'B_K4_plus_V5_oof_auc':        auc_B,
        'C_K5_oof_auc':                auc_C,
        'D_K6_oof_auc':                auc_D,
        'delta_B_vs_A_bp':             delta_B_vs_A,
        'delta_C_vs_A_bp':             delta_C_vs_A,
        'delta_D_vs_A_bp':             delta_D_vs_A,
        'delta_D_vs_C_bp':             delta_D_vs_C,
        'rho_V5_vs_V4_spearman':       rho_v5_v4,
        'rho_V5_vs_K4_bases':          rho_v5_K4,
        'v5_standalone_oof_auc':       float(roc_auc_score(y, v5)),
        'v4_standalone_oof_auc':       float(roc_auc_score(y, v4)),
        'verdict_swap_v4_v5':          swap_v4_v5,
        'verdict_add_v5_to_K5':        add_v5_to_K5,
    }
    (ART / 'probe_v5_gate.json').write_text(json.dumps(out, indent=2))
    print('\n=== V5 GATE SUMMARY ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
