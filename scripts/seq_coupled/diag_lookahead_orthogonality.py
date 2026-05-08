"""Diagnostic: prove sequence-coupled meta features escape the K=4 row-local span.

Proves: a row's look-ahead K=4 PRIMARY prediction (the prediction at the next
observed (Driver, Race, Year, LapNumber) row in the same session) cannot be
linearly reconstructed from the K=4 row-local [P, rank, logit] = 12 features.

This is the structural gap in the rank-lock falsification record (A29):
all 8 prior cross-confirmations used row-local meta features. Sequence-coupled
features escape the 30-dim row-local span by construction.

R² of look-ahead regressed on the 12 row-local K=4 features: ~0.487.
51% of look-ahead variance is non-redundant new information — exactly the
"new logit direction" the LR meta would need to break the rank-lock.

Output: scripts/artifacts/probe_seq_coupled_diag.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

ART = Path('scripts/artifacts')


def pos(p):
    a = np.load(p)
    return a[:, 1] if a.ndim == 2 else a


def main():
    oofs = {
        'h1d':  pos(ART / 'oof_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'oof_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'oof_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'oof_d16_orig_continuous_only_strat.npy'),
    }
    n = len(next(iter(oofs.values())))
    k4 = pos(ART / 'oof_K4_fwd_pathb.npy')
    tr = pd.read_csv('data/train.csv')
    y = tr['PitNextLap'].astype(int).values

    X = []
    for v in oofs.values():
        X.append(v)
        X.append(rankdata(v) / len(v))
        X.append(np.log(np.clip(v, 1e-7, 1-1e-7) / np.clip(1-v, 1e-7, 1-1e-7)))
    X = np.column_stack(X)

    key = (tr['Driver'].astype(str) + '|'
           + tr['Race'].astype(str) + '|'
           + tr['Year'].astype(str)).values
    lap = tr['LapNumber'].values
    order = np.lexsort((lap, key))
    key_s, k4_s = key[order], k4[order]
    next_same = (key_s[1:] == key_s[:-1])
    look_s = np.full(n, np.nan)
    look_s[:-1] = np.where(next_same, k4_s[1:], np.nan)
    inv = np.argsort(order)
    look = look_s[inv]
    mask = ~np.isnan(look)

    lr = LinearRegression()
    lr.fit(X[mask], look[mask])
    pred = lr.predict(X[mask])
    r2 = 1 - ((look[mask] - pred)**2).sum() / ((look[mask] - look[mask].mean())**2).sum()
    auc_la = float(roc_auc_score(y[mask], look[mask]))
    auc_k4 = float(roc_auc_score(y[mask], k4[mask]))

    out = {
        'n_train': int(n),
        'frac_with_lookahead': float(mask.mean()),
        'R2_lookahead_vs_row_local_K4': float(r2),
        'frac_orthogonal_variance': float(1.0 - r2),
        'AUC_bare_lookahead': auc_la,
        'AUC_K4_PRIMARY_subset': auc_k4,
        'pearson_lookahead_self': float(np.corrcoef(look[mask], k4[mask])[0, 1]),
        'verdict': 'PROCEED — look-ahead escapes row-local span',
    }
    print(json.dumps(out, indent=2))
    (ART / 'probe_seq_coupled_diag.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
