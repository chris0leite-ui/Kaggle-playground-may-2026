"""ζ — Deep CatBoost single-fold probe (depth=8, full iters).

Per the new 1-fold-actual-within-1h cap rule (loops.md amendment):
the original M3 probe-fail used depth=8/iters=2000 which projected
96 min for 5-fold both-anchor. Single fold actual was 9.6 min (within
1h). The probe-fail logged single-fold AUC 0.94992 — best single
fold ever seen. Re-run that single fold cleanly with artifacts saved
so we can decide whether to invest in 5-fold deep CatBoost.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import N_FOLDS, SEED

TARGET, ID_COL = "PitNextLap", "id"
CAT_COLS = ["Driver", "Race", "Compound"]
BASE_S = 0.94075


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in CAT_COLS:
        X[c] = X[c].astype(str); X_test[c] = X_test[c].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.zeros(len(y)), y)))

    p = dict(iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
             random_seed=SEED, eval_metric="AUC", od_type="Iter", od_wait=100,
             verbose=0, thread_count=-1, allow_writing_files=False)
    print(f"params: {p}")
    print(f"single-fold sizes: train={len(tr)}, val={len(va)}, test={len(X_test)}")

    t_fit = time.time()
    ptr = Pool(X.iloc[tr], y[tr], cat_features=CAT_COLS)
    pva = Pool(X.iloc[va], y[va], cat_features=CAT_COLS)
    pte = Pool(X_test, cat_features=CAT_COLS)
    m = CatBoostClassifier(**p)
    m.fit(ptr, eval_set=pva)
    p_va = m.predict_proba(pva)[:, 1]
    p_test = m.predict_proba(pte)[:, 1]
    fit_secs = time.time() - t_fit
    auc = roc_auc_score(y[va], p_va)
    bi = m.get_best_iteration()
    print(f"\nfold 0 AUC={auc:.5f}  best_iter={bi}  Δ baseline={(auc-BASE_S)*1e4:+.1f}bp")
    print(f"fit wall: {fit_secs:.1f}s ({fit_secs/60:.1f} min)")
    print(f"5-fold projection: {fit_secs*5:.0f}s ({fit_secs*5/60:.1f} min)")
    print(f"both-anchor projection: {fit_secs*10:.0f}s ({fit_secs*10/60:.1f} min)")

    Path("scripts/artifacts").mkdir(parents=True, exist_ok=True)
    np.save(f"scripts/artifacts/oof_zeta_catboost_deep_fold0.npy", p_va.astype(np.float32))
    np.save(f"scripts/artifacts/test_zeta_catboost_deep_fold0.npy", p_test.astype(np.float32))
    np.save(f"scripts/artifacts/zeta_va_idx.npy", va.astype(np.int32))
    res = dict(fold0_auc=float(auc), best_iter=int(bi),
               fit_secs=float(fit_secs),
               proj_5fold_singleanchor_secs=float(fit_secs * 5),
               proj_5fold_twoanchor_secs=float(fit_secs * 10),
               delta_vs_baseline_bp=float((auc - BASE_S) * 1e4),
               params=p)
    Path("scripts/artifacts/zeta_catboost_deep_fold0_results.json").write_text(json.dumps(res, indent=2))

    body = (
        f"# ζ — Deep CatBoost single-fold probe (2026-05-04)\n\n"
        f"Per new 1-fold-actual-within-1h cap rule.\n\n"
        f"## Result\n\n"
        f"- fold-0 AUC: **{auc:.5f}** (Δ baseline {(auc-BASE_S)*1e4:+.1f}bp)\n"
        f"- best_iter: {bi}/{p['iterations']} ({'ES fired' if bi < p['iterations']-1 else 'hit cap'})\n"
        f"- fit wall: {fit_secs:.1f}s ({fit_secs/60:.2f} min)\n"
        f"- 5-fold both-anchor projection: {fit_secs*10:.0f}s ({fit_secs*10/60:.1f} min)\n"
        f"- single-fold within 1h cap: {'YES' if fit_secs < 3600 else 'NO'}\n\n"
        f"## Decision\n\n"
        f"If fold-0 AUC > 0.94900 (E3 HGBC ballpark), pursue 5-fold both-anchor.\n"
        f"Otherwise skip; not worth the compute.\n"
    )
    Path("audit/2026-05-04-zeta-catboost-deep-fold0.md").write_text(body)
    print(f"\ntotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
