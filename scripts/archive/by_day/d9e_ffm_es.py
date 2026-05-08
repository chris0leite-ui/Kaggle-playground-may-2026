"""Day-9e take 2 — FFM with early stopping at epochs=2.

The 6-epoch FFM in d9e_ffm.py overfit catastrophically (std OOF 0.91525
vs FM's 0.92069). Per-fold val_AUC curves showed epoch 1 was the
peak. Re-run with epochs=2 (= 2 passes over the train data) which
captures peak-val behaviour. Same other hyperparams.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import d9e_ffm as d9e

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_S = 0.95070
PRIMARY_LB = 0.95029


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1]


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = np.load(ART / "oof_d9c_Sd_K20_swap_FM_strat.npy"
                          )[:, 1].astype(np.float64)
    primary_test = np.load(ART / "test_d9c_Sd_K20_swap_FM_strat.npy"
                           )[:, 1].astype(np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    print(f"\n=== FFM 5-fold (k=4, epochs=2 early-stop) ===\n")
    params = dict(embed_dim=4, epochs=2, lr=0.05, batch=8192)
    oof_ffm, test_ffm = d9e.train_ffm(train, test, y, splits, params, seed=42)
    auc = float(roc_auc_score(y, oof_ffm))
    rho_test, _ = spearmanr(test_ffm, primary_test)
    F_min = expand(np.column_stack([primary_oof, oof_ffm]))
    F_min_t = expand(np.column_stack([primary_test, test_ffm]))
    mo, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo))
    delta_mm = (auc_min - PRIMARY_S) * 1e4
    print(f"\n  FFM(es=2) std OOF: {auc:.5f}")
    print(f"  ρ vs PRIMARY test: {rho_test:.5f}")
    print(f"  Min-meta vs PRIMARY: {auc_min:.5f}  Δ {delta_mm:+.2f}bp  "
          f"{'PASS ✓' if auc_min >= PRIMARY_S else 'FAIL ✗'}")

    np.save(ART / "oof_d9e_ffm_es2_strat.npy",
            np.column_stack([1 - oof_ffm, oof_ffm]))
    np.save(ART / "test_d9e_ffm_es2_strat.npy",
            np.column_stack([1 - test_ffm, test_ffm]))

    # K=20 swap & K=21 add stacks
    base_oof, base_test, base_names = [], [], []
    for label, fname in d9e.POOL_KEEP:
        base_oof.append(np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64))
        base_test.append(np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64))
        base_names.append(label)
    d9_oof, d9_test, d9_names = [], [], []
    for label, fname in d9e.TOP_3_D9:
        d9_oof.append(np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64))
        d9_test.append(np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64))
        d9_names.append(label)
    fm_pool_oof = np.load(ART / "oof_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    fm_pool_test = np.load(ART / "test_d9c_fm_strat.npy")[:, 1].astype(np.float64)

    results = {"ffm_standalone": dict(
        std_oof=auc, rho_vs_primary=float(rho_test),
        min_meta_oof=auc_min, delta_primary_bp=float(delta_mm),
        min_meta_pass=bool(auc_min >= PRIMARY_S))}

    mo_swap, tp_swap = d9e.stack_eval(
        "K20_swap_es2", oof_ffm, test_ffm, y, primary_test,
        base_oof, base_test, base_names, d9_oof, d9_test, d9_names, results)
    mo_add, tp_add = d9e.stack_eval(
        "K21_add_es2", oof_ffm, test_ffm, y, primary_test,
        base_oof, base_test, base_names, d9_oof, d9_test, d9_names, results,
        also_keep_pool_fm=True, fm_pool_oof=fm_pool_oof,
        fm_pool_test=fm_pool_test)

    np.save(ART / "test_d9e_K20_swap_FFM_es2_strat.npy",
            np.column_stack([1 - tp_swap, tp_swap]))
    sub = sample_sub.copy(); sub[TARGET] = tp_swap
    sub.to_csv("submissions/submission_d9e_K20_swap_FFM_es2.csv", index=False)
    print("→ wrote submissions/submission_d9e_K20_swap_FFM_es2.csv")
    np.save(ART / "test_d9e_K21_add_FFM_es2_strat.npy",
            np.column_stack([1 - tp_add, tp_add]))
    sub = sample_sub.copy(); sub[TARGET] = tp_add
    sub.to_csv("submissions/submission_d9e_K21_add_FFM_es2.csv", index=False)
    print("→ wrote submissions/submission_d9e_K21_add_FFM_es2.csv")

    final = dict(results=results, params=params,
                 primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
                 total_wall_s=time.time() - t0)
    (ART / "d9e_ffm_es2_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9e_ffm_es2_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
