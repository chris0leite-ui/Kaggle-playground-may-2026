"""qBH — K=34 plain LR-meta test predictions only (backup if qBG stalls).

Reuses qBG's K=34 pool (27 K=27-era + 7 slim-kNN) but only computes
plain LR-meta. Skips Path-B amp.
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

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

POOL_KEEP_16 = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
]
TOP_3_D9 = ["d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound"]
FM_AB = ["d9f_FM_A", "d9f_FM_B"]
K27_EXTRAS = [
    "p1_single_cb_v4_gpu", "d17_h1d_yekenot_full",
    "d16_orig_continuous_only", "d18_chain_decomp",
    "d18_e2_preimage_knn", "d18_f2_constraint",
]
K27_NAMES = POOL_KEEP_16 + TOP_3_D9 + FM_AB + K27_EXTRAS

NEW_BASES = [
    ("qAT", "dgp_v3_qAT_K1_oof.npy", "dgp_v3_qAT_K1_test.npy"),
    ("qAV", "dgp_v3_qAV_K1_7feat_oof.npy", "dgp_v3_qAV_K1_7feat_test.npy"),
    ("qAO", "dgp_v3_qAO_knn_multi_oof.npy", "dgp_v3_qAO_knn_multi_test.npy"),
    ("qAA", "dgp_v3_qAA_stint_imputed_oof.npy", "dgp_v3_qAA_stint_imputed_test.npy"),
    ("qAF", "dgp_v3_qAF_d16plus_oof.npy", "dgp_v3_qAF_d16plus_test.npy"),
    ("qAK", "dgp_v3_qAK_knn3_oof.npy", "dgp_v3_qAK_knn3_test.npy"),
    ("qBA", "dgp_v3_qBA_manhattan_oof.npy", "dgp_v3_qBA_manhattan_test.npy"),
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    K27_oofs, K27_tests = [], []
    for nm in K27_NAMES:
        K27_oofs.append(_pos(ART / f"oof_{nm}_strat.npy"))
        K27_tests.append(_pos(ART / f"test_{nm}_strat.npy"))

    new_oofs, new_tests = [], []
    for nm, oof_f, test_f in NEW_BASES:
        new_oofs.append(_pos(ART / oof_f))
        new_tests.append(_pos(ART / test_f))

    all_oofs = K27_oofs + new_oofs
    all_tests = K27_tests + new_tests
    K_total = len(all_oofs)
    print(f"K={K_total} pool")

    F_oof = expand(np.column_stack(all_oofs))
    F_test = expand(np.column_stack(all_tests))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    for fold, (tr, va) in enumerate(skf.split(F_oof, y)):
        m = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)
        m.fit(F_oof[tr], y[tr])
        oof[va] = m.predict_proba(F_oof[va])[:, 1]
        test_pred += m.predict_proba(F_test)[:, 1] / N_FOLDS
        print(f"  fold {fold} done", flush=True)

    auc = float(roc_auc_score(y, oof))
    primary = _pos(ART / "oof_K4_fwd_pathb.npy")
    delta = (auc - roc_auc_score(y, primary)) * 1e4
    print(f"\nK={K_total} plain LR-meta OOF: {auc:.5f}  ΔvsK4: {delta:+.3f} bp")

    PRIMARY_test = _pos(ART / "test_K4_fwd_pathb.npy")
    rho = float(spearmanr(test_pred, PRIMARY_test).correlation)
    print(f"  ρ_test vs K=4: {rho:.5f}")

    np.save(ART / f"dgp_v3_qBH_K{K_total}_plain_oof.npy", oof)
    np.save(ART / f"dgp_v3_qBH_K{K_total}_plain_test.npy", test_pred)

    sub = pd.DataFrame({"id": test["id"].values, TARGET: test_pred})
    sub_path = Path("submissions") / f"submission_qBH_K{K_total}_plain_lr.csv"
    sub_path.parent.mkdir(exist_ok=True)
    sub.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path}; t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
