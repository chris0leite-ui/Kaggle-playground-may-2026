"""Rebuild PRIMARY K=22 (d9h S2 add aug12) OOF + test from component bases.

The d9h experiment saved only the test prediction; the OOF was not
persisted.  Reproduce per-fold LR-meta with logit-link inputs and a
single C=1 LR (matches m5h conventions in scripts/common.py).

Saves:
  scripts/artifacts/oof_PRIMARY_K22_strat.npy      (439_140,)
  scripts/artifacts/test_PRIMARY_K22_strat.npy     (188_165,)  [for sanity]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")

K22_BASES = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"),
    ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    ("realmlp", "realmlp"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
    ("FM_A_d9f", "d9f_FM_A"),
    ("FM_B_d9f", "d9f_FM_B"),
    ("FM_aug12", "d9h_FM_aug12"),
]


def load_pos(name: str, kind: str) -> np.ndarray:
    arr = np.load(ART / f"oof_{name}_{kind}.npy")
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr[:, 1]
    return arr.astype(np.float32)


def load_test_pos(name: str, kind: str) -> np.ndarray:
    arr = np.load(ART / f"test_{name}_{kind}.npy")
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr[:, 1]
    return arr.astype(np.float32)


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    train = pd.read_csv("data/train.csv")
    y = train["PitNextLap"].astype(int).to_numpy()

    # load OOFs as raw probabilities
    P = []
    Pt = []
    used = []
    for label, fn in K22_BASES:
        try:
            oof = load_pos(fn, "strat")
            te = load_test_pos(fn, "strat")
        except FileNotFoundError:
            print(f"MISSING: {fn}; skipping")
            continue
        P.append(oof)
        Pt.append(te)
        used.append(label)
    P = np.column_stack(P).astype(np.float64)
    Pt = np.column_stack(Pt).astype(np.float64)
    print(f"K = {P.shape[1]} bases used: {used}")

    def expand(M: np.ndarray) -> np.ndarray:
        n = len(M)
        rk = np.column_stack([rankdata(c) / n for c in M.T])
        logit = np.log(np.clip(M, 1e-9, 1 - 1e-9) /
                       (1 - np.clip(M, 1e-9, 1 - 1e-9)))
        return np.hstack([M, rk, logit])

    F_oof = expand(P)
    F_test = expand(Pt)

    # 5-fold OOF LR-meta (matches d9h fit_lr_meta exactly)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof_meta = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        oof_meta[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    test_meta = lr_full.predict_proba(F_test)[:, 1]

    auc = roc_auc_score(y, oof_meta)
    print(f"PRIMARY K={P.shape[1]} reconstructed OOF AUC = {auc:.6f} "
          f"(d9h target = 0.95073)")
    np.save(ART / "oof_PRIMARY_K22_strat.npy", oof_meta.astype(np.float32))
    np.save(ART / "test_PRIMARY_K22_strat.npy", test_meta.astype(np.float32))
    print("saved oof_PRIMARY_K22_strat.npy and test_PRIMARY_K22_strat.npy")


if __name__ == "__main__":
    main()
