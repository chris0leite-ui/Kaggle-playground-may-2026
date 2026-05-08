"""Shared helpers for Day-13 G1 / G2' / G3 bases.

Loads data, splits, PRIMARY anchors. Computes ρ vs PRIMARY,
min-meta gate, predicted LB, and writes artifacts in the convention
used by the rest of the pool.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"

# PRIMARY anchors (mirror d13_move_b_fm_variants.py)
PRIMARY_S = 0.95073
PRIMARY_LB = 0.95034
PRIMARY_OOF_FILE = "oof_d9c_Sd_K20_swap_FM_strat.npy"
PRIMARY_TEST_FILE = "test_d9h_S2_K22_add_aug12_strat.npy"
RHO_TIE = 0.999


def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    return train, test, sub, y


def load_primary():
    primary_oof = np.load(ART / PRIMARY_OOF_FILE)[:, 1].astype(np.float64)
    primary_test = np.load(ART / PRIMARY_TEST_FILE)[:, 1].astype(np.float64)
    return primary_oof, primary_test


def make_splits(y, train, kind="strat"):
    if kind == "strat":
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return list(skf.split(np.zeros(len(y)), y))
    if kind == "groupkf":
        grp = train.groupby(["Race", "Driver", "Year", "Stint"],
                            sort=False).ngroup().values
        gkf = GroupKFold(n_splits=N_FOLDS)
        return list(gkf.split(np.zeros(len(y)), y, groups=grp))
    raise ValueError(kind)


def expand(P):
    """[raw, rank, logit] feature expansion for LR meta."""
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y, splits):
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1]


def predicted_lb(auc, rho):
    base_lb = PRIMARY_LB + (auc - PRIMARY_S)
    if rho >= RHO_TIE:
        return base_lb
    if rho >= 0.995:
        return base_lb - 0.0001
    if rho >= 0.99:
        return base_lb - 0.00025
    return base_lb - 0.0004


def min_meta_gate(primary_oof, primary_test, cand_oof, cand_test, y, splits):
    F_oof = expand(np.column_stack([primary_oof, cand_oof]))
    F_test = expand(np.column_stack([primary_test, cand_test]))
    mo, _ = fit_lr_meta(F_oof, F_test, y, splits)
    auc_min = float(roc_auc_score(y, mo))
    return auc_min, (auc_min - PRIMARY_S) * 1e4


def report_candidate(name: str, oof: np.ndarray, test: np.ndarray, y,
                     primary_oof, primary_test, splits) -> dict:
    """Standard 4-line report + dict for results JSON."""
    std_auc = float(roc_auc_score(y, oof))
    rho, _ = spearmanr(test, primary_test)
    auc_min, dbp = min_meta_gate(primary_oof, primary_test, oof, test, y, splits)
    pred_lb = predicted_lb(auc_min, float(rho))
    info = dict(
        name=name,
        std_oof=std_auc,
        rho_vs_primary=float(rho),
        min_meta_oof=auc_min,
        min_meta_delta_bp=dbp,
        pred_lb=pred_lb,
        pred_lb_delta_bp=(pred_lb - PRIMARY_LB) * 1e4,
    )
    print(f"\n--- {name} ---")
    print(f"  std OOF      : {std_auc:.5f}")
    print(f"  ρ vs PRIMARY : {rho:.5f}")
    print(f"  min-meta OOF : {auc_min:.5f}  Δ {dbp:+.2f}bp")
    print(f"  pred LB      : {pred_lb:.5f}  Δ {(pred_lb-PRIMARY_LB)*1e4:+.2f}bp")
    return info


def save_base(name: str, oof: np.ndarray, test: np.ndarray, info: dict):
    """Save oof_<name>_strat.npy as (n, 2) array (cols: 1-p, p)."""
    oof2 = np.column_stack([1 - oof, oof]).astype(np.float32)
    test2 = np.column_stack([1 - test, test]).astype(np.float32)
    np.save(ART / f"oof_{name}_strat.npy", oof2)
    np.save(ART / f"test_{name}_strat.npy", test2)
    (ART / f"{name}_results.json").write_text(json.dumps(info, indent=2))
    print(f"  saved oof_{name}_strat.npy / test_{name}_strat.npy")
