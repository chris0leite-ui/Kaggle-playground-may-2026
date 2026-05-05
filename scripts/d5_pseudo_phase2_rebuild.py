"""D5 Path B Phase 2 — rebuild 5 fast CPU bases with pseudo-labels,
then partial-pseudo M5q LR-stack (6 pseudo + 8 original).

Phase 1 (e3_hgbc MVP) cleared both gates: +4.1bp OOF, ρ=0.99593.
Phase 2 expands to all rebuildable fast CPU GBDT bases. Defers:
  - a_horizon / b_lapsuntilpit (target reformulations don't take
    pseudo-PitNextLap directly)
  - d2a_te (TE within-fold needs pseudo-aware redesign)
  - CatBoost CPU bases (e1_cb_sub, cb_year-cat, cb_lossguide — slow)
  - Kaggle-GPU bases (realmlp, cb_slow-wide-bag)

Phase-2 bases (cumulative L1 in M5q ≈ 2.4 of total ~7-8):
  baseline_two_anchor  (LGBM, anchor 0.94075)  L1≈0.34
  m2_xgb               (XGB,  anchor 0.94507)  L1≈0.18
  e5_optuna_lgbm       (LGBM tuned, 0.94736)   L1≈0.84
  f1_hgbc_deep         (HGBC, 0.94870)         L1≈0.36
  f2_hgbc_shallow      (HGBC, 0.94861)         L1≈0.43

Decision rule for Phase 3 (slow + GPU rebuilds):
  partial-pseudo M5q OOF (6 pseudo + 8 orig, K=14) >= 0.95057 + 1bp
  → expand. Otherwise abandon Path B.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
M5Q_ANCHOR = 0.95057

M5H_TEST_NAMES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat",
    "cb_lossguide", "cb_slow-wide-bag",
]

# Phase 2 base specs: name, model_kind, anchor_oof
PHASE2_SPECS = [
    ("baseline_two_anchor", "lgbm", 0.94075),
    ("m2_xgb", "xgb", 0.94507),
    ("e5_optuna_lgbm", "lgbm_optuna", 0.94736),
    ("f1_hgbc_deep", "hgbc", 0.94870),
    ("f2_hgbc_shallow", "hgbc", 0.94861),
]


def prep_lgbm_xgb(train: pd.DataFrame, test: pd.DataFrame):
    """All string-like cols → category with unified vocab (preserves dtype across concat)."""
    train, test = train.copy(), test.copy()
    cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in cat_cols:
        union = pd.concat([train[c].astype(str), test[c].astype(str)],
                           ignore_index=True)
        cat_dtype = pd.CategoricalDtype(categories=sorted(union.unique()))
        train[c] = train[c].astype(str).astype(cat_dtype)
        test[c] = test[c].astype(str).astype(cat_dtype)
    return train, test, cat_cols


def prep_hgbc(train: pd.DataFrame, test: pd.DataFrame):
    """Driver label-encoded; Compound/Race as category (HGBC native)."""
    train, test = train.copy(), test.copy()
    HIGH_CARD = ["Driver"]
    LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in train.columns:
            uniq = pd.concat([train[c], test[c]], ignore_index=True).astype(str).unique()
            mapping = {v: i for i, v in enumerate(sorted(uniq))}
            train[c] = train[c].astype(str).map(mapping).astype(np.int32)
            test[c] = test[c].astype(str).map(mapping).astype(np.int32)
    for c in LOW_CARD:
        if c in train.columns:
            train[c] = train[c].astype("category")
            test[c] = test[c].astype("category")
    return train, test, []


def make_lgb_baseline_params() -> dict:
    return dict(objective="binary", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                min_data_in_leaf=200, verbose=-1, seed=SEED)


def make_lgb_optuna_params() -> dict:
    """Best params from e5_optuna_lgbm_strat_results.json."""
    return dict(
        objective="binary", verbose=-1, seed=SEED,
        learning_rate=0.017739639246790825,
        num_leaves=229,
        min_data_in_leaf=103,
        feature_fraction=0.6575935831140413,
        bagging_fraction=0.7647859924409988,
        bagging_freq=4,
        lambda_l1=3.819433130462022e-05,
        lambda_l2=0.00019363249015436128,
        max_depth=10,
    )


def make_xgb():
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        learning_rate=0.08, max_depth=6, subsample=0.9, colsample_bytree=0.9,
        min_child_weight=20, n_estimators=1000, early_stopping_rounds=80,
        enable_categorical=True, random_state=42, n_jobs=-1, verbosity=0,
    )


F1_HP = dict(max_iter=2000, learning_rate=0.05, max_leaf_nodes=127,
             min_samples_leaf=100, l2_regularization=0.1,
             early_stopping=True, validation_fraction=0.1,
             n_iter_no_change=50, random_state=123,
             categorical_features="from_dtype")
F2_HP = dict(max_iter=1500, learning_rate=0.05, max_leaf_nodes=31,
             min_samples_leaf=400, l2_regularization=0.1,
             early_stopping=True, validation_fraction=0.1,
             n_iter_no_change=50, random_state=7,
             categorical_features="from_dtype")


def fit_lgbm(params, X_tr, y_tr, X_va, y_va, X_test, cat_cols):
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return m.predict(X_va), m.predict(X_test), int(m.best_iteration or 0)


def fit_xgb(X_tr, y_tr, X_va, y_va, X_test):
    m = make_xgb()
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return (m.predict_proba(X_va)[:, 1],
            m.predict_proba(X_test)[:, 1],
            int(getattr(m, "best_iteration", 0)))


def fit_hgbc(hp, X_tr, y_tr, X_va, X_test):
    m = HistGradientBoostingClassifier(**hp)
    m.fit(X_tr, y_tr)
    return (m.predict_proba(X_va)[:, 1],
            m.predict_proba(X_test)[:, 1],
            int(m.n_iter_))


def build_pseudo_gate(test_m5q, m5h_test):
    track_a_pos = test_m5q >= 0.95
    track_a_neg = test_m5q <= 0.05
    track_a = track_a_pos | track_a_neg
    vote_pos = (m5h_test > 0.5).sum(axis=1)
    track_b_pos = vote_pos >= 10
    track_b_neg = vote_pos <= 3
    track_b = track_b_pos | track_b_neg
    pseudo_mask = track_a | track_b
    pseudo_y = np.zeros(len(test_m5q), dtype=np.int8)
    pseudo_y[track_a_pos | track_b_pos] = 1
    pseudo_y[track_a_neg | track_b_neg] = 0
    pseudo_y[track_a_pos & track_b_neg] = 1
    pseudo_y[track_a_neg & track_b_pos] = 0
    return pseudo_mask, pseudo_y


def rebuild_one(name: str, kind: str, anchor: float,
                train_X_full: pd.DataFrame, test_X_full: pd.DataFrame,
                y_real: np.ndarray, pseudo_mask: np.ndarray,
                pseudo_y: np.ndarray, splits, sample_sub: pd.DataFrame):
    t0 = time.time()
    if kind in ("lgbm", "lgbm_optuna", "xgb"):
        X, X_test, cat_cols = prep_lgbm_xgb(train_X_full, test_X_full)
    else:
        X, X_test, cat_cols = prep_hgbc(train_X_full, test_X_full)
    test_pseudo = X_test.iloc[pseudo_mask].reset_index(drop=True)
    y_pseudo_subset = pseudo_y[pseudo_mask]

    oof = np.zeros(len(y_real), dtype=np.float32)
    tp = np.zeros(len(test_X_full), dtype=np.float32)
    fold_aucs, walls = [], []
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        X_aug = pd.concat([X.iloc[tr], test_pseudo], ignore_index=True)
        y_aug = np.concatenate([y_real[tr], y_pseudo_subset])
        if kind == "lgbm":
            p_va, p_te, biter = fit_lgbm(make_lgb_baseline_params(),
                                          X_aug, y_aug,
                                          X.iloc[va], y_real[va],
                                          X_test, cat_cols)
        elif kind == "lgbm_optuna":
            p_va, p_te, biter = fit_lgbm(make_lgb_optuna_params(),
                                          X_aug, y_aug,
                                          X.iloc[va], y_real[va],
                                          X_test, cat_cols)
        elif kind == "xgb":
            p_va, p_te, biter = fit_xgb(X_aug, y_aug,
                                         X.iloc[va], y_real[va], X_test)
        elif kind == "hgbc":
            hp = F1_HP if name == "f1_hgbc_deep" else F2_HP
            p_va, p_te, biter = fit_hgbc(hp, X_aug, y_aug, X.iloc[va], X_test)
        else:
            raise ValueError(kind)
        oof[va] = p_va
        tp += p_te / N_FOLDS
        wall = time.time() - t_fold
        s = float(roc_auc_score(y_real[va], p_va))
        fold_aucs.append(s); walls.append(wall)
        print(f"  [{name}] fold {k}: AUC={s:.5f}  iter={biter}  "
              f"wall={wall:.1f}s")

    rebuilt_oof = float(roc_auc_score(y_real, oof))
    delta_bp = (rebuilt_oof - anchor) * 1e4
    orig_test = np.load(ART / f"test_{name}_strat.npy")[:, 1]
    rho, _ = spearmanr(tp, orig_test)
    total_wall = time.time() - t0
    print(f"  [{name}] OOF={rebuilt_oof:.5f}  Δ anchor({anchor:.5f})="
          f"{delta_bp:+.2f}bp  ρ={rho:.5f}  total={total_wall:.0f}s\n")

    np.save(ART / f"oof_d5_{name}_pseudo_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / f"test_d5_{name}_pseudo_strat.npy",
            np.column_stack([1 - tp, tp]))
    return dict(rebuilt_oof=rebuilt_oof, anchor=anchor, delta_bp=delta_bp,
                rho_vs_orig=float(rho), fold_aucs=fold_aucs,
                fold_walls_s=walls, total_wall_s=total_wall, oof=oof, test=tp)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    eps = 1e-9
    Pc = np.clip(P, eps, 1 - eps)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def lr_meta(F_oof, F_test, y, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc, lr_full.coef_.ravel()


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y_real = train[TARGET].astype(int).values
    train_X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    test_X = test.drop(columns=[ID_COL], errors="ignore")

    # Build pseudo gate
    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]
    m5h_test = np.column_stack([
        np.load(ART / f"test_{n}_strat.npy")[:, 1] for n in M5H_TEST_NAMES
    ])
    pseudo_mask, pseudo_y = build_pseudo_gate(test_m5q, m5h_test)
    print(f"pseudo: {pseudo_mask.sum()} / {len(test_m5q)} rows  "
          f"pos_rate={pseudo_y[pseudo_mask].mean():.4f}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y_real)), y_real))

    # Rebuild Phase 2 bases
    base_results = {}
    for name, kind, anchor in PHASE2_SPECS:
        print(f"=== rebuild {name} ({kind}, anchor {anchor:.5f}) ===")
        r = rebuild_one(name, kind, anchor, train_X, test_X,
                        y_real, pseudo_mask, pseudo_y, splits, sample_sub)
        base_results[name] = r

    # Partial-pseudo M5q stack: 6 pseudo (Phase-1 e3 + 5 Phase-2) + 8 original
    PSEUDO_BASES = [
        ("baseline", "d5_baseline_two_anchor_pseudo"),
        ("m2_xgb", "d5_m2_xgb_pseudo"),
        ("e5_optuna_lgbm", "d5_e5_optuna_lgbm_pseudo"),
        ("f1_hgbc_deep", "d5_f1_hgbc_deep_pseudo"),
        ("f2_hgbc_shallow", "d5_f2_hgbc_shallow_pseudo"),
        ("e3_hgbc", "d5_e3_hgbc_pseudo"),
    ]
    ORIG_BASES = [
        ("d2a_te", "d2a_te"),
        ("e1_cb_sub", "e1_catboost_sub"),
        ("a_horizon", "a_horizon"),
        ("b_lapsuntilpit", "b_lapsuntilpit"),
        ("cb_year-cat", "cb_year-cat"),
        ("cb_lossguide", "cb_lossguide"),
        ("cb_slow-wide-bag", "cb_slow-wide-bag"),
        ("realmlp", "realmlp"),
    ]
    Xs_oof, Xs_test, names = [], [], []
    for label, fname in PSEUDO_BASES + ORIG_BASES:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    F_oof = expand(np.column_stack(Xs_oof))
    F_test = expand(np.column_stack(Xs_test))

    meta_oof, meta_test, auc_partial, coef = lr_meta(F_oof, F_test, y_real)
    delta_m5q = (auc_partial - M5Q_ANCHOR) * 1e4
    rho_m5q, _ = spearmanr(meta_test, test_m5q)
    print(f"=== Partial-pseudo M5q LR-stack (6 pseudo + 8 orig, K=14) ===")
    print(f"  Strat OOF: {auc_partial:.5f}  Δ M5q anchor: {delta_m5q:+.2f}bp")
    print(f"  ρ vs M5q test: {rho_m5q:.5f}")
    if rho_m5q >= 0.9997: gate_str = "TIE_EXPECTED"
    elif rho_m5q >= 0.999: gate_str = "TIE_LIKELY"
    elif rho_m5q >= 0.994: gate_str = "REAL_DELTA"
    else: gate_str = "DIVERGENT"
    print(f"  gate: {gate_str}")

    K = len(names)
    l1 = {n: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2 * K + i]))
          for i, n in enumerate(names)}
    print("\nL1 ranking partial-pseudo K=14:")
    for n, v in sorted(l1.items(), key=lambda x: -x[1]):
        marker = " ← pseudo" if any(n == p[0] for p in PSEUDO_BASES) else ""
        print(f"  {n:<22s} L1={v:.3f}{marker}")

    np.save(ART / "oof_d5_partial_pseudo_m5q_strat.npy",
            np.column_stack([1 - meta_oof, meta_oof]))
    np.save(ART / "test_d5_partial_pseudo_m5q_strat.npy",
            np.column_stack([1 - meta_test, meta_test]))
    sub = sample_sub.copy(); sub[TARGET] = meta_test
    sub.to_csv("submissions/submission_d5_partial_pseudo_m5q.csv", index=False)

    summary = dict(
        bases={k: {kk: vv for kk, vv in v.items() if kk not in ("oof", "test")}
               for k, v in base_results.items()},
        partial_pseudo_m5q=dict(
            strat_oof=auc_partial, delta_m5q_bp=delta_m5q,
            rho_vs_m5q_test=float(rho_m5q), gate=gate_str,
            l1=l1, n_bases=K,
        ),
        decision_rule="partial_pseudo > 0.95057 + 1bp → expand to slow rebuilds",
        proceed_to_phase3=bool(auc_partial > M5Q_ANCHOR + 1e-4),
        total_wall_s=time.time() - t0,
    )
    (ART / "d5_pseudo_phase2_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/d5_pseudo_phase2_results.json")
    print(f"total Phase-2 wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
