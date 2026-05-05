"""Move C / F1 — Closed-form rule + GBDT residual L1 base.

Critic-loop §3 F1 reformulation (b): "residual-from-Compound×TyreLife
rule". Mechanism class: instead of training a GBDT to predict
PitNextLap directly, train a GBDT REGRESSOR to predict
(PitNextLap - rule_proba) where rule_proba is a per-(Compound,
TyreLife-decile) lookup of the training base rate.

Hypothesis: forcing the model to spend capacity ONLY on residual
signal produces a mistake structure orthogonal to every existing
PitNextLap-direct base. Even if standalone AUC is mid-pack, the
rank ordering in the residual band should differ enough from M5q
to break the rank-lock when added to the K=14 pool.

Anchors
  e3_hgbc (best single GBDT pre-CB)         Strat 0.94876
  M5q (PRIMARY)                              Strat 0.95057  LB 0.95005

Decision rules per audit §5:
  - Standalone OOF >= e3_hgbc: minimum bar for adding to M5q pool
  - K=15 stack OOF >= M5q anchor + 1bp: required to slot for LB
  - ρ vs M5q test < 0.999: structural difference required
  - Predicted-gap gate: pred-LB >= M5q LB

Strat-only (R1).  Pool-add minimal-input-meta sanity check applied.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
M5Q_S = 0.95057
M5Q_LB = 0.95005
E3_S = 0.94876
N_TYRE_BINS = 10  # decile bins
RHO_TIE = 0.999

POOL_M5Q = [
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
]


def encode_features(X, X_test):
    HIGH_CARD = ["Driver"]
    LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True
                             ).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mp).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")
    return X, X_test


def make_hgbc_regressor():
    return HistGradientBoostingRegressor(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
        categorical_features="from_dtype",
    )


def fit_rule_lookup(comp_arr, tyre_arr, y_arr, tyre_edges):
    """Per-(Compound, TyreLife-decile) base-rate lookup with smoothing."""
    bins = np.searchsorted(tyre_edges, tyre_arr, side="right") - 1
    bins = np.clip(bins, 0, N_TYRE_BINS - 1)
    df = pd.DataFrame({"comp": comp_arr, "bin": bins, "y": y_arr})
    grp = df.groupby(["comp", "bin"], observed=True)["y"]
    counts = grp.count(); means = grp.mean()
    # Bayesian smoothing toward global mean (alpha=50 effective rows)
    glob = float(y_arr.mean()); alpha = 50.0
    smoothed = (means * counts + glob * alpha) / (counts + alpha)
    return smoothed.to_dict(), glob


def apply_rule_lookup(comp_arr, tyre_arr, lookup, glob, tyre_edges):
    bins = np.searchsorted(tyre_edges, tyre_arr, side="right") - 1
    bins = np.clip(bins, 0, N_TYRE_BINS - 1)
    out = np.full(len(comp_arr), glob, dtype=np.float64)
    for i, (c, b) in enumerate(zip(comp_arr, bins)):
        v = lookup.get((c, b))
        if v is not None:
            out[i] = v
    return out


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def main():
    t_total = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    y = train[TARGET].astype(int).values
    print(f"Train {train.shape}  test {test.shape}  prior y={y.mean():.4f}")

    # Compute global TyreLife deciles from train (used per fold w/ same edges).
    tyre_train = train["TyreLife"].values
    tyre_edges = np.quantile(tyre_train, np.linspace(0, 1, N_TYRE_BINS + 1))
    tyre_edges[0] = -np.inf; tyre_edges[-1] = np.inf
    print(f"TyreLife decile edges: {tyre_edges[1:-1].round(2).tolist()}")

    comp_train = train["Compound"].astype(str).values
    comp_test = test["Compound"].astype(str).values
    tyre_test = test["TyreLife"].values

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    X_test = test.drop(columns=[ID_COL], errors="ignore").copy()
    X, X_test = encode_features(X, X_test)

    # Standalone Strat 5-fold: rule + residual GBDT
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof_rule = np.zeros(len(y), dtype=np.float64)       # rule-only base
    oof_full = np.zeros(len(y), dtype=np.float64)       # rule + residual
    oof_resid = np.zeros(len(y), dtype=np.float64)      # residual-only diagnostic
    test_full = np.zeros(len(test), dtype=np.float64)
    test_rule = np.zeros(len(test), dtype=np.float64)
    fold_aucs_rule, fold_aucs_full = [], []
    walls = []

    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        # Per-fold rule lookup on train portion only
        lookup, glob = fit_rule_lookup(comp_train[tr], tyre_train[tr], y[tr],
                                       tyre_edges)
        rp_tr = apply_rule_lookup(comp_train[tr], tyre_train[tr], lookup,
                                  glob, tyre_edges)
        rp_va = apply_rule_lookup(comp_train[va], tyre_train[va], lookup,
                                  glob, tyre_edges)
        rp_te = apply_rule_lookup(comp_test, tyre_test, lookup, glob, tyre_edges)

        # Residual regression target
        resid_tr = y[tr].astype(np.float64) - rp_tr
        m = make_hgbc_regressor()
        m.fit(X.iloc[tr], resid_tr)
        resid_va = m.predict(X.iloc[va])
        resid_te = m.predict(X_test)

        pred_va = np.clip(rp_va + resid_va, 1e-9, 1 - 1e-9)
        pred_te = np.clip(rp_te + resid_te, 1e-9, 1 - 1e-9)

        oof_rule[va] = rp_va
        oof_resid[va] = resid_va
        oof_full[va] = pred_va
        test_full += pred_te / N_FOLDS
        test_rule += rp_te / N_FOLDS

        s_rule = float(roc_auc_score(y[va], rp_va))
        s_full = float(roc_auc_score(y[va], pred_va))
        wall = time.time() - t0
        fold_aucs_rule.append(s_rule); fold_aucs_full.append(s_full); walls.append(wall)
        print(f"  f{k}: rule_AUC={s_rule:.5f}  full_AUC={s_full:.5f}  "
              f"iters={m.n_iter_}  wall={wall:.1f}s")

    auc_rule = float(roc_auc_score(y, oof_rule))
    auc_full = float(roc_auc_score(y, oof_full))
    delta_e3 = (auc_full - E3_S) * 1e4
    delta_m5q_std = (auc_full - M5Q_S) * 1e4
    print(f"\n=== Standalone result ===")
    print(f"Rule-only OOF: {auc_rule:.5f}  (per-fold std={np.std(fold_aucs_rule):.5f})")
    print(f"Rule+residual OOF: {auc_full:.5f}  Δ e3 {delta_e3:+.2f}bp  "
          f"Δ M5q {delta_m5q_std:+.2f}bp  std={np.std(fold_aucs_full):.5f}")
    print(f"Total wall: {time.time()-t_total:.1f}s")

    # Save standalone artifacts
    np.save(ART / "oof_d6_rule_residual_strat.npy",
            np.column_stack([1 - oof_full, oof_full]))
    np.save(ART / "test_d6_rule_residual_strat.npy",
            np.column_stack([1 - test_full, test_full]))
    np.save(ART / "oof_d6_rule_only_strat.npy",
            np.column_stack([1 - oof_rule, oof_rule]))
    np.save(ART / "test_d6_rule_only_strat.npy",
            np.column_stack([1 - test_rule, test_rule]))

    # ρ vs M5q test
    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]
    rho_test, _ = spearmanr(test_full, test_m5q)
    print(f"\nρ(d6_rule_residual_test, M5q test) = {rho_test:.5f}")

    # K=15 stack: M5q pool + rule_residual
    print(f"\n=== K=15 stack: M5q pool + rule_residual ===")

    def load(name):
        oo = np.load(ART / f"oof_{name}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{name}_strat.npy")[:, 1].astype(np.float64)
        return oo, te

    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL_M5Q:
        oo, te = load(n)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    Xs_oof.append(oof_full); Xs_test.append(test_full)
    names.append("rule_residual")

    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc_k15 = float(roc_auc_score(y, mo))
    rho_k15, _ = spearmanr(tp, test_m5q)
    delta_m5q_stk = (auc_k15 - M5Q_S) * 1e4
    K = len(names)
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"K=15 LR-meta Strat OOF: {auc_k15:.5f}  Δ M5q {delta_m5q_stk:+.2f}bp  "
          f"ρ vs M5q test {rho_k15:.5f}")
    print(f"L1 ranking (top-5):")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {n:<22s} L1={v:.3f}")
    print(f"  rule_residual L1: {l1['rule_residual']:.3f}")

    # Minimal-input-meta sanity check (audit §5.2)
    print(f"\n=== Minimal-meta sanity check (M5q + rule_residual) ===")
    P_oof_min = np.column_stack([np.load(ART / "oof_m5q_strat.npy")[:, 1], oof_full])
    P_test_min = np.column_stack([test_m5q, test_full])
    F_oof_min = expand(P_oof_min); F_test_min = expand(P_test_min)
    mo_min, tp_min, _ = fit_lr_meta(F_oof_min, F_test_min, y)
    auc_min = float(roc_auc_score(y, mo_min))
    print(f"2-comp [M5q, rule_residual] LR-meta OOF: {auc_min:.5f}  "
          f"Δ M5q {(auc_min - M5Q_S)*1e4:+.2f}bp")
    if auc_min < M5Q_S:
        print(f"  ⚠️  2-comp OOF < M5q: K=15 lift would be cross-component memorization")

    # Save K=15 stack
    np.save(ART / "oof_d6_k15_rule_residual_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d6_k15_rule_residual_strat.npy",
            np.column_stack([1 - tp, tp]))
    sub = sample_sub.copy(); sub[TARGET] = tp
    sub.to_csv("submissions/submission_d6_k15_rule_residual.csv", index=False)

    # Verdict
    print(f"\n=== Verdict per audit §5 ===")
    base_pass = auc_full >= E3_S - 0.5/1e4
    stack_pass = auc_k15 >= M5Q_S + 1.0/1e4
    rho_pass = rho_k15 < RHO_TIE
    minimal_pass = auc_min >= M5Q_S
    pred_lb = M5Q_LB + (auc_k15 - M5Q_S) - (
        0 if rho_k15 >= RHO_TIE
        else 0.0001 if rho_k15 >= 0.995
        else 0.00025 if rho_k15 >= 0.99
        else 0.0004)
    print(f"  Base OOF >= e3 (0.94876):           {base_pass}  ({auc_full:.5f})")
    print(f"  K=15 OOF >= M5q + 1bp (0.95067):    {stack_pass}  ({auc_k15:.5f})")
    print(f"  ρ K=15 < 0.999:                     {rho_pass}  ({rho_k15:.5f})")
    print(f"  Minimal-meta OOF >= M5q (0.95057):  {minimal_pass}  ({auc_min:.5f})")
    print(f"  Predicted LB:                       {pred_lb:.5f}  "
          f"(vs M5q LB {M5Q_LB:.5f})")
    if base_pass and stack_pass and rho_pass and minimal_pass:
        print(f"  → SLOT-WORTHY")
    else:
        print(f"  → DO NOT SLOT (one or more gates failed)")

    results = dict(
        standalone=dict(
            rule_only_oof=auc_rule, rule_only_std=float(np.std(fold_aucs_rule)),
            rule_residual_oof=auc_full, rule_residual_std=float(np.std(fold_aucs_full)),
            delta_e3_bp=delta_e3, delta_m5q_bp=delta_m5q_std,
            fold_aucs_rule=fold_aucs_rule, fold_aucs_full=fold_aucs_full,
            walls=walls,
        ),
        rho_test_vs_m5q=float(rho_test),
        k15_stack=dict(
            oof=auc_k15, delta_m5q_bp=delta_m5q_stk,
            rho_vs_m5q_test=float(rho_k15), l1_ranking=l1,
            rule_residual_l1=l1["rule_residual"],
        ),
        minimal_meta=dict(oof=auc_min,
                          delta_m5q_bp=(auc_min - M5Q_S) * 1e4),
        gate=dict(base_pass=bool(base_pass), stack_pass=bool(stack_pass),
                  rho_pass=bool(rho_pass), minimal_pass=bool(minimal_pass),
                  pred_lb=float(pred_lb)),
    )
    (ART / "d6_rule_residual_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n→ scripts/artifacts/d6_rule_residual_results.json")


if __name__ == "__main__":
    main()
