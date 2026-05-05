"""D12 — Year-segmented specialist + Adversarial-validation reweighting.

Hypotheses (P3, P9 from `audit/2026-05-08-data-probe-results.md`):
- Year=2023 is structural mode collapse (0.96% pit rate vs ~28% other
  years, 31% of train+test). A single model fits both regimes and
  averages weak signal in the 2023 cohort.
- Train/test mismatch may exist; importance reweighting via an AV
  classifier could re-align the train objective to test.

Parts:
  A) M_active (Year ∈ {2022, 2024, 2025}) + M_2023 specialists, route
     by Year on full train (5-fold OOF) + full test.
  B) Adversarial-validation classifier (LGBM) → importance weights
     w = clip(p_test/(1-p_test), 0.1, 10), retrain e3-equivalent HGBC
     with sample_weight=w.
  C) Year-segmented + AV weights combined.
  D) Stack-add: standalone OOF, ρ vs PRIMARY test, min-meta-Δ, K=22 add.

Reuses HGBC base from `scripts/e3_hgbc_two_anchor.py` (best single
non-CB base; HGBC label-encoded driver, native cat for Compound/Race).

Saves:
  oof_d12_year_specialist_strat.npy / test_d12_year_specialist_strat.npy
  oof_d12_e3_advweight_strat.npy   / test_d12_e3_advweight_strat.npy
  oof_d12_year_advweight_strat.npy / test_d12_year_advweight_strat.npy
  d12_year_specialist_advweight_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

# Current PRIMARY (per CLAUDE.md state block): d9f K=21 swap partA+partB
# However only test_d9f_K21_swap_strat.npy exists (no matching OOF artifact);
# fall back to the immediately previous PRIMARY d9c Sd K=20 swap FM, which
# has both OOF+test arrays. Their OOF AUCs are within 0.3bp (0.95073 vs
# 0.95070), so this is a faithful surrogate for stack-add diagnostics.
# We additionally compute ρ vs the d9f K=21 test array (the actual LB best).
PRIMARY_NAME = "d9c_Sd_K20_swap_FM"
PRIMARY_S, PRIMARY_LB = 0.95070, 0.95029
PRIMARY_TEST_LB_BEST = "d9f_K21_swap"   # has only test array
RHO_TIE = 0.999


# ----------------------------- data prep -----------------------------
def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    HIGH = ["Driver"]
    LOW = ["Compound", "Race"]
    for c in HIGH:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mp).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)
    for c in LOW:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")
    return train, test, sample_sub, X, y, X_test


def make_hgbc():
    return HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
        categorical_features="from_dtype",
    )


# ----------------------------- Part A: Year specialist ---------------
def part_a_year_specialist(X, y, X_test, year_train, year_test):
    """Train 2 specialists on year cohorts, route on Year."""
    print("\n=== Part A: Year-segmented specialist ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float64)
    tp = np.zeros(len(X_test), dtype=np.float64)

    fold_walls = []
    fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        # Within fold tr: split by Year regime
        tr_active = tr[year_train[tr] != 2023]
        tr_2023 = tr[year_train[tr] == 2023]

        # M_active fit on Year != 2023 in tr
        m_act = make_hgbc()
        m_act.fit(X.iloc[tr_active], y[tr_active])

        # M_2023 fit on Year == 2023 in tr
        m_23 = make_hgbc()
        m_23.fit(X.iloc[tr_2023], y[tr_2023])

        # Route on validation
        va_year = year_train[va]
        p_va = np.zeros(len(va), dtype=np.float64)
        is_23_va = (va_year == 2023)
        # M_active predicts non-2023 val rows
        if (~is_23_va).any():
            p_va[~is_23_va] = m_act.predict_proba(X.iloc[va[~is_23_va]])[:, 1]
        # M_2023 predicts 2023 val rows
        if is_23_va.any():
            p_va[is_23_va] = m_23.predict_proba(X.iloc[va[is_23_va]])[:, 1]
        oof[va] = p_va

        # Route on test (averaged across folds)
        is_23_te = (year_test == 2023)
        p_test = np.zeros(len(X_test), dtype=np.float64)
        if (~is_23_te).any():
            p_test[~is_23_te] = m_act.predict_proba(X_test.iloc[~is_23_te])[:, 1]
        if is_23_te.any():
            p_test[is_23_te] = m_23.predict_proba(X_test.iloc[is_23_te])[:, 1]
        tp += p_test / N_FOLDS

        wall = time.time() - t0
        fold_walls.append(wall)
        s = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(s)
        print(f"  fold{k}: AUC={s:.5f}  wall={wall:.1f}s  "
              f"|active={len(tr_active)} |2023={len(tr_2023)} "
              f"|val_2023={is_23_va.sum()} val_active={(~is_23_va).sum()}")
    auc = float(roc_auc_score(y, oof))
    print(f"  STRAT OOF AUC: {auc:.5f}  fold_std={np.std(fold_aucs):.5f}  "
          f"total_wall={sum(fold_walls):.0f}s")
    return oof, tp, auc, fold_aucs, fold_walls


# ----------------------------- Part B: AV reweight -------------------
def adversarial_weights(X, X_test, y_dummy_train, y_dummy_test):
    """LGBM 5-fold OOF on is_test label; compute importance weights."""
    print("\n=== Part B step 1: Adversarial-validation classifier ===")
    # Concat train + test, label = is_test
    X_all = pd.concat([X, X_test], ignore_index=True)
    y_av = np.concatenate([np.zeros(len(X)), np.ones(len(X_test))]).astype(int)
    n_train = len(X)
    # Convert categorical / int columns for LightGBM
    cat_cols = [c for c in X_all.columns
                if str(X_all[c].dtype) == "category"]
    # LightGBM consumes pandas categoricals natively
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y_av)), y_av))
    oof_av = np.zeros(len(y_av), dtype=np.float64)
    fold_aucs = []
    walls = []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        ds_tr = lgb.Dataset(X_all.iloc[tr], y_av[tr],
                            categorical_feature=cat_cols)
        ds_va = lgb.Dataset(X_all.iloc[va], y_av[va],
                            categorical_feature=cat_cols)
        params = dict(objective="binary", metric="auc",
                      learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=200, feature_fraction=0.9,
                      bagging_fraction=0.9, bagging_freq=5,
                      verbose=-1, seed=SEED)
        m = lgb.train(params, ds_tr, num_boost_round=1500,
                      valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
        oof_av[va] = m.predict(X_all.iloc[va], num_iteration=m.best_iteration)
        s = float(roc_auc_score(y_av[va], oof_av[va]))
        fold_aucs.append(s)
        walls.append(time.time() - t0)
        print(f"  AV fold{k}: AUC={s:.5f}  best_iter={m.best_iteration}  "
              f"wall={walls[-1]:.1f}s")
    auc_av = float(roc_auc_score(y_av, oof_av))
    print(f"  AV OOF AUC: {auc_av:.5f}  total_wall={sum(walls):.0f}s")

    # Importance weights for TRAIN rows only
    p_test_train = oof_av[:n_train]  # P(is_test | x) for train rows
    # Clip to avoid extreme values (esp. p≈1)
    p_clip = np.clip(p_test_train, 1e-3, 1 - 1e-3)
    w = p_clip / (1.0 - p_clip)
    w = np.clip(w, 0.1, 10.0)
    print(f"  weights: mean={w.mean():.3f}  median={np.median(w):.3f}  "
          f"min={w.min():.3f}  max={w.max():.3f}  "
          f"frac_clipped_lo={(w==0.1).mean():.3%}  "
          f"frac_clipped_hi={(w==10.0).mean():.3%}")
    return w, auc_av, fold_aucs, walls


def part_b_e3_advweight(X, y, X_test, w):
    """Re-train e3-style HGBC with sample_weight=w."""
    print("\n=== Part B step 2: e3_hgbc with adversarial weights ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    oof = np.zeros(len(y), dtype=np.float64)
    tp = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs = []; fold_walls = []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        m = make_hgbc()
        m.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(X_test)[:, 1] / N_FOLDS
        wall = time.time() - t0; fold_walls.append(wall)
        s = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(s)
        print(f"  fold{k}: AUC={s:.5f}  iters={m.n_iter_}  wall={wall:.1f}s")
    auc = float(roc_auc_score(y, oof))
    print(f"  STRAT OOF AUC: {auc:.5f}  fold_std={np.std(fold_aucs):.5f}")
    return oof, tp, auc, fold_aucs, fold_walls


# ----------------------------- Part C: Year + AV combined ------------
def part_c_year_specialist_advweight(X, y, X_test, year_train, year_test, w):
    print("\n=== Part C: Year-specialist + AV weights ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float64)
    tp = np.zeros(len(X_test), dtype=np.float64)
    fold_walls = []; fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        tr_active = tr[year_train[tr] != 2023]
        tr_2023 = tr[year_train[tr] == 2023]

        m_act = make_hgbc()
        m_act.fit(X.iloc[tr_active], y[tr_active],
                  sample_weight=w[tr_active])

        m_23 = make_hgbc()
        m_23.fit(X.iloc[tr_2023], y[tr_2023],
                 sample_weight=w[tr_2023])

        va_year = year_train[va]
        p_va = np.zeros(len(va), dtype=np.float64)
        is_23_va = (va_year == 2023)
        if (~is_23_va).any():
            p_va[~is_23_va] = m_act.predict_proba(X.iloc[va[~is_23_va]])[:, 1]
        if is_23_va.any():
            p_va[is_23_va] = m_23.predict_proba(X.iloc[va[is_23_va]])[:, 1]
        oof[va] = p_va

        is_23_te = (year_test == 2023)
        p_test = np.zeros(len(X_test), dtype=np.float64)
        if (~is_23_te).any():
            p_test[~is_23_te] = m_act.predict_proba(X_test.iloc[~is_23_te])[:, 1]
        if is_23_te.any():
            p_test[is_23_te] = m_23.predict_proba(X_test.iloc[is_23_te])[:, 1]
        tp += p_test / N_FOLDS

        wall = time.time() - t0; fold_walls.append(wall)
        s = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(s)
        print(f"  fold{k}: AUC={s:.5f}  wall={wall:.1f}s")
    auc = float(roc_auc_score(y, oof))
    print(f"  STRAT OOF AUC: {auc:.5f}  fold_std={np.std(fold_aucs):.5f}")
    return oof, tp, auc, fold_aucs, fold_walls


# ----------------------------- per-Year segment AUC ------------------
def per_year_auc(name, oof, y, year_train):
    out = {}
    for yr in [2022, 2023, 2024, 2025]:
        mask = (year_train == yr)
        if mask.sum() > 0 and len(np.unique(y[mask])) > 1:
            out[int(yr)] = float(roc_auc_score(y[mask], oof[mask]))
        else:
            out[int(yr)] = float("nan")
    print(f"  per-Year AUC [{name}]: " +
          " ".join([f"{yr}={out[yr]:.5f}" for yr in [2022, 2023, 2024, 2025]]))
    return out


# ----------------------------- Part D: stack-add ----------------------
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
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def predicted_lb(auc, rho):
    base_lb = PRIMARY_LB + (auc - PRIMARY_S)
    if rho >= RHO_TIE: return base_lb
    if rho >= 0.995:   return base_lb - 0.0001
    if rho >= 0.99:    return base_lb - 0.00025
    return base_lb - 0.0004


def min_meta_gate(cand_oof, primary_oof, y):
    """3-feat LR over {primary, candidate, |delta|}, 5-fold OOF AUC."""
    delta = np.abs(cand_oof - primary_oof)
    F = np.column_stack([primary_oof, cand_oof, delta])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        pred[va] = lr.predict_proba(F[va])[:, 1]
    return float(roc_auc_score(y, pred))


# ----------------------------- pool loaders for K=22 -----------------
POOL_KEEP = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"), ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"), ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"), ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"), ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    ("realmlp", "realmlp"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]
TOP_3_D9 = [
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
]


def load_pool(names_files):
    Xs_oof, Xs_test, names = [], [], []
    for label, fname in names_files:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    return Xs_oof, Xs_test, names


def stack_eval(name, Xs_oof, Xs_test, names, y, primary_test):
    K = len(names)
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, primary_test)
    pred_lb = predicted_lb(auc, rho)
    delta = (auc - PRIMARY_S) * 1e4
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"  {name} (K={K}): OOF={auc:.5f}  Δprim={delta:+.2f}bp  "
          f"ρ={rho:.5f}  predLB={pred_lb:.5f} (Δ {(pred_lb-PRIMARY_LB)*1e4:+.2f}bp)")
    return dict(K=K, strat_oof=auc, delta_primary_bp=delta,
                rho_vs_primary_test=float(rho), pred_lb=float(pred_lb),
                delta_lb_bp=float((pred_lb - PRIMARY_LB) * 1e4),
                l1_top5={k: v for k, v in
                         sorted(l1.items(), key=lambda kv: -kv[1])[:5]})


# ----------------------------- main -----------------------------------
def main():
    t_all = time.time()
    train, test, sample_sub, X, y, X_test = load_data()
    year_train = train["Year"].values
    year_test = test["Year"].values

    # Load m5q OOF for per-Year baseline & PRIMARY test for ρ
    oof_m5q = np.load(ART / "oof_m5q_strat.npy")[:, 1].astype(np.float64)
    # PRIMARY OOF + test (d9c_Sd, faithful surrogate)
    primary_oof = np.load(ART / f"oof_{PRIMARY_NAME}_strat.npy")[:, 1].astype(np.float64)
    primary_test = np.load(ART / f"test_{PRIMARY_NAME}_strat.npy")[:, 1].astype(np.float64)
    # Best LB test array (d9f K=21 swap) for ρ-vs-actual-LB-best
    lb_best_test = np.load(ART / f"test_{PRIMARY_TEST_LB_BEST}_strat.npy")[:, 1].astype(np.float64)
    primary_oof_auc = float(roc_auc_score(y, primary_oof))
    print(f"PRIMARY ({PRIMARY_NAME}): OOF AUC={primary_oof_auc:.5f}  "
          f"(state-block PRIMARY_S={PRIMARY_S})")
    print(f"m5q OOF AUC: {float(roc_auc_score(y, oof_m5q)):.5f}")
    per_year_auc("m5q", oof_m5q, y, year_train)
    per_year_auc("PRIMARY", primary_oof, y, year_train)

    # ---------- Part A
    oof_A, test_A, auc_A, faucs_A, fwalls_A = part_a_year_specialist(
        X, y, X_test, year_train, year_test)
    pyA = per_year_auc("year_specialist", oof_A, y, year_train)
    np.save(ART / "oof_d12_year_specialist_strat.npy",
            np.column_stack([1 - oof_A, oof_A]))
    np.save(ART / "test_d12_year_specialist_strat.npy",
            np.column_stack([1 - test_A, test_A]))

    # ---------- Part B step 1: AV classifier
    w, auc_av, av_fold_aucs, av_walls = adversarial_weights(X, X_test, None, None)

    # ---------- Part B step 2: e3 advweight
    oof_B, test_B, auc_B, faucs_B, fwalls_B = part_b_e3_advweight(
        X, y, X_test, w)
    pyB = per_year_auc("e3_advweight", oof_B, y, year_train)
    np.save(ART / "oof_d12_e3_advweight_strat.npy",
            np.column_stack([1 - oof_B, oof_B]))
    np.save(ART / "test_d12_e3_advweight_strat.npy",
            np.column_stack([1 - test_B, test_B]))

    # ---------- Part C: combined
    oof_C, test_C, auc_C, faucs_C, fwalls_C = part_c_year_specialist_advweight(
        X, y, X_test, year_train, year_test, w)
    pyC = per_year_auc("year_advweight", oof_C, y, year_train)
    np.save(ART / "oof_d12_year_advweight_strat.npy",
            np.column_stack([1 - oof_C, oof_C]))
    np.save(ART / "test_d12_year_advweight_strat.npy",
            np.column_stack([1 - test_C, test_C]))

    # ---------- Part D: stack-add eval
    print("\n=== Part D: stack-add evaluation ===")
    base_oof, base_test, base_names = load_pool(POOL_KEEP)
    d9_oof, d9_test, d9_names = load_pool(TOP_3_D9)
    fm_oof = np.load(ART / "oof_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    fm_test = np.load(ART / "test_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    r14_l4_oof = np.load(ART / "oof_d9b_R14_L4_strat.npy")[:, 1].astype(np.float64)
    r14_l4_test = np.load(ART / "test_d9b_R14_L4_strat.npy")[:, 1].astype(np.float64)

    # Base K=21 stack equals d9c Sa (PRIMARY-keep + R6/R10/R7 + R14_L4 + FM)
    candidates = [
        ("year_specialist", oof_A, test_A, auc_A),
        ("e3_advweight", oof_B, test_B, auc_B),
        ("year_advweight", oof_C, test_C, auc_C),
    ]
    cand_results = {}
    for cname, c_oof, c_test, c_auc in candidates:
        rho_test, _ = spearmanr(c_test, primary_test)
        rho_lb, _ = spearmanr(c_test, lb_best_test)
        rho_oof_corr = float(np.corrcoef(c_oof, primary_oof)[0, 1])
        # min-meta gate vs PRIMARY OOF
        meta_auc = min_meta_gate(c_oof, primary_oof, y)
        delta_meta_vs_primary_bp = (meta_auc - primary_oof_auc) * 1e4
        print(f"\n  [{cname}] standalone OOF={c_auc:.5f}  "
              f"ρ(test vs d9c_Sd PRIMARY)={rho_test:.5f}  "
              f"ρ(test vs d9f_K21 LB-best)={rho_lb:.5f}  "
              f"min-meta auc={meta_auc:.5f}  Δvs PRIMARY={delta_meta_vs_primary_bp:+.2f}bp")
        # K=22 add stack
        Xs = base_oof + d9_oof + [r14_l4_oof, fm_oof, c_oof]
        Ts = base_test + d9_test + [r14_l4_test, fm_test, c_test]
        Ns = base_names + d9_names + ["R14_L4", "FM", cname]
        r = stack_eval(f"K22_add_{cname}", Xs, Ts, Ns, y, primary_test)
        cand_results[cname] = dict(
            standalone_oof=c_auc,
            rho_vs_primary_test=float(rho_test),
            rho_vs_lb_best_test=float(rho_lb),
            rho_oof_corrcoef=rho_oof_corr,
            min_meta_auc=meta_auc,
            min_meta_delta_bp_vs_primary=float(delta_meta_vs_primary_bp),
            min_meta_pass=bool(delta_meta_vs_primary_bp >= 0.0),
            k22_stack=r,
        )

    # Reference K=21 PRIMARY for clarity
    Xs = base_oof + d9_oof + [r14_l4_oof, fm_oof]
    Ts = base_test + d9_test + [r14_l4_test, fm_test]
    Ns = base_names + d9_names + ["R14_L4", "FM"]
    print()
    ref_k21 = stack_eval("K21_PRIMARY_ref", Xs, Ts, Ns, y, primary_test)

    # ---------- write JSON
    final = dict(
        primary=dict(name=PRIMARY_NAME, oof=primary_oof_auc,
                     state_block_oof=PRIMARY_S, state_block_lb=PRIMARY_LB),
        adversarial=dict(av_oof_auc=auc_av, fold_aucs=av_fold_aucs,
                         fold_walls_s=av_walls,
                         w_summary=dict(mean=float(w.mean()),
                                        median=float(np.median(w)),
                                        min=float(w.min()),
                                        max=float(w.max()),
                                        frac_clip_lo=float((w == 0.1).mean()),
                                        frac_clip_hi=float((w == 10.0).mean()))),
        per_year_auc=dict(m5q=per_year_auc("m5q_chk", oof_m5q, y, year_train),
                          PRIMARY=per_year_auc("PRIMARY_chk", primary_oof, y, year_train),
                          year_specialist=pyA,
                          e3_advweight=pyB,
                          year_advweight=pyC),
        candidates=cand_results,
        reference_k21=ref_k21,
        total_wall_s=time.time() - t_all,
    )
    (ART / "d12_year_specialist_advweight_results.json").write_text(
        json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d12_year_specialist_advweight_results.json")
    print(f"total wall: {time.time()-t_all:.0f}s")

    # Summary table
    print("\n" + "=" * 84)
    print(f"{'candidate':<22s} {'std_OOF':>8s} {'ρ_test':>7s} {'minMetaΔ':>9s} "
          f"{'K22_OOF':>8s} {'K22_predLB':>10s} {'ΔLB':>6s}")
    print("-" * 84)
    for nm, r in cand_results.items():
        k = r["k22_stack"]
        print(f"{nm:<22s} {r['standalone_oof']:>8.5f} "
              f"{r['rho_vs_primary_test']:>7.4f} "
              f"{r['min_meta_delta_bp_vs_primary']:>+8.2f}bp "
              f"{k['strat_oof']:>8.5f} "
              f"{k['pred_lb']:>10.5f} {k['delta_lb_bp']:>+5.2f}")


if __name__ == "__main__":
    main()
