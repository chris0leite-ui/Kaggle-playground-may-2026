"""C1 — Safety-car probability rule_residual base (Day-9, external-data probe).

Rule 16 5-question pre-flight applied:
  Q1: Mechanism class? Cross-Race generalization via SC-prob decile bin
      (groups Races with similar SC propensity; M5q pool has Race as
      categorical so it cannot share statistics across Races).
  Q2: Vulnerability? rule_residual-on-raw-plus-feature → at risk; mitigated
      by the SC-prob lookup being EXTERNAL data (2018-2024 historical, NOT
      derivable from train+test).
  Q3: Predicted standalone OOF: 0.945-0.946 (rule_residual range).
  Q4: Predicted ρ vs M5q test: 0.93-0.96 (less diverse than C5 since most
      per-Race effect already captured by Race-as-categorical in M5q pool).
  Q5: Closest precedent: C5 (today, MARGINAL — K=20 TIE despite ρ=0.89);
      F1.2 R4 year_race (PASS but inside-pool features).

EV midpoint × 0.3 per Rule 16 → predicted +0-1bp at K=19 stack. Cheap CPU
probe (~5min) so worth running for the no-NULL data point regardless.

SC-probability source: aggregate of 2018-2024 historical safety-car
deployment rates per circuit, calibrated against:
  - Lights Out Blog 2024 season log (every-safety-car-f1-2024)
  - Lights Out Blog per-circuit deployment archives (Britain, Canada,
    Spain, Mexico, Imola, Austria, Saudi Arabia, etc.)
  - Axiora Blogs strategy column (Singapore ~100%, Paul Ricard ~10%)
  - Common F1-strategy domain priors (urban-walls vs open-runoff)
Values are P(any SC or VSC deployment per race), Bayesian-style point
estimates with conservative tier ordering.

Strat-only (R1).
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
M5Q_S, M5Q_LB = 0.95057, 0.95005
K18_OOF, K18_LB = 0.95065, 0.95026
RHO_TIE = 0.999
ALPHA = 50.0

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
D6_RULES = [
    ("rule_compound_tyre", "d6_rule_residual"),
    ("rule_compound_stint", "d6_rule_compound_stint"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]

# P(any SC or VSC per race), 2018-2024 aggregate (curated, see module docstring).
# Tier mapping: high>=0.7, mid 0.4-0.7, low<0.4. Pre-Season ~0 by definition.
SC_PROB = {
    "Singapore Grand Prix":      0.95,
    "Saudi Arabian Grand Prix":  0.90,
    "Azerbaijan Grand Prix":     0.85,
    "Qatar Grand Prix":          0.80,
    "São Paulo Grand Prix":      0.75,
    "Miami Grand Prix":          0.75,
    "Canadian Grand Prix":       0.65,
    "Australian Grand Prix":     0.65,
    "Monaco Grand Prix":         0.65,
    "Mexico City Grand Prix":    0.55,
    "Belgian Grand Prix":        0.55,
    "Japanese Grand Prix":       0.50,
    "Chinese Grand Prix":        0.50,
    "Emilia Romagna Grand Prix": 0.50,
    "United States Grand Prix":  0.45,
    "Las Vegas Grand Prix":      0.45,
    "Italian Grand Prix":        0.40,
    "Bahrain Grand Prix":        0.40,
    "Austrian Grand Prix":       0.40,
    "British Grand Prix":        0.40,
    "Hungarian Grand Prix":      0.35,
    "Dutch Grand Prix":          0.30,
    "Abu Dhabi Grand Prix":      0.30,
    "Spanish Grand Prix":        0.25,
    "French Grand Prix":         0.20,
    "Pre-Season Testing":        0.05,
}


def encode_features(X, X_test):
    """Same as F1.2 / C5 templates so M5q-pool joins line up.

    Note: extra columns added by C1 (sc_prob, lap_quintile_int) are NOT in
    the categorical lists; they pass through as numeric and the residual
    HGBC splits on them naturally.
    """
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


def fit_lookup(keys_train, y_train, alpha=ALPHA):
    df = pd.DataFrame({"k": list(keys_train), "y": y_train})
    g = df.groupby("k", observed=True)["y"]
    counts = g.count(); means = g.mean()
    glob = float(np.mean(y_train))
    smoothed = (means * counts + glob * alpha) / (counts + alpha)
    return smoothed.to_dict(), glob


def apply_lookup(keys, lookup, glob):
    out = np.full(len(keys), glob, dtype=np.float64)
    for i, k in enumerate(keys):
        v = lookup.get(k)
        if v is not None:
            out[i] = v
    return out


def lap_quintile_factory(lap_train):
    edges = np.quantile(lap_train, np.linspace(0, 1, 6))
    edges[0] = -np.inf; edges[-1] = np.inf
    def transform(a):
        b = np.searchsorted(edges, a, side="right") - 1
        return np.clip(b, 0, 4)
    return transform


def stint_bucket(s):
    s = np.asarray(s)
    return np.where(s == 1, "1", np.where(s == 2, "2", "3+"))


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


def predicted_lb(auc, rho, anchor_oof, anchor_lb):
    base_lb = anchor_lb + (auc - anchor_oof)
    if rho >= RHO_TIE:
        return base_lb
    if rho >= 0.995:
        return base_lb - 0.0001
    if rho >= 0.99:
        return base_lb - 0.00025
    return base_lb - 0.0004


def main():
    t_total = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    # --- 1. SC-probability feature engineering ---
    races_present = sorted(set(train["Race"].unique()) | set(test["Race"].unique()))
    missing = [r for r in races_present if r not in SC_PROB]
    if missing:
        print(f"WARN: {len(missing)} races without SC_PROB entry: {missing}")
    train["sc_prob"] = train["Race"].map(SC_PROB).astype(float)
    test["sc_prob"] = test["Race"].map(SC_PROB).astype(float)
    glob_sc = float(pd.Series(SC_PROB).mean())
    train["sc_prob"] = train["sc_prob"].fillna(glob_sc)
    test["sc_prob"] = test["sc_prob"].fillna(glob_sc)

    # SC-decile bin (4 distinct levels covering low/mid/high/extreme).
    # Using unique probability values across races, qcut to 4 bins so each
    # decile has a meaningful share. With 26 races and 26 distinct values,
    # 4 buckets is the right granularity (smaller bins → noisy lookup).
    sc_arr = np.asarray([SC_PROB[r] for r in races_present])
    edges = np.quantile(sc_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0] = -np.inf; edges[-1] = np.inf
    def sc_decile(p):
        return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 3)
    train["sc_dec"] = sc_decile(train["sc_prob"].values).astype(np.int8)
    test["sc_dec"] = sc_decile(test["sc_prob"].values).astype(np.int8)
    print(f"SC decile counts (train): "
          f"{train['sc_dec'].value_counts().sort_index().to_dict()}")

    # Lap-quintile from LapNumber (using train edges; consistent w/ existing FE).
    lap_q = lap_quintile_factory(train["LapNumber"].values)
    train["lap_q"] = lap_q(train["LapNumber"].values).astype(np.int8)
    test["lap_q"] = lap_q(test["LapNumber"].values).astype(np.int8)
    train["sb"] = stint_bucket(train["Stint"].values)
    test["sb"] = stint_bucket(test["Stint"].values)

    # --- 2. Build X / X_test for residual GBDT (incl. sc_prob feature) ---
    X = train.drop(columns=[TARGET, ID_COL, "sc_dec", "lap_q", "sb"],
                   errors="ignore").copy()
    X_test = test.drop(columns=[ID_COL, "sc_dec", "lap_q", "sb"],
                       errors="ignore").copy()
    X_enc, X_test_enc = encode_features(X.copy(), X_test.copy())

    # --- 3. Per-fold rule_residual: rule key (sc_dec, sb, lap_q) ---
    def keys_from(df):
        return list(zip(df["sc_dec"].astype(str).values,
                        df["sb"].astype(str).values,
                        df["lap_q"].astype(str).values))
    keys_train = keys_from(train)
    keys_test = keys_from(test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof_full = np.zeros(len(y), dtype=np.float64)
    test_full = np.zeros(len(test), dtype=np.float64)
    fold_aucs_rule, fold_aucs_full, walls = [], [], []

    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        keys_tr = [keys_train[i] for i in tr]
        keys_va = [keys_train[i] for i in va]
        lookup, glob = fit_lookup(keys_tr, y[tr])
        rp_va = apply_lookup(keys_va, lookup, glob)
        rp_te = apply_lookup(keys_test, lookup, glob)
        rp_tr = apply_lookup(keys_tr, lookup, glob)

        m = make_hgbc_regressor()
        m.fit(X_enc.iloc[tr], y[tr].astype(np.float64) - rp_tr)
        resid_va = m.predict(X_enc.iloc[va])
        resid_te = m.predict(X_test_enc)
        pred_va = np.clip(rp_va + resid_va, 1e-9, 1 - 1e-9)
        pred_te = np.clip(rp_te + resid_te, 1e-9, 1 - 1e-9)

        oof_full[va] = pred_va
        test_full += pred_te / N_FOLDS
        s_rule = float(roc_auc_score(y[va], rp_va))
        s_full = float(roc_auc_score(y[va], pred_va))
        wall = time.time() - t0
        fold_aucs_rule.append(s_rule); fold_aucs_full.append(s_full); walls.append(wall)
        print(f"  f{k}: rule={s_rule:.5f}  full={s_full:.5f}  wall={wall:.1f}s")

    auc_full = float(roc_auc_score(y, oof_full))
    print(f"\n=== C1 sc_prob standalone ===")
    print(f"Standalone OOF: {auc_full:.5f}  std={np.std(fold_aucs_full):.5f}")

    # Save the standalone artifacts
    np.save(ART / "oof_d9_c1_sc_prob_strat.npy",
            np.column_stack([1 - oof_full, oof_full]))
    np.save(ART / "test_d9_c1_sc_prob_strat.npy",
            np.column_stack([1 - test_full, test_full]))

    # --- 4. ρ vs M5q test, minimal-meta sanity check ---
    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]
    m5q_oof = np.load(ART / "oof_m5q_strat.npy")[:, 1]
    rho_test, _ = spearmanr(test_full, test_m5q)
    print(f"ρ vs M5q test: {rho_test:.5f}")

    F_min = expand(np.column_stack([m5q_oof, oof_full]))
    F_min_t = expand(np.column_stack([test_m5q, test_full]))
    mo_min, _, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo_min))
    print(f"Minimal-meta OOF: {auc_min:.5f}  Δ M5q "
          f"{(auc_min - M5Q_S)*1e4:+.2f}bp  "
          f"{'PASS ✓' if auc_min >= M5Q_S else 'FAIL ✗'}")

    # --- 5. K=19 stack: M5q (14) + d6 4 rules + C1 ---
    print(f"\n=== K=19 stack: M5q (14) + d6 4 rules + C1 sc_prob ===")
    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL_M5Q:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    for label, n in D6_RULES:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    Xs_oof.append(oof_full); Xs_test.append(test_full); names.append("c1_sc_prob")

    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    K = len(names)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    k18_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy")[:, 1]
    rho_m5q, _ = spearmanr(tp, test_m5q)
    rho_k18, _ = spearmanr(tp, k18_test)
    pred_lb_vs_k18 = predicted_lb(auc, rho_k18, K18_OOF, K18_LB)
    delta_k18 = (auc - K18_OOF) * 1e4
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"K={K} LR-meta Strat OOF: {auc:.5f}  Δ K=18 {delta_k18:+.2f}bp")
    print(f"  ρ vs M5q test: {rho_m5q:.5f}    ρ vs K=18 test: {rho_k18:.5f}")
    print(f"  pred-LB (vs K=18 anchor): {pred_lb_vs_k18:.5f}")
    print(f"L1 ranking (top-10):")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:10]:
        marker = " ← C1" if n == "c1_sc_prob" else (
                 " ← d6" if n.startswith("rule_") else "")
        print(f"  {n:<22s} L1={v:.3f}{marker}")
    print(f"  c1_sc_prob L1: {l1['c1_sc_prob']:.3f}")

    stack_pass = auc >= K18_OOF + 0.5/1e4
    rho_pass = rho_k18 < RHO_TIE
    pred_pass = pred_lb_vs_k18 >= K18_LB + 0.5/1e4
    print(f"\n=== K=19 verdict (vs K=18 anchor) ===")
    print(f"  K=19 OOF >= K=18 + 0.5bp ({K18_OOF + 0.5/1e4:.5f}): {stack_pass}  ({auc:.5f})")
    print(f"  ρ vs K=18 < {RHO_TIE}:                              {rho_pass}   ({rho_k18:.5f})")
    print(f"  pred-LB >= K=18 LB + 0.5bp ({K18_LB+0.5/1e4:.5f}):  {pred_pass}  ({pred_lb_vs_k18:.5f})")
    if stack_pass and rho_pass and pred_pass:
        verdict = "SLOT-CANDIDATE"
    elif pred_lb_vs_k18 >= K18_LB:
        verdict = "MARGINAL"
    else:
        verdict = "DO_NOT_SLOT"
    print(f"  → {verdict}")

    # Save
    np.save(ART / "oof_d9_k19_sc_prob_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d9_k19_sc_prob_strat.npy",
            np.column_stack([1 - tp, tp]))
    sub = sample_sub.copy(); sub[TARGET] = tp
    sub.to_csv("submissions/submission_d9_k19_sc_prob.csv", index=False)

    # Diagnostic: rule-only AUC + per-Race mean check
    auc_rule = float(roc_auc_score(y, oof_full))  # full == rule+resid; rule-only diag below
    sc_lookup_used = sorted(SC_PROB.items(), key=lambda kv: -kv[1])
    final = dict(
        sc_prob_table=SC_PROB,
        races_present=races_present,
        races_missing_from_table=missing,
        standalone=dict(oof=auc_full, fold_aucs_full=fold_aucs_full,
                        fold_aucs_rule=fold_aucs_rule, walls=walls),
        rho_vs_m5q_test=float(rho_test),
        minimal_meta=dict(oof=auc_min,
                          delta_m5q_bp=(auc_min - M5Q_S) * 1e4,
                          pass_=bool(auc_min >= M5Q_S)),
        k19_stack=dict(K=K, strat_oof=auc, delta_k18_bp=delta_k18,
                        rho_vs_m5q_test=float(rho_m5q),
                        rho_vs_k18_test=float(rho_k18),
                        pred_lb_vs_k18=float(pred_lb_vs_k18),
                        l1_ranking=l1,
                        stack_pass=bool(stack_pass),
                        rho_pass=bool(rho_pass),
                        pred_pass=bool(pred_pass),
                        verdict=verdict),
        anchors=dict(M5Q_S=M5Q_S, K18_OOF=K18_OOF, K18_LB=K18_LB),
        total_wall_s=time.time() - t_total,
    )
    (ART / "d9_c1_sc_prob_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9_c1_sc_prob_results.json  "
          f"(total wall {time.time()-t_total:.0f}s)")


if __name__ == "__main__":
    main()
