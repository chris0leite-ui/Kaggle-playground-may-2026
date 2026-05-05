"""C5 — multi-rule extension with prev/next-compound neighbor signal (Day-9).

Rule 16 5-question pre-flight applied (rule_residual-on-raw → at risk of
T1.3-mode collapse; mitigated by using `prev_compound` / `next_compound`
which are NOT in M5q's 14 raw features → genuinely orthogonal signal).

Rules added (Bayesian-smoothed lookups, alpha=50, F1.2 template):
  R5: prev_compound × Compound × stint_bucket
        — Stint-2 specialist signal (P4: 18.9% SOFT→HARD vs 75.4% WET→HARD)
  R6: next_compound × Compound
        — strategy signal (P5: known for 68% of test rows)

prev_compound / next_compound computed via shift-1 within (Race,Driver,Year)
on TRAIN ∪ TEST stint-level table (Compound is observed in test → leak-free).

Gates per audit §5:
  - Each rule min-meta vs M5q (K=2 LR): Δ >= 0 PASS
  - K=20 stack: OOF >= d6_k18 + 0.5bp AND ρ < 0.999
  - Predicted LB >= d6_k18 LB + 0.5bp

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


def add_neighbor_compounds(train: pd.DataFrame, test: pd.DataFrame):
    """Compute prev_compound and next_compound for all rows.

    Uses (Race, Driver, Year, Stint, Compound) from train+test combined
    (Compound is observed in test → leak-free). Coverage: prev_compound
    fires for Stint>=2 (~70% of rows); next_compound fires for rows with
    a same-(Race,Driver,Year) Stint+1 entry (~70% of rows per P7).
    """
    cols = ["Race", "Driver", "Year", "Stint", "Compound"]
    stint_table = (pd.concat([train[cols], test[cols]], ignore_index=True)
                   .drop_duplicates(cols)
                   .sort_values(["Race", "Driver", "Year", "Stint"])
                   .reset_index(drop=True))
    g = stint_table.groupby(["Race", "Driver", "Year"], sort=False)["Compound"]
    stint_table["prev_compound"] = g.shift(1).fillna("NONE")
    stint_table["next_compound"] = g.shift(-1).fillna("NONE")
    join_cols = ["Race", "Driver", "Year", "Stint"]
    nbr = stint_table[join_cols + ["prev_compound", "next_compound"]]
    train_out = train.merge(nbr, on=join_cols, how="left")
    test_out = test.merge(nbr, on=join_cols, how="left")
    train_out["prev_compound"] = train_out["prev_compound"].fillna("NONE")
    train_out["next_compound"] = train_out["next_compound"].fillna("NONE")
    test_out["prev_compound"] = test_out["prev_compound"].fillna("NONE")
    test_out["next_compound"] = test_out["next_compound"].fillna("NONE")
    return train_out, test_out


def stint_bucket(s):
    s = np.asarray(s)
    out = np.where(s == 1, "1", np.where(s == 2, "2", "3+"))
    return out


def build_rule_residual(rule_name, key_cols, train, test, X_enc, X_test_enc,
                        y, splits):
    """Build one rule_residual variant in 5-fold Strat. Returns (oof, test_p)."""
    print(f"\n--- Building rule '{rule_name}' (key_cols={key_cols}) ---")

    def keys_from(df):
        cols = [df[c].astype(str).values if c != "stint_bucket"
                else stint_bucket(df["Stint"].values) for c in key_cols]
        return list(zip(*cols))

    keys_train = keys_from(train)
    keys_test = keys_from(test)

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
    print(f"  → standalone OOF: {auc_full:.5f}  std={np.std(fold_aucs_full):.5f}  "
          f"total wall={sum(walls):.1f}s")
    return oof_full, test_full, auc_full


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

    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]
    m5q_oof = np.load(ART / "oof_m5q_strat.npy")[:, 1]

    # Sanity: load K=18 anchor for the stack-vs-K=18 verdict
    k18_oof = np.load(ART / "oof_d6_k18_multi_rule_strat.npy")[:, 1]
    k18_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy")[:, 1]
    print(f"Loaded anchors: M5q OOF {roc_auc_score(y, m5q_oof):.5f} "
          f"(expect {M5Q_S:.5f}); K=18 OOF {roc_auc_score(y, k18_oof):.5f} "
          f"(expect {K18_OOF:.5f})")

    # Add neighbor compounds
    train_n, test_n = add_neighbor_compounds(train, test)
    pc = train_n["prev_compound"]
    nc = train_n["next_compound"]
    pc_t = test_n["prev_compound"]
    nc_t = test_n["next_compound"]
    print(f"prev_compound coverage train: {(pc != 'NONE').mean()*100:.1f}%, "
          f"test: {(pc_t != 'NONE').mean()*100:.1f}%")
    print(f"next_compound coverage train: {(nc != 'NONE').mean()*100:.1f}%, "
          f"test: {(nc_t != 'NONE').mean()*100:.1f}%")

    X = train_n.drop(columns=[TARGET, ID_COL, "prev_compound", "next_compound"],
                     errors="ignore").copy()
    X_test = test_n.drop(columns=[ID_COL, "prev_compound", "next_compound"],
                         errors="ignore").copy()
    X_enc, X_test_enc = encode_features(X.copy(), X_test.copy())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Define rules: pass key_cols via train_n / test_n (which have prev/next).
    rules = {
        "prev_comp_stint_bucket": ["prev_compound", "Compound", "stint_bucket"],
        "next_comp_compound": ["next_compound", "Compound"],
    }

    new_oofs, new_tests, results_per_rule = {}, {}, {}
    for rule_name, key_cols in rules.items():
        oof, tp, auc = build_rule_residual(rule_name, key_cols,
                                            train_n, test_n, X_enc, X_test_enc,
                                            y, splits)
        new_oofs[rule_name] = oof; new_tests[rule_name] = tp
        rho_test, _ = spearmanr(tp, test_m5q)
        F_min = expand(np.column_stack([m5q_oof, oof]))
        F_min_t = expand(np.column_stack([test_m5q, tp]))
        mo_min, _, _ = fit_lr_meta(F_min, F_min_t, y)
        auc_min = float(roc_auc_score(y, mo_min))
        results_per_rule[rule_name] = dict(
            standalone_oof=auc, rho_vs_m5q_test=float(rho_test),
            minimal_meta_oof=auc_min,
            minimal_meta_delta_m5q_bp=(auc_min - M5Q_S) * 1e4,
            minimal_meta_pass=bool(auc_min >= M5Q_S),
        )
        print(f"  ρ vs M5q test: {rho_test:.5f}")
        print(f"  Minimal-meta OOF: {auc_min:.5f}  Δ M5q "
              f"{(auc_min - M5Q_S)*1e4:+.2f}bp  "
              f"{'PASS ✓' if auc_min >= M5Q_S else 'FAIL ✗'}")
        np.save(ART / f"oof_d9_c5_{rule_name}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d9_c5_{rule_name}_strat.npy",
                np.column_stack([1 - tp, tp]))

    # K=20 stack: M5q (14) + d6 4 rules + 2 new rules
    print(f"\n=== K=20 stack: M5q (14) + d6 4 rules + 2 new neighbor rules ===")
    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL_M5Q:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    for label, n in D6_RULES:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    for rule_name in rules:
        Xs_oof.append(new_oofs[rule_name])
        Xs_test.append(new_tests[rule_name])
        names.append(f"c5_{rule_name}")

    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    K = len(names)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
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
        marker = ""
        if n.startswith("c5_"):
            marker = " ← NEW"
        elif n.startswith("rule_"):
            marker = " ← d6"
        print(f"  {n:<30s} L1={v:.3f}{marker}")
    for c5_name in (f"c5_{r}" for r in rules):
        print(f"  c5 L1: {c5_name}: {l1[c5_name]:.3f}")

    stack_pass = auc >= K18_OOF + 0.5/1e4
    rho_pass = rho_k18 < RHO_TIE
    pred_pass = pred_lb_vs_k18 >= K18_LB + 0.5/1e4
    print(f"\n=== K=20 verdict (vs K=18 anchor) ===")
    print(f"  K=20 OOF >= K=18 + 0.5bp ({K18_OOF + 0.5/1e4:.5f}): {stack_pass}  ({auc:.5f})")
    print(f"  ρ vs K=18 < {RHO_TIE}:                              {rho_pass}   ({rho_k18:.5f})")
    print(f"  pred-LB >= K=18 LB + 0.5bp ({K18_LB+0.5/1e4:.5f}):  {pred_pass}  ({pred_lb_vs_k18:.5f})")
    if stack_pass and rho_pass and pred_pass:
        verdict = "SLOT-CANDIDATE"
    elif pred_lb_vs_k18 >= K18_LB:
        verdict = "MARGINAL"
    else:
        verdict = "DO_NOT_SLOT"
    print(f"  → {verdict}")

    # Save K=20 stack
    np.save(ART / "oof_d9_k20_neighbor_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d9_k20_neighbor_strat.npy",
            np.column_stack([1 - tp, tp]))
    sub = sample_sub.copy(); sub[TARGET] = tp
    sub.to_csv("submissions/submission_d9_k20_neighbor.csv", index=False)

    final = dict(
        per_rule=results_per_rule,
        k20_stack=dict(
            K=K, strat_oof=auc, delta_k18_bp=delta_k18,
            rho_vs_m5q_test=float(rho_m5q),
            rho_vs_k18_test=float(rho_k18),
            pred_lb_vs_k18=float(pred_lb_vs_k18),
            l1_ranking=l1,
            stack_pass=bool(stack_pass), rho_pass=bool(rho_pass),
            pred_pass=bool(pred_pass), verdict=verdict,
        ),
        anchors=dict(M5Q_S=M5Q_S, K18_OOF=K18_OOF, K18_LB=K18_LB),
        total_wall_s=time.time() - t_total,
    )
    (ART / "d9_c5_neighbor_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9_c5_neighbor_results.json  "
          f"(total wall {time.time()-t_total:.0f}s)")


if __name__ == "__main__":
    main()
