"""F1.2 — multi-rule residual ensemble (Day-6 strengthening of Move C).

Builds 3 additional rule_residual L1 bases on different rule
lookups, then pool-adds all 4 (original + 3 new) to M5q for a
K=18 LR-meta stack.

Variants (Bayesian-smoothed lookups, alpha=50):
  R1 (existing): Compound × TyreLife-decile  → d6_rule_residual
  R2:            Compound × Stint
  R3:            Driver × Compound
  R4:            Year × Race

Hypothesis: each rule encodes a different per-row miscalibration
direction; the residual GBDT for each variant produces a different
mistake structure; stacking all 4 with M5q gives the LR meta enough
freedom to extract real LB signal beyond the K=15 quantum-bounded
result.

Decision rule per audit §5:
  - Each new base: minimal-meta sanity check vs M5q (K=2 LR) PASS
  - K=18 stack: OOF >= M5q + 1bp AND ρ < 0.999 → SLOT
  - Predicted-LB heuristic: pred_lb >= M5q LB + 1bp

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
E3_S = 0.94876
RHO_TIE = 0.999
ALPHA = 50.0  # Bayesian smoothing strength

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


def fit_lookup(keys_train, y_train, alpha=ALPHA):
    """Fit a Bayesian-smoothed lookup table; key is a tuple."""
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


def build_keys(df, spec):
    """Build a list of tuple keys per (rule_name, columns, transforms)."""
    name, cols, transforms = spec
    arrays = []
    for c, t in zip(cols, transforms):
        a = df[c].values
        if t is not None:
            a = t(a)
        arrays.append(np.asarray(a))
    return list(zip(*arrays))


def tyre_decile_factory(tyre_train):
    edges = np.quantile(tyre_train, np.linspace(0, 1, 11))
    edges[0] = -np.inf; edges[-1] = np.inf
    def transform(a):
        b = np.searchsorted(edges, a, side="right") - 1
        return np.clip(b, 0, 9)
    return transform


def build_rule_residual_base(rule_name, key_spec, train, test, X_enc, X_test_enc,
                             y, splits, t_log):
    """Build one rule_residual variant in 5-fold Strat. Returns (oof, test_p)."""
    print(f"\n--- Building rule '{rule_name}' (key={key_spec}) ---")

    # Pre-build the key-arrays for train/test (uses raw, untransformed train)
    name, cols, transforms = ("name", *key_spec)  # unpack (cols, transforms)
    keys_train = build_keys(train, ("", cols, transforms))
    keys_test = build_keys(test, ("", cols, transforms))

    oof_full = np.zeros(len(y), dtype=np.float64)
    test_full = np.zeros(len(test), dtype=np.float64)
    rule_aucs, full_aucs, walls = [], [], []

    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        keys_tr = [keys_train[i] for i in tr]
        keys_va = [keys_train[i] for i in va]
        lookup, glob = fit_lookup(keys_tr, y[tr])
        rp_va = apply_lookup(keys_va, lookup, glob)
        rp_te = apply_lookup(keys_test, lookup, glob)
        rp_tr = apply_lookup(keys_tr, lookup, glob)

        # Residual GBDT regressor
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
        rule_aucs.append(s_rule); full_aucs.append(s_full); walls.append(wall)
        print(f"  f{k}: rule={s_rule:.5f}  full={s_full:.5f}  wall={wall:.1f}s")

    auc_full = float(roc_auc_score(y, oof_full))
    print(f"  → standalone OOF: {auc_full:.5f}  std={np.std(full_aucs):.5f}  "
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


def predicted_lb(auc, rho):
    base_lb = M5Q_LB + (auc - M5Q_S)
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

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    X_test = test.drop(columns=[ID_COL], errors="ignore").copy()
    X_enc, X_test_enc = encode_features(X.copy(), X_test.copy())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    tyre_decile = tyre_decile_factory(train["TyreLife"].values)
    str_t = lambda a: a.astype(str)

    # Define rules. Format: (cols_list, transforms_list).
    rules = {
        "compound_stint": (["Compound", "Stint"], [str_t, str_t]),
        "driver_compound": (["Driver", "Compound"], [str_t, str_t]),
        "year_race": (["Year", "Race"], [str_t, str_t]),
    }

    new_oofs, new_tests, results_per_rule = {}, {}, {}
    for rule_name, key_spec in rules.items():
        oof, tp, auc = build_rule_residual_base(rule_name, key_spec, train,
                                                test, X_enc, X_test_enc, y,
                                                splits, t_total)
        new_oofs[rule_name] = oof; new_tests[rule_name] = tp
        rho_test, _ = spearmanr(tp, test_m5q)
        # Minimal-meta sanity check
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
        # Save
        np.save(ART / f"oof_d6_rule_{rule_name}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d6_rule_{rule_name}_strat.npy",
                np.column_stack([1 - tp, tp]))

    # Build K=18 stack: M5q + rule_residual (orig) + 3 new
    print(f"\n=== K=18 stack: M5q (14) + rule_residual + 3 new rules ===")
    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL_M5Q:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    # Original rule_residual
    oo = np.load(ART / "oof_d6_rule_residual_strat.npy")[:, 1].astype(np.float64)
    te = np.load(ART / "test_d6_rule_residual_strat.npy")[:, 1].astype(np.float64)
    Xs_oof.append(oo); Xs_test.append(te); names.append("rule_compound_tyre")
    # New rules
    for rule_name in rules:
        Xs_oof.append(new_oofs[rule_name])
        Xs_test.append(new_tests[rule_name])
        names.append(f"rule_{rule_name}")

    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    K = len(names)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, test_m5q)
    pred_lb = predicted_lb(auc, rho)
    delta = (auc - M5Q_S) * 1e4
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"K={K} LR-meta Strat OOF: {auc:.5f}  Δ M5q {delta:+.2f}bp  "
          f"ρ vs M5q test {rho:.5f}  pred-LB {pred_lb:.5f}")
    print(f"L1 ranking (top-10):")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:10]:
        marker = " ← rule" if n.startswith("rule_") else ""
        print(f"  {n:<22s} L1={v:.3f}{marker}")

    # Verdict
    base_lb = M5Q_LB
    stack_pass = auc >= M5Q_S + 1.0/1e4
    rho_pass = rho < RHO_TIE
    print(f"\n=== K=18 verdict ===")
    print(f"  K=18 OOF >= M5q + 1bp (0.95067):  {stack_pass}  ({auc:.5f})")
    print(f"  ρ < 0.999:                        {rho_pass}   ({rho:.5f})")
    print(f"  Predicted LB:                     {pred_lb:.5f}  "
          f"(vs M5q LB {base_lb:.5f}, Δ {(pred_lb-base_lb)*1e4:+.1f}bp)")
    if stack_pass and rho_pass and pred_lb >= base_lb + 0.5/1e4:
        print(f"  → SLOT-WORTHY")
    elif pred_lb >= base_lb:
        print(f"  → MARGINAL slot candidate (predicted lift but rho/oof gates miss)")
    else:
        print(f"  → DO NOT SLOT")

    # Save K=18
    np.save(ART / "oof_d6_k18_multi_rule_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d6_k18_multi_rule_strat.npy",
            np.column_stack([1 - tp, tp]))
    sub = sample_sub.copy(); sub[TARGET] = tp
    sub.to_csv("submissions/submission_d6_k18_multi_rule.csv", index=False)

    final = dict(
        per_rule=results_per_rule,
        k18_stack=dict(
            K=K, strat_oof=auc, delta_m5q_bp=delta,
            rho_vs_m5q_test=float(rho), pred_lb=float(pred_lb),
            l1_ranking=l1,
            stack_pass=bool(stack_pass), rho_pass=bool(rho_pass),
        ),
        total_wall_s=time.time() - t_total,
    )
    (ART / "d6_multi_rule_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d6_multi_rule_results.json  "
          f"(total wall {time.time()-t_total:.0f}s)")


if __name__ == "__main__":
    main()
