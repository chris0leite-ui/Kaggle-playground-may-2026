"""K=19 stack: M5q (14) + d6 4 rules + hazard_nn_bag (Day-9 PROMOTE).

Mirrors the F1.2 K=18 stack structure with one extra base appended.
Outer fold split is StratifiedKFold(5, random_state=42) — matches the
hazard-NN bag v2 outer split (so OOF stacking is leak-free).

Verdict gates (per audit §5):
  - K=19 OOF >= K=18 anchor + 0.5bp (0.95070)
  - ρ vs K=18 test < 0.999 (Rule 16 Q6: ρ vs full stack, not just M5q)
  - Predicted LB >= K=18 LB + 0.5bp (0.95031)
Plus Rule 16 Q6 sub-checks: ρ vs each existing base (printed for audit).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
M5Q_S, M5Q_LB = 0.95057, 0.95005
K18_OOF, K18_LB = 0.95065, 0.95026
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
D6_RULES = [
    ("rule_compound_tyre", "d6_rule_residual"),
    ("rule_compound_stint", "d6_rule_compound_stint"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]


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
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    n_test = len(test)

    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]
    m5q_oof = np.load(ART / "oof_m5q_strat.npy")[:, 1]
    k18_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy")[:, 1]
    k18_oof = np.load(ART / "oof_d6_k18_multi_rule_strat.npy")[:, 1]
    print(f"Anchors: M5q OOF={roc_auc_score(y, m5q_oof):.5f}  "
          f"K=18 OOF={roc_auc_score(y, k18_oof):.5f}")

    haz_oof = np.load(ART / "oof_d9_hazard_nn_strat.npy")[:, 1].astype(np.float64)
    haz_test = np.load(ART / "test_d9_hazard_nn_strat.npy")[:, 1].astype(np.float64)
    haz_auc = float(roc_auc_score(y, haz_oof))
    print(f"hazard_nn_bag OOF AUC: {haz_auc:.5f}")

    # Diversity scan: ρ of hazard-NN test vs every existing base in the K=18 stack
    print("\n=== Q6 ρ-vs-stack scan (Rule 16) ===")
    rho_m5q_test, _ = spearmanr(haz_test, test_m5q)
    rho_k18_test, _ = spearmanr(haz_test, k18_test)
    print(f"hazard_nn vs M5q test:  ρ={rho_m5q_test:.5f}")
    print(f"hazard_nn vs K=18 test: ρ={rho_k18_test:.5f}")
    rho_per_base = []
    for label, n in POOL_M5Q + D6_RULES:
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        r, _ = spearmanr(haz_test, te)
        rho_per_base.append((label, float(r)))
    print("hazard_nn vs each base in K=18 (sorted asc):")
    for n, r in sorted(rho_per_base, key=lambda kv: kv[1])[:20]:
        print(f"  {n:<22s} ρ={r:.5f}")

    # Min-meta sanity
    F_min = expand(np.column_stack([m5q_oof, haz_oof]))
    F_min_t = expand(np.column_stack([test_m5q, haz_test]))
    mo_min, _, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo_min))
    print(f"\nMin-meta (M5q + hazard) OOF: {auc_min:.5f}  "
          f"Δ M5q {(auc_min - M5Q_S)*1e4:+.2f}bp  "
          f"{'PASS ✓' if auc_min >= M5Q_S else 'FAIL ✗'}")

    # K=19 stack
    print(f"\n=== K=19 stack: M5q (14) + d6 4 rules + hazard_nn_bag ===")
    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL_M5Q:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    for label, n in D6_RULES:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    Xs_oof.append(haz_oof); Xs_test.append(haz_test); names.append("hazard_nn_bag")

    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    K = len(names)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho_m5q, _ = spearmanr(tp, test_m5q)
    rho_k18, _ = spearmanr(tp, k18_test)
    pred_lb = predicted_lb(auc, rho_k18, K18_OOF, K18_LB)
    delta_k18 = (auc - K18_OOF) * 1e4
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"K={K} LR-meta Strat OOF: {auc:.5f}  Δ K=18 {delta_k18:+.2f}bp")
    print(f"  ρ vs M5q test:  {rho_m5q:.5f}")
    print(f"  ρ vs K=18 test: {rho_k18:.5f}")
    print(f"  pred-LB (vs K=18 anchor): {pred_lb:.5f}")
    print(f"L1 ranking (top-10):")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:10]:
        marker = " ← NEW" if n == "hazard_nn_bag" else ""
        print(f"  {n:<22s} L1={v:.3f}{marker}")
    print(f"  hazard_nn_bag L1: {l1['hazard_nn_bag']:.3f}")

    stack_pass = auc >= K18_OOF + 0.5/1e4
    rho_pass = rho_k18 < RHO_TIE
    pred_pass = pred_lb >= K18_LB + 0.5/1e4
    print(f"\n=== K=19 verdict (vs K=18 anchor) ===")
    print(f"  K=19 OOF >= K=18 + 0.5bp ({K18_OOF + 0.5/1e4:.5f}): {stack_pass}  ({auc:.5f})")
    print(f"  ρ vs K=18 < {RHO_TIE}:                              {rho_pass}   ({rho_k18:.5f})")
    print(f"  pred-LB >= K=18 LB + 0.5bp ({K18_LB+0.5/1e4:.5f}):  {pred_pass}  ({pred_lb:.5f})")
    if stack_pass and rho_pass and pred_pass:
        verdict = "SUBMIT-CANDIDATE"
    elif pred_lb >= K18_LB:
        verdict = "MARGINAL"
    else:
        verdict = "DO_NOT_SLOT"
    print(f"  → {verdict}")

    np.save(ART / "oof_d9_k19_hazard_nn_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d9_k19_hazard_nn_strat.npy",
            np.column_stack([1 - tp, tp]))
    sub = sample_sub.copy(); sub[TARGET] = tp
    sub.to_csv("submissions/submission_d9_k19_hazard_nn_stack.csv", index=False)

    final = dict(
        hazard_oof_auc=haz_auc,
        rho_vs_m5q_test=float(rho_m5q_test),
        rho_vs_k18_test=float(rho_k18_test),
        rho_per_base=[{"name": n, "rho_vs_haz": r} for n, r in rho_per_base],
        min_meta=dict(oof=auc_min,
                      delta_m5q_bp=(auc_min - M5Q_S) * 1e4,
                      pass_=bool(auc_min >= M5Q_S)),
        k19_stack=dict(K=K, strat_oof=auc, delta_k18_bp=delta_k18,
                       rho_vs_m5q_test=float(rho_m5q),
                       rho_vs_k18_test=float(rho_k18),
                       pred_lb=float(pred_lb),
                       l1_ranking=l1,
                       stack_pass=bool(stack_pass),
                       rho_pass=bool(rho_pass),
                       pred_pass=bool(pred_pass),
                       verdict=verdict),
        anchors=dict(M5Q_S=M5Q_S, K18_OOF=K18_OOF, K18_LB=K18_LB),
    )
    (ART / "d9_k19_hazard_nn_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9_k19_hazard_nn_results.json")


if __name__ == "__main__":
    main()
