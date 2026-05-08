"""Day-9b — R14 strength ladder.

Day-9's R14 (sparse-LR over hashed Driver × Compound × Stint
interactions) was the structural outlier — ρ vs PRIMARY 0.444, but
standalone OOF only 0.794, and min-meta vs PRIMARY −0.02bp (just
short of the gate). Hypothesis: stronger feature engineering on the
same model class (sparse logistic regression with hashed categorical
interactions + binned numerics) lifts standalone OOF *without*
collapsing the diversity (because the model class is unchanged).

Levels (each adds to the previous):
  L0  baseline R14 — D, C, S + 2-way + 3-way (Day-9 R14)
  L1  + R, Y, DR, DY, CR, CY, RY
  L2  + binned TyreLife-q5, RaceProgress-q5, Position-q5,
        Stint-integer; main effects only
  L3  + 2-way of binned numerics with Compound (CT, CR_q, CP, CS_int)
  L4  + Driver × {TyreLife-q5, RaceProgress-q5}
  L5  kitchen sink: L4 + 3-way DCT, DCR_q, DCP

For each level: standalone OOF, ρ vs PRIMARY, minimal-meta vs
PRIMARY (K=2 LR). Strat-only 5-fold SEED=42. liblinear L2-LR.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
PRIMARY_S, PRIMARY_LB = 0.95065, 0.95026
RHO_TIE = 0.999


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
    return meta_oof, lr_full.predict_proba(F_test)[:, 1]


def _quantile_bin(arr_train, arr_query, n_bins):
    edges = np.quantile(arr_train, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    out_t = np.clip(np.searchsorted(edges, arr_train, side="right") - 1,
                    0, n_bins - 1).astype(int)
    out_q = np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                    0, n_bins - 1).astype(int)
    return out_t, out_q


def build_features(level, train, test, tr_idx):
    """Return (sparse_train_matrix, sparse_test_matrix). The bin
    edges are derived from the tr_idx slice only (no leakage of
    fold-validation rows into edge fitting)."""
    n_tr, n_te = len(train), len(test)
    # Categorical strings
    D = train["Driver"].astype(str).values; D_te = test["Driver"].astype(str).values
    C = train["Compound"].astype(str).values; C_te = test["Compound"].astype(str).values
    R = train["Race"].astype(str).values; R_te = test["Race"].astype(str).values
    Y = train["Year"].astype(str).values; Y_te = test["Year"].astype(str).values
    S = train["Stint"].clip(upper=5).astype(int).astype(str).values
    S_te = test["Stint"].clip(upper=5).astype(int).astype(str).values
    # Binned numerics (fit edges on tr_idx)
    if level >= 2:
        T_tr_b, T_q_b = _quantile_bin(train["TyreLife"].values[tr_idx],
                                      train["TyreLife"].values, 5)
        _, T_te_b = _quantile_bin(train["TyreLife"].values[tr_idx],
                                  test["TyreLife"].values, 5)
        Rp_tr_b, Rp_q_b = _quantile_bin(train["RaceProgress"].values[tr_idx],
                                        train["RaceProgress"].values, 5)
        _, Rp_te_b = _quantile_bin(train["RaceProgress"].values[tr_idx],
                                   test["RaceProgress"].values, 5)
        P_tr_b, P_q_b = _quantile_bin(train["Position"].values[tr_idx],
                                      train["Position"].values, 5)
        _, P_te_b = _quantile_bin(train["Position"].values[tr_idx],
                                  test["Position"].values, 5)
        T_str = T_q_b.astype(str); T_te_str = T_te_b.astype(str)
        Rp_str = Rp_q_b.astype(str); Rp_te_str = Rp_te_b.astype(str)
        P_str = P_q_b.astype(str); P_te_str = P_te_b.astype(str)
    else:
        T_str = Rp_str = P_str = None
        T_te_str = Rp_te_str = P_te_str = None

    def feats_for(idx_arr, te=False):
        if te:
            d, c, r, y, s = D_te, C_te, R_te, Y_te, S_te
            T, Rp, P = T_te_str, Rp_te_str, P_te_str
        else:
            d = D[idx_arr]; c = C[idx_arr]; r = R[idx_arr]
            y = Y[idx_arr]; s = S[idx_arr]
            T = T_str[idx_arr] if T_str is not None else None
            Rp = Rp_str[idx_arr] if Rp_str is not None else None
            P = P_str[idx_arr] if P_str is not None else None
        out = []
        for i in range(len(d)):
            row = [
                f"D={d[i]}", f"C={c[i]}", f"S={s[i]}",
                f"DC={d[i]}|{c[i]}", f"CS={c[i]}|{s[i]}",
                f"DS={d[i]}|{s[i]}",
                f"DCS={d[i]}|{c[i]}|{s[i]}",
            ]
            if level >= 1:
                row += [
                    f"R={r[i]}", f"Y={y[i]}",
                    f"DR={d[i]}|{r[i]}", f"DY={d[i]}|{y[i]}",
                    f"CR={c[i]}|{r[i]}", f"CY={c[i]}|{y[i]}",
                    f"RY={r[i]}|{y[i]}",
                ]
            if level >= 2:
                row += [
                    f"T={T[i]}", f"Rp={Rp[i]}", f"P={P[i]}",
                ]
            if level >= 3:
                row += [
                    f"CT={c[i]}|{T[i]}", f"CRp={c[i]}|{Rp[i]}",
                    f"CP={c[i]}|{P[i]}",
                    f"CSint_T={c[i]}|{s[i]}|{T[i]}",
                ]
            if level >= 4:
                row += [
                    f"DT={d[i]}|{T[i]}", f"DRp={d[i]}|{Rp[i]}",
                    f"DP={d[i]}|{P[i]}",
                ]
            if level >= 5:
                row += [
                    f"DCT={d[i]}|{c[i]}|{T[i]}",
                    f"DCRp={d[i]}|{c[i]}|{Rp[i]}",
                    f"DCP={d[i]}|{c[i]}|{P[i]}",
                    f"DRY={d[i]}|{r[i]}|{y[i]}",
                ]
            out.append(row)
        return out

    h = FeatureHasher(n_features=2**18, input_type="string", alternate_sign=False)
    X_tr = h.transform(feats_for(np.arange(n_tr)))
    X_te = h.transform(feats_for(None, te=True))
    return X_tr, X_te


def run_level(level, train, test, y, splits, primary_oof, primary_test):
    print(f"\n=== R14_L{level} ===")
    n_tr, n_te = len(train), len(test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_avg = np.zeros(n_te, dtype=np.float64)
    fold_aucs, walls = [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        X_tr_full, X_te = build_features(level, train, test, tr)
        lr = LogisticRegression(C=1.0, max_iter=300, solver="liblinear")
        lr.fit(X_tr_full[tr], y[tr])
        pred_va = lr.predict_proba(X_tr_full[va])[:, 1]
        pred_te = lr.predict_proba(X_te)[:, 1]
        oof[va] = pred_va
        test_avg += pred_te / N_FOLDS
        s = float(roc_auc_score(y[va], pred_va))
        wall = time.time() - t0
        fold_aucs.append(s); walls.append(wall)
        print(f"  f{k}: AUC={s:.5f}  wall={wall:.1f}s")
    auc = float(roc_auc_score(y, oof))
    rho_test, _ = spearmanr(test_avg, primary_test)
    F_min = expand(np.column_stack([primary_oof, oof]))
    F_min_t = expand(np.column_stack([primary_test, test_avg]))
    mo, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo))
    delta = (auc_min - PRIMARY_S) * 1e4
    print(f"  → std OOF: {auc:.5f}  fold-mean {np.mean(fold_aucs):.5f}  "
          f"total wall={sum(walls):.1f}s")
    print(f"  ρ vs PRIMARY test: {rho_test:.5f}")
    print(f"  Min-meta OOF: {auc_min:.5f}  Δ PRIMARY {delta:+.2f}bp  "
          f"{'PASS ✓' if auc_min >= PRIMARY_S else 'FAIL ✗'}")
    np.save(ART / f"oof_d9b_R14_L{level}_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / f"test_d9b_R14_L{level}_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))
    return dict(level=level, std_oof=auc, fold_mean_auc=float(np.mean(fold_aucs)),
                rho_vs_primary_test=float(rho_test),
                min_meta_oof=auc_min, delta_primary_bp=float(delta),
                min_meta_pass=bool(auc_min >= PRIMARY_S),
                wall_s=float(sum(walls)))


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_oof = np.load(ART / "oof_d6_k18_multi_rule_strat.npy"
                          )[:, 1].astype(np.float64)
    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy"
                           )[:, 1].astype(np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    results = {}
    for level in range(6):
        results[f"L{level}"] = run_level(level, train, test, y, splits,
                                         primary_oof, primary_test)
    final = dict(
        levels=results,
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        total_wall_s=time.time() - t0,
    )
    (ART / "d9b_r14_ladder_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9b_r14_ladder_results.json  "
          f"(total wall {time.time()-t0:.0f}s)")
    print("\n" + "=" * 72)
    print(f"{'level':<6s} {'std_OOF':>8s} {'ρ_PRIM':>8s} {'min-meta':>9s} {'Δprim':>7s}  verdict")
    print("-" * 72)
    for k, r in results.items():
        verdict = "PASS" if r["min_meta_pass"] else "FAIL"
        print(f"{k:<6s} {r['std_oof']:>8.5f} {r['rho_vs_primary_test']:>8.5f} "
              f"{r['min_meta_oof']:>9.5f} {r['delta_primary_bp']:>+6.2f}  {verdict}")


if __name__ == "__main__":
    main()
