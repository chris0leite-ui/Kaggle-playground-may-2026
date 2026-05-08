"""T1.2 — Multi-formulation L1 (Poisson on laps_until_next_pit).

Per audit/2026-05-08-strategic-menu-wider-steps.md Tier-1 #2.

Mechanism: LR-meta over GBDT-classifier outputs is rank-locked.
Q12 (rule_residual) confirmed: any GBDT base trained on the same
binary target collapses to ρ≥0.999 in K=N stack. The escape route
is a base that SOLVES A DIFFERENT PROBLEM on the same data.

Build: train LightGBM with Poisson objective on a NEW target
`laps_until_next_pit` — within (Driver, Race, Year, Stint) groups
sorted by LapNumber, for each row find the NEXT row with PitStop=1
(if any) and compute distance = next_pit_lap - current_lap.
Censored rows (no future pit in group) get distance = 99 (sentinel).

The Poisson base predicts an EXPECTED count → small predicted value
means "pit imminent". We use -predicted_value as the rank score.
The LOSS LANDSCAPE is fundamentally different from binary-AUC, so
the rank ordering is NOT a monotone transform of M5q.

Decision rule (same gates as F1.2):
  - Standalone OOF report.
  - Minimal-meta gate vs PRIMARY (2-comp LR).
  - K=19 stack (M5q + 4 rules + Poisson).
  - Slot if OOF ≥ PRIMARY+1bp AND ρ < 0.9995.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
PRIMARY_S = 0.95065
PRIMARY_LB = 0.95026
RHO_TIE = 0.9995

POOL_K18 = [
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


def build_laps_until_target(df_combined):
    """For each row, compute laps_until_next_pit within (Driver, Race,
    Year, Stint) group. Censored = 99.

    Sort within group by LapNumber. For each row, look forward to find
    the NEXT row with PitStop=1 in the same group (if exists). Distance
    = next_lap - current_lap. If never pits, set 99 (sentinel).
    """
    print("  Computing laps_until_next_pit ...")
    t = time.time()
    df = df_combined.sort_values(
        ["Driver", "Race", "Year", "Stint", "LapNumber"], kind="stable"
    ).reset_index(drop=True)

    # For TRAIN rows we have PitStop. For TEST rows we have PitStop too.
    # Within group: for each lap, distance to next PitStop=1 lap.
    # Vectorize per group: cumulative-from-end + lap distance.

    # Strategy: for each group, get the ASCENDING-sorted LapNumbers and
    # an array of pit indicators. For each row, find next-pit lap;
    # distance = that - current. If none, 99.
    grp_keys = ["Driver", "Race", "Year", "Stint"]
    laps_until = np.full(len(df), 99, dtype=np.int32)

    # Build mapping group -> [(lap, pit_flag, row_idx), ...] sorted by lap
    grp_idx = df.groupby(grp_keys, sort=False).indices
    print(f"  groups: {len(grp_idx)}; total rows: {len(df)}")

    pit_arr = df["PitStop"].values.astype(np.int8)
    lap_arr = df["LapNumber"].values.astype(np.int32)

    for k, idxs in grp_idx.items():
        # Sort idxs by LapNumber (already sorted within sort_values, but
        # just to be safe — note: ascending by LapNumber)
        order = np.argsort(lap_arr[idxs], kind="stable")
        idxs_sorted = idxs[order]
        laps_g = lap_arr[idxs_sorted]
        pits_g = pit_arr[idxs_sorted]
        n = len(idxs_sorted)
        # Find for each row the next pit lap (>= row's lap +1 with pit=1)
        # Walk back from end: maintain "next_pit_lap_seen" updated when pit=1.
        next_pit_lap = 99 + laps_g[-1]
        for i in range(n - 1, -1, -1):
            if pits_g[i] == 1:
                # this lap pits; for the NEXT pit decision (i.e., for
                # rows BEFORE this one), next_pit_lap = laps_g[i].
                # For this row itself, distance is set after this update.
                pass  # next_pit_lap update happens after recording dist for i
            d = next_pit_lap - laps_g[i]
            laps_until[idxs_sorted[i]] = min(d, 99)
            if pits_g[i] == 1:
                next_pit_lap = laps_g[i]

    # Restore original order (sort by id)
    df["laps_until_next_pit"] = laps_until
    print(f"  laps_until distribution:")
    vc = df["laps_until_next_pit"].value_counts().sort_index().head(15)
    for k, v in vc.items():
        print(f"    {k}: {v}")
    censored_share = (df["laps_until_next_pit"] >= 99).mean()
    print(f"  censored share (>=99): {censored_share:.4%}")
    print(f"  wall {time.time()-t:.1f}s")
    return df


def fit_lgbm_poisson(X_train, X_test, y_target, splits, y_binary_for_auc):
    """LightGBM Poisson on laps_until_next_pit.
    Returns OOF and test predictions; OOF AUC is computed against binary y."""
    print("  Training LGBM Poisson on laps_until_next_pit ...")
    params = dict(
        objective="poisson",  # log-link Poisson
        metric="poisson",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=SEED,
    )
    oof = np.zeros(len(y_target), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    walls = []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        d_tr = lgb.Dataset(X_train.iloc[tr], label=y_target[tr])
        d_va = lgb.Dataset(X_train.iloc[va], label=y_target[va])
        m = lgb.train(params, d_tr, num_boost_round=3000,
                      valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        # Predict expected count; SMALLER means pit-imminent → use -pred as
        # rank score for AUC.
        oof[va] = m.predict(X_train.iloc[va])
        test_pred += m.predict(X_test) / N_FOLDS
        wall = time.time() - t0
        # AUC of -oof[va] vs binary y
        s_auc = float(roc_auc_score(y_binary_for_auc[va], -oof[va]))
        walls.append(wall)
        print(f"  f{k}: best_iter={m.best_iteration}  AUC(-oof,y)={s_auc:.5f}  "
              f"wall={wall:.1f}s")

    auc_full = float(roc_auc_score(y_binary_for_auc, -oof))
    print(f"  → standalone OOF AUC (using -pred as rank): {auc_full:.5f}  "
          f"total wall={sum(walls):.1f}s")
    return oof, test_pred, auc_full


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    p = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(p / (1 - p))
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


def predicted_lb(auc, rho_vs_primary):
    base_lb = PRIMARY_LB + (auc - PRIMARY_S)
    if rho_vs_primary >= RHO_TIE:
        return base_lb
    if rho_vs_primary >= 0.995:
        return base_lb - 0.0001
    if rho_vs_primary >= 0.99:
        return base_lb - 0.00025
    return base_lb - 0.0004


def to_proba(neg_pred):
    """Convert -poisson_pred (higher = pit imminent) to a [0,1] proba via
    rank → quantile transform. This is for stacking convenience.
    """
    n = len(neg_pred)
    return rankdata(neg_pred) / n


def main():
    t_total = time.time()
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train_df[TARGET].astype(int).values

    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy")[:, 1]
    primary_oof = np.load(ART / "oof_d6_k18_multi_rule_strat.npy")[:, 1]
    primary_oof_auc = float(roc_auc_score(y, primary_oof))
    print(f"PRIMARY d6_k18_multi_rule OOF: {primary_oof_auc:.5f}")

    # Combined train+test for laps_until_next_pit (uses PitStop in BOTH)
    print("\n--- Building laps_until_next_pit on combined train+test ---")
    train_df["_src"] = "train"; test_df["_src"] = "test"; test_df[TARGET] = -1
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all = build_laps_until_target(df_all)
    df_all = df_all.sort_values("id", kind="stable").reset_index(drop=True)
    train = df_all[df_all["_src"] == "train"].copy()
    test = df_all[df_all["_src"] == "test"].copy()
    assert (train[TARGET].astype(int).values == y).all(), "ord mismatch"

    # Encode raw features (drop helper cols + target)
    drop_cols = [TARGET, ID_COL, "_src", "laps_until_next_pit"]
    X = train.drop(columns=drop_cols, errors="ignore").copy()
    X_test = test.drop(columns=drop_cols, errors="ignore").copy()
    X_enc, X_test_enc = encode_features(X.copy(), X_test.copy())

    # Target: laps_until_next_pit (Poisson — non-negative count)
    laps_until = train["laps_until_next_pit"].astype(np.float64).values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # === Build Poisson base ===
    oof_pois_raw, test_pois_raw, auc_std = fit_lgbm_poisson(
        X_enc, X_test_enc, laps_until, splits, y
    )
    # Convert to [0,1] proba via rank-quantile (so it stacks like other bases)
    oof_pois = to_proba(-oof_pois_raw)
    test_pois = to_proba(-test_pois_raw)

    # ρ vs PRIMARY (rank-equivalent test)
    rho_pois_primary, _ = spearmanr(test_pois, primary_test)
    print(f"\nstandalone OOF AUC: {auc_std:.5f}  "
          f"ρ vs PRIMARY test: {rho_pois_primary:.5f}")

    # Save artifact
    np.save(ART / "oof_d8_poisson_lapsuntil_strat.npy",
            np.column_stack([1 - oof_pois, oof_pois]))
    np.save(ART / "test_d8_poisson_lapsuntil_strat.npy",
            np.column_stack([1 - test_pois, test_pois]))

    # === Minimal-meta gate ===
    print(f"\n--- Minimal-meta gate (2-comp LR vs PRIMARY) ---")
    F_min = expand(np.column_stack([primary_oof, oof_pois]))
    F_min_t = expand(np.column_stack([primary_test, test_pois]))
    mo_min, _, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo_min))
    delta_min = (auc_min - primary_oof_auc) * 1e4
    verdict_min = "PASS ✓" if auc_min >= primary_oof_auc else "FAIL ✗"
    print(f"  2-comp OOF: {auc_min:.5f}  Δ PRIMARY {delta_min:+.2f}bp  "
          f"{verdict_min}")

    # === K=19 stack ===
    print(f"\n=== K=19 LR-meta stack: K=18 pool + Poisson ===")
    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL_K18:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    Xs_oof.append(oof_pois); Xs_test.append(test_pois); names.append("poisson")
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    K = len(names)

    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho_primary, _ = spearmanr(tp, primary_test)
    pred_lb = predicted_lb(auc, rho_primary)
    delta_primary_bp = (auc - primary_oof_auc) * 1e4

    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"K={K} LR-meta Strat OOF: {auc:.5f}  Δ PRIMARY {delta_primary_bp:+.2f}bp")
    print(f"ρ vs PRIMARY test: {rho_primary:.5f}")
    print(f"Predicted LB: {pred_lb:.5f}  Δ PRIMARY LB "
          f"{(pred_lb-PRIMARY_LB)*1e4:+.1f}bp")
    print(f"L1 ranking (top-10):")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:10]:
        marker = " ← Poisson" if n == "poisson" else ""
        print(f"  {n:<22s} L1={v:.3f}{marker}")

    # Verdict
    stack_pass = auc >= primary_oof_auc + 1.0/1e4
    rho_pass = rho_primary < RHO_TIE
    pred_lb_pass = pred_lb >= PRIMARY_LB + 0.5/1e4
    print(f"\n=== K=19 verdict ===")
    print(f"  K=19 OOF >= PRIMARY + 1bp: {stack_pass} ({auc:.5f})")
    print(f"  ρ < 0.9995:                 {rho_pass} ({rho_primary:.5f})")
    print(f"  Pred LB >= PRIMARY+0.5bp:   {pred_lb_pass} ({pred_lb:.5f})")
    if stack_pass and rho_pass and pred_lb_pass:
        print(f"  → SLOT-WORTHY")
        sub_status = "slot_worthy"
    elif pred_lb >= PRIMARY_LB:
        print(f"  → MARGINAL")
        sub_status = "marginal"
    else:
        print(f"  → DO NOT SLOT")
        sub_status = "skip"

    # Save K=19 stack and (if slot_worthy) submission
    np.save(ART / "oof_d8_k19_poisson_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d8_k19_poisson_strat.npy",
            np.column_stack([1 - tp, tp]))
    if sub_status != "skip":
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv("submissions/submission_d8_k19_poisson.csv", index=False)
        print(f"\n→ submission_d8_k19_poisson.csv written")

    final = dict(
        primary_oof=primary_oof_auc, primary_lb=PRIMARY_LB,
        poisson_base=dict(
            std_oof=auc_std, rho_vs_primary=float(rho_pois_primary),
            min_meta_oof=auc_min, min_meta_delta_bp=delta_min,
            min_meta_pass=bool(auc_min >= primary_oof_auc),
        ),
        k19_stack=dict(
            K=K, strat_oof=auc, delta_primary_bp=delta_primary_bp,
            rho_vs_primary=float(rho_primary), pred_lb=float(pred_lb),
            l1_ranking=l1,
            stack_pass=bool(stack_pass), rho_pass=bool(rho_pass),
            pred_lb_pass=bool(pred_lb_pass),
            sub_status=sub_status,
        ),
        wall_total_s=time.time() - t_total,
    )
    (ART / "d8_poisson_lapsuntil_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d8_poisson_lapsuntil_results.json  "
          f"(total wall {time.time()-t_total:.0f}s)")


if __name__ == "__main__":
    main()
