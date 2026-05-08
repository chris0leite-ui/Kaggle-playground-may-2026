"""T1.3 — Q12 Mandatory-2-compound rule_residual base.

F1 regulation: each driver must use ≥2 distinct dry compounds per race.
A driver still on a single compound late in a race MUST pit. This is a
HARD CONSTRAINT not encoded in any base.

Atomicity check (d8_q12_atomicity_check.py): 91% of size>=10 dry groups
have >=2 compounds — feature is well-defined.

Build (mirrors F1.2 multi-rule pattern):
  1. Within (Driver, Race, Year) groups, sort by LapNumber.
     Cumulative-track distinct compounds → `compounds_used_so_far`.
  2. RaceLength_Estimate[Race] = median(max LapNumber) per Race
     (from P8 probe).
  3. Engineer:
     - compounds_used_so_far ∈ {1, 2, 3+}
     - race_progress_norm = LapNumber / RaceLength_Estimate
     - must_change = (compounds_used_so_far == 1) AND (race_progress_norm > 0.6)
                     AND (Compound not in WET/INTERMEDIATE)
     - forced_pit_pressure = must_change × Stint
  4. Bayesian-smoothed lookup `rule_proba` keyed on
     (compounds_used_so_far_bin, Stint, lap_decile, Compound).
  5. Residual HGBC on (raw features) regressing y - rule_proba.

Decision rule:
  - Standalone OOF report (informational).
  - Minimal-input-meta gate vs PRIMARY d6_k18_multi_rule:
    2-comp LR-meta on [primary_oof, q12_oof]; if 2-comp OOF >= PRIMARY,
    Q12 base is incrementally informative.
  - K=19 stack (M5q + 4 rules + Q12) — full slot test if minimal-meta passes.

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
PRIMARY_S = 0.95065
PRIMARY_LB = 0.95026
M5Q_S = 0.95057
RHO_TIE = 0.9995
ALPHA = 50.0  # Bayesian smoothing strength

# K=18 PRIMARY pool (m5q + 4 rules) — matches d6_multi_rule.py
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
    """Same as d6_multi_rule encoding — Driver high-card → ord int;
    Compound, Race → category dtype."""
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


def build_q12_features(df_combined):
    """Compute Q12 features on combined train+test (i.i.d.-shuffled, but
    within-(Driver, Race, Year) cumulative is meaningful per atomicity).
    Returns columns: compounds_used_so_far, race_progress_norm, must_change,
    forced_pit_pressure.
    """
    print("  Computing Q12 features ...")
    t = time.time()

    # RaceLength_Estimate per Race (from P8: median of max LapNumber)
    race_max_per_yeardriver = df_combined.groupby(
        ["Race", "Year", "Driver"]
    )["LapNumber"].max()
    race_length = race_max_per_yeardriver.groupby("Race").median().to_dict()
    df_combined["RaceLength_Est"] = df_combined["Race"].map(race_length).astype(np.float32)
    df_combined["race_progress_norm"] = (
        df_combined["LapNumber"] / df_combined["RaceLength_Est"]
    ).clip(0, 1.5).astype(np.float32)

    # Sort within group; cumulative distinct-compounds
    df_combined = df_combined.sort_values(
        ["Driver", "Race", "Year", "LapNumber"], kind="stable"
    ).reset_index(drop=True)

    # For each row, cumulative distinct compounds in same group up to & incl. this lap
    grp_cols = ["Driver", "Race", "Year"]
    # Build per-group cumulative distinct count via per-row first-appearance flag
    is_first = ~df_combined.duplicated(subset=grp_cols + ["Compound"], keep="first")
    df_combined["_first_compound_in_group"] = is_first.astype(np.int8)
    df_combined["compounds_used_so_far"] = (
        df_combined.groupby(grp_cols)["_first_compound_in_group"].cumsum()
    ).astype(np.int8)
    df_combined.drop(columns=["_first_compound_in_group"], inplace=True)

    # must_change (dry-only, past 60% race progress, single compound)
    is_dry = ~df_combined["Compound"].isin(["WET", "INTERMEDIATE"])
    df_combined["must_change_compound"] = (
        (df_combined["compounds_used_so_far"] == 1)
        & (df_combined["race_progress_norm"] > 0.6)
        & is_dry
    ).astype(np.int8)

    df_combined["forced_pit_pressure"] = (
        df_combined["must_change_compound"] * df_combined["Stint"]
    ).astype(np.int8)

    # compounds_used_so_far_bin: clip to {1, 2, 3+}
    df_combined["cuf_bin"] = df_combined["compounds_used_so_far"].clip(1, 3).astype(np.int8)

    print(f"  Q12 feature stats:")
    print(f"    cuf_bin distribution: {df_combined['cuf_bin'].value_counts().to_dict()}")
    print(f"    must_change distribution: {df_combined['must_change_compound'].value_counts().to_dict()}")
    print(f"    must_change=1 share: {df_combined['must_change_compound'].mean():.4%}")
    print(f"    forced_pit_pressure mean: {df_combined['forced_pit_pressure'].mean():.4f}")
    print(f"  wall {time.time()-t:.1f}s")
    return df_combined


def build_keys(df, cols):
    arrays = [df[c].values for c in cols]
    return list(zip(*arrays))


def build_q12_rule_residual_base(train, test, X_enc, X_test_enc, y, splits):
    """Build the Q12 rule_residual base in 5-fold Strat.

    Lookup key: (cuf_bin, Stint, lap_decile, Compound) — 4-tuple.
    Residual GBDT trains on raw features regressing y - rule_proba.
    """
    print("\n--- Building Q12 forced-pit rule_residual base ---")
    print(f"  K (rule keys): {len(set(zip(train['cuf_bin'], train['Stint'], train['lap_decile'], train['Compound'].astype(str))))}")

    KEY_COLS = ["cuf_bin", "Stint", "lap_decile", "Compound"]
    train_str = train.copy()
    train_str["Compound"] = train_str["Compound"].astype(str)
    test_str = test.copy()
    test_str["Compound"] = test_str["Compound"].astype(str)

    keys_train = build_keys(train_str, KEY_COLS)
    keys_test = build_keys(test_str, KEY_COLS)

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

        # Residual GBDT regressor on (raw features, target = y - rule_proba)
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
    auc_rule = float(np.mean(rule_aucs))
    print(f"  → standalone OOF: {auc_full:.5f} (rule alone {auc_rule:.5f})  "
          f"std={np.std(full_aucs):.5f}  total wall={sum(walls):.1f}s")
    return oof_full, test_full, auc_full, auc_rule


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

    # Build Q12 features on combined data (within-group cumulative needs ALL rows)
    print("\n--- Building Q12 features on combined train+test ---")
    train_df["_src"] = "train"; test_df["_src"] = "test"; test_df[TARGET] = -1
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all = build_q12_features(df_all)

    # Add lap_decile (race_progress_norm decile)
    df_all["lap_decile"] = pd.qcut(
        df_all["race_progress_norm"], 10, labels=False, duplicates="drop"
    ).astype(np.int8)

    # Split back, preserving original train/test ordering
    df_all = df_all.sort_values("id", kind="stable").reset_index(drop=True)
    train = df_all[df_all["_src"] == "train"].copy()
    test = df_all[df_all["_src"] == "test"].copy()
    print(f"after Q12 enrich: train {train.shape}, test {test.shape}")

    # Verify ordering matches y
    assert (train[TARGET].values == y).all(), "train target order mismatch!"

    # Build encoded raw-feature matrix for HGBC (drop Q12 helper cols + target)
    drop_cols = [TARGET, ID_COL, "_src", "RaceLength_Est",
                 "compounds_used_so_far", "must_change_compound",
                 "forced_pit_pressure", "cuf_bin", "lap_decile",
                 "race_progress_norm"]
    X = train.drop(columns=drop_cols, errors="ignore").copy()
    X_test = test.drop(columns=drop_cols, errors="ignore").copy()

    # ALSO try a variant where Q12 features are FED to the residual GBDT
    # (alongside raw features) — strictly larger info than rule-lookup alone.
    X_q12 = X.copy()
    X_test_q12 = X_test.copy()
    for c in ["compounds_used_so_far", "must_change_compound",
              "forced_pit_pressure", "race_progress_norm"]:
        X_q12[c] = train[c].values
        X_test_q12[c] = test[c].values

    X_enc, X_test_enc = encode_features(X.copy(), X_test.copy())
    X_enc_q12, X_test_enc_q12 = encode_features(X_q12.copy(), X_test_q12.copy())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # === Variant A: Q12 rule_residual with raw-only residual GBDT ===
    print("\n=== VARIANT A: rule lookup, residual GBDT on RAW features only ===")
    oof_a, test_a, auc_a, _ = build_q12_rule_residual_base(
        train, test, X_enc, X_test_enc, y, splits
    )
    rho_a_primary, _ = spearmanr(test_a, primary_test)

    # === Variant B: Q12 rule_residual + Q12 features in residual GBDT ===
    print("\n=== VARIANT B: rule lookup, residual GBDT on RAW+Q12 features ===")
    oof_b, test_b, auc_b, _ = build_q12_rule_residual_base(
        train, test, X_enc_q12, X_test_enc_q12, y, splits
    )
    rho_b_primary, _ = spearmanr(test_b, primary_test)

    # Minimal-meta gate: 2-comp LR over [primary, q12]
    print(f"\n=== Minimal-meta sanity check (2-comp LR vs PRIMARY) ===")
    for label, oof, tp, std_auc, rho_p in [
        ("Variant A", oof_a, test_a, auc_a, rho_a_primary),
        ("Variant B", oof_b, test_b, auc_b, rho_b_primary),
    ]:
        F_min = expand(np.column_stack([primary_oof, oof]))
        F_min_t = expand(np.column_stack([primary_test, tp]))
        mo_min, _, _ = fit_lr_meta(F_min, F_min_t, y)
        auc_min = float(roc_auc_score(y, mo_min))
        verdict = "PASS ✓" if auc_min >= primary_oof_auc else "FAIL ✗"
        print(f"  {label}: std-OOF {std_auc:.5f}  ρ vs PRIMARY {rho_p:.5f}")
        print(f"    2-comp LR-meta OOF: {auc_min:.5f}  Δ PRIMARY "
              f"{(auc_min-primary_oof_auc)*1e4:+.2f}bp  {verdict}")

    # Pick the best variant for K=19 stack
    if auc_b >= auc_a:
        oof_q12, test_q12, name_q12 = oof_b, test_b, "q12_v_b"
        std_auc = auc_b
        rho_q12_primary = rho_b_primary
        print(f"\n  → Selecting Variant B for K=19 stack (std-OOF {auc_b:.5f})")
    else:
        oof_q12, test_q12, name_q12 = oof_a, test_a, "q12_v_a"
        std_auc = auc_a
        rho_q12_primary = rho_a_primary
        print(f"\n  → Selecting Variant A for K=19 stack (std-OOF {auc_a:.5f})")

    # Save Q12 base artifact
    np.save(ART / f"oof_d8_{name_q12}_strat.npy",
            np.column_stack([1 - oof_q12, oof_q12]))
    np.save(ART / f"test_d8_{name_q12}_strat.npy",
            np.column_stack([1 - test_q12, test_q12]))

    # === K=19 LR stack: K=18 pool + Q12 ===
    print(f"\n=== K=19 LR-meta stack: K=18 pool + Q12 ({name_q12}) ===")
    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL_K18:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    Xs_oof.append(oof_q12); Xs_test.append(test_q12); names.append(f"q12_{name_q12[-1]}")
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
    print(f"Predicted LB: {pred_lb:.5f}  (vs PRIMARY LB {PRIMARY_LB:.5f}, "
          f"Δ {(pred_lb-PRIMARY_LB)*1e4:+.1f}bp)")
    print(f"L1 ranking (top-12):")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:12]:
        marker = " ← Q12" if n.startswith("q12") else ""
        print(f"  {n:<22s} L1={v:.3f}{marker}")

    # Verdict gates per audit §5
    stack_pass = auc >= primary_oof_auc + 1.0/1e4
    rho_pass = rho_primary < RHO_TIE
    pred_lb_pass = pred_lb >= PRIMARY_LB + 0.5/1e4
    print(f"\n=== K=19 verdict ===")
    print(f"  K=19 OOF >= PRIMARY + 1bp ({primary_oof_auc + 1.0/1e4:.5f}):  "
          f"{stack_pass}  ({auc:.5f})")
    print(f"  ρ < {RHO_TIE}:                         "
          f"{rho_pass}   ({rho_primary:.5f})")
    print(f"  Predicted LB >= PRIMARY LB + 0.5bp:     {pred_lb_pass}  "
          f"({pred_lb:.5f})")
    if stack_pass and rho_pass and pred_lb_pass:
        print(f"  → SLOT-WORTHY")
        sub_status = "slot_worthy"
    elif pred_lb >= PRIMARY_LB:
        print(f"  → MARGINAL slot candidate (predicted lift but gates miss)")
        sub_status = "marginal"
    else:
        print(f"  → DO NOT SLOT")
        sub_status = "skip"

    # Save K=19 stack artifact + submission file
    np.save(ART / "oof_d8_k19_q12_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d8_k19_q12_strat.npy",
            np.column_stack([1 - tp, tp]))
    if sub_status != "skip":
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv("submissions/submission_d8_k19_q12.csv", index=False)
        print(f"\n→ submission_d8_k19_q12.csv written")

    final = dict(
        primary_oof=primary_oof_auc, primary_lb=PRIMARY_LB,
        variant_a=dict(std_oof=auc_a, rho_vs_primary=float(rho_a_primary)),
        variant_b=dict(std_oof=auc_b, rho_vs_primary=float(rho_b_primary)),
        chosen=name_q12,
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
    (ART / "d8_q12_forced_pit_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d8_q12_forced_pit_results.json  "
          f"(total wall {time.time()-t_total:.0f}s)")


if __name__ == "__main__":
    main()
