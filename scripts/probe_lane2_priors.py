"""scripts/probe_lane2_priors.py — Lane 2 (F1 pit-decision priors).

Tests whether deterministic / near-deterministic F1 strategy priors are
fully encoded by the K=4 pool, OR whether presenting them as **meta
features alongside base predictions** opens a 4th logit direction.

Critical distinction from EXP-3 (inter-stint as a base, NULL): here the
heuristic features are added as columns **at the meta layer**. The 30-
feature [P, rank, logit] expansion can reconstruct any new base's
*prediction* linearly (A29) — it CANNOT reconstruct an arbitrary new
column that's a function of raw features.

Probes:
  D2.1 — empirical pit hazard curve P(pit | TyreLife, Compound) and
         P(pit | laps_to_race_end). Identifies bins where P > 0.8 or < 0.05.
  P2.1 — heuristic features as meta inputs to K=4 LR meta.
  P2.2 — deterministic post-hoc rule clamps on K=4 PRIMARY.
  P2.3 — Compound-tier monotonic LGBM as a single base, K=4+1 gate.

All features are computed combined-frame (AV-safe per A3) using row
indices and raw features only — NOT label-conditional, so Rule 24
fold-safety is automatic.

Cost (CPU): ~45 min combined. Outputs:
  scripts/artifacts/probe_lane2_priors.json
  scripts/artifacts/oof_lane2_K4plus_heuristic_meta_strat.npy   (P2.1)
  scripts/artifacts/oof_lane2_rule_clamped_strat.npy            (P2.2)
  scripts/artifacts/oof_lane2_compound_monotonic_lgbm_strat.npy (P2.3)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 500

K4_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]

COMPOUND_TIER = {"SOFT": 1, "MEDIUM": 2, "HARD": 3,
                 "INTERMEDIATE": 4, "WET": 5}


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def build_priors_features(train: pd.DataFrame,
                          test: pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build F1-domain heuristic features. Combined-frame, label-free."""
    n_tr = len(train)
    df = pd.concat([train.assign(_split="tr"),
                    test.assign(_split="te")],
                   ignore_index=True, sort=False)
    df["row_id"] = np.arange(len(df))

    # Compound-tier ordinal
    df["compound_tier"] = df["Compound"].map(COMPOUND_TIER).fillna(0).astype(int)

    # TyreLife percentile within Compound (combined-frame; ECDF on combined
    # rows; Rule 25 AV-AUC=0.502 says combined transform is safe here)
    df["tyre_life_pctile_in_compound"] = (
        df.groupby("Compound", sort=False)["TyreLife"].rank(pct=True))

    # Race phase: laps_to_race_end (combined-frame; race_max_lap from
    # combined train+test which is AV-safe by A3)
    g_race = df.groupby("Race", sort=False)
    df["race_max_lap"] = g_race["LapNumber"].transform("max")
    df["laps_to_race_end"] = (df["race_max_lap"] - df["LapNumber"]).astype(float)
    df["is_last_3_laps"] = (df["laps_to_race_end"] <= 2).astype(int)
    df["is_last_lap"] = (df["laps_to_race_end"] <= 0).astype(int)
    df["race_progress"] = (df["LapNumber"] / df["race_max_lap"].clip(lower=1)).astype(float)

    # n_distinct_compounds_used_so_far per (Race, Driver, Year), combined-frame
    df = df.sort_values(["Race", "Driver", "Year", "LapNumber"],
                        kind="stable").reset_index(drop=True)
    g_rdy = df.groupby(["Race", "Driver", "Year"], sort=False)

    def cum_unique_count(s):
        seen = []
        out = []
        for v in s:
            if v not in seen:
                seen.append(v)
            out.append(len(seen))
        return out

    df["n_distinct_compounds_so_far"] = (
        g_rdy["Compound"].transform(cum_unique_count).astype(int))

    # Cross-driver field-state at this (Race, Year, LapNumber): how
    # many drivers have *been observed* at this lap? Cheap proxy for
    # density of pit calls in the field.
    g_rl = df.groupby(["Race", "Year", "LapNumber"], sort=False)
    df["field_size_at_lap"] = g_rl["Driver"].transform("size")

    # Stint progress relative to typical Compound stint length
    # Note: "typical" computed from observed data (combined-frame).
    df["stint_size_obs"] = df.groupby(
        ["Race", "Driver", "Year", "Stint"], sort=False)["LapNumber"].transform("size")

    # Compound-typical observed stint length (combined-frame ECDF)
    compound_stint_q = (
        df.groupby(["Compound", "Race", "Driver", "Year", "Stint"],
                   sort=False)["LapNumber"].size().reset_index(name="sz")
          .groupby("Compound")["sz"].quantile(0.95).to_dict())
    df["compound_stint_p95"] = df["Compound"].map(compound_stint_q).fillna(
        df["stint_size_obs"].quantile(0.95)).astype(float)
    df["stint_overrun"] = (df["stint_size_obs"] / df["compound_stint_p95"].clip(lower=1)).astype(float)

    feats = [
        "compound_tier",
        "tyre_life_pctile_in_compound",
        "laps_to_race_end",
        "is_last_3_laps",
        "is_last_lap",
        "race_progress",
        "n_distinct_compounds_so_far",
        "field_size_at_lap",
        "stint_overrun",
    ]
    df = df.sort_values("row_id").reset_index(drop=True)
    return (df.iloc[:n_tr].reset_index(drop=True),
            df.iloc[n_tr:].reset_index(drop=True),
            feats)


def main():
    t0 = time.time()
    print("Loading data + K=4 base OOFs ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4_FWD]
    base_tests = [_pos(ART / f"test_{n}_strat.npy") for n in K4_FWD]
    P4 = np.column_stack(base_oofs)
    P4_test = np.column_stack(base_tests)

    primary_path = ART / "oof_K4_fwd_pathb_strat.npy"
    if primary_path.exists():
        primary_oof = _pos(primary_path)
        primary_kind = "K4_fwd_pathb"
    else:
        print("  (PRIMARY composite OOF not on disk; using plain K=4 LR meta substitute)")
        F4_tmp = _expand(P4)
        F4_test_tmp = _expand(P4_test)
        primary_oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                       random_state=SEED).split(np.zeros(len(y)), y):
            lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F4_tmp[tr], y[tr])
            primary_oof[va] = lr.predict_proba(F4_tmp[va])[:, 1]
        primary_kind = "K4_LR_meta_substitute"

    print("Building priors features ...")
    tr_x, te_x, h_feats = build_priors_features(train, test)

    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED).split(np.zeros(len(y)), y))

    # ============ D2.1 — empirical pit hazard tables ================
    print("\n--- D2.1: empirical pit hazard")
    tl_q = pd.qcut(tr_x["tyre_life_pctile_in_compound"], 10,
                   labels=[f"p{i*10}" for i in range(1, 11)],
                   duplicates="drop")
    haz_tl = pd.DataFrame({"tl_q": tl_q, "y": y, "C": tr_x["Compound"]})
    print("  P(pit | TyreLife percentile within Compound):")
    by_tl = haz_tl.groupby("tl_q", observed=True)["y"].agg(["size", "mean"])
    print(by_tl.to_string())
    print("\n  P(pit | laps_to_race_end bucket):")
    le_b = pd.cut(tr_x["laps_to_race_end"],
                  bins=[-0.5, 0.5, 2.5, 5.5, 10.5, 999],
                  labels=["last", "1-2", "3-5", "6-10", "11+"])
    by_le = pd.DataFrame({"b": le_b, "y": y}).groupby("b", observed=True)["y"].agg(["size", "mean"])
    print(by_le.to_string())

    # ============ P2.1 — heuristic meta features ====================
    print("\n--- P2.1: K=4 + heuristic meta features")
    F4 = _expand(P4)
    F4_test = _expand(P4_test)
    H_tr = tr_x[h_feats].astype(float).values
    H_te = te_x[h_feats].astype(float).values
    F4H = np.hstack([F4, H_tr])
    F4H_test = np.hstack([F4_test, H_te])

    def fit_lr_meta(F, F_test, C=1.0):
        oof_pred = np.zeros(len(y))
        test_acc = np.zeros(F_test.shape[0])
        for tr, va in splits:
            lr = LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            oof_pred[va] = lr.predict_proba(F[va])[:, 1]
            test_acc += lr.predict_proba(F_test)[:, 1] / N_FOLDS
        return oof_pred, test_acc

    oof_K4, _ = fit_lr_meta(F4, F4_test)
    oof_K4H, test_K4H = fit_lr_meta(F4H, F4H_test, C=0.3)  # more reg with more feats
    auc_K4 = float(roc_auc_score(y, oof_K4))
    auc_K4H = float(roc_auc_score(y, oof_K4H))
    delta_p21_bp = (auc_K4H - auc_K4) * 1e4
    print(f"  K=4 LR meta plain          : {auc_K4:.5f}")
    print(f"  K=4 + heuristic meta input : {auc_K4H:.5f}  (Δ {delta_p21_bp:+.3f} bp)")
    np.save(ART / "oof_lane2_K4plus_heuristic_meta_strat.npy", oof_K4H)
    np.save(ART / "test_lane2_K4plus_heuristic_meta_strat.npy",
            np.column_stack([1 - test_K4H, test_K4H]))

    # ============ P2.2 — rule clamps ================================
    print("\n--- P2.2: rule clamps on PRIMARY")
    primary_clamped = primary_oof.copy()
    # Last-lap clamp: floor to bucket empirical mean
    last_mask = (tr_x["is_last_lap"] == 1).values
    if last_mask.sum() > 0:
        last_pit_rate = float(y[last_mask].mean())
        primary_clamped[last_mask] = last_pit_rate
        print(f"  is_last_lap mask: n={last_mask.sum()}, "
              f"empirical P(pit)={last_pit_rate:.4f} (was meta_mean="
              f"{primary_oof[last_mask].mean():.4f})")
    # Tyre-life cliff: if percentile > 0.99, push to high
    cliff_mask = (tr_x["tyre_life_pctile_in_compound"] > 0.99).values
    if cliff_mask.sum() > 0:
        cliff_pit_rate = float(y[cliff_mask].mean())
        primary_clamped[cliff_mask] = np.maximum(
            primary_clamped[cliff_mask], cliff_pit_rate)
        print(f"  tyre_life_pctile>0.99 mask: n={cliff_mask.sum()}, "
              f"empirical P(pit)={cliff_pit_rate:.4f}")

    auc_primary = float(roc_auc_score(y, primary_oof))
    auc_clamped = float(roc_auc_score(y, primary_clamped))
    delta_p22_bp = (auc_clamped - auc_primary) * 1e4
    print(f"  PRIMARY plain   : {auc_primary:.5f}")
    print(f"  PRIMARY + clamps: {auc_clamped:.5f}  (Δ {delta_p22_bp:+.3f} bp)")
    np.save(ART / "oof_lane2_rule_clamped_strat.npy", primary_clamped)

    # ============ P2.3 — Compound-tier monotonic LGBM ===============
    print("\n--- P2.3: Compound-tier monotonic LGBM single base")
    lgb_feats = ["TyreLife", "compound_tier", "LapNumber", "race_progress",
                 "tyre_life_pctile_in_compound"]
    X_lgb = tr_x[lgb_feats].astype(float).values
    X_lgb_te = te_x[lgb_feats].astype(float).values

    LGB = dict(
        objective="binary", metric="auc", learning_rate=0.05,
        num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
        monotone_constraints=[1, 0, 0, 0, 1],  # TyreLife +, pctile +
    )
    oof_mono = np.zeros(len(y))
    test_mono = np.zeros(X_lgb_te.shape[0])
    for fold, (tr, va) in enumerate(splits):
        ds_tr = lgb.Dataset(X_lgb[tr], label=y[tr])
        ds_va = lgb.Dataset(X_lgb[va], label=y[va], reference=ds_tr)
        booster = lgb.train(
            LGB, ds_tr, num_boost_round=400, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        oof_mono[va] = booster.predict(X_lgb[va])
        test_mono += booster.predict(X_lgb_te) / N_FOLDS
    auc_mono = float(roc_auc_score(y, oof_mono))
    print(f"  monotonic LGBM standalone OOF: {auc_mono:.5f}")

    # K=4 + 1 gate
    F4_mono = _expand(np.hstack([P4, oof_mono.reshape(-1, 1)]))
    F4_mono_test = _expand(np.hstack([P4_test, test_mono.reshape(-1, 1)]))
    oof_K4M, _ = fit_lr_meta(F4_mono, F4_mono_test)
    auc_K4M = float(roc_auc_score(y, oof_K4M))
    delta_p23_bp = (auc_K4M - auc_K4) * 1e4
    print(f"  K=4 plain : {auc_K4:.5f}")
    print(f"  K=4+1 mono: {auc_K4M:.5f}  (Δ {delta_p23_bp:+.3f} bp)")
    np.save(ART / "oof_lane2_compound_monotonic_lgbm_strat.npy", oof_mono)
    np.save(ART / "test_lane2_compound_monotonic_lgbm_strat.npy",
            np.column_stack([1 - test_mono, test_mono]))

    # ρ vs PRIMARY
    rho_p21 = float(spearmanr(oof_K4H, primary_oof)[0])
    rho_p23 = float(spearmanr(oof_mono, primary_oof)[0])

    out = {
        "K4_bases": K4_FWD,
        "heuristic_features": h_feats,
        "D2_1_p_pit_by_tyre_life_pctile": (
            haz_tl.groupby("tl_q", observed=True)["y"]
            .agg(["size", "mean"]).reset_index()
            .to_dict(orient="records")),
        "D2_1_p_pit_by_laps_to_race_end": (
            pd.DataFrame({"b": le_b, "y": y})
              .groupby("b", observed=True)["y"]
              .agg(["size", "mean"]).reset_index()
              .to_dict(orient="records")),
        "P2_1_K4_LR_meta_plain_oof": auc_K4,
        "P2_1_K4_plus_heuristic_oof": auc_K4H,
        "P2_1_delta_bp": float(delta_p21_bp),
        "P2_1_rho_vs_primary": rho_p21,
        "P2_2_PRIMARY_oof": auc_primary,
        "P2_2_PRIMARY_clamped_oof": auc_clamped,
        "P2_2_delta_bp": float(delta_p22_bp),
        "P2_3_mono_LGBM_standalone_oof": auc_mono,
        "P2_3_K4_plus_mono_oof": auc_K4M,
        "P2_3_delta_bp": float(delta_p23_bp),
        "P2_3_rho_vs_primary": rho_p23,
        "verdict_P2_1": ("PASS" if delta_p21_bp >= 0.5
                         else "AMBIG" if delta_p21_bp >= -0.1
                         else "NULL"),
        "verdict_P2_2": ("PASS" if delta_p22_bp >= 0.2
                         else "AMBIG" if delta_p22_bp >= -0.1
                         else "NULL"),
        "verdict_P2_3": ("PASS" if delta_p23_bp >= 0.5
                         else "AMBIG" if delta_p23_bp >= -0.1
                         else "NULL"),
        "wall_s": time.time() - t0,
    }
    (ART / "probe_lane2_priors.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {ART/'probe_lane2_priors.json'}. Wall {out['wall_s']:.1f}s")
    print(f"Verdicts: P2.1 {out['verdict_P2_1']} | P2.2 {out['verdict_P2_2']} "
          f"| P2.3 {out['verdict_P2_3']}")


if __name__ == "__main__":
    main()
