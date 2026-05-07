"""scripts/d19_lgbm_v4_fs.py — A5-light: LGBM with v4 yekenot recipe + field-state.

Tests the load-bearing question from friction tag
`lr-meta-rank-lock-strong-anchor` (6 cross-confirmations):
  Does fs_cum_pits + ~24 cross-row aggregates survive at meta-add level
  when integrated INTO an anchor recipe (yekenot v4) instead of stacked
  as a separate base on top of v4+h1d?

A5-full would be CB-v4-fs (CatBoost on Kaggle GPU); this is the LGBM-CPU
proxy that fits in <60 min locally. If LGBM-v4-fs lifts ≥+0.5 bp at
K=27+1, the case for spinning up CB-v4-fs on Kaggle GPU is strong.

Recipe (excludes orig-aug for CPU time budget):
  1. make_features_static (v3 base: kitchen-sink + 6 TE configs)
  2. yekenot items 2/3/4 inline: floor-cat / count-enc / KBins
  3. field-state aggregates per (Race,Year,LapNumber) ± Compound:
     fs_cum_pits, fs_n_pitting_now, fs_mean_TyreLife, etc.
  4. CV TE per-fold (TE_CONFIGS)
  5. LGBM 5-fold StratifiedKFold(seed=42) CPU

Outputs:
  scripts/artifacts/oof_d19_lgbm_v4_fs_strat.npy
  scripts/artifacts/test_d19_lgbm_v4_fs_strat.npy
  scripts/artifacts/d19_lgbm_v4_fs_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

sys.path.insert(0, str(Path(__file__).parent))
from p1_features import (  # noqa: E402
    make_features_static, fit_fs_a, apply_fs_a, cv_target_encode,
    TE_CONFIGS, feature_columns_for_lgbm,
)

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=-1,
    n_jobs=-1,
    seed=SEED,
)
NUM_BOOST = 4000
EARLY_STOP = 200


def add_yekenot_items(df_in: pd.DataFrame, fit: bool, state: dict) -> pd.DataFrame:
    """Yekenot items 2 (floor-cat), 3 (count-enc), 4 (KBins). Mirrors v4 kernel."""
    df = df_in.copy()
    floor_cols = ["lap_div_rp", "tl_div_ln",
                  "LapNumber", "TyreLife", "RaceProgress", "LapTime (s)",
                  "Cumulative_Degradation", "Position"]
    for col in floor_cols:
        if col not in df.columns:
            continue
        cat_name = f"floor_{col.replace(' (s)', '').replace('/', '_')}_"
        floor_series = pd.Series(
            np.floor(df[col].fillna(0).astype("float32").values),
            index=df.index)
        if fit:
            _, uniques = floor_series.factorize()
            state[f"floor_{col}"] = {float(v): i for i, v in enumerate(uniques)}
        df[cat_name] = (floor_series.map(state[f"floor_{col}"])
                        .fillna(-1).astype("int32"))

    count_src_cols = [
        ("Driver", "Driver"), ("Race", "Race"), ("Compound", "Compound"),
        ("Year", "Year"), ("Stint", "Stint"),
        ("Race_Compound_", "RaceCompound"),
        ("Race_Year_", "RaceYear"),
        ("Driver_Compound_", "DriverCompound"),
    ]
    for src, alias in count_src_cols:
        if src not in df.columns:
            continue
        out = f"count_{alias}"
        if fit:
            counts = df[src].value_counts()
            state[f"count_{src}"] = counts.to_dict()
        df[out] = (df[src].map(state[f"count_{src}"])
                   .fillna(0).astype("int32"))

    bin_specs = [("RaceProgress", 200, "RaceProgress_q200_"),
                 ("LapTime (s)", 7, "LapTime_q7_")]
    for col, n_bins, out in bin_specs:
        if col not in df.columns:
            continue
        vals = df[[col]].fillna(df[col].median())
        if fit:
            kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                                  strategy="quantile", subsample=None)
            kb.fit(vals)
            state[f"kbins_{col}"] = kb
        df[out] = state[f"kbins_{col}"].transform(vals).ravel().astype("int32")

    return df


def add_field_state(df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Field-state cross-row aggregates. Source is train+test combined.
    PitStop (the column not the label) is AV-safe per Rule 25 (AV-AUC 0.502).
    `cross-row-aggregates-survive-strict-fold-safe-audit` confirmed.
    """
    g = source_df.groupby(["Race", "Year", "LapNumber"])
    a = g.agg(
        fs_field_size=("id", "size"),
        fs_n_pitting_now=("PitStop", "sum"),
        fs_pit_rate_now=("PitStop", "mean"),
        fs_mean_TyreLife=("TyreLife", "mean"),
        fs_max_TyreLife=("TyreLife", "max"),
        fs_min_TyreLife=("TyreLife", "min"),
        fs_std_TyreLife=("TyreLife", "std"),
        fs_mean_Stint=("Stint", "mean"),
        fs_max_Stint=("Stint", "max"),
        fs_mean_Position=("Position", "mean"),
        fs_mean_LapTime=("LapTime (s)", "mean"),
        fs_mean_RaceProgress=("RaceProgress", "mean"),
    ).reset_index()
    df = df.merge(a, on=["Race", "Year", "LapNumber"], how="left")

    rs = (source_df.sort_values(["Race", "Year", "LapNumber"])
                  .groupby(["Race", "Year", "LapNumber"])["PitStop"]
                  .sum().reset_index())
    rs["fs_cum_pits"] = rs.groupby(["Race", "Year"])["PitStop"].cumsum()
    rs["fs_cum_pit_lap_count"] = (rs.groupby(["Race", "Year"])["PitStop"]
                                    .cumcount() + 1)
    rs["fs_cum_pit_rate"] = rs["fs_cum_pits"] / rs["fs_cum_pit_lap_count"]
    df = df.merge(
        rs[["Race", "Year", "LapNumber",
            "fs_cum_pits", "fs_cum_pit_lap_count", "fs_cum_pit_rate"]],
        on=["Race", "Year", "LapNumber"], how="left",
    )

    gc = source_df.groupby(["Race", "Year", "LapNumber", "Compound"])
    ac = gc.agg(
        fs_compound_n=("id", "size"),
        fs_compound_n_pitting=("PitStop", "sum"),
        fs_compound_pit_rate=("PitStop", "mean"),
        fs_compound_mean_TyreLife=("TyreLife", "mean"),
        fs_compound_max_TyreLife=("TyreLife", "max"),
    ).reset_index()
    df = df.merge(ac, on=["Race", "Year", "LapNumber", "Compound"], how="left")

    df["fs_TyreLife_vs_field_mean"] = df["TyreLife"] - df["fs_mean_TyreLife"]
    df["fs_TyreLife_vs_field_max"] = df["TyreLife"] - df["fs_max_TyreLife"]
    df["fs_Position_vs_field_mean"] = df["Position"] - df["fs_mean_Position"]
    df["fs_Stint_vs_field_mean"] = df["Stint"] - df["fs_mean_Stint"]
    return df


def main():
    t0 = time.time()
    print("=== d19 LGBM-v4-fs (A5-light) ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"train {train.shape}  test {test.shape}")

    # Combined train+test for field-state source (PitStop is feature, AV-safe)
    print("[fs] computing field-state source (train+test combined)...")
    src = pd.concat([train, test], ignore_index=True)

    print("[fe] make_features_static (v3 base)...")
    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)

    print("[fe] adding yekenot items 2/3/4 (floor-cat, count-enc, KBins)...")
    train_S = add_yekenot_items(train_S, fit=True, state=state)
    test_S = add_yekenot_items(test_S, fit=False, state=state)

    print("[fs] merging field-state aggregates...")
    train_S = add_field_state(train_S, src)
    test_S = add_field_state(test_S, src)
    fs_cols = [c for c in train_S.columns if c.startswith("fs_")]
    print(f"  added {len(fs_cols)} fs_* columns")

    y = train_S[TARGET].astype(int).reset_index(drop=True)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    # Sample fs_a fit to discover feature columns
    sample_ti = fold_list[0][0]
    sample_fs_a = fit_fs_a(train_S.iloc[sample_ti])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    sample_train = sample_train.reindex(
        columns=list(dict.fromkeys(list(sample_train.columns) + fs_cols)))
    feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in feats and c not in cat_cols:
            cat_cols.append(c)
    feats = feats + [n for _, _, n in TE_CONFIGS]
    print(f"feats {len(feats)} cat {len(cat_cols)}")

    n_train, n_test = len(y), len(test_S)
    oof = np.zeros(n_train, dtype=np.float32)
    test_pred = np.zeros(n_test, dtype=np.float32)
    fold_aucs, iters, walls = [], [], []

    for fold, (ti, vi) in enumerate(fold_list, 1):
        t1 = time.time()
        print(f"\n--- Fold {fold} | ti={len(ti)} va={len(vi)} ---")

        fs_a = fit_fs_a(train_S.iloc[ti])
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)

        y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
        inner_skf = StratifiedKFold(N_FOLDS, shuffle=True,
                                    random_state=SEED + fold)
        inner_folds = list(inner_skf.split(np.zeros(len(y_ti)), y_ti))
        for cols, smooth, te_name in TE_CONFIGS:
            if not all(c in train_ti.columns for c in cols):
                continue
            ti_enc, _ = cv_target_encode(
                train_ti, train_va, cols, y_ti, inner_folds, smoothing=smooth)
            train_ti[te_name] = ti_enc

            def _kfn(df, cols=cols):
                s = df[cols[0]].fillna("MISSING").astype(str)
                for c in cols[1:]:
                    s = s + "__" + df[c].fillna("MISSING").astype(str)
                return s.reset_index(drop=True)
            gm = float(y_ti.mean())
            k_ti = _kfn(train_ti)
            stats = (pd.DataFrame({"key": k_ti.values, "target": y_ti.values})
                     .groupby("key")["target"].agg(["sum", "count"]))
            stats["enc"] = ((stats["sum"] + smooth * gm) / (stats["count"] + smooth))
            mp = stats["enc"].to_dict()
            train_va[te_name] = _kfn(train_va).map(mp).fillna(gm).values
            test_fold[te_name] = _kfn(test_fold).map(mp).fillna(gm).values

        X_tr = train_ti.reindex(columns=feats, fill_value=0).copy()
        X_va = train_va.reindex(columns=feats, fill_value=0).copy()
        X_te = test_fold.reindex(columns=feats, fill_value=0).copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("int32")
            X_va[c] = X_va[c].astype("int32")
            X_te[c] = X_te[c].astype("int32")
        num_cols = [c for c in feats if c not in cat_cols]
        for X in [X_tr, X_va, X_te]:
            X[num_cols] = X[num_cols].fillna(0).astype(np.float32)

        y_tr = train_ti[TARGET].astype(int).values
        y_va = train_va[TARGET].astype(int).values
        dtr = lgb.Dataset(X_tr[feats], label=y_tr,
                          categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(X_va[feats], label=y_va,
                          categorical_feature=cat_cols, free_raw_data=False)
        booster = lgb.train(
            LGB_PARAMS, dtr, num_boost_round=NUM_BOOST,
            valid_sets=[dva], valid_names=["va"],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(0)],
        )
        p_va = booster.predict(X_va[feats], num_iteration=booster.best_iteration)
        p_te = booster.predict(X_te[feats], num_iteration=booster.best_iteration)
        oof[vi] = p_va
        test_pred += p_te / N_FOLDS

        fold_auc = float(roc_auc_score(y_va, p_va))
        wall = time.time() - t1
        fold_aucs.append(fold_auc)
        iters.append(int(booster.best_iteration or NUM_BOOST))
        walls.append(wall)
        print(f"  fold AUC {fold_auc:.5f}  iters {booster.best_iteration}  "
              f"wall {wall:.0f}s")

    oof_auc = float(roc_auc_score(y, oof))
    print(f"\n=== OOF AUC: {oof_auc:.5f}  total wall: {time.time()-t0:.0f}s ===")
    print(f"     fold AUCs: {[round(a,5) for a in fold_aucs]}")

    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / "oof_d19_lgbm_v4_fs_strat.npy", oof2)
    np.save(ART / "test_d19_lgbm_v4_fs_strat.npy", test2)
    summary = dict(
        oof_auc=oof_auc, fold_aucs=fold_aucs, iters=iters, walls=walls,
        n_feats=len(feats), n_cat=len(cat_cols), n_fs=len(fs_cols),
        wall_total_s=time.time() - t0,
    )
    (ART / "d19_lgbm_v4_fs_results.json").write_text(json.dumps(summary, indent=2))
    print(f"     saved oof_d19_lgbm_v4_fs_strat.npy + test + results.json")


if __name__ == "__main__":
    main()
