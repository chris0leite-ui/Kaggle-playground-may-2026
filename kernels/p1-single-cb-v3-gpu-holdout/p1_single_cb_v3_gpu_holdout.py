"""P1 single-CB v3 GPU **HOLDOUT** kernel — Rule 24 mandatory check.

Independent 80/20 split (HOLDOUT_SEED=99 ≠ main 5-fold seed=42) to
catch FS_A/TE/orig-aug-style cross-fold or distribution-shift leaks
before any LB submit. Trains the same CB recipe (Bernoulli+min_dil+
default CTR+Year_cat+...) on 80% with full FE state fit on 80% only,
predicts 20% holdout, reports holdout_auc.

Pass criterion (Rule 24):
  |holdout_auc − OOF_auc| ≤ 10 bp  (within fold-std)
Fail criterion: holdout ≪ OOF (FS_A leak or TE-via-full-train leak).

Why this is run on Kaggle GPU rather than locally:
  CB-on-GPU at 2500 iters × 351k rows takes ~5 min on P100; locally
  CPU would take ~60 min. We need the same recipe path-for-path.

Outputs (under /kaggle/working/):
  p1_holdout_v3_gpu_results.json   { holdout_auc, train_auc, oof_auc_ref, ... }
  oof_holdout_v3_gpu_seed42.npy    (predictions on the 80% train via inner 5-fold)
  test_holdout_v3_gpu_seed42.npy   (predictions on the 20% holdout)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# === CONFIG ===
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
HOLDOUT_SEED = 99            # independent of main 5-fold (Rule 24 origin)
WALL_BUDGET_S = 5 * 3600
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

# Holdout mode: train one CB on 80% with full FE state, eval on 20%.
WITH_ORIG_DATA = False
N_SEEDS = 1
MAX_ROUNDS = 2500
OOF_AUC_REF = 0.94993        # measured 5-fold OOF (commit a0444f3)


# === FE chain (verbatim from scripts/p1_features.py) ===
COMPOUND_MAX_LIFE_MAP = {
    "SOFT": 15, "MEDIUM": 30, "HARD": 50,
    "INTERMEDIATE": 25, "WET": 20,
}
COMBO_COLS = [("Race", "Compound"), ("Race", "Year"), ("Driver", "Compound")]


def _ndrv(s):
    return str(s).strip().split()[-1].lower()


def _nrace(s):
    s = str(s).strip().lower()
    return re.sub(r"grand\s+prix|\bgp\b", "", s).strip()


def _load_hist_priors(ext_pit_path):
    if not ext_pit_path.exists():
        return None
    df = pd.read_csv(ext_pit_path)
    df.columns = df.columns.str.strip()
    drv_col = next((c for c in df.columns if "driver" in c.lower()), None)
    race_col = next((c for c in df.columns if "grand" in c.lower() or "race" in c.lower()), None)
    lap_col = next((c for c in df.columns if "lap" in c.lower() and "time" not in c.lower()), None)
    if not (drv_col and lap_col):
        return None
    df["_dk"] = df[drv_col].map(_ndrv)
    df["_lap"] = pd.to_numeric(df[lap_col], errors="coerce")
    if race_col:
        df["_rk"] = df[race_col].map(_nrace)
    out = {"driver": {}, "circuit": {}}
    drv = (df.dropna(subset=["_lap"]).groupby("_dk")
           .agg(pit_hist_avg_lap=("_lap", "mean"),
                pit_hist_std_lap=("_lap", "std")))
    drv["pit_hist_std_lap"] = drv["pit_hist_std_lap"].fillna(8.0)
    out["driver"] = drv.to_dict(orient="index")
    if race_col:
        ckt = (df.dropna(subset=["_lap"]).groupby("_rk")
               .agg(pit_ckt_avg_lap=("_lap", "mean"),
                    pit_ckt_std_lap=("_lap", "std")))
        ckt["pit_ckt_std_lap"] = ckt["pit_ckt_std_lap"].fillna(8.0)
        out["circuit"] = ckt.to_dict(orient="index")
    return out


def fit_fs_a(df_with_labels):
    if "PitNextLap" not in df_with_labels.columns:
        raise ValueError("fit_fs_a requires PitNextLap column")
    fs_a = {}
    fs_a["pit_laps"] = (df_with_labels[df_with_labels["PitNextLap"] == 1]
        .groupby(["Race", "Year"])["LapNumber"].mean()
        .rename("race_avg_pit_lap"))
    fs_a["total_laps"] = (df_with_labels.groupby(["Race", "Year"])["LapNumber"]
        .max().rename("race_total_laps"))
    fs_a["comp_life"] = (df_with_labels[df_with_labels["PitNextLap"] == 1]
        .groupby("Compound")["TyreLife"].mean()
        .rename("compound_avg_life"))
    fs_a["race_stints"] = (df_with_labels.groupby(["Race", "Year"])["Stint"]
        .max().rename("race_max_stint"))
    fs_a["compound_race_lt"] = (df_with_labels.groupby(
        ["Race", "Year", "Compound"])["LapTime (s)"].median()
        .rename("compound_race_median_lt"))
    fs_a["race_compound_max"] = (df_with_labels.groupby(
        ["Race", "Year", "Compound"])["TyreLife"].max()
        .rename("race_compound_max_life"))
    fs_a["dc_avg_stint_life"] = (df_with_labels[df_with_labels["PitNextLap"] == 1]
        .groupby(["Driver", "Compound"])["TyreLife"].mean()
        .rename("dc_avg_stint_life"))
    return fs_a


def apply_fs_a(df, fs_a):
    df = df.copy()
    df = df.merge(fs_a["pit_laps"].reset_index(),    on=["Race", "Year"], how="left")
    df = df.merge(fs_a["total_laps"].reset_index(),  on=["Race", "Year"], how="left")
    df = df.merge(fs_a["comp_life"].reset_index(),   on="Compound",      how="left")
    df = df.merge(fs_a["race_stints"].reset_index(), on=["Race", "Year"], how="left")
    df = df.merge(fs_a["compound_race_lt"].reset_index(),
                  on=["Race", "Year", "Compound"], how="left")
    df = df.merge(fs_a["race_compound_max"].reset_index(),
                  on=["Race", "Year", "Compound"], how="left")
    df = df.merge(fs_a["dc_avg_stint_life"].reset_index(),
                  on=["Driver", "Compound"], how="left")
    df["pit_window_flag"] = (np.abs(df["LapNumber"]
        - df["race_avg_pit_lap"].fillna(35)) <= 3).astype(int)
    df["tyre_vs_comp_avg"] = df["TyreLife"] - df["compound_avg_life"].fillna(25)
    df["overdue_pit"] = (df["TyreLife"] > df["compound_avg_life"].fillna(25)).astype(int)
    df["laps_remaining_race"] = df["race_total_laps"].fillna(60) - df["LapNumber"]
    df["tyre_age_pct_race"] = df["TyreLife"] / (df["race_total_laps"].fillna(60) + 1)
    df["stint_progress"] = df["Stint"] / (df["race_max_stint"].fillna(3) + 1)
    df["tyre_life_pct"] = df["TyreLife"] / df["compound_avg_life"].fillna(25).clip(lower=1)
    df["stint_end_est"] = df["stint_start_lap"] + df["compound_avg_life"].fillna(25)
    df["laps_until_stop"] = (df["stint_end_est"] - df["LapNumber"]).clip(lower=-20)
    df["pit_imminent"] = (df["laps_until_stop"] <= 2).astype(int)
    df["pit_in_5"] = (df["laps_until_stop"] <= 5).astype(int)
    df["lap_vs_compound_baseline"] = (df["LapTime (s)"]
        - df["compound_race_median_lt"].fillna(df["LapTime (s)"])).clip(-5, 15)
    df["tyre_freshness_pct"] = (1 - df["TyreLife"] / (df["race_compound_max_life"].fillna(40) + 1)).clip(0, 1)
    df["driver_vs_avg_life"] = (df["TyreLife"] - df["dc_avg_stint_life"].fillna(25)).clip(-20, 20)
    df["driver_overdue_personal"] = (df["TyreLife"] > df["dc_avg_stint_life"].fillna(25)).astype(int)
    df["deg_x_win"] = df["Cumulative_Degradation"] * df["is_pit_window"]
    df["over_x_win"] = df["overdue_pit"] * df["is_pit_window"]
    df["tyre_x_pres"] = df["TyreLife"] * df["position_pressure"]
    return df


def make_features_static(df_in, fit=False, state=None, ext_pit_path=None):
    if state is None:
        state = {}
    df = (df_in.copy()
          .sort_values(["Driver", "Race", "Year", "LapNumber"])
          .reset_index(drop=True))
    df["tyre_life_sq"] = df["TyreLife"] ** 2
    df["tyre_life_log"] = np.log1p(df["TyreLife"])
    df["tyre_life_sqrt"] = np.sqrt(df["TyreLife"])
    df["deg_per_lap"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)
    df["compound_max_life"] = df["Compound"].map(COMPOUND_MAX_LIFE_MAP).fillna(30)
    df["compound_tyre_norm"] = (df["TyreLife"] / df["compound_max_life"]).clip(0, 2)
    df["tyre_overdue_norm"] = (df["compound_tyre_norm"] > 0.85).astype(int)
    df["est_total_laps"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-9)
                            ).round().clip(30, 80)
    df["laps_remaining"] = (df["est_total_laps"] - df["LapNumber"]).clip(lower=0)
    df["tyre_pct_remaining"] = df["TyreLife"] / (df["laps_remaining"] + 1)
    df["is_pit_window"] = ((df["RaceProgress"] >= 0.28)
                           & (df["RaceProgress"] <= 0.62)).astype(int)
    df["is_late_race"] = (df["RaceProgress"] > 0.75).astype(int)
    df["position_pressure"] = df["Position"] * (1 - df["RaceProgress"])
    df["urgency_score"] = df["Cumulative_Degradation"].abs() * (1 - df["RaceProgress"])
    df["race_phase"] = pd.cut(
        df["RaceProgress"], bins=[0, .25, .5, .75, 1.01],
        labels=[0, 1, 2, 3]).astype(float)
    df["norm_position"] = 1 - (df["Position"] - 1) / 19.0
    df["life_x_progress"] = df["TyreLife"] * df["RaceProgress"]
    grp_id = (df["Driver"].astype(str) + "|"
              + df["Race"].astype(str) + "|"
              + df["Year"].astype(str)).factorize()[0]
    df["_grp_id"] = grp_id
    grp_id_s = pd.Series(grp_id)
    df["delta_lag1"] = df["LapTime_Delta"].shift(1).where(
        df["_grp_id"] == grp_id_s.shift(1).values)
    df["delta_lag2"] = df["LapTime_Delta"].shift(2).where(
        df["_grp_id"] == grp_id_s.shift(2).values)
    df["prev_pit"] = df["PitStop"].shift(1).where(
        df["_grp_id"] == grp_id_s.shift(1).values).fillna(0)
    df["delta_accel"] = df["LapTime_Delta"] - df["delta_lag1"]
    for w in [3, 5, 7, 10, 15]:
        df[f"roll{w}_lt"] = (df.groupby("_grp_id")["LapTime (s)"]
                             .rolling(w, min_periods=1).mean()
                             .reset_index(level=0, drop=True).values)
    for w in [3, 7]:
        df[f"roll{w}_d"] = (df.groupby("_grp_id")["LapTime_Delta"]
                            .rolling(w, min_periods=1).mean()
                            .reset_index(level=0, drop=True).values)
    df["roll3_std"] = (df.groupby("_grp_id")["LapTime (s)"]
                       .rolling(3, min_periods=1).std()
                       .reset_index(level=0, drop=True).fillna(0).values)
    df["lap_vs_r3"] = df["LapTime (s)"] - df["roll3_lt"]
    df["lap_vs_r7"] = df["LapTime (s)"] - df["roll7_lt"]
    df["lap_vs_r15"] = df["LapTime (s)"] - df["roll15_lt"]
    g_stint = df.groupby(["Driver", "Race", "Year", "Stint"])
    df["lap_in_stint"] = g_stint.cumcount()
    df["stint_start_lap"] = g_stint["LapNumber"].transform("min")
    df = df.drop(columns=["_grp_id"])
    df["lap_div_rp"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype(np.float32)
    df["tl_div_ln"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype(np.float32)
    df["compound_ord"] = df["Compound"].map(
        {"SOFT": 2, "MEDIUM": 1, "HARD": 0, "INTERMEDIATE": 3, "WET": 4}).fillna(1)
    if fit and ext_pit_path is not None:
        state["hist"] = _load_hist_priors(ext_pit_path)
    hist = state.get("hist") if state else None
    df["_dk"] = df["Driver"].map(_ndrv)
    df["_rk"] = df["Race"].map(_nrace)
    if hist is not None:
        drv = hist.get("driver", {})
        ckt = hist.get("circuit", {})
        df["pit_hist_avg_lap"] = df["_dk"].map(
            lambda k: drv.get(k, {}).get("pit_hist_avg_lap", 30.0))
        df["pit_hist_std_lap"] = df["_dk"].map(
            lambda k: drv.get(k, {}).get("pit_hist_std_lap", 8.0))
        df["drv_laps_vs_hist"] = (df["TyreLife"] - df["pit_hist_avg_lap"]).clip(-25, 25)
        df["drv_hist_overdue"] = (df["TyreLife"] > df["pit_hist_avg_lap"]).astype(int)
        df["ckt_hist_avg_lap"] = df["_rk"].map(
            lambda k: ckt.get(k, {}).get("pit_ckt_avg_lap", 28.0))
        df["ckt_hist_std_lap"] = df["_rk"].map(
            lambda k: ckt.get(k, {}).get("pit_ckt_std_lap", 8.0))
        df["laps_vs_ckt_hist"] = (df["TyreLife"] - df["ckt_hist_avg_lap"]).clip(-25, 25)
        df["in_ckt_pit_window"] = (df["laps_vs_ckt_hist"].abs()
            <= df["ckt_hist_std_lap"]).astype(int)
    else:
        for c in ["pit_hist_avg_lap", "pit_hist_std_lap", "drv_laps_vs_hist",
                  "drv_hist_overdue", "ckt_hist_avg_lap", "ckt_hist_std_lap",
                  "laps_vs_ckt_hist", "in_ckt_pit_window"]:
            df[c] = 0.0
    df = df.drop(columns=["_dk", "_rk"], errors="ignore")
    for c1, c2 in COMBO_COLS:
        combo_str = df[c1].astype(str) + "_" + df[c2].astype(str)
        key = f"{c1}_{c2}_"
        if fit:
            codes, uniques = combo_str.factorize()
            state[f"combo_{key}"] = {v: i for i, v in enumerate(uniques)}
        df[key] = combo_str.map(state[f"combo_{key}"]).fillna(-1).astype("int32")
    for c in ["Driver", "Race", "Compound"]:
        if fit:
            codes, uniques = df[c].astype(str).factorize()
            state[f"cat_{c}"] = {v: i for i, v in enumerate(uniques)}
        df[f"{c}_cat"] = df[c].astype(str).map(state[f"cat_{c}"]).fillna(-1).astype("int32")
    return df, state


def cv_target_encode(train_df, test_df, group_cols, target, fold_list, smoothing=30):
    global_mean = float(target.mean())
    n = len(train_df)
    oof_enc = np.full(n, global_mean, dtype=np.float32)
    def _key(df):
        s = df[group_cols[0]].fillna("MISSING").astype(str)
        for c in group_cols[1:]:
            s = s + "__" + df[c].fillna("MISSING").astype(str)
        return s.reset_index(drop=True)
    key = _key(train_df)
    key_test = _key(test_df)
    target_arr = target.reset_index(drop=True)
    for ti, vi in fold_list:
        stats = (pd.DataFrame({"key": key.iloc[ti].values,
                               "target": target_arr.iloc[ti].values})
                 .groupby("key")["target"].agg(["sum", "count"]))
        stats["enc"] = ((stats["sum"] + smoothing * global_mean)
                        / (stats["count"] + smoothing))
        oof_enc[vi] = key.iloc[vi].map(stats["enc"].to_dict()).fillna(global_mean).values
    stats_full = (pd.DataFrame({"key": key.values, "target": target_arr.values})
                  .groupby("key")["target"].agg(["sum", "count"]))
    stats_full["enc"] = ((stats_full["sum"] + smoothing * global_mean)
                         / (stats_full["count"] + smoothing))
    test_enc = key_test.map(stats_full["enc"].to_dict()).fillna(global_mean).values
    return oof_enc.astype(np.float32), test_enc.astype(np.float32)


TE_CONFIGS = [
    (["Driver", "Race", "Year"], 20, "te_drv_race_yr"),
    (["Driver", "Race"], 30, "te_drv_race"),
    (["Race", "Compound"], 25, "te_race_comp"),
    (["Driver", "Compound"], 25, "te_drv_comp"),
    (["Race", "Year"], 20, "te_race_yr"),
    (["Driver", "Race", "Compound"], 15, "te_drv_race_comp"),
]


def feature_columns_for_lgbm(train_A):
    drop = {"id", "PitNextLap", "Driver", "Race", "Compound", "split"}
    feats = [c for c in train_A.columns if c not in drop]
    cat_cols = [c for c in feats
                if c.endswith("_cat") or c.endswith("_") or c == "race_phase"]
    return feats, cat_cols


# === Recipe ===
def cb_params(seed: int, max_iters: int, depth: int = 10) -> dict:
    # NOTE: `rsm` (column subsampling) is **not supported on GPU** with
    # binary Logloss — CatBoost only allows it for pairwise loss functions.
    # We rely on Bernoulli row-subsampling alone for regularisation on
    # the GPU path. CPU path can keep rsm=0.8.
    return dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=max_iters,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=8.0,
        one_hot_max_size=10,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        # rsm omitted (GPU restriction)
        min_data_in_leaf=20,
        od_type="Iter",
        od_wait=200,
        random_seed=seed,
        verbose=200,
        allow_writing_files=False,
        task_type="GPU",
        devices="0",  # Kaggle P100 single GPU; works on T4×2 too
        border_count=254,
    )


def find_data_dir(name="train.csv"):
    base = Path("/kaggle/input")
    matches = list(base.rglob(name))
    if not matches:
        raise RuntimeError(f"no {name} under /kaggle/input")
    return matches[0].parent


def find_orig_csv():
    base = Path("/kaggle/input")
    for cand in base.rglob("f1_strategy_dataset_v4.csv"):
        return cand
    return None


def find_pit_priors():
    base = Path("/kaggle/input")
    for cand in base.rglob("pitstops.csv"):
        if "1950" in str(cand) or "official" in str(cand):
            return cand
    return None


def main():
    t0_total = time.time()
    print("=== P1 single-CB v3 GPU ===")

    # GPU sanity
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True, timeout=10).strip()
        print(f"[boot] GPU: {out}")
    except Exception as e:
        print(f"[boot] nvidia-smi failed: {e}")

    data_dir = find_data_dir("train.csv")
    print(f"data_dir = {data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"train {train.shape}  test {test.shape}")

    pit_priors = find_pit_priors()
    print(f"pit_priors = {pit_priors}")
    train_S, state = make_features_static(train, fit=True, ext_pit_path=pit_priors)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    sample_ti = fold_list[0][0]
    sample_fs_a = fit_fs_a(train_S.iloc[sample_ti])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in feats and c not in cat_cols:
            cat_cols.append(c)
    feats = feats + [n for _, _, n in TE_CONFIGS]
    print(f"feats {len(feats)} cat {len(cat_cols)}")

    # Optional original-data augmentation
    orig_csv = find_orig_csv() if WITH_ORIG_DATA else None
    print(f"orig_csv = {orig_csv}  with_orig_data = {WITH_ORIG_DATA}")

    n_train = len(y)

    # === HOLDOUT MODE ===
    # 80/20 stratified split on the SYNTH train rows; HOLDOUT_SEED=99
    # is independent of the main 5-fold seed=42, so the FE state
    # (built on the 80%) has not seen any 20% labels.
    skf_h = StratifiedKFold(5, shuffle=True, random_state=HOLDOUT_SEED)
    train_idx, holdout_idx = next(skf_h.split(np.zeros(n_train), y.values))
    print(f"\n=== HOLDOUT 80/20 split (seed={HOLDOUT_SEED}) ===")
    print(f"  train_idx = {len(train_idx)}  holdout_idx = {len(holdout_idx)}")
    print(f"  pos rates: train={y.iloc[train_idx].mean():.4f} "
          f"holdout={y.iloc[holdout_idx].mean():.4f}")

    # Reusing the v3 fold-safe pipeline: treat train_idx as ti (the 80%
    # we fit FS_A on) and holdout_idx as vi (the held-out 20%). Apply the
    # same transforms (FS_A merge + CV-TE with full-ti stats for va).
    fold_list = [(train_idx, holdout_idx)]

    oof_seed42 = np.zeros(n_train, dtype=np.float32)
    holdout_pred = np.zeros(len(holdout_idx), dtype=np.float32)

    for fold, (ti, vi) in enumerate(fold_list, 1):
        if time.time() - t0_total > WALL_BUDGET_S:
            print(f"WALL BUDGET hit before fold {fold}; abort")
            break
        t0 = time.time()
        print(f"\n--- Fold {fold} | ti={len(ti)} va={len(vi)} ---")

        fs_a = fit_fs_a(train_S.iloc[ti])
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)

        y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
        # CV TE per-fold (inner stats for ti, ti-stats for va/test)
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
        cat_idx = [feats.index(c) for c in cat_cols]
        y_tr = train_ti[TARGET].astype(int).values
        weights = np.ones(len(X_tr), dtype=np.float32)

        # Original-data augmentation
        if WITH_ORIG_DATA and orig_csv is not None:
            orig_raw = pd.read_csv(orig_csv)
            orig_raw = orig_raw.drop(
                columns=[c for c in ("Normalized_TyreLife", "Position_Change")
                         if c in orig_raw.columns])
            next_id = int(train["id"].max()) + 1
            orig_raw["id"] = np.arange(next_id, next_id + len(orig_raw))
            orig_S, _ = make_features_static(orig_raw, fit=False, state=state)
            orig_FS = apply_fs_a(orig_S, fs_a)
            for cols, smooth, te_name in TE_CONFIGS:
                if not all(c in orig_FS.columns for c in cols):
                    continue
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
                orig_FS[te_name] = _kfn(orig_FS).map(mp).fillna(gm).values
            X_orig = orig_FS.reindex(columns=feats, fill_value=0).copy()
            for c in cat_cols:
                X_orig[c] = X_orig[c].astype("int32")
            X_orig[num_cols] = X_orig[num_cols].fillna(0).astype(np.float32)
            y_orig = orig_FS[TARGET].astype(int).values
            X_tr = pd.concat([X_tr, X_orig], ignore_index=True)
            y_tr = np.concatenate([y_tr, y_orig])
            weights = np.concatenate([weights,
                np.full(len(X_orig), 0.5, dtype=np.float32)])
            print(f"  + appended {len(X_orig)} orig rows  pos rate {y_orig.mean():.4f}")

        # === HOLDOUT MODE: single CB fit on 80%, eval on 20% ===
        sd = SEED  # single seed
        t1 = time.time()
        params = cb_params(sd, MAX_ROUNDS, depth=10)
        m = CatBoostClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=(X_va, train_va[TARGET].astype(int)),
              cat_features=cat_idx, sample_weight=weights,
              use_best_model=True)
        # Predictions on 20% holdout (vi positions)
        pred_h = m.predict_proba(X_va)[:, 1]
        holdout_pred[:] = pred_h
        oof_seed42[vi] = pred_h
        # Train self-AUC on 80% for sanity
        pred_tr = m.predict_proba(X_tr.iloc[:len(train_ti)])[:, 1]
        train_auc = float(roc_auc_score(y_tr[:len(train_ti)], pred_tr))
        holdout_auc = float(roc_auc_score(
            train_va[TARGET].astype(int).values, pred_h))
        best_iter = int(m.tree_count_)
        wall = time.time() - t1
        print(f"\n  [seed{sd}] holdout AUC = {holdout_auc:.5f}")
        print(f"  [seed{sd}] train self-AUC = {train_auc:.5f}")
        print(f"  [seed{sd}] best_iter = {best_iter}  wall = {wall:.1f}s")

    # Verdict per Rule 24: |holdout - OOF| ≤ 10 bp = clean
    delta_bp = (holdout_auc - OOF_AUC_REF) * 1e4
    verdict = "PASS" if abs(delta_bp) <= 10 else "FAIL"
    print(f"\n=== Rule 24 verdict ===")
    print(f"  OOF (5-fold, ref): {OOF_AUC_REF:.5f}")
    print(f"  Holdout AUC:       {holdout_auc:.5f}")
    print(f"  Δ holdout − OOF:   {delta_bp:+.2f} bp  ({verdict})")
    if verdict == "FAIL":
        print(f"  → DO NOT submit single-CB OOF/test artifacts to LB.")
        print(f"     The +12 bp K=21+1 lift is likely leakage-inflated.")
    else:
        print(f"  → Single-CB recipe is fold-safe; OOF→LB transfer plausible.")

    # Save holdout predictions + summary
    np.save(WORK / "oof_holdout_v3_gpu_seed42.npy",
            np.column_stack([1 - oof_seed42, oof_seed42]).astype(np.float64))
    np.save(WORK / "test_holdout_v3_gpu_seed42.npy",  # 20% predictions
            np.column_stack([1 - holdout_pred, holdout_pred]).astype(np.float64))
    results = dict(
        feats_n=len(feats), cat_n=len(cat_cols),
        with_orig_data=WITH_ORIG_DATA,
        oof_auc_ref=OOF_AUC_REF,
        holdout_auc=holdout_auc, train_auc=train_auc,
        delta_bp=float(delta_bp), verdict=verdict,
        best_iter=best_iter,
        n_train_ti=int(len(train_idx)), n_holdout=int(len(holdout_idx)),
        holdout_seed=HOLDOUT_SEED,
        wall_total_s=time.time() - t0_total,
    )
    (WORK / "p1_holdout_v3_gpu_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nDONE wall={time.time()-t0_total:.0f}s  → "
          f"p1_holdout_v3_gpu_results.json")


if __name__ == "__main__":
    main()
