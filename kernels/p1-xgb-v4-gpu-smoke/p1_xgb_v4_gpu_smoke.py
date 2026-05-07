"""P1 XGBoost v4 GPU — yekenot transfer FE on a different model class.

Same v4 FE chain as kernels/p1-single-cb-v4-gpu/ (Rozen base + arithmetic
ratios + floor-cat + count-encoding + KBins + 6 CV-TE + orig-aug per fold).
Replaces CatBoostClassifier with xgboost.XGBClassifier(tree_method="hist",
device="cuda"). Int cats are fed as numeric (XGB-GPU's enable_categorical
is less mature than CatBoost native; integer categories are already
factorized and split-friendly).

Goal: independent GBDT-class base on the v4 FE recipe. v4 single-CB is
ρ=0.971 vs PRIMARY; XGB on identical FE should land ρ < 0.97 (same FE,
different model class), making it a stack-add candidate.

Outputs:
  oof_p1_xgb_v4_gpu_seed42.npy
  test_p1_xgb_v4_gpu_seed42.npy
  p1_xgb_v4_gpu_results.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

# === CONFIG ===
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
WALL_BUDGET_S = 5 * 3600
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

# v4 single-model max-effort
WITH_ORIG_DATA = True       # item 7
N_SEEDS = 1
SEEDS_BAG = [42, 13, 71]    # used SEEDS_BAG[:N_SEEDS]
MAX_ROUNDS = 3000           # smoke v2 hit 2500 cap at fold-1 with model still improving
SMOKE = True


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

    # === V4 ITEM 2: Floor-cat (np.floor + factorize) for ratios + key continuous ===
    floor_cols = [
        "lap_div_rp", "tl_div_ln",  # arithmetic ratios already computed
        "LapNumber", "TyreLife", "RaceProgress", "LapTime (s)",
        "Cumulative_Degradation", "Position",
    ]
    for col in floor_cols:
        if col not in df.columns:
            continue
        cat_name = f"floor_{col.replace(' (s)','').replace('/','_')}_"
        floor_series = pd.Series(
            np.floor(df[col].fillna(0).astype("float32").values),
            index=df.index)
        if fit:
            codes, uniques = floor_series.factorize()
            state[f"floor_{col}"] = {float(v): i for i, v in enumerate(uniques)}
        df[cat_name] = (floor_series.map(state[f"floor_{col}"])
                        .fillna(-1).astype("int32"))

    # === V4 ITEM 3: Count encoding for raw cats + combo cats + Year/Stint ===
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

    # === V4 ITEM 4: KBinsDiscretizer (yekenot's exact: 200/RaceProgress, 7/LapTime) ===
    bin_specs = [("RaceProgress", 200, "RaceProgress_q200_"),
                 ("LapTime (s)",    7, "LapTime_q7_")]
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
def xgb_params(seed: int, max_iters: int, depth: int = 8) -> dict:
    """XGBoost-GPU parameters analogous to v4 CatBoost recipe.

    Matches v4 CB intent where applicable; XGB-specific defaults
    elsewhere (no CTR, no rsm-equivalent on GPU, depth caps lower).
    """
    return dict(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=max_iters,
        learning_rate=0.03,
        max_depth=depth,
        reg_lambda=8.0,
        min_child_weight=20.0,    # analogous to min_data_in_leaf=20
        subsample=0.8,
        colsample_bytree=0.8,     # XGB column subsampling on GPU is OK
        random_state=seed,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
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

    n_train, n_test = len(y), len(test_S)

    # Per-seed OOF/test arrays
    oofs_seed = {s: np.zeros(n_train, dtype=np.float32) for s in SEEDS_BAG[:N_SEEDS]}
    tests_seed = {s: np.zeros(n_test, dtype=np.float32) for s in SEEDS_BAG[:N_SEEDS]}
    fold_aucs_seed = {s: [] for s in SEEDS_BAG[:N_SEEDS]}
    iters_seed = {s: [] for s in SEEDS_BAG[:N_SEEDS]}
    walls_seed = {s: [] for s in SEEDS_BAG[:N_SEEDS]}

    n_eff_folds = 1 if SMOKE else N_FOLDS
    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
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

        # Per-seed CB fit
        for sd in SEEDS_BAG[:N_SEEDS]:
            t1 = time.time()
            params = xgb_params(sd, MAX_ROUNDS, depth=8)
            m = xgb.XGBClassifier(
                **params,
                early_stopping_rounds=200,
            )
            m.fit(X_tr, y_tr,
                  eval_set=[(X_va, train_va[TARGET].astype(int))],
                  sample_weight=weights, verbose=200)
            sorted_vi = train_S.iloc[vi].index.values
            oofs_seed[sd][sorted_vi] = m.predict_proba(X_va)[:, 1]
            tests_seed[sd] += m.predict_proba(X_te)[:, 1] / N_FOLDS
            fold_aucs_seed[sd].append(float(roc_auc_score(
                train_va[TARGET].astype(int).values,
                m.predict_proba(X_va)[:, 1])))
            iters_seed[sd].append(int(m.best_iteration or m.n_estimators))
            walls_seed[sd].append(time.time() - t1)
            print(f"  [seed{sd}] fold{fold} AUC={fold_aucs_seed[sd][-1]:.5f} "
                  f"iters={iters_seed[sd][-1]} wall={walls_seed[sd][-1]:.1f}s")

        print(f"--- fold {fold} total wall {time.time()-t0:.1f}s "
              f"(elapsed {time.time()-t0_total:.0f}s) ---")

    # Persist per-seed and rank-bag
    results = {"feats_n": len(feats), "cat_n": len(cat_cols),
               "with_orig_data": WITH_ORIG_DATA, "n_seeds": N_SEEDS,
               "wall_total_s": time.time() - t0_total, "by_seed": {}}
    completed = []
    for sd in SEEDS_BAG[:N_SEEDS]:
        if not fold_aucs_seed[sd]:
            continue
        # Map back to original train.csv id order
        order = train_S["id"].values
        sort_back = np.argsort(order)
        oof_aligned = oofs_seed[sd][sort_back]
        order_te = test_S["id"].values
        id_to_pos = {tid: i for i, tid in enumerate(order_te)}
        orig_te = pd.read_csv(data_dir / "test.csv", usecols=[ID_COL])[ID_COL].values
        test_aligned = np.array([tests_seed[sd][id_to_pos[t]] for t in orig_te])
        auc = float(roc_auc_score(y, oofs_seed[sd]))
        np.save(WORK / f"oof_p1_xgb_v4_gpu_seed{sd}.npy",
                np.column_stack([1 - oof_aligned, oof_aligned]).astype(np.float64))
        np.save(WORK / f"test_p1_xgb_v4_gpu_seed{sd}.npy",
                np.column_stack([1 - test_aligned, test_aligned]).astype(np.float64))
        results["by_seed"][f"seed{sd}"] = dict(
            oof_auc=auc, fold_aucs=fold_aucs_seed[sd],
            iters=iters_seed[sd], walls=walls_seed[sd])
        completed.append(sd)
        print(f"[seed{sd}] OOF AUC = {auc:.5f}")

    if len(completed) >= 2:
        oof_bag = np.mean([rankdata(oofs_seed[sd]) / len(oofs_seed[sd])
                           for sd in completed], axis=0)
        test_bag = np.mean([rankdata(tests_seed[sd]) / len(tests_seed[sd])
                            for sd in completed], axis=0)
        bag_auc = float(roc_auc_score(y, oof_bag))
        # align bag arrays to original train.csv id order
        order = train_S["id"].values
        sort_back = np.argsort(order)
        oof_bag_al = oof_bag[sort_back]
        order_te = test_S["id"].values
        id_to_pos = {tid: i for i, tid in enumerate(order_te)}
        orig_te = pd.read_csv(data_dir / "test.csv", usecols=[ID_COL])[ID_COL].values
        test_bag_al = np.array([test_bag[id_to_pos[t]] for t in orig_te])
        np.save(WORK / "oof_p1_xgb_v4_gpu_bag.npy",
                np.column_stack([1 - oof_bag_al, oof_bag_al]).astype(np.float64))
        np.save(WORK / "test_p1_xgb_v4_gpu_bag.npy",
                np.column_stack([1 - test_bag_al, test_bag_al]).astype(np.float64))
        # write submission for the bag
        sub = pd.read_csv(data_dir / "sample_submission.csv")
        sub[TARGET] = np.clip(test_bag_al, 0.001, 0.999)
        sub.to_csv(WORK / "submission_p1_xgb_v4_gpu_bag.csv", index=False)
        results["bag"] = dict(oof_auc=bag_auc, seeds=completed)
        print(f"[bag] OOF AUC = {bag_auc:.5f} ({len(completed)} seeds)")

    (WORK / "p1_xgb_v4_gpu_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nDONE wall={time.time()-t0_total:.0f}s")


if __name__ == "__main__":
    main()
