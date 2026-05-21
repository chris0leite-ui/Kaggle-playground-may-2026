"""R39 inventor — Pointwise CB classifier on K=17 + pilkwang domain features +
rohit categorical crossings/binning + original-data concatenation per fold.

Diagnostic of R38's failure: listwise ranker can't use cohort-non-discriminative
domain features. R39 uses POINTWISE Logloss instead — same features, different
loss family. Mirrors rohit/ps-s6e5-catboost-10-fold-cv's recipe distilled to
our 5-fold + K=17 input.

Key recipe differences from cb_v4 in our K=17 pool:
- rohit's tuned hyperparams: lr 0.018, depth 8, l2 8.5, random_strength 0.65,
  bagging_temp 0.45, min_data_in_leaf 48, max_ctr_complexity 4,
  auto_class_weights="Balanced"
- 15 pilkwang domain features (compound priors, lap math)
- 10 rohit categorical crossings (Race_Compound, Compound_Stint, etc.)
- Original f1_strategy_dataset_v4 CONCATENATED to train per fold (rohit-style)
- IsOriginalData flag

Output: oof_R39_cb_pointwise_strat.npy + test_R39_cb_pointwise_strat.npy
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

K17_FILES = [
    ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy",     "test_d17_h1d_yekenot_full_strat.npy"),
    ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy",      "test_p1_single_cb_v4_gpu_strat.npy"),
    ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy",             "test_f1_hgbc_deep_strat.npy"),
    ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ("qAT",       "dgp_v3_qAT_K1_oof.npy",                  "dgp_v3_qAT_K1_test.npy"),
    ("qAV",       "dgp_v3_qAV_K1_7feat_oof.npy",            "dgp_v3_qAV_K1_7feat_test.npy"),
    ("qAO",       "dgp_v3_qAO_knn_multi_oof.npy",           "dgp_v3_qAO_knn_multi_test.npy"),
    ("qAA",       "dgp_v3_qAA_stint_imputed_oof.npy",       "dgp_v3_qAA_stint_imputed_test.npy"),
    ("qAF",       "dgp_v3_qAF_d16plus_oof.npy",             "dgp_v3_qAF_d16plus_test.npy"),
    ("qAK",       "dgp_v3_qAK_knn3_oof.npy",                "dgp_v3_qAK_knn3_test.npy"),
    ("K27_100k",  "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
                  "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"),
    ("seg_fe",    "oof_r4_segment_fe_strat.npy",            "test_r4_segment_fe_strat.npy"),
    ("HMM",       "oof_r4_hmm_seq_strat.npy",               "test_r4_hmm_seq_strat.npy"),
    ("R12_cb_horizon",          "oof_R12_cb_horizon_strat.npy",         "test_R12_cb_horizon_strat.npy"),
    ("R13_cb_stint_completion", "oof_R13_cb_stint_completion_strat.npy","test_R13_cb_stint_completion_strat.npy"),
    ("R14_tabm",                "oof_R14_tabm_strat.npy",               "test_R14_tabm_strat.npy"),
    ("R15_xendcg_per_seg",      "oof_R15_xendcg_per_seg_strat.npy",     "test_R15_xendcg_per_seg_strat.npy"),
]

COMPOUND_HARDNESS = {"SOFT": 1.0, "MEDIUM": 2.0, "HARD": 3.0, "INTERMEDIATE": 4.0, "WET": 5.0}
COMPOUND_EXPECTED_LIFE = {"SOFT": 25.0, "MEDIUM": 35.0, "HARD": 45.0, "INTERMEDIATE": 30.0, "WET": 40.0}
COMPOUND_WINDOW_START = {"SOFT": 0.64, "MEDIUM": 0.68, "HARD": 0.72, "INTERMEDIATE": 0.62, "WET": 0.60}

ORIGINAL_PATHS = [
    "/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv",
    "/kaggle/input/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv",
]


def gpu_boot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=10).strip()
        print(f"[boot] GPU: {out}", flush=True)
    except Exception as e:
        print(f"[boot] nvidia-smi failed: {e}", flush=True)


def find_data_dir():
    for p in Path("/kaggle/input").rglob("train.csv"):
        return p.parent
    raise RuntimeError("train.csv not found")


def find_artifacts_dir():
    for p in Path("/kaggle/input").rglob("oof_R15_xendcg_per_seg_strat.npy"):
        return p.parent
    raise RuntimeError("artifacts dir not found")


def find_original():
    for p in ORIGINAL_PATHS:
        if Path(p).exists():
            return Path(p)
    for p in Path("/kaggle/input").rglob("f1_strategy_dataset_v4.csv"):
        return p
    return None


def _pos(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[:, -1] if a.shape[1] >= 2 else a.ravel()
    return a.astype(np.float32)


def _safe_divide(a, b):
    return np.divide(a, np.where(np.abs(b) < 1e-9, np.nan, b))


def build_features(df, k17_cols, is_original=False):
    """K=17 logits + pilkwang domain + rohit crossings. Returns
    (feature_df, cat_feature_indices)."""
    out = pd.DataFrame(index=df.index)

    # K=17 logits as columns
    for i, name in enumerate([n for n, _, _ in K17_FILES]):
        out[f"k17_{name}"] = k17_cols[:, i]

    compound = df["Compound"].astype(str).fillna("MISSING")
    tyre_life = df["TyreLife"].astype(float)
    lap_number = df["LapNumber"].astype(float)
    race_progress = df["RaceProgress"].astype(float).clip(0.001, 1.05)
    stint = df["Stint"].astype("Int64")
    race = df["Race"].astype(str).fillna("MISSING")
    driver = df["Driver"].astype(str).fillna("MISSING")
    year = df["Year"].astype(int)

    # Pilkwang domain priors
    hardness = compound.map(COMPOUND_HARDNESS).fillna(2.5).astype(np.float32)
    expected_life = compound.map(COMPOUND_EXPECTED_LIFE).fillna(35.0).astype(np.float32)
    window_start = compound.map(COMPOUND_WINDOW_START).fillna(0.68).astype(np.float32)

    estimated_laps = (_safe_divide(lap_number, race_progress)
                      .clip(lower=1, upper=120).astype(np.float32))

    out["CompoundHardness"] = hardness
    out["ExpectedMaxLife"] = expected_life
    out["EstimatedRaceLaps"] = estimated_laps
    out["LapsRemainingEstimate"] = (estimated_laps - lap_number).clip(-10, 120).astype(np.float32)
    out["RemainingRaceProgress"] = (1.0 - race_progress).astype(np.float32)
    out["TyreLifePct"] = _safe_divide(tyre_life, expected_life).clip(0, 3).astype(np.float32)
    out["InCompoundPitWindow"] = (out["TyreLifePct"] >= window_start).astype(np.float32)
    out["NormalizedTyreLifeEstimate"] = _safe_divide(tyre_life, estimated_laps).astype(np.float32)
    out["TyreLife_x_CompoundHardness"] = (tyre_life * hardness).astype(np.float32)
    out["CompoundIsDry"] = compound.isin(["SOFT", "MEDIUM", "HARD"]).astype(np.float32)
    out["CompoundIsWetLike"] = compound.isin(["INTERMEDIATE", "WET"]).astype(np.float32)

    # Raw features
    out["TyreLife"] = tyre_life.astype(np.float32)
    out["LapNumber"] = lap_number.astype(np.float32)
    out["RaceProgress"] = race_progress.astype(np.float32)
    if "Position" in df.columns:
        out["Position"] = df["Position"].astype(float).astype(np.float32)
    if "Position_Change" in df.columns:
        out["Position_Change"] = df["Position_Change"].astype(float).astype(np.float32)
    if "LapTime_Delta" in df.columns:
        out["LapTime_Delta"] = df["LapTime_Delta"].astype(float).astype(np.float32)
    if "Cumulative_Degradation" in df.columns:
        out["Cumulative_Degradation"] = df["Cumulative_Degradation"].astype(float).astype(np.float32)
    out["Year"] = year.astype(np.float32)
    out["Stint"] = stint.astype("Int64").astype(np.float32).fillna(-1)

    # Rohit-style categorical crossings (strings)
    out["Compound"] = compound.values
    out["Race"] = race.values
    out["Driver"] = driver.values
    out["Race_Compound"] = (race + "__" + compound).values
    out["Driver_Compound"] = (driver + "__" + compound).values
    out["Race_Year"] = (race + "__" + year.astype(str)).values
    out["Compound_Stint"] = (compound + "__" + stint.astype("Int64").astype(str).fillna("NA")).values

    # IsOriginalData
    out["IsOriginalData"] = float(is_original)

    # Determine categorical feature indices
    cat_cols = ["Compound", "Race", "Driver", "Race_Compound", "Driver_Compound",
                "Race_Year", "Compound_Stint"]
    cat_idx = [out.columns.get_loc(c) for c in cat_cols]
    return out, cat_idx, cat_cols


def main():
    t0 = time.time()
    print("== R39 inventor: Pointwise CB on K=17 + domain + rohit crossings + orig ==",
          flush=True)
    gpu_boot()

    import catboost as cb
    print(f"[setup] CatBoost {cb.__version__}", flush=True)

    data_dir = find_data_dir()
    art_dir = find_artifacts_dir()
    orig_path = find_original()
    print(f"[setup] data at {data_dir}", flush=True)
    print(f"[setup] artifacts at {art_dir}", flush=True)
    print(f"[setup] original at {orig_path}", flush=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y_train = train[TARGET].astype(int).values
    print(f"[data] train {train.shape}  test {test.shape}", flush=True)

    # K=17 logits for train and test
    oof_cols, test_cols = [], []
    for name, of, tf in K17_FILES:
        oof_cols.append(_pos(np.load(art_dir / of)))
        test_cols.append(_pos(np.load(art_dir / tf)))
    K17_train = np.column_stack(oof_cols)
    K17_test = np.column_stack(test_cols)

    F_train, cat_idx, cat_cols = build_features(train, K17_train, is_original=False)
    F_test, _, _ = build_features(test, K17_test, is_original=False)
    print(f"[data] features train {F_train.shape} test {F_test.shape}", flush=True)
    print(f"[cat_features] {cat_cols}", flush=True)

    # Original data: only used as training augmentation (no K=17 logits available
    # for original rows — fill with NaN/0 so CB sees them as missing).
    F_orig = None
    y_orig = None
    if orig_path is not None:
        try:
            original = pd.read_csv(orig_path)
            if TARGET in original.columns:
                # Build "fake" K=17 cols filled with 0.5 (neutral - CB will treat as
                # missing-like signal, IsOriginalData flag distinguishes)
                K17_orig_fake = np.full((len(original), len(K17_FILES)), 0.5, dtype=np.float32)
                F_orig_df, _, _ = build_features(original, K17_orig_fake, is_original=True)
                # Align columns to F_train
                missing_cols = set(F_train.columns) - set(F_orig_df.columns)
                for c in missing_cols:
                    F_orig_df[c] = 0.0 if F_train[c].dtype != object else ""
                F_orig_df = F_orig_df[F_train.columns]
                F_orig = F_orig_df
                y_orig = original[TARGET].astype(int).values
                print(f"[orig] augmenting train with {F_orig.shape}", flush=True)
        except Exception as e:
            print(f"[orig] skipping original data: {e}", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_train)), y_train))

    oof_meta = np.zeros(len(y_train), dtype=np.float64)
    test_meta = np.zeros(len(F_test), dtype=np.float64)
    fold_aucs, walls = [], []

    cb_params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=4000,
        learning_rate=0.018,
        depth=8,
        l2_leaf_reg=8.5,
        random_strength=0.65,
        bootstrap_type="Bayesian",
        bagging_temperature=0.45,
        auto_class_weights="Balanced",
        border_count=254,
        one_hot_max_size=10,
        min_data_in_leaf=48,
        max_ctr_complexity=4,
        grow_policy="SymmetricTree",
        random_seed=SEED,
        early_stopping_rounds=300,
        task_type="GPU",
        devices="0",
        verbose=400,
        allow_writing_files=False,
    )

    for k, (ti, vi) in enumerate(fold_list, 1):
        t_f = time.time()
        X_tr = F_train.iloc[ti].reset_index(drop=True)
        y_tr = y_train[ti]
        X_val = F_train.iloc[vi].reset_index(drop=True)
        y_val = y_train[vi]

        # Augment with original data per fold (rohit-style)
        if F_orig is not None:
            X_tr_aug = pd.concat([X_tr, F_orig], axis=0, ignore_index=True)
            y_tr_aug = np.concatenate([y_tr, y_orig])
        else:
            X_tr_aug, y_tr_aug = X_tr, y_tr

        print(f"\n--- Fold {k}/{N_FOLDS} | aug_train={len(X_tr_aug)} val={len(vi)} ---",
              flush=True)

        booster = cb.CatBoostClassifier(**cb_params)
        booster.fit(X_tr_aug, y_tr_aug,
                    eval_set=(X_val, y_val),
                    cat_features=cat_idx,
                    use_best_model=True)

        pred_va = booster.predict_proba(X_val)[:, 1]
        oof_meta[vi] = pred_va
        pred_te = booster.predict_proba(F_test)[:, 1]
        test_meta += pred_te / N_FOLDS

        auc_va = float(roc_auc_score(y_val, pred_va))
        wall = time.time() - t_f
        fold_aucs.append(auc_va)
        walls.append(wall)
        print(f"  fold {k}: AUC(val)={auc_va:.5f}  wall={wall:.0f}s", flush=True)

    auc_full = float(roc_auc_score(y_train, oof_meta))
    print(f"\n[result] R39 pointwise CB OOF AUC: {auc_full:.6f}", flush=True)

    np.save(WORK / "oof_R39_cb_pointwise_strat.npy", oof_meta.astype(np.float32))
    np.save(WORK / "test_R39_cb_pointwise_strat.npy", test_meta.astype(np.float32))
    print(f"[save] wrote oof_R39_cb_pointwise_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R39_cb_pointwise_gpu",
        oof_auc=auc_full,
        fold_aucs=fold_aucs,
        fold_walls_s=walls,
        wall_total_s=time.time() - t0,
        n_features=int(F_train.shape[1]),
        cat_features=cat_cols,
        used_original=F_orig is not None,
        cb_params={k: v for k, v in cb_params.items() if k != "devices"},
    )
    (WORK / "r39_pointwise_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] wrote summary  total wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
