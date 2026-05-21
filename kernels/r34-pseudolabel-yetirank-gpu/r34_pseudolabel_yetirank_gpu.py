"""R34 — pseudo-label augmented CatBoost YetiRank on K=17 (Y,R,L) cohort.

Mechanism: take K=18 PRIMARY-equivalent test predictions (calibrated), threshold
to pseudo-label high-confidence test rows, concat with train, retrain R21
architecture (CB YetiRank on K=17 with (Year,Race,LapNumber) cohort).

Why: closes the train/test distribution gap by exposing R21-style model to
test-row K=17 features during training. Untried per HANDOVER. The
K=18 base column (oof_K18_pathb_driverclass_stint_tau100000.npy +
test_K18_pathb_driverclass_stint_tau100000.npy) is well-calibrated; thresholds
at >=0.85 -> label 1, <=0.02 -> label 0 give ~80k pseudo-labeled rows.

OOF discipline: pseudo-labeled test rows go ONLY into the train_fold of each
fold. OOF predictions are made on held-out TRAIN rows only — pseudo-labeled
rows are never used as OOF targets.

Output: oof_R34_pseudolabel_yrl_strat.npy + test_R34_pseudolabel_yrl_strat.npy
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

# Same K=17 pool as R21
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

# Pseudo-label source = K18 PRIMARY-equivalent test predictions
K18_TEST_FILE = "test_K18_pathb_driverclass_stint_tau100000.npy"


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


def _pos(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[:, -1] if a.shape[1] >= 2 else a.ravel()
    return a.astype(np.float32)


def expand(M):
    n, k = M.shape
    rank_M = np.argsort(np.argsort(M, axis=0), axis=0) / max(n - 1, 1)
    eps = 1e-6
    P = np.clip(M, eps, 1 - eps)
    logit_M = np.log(P / (1 - P))
    return np.column_stack([M, rank_M, logit_M]).astype(np.float32)


def main():
    t0 = time.time()
    print("== R34 inventor: Pseudo-label CB YetiRank on (Y,R,L) cohort ==", flush=True)
    gpu_boot()

    import catboost as cb
    print(f"[setup] CatBoost {cb.__version__}", flush=True)

    data_dir = find_data_dir()
    art_dir = find_artifacts_dir()

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y_train = train[TARGET].astype(int).values
    print(f"[data] train {train.shape}  test {test.shape}", flush=True)

    # K=17 features for train and test
    oof_cols, test_cols = [], []
    for name, of, tf in K17_FILES:
        oof_cols.append(_pos(np.load(art_dir / of)))
        test_cols.append(_pos(np.load(art_dir / tf)))
    K17_train = np.column_stack(oof_cols)
    K17_test = np.column_stack(test_cols)
    F_train = expand(K17_train)
    F_test = expand(K17_test)
    print(f"[data] features train {F_train.shape}  test {F_test.shape}", flush=True)

    # K18 test predictions for pseudo-labels
    K18_test_probs = _pos(np.load(art_dir / K18_TEST_FILE))
    HI_THRESH, LO_THRESH = 0.85, 0.02
    pos_mask = K18_test_probs >= HI_THRESH
    neg_mask = K18_test_probs <= LO_THRESH
    pseudo_mask = pos_mask | neg_mask
    pseudo_y = pos_mask[pseudo_mask].astype(int)
    print(f"[pseudo] >={HI_THRESH}: {pos_mask.sum()} rows; <={LO_THRESH}: {neg_mask.sum()} rows", flush=True)
    print(f"[pseudo] total pseudo-labeled: {pseudo_mask.sum()} ({100*pseudo_mask.mean():.1f}% of test)", flush=True)
    print(f"[pseudo] positive rate of pseudo set: {pseudo_y.mean():.4f}", flush=True)
    print(f"[pseudo] train positive rate: {y_train.mean():.4f}", flush=True)

    F_pseudo = F_test[pseudo_mask]
    test_pseudo_df = test.iloc[pseudo_mask].copy().reset_index(drop=True)

    # Cohort keys for train (used for fold grouping) and pseudo (used for ranking groups)
    train_cohort = (train["Year"].astype(str) + "|" +
                    train["Race"].astype(str) + "|" +
                    train["LapNumber"].astype(str))
    pseudo_cohort = (test_pseudo_df["Year"].astype(str) + "|" +
                     test_pseudo_df["Race"].astype(str) + "|" +
                     test_pseudo_df["LapNumber"].astype(str))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_train)), y_train))

    oof_meta = np.zeros(len(y_train), dtype=np.float64)
    test_meta = np.zeros(len(F_test), dtype=np.float64)
    fold_aucs, walls = [], []

    cb_params = dict(
        loss_function="YetiRank",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        task_type="GPU",
        devices="0",
        verbose=200,
    )

    for k, (ti, vi) in enumerate(fold_list, 1):
        t_f = time.time()
        # Build augmented training: train_fold rows + ALL pseudo-labeled test rows
        ti_train_df = train.iloc[ti].copy().reset_index(drop=True)
        # Concat features and labels
        X_aug = np.vstack([F_train[ti], F_pseudo])
        y_aug = np.concatenate([y_train[ti], pseudo_y])
        # Build augmented cohort id (string for stable mapping)
        aug_cohort = pd.concat([
            train_cohort.iloc[ti].reset_index(drop=True),
            pseudo_cohort.reset_index(drop=True)
        ], ignore_index=True)
        # Sort by cohort, then break ties by (Year, Race, LapNumber)
        sort_keys = aug_cohort.values
        sort_idx = np.argsort(sort_keys, kind="stable")
        X_aug_sorted = X_aug[sort_idx]
        y_aug_sorted = y_aug[sort_idx]
        codes = pd.Categorical(aug_cohort.iloc[sort_idx].values).codes.astype(np.int64)
        n_groups = int(codes.max() + 1)
        # Max group size for safety check
        _, sizes = np.unique(codes, return_counts=True)
        max_group = int(sizes.max())
        print(f"\n--- Fold {k}/{N_FOLDS} | aug_rows={len(X_aug)} groups={n_groups} "
              f"max_group={max_group} ---", flush=True)
        if max_group > 1023:
            print(f"[warn] max group size {max_group} > 1023 (CB GPU YetiRank limit)", flush=True)

        train_pool = cb.Pool(X_aug_sorted, label=y_aug_sorted.astype(np.float32), group_id=codes)
        booster = cb.CatBoost(cb_params)
        booster.fit(train_pool)

        # Predict on TRAIN held-out fold
        pred_va = booster.predict(F_train[vi])
        oof_meta[vi] = pred_va

        # Predict on test
        pred_te = booster.predict(F_test)
        test_meta += pred_te / N_FOLDS

        auc_va = float(roc_auc_score(y_train[vi], pred_va))
        wall = time.time() - t_f
        fold_aucs.append(auc_va)
        walls.append(wall)
        print(f"  fold {k}: AUC(val)={auc_va:.5f}  wall={wall:.0f}s", flush=True)

    auc_full = float(roc_auc_score(y_train, oof_meta))
    print(f"\n[result] R34 pseudo-label YetiRank cohort OOF AUC: {auc_full:.6f}", flush=True)

    np.save(WORK / "oof_R34_pseudolabel_yrl_strat.npy", oof_meta.astype(np.float32))
    np.save(WORK / "test_R34_pseudolabel_yrl_strat.npy", test_meta.astype(np.float32))
    print(f"[save] wrote oof_R34_pseudolabel_yrl_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R34_pseudolabel_yrl_gpu",
        cohort="(Year, Race, LapNumber)",
        pseudo_thresholds=dict(hi=HI_THRESH, lo=LO_THRESH),
        pseudo_pos_rows=int(pos_mask.sum()),
        pseudo_neg_rows=int(neg_mask.sum()),
        pseudo_total=int(pseudo_mask.sum()),
        oof_auc=auc_full,
        fold_aucs=fold_aucs,
        fold_walls_s=walls,
        wall_total_s=time.time() - t0,
        bases=[name for name, _, _ in K17_FILES],
        cb_params={k: v for k, v in cb_params.items() if k != "devices"},
    )
    (WORK / "r34_pseudolabel_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] wrote r34_pseudolabel_summary.json  "
          f"total wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
