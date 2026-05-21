"""R32 inventor — CatBoost YetiRank META on K=17 with (Year, Race, Stint)
cohort grouping. Sibling of R21 (CB YetiRank on (Y,R,L)): same model family
and loss, different cohort axis.

Why this cohort: (Year, Race, Stint) groups laps within a single stint (period
between pit stops) of a single race. The listwise loss ranks laps by pit
probability *within that stint*. R21's (Y,R,L) was per-lap cross-driver;
this is per-stint cross-driver-and-lap, a coarser axis.

Group stats: 528 groups, median 358 laps/group, max 5,621 — large but
YetiRank tolerates large groups.

Output: oof_R32_yetirank_yrs_strat.npy + test_R32_yetirank_yrs_strat.npy
to /kaggle/working/.
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
    print("== R32 inventor: CatBoost YetiRank META on (Year,Race,Stint) cohort ==",
          flush=True)
    gpu_boot()

    import catboost as cb
    print(f"[setup] CatBoost {cb.__version__}", flush=True)

    data_dir = find_data_dir()
    art_dir = find_artifacts_dir()
    print(f"[setup] data at {data_dir}", flush=True)
    print(f"[setup] artifacts at {art_dir}", flush=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"[data] train {train.shape}  test {test.shape}", flush=True)

    oof_cols, test_cols, names = [], [], []
    for name, of, tf in K17_FILES:
        oof_cols.append(_pos(np.load(art_dir / of)))
        test_cols.append(_pos(np.load(art_dir / tf)))
        names.append(name)
    K17_oof = np.column_stack(oof_cols)
    K17_test = np.column_stack(test_cols)
    print(f"[data] K=17 OOF: {K17_oof.shape}  test: {K17_test.shape}", flush=True)

    F_oof = expand(K17_oof)
    F_test = expand(K17_test)
    print(f"[data] expanded OOF: {F_oof.shape}  test: {F_test.shape}", flush=True)

    cohort_keys = (train["Year"].astype(str) + "|" +
                   train["Race"].astype(str) + "|" +
                   train["Stint"].astype(str))
    sizes = cohort_keys.value_counts().values
    print(f"[cohort] (Y,R,Stint) count: {len(sizes)} "
          f"median={int(np.median(sizes))} mean={sizes.mean():.1f} "
          f"max={sizes.max()} min={sizes.min()}", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_all)), y_all))
    oof_meta = np.zeros(len(y_all), dtype=np.float64)
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
        ti_df = train.iloc[ti].copy().reset_index(drop=True)
        # Sort to ensure contiguous groups: (Driver, Year, Race, LapNumber)
        sort_idx = np.lexsort((
            ti_df["LapNumber"].astype(int).values,
            ti_df["Stint"].astype(int).values,
            ti_df["Race"].astype(str).values,
            ti_df["Year"].astype(int).values,
        ))
        ti_df_sorted = ti_df.iloc[sort_idx].reset_index(drop=True)
        X_tr = F_oof[ti][sort_idx]
        y_tr = y_all[ti][sort_idx]
        group_id_tr = (ti_df_sorted["Year"].astype(str) + "|" +
                       ti_df_sorted["Race"].astype(str) + "|" +
                       ti_df_sorted["Stint"].astype(str))
        codes = pd.Categorical(group_id_tr).codes.astype(np.int64)
        n_groups = int(codes.max() + 1)
        print(f"\n--- Fold {k}/{N_FOLDS} | ti={len(ti)} groups={n_groups} ---",
              flush=True)

        train_pool = cb.Pool(X_tr, label=y_tr.astype(np.float32), group_id=codes)
        booster = cb.CatBoost(cb_params)
        booster.fit(train_pool)

        pred_va = booster.predict(F_oof[vi])
        oof_meta[vi] = pred_va
        pred_te = booster.predict(F_test)
        test_meta += pred_te / N_FOLDS

        auc_va = float(roc_auc_score(y_all[vi], pred_va))
        wall = time.time() - t_f
        fold_aucs.append(auc_va)
        walls.append(wall)
        print(f"  fold {k}: AUC(val)={auc_va:.5f}  wall={wall:.0f}s", flush=True)

    auc_full = float(roc_auc_score(y_all, oof_meta))
    print(f"\n[result] R32 YetiRank (Y,R,Stint) cohort OOF AUC: {auc_full:.6f}",
          flush=True)

    np.save(WORK / "oof_R32_yetirank_yrs_strat.npy", oof_meta.astype(np.float32))
    np.save(WORK / "test_R32_yetirank_yrs_strat.npy", test_meta.astype(np.float32))
    print(f"[save] wrote oof_R32_yetirank_yrs_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R32_yetirank_yrs_gpu",
        cohort="(Year, Race, Stint)",
        oof_auc=auc_full,
        fold_aucs=fold_aucs,
        fold_walls_s=walls,
        wall_total_s=time.time() - t0,
        bases=names,
        cb_params={k: v for k, v in cb_params.items() if k != "devices"},
    )
    (WORK / "r32_yetirank_yrs_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] wrote r32_yetirank_yrs_summary.json  "
          f"total wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
