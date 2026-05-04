"""CB slow-wide on Kaggle GPU — single seed full 5-fold two-anchor.

This kernel is the GPU counterpart of the CPU `slow-wide` variant. Same
pinned splits (StratKF seed=42 + GroupKF on Race), same params except:
  - task_type='GPU', devices='0'
  - iterations=4000 (CPU was capped at 2500), od_wait=200 (let ES fire)
  - lr=0.03, depth=6, l2=8.0, Year in CAT_COLS

Outputs (all under /kaggle/working/):
  oof_cb_slow-wide-gpu_seed42_strat.npy / test_*.npy
  oof_cb_slow-wide-gpu_seed42_groupkf.npy / test_*.npy
  cb_slow-wide-gpu_results.json

If wall < 30min, also runs seed=123 + seed=456 and rank-averages all 3
into oof_cb_slow-wide-gpu-bag_strat.npy / oof_cb_slow-wide-gpu-bag_groupkf.npy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

TARGET, ID_COL = "PitNextLap", "id"
CAT_COLS = ["Driver", "Race", "Compound", "Year"]
SEED, N_FOLDS = 42, 5
BASE_S, BASE_G = 0.94075, 0.92059

P = dict(iterations=4000, learning_rate=0.03, depth=6, l2_leaf_reg=8.0,
         eval_metric="AUC", od_type="Iter", od_wait=200,
         verbose=200, allow_writing_files=False,
         task_type="GPU", devices="0")

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def run_anchor(seed, anchor_name, splits, X, y, X_test):
    oof = np.zeros(len(y), dtype=np.float32)
    test_p = np.zeros(len(X_test), dtype=np.float32)
    scores, biters, walls = [], [], []
    pool_test = Pool(X_test, cat_features=CAT_COLS)
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        ptr = Pool(X.iloc[tr], y[tr], cat_features=CAT_COLS)
        pva = Pool(X.iloc[va], y[va], cat_features=CAT_COLS)
        params = dict(P); params["random_seed"] = seed
        m = CatBoostClassifier(**params); m.fit(ptr, eval_set=pva)
        p_va = m.predict_proba(pva)[:, 1]
        oof[va] = p_va
        test_p += m.predict_proba(pool_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        bi = int(m.get_best_iteration()); wall = time.time() - t0
        scores.append(s); biters.append(bi); walls.append(wall)
        print(f"  [seed{seed}/{anchor_name}] f{k}: AUC={s:.5f} bi={bi} wall={wall:.1f}s")
    auc = float(roc_auc_score(y, oof))
    return oof, test_p, auc, scores, biters, walls


def find_data_dir():
    import os
    base = Path("/kaggle/input")
    if not base.exists():
        raise RuntimeError(f"/kaggle/input does not exist; ls /kaggle: {os.listdir('/kaggle')}")
    print(f"DEBUG /kaggle/input = {os.listdir(base)}")
    for sub in base.iterdir():
        if sub.is_dir():
            files = list(sub.glob("*.csv"))
            print(f"DEBUG {sub} contains {[f.name for f in files]}")
            if any(f.name == "train.csv" for f in files):
                return sub
    raise RuntimeError("could not locate train.csv under /kaggle/input")


def main():
    t0 = time.time()
    data_dir = find_data_dir()
    print(f"Using data_dir={data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in CAT_COLS:
        X[c] = X[c].astype(str); X_test[c] = X_test[c].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_strat = list(skf.split(np.zeros(len(y)), y))
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_group = list(gkf.split(np.zeros(len(y)), y, train["Race"].values))

    results = {}
    seeds_to_run = [42, 123, 456]
    completed_seeds = []

    for seed in seeds_to_run:
        # Time gate: only run additional seeds if total wall < 5h (Kaggle 9h cap)
        if time.time() - t0 > 5 * 3600 and seed != 42:
            print(f"Skipping seed={seed} -- wall budget exceeded")
            break
        print(f"=== seed={seed} STRAT ===")
        oof_s, test_s, auc_s, fs_s, bi_s, w_s = run_anchor(
            seed, "strat", splits_strat, X, y, X_test)
        d_s = (auc_s - BASE_S) * 1e4
        print(f"[seed{seed}/strat] OOF={auc_s:.5f}  Δbase={d_s:+.1f}bp")
        np.save(WORK / f"oof_cb_slow-wide-gpu_seed{seed}_strat.npy",
                np.column_stack([1 - oof_s, oof_s]))
        np.save(WORK / f"test_cb_slow-wide-gpu_seed{seed}_strat.npy",
                np.column_stack([1 - test_s, test_s]))

        print(f"=== seed={seed} GROUPKF ===")
        oof_g, test_g, auc_g, fs_g, bi_g, w_g = run_anchor(
            seed, "groupkf", splits_group, X, y, X_test)
        d_g = (auc_g - BASE_G) * 1e4
        print(f"[seed{seed}/groupkf] OOF={auc_g:.5f}  Δbase={d_g:+.1f}bp")
        np.save(WORK / f"oof_cb_slow-wide-gpu_seed{seed}_groupkf.npy",
                np.column_stack([1 - oof_g, oof_g]))
        np.save(WORK / f"test_cb_slow-wide-gpu_seed{seed}_groupkf.npy",
                np.column_stack([1 - test_g, test_g]))

        results[f"seed{seed}"] = dict(
            strat_auc=auc_s, strat_delta_bp=d_s, strat_fold_scores=fs_s,
            strat_best_iters=bi_s, strat_walls=w_s,
            groupkf_auc=auc_g, groupkf_delta_bp=d_g, groupkf_fold_scores=fs_g,
            groupkf_best_iters=bi_g, groupkf_walls=w_g,
        )
        completed_seeds.append(seed)

    # rank-average bag if 2+ seeds completed
    if len(completed_seeds) >= 2:
        for anchor in ["strat", "groupkf"]:
            oofs = []
            tests = []
            for s in completed_seeds:
                o = np.load(WORK / f"oof_cb_slow-wide-gpu_seed{s}_{anchor}.npy")[:, 1]
                t = np.load(WORK / f"test_cb_slow-wide-gpu_seed{s}_{anchor}.npy")[:, 1]
                oofs.append(rankdata(o) / len(o))
                tests.append(rankdata(t) / len(t))
            bag_oof = np.mean(oofs, axis=0)
            bag_test = np.mean(tests, axis=0)
            np.save(WORK / f"oof_cb_slow-wide-gpu-bag_{anchor}.npy",
                    np.column_stack([1 - bag_oof, bag_oof]))
            np.save(WORK / f"test_cb_slow-wide-gpu-bag_{anchor}.npy",
                    np.column_stack([1 - bag_test, bag_test]))
            base = BASE_S if anchor == "strat" else BASE_G
            auc = roc_auc_score(y, bag_oof)
            results[f"bag_{anchor}"] = dict(auc=float(auc),
                                            delta_bp=(auc - base) * 1e4,
                                            seeds=completed_seeds)
            print(f"BAG {anchor}: AUC={auc:.5f}  Δbase={(auc-base)*1e4:+.1f}bp  "
                  f"(seeds={completed_seeds})")

    results["total_wall_s"] = time.time() - t0
    results["completed_seeds"] = completed_seeds
    (WORK / "cb_slow-wide-gpu_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print("DONE", json.dumps({k: v for k, v in results.items()
                              if not k.startswith("seed") or k == "bag_strat"
                              or k == "bag_groupkf"}, indent=2, default=str))


if __name__ == "__main__":
    main()
