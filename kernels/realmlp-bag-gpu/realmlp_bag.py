"""Move F — RealMLP-TD multi-seed bag on Kaggle T4 GPU.

Critic-loop §4 Move F: HANDOVER A.1 + nn-stack-priorities #1.
RealMLP at SEED=42 gave +52bp standalone Strat OOF and 10x OOF→LB
amplification when added to M5h pool (M5q LB +14bp). Seed-bag
prior: +1-3bp at known cost. Mechanism: variance reduction across
init / data-shuffle paths.

This kernel runs 2 NEW seeds (123, 456); existing seed-42 OOF/test
will be rank-averaged with these locally.

Per-fold checkpointing: save oof / test per seed after every fold,
so partial state survives if kernel hits the 9h cap.

Strat-only (R1).  Pinned: N_FOLDS=5, StratifiedKFold(shuffle=True).

Outputs:
  oof_realmlp_seed{S}_strat.npy     (n_train, 2)  per seed
  test_realmlp_seed{S}_strat.npy    (n_test, 2)   per seed
  realmlp_bag_results.json          per-seed AUCs + walls
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID_COL = "PitNextLap", "id"
SEEDS = [123, 456]
N_FOLDS = 5
SPLIT_SEED = 42                # SAME splits as seed-42 run for clean rank-average
BASE_S = 0.94075               # baseline_two_anchor Strat OOF (LB-proxy anchor)
SEED42_S = 0.94582             # known seed-42 OOF for sanity logging

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def gpu_boot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=10).strip()
        print(f"[boot] GPU info: {out}")
    except Exception as e:
        print(f"[boot] nvidia-smi failed: {e}")


def install_pytabkit():
    """Install pytabkit (RealMLP). Force torch 2.4 for sm_60 P100 compat
    in case Kaggle ignores machine_shape=GpuT4x2 (seen in kernels/realmlp-gpu)."""
    print("[setup] force-reinstall torch 2.4 (sm_60 P100 support) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
    print("[setup] installing pytabkit (--no-deps to preserve torch pin) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
        "pytabkit",
    ])
    print("[setup] installing pytabkit transitive deps ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "pytorch-lightning>=2.0,<2.5", "skorch", "torchmetrics<1.5",
    ])
    import importlib
    if "torch" in sys.modules:
        importlib.reload(sys.modules["torch"])
    import torch
    print(f"[setup] torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[setup] CUDA: {torch.version.cuda}, "
              f"device 0: {torch.cuda.get_device_name(0)}, "
              f"capability: {torch.cuda.get_device_capability(0)}")


def find_data_dir():
    base = Path("/kaggle/input")
    if not base.exists():
        raise RuntimeError(f"/kaggle/input missing; ls /kaggle: {os.listdir('/kaggle')}")
    matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv anywhere under {base}")
    return matches[0].parent


def run_one_seed(seed: int, X, y, X_test, splits, n_test: int) -> dict:
    """Train RealMLP-TD 5-fold for one seed; checkpoint after each fold."""
    from pytabkit import RealMLP_TD_Classifier

    oof = np.zeros(len(y), dtype=np.float32)
    test_proba = np.zeros(n_test, dtype=np.float32)
    fold_scores, fold_walls = [], []

    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        print(f"=== seed {seed} fold {k} (train={len(tr)} val={len(va)}) ===")
        model = RealMLP_TD_Classifier(
            device="cuda",
            random_state=seed,
            n_cv=1,
            val_metric_name="cross_entropy",
            use_ls=False,
            verbosity=1,
        )
        model.fit(X.iloc[tr], y[tr])
        p_va = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p_va
        test_proba += model.predict_proba(X_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        wall = time.time() - t_fold
        fold_scores.append(s); fold_walls.append(wall)
        print(f"  seed {seed} fold {k}: AUC={s:.5f}  wall={wall:.0f}s ({wall/60:.1f}min)")

        # Per-fold checkpoint
        np.save(WORK / f"oof_realmlp_seed{seed}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(WORK / f"test_realmlp_seed{seed}_strat.npy",
                np.column_stack([1 - test_proba, test_proba]))

    auc_full = float(roc_auc_score(y, oof))
    return dict(seed=seed, oof_score=auc_full,
                fold_std=float(np.std(fold_scores)),
                fold_scores=fold_scores, fold_walls=fold_walls,
                delta_vs_baseline_bp=(auc_full - BASE_S) * 1e4,
                delta_vs_seed42_bp=(auc_full - SEED42_S) * 1e4)


def main():
    t0 = time.time()
    gpu_boot()
    install_pytabkit()

    data_dir = find_data_dir()
    print(f"Using data_dir={data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"cat cols: {cat_cols}")
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    results = {"seeds": [], "split_seed": SPLIT_SEED}
    for seed in SEEDS:
        print(f"\n##### Starting seed {seed} (cumulative wall {time.time()-t0:.0f}s) #####")
        r = run_one_seed(seed, X, y, X_test, splits, len(test))
        results["seeds"].append(r)
        # Save running results JSON after each seed completes
        results["total_wall_s"] = time.time() - t0
        (WORK / "realmlp_bag_results.json").write_text(json.dumps(results, indent=2))
        print(f"seed {seed} done: OOF {r['oof_score']:.5f}  "
              f"Δ baseline {r['delta_vs_baseline_bp']:+.1f}bp  "
              f"Δ seed42 {r['delta_vs_seed42_bp']:+.1f}bp  "
              f"std {r['fold_std']:.5f}")

    print(f"\n=== ALL DONE (wall {time.time()-t0:.0f}s = "
          f"{(time.time()-t0)/60:.1f}min) ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
