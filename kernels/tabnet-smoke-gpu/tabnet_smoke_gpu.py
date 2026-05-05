"""TabNet 1-fold SMOKE on Kaggle GPU — Strat fold 0 only (HANDOVER Path A.2).

Rule 2: 1-fold smoke before any GPU 5-fold. Day-3 RealMLP burned 175min
by skipping smoke. This kernel runs fold 0 of the 5-fold Strat split,
reports wall-time and AUC, and projects 5-fold cost. NO submission file
is written; that comes after the smoke clears the gate.

Why TabNet (vs RealMLP, our existing NN base):
  RealMLP-TD: MLP + numerical/categorical embeddings. Already in M5q
  pool (single seed), gave +14bp LB on a 10x OOF-to-LB amplification.
  TabNet:     attention-based feature selection, sparsemax masks per
  decision step. Different inductive bias = orthogonality candidate
  rather than another MLP seed. Diversity > tuning per
  audit/2026-05-05-nn-stack-priorities.md.

Pinned: SEED=42, N_FOLDS=5, StratifiedKFold(shuffle=True, random_state=SEED).
Runs fold 0 only (k=0) for the smoke.

Outputs (under /kaggle/working/):
  tabnet_smoke_results.json  -- AUC fold 0, wall, 5-fold projection
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
SMOKE_FOLD = 0
BASE_S = 0.94075           # baseline_two_anchor Strat OOF
REALMLP_FOLD0_REF = 0.94722  # E4 fold-0 reference for sanity comparison

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


def install_tabnet():
    """pytorch-tabnet pulls torch as a dep; pin to a version with sm_60.

    P100 (sm_60) compat: same lesson as RealMLP kernel — Kaggle silently
    routes GpuT4x2 jobs to P100 sometimes, and recent torch wheels drop
    sm_60. Force-reinstall torch 2.4 first, then pytorch-tabnet --no-deps.
    """
    print("[setup] force-reinstall torch 2.4 (sm_60 P100 support) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
    print("[setup] installing pytorch-tabnet (--no-deps to preserve torch pin) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
        "pytorch-tabnet",
    ])
    print("[setup] installing pytorch-tabnet transitive deps ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "scipy", "scikit-learn", "tqdm",
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
    train_path = matches[0]
    print(f"Found train.csv at {train_path}")
    return train_path.parent


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    """TabNet wants:
      - numerics standard-scaled (training stability under sparsemax),
      - categoricals label-encoded to integers + cat_idxs / cat_dims arrays.
    Unlike RealMLP, TabNet does NOT learn cat embeddings from string
    columns; we have to materialize integer codes ourselves.
    """
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    X_test = test.drop(columns=[ID_COL], errors="ignore").copy()

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"cat_cols={cat_cols}")
    print(f"num_cols ({len(num_cols)}): {num_cols[:8]}{' ...' if len(num_cols)>8 else ''}")

    # Label-encode each cat using train+test vocab so cat_dims is correct
    cat_dims = []
    cat_idxs = []
    feat_order = num_cols + cat_cols  # numerics first, cats last
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[c].astype(str), X_test[c].astype(str)])
        le.fit(combined.values)
        X[c] = le.transform(X[c].astype(str).values)
        X_test[c] = le.transform(X_test[c].astype(str).values)
        cat_dims.append(int(len(le.classes_)))
    # Index of each cat in feat_order
    for c in cat_cols:
        cat_idxs.append(feat_order.index(c))

    # NaN-fill numerics with median, then standard-scale
    for c in num_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med).astype(np.float32)
        X_test[c] = X_test[c].fillna(med).astype(np.float32)
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    X_arr = X[feat_order].values.astype(np.float32)
    X_test_arr = X_test[feat_order].values.astype(np.float32)
    print(f"X_arr={X_arr.shape}  cat_idxs={cat_idxs}  cat_dims={cat_dims}")
    return X_arr, X_test_arr, y, cat_idxs, cat_dims


def main():
    t0 = time.time()
    gpu_boot()
    install_tabnet()

    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    X_arr, X_test_arr, y, cat_idxs, cat_dims = prepare_features(train, test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[SMOKE_FOLD]
    print(f"=== SMOKE fold {SMOKE_FOLD} (train={len(tr)} val={len(va)}) ===")

    # TabNet defaults from the paper / pytorch-tabnet README, modest size:
    #   n_d / n_a = 32, n_steps = 5, gamma = 1.5, lambda_sparse = 1e-3.
    # cat_emb_dim auto-sized per cat_dim is fine here.
    model = TabNetClassifier(
        n_d=32, n_a=32,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-3,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=20, gamma=0.9),
        seed=SEED,
        verbose=10,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    t_fold = time.time()
    model.fit(
        X_arr[tr], y[tr],
        eval_set=[(X_arr[va], y[va])],
        eval_metric=["auc"],
        max_epochs=120,
        patience=15,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=2,
        drop_last=False,
    )
    p_va = model.predict_proba(X_arr[va])[:, 1]
    fold_auc = float(roc_auc_score(y[va], p_va))
    fold_wall = time.time() - t_fold
    delta_bp = (fold_auc - BASE_S) * 1e4
    delta_vs_realmlp_bp = (fold_auc - REALMLP_FOLD0_REF) * 1e4

    print(f"\nFold-0 AUC={fold_auc:.5f}  wall={fold_wall:.0f}s ({fold_wall/60:.1f}min)")
    print(f"  Δ baseline(BASE_S={BASE_S:.5f})={delta_bp:+.1f}bp")
    print(f"  Δ realmlp_e4_fold0({REALMLP_FOLD0_REF:.5f})={delta_vs_realmlp_bp:+.1f}bp")

    five_fold_proj_min = (fold_wall * N_FOLDS) / 60.0
    print(f"5-fold wall projection: {five_fold_proj_min:.1f}min "
          f"({'OK <60min' if five_fold_proj_min < 60 else 'WARN >=60min — Rule 2 shrink'})")

    res = dict(
        smoke_fold=SMOKE_FOLD,
        fold_auc=fold_auc,
        fold_wall_s=fold_wall,
        delta_vs_baseline_bp=delta_bp,
        delta_vs_realmlp_e4_fold0_bp=delta_vs_realmlp_bp,
        five_fold_projection_min=five_fold_proj_min,
        cv="StratifiedKFold(5)",
        metric="roc_auc",
        seed=SEED,
        n_folds=N_FOLDS,
        total_wall_s=time.time() - t0,
        notes=("TabNet 1-fold smoke. Compare fold-0 AUC vs RealMLP E4 "
               "(0.94722). Decision rule: if AUC < 0.945 OR 5-fold proj "
               ">=60min, shrink before promoting to 5-fold."),
        cat_idxs=cat_idxs, cat_dims=cat_dims,
    )
    (WORK / "tabnet_smoke_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
