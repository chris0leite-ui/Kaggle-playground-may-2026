"""RealMLP-TD on Kaggle GPU — Strat 5-fold OOF (HANDOVER Step 3).

Genuinely orthogonal mechanism family (NN ≠ all 13/15 GBDTs in M5h pool).
yekenot's 56-vote public notebook for this exact comp uses RealMLP.

E4 (CPU fold-0) gave AUC=0.94722 (+64.7bp baseline) in 39.5 min →
strong base, just unaffordable on CPU at 5-fold (3.3h projection).
GPU port should fit comfortably in Kaggle's 9h cap.

R1: GroupKF dropped Day-3+ (test is i.i.d. row split per U3 probe).
Strat-only.

Pinned: SEED=42, N_FOLDS=5, StratifiedKFold(shuffle=True, random_state=SEED).

Outputs (under /kaggle/working/):
  oof_realmlp_strat.npy  shape (n_train, 2)
  test_realmlp_strat.npy shape (n_test, 2)
  realmlp_results.json
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
SEED, N_FOLDS = 42, 5
BASE_S = 0.94075   # baseline_two_anchor Strat OOF (LB-proxy anchor)

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
    """Install pytabkit (RealMLP) into the kernel runtime.

    Kaggle's GPU base image does not include pytabkit. Internet is enabled
    in kernel-metadata.json. Pin a recent stable version; pytabkit ships
    RealMLP_TD as published by Holzmüller et al. (2024).
    """
    print("[setup] installing pytabkit ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location",
        "pytabkit",
    ])
    print("[setup] pytabkit installed.")


def find_data_dir():
    """rglob train.csv anywhere under /kaggle/input.

    Comp data path varies. Pattern from `cb-slow-wide-gpu`.
    """
    base = Path("/kaggle/input")
    if not base.exists():
        raise RuntimeError(f"/kaggle/input missing; ls /kaggle: {os.listdir('/kaggle')}")
    print("DEBUG ls /kaggle/input recursive (first 30):")
    for i, p in enumerate(sorted(base.rglob("*"))):
        if i >= 30:
            print("  ...")
            break
        print(f"  {p}")
    matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv anywhere under {base}")
    train_path = matches[0]
    print(f"Found train.csv at {train_path}")
    return train_path.parent


def main():
    t0 = time.time()
    gpu_boot()
    install_pytabkit()

    # Import after install
    from pytabkit import RealMLP_TD_Classifier

    data_dir = find_data_dir()
    print(f"Using data_dir={data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")

    # RealMLP handles native categoricals via embeddings; pass as object string
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"cat cols: {cat_cols}")
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_strat = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float32)
    test_proba = np.zeros(len(test), dtype=np.float32)
    fold_scores = []
    fold_walls = []
    for k, (tr, va) in enumerate(splits_strat):
        t_fold = time.time()
        print(f"=== fold {k} (train={len(tr)} val={len(va)}) ===")
        model = RealMLP_TD_Classifier(
            device="cuda",
            random_state=SEED,
            n_cv=1,                          # internal val from train portion
            val_metric_name="cross_entropy",
            use_ls=False,                    # AUC-friendly (no label smoothing)
            verbosity=1,
        )
        model.fit(X.iloc[tr], y[tr])
        p_va = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p_va
        test_proba += model.predict_proba(X_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        wall = time.time() - t_fold
        fold_scores.append(s)
        fold_walls.append(wall)
        print(f"  fold {k}: AUC={s:.5f}  wall={wall:.0f}s ({wall/60:.1f}min)")

        # Save partial state after every fold so we never lose work
        np.save(WORK / f"oof_realmlp_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(WORK / f"test_realmlp_strat.npy",
                np.column_stack([1 - test_proba, test_proba]))

    auc_full = float(roc_auc_score(y, oof))
    delta_bp = (auc_full - BASE_S) * 1e4
    print(f"\nOOF Strat: {auc_full:.5f}  std={np.std(fold_scores):.5f}  "
          f"Δ baseline={delta_bp:+.1f}bp")
    print(f"Total wall: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

    # Final save (in case last partial save was for fold N-1)
    np.save(WORK / "oof_realmlp_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(WORK / "test_realmlp_strat.npy",
            np.column_stack([1 - test_proba, test_proba]))

    # Submission
    sub = sample_sub.copy()
    sub[TARGET] = test_proba
    sub.to_csv(WORK / "submission_realmlp.csv", index=False)

    res = dict(
        oof_score=auc_full,
        fold_std=float(np.std(fold_scores)),
        fold_scores=fold_scores,
        fold_walls=fold_walls,
        cv="StratifiedKFold(5)",
        metric="roc_auc",
        delta_vs_baseline_bp=delta_bp,
        seed=SEED,
        n_folds=N_FOLDS,
        total_wall_s=time.time() - t0,
        e4_cpu_fold0=0.94722,
        notes="RealMLP-TD GPU Strat-only (R1 dropped GroupKF Day-3+).",
    )
    (WORK / "realmlp_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
