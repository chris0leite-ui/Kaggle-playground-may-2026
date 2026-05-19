"""R14 Phase 2 — TabM full 5-fold on Kaggle T4×2.

NN-class diversity for the K=16 stack. Brainstorm S2 predicted +0.3
to +0.8 bp standalone (pytabkit Tabular Mixture, Yandex 2024). The
prior smoke (fold-0) hit 0.94039 — below K=13 base AUCs but
structurally novel. K=16 add via Path-B DCS τ=100k.

Fold-safety: 5-fold StratifiedKFold(seed=42) same as the K=13 pool.
Each fold trains TabM on ti rows; predict on val + test rows. Test
predictions averaged across folds.

Outputs (to /kaggle/working):
- oof_R14_tabm_strat.npy  (439140,) probabilities in [0, 1]
- test_R14_tabm_strat.npy (188165,) probabilities in [0, 1]
- r14_tabm_results.json — per-fold AUC, walls, config used
"""
from __future__ import annotations

import inspect
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
TARGET_N_EPOCHS = 200
TARGET_PATIENCE = 50
TARGET_LR = 3e-4

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def gpu_boot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=10).strip()
        print(f"[boot] GPU info: {out}", flush=True)
    except Exception as e:
        print(f"[boot] nvidia-smi failed: {e}", flush=True)


def install_tabm():
    print("[setup] force-reinstall torch 2.4 (sm_60 P100 support) ...",
          flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
    print("[setup] installing pytabkit (--no-deps) ...", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
        "pytabkit",
    ])
    print("[setup] installing pytabkit transitive deps ...", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "pytorch-lightning>=2.0,<2.5", "skorch", "torchmetrics<1.5",
    ])
    import importlib
    if "torch" in sys.modules:
        importlib.reload(sys.modules["torch"])
    import torch
    print(f"[setup] torch version: {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"[setup] CUDA: {torch.version.cuda}, "
              f"device 0: {torch.cuda.get_device_name(0)}, "
              f"capability: {torch.cuda.get_device_capability(0)}",
              flush=True)


def resolve_tabm_class():
    import pytabkit
    for name in ("TabM_D_Classifier", "TabM_Classifier",
                 "TabM_HPO_Classifier"):
        cls = getattr(pytabkit, name, None)
        if cls is not None:
            print(f"[setup] using pytabkit.{name}", flush=True)
            return cls
    raise RuntimeError(f"no TabM class in pytabkit. dir: "
                       f"{[n for n in dir(pytabkit) if 'TabM' in n]}")


def first_supported_kwarg(allowed: set, *candidates: str) -> str | None:
    for c in candidates:
        if c in allowed:
            return c
    return None


def build_kwargs(cls) -> dict:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
    proposed = dict(
        device="cuda",
        random_state=SEED,
        n_cv=1,
        val_metric_name="cross_entropy",
        use_ls=False,
        verbosity=1,
    )
    e = first_supported_kwarg(allowed, "n_epochs", "max_epochs", "epochs")
    if e:
        proposed[e] = TARGET_N_EPOCHS
    elif has_var_kw:
        proposed["n_epochs"] = TARGET_N_EPOCHS
    p = first_supported_kwarg(allowed, "patience",
                              "early_stopping_patience", "es_patience")
    if p:
        proposed[p] = TARGET_PATIENCE
    elif has_var_kw:
        proposed["patience"] = TARGET_PATIENCE
    lr = first_supported_kwarg(allowed, "lr", "learning_rate")
    if lr:
        proposed[lr] = TARGET_LR
    elif has_var_kw:
        proposed["lr"] = TARGET_LR
    print(f"[knobs] applied kwargs: {proposed}", flush=True)
    if has_var_kw:
        return proposed
    return {k: v for k, v in proposed.items() if k in allowed}


def find_data_dir():
    base = Path("/kaggle/input")
    if not base.exists():
        raise RuntimeError(f"/kaggle/input missing; ls /kaggle: "
                           f"{os.listdir('/kaggle')}")
    matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv under {base}")
    return matches[0].parent


def main() -> None:
    t0 = time.time()
    gpu_boot()
    install_tabm()
    TabMCls = resolve_tabm_class()

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"loaded train={train.shape} test={test.shape}  "
          f"t={time.time()-t0:.1f}s", flush=True)

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"cat cols: {cat_cols}", flush=True)
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs = []
    walls = []

    for k, (tr, va) in enumerate(fold_list, 1):
        t_f = time.time()
        print(f"\n=== Fold {k}/{N_FOLDS} train={len(tr)} val={len(va)} ===",
              flush=True)
        kwargs = build_kwargs(TabMCls)
        model = TabMCls(**kwargs)
        model.fit(X.iloc[tr], y[tr])
        p_va = model.predict_proba(X.iloc[va])[:, 1]
        p_te = model.predict_proba(X_test)[:, 1]
        oof[va] = p_va
        test_pred += p_te / N_FOLDS
        auc = float(roc_auc_score(y[va], p_va))
        wall = time.time() - t_f
        fold_aucs.append(auc)
        walls.append(wall)
        print(f"  Fold {k}: AUC={auc:.5f}  wall={wall:.0f}s "
              f"({wall/60:.1f}min)", flush=True)

    auc_full = float(roc_auc_score(y, oof))
    print(f"\n=== Full OOF AUC: {auc_full:.5f}  total={time.time()-t0:.0f}s ===",
          flush=True)

    np.save(WORK / "oof_R14_tabm_strat.npy", oof.astype(np.float32))
    np.save(WORK / "test_R14_tabm_strat.npy", test_pred.astype(np.float32))
    res = dict(
        round="R14_Phase2_tabm",
        oof_auc_full=auc_full,
        fold_aucs=fold_aucs,
        walls=walls,
        target_n_epochs=TARGET_N_EPOCHS,
        target_patience=TARGET_PATIENCE,
        target_lr=TARGET_LR,
        seed=SEED,
        n_folds=N_FOLDS,
        total_wall_s=time.time() - t0,
    )
    (WORK / "r14_tabm_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
