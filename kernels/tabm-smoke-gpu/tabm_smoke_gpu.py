"""TabM 1-fold SMOKE on Kaggle T4 — Strat fold 0 only (HANDOVER Day-9 Step 1).

Rule 2: 1-fold smoke before any GPU 5-fold. RealMLP fold-0 reference is
0.94722 (E4 CPU). The smoke gate per HANDOVER:
  - fold-0 val AUC >= 0.945 → schedule 5-fold + 3-seed bag overnight
  - 5-fold projection < 60 min → safe under 1h GPU cap
  - either failing → don't proceed; pivot to T1.4 hazard-NN

Why TabM (vs RealMLP, our existing NN base):
  RealMLP-TD: MLP + numerical/categorical embeddings (single base in M5q
    pool). Day-7 partial-bag salvage was NULL (Tier-3 variance-reduction
    cap). Sole remaining unfalsified Tier-1 GPU candidate per Day-8
    falsifications.
  TabM:       parameter-efficient ensemble (BatchEnsemble) of K=32 internal
    sub-models sharing most weights. Different optimization landscape
    (averaged outputs trained jointly), distinct from RealMLP-TD's single-
    pass MLP. Gomes/Gorishniy 2024 result: median AUC lift over MLP/FT-
    Transformer on tabular benchmarks. Mechanism class NOT in M5q pool.

Pinned: SEED=42, N_FOLDS=5, StratifiedKFold(shuffle=True, random_state=SEED).
Runs fold 0 only (k=0) for the smoke. NO submission file written.

Outputs (under /kaggle/working/):
  tabm_smoke_results.json -- AUC fold 0, wall, 5-fold projection
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
SMOKE_FOLD = 0
BASE_S = 0.94075           # baseline_two_anchor Strat OOF
REALMLP_FOLD0_REF = 0.94722  # E4 fold-0 reference for sanity comparison
SMOKE_GATE_AUC = 0.945     # HANDOVER: minimum to promote to 5-fold

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


def install_tabm():
    """Install pytabkit (which bundles TabM_D_Classifier) into the runtime.

    P100 (sm_60) compat lesson from realmlp_gpu.py: Kaggle silently routes
    GpuT4x2 jobs to P100 sometimes; recent torch wheels drop sm_60. Force-
    reinstall torch 2.4 first, then pytabkit --no-deps to preserve the pin.
    """
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


def resolve_tabm_class():
    """Return a TabM classifier class. Tries pytabkit naming variants;
    falls back to standalone `tabm` PyPI package if pytabkit lacks it."""
    import pytabkit
    for name in ("TabM_D_Classifier", "TabM_Classifier",
                 "TabM_HPO_Classifier"):
        cls = getattr(pytabkit, name, None)
        if cls is not None:
            print(f"[setup] using pytabkit.{name}")
            return cls
    print("[setup] pytabkit lacks TabM; installing standalone `tabm` ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "tabm",
    ])
    import tabm
    for name in ("TabMClassifier", "TabM_Classifier", "Classifier"):
        cls = getattr(tabm, name, None)
        if cls is not None:
            print(f"[setup] using tabm.{name}")
            return cls
    raise RuntimeError(
        f"no TabM class found. pytabkit dir: {dir(pytabkit)[:30]}; "
        f"tabm dir: {dir(tabm)[:30]}"
    )


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


def safe_kwargs(cls, **proposed):
    """Filter kwargs to only those the class' __init__ accepts.

    pytabkit classes do NOT share a constructor signature: TabM_D_Classifier
    rejects RealMLP-specific args like `use_ls`. Use signature introspection
    so we don't crash on benign tuning hints. Falls back to **kwargs accept
    when signature is variadic.
    """
    import inspect
    sig = inspect.signature(cls.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
    if has_var_kw:
        return proposed
    allowed = set(sig.parameters.keys())
    keep = {k: v for k, v in proposed.items() if k in allowed}
    dropped = sorted(set(proposed) - set(keep))
    if dropped:
        print(f"[setup] dropped unsupported kwargs for {cls.__name__}: {dropped}")
    return keep


def main():
    t0 = time.time()
    gpu_boot()
    install_tabm()
    TabMCls = resolve_tabm_class()

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    # Smoke does not predict test; skip the test prep entirely.
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"cat cols: {cat_cols}")
    for c in cat_cols:
        X[c] = X[c].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[SMOKE_FOLD]
    print(f"=== SMOKE fold {SMOKE_FOLD} (train={len(tr)} val={len(va)}) ===")

    t_fold = time.time()
    # TabM_D_* uses tuned defaults from Gomes/Gorishniy 2024 (k=32 internal
    # heads, BatchEnsemble). Constructor args differ from RealMLP_TD; use
    # signature filtering to avoid TypeError on RealMLP-only kwargs.
    proposed = dict(
        device="cuda",
        random_state=SEED,
        n_cv=1,                         # outer 5-fold already gives val
        val_metric_name="cross_entropy",
        use_ls=False,                   # AUC-friendly (RealMLP-only; filtered)
        verbosity=1,
    )
    model = TabMCls(**safe_kwargs(TabMCls, **proposed))
    model.fit(X.iloc[tr], y[tr])
    p_va = model.predict_proba(X.iloc[va])[:, 1]
    fold_auc = float(roc_auc_score(y[va], p_va))
    fold_wall = time.time() - t_fold

    delta_bp_baseline = (fold_auc - BASE_S) * 1e4
    delta_bp_realmlp = (fold_auc - REALMLP_FOLD0_REF) * 1e4
    five_fold_proj_min = (fold_wall * N_FOLDS) / 60.0

    auc_gate = "PASS" if fold_auc >= SMOKE_GATE_AUC else "FAIL"
    wall_gate = "PASS" if five_fold_proj_min < 60 else "FAIL"
    overall = "PROMOTE" if (auc_gate == "PASS" and wall_gate == "PASS") else "HOLD"

    print(f"\nFold-0 AUC={fold_auc:.5f}  wall={fold_wall:.0f}s ({fold_wall/60:.1f}min)")
    print(f"  Δ baseline(BASE_S={BASE_S:.5f})={delta_bp_baseline:+.1f}bp")
    print(f"  Δ realmlp_e4_fold0({REALMLP_FOLD0_REF:.5f})={delta_bp_realmlp:+.1f}bp")
    print(f"  smoke_gate AUC>={SMOKE_GATE_AUC}: {auc_gate}")
    print(f"5-fold wall projection: {five_fold_proj_min:.1f}min  gate<60min: {wall_gate}")
    print(f"OVERALL: {overall}")

    res = dict(
        smoke_fold=SMOKE_FOLD,
        fold_auc=fold_auc,
        fold_wall_s=fold_wall,
        delta_vs_baseline_bp=delta_bp_baseline,
        delta_vs_realmlp_e4_fold0_bp=delta_bp_realmlp,
        five_fold_projection_min=five_fold_proj_min,
        smoke_gate_auc=SMOKE_GATE_AUC,
        auc_gate=auc_gate,
        wall_gate=wall_gate,
        overall=overall,
        cv="StratifiedKFold(5)",
        metric="roc_auc",
        seed=SEED,
        n_folds=N_FOLDS,
        total_wall_s=time.time() - t0,
        notes=("TabM 1-fold smoke. Sole unfalsified Tier-1 (HANDOVER Day-9). "
               "PROMOTE -> 5-fold + 3-seed bag; HOLD -> pivot to T1.4 "
               "hazard-NN."),
    )
    (WORK / "tabm_smoke_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
