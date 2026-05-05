"""TabM 1-fold SMOKE v3 — extended training (HANDOVER Day-10 Path A).

v2 result (d9_tabm_smoke_results.json): fold-0 AUC 0.94039, FAIL gate
0.945 by 46bp; vs RealMLP fold-0 ref 0.94722 by 68bp. Best val
cross-entropy at epoch 5 then 20 epochs of oscillation. HANDOVER Day-10
hypothesis: under-trained at the margin or stuck in a local basin from
default early-stopping (patience too small).

Path A intervention:
  - introspect pytabkit.TabM_D_Classifier.__init__ for n_epochs /
    patience / lr knobs and pass max-epochs >= 200, patience >= 50,
    smaller lr where the API exposes it
  - if the pytabkit API does NOT expose those knobs, fall back to the
    standalone `tabm` PyPI package which has the simpler `(n_epochs,
    patience, lr)` API directly
  - dump the resolved class' full __init__ signature to stdout so the
    Kaggle log shows exactly which knobs were applied (debuggable)

Same gate as v2: fold-0 AUC >= 0.945 PROMOTE -> 5-fold + 3-seed bag;
else HOLD and pivot per HANDOVER (G4 SCARF / F2 multi-rule rebuild).
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
SMOKE_FOLD = 0
BASE_S = 0.94075
REALMLP_FOLD0_REF = 0.94722
SMOKE_GATE_AUC = 0.945
V2_FOLD0_AUC = 0.94039  # for printed delta diagnostic

# Path A: extended-training targets. Apply via whichever knob name the
# resolved class exposes. Defensive over pytabkit / tabm naming drift.
TARGET_N_EPOCHS = 200
TARGET_PATIENCE = 50
TARGET_LR = 3e-4  # lower than typical defaults (1e-3) — counters local-basin

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

    P100 (sm_60) compat lesson from realmlp_gpu.py + v2: Kaggle silently
    routes GpuT4x2 jobs to P100 sometimes; recent torch wheels drop sm_60.
    Force-reinstall torch 2.4 first, then pytabkit --no-deps to preserve
    the pin. This block is unchanged from v2.
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


def install_standalone_tabm():
    print("[setup] installing standalone `tabm` package ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "tabm",
    ])


def dump_init_signature(cls, label: str):
    """Print every parameter the resolved classifier accepts so the
    Kaggle log shows what knobs were available. Path A specifically
    needs to know whether n_epochs / patience / lr are exposed."""
    try:
        sig = inspect.signature(cls.__init__)
        print(f"[introspect] {label} {cls.__name__}.__init__ params:")
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            default = p.default if p.default is not inspect.Parameter.empty else "<required>"
            print(f"             {name} = {default!r}")
    except Exception as e:
        print(f"[introspect] failed to inspect {cls}: {e}")


def resolve_tabm_class():
    """Return (cls, source) where source in {'pytabkit', 'tabm'}.
    Prefers pytabkit (proven to load on Kaggle in v2)."""
    try:
        import pytabkit
    except Exception as e:
        print(f"[setup] pytabkit import failed: {e}; going straight to tabm")
        install_standalone_tabm()
        import tabm
        for name in ("TabMClassifier", "TabM_Classifier", "Classifier"):
            cls = getattr(tabm, name, None)
            if cls is not None:
                print(f"[setup] using tabm.{name}")
                return cls, "tabm"
        raise RuntimeError(f"no TabM class in tabm. dir: {dir(tabm)[:30]}")

    for name in ("TabM_D_Classifier", "TabM_Classifier", "TabM_HPO_Classifier"):
        cls = getattr(pytabkit, name, None)
        if cls is not None:
            print(f"[setup] using pytabkit.{name}")
            return cls, "pytabkit"

    print("[setup] pytabkit lacks TabM; installing standalone `tabm` ...")
    install_standalone_tabm()
    import tabm
    for name in ("TabMClassifier", "TabM_Classifier", "Classifier"):
        cls = getattr(tabm, name, None)
        if cls is not None:
            print(f"[setup] using tabm.{name}")
            return cls, "tabm"
    raise RuntimeError(
        f"no TabM class found. pytabkit dir: {dir(pytabkit)[:30]}; "
        f"tabm dir: {dir(tabm)[:30]}"
    )


def first_supported_kwarg(allowed: set, *candidates: str) -> str | None:
    """Return the first candidate name that's in `allowed`, else None.
    Naming drifts across pytabkit versions (n_epochs vs max_epochs vs
    epochs) — try a small ordered list."""
    for c in candidates:
        if c in allowed:
            return c
    return None


def build_extended_kwargs(cls, source: str) -> dict:
    """Pick proposed kwargs for extended training, mapped to whichever
    names the resolved class actually accepts. Logs the mapping it
    chose so the Kaggle log shows the active knobs."""
    sig = inspect.signature(cls.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
    allowed = set(sig.parameters.keys())

    proposed: dict = {}
    if source == "pytabkit":
        proposed.update(dict(
            device="cuda",
            random_state=SEED,
            n_cv=1,
            val_metric_name="cross_entropy",
            use_ls=False,
            verbosity=1,
        ))

    # Path A targets — applied via whichever knob name exists.
    # Source: pytabkit RealMLP_TD_Classifier signature uses `n_epochs`
    # and `patience`; tabm uses similar; fall through if neither exposed.
    epoch_knob = first_supported_kwarg(
        allowed, "n_epochs", "max_epochs", "epochs"
    )
    if epoch_knob:
        proposed[epoch_knob] = TARGET_N_EPOCHS
    elif has_var_kw:
        proposed["n_epochs"] = TARGET_N_EPOCHS  # let **kwargs swallow it

    patience_knob = first_supported_kwarg(
        allowed, "patience", "early_stopping_patience", "es_patience"
    )
    if patience_knob:
        proposed[patience_knob] = TARGET_PATIENCE
    elif has_var_kw:
        proposed["patience"] = TARGET_PATIENCE

    lr_knob = first_supported_kwarg(
        allowed, "lr", "learning_rate"
    )
    if lr_knob:
        proposed[lr_knob] = TARGET_LR
    elif has_var_kw:
        proposed["lr"] = TARGET_LR

    print(f"[knobs] epoch_knob={epoch_knob} patience_knob={patience_knob} "
          f"lr_knob={lr_knob}")
    print(f"[knobs] proposed kwargs: {proposed}")

    if has_var_kw:
        return proposed
    keep = {k: v for k, v in proposed.items() if k in allowed}
    dropped = sorted(set(proposed) - set(keep))
    if dropped:
        print(f"[knobs] dropped unsupported: {dropped}")

    # Final guard: if NEITHER pytabkit accepted epoch/patience knobs nor
    # we have **kwargs, the extended-training intervention is a no-op.
    # That is itself signal — fail loud so the result reflects "v3 had
    # no lever". Caller decides whether to escalate to standalone tabm.
    if epoch_knob is None and patience_knob is None and not has_var_kw:
        print("[knobs] WARNING: pytabkit class exposes neither epoch nor "
              "patience knob; v3 reduces to v2 config")
    return keep


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


def main():
    t0 = time.time()
    gpu_boot()
    install_tabm()
    TabMCls, source = resolve_tabm_class()
    dump_init_signature(TabMCls, "primary")

    # If pytabkit doesn't expose epoch / patience knobs at all, escalate
    # to the standalone `tabm` package (HANDOVER Path A fallback).
    sig = inspect.signature(TabMCls.__init__)
    allowed = set(sig.parameters.keys())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
    epoch_present = first_supported_kwarg(
        allowed, "n_epochs", "max_epochs", "epochs"
    ) is not None
    if source == "pytabkit" and not epoch_present and not has_var_kw:
        print("[setup] pytabkit lacks epoch knob; escalating to standalone tabm")
        install_standalone_tabm()
        import tabm
        for name in ("TabMClassifier", "TabM_Classifier", "Classifier"):
            cls = getattr(tabm, name, None)
            if cls is not None:
                TabMCls, source = cls, "tabm"
                print(f"[setup] using tabm.{name}")
                dump_init_signature(TabMCls, "fallback")
                break

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"cat cols: {cat_cols}")
    for c in cat_cols:
        X[c] = X[c].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[SMOKE_FOLD]
    print(f"=== SMOKE fold {SMOKE_FOLD} (train={len(tr)} val={len(va)}) ===")

    t_fold = time.time()
    kwargs = build_extended_kwargs(TabMCls, source)
    model = TabMCls(**kwargs)
    model.fit(X.iloc[tr], y[tr])
    p_va = model.predict_proba(X.iloc[va])[:, 1]
    fold_auc = float(roc_auc_score(y[va], p_va))
    fold_wall = time.time() - t_fold

    delta_bp_baseline = (fold_auc - BASE_S) * 1e4
    delta_bp_realmlp = (fold_auc - REALMLP_FOLD0_REF) * 1e4
    delta_bp_v2 = (fold_auc - V2_FOLD0_AUC) * 1e4
    five_fold_proj_min = (fold_wall * N_FOLDS) / 60.0

    auc_gate = "PASS" if fold_auc >= SMOKE_GATE_AUC else "FAIL"
    wall_gate = "PASS" if five_fold_proj_min < 60 else "FAIL"
    overall = "PROMOTE" if (auc_gate == "PASS" and wall_gate == "PASS") else "HOLD"

    print(f"\nFold-0 AUC={fold_auc:.5f}  wall={fold_wall:.0f}s ({fold_wall/60:.1f}min)")
    print(f"  Δ baseline(BASE_S={BASE_S:.5f})={delta_bp_baseline:+.1f}bp")
    print(f"  Δ realmlp_e4_fold0({REALMLP_FOLD0_REF:.5f})={delta_bp_realmlp:+.1f}bp")
    print(f"  Δ v2_fold0({V2_FOLD0_AUC:.5f})={delta_bp_v2:+.1f}bp  (Path A lift)")
    print(f"  smoke_gate AUC>={SMOKE_GATE_AUC}: {auc_gate}")
    print(f"5-fold wall projection: {five_fold_proj_min:.1f}min  gate<60min: {wall_gate}")
    print(f"OVERALL: {overall}")

    res = dict(
        version="v3_extended_training",
        smoke_fold=SMOKE_FOLD,
        fold_auc=fold_auc,
        fold_wall_s=fold_wall,
        delta_vs_baseline_bp=delta_bp_baseline,
        delta_vs_realmlp_e4_fold0_bp=delta_bp_realmlp,
        delta_vs_v2_fold0_bp=delta_bp_v2,
        five_fold_projection_min=five_fold_proj_min,
        smoke_gate_auc=SMOKE_GATE_AUC,
        auc_gate=auc_gate,
        wall_gate=wall_gate,
        overall=overall,
        cv="StratifiedKFold(5)",
        metric="roc_auc",
        seed=SEED,
        n_folds=N_FOLDS,
        target_n_epochs=TARGET_N_EPOCHS,
        target_patience=TARGET_PATIENCE,
        target_lr=TARGET_LR,
        applied_kwargs=kwargs,
        source=source,
        total_wall_s=time.time() - t0,
        notes=("TabM v3 — Path A extended training (HANDOVER Day-10). "
               "PROMOTE -> 5-fold + 3-seed bag; HOLD -> pivot to G4 SCARF "
               "or F2 multi-rule rebuild."),
    )
    (WORK / "tabm_smoke_v3_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
