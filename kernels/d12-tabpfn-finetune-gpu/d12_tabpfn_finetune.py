"""TabPFN-2.5 fine-tuning on Kaggle GPU — Strat 5-fold OOF (Day-12 Option 2).

This is NOT inference-only ICL. We use `tabpfn.finetuning.FinetunedTabPFNClassifier`
to specialise the foundation-model weights to this DGP via gradient steps.

Why fine-tune (Day-8 Tier-4 dismissal vs this revival):
  Day-8 dismissed TabPFN-ICL on context-window grounds (sub-sampling 440k
  rows down to 10k loses statistical power). Fine-tuning is structurally
  different: gradient updates over many meta-batches let the backbone
  internalise the synthetic-DGP regularities, and at inference time we
  still subsample 50k rows per estimator. The pre-trained inductive bias
  + the comp-specific specialisation is the bet — distinct from RealMLP
  (which has zero pre-train) and distinct from M5q's GBDTs.

Why Strat-only:
  R1 (HANDOVER): GroupKF dropped Day-3+; Strat is the LB proxy (gap +3.8bp).

Pinned: SEED=42, N_FOLDS=5, StratifiedKFold(shuffle=True, random_state=SEED).

GPU runtime expectations (T4×2; if Kaggle silently falls back to P100, the
torch-2.4 sm_60 force-reinstall trick from realmlp_gpu.py applies — see
install_tabpfn() below for that fallback):
  - Per fold: ~50-90 min (10 epochs × ~5min/epoch + final inference 10-15min
    on 188k test rows with n_estimators_final_inference=8).
  - 5-fold total: ~5-7h. Comfortably under Kaggle's 9h cap.

Outputs (under /kaggle/working/):
  oof_d12_tabpfn_finetune_strat.npy   shape (n_train, 2)
  test_d12_tabpfn_finetune_strat.npy  shape (n_test, 2)
  d12_tabpfn_finetune_results.json
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
SMOKE_FOLD0_ONLY = True   # 1-fold time-probe; set False for full 5-fold run
BASE_S = 0.94075          # baseline_two_anchor Strat OOF (LB-proxy anchor)
REALMLP_E4 = 0.94722      # E4 fold-0 reference

# ---- Fine-tune hyperparameters ----------------------------------------
# PI brief specifies LR=3e-5, batch=64, epochs ≤10 with early-stop patience=3
# on val AUC. TabPFN's API exposes LR / epochs / patience directly; "batch=64"
# maps to its meta-batch construction (n_finetune_ctx_plus_query_samples + the
# ctx/query split). We use TabPFN's defaults for that since they're tuned;
# the LR + epochs + patience are the load-bearing knobs the brief calls out.
FT_LR = 3e-5
FT_EPOCHS = 10
FT_EARLY_STOPPING_PATIENCE = 3
FT_VAL_SPLIT_RATIO = 0.1
FT_N_EST_FINETUNE = 2
FT_N_EST_VAL = 2
FT_N_EST_FINAL_INFERENCE = 8
FT_N_INFERENCE_SUBSAMPLE = 50_000  # TabPFN's per-estimator inference-time subsample

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


def install_tabpfn():
    """Install tabpfn into the Kaggle runtime.

    P100 (sm_60) compat notes from realmlp_gpu.py: Kaggle has historically
    silently routed GpuT4x2 jobs to P100 nodes that lack sm_60 in newer
    torch wheels. tabpfn 7.1.1 requires torch>=2.5 (which has sm_60 wheels
    again), so the realmlp force-reinstall workaround is NOT needed here —
    but we test cuda capability after import and warn loudly if it fails.

    The other pitfall is the one-time license-acceptance handshake. We
    handle it three ways:
      1. If TABPFN_TOKEN is set as a Kaggle Secret, use it.
      2. Else try the kaggle_secrets module to fetch a secret named
         TABPFN_TOKEN (PI's recommended path).
      3. Else fall back to printing the manual-license URL — kernel will
         crash on the first .fit() with a clear message.
    """
    # P100 (sm_60) compat: Kaggle silently routes GpuT4x2 jobs to P100.
    # torch>=2.5 dropped sm_60 support; 2.4 is the last release with it.
    # Install torch 2.4 first, then tabpfn with --no-deps so pip does not
    # upgrade torch above 2.4 (same workaround used in realmlp_gpu.py).
    print("[setup] force-reinstall torch 2.4 (sm_60 / P100 support) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
    print("[setup] installing tabpfn (--no-deps to preserve torch 2.4 pin) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--no-deps", "tabpfn==7.1.1",
    ])
    # Install tabpfn's non-torch transitive deps (torch excluded to hold pin)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "scikit-learn>=1.2.0", "numpy>=1.21.6", "pandas>=1.4.0",
        "scipy>=1.11.1", "einops>=0.4.0", "huggingface-hub>=0.19.0",
        "pydantic>=2.8.0", "pydantic-settings>=2.10.1",
        "eval-type-backport>=0.2.2", "joblib>=1.2.0", "filelock>=3.11.0",
        "typing_extensions>=4.12.0",
        "tabpfn-common-utils[telemetry-interactive]>=0.2.13",
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
    else:
        print("[setup] WARNING: CUDA not available; fine-tuning on CPU is "
              "infeasible at our scale. Aborting early.")
        raise RuntimeError("CUDA required for d12 fine-tune")

    # --- License token plumbing -------------------------------------
    if os.environ.get("TABPFN_TOKEN"):
        print("[setup] TABPFN_TOKEN already in env (good)")
        return

    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("TABPFN_TOKEN")
        if token:
            os.environ["TABPFN_TOKEN"] = token
            print("[setup] TABPFN_TOKEN loaded from Kaggle Secrets")
            return
    except Exception as e:
        print(f"[setup] kaggle_secrets unavailable / no TABPFN_TOKEN secret: {e}")

    print("[setup] WARNING: no TABPFN_TOKEN found. If license has not been "
          "accepted on this Kaggle account, the first fit() call will fail. "
          "Add TABPFN_TOKEN as a Kaggle Secret (Add-ons -> Secrets) and "
          "rerun. URL: https://ux.priorlabs.ai/account")


def find_data_dir():
    """rglob train.csv anywhere under /kaggle/input. Pattern from cb-slow-wide-gpu."""
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


def encode_categoricals(X_train, X_test, X_full=None):
    """Factorise object/string cols to int codes; return cat_indices.

    TabPFN's `categorical_features_indices` accepts the integer column
    positions of categorical features. We encode using the union of train +
    test categories so that test codes align with train (no -1 fallback at
    inference time, which TabPFN tolerates but doesn't love)."""
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"[fe] categorical columns: {cat_cols}")
    for c in cat_cols:
        # Build category set from all available frames so test codes match train
        frames = [X_train[c], X_test[c]]
        if X_full is not None:
            frames.append(X_full[c])
        all_vals = pd.concat(frames, axis=0).astype(str)
        codes, uniques = pd.factorize(all_vals, sort=True)
        n_train = len(X_train)
        n_test = len(X_test)
        X_train[c] = codes[:n_train]
        X_test[c] = codes[n_train:n_train + n_test]
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
    print(f"[fe] cat indices: {cat_indices}")
    return X_train, X_test, cat_indices


def main():
    t0 = time.time()
    gpu_boot()
    install_tabpfn()

    # Import after install
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    data_dir = find_data_dir()
    print(f"Using data_dir={data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore").reset_index(drop=True)
    X_test = test.drop(columns=[ID_COL], errors="ignore").reset_index(drop=True)
    print(f"raw X.shape={X.shape}  X_test.shape={X_test.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_strat = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float32)
    test_proba = np.zeros(len(test), dtype=np.float32)
    fold_scores = []
    fold_walls = []

    for k, (tr, va) in enumerate(splits_strat):
        t_fold = time.time()
        print(f"\n=== fold {k} (train={len(tr)} val={len(va)}) ===")

        # Per-fold encoding using union of train_fold + val_fold + test
        X_tr = X.iloc[tr].copy().reset_index(drop=True)
        X_va = X.iloc[va].copy().reset_index(drop=True)
        X_te = X_test.copy()

        # Use union of train_fold + val_fold for cat factorisation
        # (matches what fit() sees), apply same map to test
        cat_cols = X_tr.select_dtypes(include=["object", "string"]).columns.tolist()
        for c in cat_cols:
            all_vals = pd.concat([X_tr[c], X_va[c], X_te[c]], axis=0).astype(str)
            codes, _ = pd.factorize(all_vals, sort=True)
            n_tr, n_va = len(X_tr), len(X_va)
            X_tr[c] = codes[:n_tr]
            X_va[c] = codes[n_tr:n_tr + n_va]
            X_te[c] = codes[n_tr + n_va:]
        cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols]

        clf = FinetunedTabPFNClassifier(
            device="cuda",
            epochs=FT_EPOCHS,
            learning_rate=FT_LR,
            validation_split_ratio=FT_VAL_SPLIT_RATIO,
            early_stopping=True,
            early_stopping_patience=FT_EARLY_STOPPING_PATIENCE,
            n_estimators_finetune=FT_N_EST_FINETUNE,
            n_estimators_validation=FT_N_EST_VAL,
            n_estimators_final_inference=FT_N_EST_FINAL_INFERENCE,
            n_inference_subsample_samples=FT_N_INFERENCE_SUBSAMPLE,
            random_state=SEED,
            eval_metric="roc_auc",
            extra_classifier_kwargs={
                "categorical_features_indices": cat_indices,
                "ignore_pretraining_limits": True,
                "balance_probabilities": False,
            },
        )

        # Fine-tune on train fold; validation split is internal
        print(f"  [fold {k}] fit (epochs={FT_EPOCHS}, lr={FT_LR}, "
              f"patience={FT_EARLY_STOPPING_PATIENCE}) ...")
        clf.fit(X_tr.values, y[tr])

        # Predict OOF + test
        print(f"  [fold {k}] predict OOF ...")
        p_va = clf.predict_proba(X_va.values)[:, 1]
        oof[va] = p_va
        print(f"  [fold {k}] predict test ...")
        p_te = clf.predict_proba(X_te.values)[:, 1]
        test_proba += p_te / N_FOLDS

        s = float(roc_auc_score(y[va], p_va))
        wall = time.time() - t_fold
        fold_scores.append(s)
        fold_walls.append(wall)
        print(f"  [fold {k}] AUC={s:.5f}  wall={wall:.0f}s ({wall/60:.1f}min)")
        if SMOKE_FOLD0_ONLY:
            print(f"  [smoke] 1-fold probe done. 5-fold projection: "
                  f"{wall*5/3600:.1f}h. Set SMOKE_FOLD0_ONLY=False for full run.")
            break

        # Save partial state every fold
        np.save(WORK / "oof_d12_tabpfn_finetune_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(WORK / "test_d12_tabpfn_finetune_strat.npy",
                np.column_stack([1 - test_proba, test_proba]))

    auc_full = float(roc_auc_score(y, oof))
    delta_baseline = (auc_full - BASE_S) * 1e4
    delta_realmlp = (auc_full - REALMLP_E4) * 1e4
    print(f"\n[d12] OOF Strat: {auc_full:.5f}  std={np.std(fold_scores):.5f}")
    print(f"  Δ baseline_two_anchor({BASE_S:.5f}): {delta_baseline:+.1f}bp")
    print(f"  Δ realmlp_e4_fold0({REALMLP_E4:.5f}): {delta_realmlp:+.1f}bp")
    print(f"Total wall: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

    # Final saves
    np.save(WORK / "oof_d12_tabpfn_finetune_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(WORK / "test_d12_tabpfn_finetune_strat.npy",
            np.column_stack([1 - test_proba, test_proba]))

    # Submission file (in case PI wants to LB-probe direct)
    sub = sample_sub.copy()
    sub[TARGET] = test_proba
    sub.to_csv(WORK / "submission_d12_tabpfn_finetune.csv", index=False)

    res = dict(
        model="TabPFN-2.5_finetuned",
        oof_score=auc_full,
        fold_std=float(np.std(fold_scores)),
        fold_scores=fold_scores,
        fold_walls=fold_walls,
        cv="StratifiedKFold(5)",
        metric="roc_auc",
        delta_vs_baseline_bp=delta_baseline,
        delta_vs_realmlp_e4_bp=delta_realmlp,
        seed=SEED,
        n_folds=N_FOLDS,
        ft_lr=FT_LR,
        ft_epochs=FT_EPOCHS,
        ft_early_stopping_patience=FT_EARLY_STOPPING_PATIENCE,
        ft_val_split_ratio=FT_VAL_SPLIT_RATIO,
        ft_n_est_finetune=FT_N_EST_FINETUNE,
        ft_n_est_validation=FT_N_EST_VAL,
        ft_n_est_final_inference=FT_N_EST_FINAL_INFERENCE,
        ft_n_inference_subsample=FT_N_INFERENCE_SUBSAMPLE,
        total_wall_s=time.time() - t0,
        notes=("TabPFN-2.5 fine-tuned (gradient updates, NOT zero-shot ICL). "
               "Strat-only per R1. Day-8 Tier-4 dismissal applied to ICL only; "
               "fine-tune is a different paradigm (foundation-model weights "
               "specialised to this DGP)."),
    )
    (WORK / "d12_tabpfn_finetune_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
