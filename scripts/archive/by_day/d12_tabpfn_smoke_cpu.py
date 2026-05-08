"""d12 — TabPFN-2.5 zero-shot CPU smoke (subsampled).

Goal: sanity-check the TabPFN package + measure zero-shot AUC on a 10k-train
/ 5k-test subsample of the comp data, before committing to a Kaggle GPU
fine-tune kernel run.

Why subsample: TabPFN-2.5 is built for ≤50k training rows; CPU-only inference
is slow (the docs explicitly say CPU is feasible only at <1k samples).
We accept that the smoke is a *coarse* signal — the goal is package /
preprocessing / API confirmation, not a representative AUC.

Outputs:
  scripts/artifacts/d12_tabpfn_smoke_cpu_results.json
  scripts/artifacts/oof_d12_tabpfn_smoke10k.npy   shape (5000, 2)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "scripts" / "artifacts"
ART.mkdir(exist_ok=True)

TARGET, ID_COL = "PitNextLap", "id"
SEED = 42
N_TRAIN_SUBSAMPLE = 10_000
N_TEST_SUBSAMPLE = 5_000
TIME_LIMIT_S = 60 * 25   # 25 min cap; Rule 2 says 30 min CPU smoke cap


def main():
    t0 = time.time()
    print("[d12] loading train.csv ...")
    train = pd.read_csv(DATA / "train.csv")
    print(f"[d12] full train shape: {train.shape}")

    y_full = train[TARGET].astype(int).values
    X_full = train.drop(columns=[TARGET, ID_COL], errors="ignore")

    # Stratified subsample (preserve class prior)
    idx_pool = np.arange(len(train))
    idx_sub, _ = train_test_split(
        idx_pool,
        train_size=N_TRAIN_SUBSAMPLE + N_TEST_SUBSAMPLE,
        random_state=SEED,
        stratify=y_full,
    )
    X_sub = X_full.iloc[idx_sub].reset_index(drop=True)
    y_sub = y_full[idx_sub]

    # Split into train / test inside the subsample
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y_sub,
        train_size=N_TRAIN_SUBSAMPLE,
        random_state=SEED,
        stratify=y_sub,
    )
    print(f"[d12] sub train: {X_tr.shape}  sub test: {X_te.shape}")
    print(f"[d12] class prior train: {y_tr.mean():.4f}  test: {y_te.mean():.4f}")

    # TabPFN expects numeric/string features; categoricals via indices.
    cat_cols = X_tr.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"[d12] cat cols: {cat_cols}")
    # Encode categoricals as integer codes (TabPFN handles cats given indices)
    for c in cat_cols:
        codes_tr, uniques = pd.factorize(X_tr[c], sort=True)
        # Map test using same uniques; unseen -> -1 (TabPFN tolerates)
        u_idx = {v: i for i, v in enumerate(uniques)}
        codes_te = X_te[c].map(u_idx).fillna(-1).astype(int).values
        X_tr[c] = codes_tr
        X_te[c] = codes_te
    cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols]
    print(f"[d12] cat indices: {cat_indices}")

    print(f"[d12] importing TabPFN (t={time.time()-t0:.1f}s)")
    from tabpfn import TabPFNClassifier

    print("[d12] building TabPFNClassifier (zero-shot, CPU) ...")
    clf = TabPFNClassifier(
        device="cpu",
        n_estimators=2,                      # smoke: minimal estimators
        categorical_features_indices=cat_indices,
        ignore_pretraining_limits=True,      # 10k is well within limits anyway
        random_state=SEED,
    )
    print(f"[d12] fit (t={time.time()-t0:.1f}s) ...")
    t_fit = time.time()
    clf.fit(X_tr.values, y_tr)
    fit_wall = time.time() - t_fit
    print(f"[d12] fit done ({fit_wall:.1f}s); predicting ...")

    if (time.time() - t0) > TIME_LIMIT_S:
        print(f"[d12] OVER 25min cap before predict; abort.")
        return

    t_pred = time.time()
    proba = clf.predict_proba(X_te.values)
    pred_wall = time.time() - t_pred
    print(f"[d12] predict done ({pred_wall:.1f}s)")

    auc = float(roc_auc_score(y_te, proba[:, 1]))
    total_wall = time.time() - t0
    print(f"\n[d12] zero-shot AUC on 5k holdout: {auc:.5f}")
    print(f"[d12] wall: fit={fit_wall:.0f}s  predict={pred_wall:.0f}s  total={total_wall:.0f}s")

    # Save smoke artifacts (subsample-only, NOT for stacking)
    np.save(ART / "oof_d12_tabpfn_smoke10k.npy", proba.astype(np.float32))

    # Reference comparisons
    BASE_S = 0.94075   # baseline_two_anchor Strat OOF
    REALMLP_E4 = 0.94722  # E4 fold-0 reference
    delta_baseline = (auc - BASE_S) * 1e4
    delta_realmlp = (auc - REALMLP_E4) * 1e4

    res = dict(
        smoke="d12_tabpfn_zero_shot_cpu_subsample",
        n_train_sub=int(N_TRAIN_SUBSAMPLE),
        n_test_sub=int(N_TEST_SUBSAMPLE),
        cat_cols=cat_cols,
        cat_indices=cat_indices,
        zero_shot_auc=auc,
        delta_vs_baseline_bp=delta_baseline,
        delta_vs_realmlp_e4_bp=delta_realmlp,
        fit_wall_s=fit_wall,
        predict_wall_s=pred_wall,
        total_wall_s=total_wall,
        seed=SEED,
        device="cpu",
        tabpfn_version="7.1.1",
        notes=(
            "Zero-shot ICL only — NO fine-tuning. Subsample-AUC is a "
            "coarse package sanity check, NOT a stack-eligible OOF. "
            "Full fine-tuning runs in kernels/d12-tabpfn-finetune-gpu/."
        ),
    )
    (ART / "d12_tabpfn_smoke_cpu_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
