"""scripts/d17_h1_yekenot_realmlp_orig.py — H1 with orig-merge data augmentation.

Per pit-or-stay-f1-strategy-1 cell 58 (the load-bearing detail H1 v1
missed): each fold trains on (train_fold + ORIG full), validates on
train_fold's val rows. ORIG = data/original/f1_strategy_dataset_v4.csv.
This is fold-safe (Rule 24) — orig is independent samples, not a
target-aggregated feature. Pass on Rule 25 — train+test AV-AUC = 0.502.

Reuses build_features / fit_realmlp from d17_h1_yekenot_realmlp.py.

Usage:
  python scripts/d17_h1_yekenot_realmlp_orig.py --mode strong --out-suffix _strong_orig
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Reuse from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from d17_h1_yekenot_realmlp import (  # noqa: E402
    ART, TARGET, ID_COL, SEED, N_FOLDS,
    CPU_FAST, CPU_FAST_FALLBACK, CPU_STRONG, CPU_STRONGER,
    YEKENOT_PARAMS_BASE, RAW_NUM_COLS, RAW_CAT_COLS,
    build_features, fit_realmlp,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "fallback", "strong", "stronger"], default="strong")
    ap.add_argument("--out-suffix", default="_orig")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--time-budget-min", type=float, default=85.0)
    args = ap.parse_args()

    cpu_params = {"fast": CPU_FAST, "fallback": CPU_FAST_FALLBACK,
                  "strong": CPU_STRONG, "stronger": CPU_STRONGER}[args.mode]
    print(f"=== D17 H1 yekenot RealMLP + ORIG MERGE ({args.mode}) ===")
    print(f"  CPU params: {cpu_params}")
    print(f"  budget: {args.time_budget_min:.0f} min  folds: {args.folds}")

    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    print(f"  train={train.shape} test={test.shape} orig={orig.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    y_orig = orig[TARGET].astype(int).values
    train_x = train.drop(columns=[TARGET, ID_COL])
    test_x = test.drop(columns=[ID_COL])
    orig_x = orig.drop(columns=[TARGET])
    if ID_COL in orig_x.columns:
        orig_x = orig_x.drop(columns=[ID_COL])

    # Align orig columns to train_x: keep only the cols both have, in train_x order
    common_cols = [c for c in train_x.columns if c in orig_x.columns]
    missing_in_orig = [c for c in train_x.columns if c not in orig_x.columns]
    extra_in_orig = [c for c in orig_x.columns if c not in train_x.columns]
    print(f"  common cols: {len(common_cols)}  missing in orig: {missing_in_orig}  extra in orig: {extra_in_orig}")
    train_x = train_x[common_cols]
    test_x = test_x[common_cols]
    orig_x = orig_x[common_cols]

    print(f"  building features (joint train+test+orig encoding)...")
    # Combined feature build: stack train + test + orig (so cat encoders / KBins
    # see all values consistently). Then split back.
    n_tr, n_te, n_or = len(train_x), len(test_x), len(orig_x)
    combined = pd.concat([train_x, test_x, orig_x], axis=0, ignore_index=True)
    # Use build_features by passing combined as both train and test then re-splitting
    X_combined, X_te_dummy, feat_cols, cat_cols = build_features(
        combined.head(n_tr + n_or), combined.tail(n_te).reset_index(drop=True)
    )
    # X_combined has n_tr + n_or rows (train then orig). Split.
    X_train = X_combined.iloc[:n_tr].reset_index(drop=True)
    X_orig = X_combined.iloc[n_tr:n_tr + n_or].reset_index(drop=True)
    X_test = X_te_dummy.reset_index(drop=True)
    print(f"  X_train shape: {X_train.shape}  X_orig shape: {X_orig.shape}  X_test shape: {X_test.shape}")
    print(f"  feature-build done t={time.time()-t0:.1f}s")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred_sum = np.zeros(len(test_x), dtype=np.float64)
    test_pred_n = 0
    fold_aucs = []
    fold_walls = []

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        if fold > args.folds:
            print(f"  hit --folds cap; stopping after fold {fold-1}")
            break
        if (time.time() - t0) / 60 > args.time_budget_min:
            print(f"  HIT BUDGET ({args.time_budget_min:.0f} min) before fold {fold}; abort")
            break
        print(f"\n--- Fold {fold}/{N_FOLDS} | t_elapsed={(time.time()-t0)/60:.1f} min ---")
        t1 = time.time()
        # Concat orig into training side
        X_tr_aug = pd.concat([X_train.iloc[tr], X_orig], axis=0, ignore_index=True)
        y_tr_aug = np.concatenate([y[tr], y_orig])
        print(f"  X_tr_aug shape: {X_tr_aug.shape} (train_fold={len(tr)} + orig={len(X_orig)})")
        model, p_va = fit_realmlp(X_tr_aug, y_tr_aug, X_train.iloc[va],
                                  cpu_params, n_threads=2)
        oof[va] = p_va
        fold_auc = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(fold_auc)
        fold_walls.append(time.time() - t1)
        print(f"  fold {fold} AUC = {fold_auc:.5f}  ({fold_walls[-1]:.0f}s)")

        p_test_fold = model.predict_proba(X_test)[:, 1]
        test_pred_sum += p_test_fold
        test_pred_n += 1
        del model

    if test_pred_n == 0:
        print("ABORT: no folds completed")
        return False

    test_pred = test_pred_sum / test_pred_n
    folds_done = test_pred_n

    covered_mask = np.zeros(len(y), dtype=bool)
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fi, (_, va) in enumerate(skf2.split(np.zeros(len(y)), y), 1):
        if fi <= folds_done:
            covered_mask[va] = True
    cov_pct = covered_mask.mean() * 100
    print(f"\n=== summary (folds completed: {folds_done}/{N_FOLDS}) ===")
    print(f"  coverage: {cov_pct:.1f}%  fold_aucs: {fold_aucs}")
    if folds_done == N_FOLDS:
        oof_auc = float(roc_auc_score(y, oof))
        print(f"  full OOF AUC: {oof_auc:.5f}")
    else:
        oof_auc = float(roc_auc_score(y[covered_mask], oof[covered_mask]))
        print(f"  partial-OOF AUC ({cov_pct:.0f}% cover): {oof_auc:.5f}")

    print(f"  vs default-config realmlp (0.94582): {(oof_auc-0.94582)*1e4:+.1f}bp")
    print(f"  vs h1 strong train-only   (0.94516): {(oof_auc-0.94516)*1e4:+.1f}bp")
    print(f"  vs yekenot published      (0.95273): {(oof_auc-0.95273)*1e4:+.1f}bp")
    print(f"  total wall: {(time.time()-t0)/60:.1f} min")

    suf = args.out_suffix
    oof2 = np.column_stack([1 - oof, oof]).astype(np.float32)
    test2 = np.column_stack([1 - test_pred, test_pred]).astype(np.float32)
    np.save(ART / f"oof_d17_h1_yekenot_realmlp{suf}_strat.npy", oof2)
    np.save(ART / f"test_d17_h1_yekenot_realmlp{suf}_strat.npy", test2)
    print(f"  saved oof/test_d17_h1_yekenot_realmlp{suf}_strat.npy")

    res = dict(
        mode=args.mode, cpu_params=cpu_params,
        with_orig=True, orig_shape=list(orig.shape),
        folds_done=folds_done, coverage_pct=cov_pct,
        fold_aucs=fold_aucs, fold_walls_s=fold_walls,
        oof_auc=oof_auc,
        delta_vs_default_realmlp_bp=(oof_auc - 0.94582) * 1e4,
        delta_vs_h1_strong_no_orig_bp=(oof_auc - 0.94516) * 1e4,
        delta_vs_yekenot_pub_bp=(oof_auc - 0.95273) * 1e4,
        total_wall_s=time.time() - t0,
    )
    (ART / f"d17_h1_yekenot_realmlp{suf}_results.json").write_text(json.dumps(res, indent=2))
    print(f"  saved d17_h1_yekenot_realmlp{suf}_results.json")
    return True


if __name__ == "__main__":
    main()
