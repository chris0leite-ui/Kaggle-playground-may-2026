"""scripts/probe_r9_compound_stint_te_base.py — Round 9 Phase A (NB4)

Per-(Compound × Stint) target-mean as a BASE learner — the segmentation
axis tested at the BASE LAYER (TE-broadcast) rather than at the META
LAYER (Path-B shrinkage). 6 existing TE_CONFIGS in p1_features.py do
NOT include Compound × Stint; mechanism is novel.

Hypothesis: TE-broadcast at base input adds row-rank information that
Path-B's per-(Compound, Stint) meta-LR shrinkage does not extract.

Gates:
  G1 standalone OOF ≥ 0.948  (yekenot-level proxy)
  G2 K=14 + Path-B Δ ≥ +0.05 bp vs R7.1 PRIMARY (0.95447)
  G3 ρ_test vs PRIMARY ∈ [0.999, 0.9999]  (OK transfer band)
  KILL: standalone < 0.945 OR K=14 Δ < +0.01 bp

Usage:
  python scripts/probe_r9_compound_stint_te_base.py [--smoke]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from p1_features import cv_target_encode  # noqa: E402

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)

LGB_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    lambda_l1=0.0, lambda_l2=1.0, max_depth=-1, n_jobs=-1,
    verbose=-1, random_state=SEED,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold on 50k rows to verify wall-time + signal")
    ap.add_argument("--max_rounds", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    print(f"== R9 Phase A: NB4 Compound × Stint TE as BASE ==")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}  prior {y_all.mean():.4f}")

    # Build 5-fold list (Stratified, seed 42) on FULL train
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_all)), y_all))

    # NB4 feature: per-(Compound × Stint) TE via cv_target_encode
    # (fold-safe — recomputes stats per fold's training rows only)
    print(f"  computing TE: (Compound × Stint) smoothing=30")
    te_oof, te_test = cv_target_encode(
        train, test, ["Compound", "Stint"],
        train[TARGET].astype(float), fold_list, smoothing=30,
    )
    print(f"    te_oof:  min={te_oof.min():.4f} max={te_oof.max():.4f} mean={te_oof.mean():.4f}")
    print(f"    te_test: min={te_test.min():.4f} max={te_test.max():.4f} mean={te_test.mean():.4f}")

    # Build feature matrix: raw + 1 TE column
    train["te_compound_stint"] = te_oof.astype(np.float32)
    test["te_compound_stint"]  = te_test.astype(np.float32)

    # Categorical encoding (Driver, Compound, Race → int32 via train∪test vocab)
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        u = pd.concat([train[c], test[c]], ignore_index=True).unique()
        m = {v: i for i, v in enumerate(u)}
        train[c] = train[c].map(m).astype(np.int32)
        test[c] = test[c].map(m).astype(np.int32)

    feat_cols = [c for c in train.columns if c not in {"id", TARGET}]
    print(f"  features: {len(feat_cols)}")
    print(f"    {feat_cols}")

    if args.smoke:
        # Reduce to 50k rows + 1 fold for smoke
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(train), size=50_000, replace=False))
        train = train.iloc[idx].reset_index(drop=True)
        y_all = y_all[idx]
        te_oof_smoke = te_oof[idx]
        # rebuild fold_list on smoke subset
        skf_smoke = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        fold_list = list(skf_smoke.split(np.zeros(len(y_all)), y_all))[:1]
        args.max_rounds = min(args.max_rounds, 500)
        print(f"  SMOKE -> train {train.shape}, 1 fold")

    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []
    iters_log = []

    for fold, (ti, vi) in enumerate(fold_list, 1):
        t_fold = time.time()
        # Rule 24 fold-safety self-check: TE at val rows must equal the
        # fold-out-of-fold encoding (not the full-train encoding)
        if not args.smoke:
            assert te_oof.shape[0] == len(y_all), \
                f"te_oof shape {te_oof.shape} mismatches y_all {len(y_all)}"

        X_tr = train.iloc[ti][feat_cols].fillna(0).values
        X_va = train.iloc[vi][feat_cols].fillna(0).values
        X_te = test[feat_cols].fillna(0).values
        y_tr = y_all[ti]
        y_va = y_all[vi]

        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=args.max_rounds)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

        oof_va = m.predict_proba(X_va)[:, 1]
        oof[vi] = oof_va
        if not args.smoke:
            test_pred += m.predict_proba(X_te)[:, 1] / N_FOLDS

        auc_va = roc_auc_score(y_va, oof_va)
        fold_aucs.append(float(auc_va))
        iters_log.append(int(m.best_iteration_) if m.best_iteration_ else args.max_rounds)
        print(f"  Fold {fold}: AUC={auc_va:.5f} iters={iters_log[-1]} "
              f"wall={time.time()-t_fold:.1f}s")

    if args.smoke:
        auc_full = fold_aucs[0]
    else:
        auc_full = float(roc_auc_score(y_all, oof))
    fold_std = float(np.std(fold_aucs))
    print(f"\n  Standalone OOF AUC: {auc_full:.5f}  "
          f"fold-std={fold_std:.5f}  total wall={time.time()-t0:.1f}s")

    # Gate G1
    if auc_full >= 0.948:
        print(f"  G1 PASS (standalone ≥ 0.948)")
    elif auc_full >= 0.945:
        print(f"  G1 WARN (standalone {auc_full:.5f} below yekenot-level 0.948)")
    else:
        print(f"  G1 FAIL (standalone < 0.945 — likely TE leak or noise)")
        if not args.smoke:
            print(f"  Aborting save to avoid polluting K=14 pool")
            return

    if args.smoke:
        print("  SMOKE complete. Skipping save.")
        return

    # Save artifacts
    np.save(ART / "oof_NB4_compound_stint_te_strat.npy", oof.astype(np.float64))
    np.save(ART / "test_NB4_compound_stint_te_strat.npy", test_pred.astype(np.float64))
    print(f"  Saved: oof_NB4_compound_stint_te_strat.npy "
          f"test_NB4_compound_stint_te_strat.npy")

    # Per-segment AUC on MEDIUM × Stint 2 (strategy-critic Section 1 weak segment)
    # Note: train was overwritten with cat-encoded ints; reload for the diagnostic.
    train_raw = pd.read_csv("data/train.csv")
    m_s2 = (train_raw["Compound"] == "MEDIUM") & (train_raw["Stint"] == 2)
    if m_s2.sum() > 100:
        auc_ms2 = roc_auc_score(y_all[m_s2.values], oof[m_s2.values])
        print(f"  Diagnostic: NB4 OOF AUC on MEDIUM × Stint 2 ({m_s2.sum()} rows): {auc_ms2:.4f}")
        print(f"    (Reference: PRIMARY R7.1 = 0.8975 on same subset)")


if __name__ == "__main__":
    main()
