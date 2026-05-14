"""Tree-based recalibration of K=11.

Builds a small LightGBM on top of K=11's stacker output plus a few raw
+ K=11-relative context features. The tree can capture non-linear
interactions between K=11 and (TyreLife, Stint, RaceProgress, Position),
plus per-group context (K=11 minus per-Race-Year mean) that K=11's
LR combiner cannot.

Mechanism: same family as V4 historical +0.8 bp lift (tree-internal
non-linear feature extraction on top of LR-class signals).

Inputs: K=11 OOF + test predictions, train + test feature columns.

Output:
  artifacts/K11_recal_oof.npy
  artifacts/K11_recal_test.npy
  artifacts/K11_recal.json

Reports standalone OOF AUC + K=11+1 LR-meta lift + rho diagnostics. The
caller decides whether to submit based on the lift + rho transfer band.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
DATA = Path("data")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 500


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def lr_meta_oof(Xm, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    om = np.zeros(len(y))
    for tr, va in skf.split(Xm, y):
        m = LogisticRegression(C=1.0, max_iter=MAX_ITER, random_state=SEED)
        m.fit(Xm[tr], y[tr])
        om[va] = m.predict_proba(Xm[va])[:, 1]
    return om


def add_K11_context_features(df: pd.DataFrame, K11_pred: np.ndarray) -> pd.DataFrame:
    """Return df augmented with K=11 prediction and K=11-relative context."""
    out = df.copy()
    out["K11"] = K11_pred
    # Per-(Race,Year) mean and the row's deviation from it
    grp_ry = out.groupby(["Race", "Year"])["K11"]
    out["K11_ry_mean"] = grp_ry.transform("mean")
    out["K11_ry_std"] = grp_ry.transform("std").fillna(0)
    out["K11_minus_ry_mean"] = out["K11"] - out["K11_ry_mean"]
    # Per-(Driver, Race, Year) sequence: lap-trajectory of K=11
    out = out.sort_values(["Driver", "Race", "Year", "LapNumber"]).reset_index(drop=True)
    grp_drv = out.groupby(["Driver", "Race", "Year"])["K11"]
    out["K11_lag1"] = grp_drv.shift(1).fillna(out["K11"])
    out["K11_lag2"] = grp_drv.shift(2).fillna(out["K11"])
    out["K11_delta1"] = out["K11"] - out["K11_lag1"]
    # Rank within (Race, Year)
    out["K11_ry_rank"] = grp_ry.rank(pct=True)
    return out


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values

    K11_oof = _pos(ART / "K11_full_pathb_tau100000_oof.npy")
    K11_test = _pos(ART / "K11_full_pathb_tau100000_test.npy")
    K11_auc = float(roc_auc_score(y, K11_oof))
    print(f"K=11 OOF AUC: {K11_auc:.5f}")

    # Augment train + test with K=11 + context features
    print("Building context features...", flush=True)
    train_aug = add_K11_context_features(train, K11_oof)
    test_aug = add_K11_context_features(test, K11_test)

    # Feature set: K=11 + 5 context + 6 raw informative features
    feats = [
        "K11", "K11_lag1", "K11_lag2", "K11_delta1",
        "K11_minus_ry_mean", "K11_ry_std", "K11_ry_rank",
        "TyreLife", "Stint", "LapNumber", "Position", "RaceProgress",
        "Cumulative_Degradation", "PitStop",
    ]
    print(f"feats ({len(feats)}): {feats}", flush=True)

    # Re-align train_aug back to original train.csv row order (we sorted earlier)
    train_aug = train_aug.sort_values("id").reset_index(drop=True)
    test_aug = test_aug.sort_values("id").reset_index(drop=True)
    X_train = train_aug[feats].fillna(0).astype(np.float32)
    X_test = test_aug[feats].fillna(0).astype(np.float32)
    # y is in original train.csv order (matches K11_oof which is in original order)

    print("\nTraining LightGBM recalibrator (5-fold CV)...", flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test_aug))
    walls = []
    for fold, (ti, vi) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        t1 = time.time()
        m = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.02, num_leaves=31,
            min_child_samples=200, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, reg_alpha=0.0, reg_lambda=0.0,
            verbose=-1, random_state=SEED, n_jobs=-1,
        )
        m.fit(X_train.iloc[ti], y[ti],
              eval_set=[(X_train.iloc[vi], y[vi])],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])
        oof[vi] = m.predict_proba(X_train.iloc[vi])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        walls.append(time.time() - t1)
        print(f"  fold {fold}: AUC={roc_auc_score(y[vi], oof[vi]):.5f}",
              f"iters={m.best_iteration_}  wall={walls[-1]:.1f}s", flush=True)

    standalone = float(roc_auc_score(y, oof))
    print(f"\nK=11_recal standalone OOF AUC: {standalone:.5f}",
          f"(delta vs K=11 plain {(standalone - K11_auc) * 1e4:+.3f} bp)", flush=True)

    # K=11+1 gate (LR-meta on [K=11_oof, recal_oof])
    Xm = expand(np.column_stack([K11_oof, oof]))
    om = lr_meta_oof(Xm, y)
    plus1_auc = float(roc_auc_score(y, om))
    plus1_delta = (plus1_auc - K11_auc) * 1e4
    print(f"K=11+1 (K=11 + recal) LR-meta OOF: {plus1_auc:.5f} ({plus1_delta:+.3f} bp)",
          flush=True)

    # Rho diagnostics
    rho_oof = float(spearmanr(oof, K11_oof).statistic)
    rho_test = float(spearmanr(test_pred, K11_test).statistic)
    print(f"\nrho_oof  K11_recal vs K=11: {rho_oof:.6f}", flush=True)
    print(f"rho_test K11_recal vs K=11: {rho_test:.6f}", flush=True)
    if rho_test >= 0.9999:
        verdict = "TIE_ZONE"
    elif rho_test < 0.999:
        verdict = "REGRESSION_RISK"
    else:
        verdict = "OK_TO_BLEND"
    print(f"Verdict: {verdict}", flush=True)

    np.save(ART / "K11_recal_oof.npy", oof)
    np.save(ART / "K11_recal_test.npy", test_pred)
    summary = {
        "K11_oof_auc": K11_auc,
        "K11_recal_standalone_oof": standalone,
        "K11_plus1_recal_oof": plus1_auc,
        "K11_plus1_lift_bp": plus1_delta,
        "rho_oof": rho_oof,
        "rho_test": rho_test,
        "verdict": verdict,
        "elapsed_sec": time.time() - t0,
    }
    (ART / "K11_recal.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
