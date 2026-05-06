"""d14 DGP-residuals probe — masked-column self-prediction features.

Thesis (PI-proposed 2026-05-06): synthetic data is NN-generated, so
features are physical only in a fuzzy way. Train one regressor per
"DGP-fingerprint" column to predict it from the rest, then use the
OOF residuals (and a composite anomaly score) as new features for the
PitNextLap LGBM. If reconstruction errors carry signal orthogonal to
the K=22 pool, we'd see (a) std-OOF >= e3 baseline and (b) min-meta
> 0bp at ρ < 0.999 vs PRIMARY.

Family: `single_base_fe_addition` (P=0.05, 4-of-4 NULL precedent
on Day-13/14 alt-axis). Cheap probe; rule-out is a valid result.

Target columns to reconstruct:
  - LapTime_Delta            (continuous, NN-noisy)
  - Cumulative_Degradation   (continuous, NN-noisy)
  - Position                 (continuous integer 1-20, light NN noise)
  - LapNumber                (semi-deterministic from Stint+TyreLife;
                              residual exposes within-stint anomalies)

Excluded as targets (deterministic / stint-arithmetic):
  - RaceProgress (= LapNumber / race_total_laps; trivial residual)
  - TyreLife (= laps since stint start; trivial)
  - Stint, Year (categorical-ish; residual not meaningful)

Run:
  python scripts/d14_dgp_residuals.py --smoke         # 1-fold sanity
  python scripts/d14_dgp_residuals.py                 # full 5-fold
  python scripts/d14_dgp_residuals.py --gate          # post-run gate report

After artifacts exist:
  python scripts/probe_min_meta.py --candidates d14_dgp_residuals
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
TARGET = "PitNextLap"
ID_COL = "id"
SEED, N_FOLDS = 42, 5
NAME = "d14_dgp_residuals"

CAT_COLS = ["Driver", "Compound", "Race"]
RECON_TARGETS = ["LapTime_Delta", "Cumulative_Degradation", "Position", "LapNumber"]

PRIMARY_OOF_FILE = "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST_FILE = "test_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_LB = 0.95049


# ---------- LGBM params ----------------------------------------------

def regressor_params() -> dict:
    return dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=200,
        verbose=-1,
        seed=SEED,
    )


def classifier_params() -> dict:
    return dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=200,
        verbose=-1,
        seed=SEED,
    )


# ---------- Reconstruction step --------------------------------------

def fit_recon_oof(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    target_col: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_rounds: int,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train LGBM regressor predicting `target_col` from rest of X.

    Excludes target_col from features. Categorical cols stay categorical.
    Returns (oof_pred, test_pred_avg, fold_rmse_list).
    """
    feat_cols = [c for c in X_train.columns if c != target_col]
    cats = [c for c in cat_cols if c in feat_cols]
    Xtr = X_train[feat_cols].copy()
    Xte = X_test[feat_cols].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    y_target = X_train[target_col].astype(np.float64).values

    oof_pred = np.zeros(len(Xtr), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    fold_rmse = []

    for k, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(Xtr.iloc[tr], y_target[tr], categorical_feature=cats)
        dva = lgb.Dataset(Xtr.iloc[va], y_target[va], categorical_feature=cats)
        m = lgb.train(
            regressor_params(), dtr, num_boost_round=n_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof_pred[va] = m.predict(Xtr.iloc[va])
        test_pred += m.predict(Xte) / len(splits)
        rmse = float(np.sqrt(mean_squared_error(y_target[va], oof_pred[va])))
        fold_rmse.append(rmse)
        print(f"    [{target_col}] fold {k}: RMSE={rmse:.4f}  "
              f"(best_iter={m.best_iteration})")

    return oof_pred, test_pred, fold_rmse


def make_residual_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    recon_targets: list[str],
    n_rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate residual + composite-anomaly features.

    Returns (train_with_features, test_with_features, recon_diagnostics).
    """
    train = train.copy()
    test = test.copy()
    diagnostics: dict[str, dict] = {}

    # Cap reconstruction-feature input set: drop target + id BEFORE
    # reconstruction (target is a label, id has no signal).
    drop_in = [TARGET, ID_COL]
    Xtr_full = train.drop(columns=drop_in, errors="ignore")
    Xte_full = test.drop(columns=[ID_COL], errors="ignore")

    z_cols: list[str] = []
    for col in recon_targets:
        if col not in Xtr_full.columns:
            print(f"  [warn] target column {col} missing — skipping")
            continue
        print(f"  --- reconstructing {col} (5-fold OOF) ---")
        oof_pred, test_pred, rmse_list = fit_recon_oof(
            Xtr_full, Xte_full, col, splits, n_rounds, CAT_COLS
        )
        # Residual = actual − predicted. Z-score across train for stability.
        actual_tr = train[col].astype(np.float64).values
        actual_te = test[col].astype(np.float64).values
        resid_tr = actual_tr - oof_pred
        resid_te = actual_te - test_pred
        mu, sigma = float(resid_tr.mean()), float(resid_tr.std() + 1e-9)
        z_tr = (resid_tr - mu) / sigma
        z_te = (resid_te - mu) / sigma
        z_name = f"dgp_z_{col}"
        train[z_name] = z_tr.astype(np.float32)
        test[z_name] = z_te.astype(np.float32)
        z_cols.append(z_name)
        diagnostics[col] = dict(
            fold_rmse_mean=float(np.mean(rmse_list)),
            fold_rmse_std=float(np.std(rmse_list)),
            resid_mean=mu, resid_sigma=sigma,
        )

    # Composite anomaly = sum |z|. Captures rows poorly explained by
    # the conditional joint of remaining features.
    if z_cols:
        train["dgp_anomaly_L1"] = np.abs(train[z_cols].values).sum(axis=1).astype(np.float32)
        test["dgp_anomaly_L1"] = np.abs(test[z_cols].values).sum(axis=1).astype(np.float32)
        z_cols.append("dgp_anomaly_L1")

    diagnostics["new_feature_cols"] = z_cols
    return train, test, diagnostics


# ---------- Target classifier ----------------------------------------

def train_target_classifier(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_rounds: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    drop_cols = [ID_COL, TARGET]
    feat_cols = [c for c in train.columns if c not in drop_cols]
    cats = [c for c in CAT_COLS if c in feat_cols]
    Xtr = train[feat_cols].copy()
    Xte = test[feat_cols].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")

    print(f"  feature_count={len(feat_cols)}; "
          f"new dgp cols: {[c for c in feat_cols if c.startswith('dgp_')]}")

    oof = np.zeros(len(y), dtype=np.float64)
    test_avg = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(Xtr.iloc[tr], y[tr], categorical_feature=cats)
        dva = lgb.Dataset(Xtr.iloc[va], y[va], categorical_feature=cats)
        m = lgb.train(
            classifier_params(), dtr, num_boost_round=n_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        p_va = m.predict(Xtr.iloc[va])
        oof[va] = p_va
        test_avg += m.predict(Xte) / len(splits)
        s = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(s)
        print(f"  [target] fold {k}: AUC={s:.5f}  (best_iter={m.best_iteration})")
    return oof, test_avg, fold_aucs


# ---------- Gate / report --------------------------------------------

def _expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def gate_report(name: str, oof: np.ndarray, test: np.ndarray,
                y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    primary_oof = np.load(ART / PRIMARY_OOF_FILE)[:, 1].astype(np.float64)
    primary_test = np.load(ART / PRIMARY_TEST_FILE)[:, 1].astype(np.float64)

    std_auc = float(roc_auc_score(y, oof))
    primary_auc = float(roc_auc_score(y, primary_oof))
    d_oof_bp = (std_auc - primary_auc) * 1e4
    rho, _ = spearmanr(test, primary_test)
    rho = float(rho)

    # Min-meta gate vs PRIMARY (K=2 LR meta).
    F_oof = _expand(np.column_stack([primary_oof, oof]))
    F_test = _expand(np.column_stack([primary_test, test]))
    mm_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        mm_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    mm_auc = float(roc_auc_score(y, mm_oof))
    mm_delta_bp = (mm_auc - primary_auc) * 1e4

    # Top-1% flip ratio (G3).
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    cand_pos = test >= rare_thr
    fpn = int(np.sum(primary_pos & ~cand_pos))
    fnp = int(np.sum(~primary_pos & cand_pos))
    flip_ratio = (min(fpn, fnp) / max(fpn, fnp)) if max(fpn, fnp) > 0 else 1.0

    info = dict(
        name=name,
        std_oof=std_auc,
        primary_oof=primary_auc,
        delta_oof_bp=float(d_oof_bp),
        rho_vs_primary=rho,
        min_meta_oof=mm_auc,
        min_meta_delta_bp=float(mm_delta_bp),
        g3_flip_ratio=float(flip_ratio),
        flips_primary_to_neg=fpn,
        flips_primary_to_pos=fnp,
    )
    print(f"\n=== GATE: {name} ===")
    print(f"  std OOF      : {std_auc:.5f}  vs PRIMARY {primary_auc:.5f}  "
          f"Δ {d_oof_bp:+.2f}bp")
    print(f"  ρ vs PRIMARY : {rho:.6f}")
    print(f"  min-meta OOF : {mm_auc:.5f}  Δ {mm_delta_bp:+.3f}bp")
    print(f"  G3 flip ratio (top-1%): {flip_ratio:.3f}  "
          f"(+→−: {fpn}, −→+: {fnp})")
    if mm_delta_bp >= 0.3 and rho < 0.999:
        verdict = "PASS — promote to K=22 stack-add probe"
    elif mm_delta_bp >= 0:
        verdict = "WEAK_PASS — marginal; check K=21+N min-meta"
    else:
        verdict = "FAIL — rule out single-base FE addition (5th NULL)"
    print(f"  verdict: {verdict}")
    info["verdict"] = verdict
    return info


# ---------- Main -----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 200 rounds, 50k row subsample for sanity check")
    ap.add_argument("--gate-only", action="store_true",
                    help="skip training; load existing oof/test artifacts and gate")
    ap.add_argument("--targets", nargs="+", default=RECON_TARGETS,
                    help=f"columns to reconstruct (default: {RECON_TARGETS})")
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== d14 DGP-residuals probe ===  smoke={args.smoke}")

    if args.gate_only:
        oof = np.load(ART / f"oof_{NAME}_strat.npy")[:, 1].astype(np.float64)
        test = np.load(ART / f"test_{NAME}_strat.npy")[:, 1].astype(np.float64)
        y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(skf.split(np.zeros(len(y)), y))
        info = gate_report(NAME, oof, test, y, splits)
        (ART / f"{NAME}_results.json").write_text(json.dumps(info, indent=2))
        return

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"train={len(train)}  test={len(test)}  pos_rate={y.mean():.4f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    if args.smoke:
        sub_idx = np.random.RandomState(0).choice(
            len(y), size=min(50000, len(y)), replace=False)
        splits = [(np.intersect1d(splits[0][0], sub_idx), splits[0][1])]
        n_recon_rounds = 150
        n_target_rounds = 200
    else:
        n_recon_rounds = 1500
        n_target_rounds = 2000

    print(f"\n[1/2] reconstruction step — targets: {args.targets}")
    train_aug, test_aug, recon_diag = make_residual_features(
        train, test, splits, args.targets, n_recon_rounds
    )

    print(f"\n[2/2] PitNextLap classifier on enriched features")
    oof, test_avg, fold_aucs = train_target_classifier(
        train_aug, test_aug, y, splits, n_target_rounds
    )

    if args.smoke:
        # Smoke = 1 fold; can't compute valid 5-fold OOF for gate.
        print("\n=== SMOKE complete ===")
        print(f"  fold-0 AUC: {fold_aucs[0]:.5f}  ({time.time()-t0:.0f}s)")
        print("  Re-run without --smoke for full 5-fold OOF + gate.")
        return

    # Save artifacts in pool convention.
    oof2 = np.column_stack([1 - oof, oof]).astype(np.float32)
    test2 = np.column_stack([1 - test_avg, test_avg]).astype(np.float32)
    np.save(ART / f"oof_{NAME}_strat.npy", oof2)
    np.save(ART / f"test_{NAME}_strat.npy", test2)
    print(f"  saved oof_{NAME}_strat.npy / test_{NAME}_strat.npy")

    # Gate report.
    info = gate_report(NAME, oof, test_avg, y, splits)
    info["fold_aucs"] = fold_aucs
    info["recon_diagnostics"] = recon_diag
    info["wall_seconds"] = time.time() - t0
    (ART / f"{NAME}_results.json").write_text(json.dumps(info, indent=2))
    print(f"  → {ART / f'{NAME}_results.json'}")


if __name__ == "__main__":
    main()
