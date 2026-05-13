"""Loss-function diversity probe — 5 LightGBM variants on the same features.

Builds 5 binary-classification bases differing only in the loss function:
  1. binary log-loss (control)
  2. binary + scale_pos_weight ~= 4.02 (balanced weighting)
  3. binary + is_unbalance=True (auto-balanced)
  4. xentropy (LightGBM xentropy objective)
  5. custom focal loss (gamma=2.0, alpha=0.25)

Same 5-fold StratifiedKFold (seed=42) as every other base. Same feature
set: 14 raw columns + 3 categorical encodings + 6 cross-validated
target encodings. Loss is the only axis of variation.

Audit context: at the meta layer, loss-function diversity has been
falsified six different ways (LambdaRank, YetiRank, RankNet, xentropy
meta, LightGBM meta on 36 features, AUC-pairwise XGBoost). At the base
layer the team has only tested CatBoost YetiRank (a ranking model
entirely, not a re-weighted log-loss).

K=11+1 gate: each variant's OOF is stacked with the K=11 OOF under
LR-meta. Variant qualifies for K=12 inclusion if delta > 0.3 bp.

Outputs:
  artifacts/loss_div_<variant>_oof.npy
  artifacts/loss_div_<variant>_test.npy
  artifacts/loss_div_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from p1_features import TE_CONFIGS, cv_target_encode  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
TARGET = "PitNextLap"
ID_COL = "id"
SEED, N_FOLDS = 42, 5

# Base LightGBM hparams (matches PROJECT_LGB from p1_single_lgbm.py).
BASE_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    reg_alpha=0.0,
    reg_lambda=0.0,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    verbose=-1,
    random_state=SEED,
    n_jobs=-1,
)

# Variants — each appends or overrides BASE_PARAMS.
# (display_name, params_override_or_callable, post_predict_fn)
# For focal loss we use lgb.train with a custom fobj (callable below).


# -------------------- focal loss --------------------
GAMMA, ALPHA = 2.0, 0.25


def focal_obj(y_true, y_pred):
    """Binary focal-loss gradient and hessian for LightGBM.

    LightGBM passes raw scores (before sigmoid). We compute gradient and
    hessian of -alpha * (1-p)^gamma * y * log(p)
                 - (1-alpha) * p^gamma * (1-y) * log(1-p)
    with respect to the raw score z. (Standard focal loss for binary.)
    """
    z = y_pred
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    y = y_true.astype(np.float64)
    # Gradient: d L / d z
    g_pos = -ALPHA * y * (1 - p) ** GAMMA * (
        GAMMA * p * np.log(p) + (1 - p)
    )
    g_neg = (1 - ALPHA) * (1 - y) * p ** GAMMA * (
        GAMMA * (1 - p) * np.log(1 - p) + p
    )
    grad = -(g_pos + g_neg)
    # Hessian approximation: d^2 L / d z^2 (diagonal element)
    h_pos = ALPHA * y * (1 - p) ** GAMMA * p * (1 - p) * (
        GAMMA * (1 - p) + GAMMA ** 2 * np.log(p) * (1 - p)
        + 2 * GAMMA * p - 1
    )
    h_neg = (1 - ALPHA) * (1 - y) * p ** GAMMA * p * (1 - p) * (
        GAMMA * p + GAMMA ** 2 * np.log(1 - p) * p
        + 2 * GAMMA * (1 - p) - 1
    )
    hess = np.abs(h_pos + h_neg) + 1e-6  # numerical safety
    return grad, hess


def focal_eval(y_true, y_pred):
    """Evaluation metric paired with focal objective: returns negative
    AUC so LightGBM's early-stopping treats higher as better."""
    auc = roc_auc_score(y_true, y_pred)
    return "auc_eval", auc, True  # higher is better


# -------------------- variants --------------------
def make_variants(base_rate: float) -> list[tuple[str, dict, bool]]:
    """Returns list of (name, params, use_custom_objective_flag)."""
    spw = (1 - base_rate) / base_rate
    return [
        ("control_logloss",
         {**BASE_PARAMS, "objective": "binary", "metric": "auc"}, False),
        ("scale_pos_weight",
         {**BASE_PARAMS, "objective": "binary", "metric": "auc",
          "scale_pos_weight": spw}, False),
        ("is_unbalance",
         {**BASE_PARAMS, "objective": "binary", "metric": "auc",
          "is_unbalance": True}, False),
        ("xentropy",
         {**BASE_PARAMS, "objective": "xentropy", "metric": "auc"}, False),
        ("focal_g2_a025",
         {**BASE_PARAMS, "metric": "None"}, True),  # focal via lgb.train with fobj
    ]


# -------------------- feature pipeline --------------------
def build_features(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
                   fold_list: list[tuple[np.ndarray, np.ndarray]]):
    """Raw cols + cat encodings + cross-validated target encodings.

    Returns X_train, X_test, feature names, categorical feature list.
    """
    # 14 raw + cat encodings
    raw_cols = ["LapNumber", "PitStop", "TyreLife", "Position", "Stint",
                "Year", "LapTime (s)", "LapTime_Delta",
                "Cumulative_Degradation", "RaceProgress", "Position_Change"]
    cat_cols_raw = ["Driver", "Race", "Compound"]
    # build label-encoded categorical features (id-stable across train+test)
    out_cat_cols = []
    train_X = train[raw_cols].copy()
    test_X = test[raw_cols].copy()
    for c in cat_cols_raw:
        uniques = pd.concat([train[c], test[c]]).astype(str).unique()
        mp = {v: i for i, v in enumerate(sorted(uniques))}
        train_X[f"{c}_cat"] = train[c].astype(str).map(mp).astype("int32")
        test_X[f"{c}_cat"] = test[c].astype(str).map(mp).astype("int32")
        out_cat_cols.append(f"{c}_cat")

    # Target encodings (cross-validated, fold-safe)
    train_te_input = train.copy()
    test_te_input = test.copy()
    for cols, smooth, te_name in TE_CONFIGS:
        if all(c in train.columns for c in cols):
            oof_enc, te_enc = cv_target_encode(
                train_te_input, test_te_input, cols, pd.Series(y), fold_list,
                smoothing=smooth)
            train_X[te_name] = oof_enc
            test_X[te_name] = te_enc

    feats = list(train_X.columns)
    cat_cols = [c for c in out_cat_cols if c in feats]
    # fillna numeric
    num_cols = [c for c in feats if c not in cat_cols]
    train_X[num_cols] = train_X[num_cols].fillna(0).astype(np.float32)
    test_X[num_cols] = test_X[num_cols].fillna(0).astype(np.float32)
    return train_X, test_X, feats, cat_cols


# -------------------- train one variant --------------------
def train_variant(name: str, params: dict, use_focal: bool,
                  X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y: np.ndarray, feats: list, cat_cols: list,
                  fold_list: list) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n=== variant {name} ===", flush=True)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    aucs, iters, walls = [], [], []
    for fold, (ti, vi) in enumerate(fold_list, 1):
        t0 = time.time()
        Xtr, Xva = X_train.iloc[ti], X_train.iloc[vi]
        ytr, yva = y[ti], y[vi]

        if use_focal:
            dtr = lgb.Dataset(Xtr[feats], label=ytr,
                              categorical_feature=cat_cols)
            dva = lgb.Dataset(Xva[feats], label=yva,
                              categorical_feature=cat_cols)
            booster = lgb.train(
                params, dtr, num_boost_round=params["n_estimators"],
                valid_sets=[dva], valid_names=["va"],
                fobj=focal_obj, feval=focal_eval,
                callbacks=[lgb.early_stopping(150, verbose=False),
                           lgb.log_evaluation(0)],
            )
            # focal_obj outputs raw scores; need sigmoid
            raw_va = booster.predict(Xva[feats], num_iteration=booster.best_iteration)
            raw_te = booster.predict(X_test[feats], num_iteration=booster.best_iteration)
            p_va = 1.0 / (1.0 + np.exp(-raw_va))
            p_te = 1.0 / (1.0 + np.exp(-raw_te))
            best = booster.best_iteration
        else:
            m = lgb.LGBMClassifier(**params)
            m.fit(Xtr[feats], ytr,
                  eval_set=[(Xva[feats], yva)],
                  categorical_feature=cat_cols,
                  callbacks=[lgb.early_stopping(150, verbose=False),
                             lgb.log_evaluation(0)])
            p_va = m.predict_proba(Xva[feats])[:, 1]
            p_te = m.predict_proba(X_test[feats])[:, 1]
            best = int(m.best_iteration_ or params["n_estimators"])

        oof[vi] = p_va
        test_pred += p_te / N_FOLDS
        aucs.append(float(roc_auc_score(yva, p_va)))
        iters.append(best)
        walls.append(time.time() - t0)
        print(f"  fold {fold} AUC={aucs[-1]:.5f}  iters={best}  wall={walls[-1]:.1f}s",
              flush=True)

    auc_full = float(roc_auc_score(y, oof))
    print(f"  standalone OOF AUC = {auc_full:.5f}  walls_sum={sum(walls):.1f}s",
          flush=True)

    np.save(ART / f"loss_div_{name}_oof.npy", oof)
    np.save(ART / f"loss_div_{name}_test.npy", test_pred)
    return oof, test_pred


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def lr_meta_oof(Xm: np.ndarray, y: np.ndarray) -> np.ndarray:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    om = np.zeros(len(y))
    for tr, va in skf.split(Xm, y):
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        m.fit(Xm[tr], y[tr])
        om[va] = m.predict_proba(Xm[va])[:, 1]
    return om


def k11_plus_one_gate(K11_oof: np.ndarray, variant_oof: np.ndarray,
                      y: np.ndarray) -> float:
    """Returns the K=11+1 lift in basis points."""
    Xm = expand(np.column_stack([K11_oof, variant_oof]))
    om = lr_meta_oof(Xm, y)
    auc_K11_plus = float(roc_auc_score(y, om))
    auc_K11 = float(roc_auc_score(y, K11_oof))
    return (auc_K11_plus - auc_K11) * 1e4


def main() -> None:
    t0_total = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values
    base_rate = float(y.mean())
    print(f"train {train.shape}  test {test.shape}  base_rate {base_rate:.4f}",
          flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    print("\nbuilding features...", flush=True)
    X_train, X_test, feats, cat_cols = build_features(train, test, y, fold_list)
    print(f"  feats={len(feats)}  cat_cols={cat_cols}", flush=True)

    K11_oof = np.load(ART / "K11_full_pathb_tau100000_oof.npy").astype(np.float64)
    K11_test = np.load(ART / "K11_full_pathb_tau100000_test.npy").astype(np.float64)
    K11_auc = float(roc_auc_score(y, K11_oof))
    print(f"\nK=11 OOF reference: {K11_auc:.5f}", flush=True)

    variants = make_variants(base_rate)
    results = []
    for name, params, use_focal in variants:
        oof, test_pred = train_variant(
            name, params, use_focal,
            X_train, X_test, y, feats, cat_cols, fold_list)
        standalone = float(roc_auc_score(y, oof))
        rho_oof = float(spearmanr(oof, K11_oof).statistic)
        rho_test = float(spearmanr(test_pred, K11_test).statistic)
        lift = k11_plus_one_gate(K11_oof, oof, y)
        verdict = "STRONG" if lift > 0.5 else ("WEAK" if lift > 0.1 else "NULL")
        print(f"  -> standalone {standalone:.5f}  rho_oof_K11 {rho_oof:.6f}  "
              f"rho_test_K11 {rho_test:.6f}  K=11+1 lift {lift:+.3f} bp  [{verdict}]",
              flush=True)
        results.append({
            "name": name,
            "standalone_oof": standalone,
            "rho_oof_vs_K11": rho_oof,
            "rho_test_vs_K11": rho_test,
            "K11_plus_1_lift_bp": lift,
            "verdict": verdict,
        })

    print("\n=== SUMMARY ===", flush=True)
    df = pd.DataFrame(results)
    print(df.to_string(index=False), flush=True)

    (ART / "loss_div_summary.json").write_text(
        json.dumps({"K11_oof_auc": K11_auc, "results": results,
                    "elapsed_sec": time.time() - t0_total}, indent=2,
                   default=str))
    print(f"\ntotal elapsed: {time.time()-t0_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
