"""scripts/d15_orig_multi_arch.py — multi-arch orig-transfer bag.

Build 3 additional orig-trained bases beyond d15_orig_transfer (LGBM):
  - d15_orig_cb       : CatBoost with native cat features
  - d15_orig_xgb      : XGBoost (cat encoded as int)
  - d15_orig_lgbm_t   : Optuna-flavored LGBM tuned for orig (deeper)

Each is trained ONCE on the aadigupta1601 99k-row original, predicts
on synth train+test. By construction:
  - All ρ ≈ 0.56 vs PRIMARY single-row (orthogonal-class)
  - Inter-architecture ρ likely 0.85-0.95 (some diversity within the
    orig-trained family — different inductive biases)

Adds 3 new artifacts to scripts/artifacts/. Use with hier-meta probe
of K=22/23/24 to test if additional orig-arch's stack incrementally.

~5-10 min total (orig is small).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET = "PitNextLap"


def load():
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    # Normalized_TyreLife from synth via stint-fraction estimate
    for df in [tr, te]:
        g = df.groupby(["Driver", "Race", "Year", "Stint"])["TyreLife"].transform("max")
        df["Normalized_TyreLife"] = (df["TyreLife"] / g.clip(lower=1)).astype(np.float32)
    return tr, te, orig


def align_categories(tr, te, orig, cat_cols):
    for c in cat_cols:
        union = pd.concat([tr[c], te[c], orig[c]], axis=0).astype(str)
        cats = sorted(union.unique())
        for df in [tr, te, orig]:
            df[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return tr, te, orig


def save_pred(pred_tr, pred_te, name):
    oof2 = np.column_stack([1 - pred_tr, pred_tr])
    test2 = np.column_stack([1 - pred_te, pred_te])
    np.save(ART / f"oof_{name}_strat.npy", oof2)
    np.save(ART / f"test_{name}_strat.npy", test2)
    print(f"  → saved oof/test_{name}_strat.npy")


def train_lgbm_tuned(orig, tr, te, cat_cols, num_cols, name):
    import lightgbm as lgb
    print(f"\n=== {name} (LGBM tuned) ===")
    feat = cat_cols + num_cols
    Xo = orig[feat]; yo = orig[TARGET].astype(int).values
    Xt = tr[feat]; Xe = te[feat]
    Xo_tr, Xo_va, yo_tr, yo_va = train_test_split(
        Xo, yo, test_size=0.15, random_state=SEED, stratify=yo)
    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.025, num_leaves=255, max_depth=-1,
        min_data_in_leaf=50, feature_fraction=0.85, bagging_fraction=0.85,
        bagging_freq=5, lambda_l2=2.0, verbose=-1, n_jobs=-1, seed=SEED,
    )
    dtr = lgb.Dataset(Xo_tr, label=yo_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(Xo_va, label=yo_va, categorical_feature=cat_cols, reference=dtr)
    t0 = time.time()
    m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False),
                             lgb.log_evaluation(0)])
    print(f"  wall {time.time()-t0:.0f}s, best_iter={m.best_iteration}, "
          f"orig held-out AUC={m.best_score['valid_0']['auc']:.5f}")
    p_tr = m.predict(Xt); p_te = m.predict(Xe)
    print(f"  synth-train AUC: {roc_auc_score(tr[TARGET].astype(int), p_tr):.5f}")
    save_pred(p_tr, p_te, name)


def train_xgb(orig, tr, te, cat_cols, num_cols, name):
    import xgboost as xgb
    print(f"\n=== {name} (XGBoost) ===")
    feat = cat_cols + num_cols
    Xo = orig[feat].copy(); yo = orig[TARGET].astype(int).values
    Xt = tr[feat].copy(); Xe = te[feat].copy()
    # XGB: convert cat → integer codes (use shared codes via earlier alignment)
    for c in cat_cols:
        Xo[c] = Xo[c].cat.codes.astype(np.int32)
        Xt[c] = Xt[c].cat.codes.astype(np.int32)
        Xe[c] = Xe[c].cat.codes.astype(np.int32)
    # Coerce numerics
    for c in num_cols:
        Xo[c] = Xo[c].astype(np.float32).fillna(-999)
        Xt[c] = Xt[c].astype(np.float32).fillna(-999)
        Xe[c] = Xe[c].astype(np.float32).fillna(-999)
    Xo_tr, Xo_va, yo_tr, yo_va = train_test_split(
        Xo, yo, test_size=0.15, random_state=SEED, stratify=yo)
    params = dict(
        objective="binary:logistic", eval_metric="auc",
        learning_rate=0.04, max_depth=8, min_child_weight=10,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
        tree_method="hist", n_jobs=-1, random_state=SEED,
        max_bin=256,
    )
    t0 = time.time()
    m = xgb.XGBClassifier(n_estimators=3000, **params)
    m.fit(Xo_tr, yo_tr, eval_set=[(Xo_va, yo_va)],
          verbose=False)
    # XGBClassifier 3.x uses early_stopping_rounds via constructor
    # but we'll just take the full model; rely on params for control.
    print(f"  wall {time.time()-t0:.0f}s")
    auc_va = roc_auc_score(yo_va, m.predict_proba(Xo_va)[:, 1])
    print(f"  orig held-out AUC={auc_va:.5f}")
    p_tr = m.predict_proba(Xt)[:, 1]
    p_te = m.predict_proba(Xe)[:, 1]
    print(f"  synth-train AUC: {roc_auc_score(tr[TARGET].astype(int), p_tr):.5f}")
    save_pred(p_tr, p_te, name)


def train_cb(orig, tr, te, cat_cols, num_cols, name):
    import catboost as cb
    print(f"\n=== {name} (CatBoost) ===")
    feat = cat_cols + num_cols
    Xo = orig[feat].copy(); yo = orig[TARGET].astype(int).values
    Xt = tr[feat].copy(); Xe = te[feat].copy()
    # CatBoost takes string cats fine
    for c in cat_cols:
        Xo[c] = Xo[c].astype(str)
        Xt[c] = Xt[c].astype(str)
        Xe[c] = Xe[c].astype(str)
    Xo_tr, Xo_va, yo_tr, yo_va = train_test_split(
        Xo, yo, test_size=0.15, random_state=SEED, stratify=yo)
    cat_idx = [feat.index(c) for c in cat_cols]
    pool_tr = cb.Pool(Xo_tr, label=yo_tr, cat_features=cat_idx)
    pool_va = cb.Pool(Xo_va, label=yo_va, cat_features=cat_idx)
    m = cb.CatBoostClassifier(
        iterations=3000, learning_rate=0.04, depth=7,
        l2_leaf_reg=3.0, eval_metric="AUC", od_type="Iter",
        od_wait=200, random_seed=SEED, allow_writing_files=False,
        verbose=False,
    )
    t0 = time.time()
    m.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    print(f"  wall {time.time()-t0:.0f}s, best_iter={m.tree_count_}")
    auc_va = roc_auc_score(yo_va, m.predict_proba(Xo_va)[:, 1])
    print(f"  orig held-out AUC={auc_va:.5f}")
    p_tr = m.predict_proba(Xt)[:, 1]
    p_te = m.predict_proba(Xe)[:, 1]
    print(f"  synth-train AUC: {roc_auc_score(tr[TARGET].astype(int), p_tr):.5f}")
    save_pred(p_tr, p_te, name)


def main():
    print("Loading data...")
    tr, te, orig = load()
    print(f"  synth train: {tr.shape} | synth test: {te.shape} | orig: {orig.shape}")

    cat_cols = ["Driver", "Compound", "Race"]
    num_cols = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
                "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                "RaceProgress", "Position_Change", "Normalized_TyreLife"]
    tr, te, orig = align_categories(tr, te, orig, cat_cols)

    train_cb(orig, tr, te, cat_cols, num_cols, "d15_orig_cb")
    train_xgb(orig, tr, te, cat_cols, num_cols, "d15_orig_xgb")
    train_lgbm_tuned(orig, tr, te, cat_cols, num_cols, "d15_orig_lgbm_t")

    # Quick ρ matrix among the 4 orig-trained variants (synth test):
    print("\n=== ρ matrix on synth TEST predictions ===")
    from scipy.stats import spearmanr
    names = ["d15_orig_transfer", "d15_orig_cb", "d15_orig_xgb", "d15_orig_lgbm_t"]
    primary = np.load(ART / "test_d13e_compound_stint_tau20000_strat.npy")[:, 1]
    preds = []
    for n in names:
        p = np.load(ART / f"test_{n}_strat.npy")
        preds.append(p[:, 1] if p.ndim == 2 else p)
    print(f"\n  {'name':<26s} ρ vs PRIMARY")
    for n, p in zip(names, preds):
        rho, _ = spearmanr(p, primary)
        print(f"  {n:<26s} {rho:+.5f}")
    print(f"\n  pairwise ρ among orig-trained:")
    print(f"  {'':26s} " + "  ".join(f"{n[:14]:>14s}" for n in names))
    for i, ni in enumerate(names):
        row = f"  {ni:<26s} "
        for j, nj in enumerate(names):
            r, _ = spearmanr(preds[i], preds[j])
            row += f"  {r:>+14.5f}"
        print(row)


if __name__ == "__main__":
    main()
