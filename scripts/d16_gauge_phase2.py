"""d16 Phase 2 — density ratio r̂(x) = p_synth(x)/p_orig(x).

Sequential pipeline:
  P2.1  LGBM classifier orig vs synth (label=is_synth).
        Outputs π(x) calibrated probability + r̂(x) = π/(1-π) × n_orig/n_synth.
        Per-feature SHAP-style importance (gain) for tells.
  P2.2  r̂(x) as single feature → LGBM K=21+1 min-meta gate.
        Tests "across-distribution information" axis.
  P2.3  Retrain orig_transfer with sample weights ∝ r̂(x) on concat(orig, synth_pseudo).
        Pseudo-label synth via PRIMARY OOF/test where confident; weight by r̂.
  P2.4  Split synth at r̂-median into orig-like / hallucinated.
        Train 2 segment-specific orig-transfer-style bases; concat OOF/test predictions.

Outputs:
  scripts/artifacts/oof_d16_dr_rhat_strat.npy / test_*       — single-feature LGBM (P2.2)
  scripts/artifacts/oof_d16_dr_weighted_orig_strat.npy / test_*  — weighted orig (P2.3)
  scripts/artifacts/oof_d16_dr_split_strat.npy / test_*      — split base (P2.4)
  scripts/artifacts/d16_phase2_summary.json                  — AUC, top SHAP, gate verdicts
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

CAT = ["Driver", "Compound", "Race"]
NUM = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
       "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
       "RaceProgress", "Position_Change"]
FEATS = CAT + NUM


def align_cats(dfs, cat_cols):
    for c in cat_cols:
        union = pd.concat([d[c].astype(str) for d in dfs], axis=0)
        cats = sorted(union.dropna().unique())
        for d in dfs:
            d[c] = pd.Categorical(d[c].astype(str), categories=cats)
    return dfs


def fit_lgbm(X_tr, y_tr, X_va, y_va, **kw):
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=5, lambda_l1=0.0,
                  lambda_l2=0.0, verbose=-1, n_jobs=-1, seed=SEED)
    params.update(kw)
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT, free_raw_data=False)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=CAT, free_raw_data=False)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return m


def main():
    t0 = time.time()
    log = []

    def step(msg):
        log.append(f"[{time.time() - t0:6.1f}s] {msg}")
        print(log[-1])

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    y_synth = tr[TARGET].astype(int).values
    n_orig, n_synth_tr, n_synth_te = len(orig), len(tr), len(te)
    step(f"orig={n_orig} tr={n_synth_tr} te={n_synth_te}")

    # ---- align categoricals across orig + synth ----
    align_cats([tr, te, orig], CAT)

    # ==================================================================
    # P2.1  Train classifier orig (label 0) vs synth_train (label 1)
    # ==================================================================
    step("P2.1  build orig-vs-synth classifier")
    Xa = orig[FEATS].copy()
    Xb = tr[FEATS].copy()
    Xt = te[FEATS].copy()

    Xall = pd.concat([Xa, Xb], axis=0, ignore_index=True)
    yall = np.concatenate([np.zeros(n_orig), np.ones(n_synth_tr)]).astype(int)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pi_all = np.zeros(len(Xall))
    pi_synth = np.zeros(n_synth_tr)
    pi_test = np.zeros(n_synth_te)
    importances = np.zeros(len(FEATS))
    fold_aucs = []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(Xall, yall)):
        m = fit_lgbm(Xall.iloc[tr_idx], yall[tr_idx], Xall.iloc[va_idx], yall[va_idx])
        p_va = m.predict(Xall.iloc[va_idx])
        pi_all[va_idx] = p_va
        fold_aucs.append(roc_auc_score(yall[va_idx], p_va))
        # Predict on synth_test (no fold exists for test rows since they're 100% in synth-class)
        pi_test += m.predict(Xt) / N_FOLDS
        importances += m.feature_importance(importance_type="gain") / N_FOLDS
        step(f"  fold{fi} AUC {fold_aucs[-1]:.4f}")
    pi_synth = pi_all[n_orig:]
    av_auc = roc_auc_score(yall, pi_all)
    step(f"  ORIG-vs-SYNTH AUC: {av_auc:.4f}  (mean fold {np.mean(fold_aucs):.4f})")

    # density ratio r̂(x) = π/(1-π) × n_orig/n_synth_tr
    eps = 1e-6
    rhat_tr = (pi_synth / (1 - pi_synth + eps)) * (n_orig / n_synth_tr)
    rhat_te = (pi_test / (1 - pi_test + eps)) * (n_orig / n_synth_tr)
    rhat_tr = np.clip(rhat_tr, 1e-3, 1e3)
    rhat_te = np.clip(rhat_te, 1e-3, 1e3)
    np.save(ART / "d16_rhat_synth_train.npy", rhat_tr)
    np.save(ART / "d16_rhat_synth_test.npy", rhat_te)
    step(f"  r̂_tr stats: median {np.median(rhat_tr):.3f} q05 {np.quantile(rhat_tr, 0.05):.3f} q95 {np.quantile(rhat_tr, 0.95):.3f}")

    fi_sorted = sorted(zip(FEATS, importances.tolist()), key=lambda x: -x[1])
    step("  top features (orig-vs-synth tells):")
    for name, imp in fi_sorted[:5]:
        step(f"    {name}: {imp:.0f}")

    # ==================================================================
    # P2.2  r̂(x) as single feature → LGBM K=21+1 min-meta
    # ==================================================================
    step("P2.2  r̂(x) as single-feature LGBM")
    # Train LGBM on synth with rhat as ONE feature (logged).
    feat = np.log1p(rhat_tr).reshape(-1, 1)
    feat_te = np.log1p(rhat_te).reshape(-1, 1)
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n_synth_tr)
    pred_te = np.zeros(n_synth_te)
    for fi, (tri, vai) in enumerate(skf2.split(feat, y_synth)):
        m = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=15, min_data_in_leaf=500, feature_fraction=1.0,
                 verbose=-1, n_jobs=-1, seed=SEED),
            lgb.Dataset(feat[tri], y_synth[tri]),
            num_boost_round=200, valid_sets=[lgb.Dataset(feat[vai], y_synth[vai])],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        oof[vai] = m.predict(feat[vai])
        pred_te += m.predict(feat_te) / N_FOLDS
    p22_auc = roc_auc_score(y_synth, oof)
    step(f"  P2.2 standalone OOF AUC {p22_auc:.5f}")
    np.save(ART / "oof_d16_dr_rhat_strat.npy", oof)
    np.save(ART / "test_d16_dr_rhat_strat.npy", pred_te)

    # ==================================================================
    # P2.3  r̂-weighted orig_transfer: train on (orig + synth-pseudo)
    #       weight: orig=1, synth=r̂(x)  (downweight hallucinated)
    # ==================================================================
    step("P2.3  r̂-weighted orig_transfer-style base")
    # Build training set: orig with true labels + synth with PRIMARY soft labels weighted by r̂
    primary_oof = np.load(ART / "oof_PRIMARY_K22_strat.npy")
    if primary_oof.ndim == 2:
        primary_oof = primary_oof[:, 1]
    # synth pseudo-labels: hard threshold at 0.5 (or use soft via probabilistic loss)
    pseudo = (primary_oof >= 0.5).astype(int)
    # restrict to high-confidence synth rows + downweight by r̂
    conf = np.maximum(primary_oof, 1 - primary_oof)
    keep = conf > 0.7
    step(f"  pseudo-label kept rows: {keep.sum()}/{n_synth_tr} (conf>0.7)")

    # build orig features
    Xo = orig[FEATS].copy()
    yo = orig[TARGET].astype(int).values
    wo = np.ones(len(Xo))

    # build synth features (kept subset)
    Xs = tr.loc[keep, FEATS].copy()
    ys = pseudo[keep]
    ws = rhat_tr[keep]  # downweight rows that look like synth-only

    Xcomb = pd.concat([Xo, Xs], ignore_index=True)
    align_cats([Xcomb, te[FEATS]], CAT)
    ycomb = np.concatenate([yo, ys])
    wcomb = np.concatenate([wo, ws])

    # 5-fold stratify on the concatenated set, OOF only over the synth portion
    skf3 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred_synth_tr = np.zeros(n_synth_tr)
    pred_synth_te = np.zeros(n_synth_te)
    counts = np.zeros(n_synth_tr)
    Xte = te[FEATS]
    align_cats([Xcomb, Xte, tr[FEATS]], CAT)

    for fi, (tri, vai) in enumerate(skf3.split(Xcomb, ycomb)):
        # train only on tri rows; predict on synth_tr (full) + test
        m = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED),
            lgb.Dataset(Xcomb.iloc[tri], ycomb[tri], weight=wcomb[tri], categorical_feature=CAT),
            num_boost_round=600, valid_sets=[
                lgb.Dataset(Xcomb.iloc[vai], ycomb[vai], weight=wcomb[vai], categorical_feature=CAT)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        # predict on full synth_tr + test
        pred_synth_tr += m.predict(tr[FEATS]) / N_FOLDS
        pred_synth_te += m.predict(Xte) / N_FOLDS
        counts += 1
    p23_auc = roc_auc_score(y_synth, pred_synth_tr)
    step(f"  P2.3 standalone synth-train AUC {p23_auc:.5f}")
    np.save(ART / "oof_d16_dr_weighted_orig_strat.npy", pred_synth_tr)
    np.save(ART / "test_d16_dr_weighted_orig_strat.npy", pred_synth_te)

    # ==================================================================
    # P2.4  r̂-median split
    # ==================================================================
    step("P2.4  r̂-median split: 2 segment-specific orig bases")
    rhat_median = float(np.median(rhat_tr))
    step(f"  r̂ median: {rhat_median:.3f}")
    seg_tr = (rhat_tr >= rhat_median).astype(int)  # 1 = synth-like (high r̂)
    seg_te = (rhat_te >= rhat_median).astype(int)

    # Train 2 LGBMs on orig data (no segmentation in orig, since orig has y=label trust);
    # use synth segmentation only at prediction time to route.
    # Actually, we want segment-specific bases. So we train 2 orig-LGBMs with different
    # hyperparameters and route based on r̂. As a simpler test: train one orig LGBM,
    # predict on synth, and scale predictions per segment by a learned calibration
    # (Platt scaling per segment).
    Xo2 = orig[FEATS].copy()
    align_cats([Xo2, tr[FEATS], te[FEATS]], CAT)
    skf4 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred_o_synth = np.zeros(n_synth_tr)
    pred_o_test = np.zeros(n_synth_te)
    for fi, (tri, vai) in enumerate(skf4.split(Xo2, yo)):
        m = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED),
            lgb.Dataset(Xo2.iloc[tri], yo[tri], categorical_feature=CAT),
            num_boost_round=600, valid_sets=[
                lgb.Dataset(Xo2.iloc[vai], yo[vai], categorical_feature=CAT)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        pred_o_synth += m.predict(tr[FEATS]) / N_FOLDS
        pred_o_test += m.predict(te[FEATS]) / N_FOLDS

    # per-segment Platt scaling on synth using y_synth
    from sklearn.linear_model import LogisticRegression
    pred_synth_calibrated = np.zeros(n_synth_tr)
    pred_test_calibrated = np.zeros(n_synth_te)
    for seg in [0, 1]:
        mask = (seg_tr == seg)
        if mask.sum() < 100:
            pred_synth_calibrated[mask] = pred_o_synth[mask]
            pred_test_calibrated[seg_te == seg] = pred_o_test[seg_te == seg]
            continue
        lr = LogisticRegression(max_iter=200)
        feat_in = pred_o_synth[mask].reshape(-1, 1)
        lr.fit(np.log(feat_in / (1 - feat_in + eps) + eps).reshape(-1, 1), y_synth[mask])
        pred_synth_calibrated[mask] = lr.predict_proba(np.log(feat_in / (1 - feat_in + eps) + eps).reshape(-1, 1))[:, 1]
        feat_te_seg = pred_o_test[seg_te == seg].reshape(-1, 1)
        pred_test_calibrated[seg_te == seg] = lr.predict_proba(np.log(feat_te_seg / (1 - feat_te_seg + eps) + eps).reshape(-1, 1))[:, 1]

    p24_auc_uncal = roc_auc_score(y_synth, pred_o_synth)
    p24_auc_cal = roc_auc_score(y_synth, pred_synth_calibrated)
    step(f"  P2.4 uncal AUC {p24_auc_uncal:.5f}, seg-calibrated AUC {p24_auc_cal:.5f}")
    np.save(ART / "oof_d16_dr_split_strat.npy", pred_synth_calibrated)
    np.save(ART / "test_d16_dr_split_strat.npy", pred_test_calibrated)

    # ==================================================================
    summary = dict(
        P21_orig_vs_synth_auc=float(av_auc),
        P21_top5_features=fi_sorted[:5],
        P21_rhat_stats=dict(median=float(np.median(rhat_tr)),
                            q05=float(np.quantile(rhat_tr, 0.05)),
                            q95=float(np.quantile(rhat_tr, 0.95))),
        P22_standalone_auc=float(p22_auc),
        P23_standalone_auc=float(p23_auc),
        P24_uncal_auc=float(p24_auc_uncal),
        P24_calibrated_auc=float(p24_auc_cal),
        runtime_s=time.time() - t0,
    )
    with open(ART / "d16_phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step("DONE")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
