"""d17 Phase 0 — leakage cleanup for d16 dr_split + dr_weighted_orig.

dr_split v1 leak: per-segment Platt scaling fit on y_synth[seg_mask] then
predicts on the SAME rows. Fix: 5-fold the Platt step.

dr_weighted_orig v1 leak: synth pseudo-labels added to training set;
predictions over synth_train include rows that were in 4/5 fold training
sets. Fix: source-stratified 5-fold (fold split honors orig vs synth_pseudo
sources separately so synth_pseudo rows in val are never used in training).

Output:
  oof_d17_dr_split_v2_strat.npy / test_*
  oof_d17_dr_weighted_orig_v2_strat.npy / test_*
  d17_phase0_leakage_summary.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"
EPS = 1e-6

CAT = ["Driver", "Compound", "Race"]
NUM = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
       "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
       "RaceProgress", "Position_Change"]
FEATS = CAT + NUM


def align_cats(dfs):
    for c in CAT:
        union = pd.concat([d[c].astype(str) for d in dfs], axis=0)
        cats = sorted(union.dropna().unique())
        for d in dfs:
            d[c] = pd.Categorical(d[c].astype(str), categories=cats)


def main():
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    y_synth = tr[TARGET].astype(int).values
    y_orig = orig[TARGET].astype(int).values
    n_orig = len(orig)
    n_synth = len(tr)
    align_cats([tr, te, orig])

    rhat_tr = np.load(ART / "d16_rhat_synth_train.npy")
    rhat_te = np.load(ART / "d16_rhat_synth_test.npy")
    step(f"  rhat loaded; median {np.median(rhat_tr):.3f}")

    summary = {}

    # ==================================================================
    # L1. dr_split v2 — 5-fold Platt (per fold: fit on tri, predict on vai)
    # ==================================================================
    step("L1  dr_split v2 (proper 5-fold Platt)")

    # Train base orig-LGBM 5-fold on orig (same as v1)
    skf_orig = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    Xo = orig[FEATS].copy()
    pred_o_synth = np.zeros(n_synth)
    pred_o_test = np.zeros(len(te))
    for fi, (tri, vai) in enumerate(skf_orig.split(Xo, y_orig)):
        m = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=4, seed=SEED),
            lgb.Dataset(Xo.iloc[tri], y_orig[tri], categorical_feature=CAT),
            num_boost_round=400,
            valid_sets=[lgb.Dataset(Xo.iloc[vai], y_orig[vai], categorical_feature=CAT)],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )
        pred_o_synth += m.predict(tr[FEATS]) / N_FOLDS
        pred_o_test += m.predict(te[FEATS]) / N_FOLDS

    rhat_median = float(np.median(rhat_tr))
    seg_tr = (rhat_tr >= rhat_median).astype(int)
    seg_te = (rhat_te >= rhat_median).astype(int)

    # PROPER 5-fold Platt: split synth_train into 5 folds; for each fold,
    # fit Platt PER SEGMENT on the 4 training folds, predict on val fold.
    # For test: refit Platt on ALL synth_train per segment.
    skf_platt = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1)
    pred_synth_calibrated = np.zeros(n_synth)
    pred_test_calibrated = np.zeros(len(te))

    for fi, (tri, vai) in enumerate(skf_platt.split(np.zeros(n_synth), y_synth)):
        for seg in [0, 1]:
            tri_seg = tri[seg_tr[tri] == seg]
            vai_seg = vai[seg_tr[vai] == seg]
            if len(tri_seg) < 100 or len(vai_seg) == 0:
                pred_synth_calibrated[vai_seg] = pred_o_synth[vai_seg]
                continue
            x_tr_logit = np.log(np.clip(pred_o_synth[tri_seg], EPS, 1 - EPS) /
                                 (1 - np.clip(pred_o_synth[tri_seg], EPS, 1 - EPS)))
            lr = LogisticRegression(max_iter=200)
            lr.fit(x_tr_logit.reshape(-1, 1), y_synth[tri_seg])
            x_va_logit = np.log(np.clip(pred_o_synth[vai_seg], EPS, 1 - EPS) /
                                 (1 - np.clip(pred_o_synth[vai_seg], EPS, 1 - EPS)))
            pred_synth_calibrated[vai_seg] = lr.predict_proba(x_va_logit.reshape(-1, 1))[:, 1]

    # Test: refit Platt on full synth_train per segment (uses all labels but only for
    # test prediction, no leak into OOF).
    for seg in [0, 1]:
        m_tr = (seg_tr == seg)
        m_te = (seg_te == seg)
        if m_tr.sum() < 100:
            pred_test_calibrated[m_te] = pred_o_test[m_te]
            continue
        x_tr_logit = np.log(np.clip(pred_o_synth[m_tr], EPS, 1 - EPS) /
                             (1 - np.clip(pred_o_synth[m_tr], EPS, 1 - EPS)))
        lr = LogisticRegression(max_iter=200)
        lr.fit(x_tr_logit.reshape(-1, 1), y_synth[m_tr])
        x_te_logit = np.log(np.clip(pred_o_test[m_te], EPS, 1 - EPS) /
                             (1 - np.clip(pred_o_test[m_te], EPS, 1 - EPS)))
        pred_test_calibrated[m_te] = lr.predict_proba(x_te_logit.reshape(-1, 1))[:, 1]

    auc_v2 = roc_auc_score(y_synth, pred_synth_calibrated)
    auc_v1 = roc_auc_score(y_synth, np.load(ART / "oof_d16_dr_split_strat.npy"))
    np.save(ART / "oof_d17_dr_split_v2_strat.npy", pred_synth_calibrated)
    np.save(ART / "test_d17_dr_split_v2_strat.npy", pred_test_calibrated)
    step(f"  v1 OOF AUC {auc_v1:.5f}  vs v2 OOF AUC {auc_v2:.5f}  Δ {(auc_v2 - auc_v1) * 1e4:+.2f} bp")
    summary["L1_dr_split"] = dict(v1_auc=float(auc_v1), v2_auc=float(auc_v2),
                                    inflation_bp=float((auc_v1 - auc_v2) * 1e4))

    # ==================================================================
    # L2. dr_weighted_orig v2 — source-stratified folds
    # ==================================================================
    step("L2  dr_weighted_orig v2 (source-stratified folds)")
    primary_oof = np.load(ART / "oof_PRIMARY_K22_strat.npy")
    if primary_oof.ndim == 2:
        primary_oof = primary_oof[:, 1]
    pseudo = (primary_oof >= 0.5).astype(int)
    conf = np.maximum(primary_oof, 1 - primary_oof)
    keep = conf > 0.7  # boolean array length n_synth
    keep_idx = np.where(keep)[0]
    n_kept = len(keep_idx)
    step(f"  pseudo kept rows: {n_kept}/{n_synth}")

    # Split synth_train into 5 folds (over ALL synth_train, not just kept).
    # For each fold k:
    #   train_set = orig_full + synth_pseudo[keep AND fold != k]
    #   val_set   = synth_train rows in fold k (any keep status)
    #   model trained on train_set, predicts on val_set.
    # This guarantees no synth row in val_k is in the training set of fold k,
    # closing the leak from v1.
    skf_synth = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 2)
    pred_synth = np.zeros(n_synth)
    pred_test = np.zeros(len(te))
    Xo_full = orig[FEATS].copy()
    yo_full = y_orig

    for fi, (tri, vai) in enumerate(skf_synth.split(np.zeros(n_synth), y_synth)):
        # synth rows in train fold AND high-conf-pseudo
        synth_tri_keep = np.intersect1d(tri, keep_idx)
        Xs_tri = tr.iloc[synth_tri_keep][FEATS].copy()
        ys_tri = pseudo[synth_tri_keep]
        ws_tri = rhat_tr[synth_tri_keep]
        wo_tri = np.ones(n_orig)
        Xcomb = pd.concat([Xo_full, Xs_tri], ignore_index=True)
        align_cats([Xcomb, tr[FEATS], te[FEATS]])
        ycomb = np.concatenate([yo_full, ys_tri])
        wcomb = np.concatenate([wo_tri, ws_tri])

        m = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=4, seed=SEED),
            lgb.Dataset(Xcomb, ycomb, weight=wcomb, categorical_feature=CAT),
            num_boost_round=300,
        )
        # predict on val rows of synth_train (fold k)
        pred_synth[vai] = m.predict(tr.iloc[vai][FEATS])
        pred_test += m.predict(te[FEATS]) / N_FOLDS
        step(f"  fold {fi} done, val n={len(vai)}, train_synth_kept n={len(synth_tri_keep)}")

    auc_v2_w = roc_auc_score(y_synth, pred_synth)
    auc_v1_w = roc_auc_score(y_synth, np.load(ART / "oof_d16_dr_weighted_orig_strat.npy"))
    np.save(ART / "oof_d17_dr_weighted_orig_v2_strat.npy", pred_synth)
    np.save(ART / "test_d17_dr_weighted_orig_v2_strat.npy", pred_test)
    step(f"  v1 OOF AUC {auc_v1_w:.5f}  vs v2 OOF AUC {auc_v2_w:.5f}  Δ {(auc_v2_w - auc_v1_w) * 1e4:+.2f} bp")
    summary["L2_dr_weighted_orig"] = dict(v1_auc=float(auc_v1_w), v2_auc=float(auc_v2_w),
                                            inflation_bp=float((auc_v1_w - auc_v2_w) * 1e4))

    summary["runtime_s"] = time.time() - t0
    with open(ART / "d17_phase0_leakage_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step("DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
