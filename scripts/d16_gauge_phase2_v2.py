"""d16 Phase 2 v2 — density ratio r̂(x), Driver/Race excluded.

v1 (killed): used Driver+Race as classifier inputs, hit AUC 0.9985 because
synth invented 856 ghost-Driver codes absent from orig (chi-sq 383k per
Phase 1). r̂(x) was degenerate (π(x)≈1 for nearly every synth row).

v2: classifier uses only the natural-joint features (Compound + 11 numerics).
This produces a meaningful r̂(x) over the actual feature distribution we
care about.

Sub-probes (same as v1):
  P2.1  classifier orig vs synth → π(x), r̂(x), top-feature tells
  P2.2  r̂(x) as single-feature LGBM K=21+1
  P2.3  r̂-weighted orig + synth-pseudo retraining
  P2.4  r̂-median split: 2 segment-specific orig bases
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

# Excluded from classifier: Driver (887 ghost levels) and Race (host removed pre-season)
NUM = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
       "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
       "RaceProgress", "Position_Change"]
CAT_CLF = ["Compound"]   # only low-cardinality cat for the classifier
FEATS_CLF = CAT_CLF + NUM

# For orig-transfer (P2.3, P2.4) we DO use Driver/Race because LGBM treats them as
# bag-of-categories and may use them; but classifier is restricted to natural joint.
CAT_FULL = ["Driver", "Compound", "Race"]
FEATS_FULL = CAT_FULL + NUM


def align_cats(dfs, cat_cols):
    for c in cat_cols:
        union = pd.concat([d[c].astype(str) for d in dfs], axis=0)
        cats = sorted(union.dropna().unique())
        for d in dfs:
            d[c] = pd.Categorical(d[c].astype(str), categories=cats)
    return dfs


def main():
    t0 = time.time()
    log = []

    def step(msg):
        log.append(f"[{time.time() - t0:6.1f}s] {msg}")
        print(log[-1], flush=True)

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    y_synth = tr[TARGET].astype(int).values
    y_orig = orig[TARGET].astype(int).values
    n_orig, n_synth_tr, n_synth_te = len(orig), len(tr), len(te)
    step(f"orig={n_orig} tr={n_synth_tr} te={n_synth_te}")

    align_cats([tr, te, orig], CAT_FULL)

    # ==================================================================
    # P2.1  classifier orig vs synth WITHOUT Driver/Race
    # ==================================================================
    step("P2.1  orig-vs-synth classifier (Driver/Race excluded)")
    Xa = orig[FEATS_CLF].copy()
    Xb = tr[FEATS_CLF].copy()
    Xt = te[FEATS_CLF].copy()
    Xall = pd.concat([Xa, Xb], axis=0, ignore_index=True)
    yall = np.concatenate([np.zeros(n_orig), np.ones(n_synth_tr)]).astype(int)

    # single 80/20 split is fine — we only need π(x) on synth rows and test
    Xtr_clf, Xva_clf, ytr_clf, yva_clf = train_test_split(
        Xall, yall, test_size=0.2, random_state=SEED, stratify=yall)
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=4, seed=SEED)
    m = lgb.train(
        params,
        lgb.Dataset(Xtr_clf, ytr_clf, categorical_feature=CAT_CLF),
        num_boost_round=600,
        valid_sets=[lgb.Dataset(Xva_clf, yva_clf, categorical_feature=CAT_CLF)],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )
    av_auc_held = roc_auc_score(yva_clf, m.predict(Xva_clf))
    step(f"  held-out AUC: {av_auc_held:.5f}")

    # predict on synth_train + test
    pi_synth = m.predict(tr[FEATS_CLF])
    pi_test = m.predict(te[FEATS_CLF])
    pi_orig = m.predict(orig[FEATS_CLF])
    av_auc_full = roc_auc_score(np.concatenate([np.zeros(n_orig), np.ones(n_synth_tr)]),
                                  np.concatenate([pi_orig, pi_synth]))
    step(f"  full orig-vs-synth AUC: {av_auc_full:.5f}")

    importances = m.feature_importance(importance_type="gain")
    fi_sorted = sorted(zip(FEATS_CLF, importances.tolist()), key=lambda x: -x[1])
    step("  top-5 tells:")
    for name, imp in fi_sorted[:5]:
        step(f"    {name}: {imp:.0f}")

    eps = 1e-6
    rhat_tr = (pi_synth / (1 - pi_synth + eps)) * (n_orig / n_synth_tr)
    rhat_te = (pi_test / (1 - pi_test + eps)) * (n_orig / n_synth_tr)
    rhat_tr = np.clip(rhat_tr, 1e-3, 1e3)
    rhat_te = np.clip(rhat_te, 1e-3, 1e3)
    np.save(ART / "d16_rhat_synth_train.npy", rhat_tr)
    np.save(ART / "d16_rhat_synth_test.npy", rhat_te)
    step(f"  r̂_tr: med {np.median(rhat_tr):.3f} q05 {np.quantile(rhat_tr, 0.05):.3f} "
         f"q95 {np.quantile(rhat_tr, 0.95):.3f} q99 {np.quantile(rhat_tr, 0.99):.3f}")

    # ==================================================================
    # P2.2  r̂(x) as single-feature LGBM
    # ==================================================================
    step("P2.2  r̂(x) single-feature LGBM K=2 with PRIMARY")
    feat_tr = np.log1p(rhat_tr).reshape(-1, 1)
    feat_te = np.log1p(rhat_te).reshape(-1, 1)
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n_synth_tr)
    pred_te = np.zeros(n_synth_te)
    for fi, (tri, vai) in enumerate(skf2.split(feat_tr, y_synth)):
        m2 = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=15, min_data_in_leaf=500, feature_fraction=1.0,
                 verbose=-1, n_jobs=4, seed=SEED),
            lgb.Dataset(feat_tr[tri], y_synth[tri]),
            num_boost_round=200,
            valid_sets=[lgb.Dataset(feat_tr[vai], y_synth[vai])],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        oof[vai] = m2.predict(feat_tr[vai])
        pred_te += m2.predict(feat_te) / N_FOLDS
    p22_auc = roc_auc_score(y_synth, oof)
    step(f"  P2.2 standalone OOF AUC {p22_auc:.5f}")
    np.save(ART / "oof_d16_dr_rhat_strat.npy", oof)
    np.save(ART / "test_d16_dr_rhat_strat.npy", pred_te)

    # ==================================================================
    # P2.3  r̂-weighted retraining: orig (w=1) + synth-pseudo (w=r̂)
    # ==================================================================
    step("P2.3  r̂-weighted orig + synth-pseudo")
    primary_oof = np.load(ART / "oof_PRIMARY_K22_strat.npy")
    if primary_oof.ndim == 2:
        primary_oof = primary_oof[:, 1]
    pseudo = (primary_oof >= 0.5).astype(int)
    conf = np.maximum(primary_oof, 1 - primary_oof)
    keep = conf > 0.7
    step(f"  pseudo kept rows: {keep.sum()}/{n_synth_tr} (conf>0.7)")

    Xo = orig[FEATS_FULL].copy()
    yo_train = y_orig
    wo = np.ones(len(Xo))
    Xs = tr.loc[keep, FEATS_FULL].copy()
    ys_train = pseudo[keep]
    ws = rhat_tr[keep]
    Xcomb = pd.concat([Xo, Xs], ignore_index=True)
    align_cats([Xcomb, te[FEATS_FULL], tr[FEATS_FULL]], CAT_FULL)
    ycomb = np.concatenate([yo_train, ys_train])
    wcomb = np.concatenate([wo, ws])

    pred_synth_tr = np.zeros(n_synth_tr)
    pred_synth_te = np.zeros(n_synth_te)
    skf3 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []
    for fi, (tri, vai) in enumerate(skf3.split(Xcomb, ycomb)):
        m3 = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=4, seed=SEED),
            lgb.Dataset(Xcomb.iloc[tri], ycomb[tri], weight=wcomb[tri], categorical_feature=CAT_FULL),
            num_boost_round=400,
            valid_sets=[lgb.Dataset(Xcomb.iloc[vai], ycomb[vai], weight=wcomb[vai], categorical_feature=CAT_FULL)],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )
        # only count val rows that are synth (have true label) for AUC
        pred_synth_tr += m3.predict(tr[FEATS_FULL]) / N_FOLDS
        pred_synth_te += m3.predict(te[FEATS_FULL]) / N_FOLDS
        fold_aucs.append(m3.best_score["valid_0"]["auc"])
        step(f"  fold{fi} val-AUC {fold_aucs[-1]:.5f}")
    p23_auc = roc_auc_score(y_synth, pred_synth_tr)
    step(f"  P2.3 synth-train AUC {p23_auc:.5f}")
    np.save(ART / "oof_d16_dr_weighted_orig_strat.npy", pred_synth_tr)
    np.save(ART / "test_d16_dr_weighted_orig_strat.npy", pred_synth_te)

    # ==================================================================
    # P2.4  r̂-median split: orig base with per-segment Platt scaling
    # ==================================================================
    step("P2.4  r̂-median split, segment Platt scaling")
    rhat_median = float(np.median(rhat_tr))
    seg_tr = (rhat_tr >= rhat_median).astype(int)
    seg_te = (rhat_te >= rhat_median).astype(int)
    step(f"  median split r̂={rhat_median:.3f} → {(seg_tr==0).sum()} orig-like, {(seg_tr==1).sum()} hallucinated")

    Xo2 = orig[FEATS_FULL].copy()
    align_cats([Xo2, tr[FEATS_FULL], te[FEATS_FULL]], CAT_FULL)
    skf4 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred_o_synth = np.zeros(n_synth_tr)
    pred_o_test = np.zeros(n_synth_te)
    for fi, (tri, vai) in enumerate(skf4.split(Xo2, y_orig)):
        m4 = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=4, seed=SEED),
            lgb.Dataset(Xo2.iloc[tri], y_orig[tri], categorical_feature=CAT_FULL),
            num_boost_round=400,
            valid_sets=[lgb.Dataset(Xo2.iloc[vai], y_orig[vai], categorical_feature=CAT_FULL)],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )
        pred_o_synth += m4.predict(tr[FEATS_FULL]) / N_FOLDS
        pred_o_test += m4.predict(te[FEATS_FULL]) / N_FOLDS
    p24_auc_uncal = roc_auc_score(y_synth, pred_o_synth)

    from sklearn.linear_model import LogisticRegression
    pred_synth_calibrated = np.zeros(n_synth_tr)
    pred_test_calibrated = np.zeros(n_synth_te)
    for seg in [0, 1]:
        m_tr = (seg_tr == seg)
        m_te = (seg_te == seg)
        if m_tr.sum() < 100:
            pred_synth_calibrated[m_tr] = pred_o_synth[m_tr]
            pred_test_calibrated[m_te] = pred_o_test[m_te]
            continue
        lr = LogisticRegression(max_iter=200)
        x_in = np.log(np.clip(pred_o_synth[m_tr], 1e-6, 1 - 1e-6) /
                      (1 - np.clip(pred_o_synth[m_tr], 1e-6, 1 - 1e-6))).reshape(-1, 1)
        lr.fit(x_in, y_synth[m_tr])
        pred_synth_calibrated[m_tr] = lr.predict_proba(x_in)[:, 1]
        x_te = np.log(np.clip(pred_o_test[m_te], 1e-6, 1 - 1e-6) /
                      (1 - np.clip(pred_o_test[m_te], 1e-6, 1 - 1e-6))).reshape(-1, 1)
        pred_test_calibrated[m_te] = lr.predict_proba(x_te)[:, 1]
    p24_auc_cal = roc_auc_score(y_synth, pred_synth_calibrated)
    step(f"  P2.4 uncal AUC {p24_auc_uncal:.5f}, seg-calibrated AUC {p24_auc_cal:.5f}")
    np.save(ART / "oof_d16_dr_split_strat.npy", pred_synth_calibrated)
    np.save(ART / "test_d16_dr_split_strat.npy", pred_test_calibrated)

    # ==================================================================
    summary = dict(
        v="2 (Driver/Race excluded from classifier)",
        P21_orig_vs_synth_auc_held=float(av_auc_held),
        P21_orig_vs_synth_auc_full=float(av_auc_full),
        P21_top5_features=fi_sorted[:5],
        P21_rhat_stats=dict(median=float(np.median(rhat_tr)),
                             q05=float(np.quantile(rhat_tr, 0.05)),
                             q50=float(np.quantile(rhat_tr, 0.50)),
                             q95=float(np.quantile(rhat_tr, 0.95)),
                             q99=float(np.quantile(rhat_tr, 0.99))),
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
