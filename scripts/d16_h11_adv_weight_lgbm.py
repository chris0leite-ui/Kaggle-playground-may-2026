"""Day-16 H11 — Adversarial-validation-weighted LGBM base.

ε-axis variant. Friction `adversarial_validation_reweight FALSIFIED`
ran AV-reweight as +e3 stack (BCE on AV-prob target as feature). Failed.

This is different: train a NEW base LGBM with sample weights set to the
AV-probability of "looks-like-test". Even though global AV-AUC=0.502,
the LOCAL sample weights re-route GBDT splits — emphasizing rows that
ARE more test-like at a feature-vector level (per-row weights, not
per-row features).

Pipeline:
  1. Build AV target: 0=train, 1=test. Concat both, train LGBM, OOF
     AV-prob predictions on train rows.
  2. Use AV-prob[train] as sample weights when training a new base
     LGBM (objective=binary, target=PitNextLap) on synth_train only.
  3. 5-fold OOF + test predictions -> stack candidate.

Output:
  oof_d16_h11_adv_weight_strat.npy   (n_train, 2)
  test_d16_h11_adv_weight_strat.npy  (n_test, 2)
  d16_h11_adv_weight_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
CATS = ["Driver", "Compound", "Race", "Year"]

LGBM_AV = dict(objective="binary", metric="auc",
               num_leaves=63, learning_rate=0.05,
               min_child_samples=200, feature_fraction=0.85,
               bagging_fraction=0.85, bagging_freq=1,
               verbose=-1, seed=SEED+7)
LGBM_BASE = dict(objective="binary", metric="auc",
                 num_leaves=63, learning_rate=0.05,
                 min_child_samples=200, feature_fraction=0.85,
                 bagging_fraction=0.85, bagging_freq=1,
                 verbose=-1, seed=SEED+11)


def main():
    t0 = time.time()
    print("[h11] loading data ...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).values
    n_train, n_test = len(train), len(test)

    encoders = {}
    full = pd.concat([train[CATS], test[CATS]], axis=0, ignore_index=True)
    for c in CATS:
        vals = full[c].astype(str).unique().tolist()
        encoders[c] = {v: i for i, v in enumerate(vals)}
    for df in (train, test):
        for c in CATS:
            df[c + "_idx"] = df[c].astype(str).map(encoders[c]).astype(np.int32)
    feat_cols = NUMERICS + [c + "_idx" for c in CATS]
    cat_idx_cols = [c + "_idx" for c in CATS]
    Xtr = train[feat_cols]
    Xte = test[feat_cols]

    # Step 1: AV classifier (train vs test)
    print("[h11] AV classifier (train vs test) ...", flush=True)
    av_tag = np.concatenate([np.zeros(n_train, np.int32),
                             np.ones(n_test, np.int32)])
    av_X = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    skf_av = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    av_splits = list(skf_av.split(np.zeros(len(av_X)), av_tag))
    av_oof = np.zeros(len(av_X), dtype=np.float32)
    av_aucs = []
    for fold, (tr, va) in enumerate(av_splits):
        dtr = lgb.Dataset(av_X.iloc[tr], av_tag[tr],
                          categorical_feature=cat_idx_cols)
        dva = lgb.Dataset(av_X.iloc[va], av_tag[va],
                          categorical_feature=cat_idx_cols, reference=dtr)
        m = lgb.train(LGBM_AV, dtr, num_boost_round=600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        av_oof[va] = m.predict(av_X.iloc[va], num_iteration=m.best_iteration)
        av_aucs.append(float(roc_auc_score(av_tag[va], av_oof[va])))
    av_auc = float(roc_auc_score(av_tag, av_oof))
    print(f"  AV OOF AUC = {av_auc:.5f}  per-fold {av_aucs}", flush=True)
    av_train_prob = av_oof[:n_train]   # P(test-like | features) for train rows

    # Step 2: train base LGBM with sample-weight = AV-prob (clipped above 0.05
    # to keep the GBDT engaged on weakly test-like rows).
    sw = np.clip(av_train_prob, 0.05, 1.0).astype(np.float32)
    print(f"[h11] sample weight stats: "
          f"mean={sw.mean():.3f} std={sw.std():.3f} "
          f"q10={np.quantile(sw, 0.1):.3f} q90={np.quantile(sw, 0.9):.3f}",
          flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n_train), y))
    oof = np.zeros(n_train, dtype=np.float32)
    test_pred = np.zeros(n_test, dtype=np.float32)
    base_aucs = []
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(Xtr.iloc[tr], y[tr], weight=sw[tr],
                          categorical_feature=cat_idx_cols)
        dva = lgb.Dataset(Xtr.iloc[va], y[va], weight=sw[va],
                          categorical_feature=cat_idx_cols, reference=dtr)
        m = lgb.train(LGBM_BASE, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        n_iter = m.best_iteration or 1500
        oof[va] = m.predict(Xtr.iloc[va], num_iteration=n_iter)
        test_pred += m.predict(Xte, num_iteration=n_iter) / N_FOLDS
        base_aucs.append(float(roc_auc_score(y[va], oof[va])))
        print(f"  base fold {fold}: best_iter={n_iter}  fold_auc={base_aucs[-1]:.5f}",
              flush=True)
    full_auc = float(roc_auc_score(y, oof))
    print(f"\n[h11] base OOF AUC = {full_auc:.6f}", flush=True)

    np.save(ART / "oof_d16_h11_adv_weight_strat.npy",
            np.column_stack([1.0 - oof, oof]))
    np.save(ART / "test_d16_h11_adv_weight_strat.npy",
            np.column_stack([1.0 - test_pred, test_pred]))
    res = dict(av_oof_auc=av_auc, av_per_fold=av_aucs,
               base_oof_auc=full_auc, base_per_fold=base_aucs,
               sw_mean=float(sw.mean()), sw_std=float(sw.std()),
               n_train=n_train, n_test=n_test,
               wall_s=time.time() - t0)
    (ART / "d16_h11_adv_weight_results.json").write_text(json.dumps(res, indent=2))
    print(f"[h11] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
