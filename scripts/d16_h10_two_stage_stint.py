"""Day-16 H10 — Two-stage stint α5 (E[T_stint] -> soft assignment).

α5 axis from d13 problem-decomposition tree (untouched).

Stage 1: regress E[T_stint] = expected stint length, per (Driver,Race,
  Year,Stint) group. Per-row LGBM regression with target =
  group_size_in_train (number of laps observed for this stint group).
  Inference: assign per-row a "remaining-laps-in-stint" estimate
  ~ E[T_stint] - laps_so_far_in_stint.

Stage 2: from remaining-laps estimate, derive soft P(PitNextLap=1):
  if remaining ≈ 0  -> P high
  if remaining > 1  -> P low (decays)
  We model this with a logistic shape:
    p_pit_soft = sigmoid(α + β * (-remaining_estimate))
  fit (α, β) on the train OOF rows by maximizing AUC.

Final base feeds K=21+1 LR meta as a new orthogonal base.

Output:
  oof_d16_h10_two_stage_stint_strat.npy   (n_train, 2)
  test_d16_h10_two_stage_stint_strat.npy  (n_test, 2)
  d16_h10_two_stage_stint_results.json
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
from scipy.optimize import minimize

ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
CATS = ["Driver", "Compound", "Race", "Year"]


def main():
    t0 = time.time()
    print("[h10] loading data ...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).values
    n_train, n_test = len(train), len(test)
    print(f"[h10] train {n_train}  test {n_test}", flush=True)

    # Build group key (Driver, Race, Year, Stint)
    for df in (train, test):
        df["__gkey"] = (df["Driver"].astype(str) + "|" +
                        df["Race"].astype(str) + "|" +
                        df["Year"].astype(str) + "|" +
                        df["Stint"].astype(str))

    # Stage 1 target: stint length T_stint = #rows in group
    # (only meaningful per group; same for all rows in the group).
    group_sizes_train = train.groupby("__gkey").size()
    train["__stint_len"] = train["__gkey"].map(group_sizes_train).astype(np.float32)
    group_sizes_test = test.groupby("__gkey").size()
    test["__stint_len_obs"] = test["__gkey"].map(group_sizes_test).astype(np.float32)
    print(f"[h10] train stint-len: mean {train['__stint_len'].mean():.2f} "
          f"min {train['__stint_len'].min():.0f} "
          f"max {train['__stint_len'].max():.0f}", flush=True)

    # Within-stint: laps_so_far = order within group sorted by LapNumber
    print("[h10] building laps_so_far ...", flush=True)
    train = train.sort_values(["__gkey", "LapNumber"]).reset_index(drop=False)
    train["__laps_so_far"] = train.groupby("__gkey").cumcount() + 1
    train = train.sort_values("index").drop(columns=["index"]).reset_index(drop=True)
    test = test.sort_values(["__gkey", "LapNumber"]).reset_index(drop=False)
    test["__laps_so_far"] = test.groupby("__gkey").cumcount() + 1
    test = test.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

    # Encode categoricals
    encoders = {}
    full = pd.concat([train[CATS], test[CATS]], axis=0, ignore_index=True)
    for c in CATS:
        vals = full[c].astype(str).unique().tolist()
        enc = {v: i for i, v in enumerate(vals)}
        encoders[c] = enc
    for df in (train, test):
        for c in CATS:
            df[c + "_idx"] = df[c].astype(str).map(encoders[c]).astype(np.int32)

    feat_cols = NUMERICS + [c + "_idx" for c in CATS]
    cat_idx_cols = [c + "_idx" for c in CATS]

    # Stage 1: 5-fold OOF regression for E[T_stint]
    print("[h10] Stage 1: 5-fold LGBM regression on stint-length ...", flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n_train), y))

    oof_T = np.zeros(n_train, dtype=np.float32)
    test_T = np.zeros(n_test, dtype=np.float32)
    Xtr_full = train[feat_cols]
    Xte_full = test[feat_cols]

    for fold, (tr, va) in enumerate(splits):
        Xtr = Xtr_full.iloc[tr]
        Xva = Xtr_full.iloc[va]
        Ttr = train["__stint_len"].iloc[tr].values
        Tva = train["__stint_len"].iloc[va].values
        dtr = lgb.Dataset(Xtr, Ttr, categorical_feature=cat_idx_cols)
        dva = lgb.Dataset(Xva, Tva, categorical_feature=cat_idx_cols, reference=dtr)
        params = dict(objective="regression", metric="rmse",
                      num_leaves=63, learning_rate=0.05,
                      feature_fraction=0.85, bagging_fraction=0.85,
                      bagging_freq=1, min_child_samples=200,
                      verbose=-1, seed=SEED)
        model = lgb.train(params, dtr, num_boost_round=600,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_T[va] = model.predict(Xva, num_iteration=model.best_iteration)
        test_T += model.predict(Xte_full, num_iteration=model.best_iteration) / N_FOLDS
        print(f"  fold {fold} best_iter={model.best_iteration}", flush=True)

    # Stage 2: convert (E[T_stint], laps_so_far) -> P(PitNextLap)
    # remaining = max(0, T_hat - laps_so_far)
    laps_so_far_tr = train["__laps_so_far"].values.astype(np.float32)
    laps_so_far_te = test["__laps_so_far"].values.astype(np.float32)
    remaining_tr = np.maximum(0.0, oof_T - laps_so_far_tr)
    remaining_te = np.maximum(0.0, test_T - laps_so_far_te)
    # +1 lap of "remaining" means "pit happens 1 lap from now" ≈ PitNextLap
    # We treat (remaining < 1) as "pit imminent". Fit logistic on train.
    print("[h10] Stage 2: fit logistic (α, β) on (remaining -> P_pit) ...",
          flush=True)
    feat = -remaining_tr  # higher when fewer laps remain
    def neg_auc(theta):
        a, b = theta
        z = a + b * feat
        p = 1.0 / (1.0 + np.exp(-z))
        return -float(roc_auc_score(y, p))
    # crude grid search then minimize
    best = None
    for a0 in np.linspace(-3, 3, 7):
        for b0 in np.linspace(0.1, 5, 10):
            v = neg_auc([a0, b0])
            if best is None or v < best[0]:
                best = (v, [a0, b0])
    res_opt = minimize(neg_auc, best[1], method="Nelder-Mead",
                       options=dict(xatol=1e-3, fatol=1e-5, maxiter=200))
    a_hat, b_hat = res_opt.x
    print(f"  fitted α={a_hat:.4f}  β={b_hat:.4f}  AUC={-res_opt.fun:.6f}",
          flush=True)
    z_tr = a_hat + b_hat * (-remaining_tr)
    p_tr = 1.0 / (1.0 + np.exp(-z_tr))
    z_te = a_hat + b_hat * (-remaining_te)
    p_te = 1.0 / (1.0 + np.exp(-z_te))

    auc = float(roc_auc_score(y, p_tr))
    print(f"[h10] standalone OOF AUC = {auc:.6f}", flush=True)

    # Save artifacts
    np.save(ART / "oof_d16_h10_two_stage_stint_strat.npy",
            np.column_stack([1.0 - p_tr, p_tr]))
    np.save(ART / "test_d16_h10_two_stage_stint_strat.npy",
            np.column_stack([1.0 - p_te, p_te]))
    res = dict(stage1_oof_T_rmse_proxy=float(np.std(oof_T - train["__stint_len"].values)),
               stage1_oof_T_mean=float(oof_T.mean()),
               stage2_alpha=float(a_hat),
               stage2_beta=float(b_hat),
               standalone_oof_auc=auc,
               n_train=n_train, n_test=n_test,
               wall_s=time.time() - t0)
    (ART / "d16_h10_two_stage_stint_results.json").write_text(json.dumps(res, indent=2))
    print(f"[h10] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
