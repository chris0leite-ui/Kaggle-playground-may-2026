"""scripts/d17_h1d_yekenot_full_recipe.py — H1d full yekenot recipe replication.

Faithful port of external/kernels/ps-s6-e5-realmlp-pytabkit/ cells 6 + 8 + 10.
The 6 load-bearing items missed in H1 v1/v2/v3:

  1. Arithmetic interaction features (LapNumber/RaceProgress, TyreLife/LapNumber)
  2. Floor-based numeric→categorical (every numeric col + 2 ratios)
  3. Count encoding on every categorical
  4. KBins discretization (RaceProgress=200, LapTime=7)
  5. Per-fold stratified orig concat (orig is also 5-fold split; take orig_tr_fold)
  6. CV target encoding (TargetEncoder cv=5, smooth='auto') on combo_names =
     [Race_Compound, Race_Year] inside each outer fold

Engineering choices for CPU box (yekenot used Kaggle GPU):
  - n_ens=4 instead of 24 (FE pipeline is load-bearing per ISSUES leaf 9d
    diagnosis; ensemble gain is secondary and adds ~5 bp ceiling).
  - n_threads=2 to keep the box responsive (single-job).
  - batch_size=256 (yekenot's default).

Saves:
  scripts/artifacts/oof_d17_h1d_yekenot_full_strat.npy
  scripts/artifacts/test_d17_h1d_yekenot_full_strat.npy
  scripts/artifacts/d17_h1d_yekenot_full_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

from pytabkit import RealMLP_TD_Classifier

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)

ID = "id"
TARGET = "PitNextLap"
SEED = 42
N_FOLDS = 5

# Yekenot params (cell 8 verbatim) with n_ens scaled for CPU
YEKENOT_PARAMS = dict(
    random_state=SEED,
    verbosity=1,
    val_metric_name="1-auc_ovr",
    use_early_stopping=False,
    early_stopping_additive_patience=10,
    early_stopping_multiplicative_patience=1,
    lr=0.03,
    wd=0.018,
    sq_mom=0.98,
    lr_sched="lin_cos_log_15",
    first_layer_lr_factor=0.25,
    embedding_size=6,
    max_one_hot_cat_size=18,
    hidden_sizes=[512, 256, 128],
    act="silu",
    p_drop=0.05,
    p_drop_sched="expm4t",
    plr_hidden_1=16,
    plr_hidden_2=8,
    plr_act_name="gelu",
    plr_lr_factor=0.1151,
    plr_sigma=2.33,
    ls_eps=0.01,
    ls_eps_sched="sqrt_cos",
    add_front_scale=False,
    bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot", "median_center", "robust_scale",
          "smooth_clip", "embedding", "l2_normalize"],
)

# important_combos verbatim from cell 6
IMPORTANT_COMBOS = [
    ("Race", "Compound"),
    ("Race", "Year"),
]


def feature_engineering(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
                        category_map: dict, fit: bool):
    """Verbatim port of yekenot's cell 6 feature_engineering()."""
    # 1) Arithmetic interactions
    df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
    df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")

    # 2) Floor-based numeric → cat
    for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype(str)

    # 3) Count encoding on cat_cols + Year_cat_/PitStop_cat_
    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype("int32")

    # 4) KBins: RaceProgress=200, LapTime (s)=7  (quantile, ordinal)
    bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ["quantile"]:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins, encode="ordinal",
                        strategy=strategy, subsample=None,
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype("int32")
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype(str)

    # 5) Combo cats (interaction categories with trailing underscore)
    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype(str)

    new_cat_cols = [c for c in df.columns if c.endswith("_")]
    new_num_cols = [c for c in df.columns if c.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ens", type=int, default=4, help="ensemble size (yekenot used 24 on GPU)")
    ap.add_argument("--n-epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--time-budget-min", type=float, default=85.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    params = dict(YEKENOT_PARAMS)
    params["n_ens"] = args.n_ens
    params["n_epochs"] = args.n_epochs
    params["batch_size"] = args.batch_size

    print(f"=== H1d full-recipe yekenot RealMLP ===")
    print(f"  params: n_ens={args.n_ens} n_epochs={args.n_epochs} batch={args.batch_size}")
    print(f"  budget: {args.time_budget_min:.0f} min  folds: {args.folds}")

    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    print(f"  loaded train={train.shape} test={test.shape} orig={orig.shape}  t={time.time()-t0:.1f}s")

    # Yekenot drops Normalized_TyreLife from orig (cell 6)
    orig = orig.drop(["Normalized_TyreLife"], axis=1)
    y_orig = orig[TARGET].copy()
    orig = orig.drop([TARGET], axis=1)

    X = train.drop([ID, TARGET], axis=1).copy()
    train_id = train[ID].copy()
    y = train[TARGET].copy()
    X_test = test.drop([ID], axis=1).copy()
    test_id = test[ID].copy()
    del train, test

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    print(f"  init cat_cols={len(cat_cols)} num_cols={len(num_cols)}")

    category_map: dict = {}
    X, new_cat, new_num, combo_names = feature_engineering(X, num_cols, cat_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(X_test, num_cols, cat_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(orig, num_cols, cat_cols, category_map, fit=False)
    cat_cols += new_cat
    num_cols += new_num
    print(f"  prep cat_cols={len(cat_cols)} num_cols={len(num_cols)}")
    print(f"  X={X.shape} X_test={X_test.shape} orig={orig.shape}")
    print(f"  combo_names={combo_names}")
    print(f"  feature-build done t={time.time()-t0:.1f}s")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs = []
    fold_walls = []
    folds_done = 0

    splits_train = list(skf.split(X, y))
    splits_orig = list(skf.split(orig, y_orig))

    for fold in range(args.folds):
        if (time.time() - t0) / 60 > args.time_budget_min:
            print(f"  HIT BUDGET ({args.time_budget_min:.0f} min) before fold {fold+1}; abort")
            break
        tr_idx, val_idx = splits_train[fold]
        or_tr_idx, _ = splits_orig[fold]
        print(f"\n--- Fold {fold+1}/{N_FOLDS} | t_elapsed={(time.time()-t0)/60:.1f} min ---")
        t1 = time.time()

        X_tr = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]

        # CV target encoding on combo_names (yekenot's #6 — load-bearing)
        TE = TargetEncoder(cv=N_FOLDS, smooth="auto", shuffle=True, random_state=SEED)
        tr_enc = TE.fit_transform(X_tr[combo_names], y_tr)
        val_enc = TE.transform(X_val[combo_names])
        tst_enc = TE.transform(X_test[combo_names])
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr[te_names] = tr_enc
        X_val[te_names] = val_enc
        X_tst = X_test.copy()
        X_tst[te_names] = tst_enc

        if fold == 0:
            print(f"  X_tr cols ({len(X_tr.columns)}): first 30 = {list(X_tr.columns)[:30]}")

        # Cap n_threads via env (PyTabKit honors via PyTorch threading)
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)

        model = RealMLP_TD_Classifier(**params, device="cpu", n_threads=2)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_tst)[:, 1]

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / N_FOLDS

        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)
        fold_walls.append(time.time() - t1)
        folds_done += 1
        print(f"  fold {fold+1} AUC = {fold_auc:.5f}  ({fold_walls[-1]:.0f}s)")
        del model, TE, X_tr, X_val, X_tst

    if folds_done == 0:
        print("ABORT: no folds completed")
        return

    if folds_done == N_FOLDS:
        oof_auc = float(roc_auc_score(y, oof_preds))
        print(f"\n=== summary (5/5 folds) ===")
        print(f"  Overall OOF AUC: {oof_auc:.5f}")
    else:
        # Partial: rescale test_preds (only K folds contributed)
        test_preds *= N_FOLDS / folds_done
        cov_idx = np.concatenate([splits_train[i][1] for i in range(folds_done)])
        oof_auc = float(roc_auc_score(y.iloc[cov_idx], oof_preds[cov_idx]))
        print(f"\n=== summary (partial {folds_done}/{N_FOLDS}) ===")
        print(f"  partial OOF AUC: {oof_auc:.5f}  cov={len(cov_idx)/len(y)*100:.1f}%")

    print(f"  fold_aucs: {fold_aucs}")
    print(f"  vs default-config realmlp (0.94582): {(oof_auc-0.94582)*1e4:+.1f}bp")
    print(f"  vs yekenot published (0.95273):     {(oof_auc-0.95273)*1e4:+.1f}bp")
    print(f"  total wall: {(time.time()-t0)/60:.1f} min")

    suf = args.out_suffix
    np.save(ART / f"oof_d17_h1d_yekenot_full{suf}_strat.npy",
            np.column_stack([1 - oof_preds, oof_preds]).astype(np.float32))
    np.save(ART / f"test_d17_h1d_yekenot_full{suf}_strat.npy",
            np.column_stack([1 - test_preds, test_preds]).astype(np.float32))

    res = dict(
        n_ens=args.n_ens, n_epochs=args.n_epochs, batch_size=args.batch_size,
        folds_done=folds_done, fold_aucs=fold_aucs, fold_walls_s=fold_walls,
        oof_auc=oof_auc,
        delta_vs_default_realmlp_bp=(oof_auc - 0.94582) * 1e4,
        delta_vs_yekenot_pub_bp=(oof_auc - 0.95273) * 1e4,
        total_wall_s=time.time() - t0,
        combo_names=combo_names,
    )
    (ART / f"d17_h1d_yekenot_full{suf}_results.json").write_text(json.dumps(res, indent=2))
    print(f"  saved oof/test_d17_h1d_yekenot_full{suf}_strat.npy + results.json")


if __name__ == "__main__":
    main()
