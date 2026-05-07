"""D17 H1 — Yekenot RealMLP recipe replication.

Target: replicate yekenot's published RealMLP recipe from
external/kernels/ps-s6-e5-realmlp-pytabkit/ as a NEW K=22 base candidate.
Our current `realmlp` slot was a default-config single-fold smoke
(0.94582 OOF). Yekenot's published config scores ~0.95273 OOF.

Differences from yekenot:
  - train.csv only (no orig f1_strategy_dataset_v4 merge).
  - CPU speed mode: n_ens, n_epochs, batch_size scaled down for 4-core CPU.
  - No TargetEncoder (yekenot uses one on Race_Compound, Race_Year combos).

Engineered features per task brief:
  * 11 raw numerics (LapNumber, TyreLife, Position, LapTime (s),
    LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change,
    PitStop, Year, Stint).
  * 3 raw cats: Driver, Compound, Race.
  * Year_str (str-version of Year).
  * Driver_Compound, Race_Compound, Race_Year, Driver_Race, Driver_Year,
    Compound_TyreLifeBin, Compound_RaceProgressBin, Stint_Compound.
  * OrdinalEncode all cats (handle_unknown='use_encoded_value', -1) on
    combined train+test.

Outputs (Strat seed=42 5-fold):
  scripts/artifacts/oof_d17_h1_yekenot_realmlp_strat.npy   (n_train, 2)
  scripts/artifacts/test_d17_h1_yekenot_realmlp_strat.npy  (n_test, 2)
  scripts/artifacts/d17_h1_yekenot_realmlp_results.json

Plus 80/20 honest holdout (seed=99) sanity check (Rule 24/25).

Hard wall-time cap: 90 min. If projected > cap with CPU speed mode,
fall back to n_ens=1, n_epochs=3.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED = 42
HOLDOUT_SEED = 99
N_FOLDS = 5

# CPU speed mode params (per task brief). May be downgraded if too slow.
CPU_FAST = dict(n_ens=2, n_epochs=4, batch_size=512)
CPU_FAST_FALLBACK = dict(n_ens=1, n_epochs=3, batch_size=512)
# Stronger configs (closer to yekenot's published) — only viable since
# fold-0 timing showed ~37s for n_ens=1 n_epochs=2 batch=512.
CPU_STRONG = dict(n_ens=3, n_epochs=6, batch_size=256)
CPU_STRONGER = dict(n_ens=4, n_epochs=6, batch_size=256)

YEKENOT_PARAMS_BASE = dict(
    random_state=SEED,
    verbosity=1,
    val_metric_name='1-auc_ovr',
    use_early_stopping=False,
    early_stopping_additive_patience=10,
    early_stopping_multiplicative_patience=1,
    lr=0.03,
    wd=0.018,
    sq_mom=0.98,
    lr_sched='lin_cos_log_15',
    first_layer_lr_factor=0.25,
    embedding_size=6,
    max_one_hot_cat_size=18,
    hidden_sizes=[512, 256, 128],
    act='silu',
    p_drop=0.05,
    p_drop_sched='expm4t',
    plr_hidden_1=16,
    plr_hidden_2=8,
    plr_act_name='gelu',
    plr_lr_factor=0.1151,
    plr_sigma=2.33,
    ls_eps=0.01,
    ls_eps_sched='sqrt_cos',
    add_front_scale=False,
    bias_init_mode='neg-uniform-dynamic-2',
    tfms=['one_hot', 'median_center', 'robust_scale',
          'smooth_clip', 'embedding', 'l2_normalize'],
)


# Engineered features per task brief
RAW_NUM_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
]
RAW_CAT_COLS = ["Driver", "Compound", "Race"]


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Engineer features per the task brief.
    Combined-feature transforms on train+test stack are SAFE here
    (AV-AUC=0.502, U3 audit; per Rule 25). Label-conditional
    transforms (TE) are NOT used to keep this fold-safe.
    """
    n_tr, n_te = len(train), len(test)

    df = pd.concat([train, test], axis=0, ignore_index=True)

    # Year_str
    df["Year_str"] = df["Year"].astype(int).astype(str)

    # Combo cats (string-concatenation cats)
    df["Driver_Compound"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
    df["Race_Compound"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year_str"]
    df["Driver_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
    df["Driver_Year"] = df["Driver"].astype(str) + "_" + df["Year_str"]
    df["Stint_Compound"] = df["Stint"].astype(str) + "_" + df["Compound"].astype(str)

    # KBins on TyreLife / RaceProgress (feature-only; no label) — combined-fit safe
    # Use 5 bins each per brief. Fit on the combined frame.
    for src, name in [("TyreLife", "TyreLifeBin"), ("RaceProgress", "RaceProgressBin")]:
        # Fill any NaN with median
        x = df[src].astype(float).fillna(df[src].median()).values.reshape(-1, 1)
        kb = KBinsDiscretizer(n_bins=5, encode='ordinal',
                              strategy='quantile', subsample=None)
        bins = kb.fit_transform(x).ravel().astype(int)
        df[name] = bins.astype(str)

    df["Compound_TyreLifeBin"] = df["Compound"].astype(str) + "_" + df["TyreLifeBin"]
    df["Compound_RaceProgressBin"] = df["Compound"].astype(str) + "_" + df["RaceProgressBin"]

    # Drop helper bin cols (only the cross-cats are used)
    df = df.drop(columns=["TyreLifeBin", "RaceProgressBin"])

    cat_cols_final = (RAW_CAT_COLS + ["Year_str", "Driver_Compound",
                      "Race_Compound", "Race_Year", "Driver_Race",
                      "Driver_Year", "Compound_TyreLifeBin",
                      "Compound_RaceProgressBin", "Stint_Compound"])

    # OrdinalEncode all cats on combined frame (Rule 25 transductive
    # safe at AV-AUC=0.502).
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols_final] = enc.fit_transform(df[cat_cols_final].astype(str))

    # Float types for numerics; fill NaN with column median.
    for c in RAW_NUM_COLS:
        df[c] = df[c].astype(float)
        df[c] = df[c].fillna(df[c].median())

    # Cast cats to int32 (required by some pytabkit codepaths) but
    # we'll pass them as float; pytabkit one_hot tfm will read them.
    # Per pytabkit conventions, int columns are typically treated as
    # categoricals via cat_features hint, BUT RealMLP_TD_Classifier
    # auto-detects via dtype. Use object dtype for cats to be safe.
    for c in cat_cols_final:
        df[c] = df[c].astype(int).astype('category')

    feat_cols = RAW_NUM_COLS + cat_cols_final
    X_all = df[feat_cols].copy()

    X_train = X_all.iloc[:n_tr].reset_index(drop=True)
    X_test = X_all.iloc[n_tr:].reset_index(drop=True)
    assert len(X_train) == n_tr and len(X_test) == n_te
    return X_train, X_test, feat_cols, cat_cols_final


def fit_realmlp(X_tr, y_tr, X_va, params, n_threads=2):
    from pytabkit import RealMLP_TD_Classifier
    full = dict(YEKENOT_PARAMS_BASE)
    full.update(params)
    full["device"] = "cpu"
    full["n_threads"] = n_threads
    model = RealMLP_TD_Classifier(**full)
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    return model, p_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "fallback", "strong", "stronger"], default="fast")
    ap.add_argument("--out-suffix", default="",
                    help="Append to output artifact name "
                         "(e.g. --out-suffix _strong → oof_d17_h1_yekenot_realmlp_strong_strat.npy)")
    ap.add_argument("--folds", type=int, default=5,
                    help="Number of folds to run (1..5). For wall-time tuning.")
    ap.add_argument("--skip-holdout", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=85.0,
                    help="Soft wall-time cap; abort start of new fold if exceeded.")
    args = ap.parse_args()

    if args.mode == "fast":
        cpu_params = CPU_FAST
    elif args.mode == "fallback":
        cpu_params = CPU_FAST_FALLBACK
    elif args.mode == "strong":
        cpu_params = CPU_STRONG
    elif args.mode == "stronger":
        cpu_params = CPU_STRONGER
    print(f"=== D17 H1 yekenot RealMLP ({args.mode}) ===")
    print(f"  CPU params: {cpu_params}")
    print(f"  folds: {args.folds}  budget: {args.time_budget_min:.0f} min")

    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"  loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    train_x = train.drop(columns=[TARGET, ID_COL])
    test_x = test.drop(columns=[ID_COL])

    print(f"  building features...")
    X_train, X_test, feat_cols, cat_cols = build_features(train_x, test_x)
    print(f"  feat_cols ({len(feat_cols)}): {feat_cols}")
    print(f"  cat_cols ({len(cat_cols)}): {cat_cols}")
    print(f"  X_train shape: {X_train.shape}  X_test shape: {X_test.shape}")
    print(f"  feature-build done t={time.time()-t0:.1f}s")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred_sum = np.zeros(len(test_x), dtype=np.float64)
    test_pred_n = 0
    fold_aucs = []
    fold_walls = []

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        if fold > args.folds:
            print(f"  hit --folds cap; stopping after fold {fold-1}")
            break
        if (time.time() - t0) / 60 > args.time_budget_min:
            print(f"  HIT BUDGET ({args.time_budget_min:.0f} min) before fold {fold}; abort")
            break
        print(f"\n--- Fold {fold}/{N_FOLDS} | t_elapsed={(time.time()-t0)/60:.1f} min ---")
        t1 = time.time()
        model, p_va = fit_realmlp(X_train.iloc[tr], y[tr], X_train.iloc[va],
                                  cpu_params, n_threads=2)
        oof[va] = p_va
        fold_auc = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(fold_auc)
        fold_walls.append(time.time() - t1)
        print(f"  fold {fold} AUC = {fold_auc:.5f}  ({fold_walls[-1]:.0f}s)")

        # test pred each fold (averaged)
        p_test_fold = model.predict_proba(X_test)[:, 1]
        test_pred_sum += p_test_fold
        test_pred_n += 1
        del model

    if test_pred_n == 0:
        print("ABORT: no folds completed")
        return False

    test_pred = test_pred_sum / test_pred_n
    folds_done = test_pred_n

    # If we didn't cover all rows, the OOF positions in unseen folds
    # are zero — that's bad. Compute partial OOF as cover-only.
    covered_mask = np.zeros(len(y), dtype=bool)
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fi, (_, va) in enumerate(skf2.split(np.zeros(len(y)), y), 1):
        if fi <= folds_done:
            covered_mask[va] = True
    cov_pct = covered_mask.mean() * 100
    print(f"\n=== summary (folds completed: {folds_done}/{N_FOLDS}) ===")
    print(f"  coverage: {cov_pct:.1f}%  fold_aucs: {fold_aucs}")
    if folds_done == N_FOLDS:
        oof_auc = float(roc_auc_score(y, oof))
        print(f"  full OOF AUC: {oof_auc:.5f}")
    else:
        oof_auc = float(roc_auc_score(y[covered_mask], oof[covered_mask]))
        print(f"  partial-OOF AUC ({cov_pct:.0f}% cover): {oof_auc:.5f}")

    print(f"  vs default-config realmlp (0.94582): {(oof_auc-0.94582)*1e4:+.1f}bp")
    print(f"  vs yekenot published (0.95273):     {(oof_auc-0.95273)*1e4:+.1f}bp")
    print(f"  total wall: {(time.time()-t0)/60:.1f} min")

    # Save outputs (2-col format)
    oof2 = np.column_stack([1 - oof, oof]).astype(np.float32)
    test2 = np.column_stack([1 - test_pred, test_pred]).astype(np.float32)
    suf = args.out_suffix
    np.save(ART / f"oof_d17_h1_yekenot_realmlp{suf}_strat.npy", oof2)
    np.save(ART / f"test_d17_h1_yekenot_realmlp{suf}_strat.npy", test2)

    res = dict(
        mode=args.mode,
        cpu_params=cpu_params,
        yekenot_params_base=YEKENOT_PARAMS_BASE,
        folds_done=folds_done,
        coverage_pct=cov_pct,
        fold_aucs=fold_aucs,
        fold_walls_s=fold_walls,
        oof_auc=oof_auc,
        delta_vs_default_realmlp_bp=(oof_auc - 0.94582) * 1e4,
        delta_vs_yekenot_pub_bp=(oof_auc - 0.95273) * 1e4,
        total_wall_s=time.time() - t0,
    )
    Path(ART / f"d17_h1_yekenot_realmlp{suf}_results.json").write_text(json.dumps(res, indent=2))
    print(f"  saved oof+test+results")

    # ===== 80/20 holdout (Rule 24 / Rule 25) =====
    if not args.skip_holdout and folds_done == N_FOLDS:
        # Skip if we're tight on budget; the OOF speaks louder than holdout
        budget_left = args.time_budget_min - (time.time() - t0) / 60
        if budget_left < 15:
            print(f"  budget left {budget_left:.1f} min — skip holdout")
        else:
            print(f"\n=== honest 80/20 holdout (seed={HOLDOUT_SEED}) ===")
            t_h = time.time()
            skf_h = StratifiedKFold(n_splits=5, shuffle=True, random_state=HOLDOUT_SEED)
            tr_idx, h_idx = next(iter(skf_h.split(np.zeros(len(y)), y)))
            print(f"  tr_idx: {len(tr_idx)}  h_idx: {len(h_idx)}")
            # Re-build features on the 80% slice only (test-style apply on 20%)
            train80 = train.iloc[tr_idx].reset_index(drop=True)
            train20 = train.iloc[h_idx].reset_index(drop=True)
            tr_x = train80.drop(columns=[TARGET, ID_COL])
            ho_x = train20.drop(columns=[TARGET, ID_COL])
            y80 = train80[TARGET].astype(int).values
            y20 = train20[TARGET].astype(int).values
            X80, X20, _, _ = build_features(tr_x, ho_x)
            print(f"  feature-build (holdout) t={time.time()-t_h:.1f}s")
            model, p_h = fit_realmlp(X80, y80, X20, cpu_params, n_threads=2)
            holdout_auc = float(roc_auc_score(y20, p_h))
            print(f"  HOLDOUT AUC: {holdout_auc:.5f}")
            print(f"  OOF AUC:     {oof_auc:.5f}")
            gap_bp = (oof_auc - holdout_auc) * 1e4
            print(f"  OOF − HOLDOUT: {gap_bp:+.1f} bp"
                  f"  ({'CLEAN' if abs(gap_bp) <= 10 else 'SUSPECT-LEAK' if gap_bp > 10 else 'OK'})")
            res["holdout_auc"] = holdout_auc
            res["holdout_minus_oof_bp"] = -gap_bp
            res["holdout_wall_s"] = time.time() - t_h
            Path(ART / f"d17_h1_yekenot_realmlp{suf}_results.json").write_text(json.dumps(res, indent=2))

    print(f"\nFINAL total wall: {(time.time()-t0)/60:.1f} min")
    return True


if __name__ == "__main__":
    main()
