"""M4 — relative-state FE + LGBM, two-anchor 5-fold (Strat & GroupKFold-Race)."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof
from m4_relstate_smoke import LAPTIME_COL, add_relstate_features, make_lgb_params

TARGET = "PitNextLap"
ID_COL = "id"
BASELINE_OOF_STRAT = 0.94075
BASELINE_OOF_GROUPKF = 0.92059


def run_anchor(name, splits, X, y, X_test, cat_cols):
    oof = np.zeros(len(y), dtype=np.float32)
    test_proba = np.zeros(len(X_test), dtype=np.float32)
    fold_scores = []
    importances = None
    for k, (tr, va) in enumerate(splits):
        dtrain = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dval = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
        model = lgb.train(make_lgb_params(), dtrain, num_boost_round=2000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        p_va = model.predict(X.iloc[va])
        oof[va] = p_va
        test_proba += model.predict(X_test) / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        fold_scores.append(s)
        if k == 0 and name == "STRAT":
            imp = model.feature_importance(importance_type="gain")
            importances = list(zip(X.columns.tolist(), imp.tolist()))
        print(f"  [{name}] fold {k}: AUC={s:.5f}  (best_iter={model.best_iteration})")
    oof_auc = float(roc_auc_score(y, oof))
    fold_std = float(np.std(fold_scores))
    return oof, test_proba, oof_auc, fold_scores, fold_std, importances


def fe_pipeline(train, test):
    train = train.copy()
    test = test.copy()
    train["__src"] = 0
    test["__src"] = 1
    test[TARGET] = -1
    full = pd.concat([train, test], axis=0, ignore_index=True)
    full_id_order = full[ID_COL].values.copy()
    fe_full, added, skipped = add_relstate_features(full)
    fe_full = fe_full.set_index(ID_COL).loc[full_id_order].reset_index()
    train_fe = fe_full[fe_full["__src"] == 0].drop(columns=["__src"]).reset_index(drop=True)
    test_fe = fe_full[fe_full["__src"] == 1].drop(columns=["__src", TARGET]).reset_index(drop=True)
    train_fe = train_fe.set_index(ID_COL).loc[train[ID_COL].values].reset_index()
    test_fe = test_fe.set_index(ID_COL).loc[test[ID_COL].values].reset_index()
    return train_fe, test_fe, added, skipped


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    print(f"loaded train={train.shape} test={test.shape}")

    train_fe, test_fe, added, skipped = fe_pipeline(train, test)
    assert (train_fe[ID_COL].values == train[ID_COL].values).all(), "train order broken"
    assert (test_fe[ID_COL].values == test[ID_COL].values).all(), "test order broken"
    print(f"FE: added={added}  skipped={skipped}  t={time.time()-t0:.1f}s")

    y = train_fe[TARGET].astype(int).values
    X = train_fe.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test_fe.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    print(f"features ({len(X.columns)}): {list(X.columns)}")

    print("=== Anchor A: StratifiedKFold(5, seed=42) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y)), y))
    t1 = time.time()
    oof_a, test_a, auc_a, folds_a, std_a, imps_a = run_anchor("STRAT", splits_a, X, y, X_test, cat_cols)
    wall_a = time.time() - t1

    print("=== Anchor B: GroupKFold(5) on Race ===")
    groups = train_fe["Race"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, groups))
    t2 = time.time()
    oof_b, test_b, auc_b, folds_b, std_b, _ = run_anchor("GROUP", splits_b, X, y, X_test, cat_cols)
    wall_b = time.time() - t2

    delta_a = (auc_a - BASELINE_OOF_STRAT) * 1e4
    delta_b = (auc_b - BASELINE_OOF_GROUPKF) * 1e4
    print()
    print(f"OOF strat:   {auc_a:.5f}  std={std_a:.5f}  Δvs baseline: {delta_a:+.1f}bp  wall={wall_a:.0f}s")
    print(f"OOF groupkf: {auc_b:.5f}  std={std_b:.5f}  Δvs baseline: {delta_b:+.1f}bp  wall={wall_b:.0f}s")

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_m4_relstate.csv", index=False)

    save_oof("m4_relstate_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=std_a, fold_scores=folds_a,
                  cv="StratifiedKFold(5, seed=42)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_a, features_added=added))
    save_oof("m4_relstate_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=std_b, fold_scores=folds_b,
                  cv="GroupKFold(5) on Race", metric="roc_auc",
                  delta_vs_baseline_bp=delta_b, features_added=added))

    # Audit md
    imps_sorted = sorted(imps_a, key=lambda x: -x[1])
    top10 = imps_sorted[:10]
    g1_strat = "PASS" if delta_a >= -5 else ("SOFT" if delta_a >= -10 else "FAIL")
    g1_groupkf = "PASS" if delta_b >= -5 else ("SOFT" if delta_b >= -10 else "FAIL")

    md = [
        f"# M4 — Relative-state FE + LGBM two-anchor ({dt.date.today()})",
        "",
        "Method: concat train+test, sort by (Race, Driver, LapNumber), compute "
        "relative-state features (skipping ones already present in host data), "
        "restore original row order, run LGBM 5-fold under two CV anchors.",
        "",
        f"Features added: {added}",
        f"Features already present (skipped): {skipped}",
        f"Final feature count: {len(X.columns)} = 14 baseline cols + 2 added",
        "",
        "## Sort/restore-order diagnostic",
        f"- assert train_fe[id] == train[id]: PASS",
        f"- assert test_fe[id] == test[id]: PASS",
        f"- OOF .npy aligned with train.csv row order (verified by id-equality before save).",
        "",
        "## Two-anchor results",
        "",
        "| anchor | OOF AUC | fold_std | per-fold | Δ vs baseline |",
        "|---|---:|---:|---|---:|",
        f"| Strat (seed=42) | **{auc_a:.5f}** | {std_a:.5f} | {[f'{x:.4f}' for x in folds_a]} | {delta_a:+.1f}bp |",
        f"| GroupKFold(Race) | **{auc_b:.5f}** | {std_b:.5f} | {[f'{x:.4f}' for x in folds_b]} | {delta_b:+.1f}bp |",
        "",
        "## G1 verdict",
        f"- Strat anchor: **{g1_strat}** (Δ={delta_a:+.1f}bp; PASS≥−5, SOFT≥−10)",
        f"- GroupKFold anchor: **{g1_groupkf}** (Δ={delta_b:+.1f}bp)",
        "",
        "## Wall times",
        f"- smoke: see scripts/m4_relstate_smoke.py output",
        f"- probe: see scripts/m4_relstate_probe.py output",
        f"- full Strat 5-fold: {wall_a:.0f}s",
        f"- full GroupKF 5-fold: {wall_b:.0f}s",
        f"- total: {time.time()-t0:.0f}s",
        "",
        "## Top 10 fold-0 Strat feature importances (gain)",
        "",
        "| rank | feature | gain |",
        "|---:|---|---:|",
    ]
    for i, (n, g) in enumerate(top10, 1):
        marker = " (NEW)" if n in added else ""
        md.append(f"| {i} | {n}{marker} | {g:.0f} |")
    md.append("")
    Path(f"audit/2026-05-04-m4-relstate-fe.md").write_text("\n".join(md))

    summary = dict(agent="S3_M4_RELSTATE",
                   auc_strat=auc_a, auc_groupkf=auc_b,
                   std_strat=std_a, std_groupkf=std_b,
                   delta_strat_bp=delta_a, delta_groupkf_bp=delta_b,
                   features_added=added, top10=top10)
    print("\nSUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
