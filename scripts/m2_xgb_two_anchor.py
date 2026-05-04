"""M2 — two-anchor 5-fold XGBoost (native categorical).

Anchor A: StratifiedKFold(5, seed=42); Anchor B: GroupKFold(5) on Race.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
BASE_S, BASE_G = 0.94075, 0.92059


def make_xgb():
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        learning_rate=0.08, max_depth=6, subsample=0.9, colsample_bytree=0.9,
        min_child_weight=20, n_estimators=1000, early_stopping_rounds=80,
        enable_categorical=True, random_state=42, n_jobs=-1, verbosity=0,
    )


def run_anchor(name, splits, X, y, X_test):
    oof = np.zeros(len(y), dtype=np.float32)
    tp = np.zeros(len(X_test), dtype=np.float32)
    fs, m0 = [], None
    for k, (tr, va) in enumerate(splits):
        m = make_xgb()
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        tp += m.predict_proba(X_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], p))
        fs.append(s)
        print(f"  [{name}] fold {k}: AUC={s:.5f}  best_iter={m.best_iteration}")
        if k == 0:
            m0 = m
    return oof, tp, float(roc_auc_score(y, oof)), fs, float(np.std(fs)), m0


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    print("=== Anchor A: StratifiedKFold(5, seed=42) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y)), y))
    ta = time.time()
    oof_a, test_a, auc_a, folds_a, std_a, m0 = run_anchor(
        "STRAT", splits_a, X, y, X_test)
    secs_a = time.time() - ta

    print("=== Anchor B: GroupKFold(5) on Race ===")
    groups = train["Race"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, groups))
    tb = time.time()
    oof_b, test_b, auc_b, folds_b, std_b, _ = run_anchor(
        "GROUP", splits_b, X, y, X_test)
    secs_b = time.time() - tb

    da = (auc_a - BASE_S) * 1e4
    db = (auc_b - BASE_G) * 1e4
    print(f"\nOOF_A: {auc_a:.5f} (Δ {da:+.1f}bp)  std={std_a:.5f}")
    print(f"OOF_B: {auc_b:.5f} (Δ {db:+.1f}bp)  std={std_b:.5f}")

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_m2_xgb.csv", index=False)

    save_oof("m2_xgb_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=std_a, fold_scores=folds_a,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=da))
    save_oof("m2_xgb_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=std_b, fold_scores=folds_b,
                  cv="GroupKFold(Race)", metric="roc_auc",
                  delta_vs_baseline_bp=db))

    gain = m0.get_booster().get_score(importance_type="gain")
    feat = X.columns.tolist()
    gn = sorted(((feat[int(k[1:])] if k.startswith("f") else k, float(v))
                 for k, v in gain.items()), key=lambda x: -x[1])[:10]
    print("Top10 gain (fold0 Strat):", gn)

    g1s = "PASS" if auc_a >= BASE_S - 5e-4 else ("SOFT" if auc_a >= BASE_S - 1e-3 else "FAIL")
    g1g = "PASS" if auc_b >= BASE_G - 5e-4 else ("SOFT" if auc_b >= BASE_G - 1e-3 else "FAIL")
    total = time.time() - t0

    out = Path("audit/2026-05-04-m2-xgb.md")
    body = (
        f"# M2 — two-anchor 5-fold XGBoost (2026-05-04)\n\n"
        f"Method: XGBClassifier(hist, lr=0.05, depth=8, ss=0.9, csb=0.9, "
        f"min_child=20, n_est=2000, ES=100, native categorical).\n\n"
        f"## Two-anchor results\n\n"
        f"| anchor | OOF AUC | fold_std | per-fold | Δ baseline (bp) | G1 |\n"
        f"|---|---:|---:|---|---:|---|\n"
        f"| StratKFold | **{auc_a:.5f}** | {std_a:.5f} | "
        f"{[f'{x:.4f}' for x in folds_a]} | {da:+.1f} | {g1s} |\n"
        f"| GroupKFold(Race) | **{auc_b:.5f}** | {std_b:.5f} | "
        f"{[f'{x:.4f}' for x in folds_b]} | {db:+.1f} | {g1g} |\n\n"
        f"Baseline: Strat {BASE_S}, Group {BASE_G}. "
        f"G1 PASS if ≥ baseline−5bp; soft floor −10bp.\n\n"
        f"## Wall times\n\n"
        f"- Smoke (1 fold, 50k rows, ES=50): ~3s\n"
        f"- Probe (1 fold full data, full hp): 48s; projection 481s\n"
        f"- Anchor A: {secs_a:.0f}s; Anchor B: {secs_b:.0f}s; "
        f"Total: {total:.0f}s\n\n"
        f"## Top-10 gain importances (fold0 StratKFold)\n\n"
        f"| feature | gain |\n|---|---:|\n"
        + "".join(f"| {n} | {v:.1f} |\n" for n, v in gn)
    )
    out.write_text(body)
    print(f"\n→ {out}  total={total:.0f}s")

    summary = dict(
        agent="S1_M2_XGB", auc_strat=round(auc_a, 5),
        auc_groupkf=round(auc_b, 5), std_strat=round(std_a, 5),
        std_groupkf=round(std_b, 5), secs_strat=round(secs_a, 1),
        secs_groupkf=round(secs_b, 1),
        g1_pass_strat=(g1s in ("PASS", "SOFT")),
        g1_pass_groupkf=(g1g in ("PASS", "SOFT")),
    )
    print("SUMMARY=" + json.dumps(summary))


if __name__ == "__main__":
    main()
