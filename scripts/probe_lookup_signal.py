"""scripts/probe_lookup_signal.py — H4: PitStop[next-observation] lookup.

For each row at (Race, Driver, Year, Lap=L), find the next observation
in the COMBINED train+test frame at (Race, Driver, Year) with Lap > L.
Compute three lookup features:
  next_pitstop      — PitStop value at next observation
  next_stint_diff   — Stint[next] != Stint[this] (compound-change proxy)
  next_lap_gap      — Lap[next] - Lap[this]

Then test:
  (i)  single-feature OOF AUC of next_pitstop alone (vs PRIMARY 0.95432).
  (ii) single-LGBM with these 3 features only.
  (iii) coverage: what fraction of rows get a successor at all?

The team's R6_next_compound rule has standalone OOF 0.9444 on the same
mechanism via compound-change. This probe tests whether the DIRECT
PitStop lookup is sharper, and whether anything new survives after R6.

Cost: <5 min CPU.
Outputs scripts/artifacts/probe_lookup_signal.json.
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
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def main() -> None:
    t0 = time.time()
    print("Loading train + test ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    n_train = len(train)
    print(f"  train: {n_train:,}  test: {len(test):,}")

    # Combined frame
    train["_split"] = "tr"
    test["_split"] = "te"
    cols = ["Driver", "Race", "Year", "LapNumber", "Stint", "PitStop",
            "Compound", "_split"]
    if TARGET in train.columns:
        train_cols = cols + [TARGET]
    else:
        train_cols = cols
    df = pd.concat([train[train_cols], test[cols]], ignore_index=True,
                   sort=False)
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["Race", "Driver", "Year", "LapNumber"],
                        kind="stable").reset_index(drop=True)

    # Per-(Race, Driver, Year) group: use shift(-1) within group
    g = df.groupby(["Race", "Driver", "Year"], sort=False)
    df["next_pitstop"] = g["PitStop"].shift(-1)
    df["next_stint"] = g["Stint"].shift(-1)
    df["next_lap"] = g["LapNumber"].shift(-1)
    df["next_compound"] = g["Compound"].shift(-1)

    df["has_successor"] = df["next_lap"].notna()
    df["next_stint_diff"] = (
        df["next_stint"].astype("Float64") -
        df["Stint"].astype("Float64")
    ).fillna(0).astype(float)
    df["next_lap_gap"] = (
        df["next_lap"].astype("Float64") -
        df["LapNumber"].astype("Float64")
    ).fillna(-1).astype(float)
    df["next_compound_diff"] = (
        df["next_compound"].fillna("__none__")
        != df["Compound"]
    ).astype(int)
    df["next_pitstop_filled"] = df["next_pitstop"].fillna(0).astype(int)

    # Restore original index order
    df = df.sort_values("row_id").reset_index(drop=True)
    train_lookup = df.iloc[:n_train].reset_index(drop=True)
    test_lookup = df.iloc[n_train:].reset_index(drop=True)
    assert (train_lookup["_split"] == "tr").all()

    cov_tr = float(train_lookup["has_successor"].mean())
    cov_te = float(test_lookup["has_successor"].mean())
    print(f"\nSuccessor coverage:  train {cov_tr:.4f}   test {cov_te:.4f}")

    y = train_lookup[TARGET].astype(int).values
    print(f"  train pos rate: {y.mean():.4f}")

    # (i) next_pitstop alone as a feature: single-feature AUC
    # Restrict to rows with a successor; compare to global y
    has_s = train_lookup["has_successor"].values.astype(bool)
    npp = train_lookup["next_pitstop_filled"].values
    auc_only_w_succ = roc_auc_score(y[has_s], npp[has_s])
    auc_overall = roc_auc_score(y, npp)
    print(f"\nnext_pitstop direct AUC:")
    print(f"  on rows with successor (n={int(has_s.sum()):,}): "
          f"{auc_only_w_succ:.5f}")
    print(f"  on all train rows (filled=0):  {auc_overall:.5f}")

    # Within-stint vs cross-stint
    same_stint = has_s & (train_lookup["next_stint_diff"].values == 0)
    cross_stint = has_s & (train_lookup["next_stint_diff"].values != 0)
    auc_same = (
        roc_auc_score(y[same_stint], npp[same_stint])
        if same_stint.sum() > 100 and len(set(y[same_stint])) == 2 else None
    )
    auc_cross = (
        roc_auc_score(y[cross_stint], npp[cross_stint])
        if cross_stint.sum() > 100 and len(set(y[cross_stint])) == 2 else None
    )
    print(f"  same-stint successor (n={int(same_stint.sum()):,}): {auc_same}")
    print(f"  cross-stint successor (n={int(cross_stint.sum()):,}): {auc_cross}")

    # Pos rate by has_successor + next_pitstop
    print("\n  pos rate by (has_successor, next_pitstop):")
    for hs in [True, False]:
        for nps in [0, 1]:
            m = (has_s == hs) & (npp == nps)
            if m.sum() > 0:
                print(f"    hs={hs} nps={nps}: n={int(m.sum()):>6d}  "
                      f"pos_rate={y[m].mean():.4f}")

    # (ii) single-LGBM on the 3 lookup features
    feats = ["next_pitstop_filled", "next_stint_diff", "next_lap_gap",
             "next_compound_diff", "has_successor"]
    X_lookup = train_lookup[feats].astype(float).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    LGB_PARAMS = dict(
        objective="binary", metric="auc", learning_rate=0.05,
        num_leaves=31, min_data_in_leaf=200, feature_fraction=1.0,
        verbose=-1, n_jobs=-1, seed=SEED,
    )
    for fold, (tr, va) in enumerate(skf.split(X_lookup, y)):
        ds_tr = lgb.Dataset(X_lookup[tr], label=y[tr])
        ds_va = lgb.Dataset(X_lookup[va], label=y[va], reference=ds_tr)
        booster = lgb.train(
            LGB_PARAMS, ds_tr, num_boost_round=500, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )
        oof[va] = booster.predict(X_lookup[va])
    auc_lgb = roc_auc_score(y, oof)
    print(f"\nLGBM on 5 lookup features OOF AUC: {auc_lgb:.5f}")

    # Compare to R6_next_compound (existing pool member)
    r6 = np.load(ART / "oof_d9_R6_next_compound_strat.npy")
    r6 = r6[:, 1] if r6.ndim == 2 else r6
    auc_r6 = roc_auc_score(y, r6)
    print(f"R6 next_compound (existing) standalone AUC: {auc_r6:.5f}")

    # min-meta gate K=27 + LGBM-lookup vs K=27
    prim_oof = np.load(
        ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")[:, 1]
    auc_prim = roc_auc_score(y, prim_oof)
    print(f"\nPRIMARY (K=27 Path-B) OOF AUC: {auc_prim:.5f}")
    # K=2 LR-meta [PRIMARY, lookup_LGBM]
    from scipy.stats import rankdata
    Pc = np.clip(np.column_stack([prim_oof, oof]), 1e-9, 1 - 1e-9)
    rk = np.column_stack([rankdata(c) / len(c) for c in Pc.T])
    lg = np.log(Pc / (1 - Pc))
    F = np.hstack([Pc, rk, lg])
    from sklearn.linear_model import LogisticRegression
    meta_oof = np.zeros(len(y))
    for tr, va in skf.split(F, y):
        lr = LogisticRegression(C=1.0, max_iter=500)
        lr.fit(F[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F[va])[:, 1]
    auc_meta = roc_auc_score(y, meta_oof)
    print(f"K=2 LR-meta [PRIMARY, lookup-LGBM] OOF AUC: {auc_meta:.5f}  "
          f"(Δ vs PRIMARY: {(auc_meta - auc_prim) * 1e4:+.2f} bp)")

    # Save lookup OOF + test pred for downstream stack-add
    np.save(ART / "oof_lookup_lgbm_strat.npy", oof)
    # Test predictions: refit on full train, predict test
    booster = lgb.train(
        LGB_PARAMS, lgb.Dataset(X_lookup, label=y),
        num_boost_round=int(np.median([
            booster.best_iteration for _ in [None]] or [200])),
        callbacks=[lgb.log_evaluation(0)],
    )
    X_test_lookup = test_lookup[feats].astype(float).values
    test_pred = booster.predict(X_test_lookup)
    np.save(ART / "test_lookup_lgbm_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))

    out = {
        "coverage_train": cov_tr,
        "coverage_test": cov_te,
        "next_pitstop_auc_with_successor": float(auc_only_w_succ),
        "next_pitstop_auc_overall": float(auc_overall),
        "next_pitstop_auc_same_stint": auc_same,
        "next_pitstop_auc_cross_stint": auc_cross,
        "lookup_lgbm_oof_auc": float(auc_lgb),
        "r6_next_compound_oof_auc": float(auc_r6),
        "primary_K27_path_b_oof_auc": float(auc_prim),
        "K2_meta_with_lookup_oof_auc": float(auc_meta),
        "delta_K2_meta_vs_primary_bp":
            float((auc_meta - auc_prim) * 1e4),
        "wall_s": time.time() - t0,
    }
    (ART / "probe_lookup_signal.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote scripts/artifacts/probe_lookup_signal.json. "
          f"Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
