"""kNN classifier on small feature subsets — random-subspace bagging.

PI prompt: "knn for the classification, many subsets of features, max
5 each."

10 hand-picked subsets, each with at most 5 features, drawn to span
the natural physical groupings of the feature schema:

  s1  TyreLife, LapNumber, Stint                            — top-3 EDA signals
  s2  + RaceProgress, Cumulative_Degradation                — top-5 numeric
  s3  Compound_LE, TyreLife, Stint, LapNumber, Position     — Compound-anchored
  s4  Race_freq, LapNumber, Position, RaceProgress          — Race-anchored
  s5  Driver_freq, TyreLife, LapNumber, Stint, Position     — Driver-anchored
  s6  Year, Compound_LE, TyreLife, Stint, RaceProgress      — multi-categorical
  s7  LapTime, LapTime_Delta, Cumulative_Degradation, ...   — lap-timing physics
  s8  PitStop, LapNumber, TyreLife, Stint                   — pit-history focus
  s9  TyreLife, Stint                                       — minimal 2-feat
  s10 TyreLife, Stint, RaceProgress, Compound_LE, LapNumber — strong-recipe-5

Each subset: standardised features, kNN with K=50, distance-weighted.
Output: oof_knn_<NAME>_strat.npy and test_knn_<NAME>_strat.npy.

Cost: ~2–4 min per subset on 350k train rows (KDTree at d≤5 is fast).

Pooling: at the end, fit a thin LR-meta over the 10 subset OOFs as a
single bagged-kNN base candidate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import N_FOLDS, SEED, folds, save_oof

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"


SUBSETS = {
    "s1_top3":         ["TyreLife", "LapNumber", "Stint"],
    "s2_top5_num":     ["TyreLife", "LapNumber", "Stint",
                        "RaceProgress", "Cumulative_Degradation"],
    "s3_cmp":          ["Compound_LE", "TyreLife", "Stint", "LapNumber",
                        "Position"],
    "s4_race":         ["Race_freq", "LapNumber", "Position", "RaceProgress"],
    "s5_drv":          ["Driver_freq", "TyreLife", "LapNumber", "Stint",
                        "Position"],
    "s6_multicat":     ["Year", "Compound_LE", "TyreLife", "Stint",
                        "RaceProgress"],
    "s7_laptime":      ["LapTime (s)", "LapTime_Delta",
                        "Cumulative_Degradation", "Position_Change",
                        "TyreLife"],
    "s8_pit":          ["PitStop", "LapNumber", "TyreLife", "Stint"],
    "s9_minimal":      ["TyreLife", "Stint"],
    "s10_recipe5":     ["TyreLife", "Stint", "RaceProgress", "Compound_LE",
                        "LapNumber"],
}


def build_feature_pool(train: pd.DataFrame, test: pd.DataFrame
                       ) -> pd.DataFrame:
    """Return a single dataframe with all candidate columns (numeric +
    encoded categoricals), shared between train and test rows. Train rows
    come first. Encodings are leak-safe (frequencies fit on train only).
    """
    n_tr = len(train)
    df = pd.concat([train.assign(__split="tr"),
                    test.assign(__split="te")], ignore_index=True)

    out = pd.DataFrame(index=df.index)
    # Pure numeric
    num_cols = [c for c in train.columns
                if c not in ["Driver", "Compound", "Race", TARGET, "id"]
                and pd.api.types.is_numeric_dtype(train[c])]
    for c in num_cols:
        out[c] = df[c].astype(np.float64).values

    # Compound: label-encode (5 levels, ordering preserved arbitrarily)
    comp_levels = sorted(df["Compound"].astype(str).unique())
    comp_map = {v: i for i, v in enumerate(comp_levels)}
    out["Compound_LE"] = df["Compound"].astype(str).map(comp_map).astype(int)

    # Race: frequency-encoded from train rows only
    race_counts = train["Race"].value_counts()
    out["Race_freq"] = df["Race"].map(race_counts).fillna(0).astype(np.float64)

    # Driver: frequency-encoded from train rows only
    drv_counts = train["Driver"].value_counts()
    out["Driver_freq"] = df["Driver"].map(drv_counts).fillna(0).astype(np.float64)

    return out, n_tr


def run_subset(name: str, cols: list[str], pool: pd.DataFrame, n_tr: int,
               y: np.ndarray, *, k: int = 50, weights: str = "distance"):
    """5-fold OOF + averaged test predictions for one feature subset."""
    print(f"\n=== {name}: {cols} ===")
    X_full = pool[cols].values.astype(np.float32)
    X_train = X_full[:n_tr]
    X_test = X_full[n_tr:]

    oof = np.zeros(n_tr, dtype=np.float32)
    test_proba = np.zeros(len(X_test), dtype=np.float32)
    fold_aucs, fold_secs = [], []

    for kf, tr, va in folds(y, task="classification"):
        t0 = time.time()
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train[tr]).astype(np.float32)
        X_va_s = sc.transform(X_train[va]).astype(np.float32)
        X_te_s = sc.transform(X_test).astype(np.float32)

        clf = KNeighborsClassifier(
            n_neighbors=k, weights=weights, algorithm="auto",
            leaf_size=40, n_jobs=-1,
        )
        clf.fit(X_tr_s, y[tr])
        p_va = clf.predict_proba(X_va_s)[:, 1]
        p_te = clf.predict_proba(X_te_s)[:, 1]

        oof[va] = p_va
        test_proba += p_te.astype(np.float32) / N_FOLDS

        secs = time.time() - t0
        auc = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(auc)
        fold_secs.append(secs)
        print(f"   fold {kf}: AUC={auc:.5f}  ({secs:.1f}s)")

    oof_full = float(roc_auc_score(y, oof))
    print(f"   full OOF: {oof_full:.5f}  total {sum(fold_secs):.0f}s")
    save_oof(f"knn_{name}",
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(subset=name, cols=cols, k=k, weights=weights,
                  oof_score=oof_full, fold_aucs=fold_aucs,
                  fold_secs=fold_secs))
    return oof, test_proba, oof_full


def lr_pool_meta(oofs: dict, tests: dict, y: np.ndarray, splits) -> tuple:
    """Pool subset OOFs through a thin LR meta-stacker."""
    names = sorted(oofs)
    P_oof = np.column_stack([oofs[n] for n in names])
    P_test = np.column_stack([tests[n] for n in names])
    oof = np.zeros(len(y))
    test = np.zeros(P_test.shape[0])
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(P_oof[tr], y[tr])
        oof[va] = lr.predict_proba(P_oof[va])[:, 1]
        test += lr.predict_proba(P_test)[:, 1] / len(splits)
    return oof, test, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", type=str, default="all",
                    help="Comma-separated subset names, or 'all'.")
    ap.add_argument("--k", type=int, default=50,
                    help="kNN neighbour count.")
    ap.add_argument("--weights", default="distance",
                    choices=["uniform", "distance"])
    args = ap.parse_args()

    print("Loading data ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    pool, n_tr = build_feature_pool(train, test)
    print(f"  train: {len(train):,}  test: {len(test):,}  "
          f"pool cols: {pool.shape[1]}")

    # Validate subset columns exist
    for nm, cols in SUBSETS.items():
        missing = [c for c in cols if c not in pool.columns]
        if missing:
            raise SystemExit(f"subset {nm}: missing cols {missing}")

    target_subsets = list(SUBSETS) if args.subsets == "all" \
        else args.subsets.split(",")

    oofs, tests, scores = {}, {}, {}
    t0 = time.time()
    for nm in target_subsets:
        oof, test_p, score = run_subset(nm, SUBSETS[nm], pool, n_tr, y,
                                        k=args.k, weights=args.weights)
        oofs[nm] = oof
        tests[nm] = test_p.astype(np.float64)
        scores[nm] = score
    print(f"\n=== per-subset summary ===")
    for nm, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"   {nm:<14s} OOF={s:.5f}   cols={SUBSETS[nm]}")

    # Pool the subset OOFs through an LR meta
    if len(oofs) >= 2:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(skf.split(np.zeros(len(y)), y))
        print("\n=== LR-pool over subset OOFs ===")
        oof_pool, test_pool, names = lr_pool_meta(oofs, tests, y, splits)
        auc_pool = float(roc_auc_score(y, oof_pool))
        print(f"   pooled OOF: {auc_pool:.5f}  ({len(names)} subsets)")
        save_oof("knn_pool_lrmeta",
                 np.column_stack([1 - oof_pool, oof_pool]),
                 np.column_stack([1 - test_pool, test_pool]),
                 dict(variant="knn_pool_lrmeta", n_subsets=len(names),
                      subsets=names, oof_score=auc_pool))
    print(f"\ntotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
