"""scripts/probe_rain_weighted.py — rain-row sample-weighted *global* LGBM.

Distinct from probe_rain_specialist.py (a rain-only model that loses
cross-Compound transfer, -152 bp on the segment). This probe trains a
single LGBM on the FULL train set with per-row sample weights up-
weighting INTERMEDIATE / WET rows. The weighted model preserves cross-
Compound transfer while addressing the minority-class concern.

Decision rule:
  * weighted - unweighted (on rain rows) >= 2 bp -> proceed to min-meta
    gate vs PRIMARY at K=27+1.
  * < 2 bp -> log as null and close.

Sample-weight schedule: tested w in {1, 3, 5, 10}. Q6 metric-aligned
(binary, AUC). Cost: ~6 min CPU.

Outputs scripts/artifacts/probe_rain_weighted.json + console.
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
ART.mkdir(parents=True, exist_ok=True)
OUT = ART / "probe_rain_weighted.json"
PRIMARY_OOF = ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"

TARGET = "PitNextLap"
RAIN = ["INTERMEDIATE", "WET"]
SEED, N_FOLDS = 42, 5
WEIGHTS = [1.0, 3.0, 5.0, 10.0]

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbose=-1,
    n_jobs=-1,
    seed=SEED,
)

NUM_COLS = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
CAT_COLS = ["Driver", "Compound", "Race"]


def encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CAT_COLS:
        df[c + "_cat"] = df[c].astype("category").cat.codes.astype("int32")
    return df


def fold_train(X, y, feats, w_arr) -> tuple[np.ndarray, list[float]]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        ds_tr = lgb.Dataset(X.iloc[tr_idx][feats], label=y[tr_idx],
                            weight=w_arr[tr_idx])
        ds_va = lgb.Dataset(X.iloc[va_idx][feats], label=y[va_idx],
                            reference=ds_tr)
        booster = lgb.train(
            LGB_PARAMS, ds_tr, num_boost_round=2000,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        oof[va_idx] = booster.predict(X.iloc[va_idx][feats])
        fold_aucs.append(roc_auc_score(y[va_idx], oof[va_idx]))
    return oof, fold_aucs


def main() -> None:
    t0 = time.time()
    print("Loading train + PRIMARY OOF ...")
    tr = pd.read_csv("data/train.csv")
    prim = np.load(PRIMARY_OOF)[:, 1]
    rain_mask = tr["Compound"].isin(RAIN).values
    rain_idx = np.where(rain_mask)[0]
    print(f"Rain rows: {len(rain_idx):,} ({rain_mask.mean()*100:.2f}%)")

    tr_enc = encode_cats(tr)
    feats = NUM_COLS + [c + "_cat" for c in CAT_COLS]
    y = tr[TARGET].astype(int).values

    primary_rain_auc = roc_auc_score(y[rain_mask], prim[rain_mask])
    primary_global_auc = roc_auc_score(y, prim)
    print(f"  PRIMARY global AUC = {primary_global_auc:.5f}")
    print(f"  PRIMARY rain AUC   = {primary_rain_auc:.5f}")

    results: dict = {"weights": [], "primary_rain_auc": float(primary_rain_auc),
                     "primary_global_auc": float(primary_global_auc)}

    for w in WEIGHTS:
        print(f"\n=== Weight rain x {w} ===")
        w_arr = np.where(rain_mask, w, 1.0)
        oof, fold_aucs = fold_train(tr_enc, y, feats, w_arr)
        global_auc = roc_auc_score(y, oof)
        rain_auc = roc_auc_score(y[rain_mask], oof[rain_mask])
        dry_auc = roc_auc_score(y[~rain_mask], oof[~rain_mask])
        print(f"  global  OOF AUC = {global_auc:.5f}")
        print(f"  rain    OOF AUC = {rain_auc:.5f}  (PRIMARY {primary_rain_auc:.5f}; "
              f"d = {(rain_auc - primary_rain_auc) * 1e4:+.2f} bp)")
        print(f"  dry     OOF AUC = {dry_auc:.5f}")

        # Min-meta gate: K=27 PRIMARY + this single base => does it lift?
        # Use simple LR at K=2 with [PRIMARY, weighted_oof] features.
        from sklearn.linear_model import LogisticRegression
        from scipy.stats import rankdata
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        meta_oof = np.zeros(len(y))
        # expand: [raw, rank/n, logit] for each base
        Pc = np.clip(np.column_stack([prim, oof]), 1e-9, 1 - 1e-9)
        rk = np.column_stack([rankdata(c) / len(c) for c in Pc.T])
        lg = np.log(Pc / (1 - Pc))
        F = np.hstack([Pc, rk, lg])
        for tr_idx, va_idx in skf.split(F, y):
            lr = LogisticRegression(C=1.0, max_iter=500)
            lr.fit(F[tr_idx], y[tr_idx])
            meta_oof[va_idx] = lr.predict_proba(F[va_idx])[:, 1]
        meta_auc = roc_auc_score(y, meta_oof)
        d_meta_bp = (meta_auc - primary_global_auc) * 1e4
        print(f"  K=2 LR-meta AUC = {meta_auc:.5f}  d vs PRIMARY = {d_meta_bp:+.2f} bp")

        results["weights"].append({
            "w": float(w), "global_auc": float(global_auc),
            "rain_auc": float(rain_auc), "dry_auc": float(dry_auc),
            "delta_rain_vs_primary_bp": float((rain_auc - primary_rain_auc) * 1e4),
            "delta_global_vs_primary_bp": float((global_auc - primary_global_auc) * 1e4),
            "meta_K2_auc": float(meta_auc),
            "delta_meta_K2_vs_primary_bp": float(d_meta_bp),
            "fold_aucs": [float(x) for x in fold_aucs],
        })

    results["wall_s"] = time.time() - t0
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}.  Wall {results['wall_s']:.1f}s.")


if __name__ == "__main__":
    main()
