"""scripts/probe_rain_specialist.py — rain-condition specialist probe.

Surfaced by Day-19 Probe A: PRIMARY's residual loss concentrates entirely
in INTERMEDIATE / WET compound rows; the 8 worst (Compound x Stint x
position-in-stint) cells are all rain-condition (per-cell AUC 0.68-0.86
vs global 0.954). Total rain rows = 18,737 (4.27% of train), of which
INTERMEDIATE = 17,382 and WET = 1,355.

The diagnostic question: is the rain-row residual *intrinsic* (the
14 columns don't carry signal there) or *recoverable* (the global model
under-fits because rain rows are a minority class)?

This probe trains a single LightGBM on rain rows only with stratified
5-fold and compares its OOF AUC against the PRIMARY's OOF on the same
rows. Decision rule:

  * specialist_auc - primary_auc_on_rain >= 2 bp -> recoverable, run
    K=27+1 min-meta gate next.
  * < 2 bp -> intrinsic, mark this axis closed.

Q6 metric-aligned (binary, AUC). No label aggregates, no transductive
features. Cost: ~3 min CPU.

Outputs scripts/artifacts/probe_rain_specialist.json + console.
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
OUT = ART / "probe_rain_specialist.json"
PRIMARY_OOF = ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"

TARGET = "PitNextLap"
RAIN = ["INTERMEDIATE", "WET"]
SEED, N_FOLDS = 42, 5

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=80,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbose=-1,
    n_jobs=-1,
    seed=SEED,
)

# Same 14 raw columns the team's bases use; keep it apples-to-apples.
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


def fold_train(X: pd.DataFrame, y: np.ndarray, feats: list[str]) -> tuple[np.ndarray, list[float]]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        ds_tr = lgb.Dataset(X.iloc[tr_idx][feats], label=y[tr_idx])
        ds_va = lgb.Dataset(X.iloc[va_idx][feats], label=y[va_idx], reference=ds_tr)
        booster = lgb.train(
            LGB_PARAMS,
            ds_tr,
            num_boost_round=2000,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        oof[va_idx] = booster.predict(X.iloc[va_idx][feats])
        fold_aucs.append(roc_auc_score(y[va_idx], oof[va_idx]))
        print(f"  fold {fold}: AUC = {fold_aucs[-1]:.5f}  rounds = {booster.best_iteration}")
    return oof, fold_aucs


def main() -> None:
    t0 = time.time()
    print("Loading train + PRIMARY OOF ...")
    tr = pd.read_csv("data/train.csv")
    prim = np.load(PRIMARY_OOF)[:, 1]  # col 1 = positive class

    rain_mask = tr["Compound"].isin(RAIN).values
    rain_idx = np.where(rain_mask)[0]
    n_rain = len(rain_idx)
    print(f"Rain rows: {n_rain:,} ({n_rain / len(tr) * 100:.2f}%)")

    rain = tr.iloc[rain_idx].reset_index(drop=True)
    rain = encode_cats(rain)
    feats = NUM_COLS + [c + "_cat" for c in CAT_COLS]
    y_rain = rain[TARGET].astype(int).values

    print("\nBaseline: PRIMARY OOF on rain rows")
    primary_rain_auc = roc_auc_score(y_rain, prim[rain_idx])
    print(f"  PRIMARY OOF AUC on rain = {primary_rain_auc:.5f}")
    print(f"  PRIMARY global OOF AUC  = {roc_auc_score(tr[TARGET].astype(int).values, prim):.5f}")
    # per-Compound break-down
    for c in RAIN:
        m = (rain["Compound"] == c).values
        if m.sum() > 50 and y_rain[m].sum() > 5:
            auc_c = roc_auc_score(y_rain[m], prim[rain_idx][m])
            print(f"    {c:>12s}: n={m.sum():>5d}  pos={int(y_rain[m].sum()):>4d}  PRIMARY auc={auc_c:.4f}")

    print("\nTraining rain specialist (LGBM, 5-fold stratified) ...")
    spec_oof, fold_aucs = fold_train(rain, y_rain, feats)
    spec_auc = roc_auc_score(y_rain, spec_oof)
    print(f"\nSpecialist OOF AUC on rain = {spec_auc:.5f}")
    delta_bp = (spec_auc - primary_rain_auc) * 1e4
    print(f"Specialist - PRIMARY (rain rows only) = {delta_bp:+.2f} bp")
    for c in RAIN:
        m = (rain["Compound"] == c).values
        if m.sum() > 50 and y_rain[m].sum() > 5:
            auc_c = roc_auc_score(y_rain[m], spec_oof[m])
            d_c = (auc_c - roc_auc_score(y_rain[m], prim[rain_idx][m])) * 1e4
            print(f"    {c:>12s}: n={m.sum():>5d}  pos={int(y_rain[m].sum()):>4d}  "
                  f"specialist auc={auc_c:.4f}  vs PRIMARY {d_c:+.2f} bp")

    # Mixed prediction: rain rows = specialist, dry rows = PRIMARY.
    # Effect on the global AUC of the *mixed* prediction.
    mixed = prim.copy()
    mixed[rain_idx] = spec_oof
    global_y = tr[TARGET].astype(int).values
    global_prim_auc = roc_auc_score(global_y, prim)
    global_mixed_auc = roc_auc_score(global_y, mixed)
    delta_global_bp = (global_mixed_auc - global_prim_auc) * 1e4
    print(f"\nGlobal AUC: PRIMARY {global_prim_auc:.5f} -> "
          f"mixed (rain=spec) {global_mixed_auc:.5f}   "
          f"delta = {delta_global_bp:+.2f} bp")

    # Save specialist OOF for downstream meta-gate use if it passes.
    if delta_bp >= 0.0:
        spec_full = np.full(len(tr), np.nan)
        spec_full[rain_idx] = spec_oof
        np.save(ART / "oof_rain_specialist_strat.npy", spec_full)
        print(f"  -> saved oof_rain_specialist_strat.npy (NaN on dry rows)")

    out = {
        "n_rain": int(n_rain),
        "rain_pos_rate": float(y_rain.mean()),
        "primary_rain_auc": float(primary_rain_auc),
        "specialist_rain_auc": float(spec_auc),
        "delta_rain_bp": float(delta_bp),
        "global_primary_auc": float(global_prim_auc),
        "global_mixed_auc": float(global_mixed_auc),
        "delta_global_bp": float(delta_global_bp),
        "fold_aucs": [float(x) for x in fold_aucs],
        "wall_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}.  Wall {out['wall_s']:.1f}s.")


if __name__ == "__main__":
    main()
