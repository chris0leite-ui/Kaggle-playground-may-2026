"""scripts/probe_exp3_inter_stint_features.py — EXP-3.

Inter-stint features. F1 strategy is sequential ACROSS stints, not just
within. Build combined-frame features for each row (Race, Driver, Year,
Stint, LapNumber):
  prev_stint_length: rows in the immediately-previous Stint of same RDY
  prev_compound: compound used in immediately-previous Stint
  prev_pit_lap_in_race: max LapNumber observed in previous Stint
  stints_completed_so_far: count of stints with Stint < current Stint
                            in same RDY group
  race_pit_count_so_far: same as stints_completed_so_far
  cur_stint_length_so_far: how many laps observed in CURRENT Stint up to
                           now (combined-frame)
  laps_since_last_pit: LapNumber - prev_pit_lap_in_race

Combined-frame, AV-safe at row level (per A3, AV-AUC=0.502).

Cost: ~25 min CPU.
Outputs scripts/artifacts/probe_exp3_inter_stint.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 500

K10_FWD = [
    "d17_h1d_yekenot_full", "p1_single_cb_v4_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]

LGB = dict(
    objective="binary", metric="auc", learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=80, feature_fraction=0.9,
    bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
)


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def build_inter_stint_features(train: pd.DataFrame,
                                test: pd.DataFrame
                                ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    n_tr = len(train)
    df = pd.concat([train.assign(_split="tr"),
                    test.assign(_split="te")],
                   ignore_index=True, sort=False)
    df["row_id"] = np.arange(len(df))
    # Per-(Race, Driver, Year, Stint) aggregates
    df = df.sort_values(["Race", "Driver", "Year", "Stint", "LapNumber"],
                        kind="stable").reset_index(drop=True)
    g_stint = df.groupby(["Race", "Driver", "Year", "Stint"], sort=False)
    df["stint_max_lap"] = g_stint["LapNumber"].transform("max")
    df["stint_min_lap"] = g_stint["LapNumber"].transform("min")
    df["stint_length"] = g_stint["LapNumber"].transform("size")
    df["stint_first_compound"] = g_stint["Compound"].transform("first")

    # Per-(Race, Driver, Year) ordered by Stint -> map prev-stint info
    df = df.sort_values(["Race", "Driver", "Year", "Stint", "LapNumber"],
                        kind="stable").reset_index(drop=True)
    # Take one row per (RDY, Stint) for prev-stint lookup
    one_per_stint = (df.groupby(["Race", "Driver", "Year", "Stint"],
                                 sort=False)
                       .agg(stint_length=("stint_length", "first"),
                            stint_max_lap=("stint_max_lap", "first"),
                            stint_first_compound=("stint_first_compound",
                                                   "first"))
                       .reset_index()
                       .sort_values(["Race", "Driver", "Year", "Stint"],
                                    kind="stable"))
    g_rdy = one_per_stint.groupby(["Race", "Driver", "Year"], sort=False)
    one_per_stint["prev_stint_length"] = g_rdy["stint_length"].shift(1)
    one_per_stint["prev_pit_lap_in_race"] = g_rdy["stint_max_lap"].shift(1)
    one_per_stint["prev_compound"] = g_rdy["stint_first_compound"].shift(1)
    one_per_stint["stints_completed_so_far"] = g_rdy.cumcount()
    # Merge back to row-level
    keys = ["Race", "Driver", "Year", "Stint"]
    df = df.merge(one_per_stint[keys + ["prev_stint_length",
                                          "prev_pit_lap_in_race",
                                          "prev_compound",
                                          "stints_completed_so_far"]],
                   on=keys, how="left")

    # Within-stint progression features (combined-frame)
    df["cur_stint_lap_idx"] = df.groupby(
        ["Race", "Driver", "Year", "Stint"], sort=False)["LapNumber"].rank("dense").astype(int) - 1
    df["laps_since_last_pit"] = (df["LapNumber"] -
                                 df["prev_pit_lap_in_race"]).fillna(-1)
    # Encode prev_compound as integer (combined-frame)
    df["prev_compound_cat"] = (
        df["prev_compound"].fillna("__none__")
        .astype("category").cat.codes.astype(int)
    )
    # Fill NaNs for first stint
    df["prev_stint_length"] = df["prev_stint_length"].fillna(-1).astype(int)
    df["prev_pit_lap_in_race"] = df["prev_pit_lap_in_race"].fillna(-1).astype(float)
    df["stints_completed_so_far"] = df["stints_completed_so_far"].astype(int)

    feats = ["prev_stint_length", "prev_pit_lap_in_race",
             "prev_compound_cat", "stints_completed_so_far",
             "cur_stint_lap_idx", "laps_since_last_pit",
             "stint_length"]
    df = df.sort_values("row_id").reset_index(drop=True)
    return df.iloc[:n_tr].reset_index(drop=True), df.iloc[n_tr:].reset_index(drop=True), feats


def main():
    t0 = time.time()
    print("Loading data + building inter-stint features ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    tr_x, te_x, feats = build_inter_stint_features(train, test)
    print(f"  built {len(feats)} inter-stint features:")
    for f in feats:
        v = tr_x[f].astype(float).values
        try:
            auc = roc_auc_score(y, v); auc = max(auc, 1 - auc)
        except Exception:
            auc = float("nan")
        print(f"    {f:>30s}  single-feat AUC: {auc:.5f}")

    # Single-LGBM 5-fold strat on inter-stint features only
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    X = tr_x[feats].astype(float).values
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        ds_tr = lgb.Dataset(X[tr], label=y[tr])
        ds_va = lgb.Dataset(X[va], label=y[va], reference=ds_tr)
        booster = lgb.train(
            LGB, ds_tr, num_boost_round=500, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        oof[va] = booster.predict(X[va])
    auc_is = float(roc_auc_score(y, oof))
    print(f"\nInter-stint-LGBM standalone OOF AUC: {auc_is:.5f}")

    # K=10+1 plain LR-meta gate
    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K10_FWD]
    F_K10 = expand(np.column_stack(base_oofs))
    F_K11 = expand(np.column_stack(base_oofs + [oof]))
    splits = list(skf.split(np.zeros(len(y)), y))

    def fit_plain(F):
        out = np.zeros(len(y))
        for tr, va in splits:
            lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            out[va] = lr.predict_proba(F[va])[:, 1]
        return float(roc_auc_score(y, out))

    auc_K10 = fit_plain(F_K10)
    auc_K11 = fit_plain(F_K11)
    delta_bp = (auc_K11 - auc_K10) * 1e4
    rho = float(spearmanr(oof, _pos(
        ART / "oof_K10_fwd_pathb_strat.npy") if (
            ART / "oof_K10_fwd_pathb_strat.npy").exists() else
        np.column_stack(base_oofs).mean(axis=1))[0])
    print(f"\nK=10 plain LR-meta:        {auc_K10:.5f}")
    print(f"K=10+inter-stint plain:    {auc_K11:.5f}  (Δ {delta_bp:+.3f} bp)")
    print(f"ρ(inter-stint, K=10 ref):  {rho:.5f}")

    if delta_bp >= 0.5:
        verdict = "PASS — 4th-direction candidate"
    elif delta_bp >= -0.1:
        verdict = "AMBIGUOUS"
    else:
        verdict = "NULL"
    print(f"Verdict: {verdict}")

    np.save(ART / "oof_exp3_inter_stint_lgbm_strat.npy", oof)
    # Also build test-set predictions for downstream stack-add use
    booster = lgb.train(
        LGB, lgb.Dataset(X, label=y), num_boost_round=200,
        callbacks=[lgb.log_evaluation(0)])
    X_test = te_x[feats].astype(float).values
    test_pred = booster.predict(X_test)
    np.save(ART / "test_exp3_inter_stint_lgbm_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))

    out = {
        "feats": feats,
        "single_feature_aucs": {
            f: float(max(roc_auc_score(y, tr_x[f].astype(float).values),
                         1 - roc_auc_score(y, tr_x[f].astype(float).values)))
            for f in feats
        },
        "inter_stint_lgbm_oof_auc": auc_is,
        "K10_plain_oof": auc_K10,
        "K10_plus_inter_stint_plain_oof": auc_K11,
        "delta_K10_plus_bp": float(delta_bp),
        "rho_inter_stint_vs_K10_ref": rho,
        "verdict": verdict,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_exp3_inter_stint.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote scripts/artifacts/probe_exp3_inter_stint.json. "
          f"Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
