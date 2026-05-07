"""d16 Phase 4 — orig-transfer feature-subset diversification.

Per friction tag `external-data-arch-bag-redundant-when-shared-training-data`:
varying architecture on shared orig training-data gives diminishing returns.
Vary FEATURE SUBSET instead. 4 LGBM variants on the original:

  P4.1  no_laptime          drop {LapTime (s), LapTime_Delta}
  P4.2  no_tyrelife_rp      drop {TyreLife, RaceProgress}
  P4.3  categorical_only    use {Driver, Race, Year, Compound, Stint, PitStop}
  P4.4  continuous_only     use {LapTime, LapTime_Delta, TyreLife, RaceProgress, Cumulative_Degradation, Position}

Outputs (4 OOF/test pairs):
  scripts/artifacts/oof_d16_orig_no_laptime_strat.npy / test_*
  scripts/artifacts/oof_d16_orig_no_tyrelife_rp_strat.npy / test_*
  scripts/artifacts/oof_d16_orig_categorical_only_strat.npy / test_*
  scripts/artifacts/oof_d16_orig_continuous_only_strat.npy / test_*
  scripts/artifacts/d16_phase4_summary.json   — AUCs, cross-rho matrix
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET = "PitNextLap"

CAT_ALL = ["Driver", "Compound", "Race"]
NUM_ALL = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
           "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
           "RaceProgress", "Position_Change"]

VARIANTS = {
    "no_laptime":       [c for c in CAT_ALL] + [c for c in NUM_ALL if c not in {"LapTime (s)", "LapTime_Delta"}],
    "no_tyrelife_rp":   [c for c in CAT_ALL] + [c for c in NUM_ALL if c not in {"TyreLife", "RaceProgress"}],
    "categorical_only": ["Driver", "Race", "Compound"] + ["Year", "Stint", "PitStop"],
    "continuous_only":  ["LapTime (s)", "LapTime_Delta", "TyreLife", "RaceProgress",
                         "Cumulative_Degradation", "Position", "LapNumber"],
}


def main():
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}")

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    y_orig = orig[TARGET].astype(int).values
    y_synth = tr[TARGET].astype(int).values

    # cat alignment
    for c in CAT_ALL:
        union = pd.concat([tr[c], te[c], orig[c]], axis=0).astype(str)
        cats = sorted(union.unique())
        for d in [tr, te, orig]:
            d[c] = pd.Categorical(d[c].astype(str), categories=cats)

    out_oof, out_te = {}, {}
    summary = {}

    params = dict(objective="binary", metric="auc", learning_rate=0.04,
                  num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED)

    for vname, feats in VARIANTS.items():
        step(f"variant {vname}: feats={feats}")
        cat_in = [c for c in CAT_ALL if c in feats]
        Xo = orig[feats].copy()
        Xs = tr[feats].copy()
        Xt = te[feats].copy()

        # train on orig (held-out 20%), predict on synth_train + synth_test
        from sklearn.model_selection import train_test_split
        Xtr, Xva, ytr, yva = train_test_split(Xo, y_orig, test_size=0.2, random_state=SEED, stratify=y_orig)
        m = lgb.train(
            params,
            lgb.Dataset(Xtr, ytr, categorical_feature=cat_in),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(Xva, yva, categorical_feature=cat_in)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        held_auc = roc_auc_score(yva, m.predict(Xva))
        oof_synth = m.predict(Xs)  # this is the orig-trained prediction on each synth-train row
        pred_te = m.predict(Xt)
        synth_auc = roc_auc_score(y_synth, oof_synth)

        np.save(ART / f"oof_d16_orig_{vname}_strat.npy", oof_synth)
        np.save(ART / f"test_d16_orig_{vname}_strat.npy", pred_te)
        out_oof[vname] = oof_synth
        out_te[vname] = pred_te
        summary[vname] = dict(orig_held_auc=float(held_auc),
                               synth_train_auc=float(synth_auc),
                               n_features=len(feats))
        step(f"  {vname}: orig-held AUC {held_auc:.5f}, synth-train AUC {synth_auc:.5f}")

    # cross-rho matrix between variants and base orig_transfer + PRIMARY
    step("computing cross-rho matrix")
    base_orig = np.load(ART / "test_d15_orig_transfer_strat.npy")
    primary_test = np.load(ART / "test_PRIMARY_K22_strat.npy")
    if primary_test.ndim == 2:
        primary_test = primary_test[:, 1]
    if base_orig.ndim == 2:
        base_orig = base_orig[:, 1]

    matrix_names = list(VARIANTS.keys()) + ["base_orig_transfer", "PRIMARY"]
    cols = [out_te[v] for v in VARIANTS] + [base_orig, primary_test]
    rho_mat = np.corrcoef(np.array(cols))
    summary["cross_rho_test"] = {n1: {n2: float(rho_mat[i, j]) for j, n2 in enumerate(matrix_names)}
                                  for i, n1 in enumerate(matrix_names)}

    summary["runtime_s"] = time.time() - t0
    with open(ART / "d16_phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step("DONE")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
