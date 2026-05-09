"""qZ — d16++ free LB probe.

The current K=4 PRIMARY pool's `d16_orig_continuous_only` is a LightGBM
trained on orig's 7 KS-low features predicting PitNextLap, applied on
synth. ρ to PRIMARY is 0.85 (most-diverse member).

qZ: extend d16's input set with the qM cell-key features
  (Year, Compound, PitStop, Race, Stint, LapNumber)
which we now know are the host's cond-vector axes. Predicted to
slightly improve d16's standalone OOF and possibly drop ρ to PRIMARY
(more diverse).

This is the only "translates the decode insights into K=4+1 lift"
probe available without further generator training. ~5 min CPU.

Output: scripts/artifacts/dgp_v3_qZ_d16plus.json
        scripts/artifacts/dgp_v3_qZ_oof_strat.npy (OOF predictions)
        scripts/artifacts/dgp_v3_qZ_test.npy (test predictions)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main() -> None:
    out: dict = {}
    ts = time.time()

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna()
    orig = orig.reset_index(drop=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
        test = test.rename(columns={"LapTime (s)": "LapTime"})
    t(f"orig {orig.shape} train {train.shape} test {test.shape}", ts)

    # Feature set: 7 KS-low + cell key categoricals
    cont_feats = ["LapTime", "LapTime_Delta", "Cumulative_Degradation",
                  "RaceProgress", "Position", "TyreLife", "Position_Change"]
    cat_feats = ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]
    feat_cols = cont_feats + cat_feats
    out["feat_cols"] = feat_cols
    out["cont_feats"] = cont_feats
    out["cat_feats"] = cat_feats

    # Encode categoricals
    enc = {}
    for c in ["Compound", "Race"]:
        enc[c] = pd.Categorical(pd.concat([orig[c], train[c], test[c]],
                                          ignore_index=True)).categories
    def encode(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in ["Compound", "Race"]:
            df[c] = pd.Categorical(df[c], categories=enc[c]).codes
        return df

    orig_e = encode(orig)
    train_e = encode(train)
    test_e = encode(test)

    X_orig = orig_e[feat_cols].values
    y_orig = orig_e["PitNextLap"].values
    X_train_synth = train_e[feat_cols].values
    X_test_synth = test_e[feat_cols].values

    # 5-fold OOF on orig as a sanity check
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_orig = np.zeros(len(X_orig))
    test_preds = []
    train_preds = []
    for fold, (tr, va) in enumerate(skf.split(X_orig, y_orig)):
        m = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, n_jobs=-1, verbosity=-1,
        )
        m.fit(X_orig[tr], y_orig[tr])
        oof_orig[va] = m.predict_proba(X_orig[va])[:, 1]
        test_preds.append(m.predict_proba(X_test_synth)[:, 1])
        train_preds.append(m.predict_proba(X_train_synth)[:, 1])

    auc_orig = float(roc_auc_score(y_orig, oof_orig))
    t(f"orig 5-fold OOF AUC = {auc_orig:.5f}", ts)
    out["orig_5fold_oof_auc"] = auc_orig

    # Average test/train predictions across folds for application to synth
    test_pred = np.mean(test_preds, axis=0)
    train_pred = np.mean(train_preds, axis=0)

    # Sanity: how does it score on synth train?
    synth_train_auc = float(roc_auc_score(train_e["PitNextLap"].values, train_pred))
    t(f"applied to synth-train (full-fit, 5-fold avg): AUC = {synth_train_auc:.5f}", ts)
    out["synth_train_auc"] = synth_train_auc

    # Save OOF + test predictions
    np.save(ART / "dgp_v3_qZ_oof_strat.npy", oof_orig)  # this is OOF on orig
    np.save(ART / "dgp_v3_qZ_test.npy", test_pred)      # test pred for synth
    np.save(ART / "dgp_v3_qZ_train_synth.npy", train_pred)  # train-synth pred

    fp = ART / "dgp_v3_qZ_d16plus.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qZ d16++ summary ===")
    print(f"  orig 5-fold OOF AUC: {auc_orig:.5f}")
    print(f"  applied to synth-train AUC: {synth_train_auc:.5f}")
    print(f"  reference: existing d16_orig_continuous_only ρ to PRIMARY: 0.85")
    print(f"  artifacts saved: oof_strat.npy, test.npy, train_synth.npy")


if __name__ == "__main__":
    main()
