"""qG — feature importance of disc(noisy-orig sigma=0 vs synth).

The sigma=0 noisy-orig (orig sampled with synth marginal + Driver/Stint
scramble) gives disc-AUC 0.9716. Identify which features the disc uses
to detect "this is the resample, not the host."

Run a single LightGBM and print top feature importances.
Also run the disc with progressively more dropped features.
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

PERTURB_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation",
                "RaceProgress"]


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def make_replay(orig: pd.DataFrame, synth_marginal: pd.DataFrame,
                synth_drivers: list, synth_stints: list, n: int, seed: int = 0,
                scramble_driver: bool = True, scramble_stint: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = synth_marginal.sample(n, weights="prob", replace=True, random_state=seed)
    rows = []
    orig_by_cell = {k: g for k, g in orig.groupby(["Year", "Compound", "PitStop"])}
    for cell_yr, cell_cmp, cell_ps, _ in cells.itertuples(index=False):
        key = (cell_yr, cell_cmp, cell_ps)
        if key in orig_by_cell:
            sub = orig_by_cell[key]
            i = rng.integers(0, len(sub))
            rows.append(sub.iloc[i])
    df = pd.DataFrame(rows).reset_index(drop=True)

    if scramble_driver:
        df["Driver"] = rng.choice(synth_drivers, size=len(df))
    if scramble_stint:
        df["Stint"] = rng.choice(synth_stints, size=len(df))

    return df


def disc_auc_with_importance(replay: pd.DataFrame, synth: pd.DataFrame,
                              drop: list[str] = ()) -> tuple[float, dict]:
    common = sorted(set(replay.columns) & set(synth.columns))
    common = [c for c in common if c not in drop]
    df = pd.concat(
        [replay[common].assign(_lbl=0), synth[common].assign(_lbl=1)],
        ignore_index=True,
    )
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df[c] = pd.Categorical(df[c]).codes
    X = df.drop(columns=["_lbl"])
    y = df["_lbl"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    importances = np.zeros(X.shape[1])
    for tr, va in skf.split(X.values, y):
        m = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, n_jobs=-1, verbosity=-1,
        )
        m.fit(X.values[tr], y[tr])
        oof[va] = m.predict_proba(X.values[va])[:, 1]
        importances += m.feature_importances_
    auc = float(roc_auc_score(y, oof))
    importances /= 5
    imp_dict = {col: float(importances[i]) for i, col in enumerate(X.columns)}
    return auc, imp_dict


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
    synth = pd.concat(
        [train[[c for c in train.columns if c != "PitNextLap"]], test],
        ignore_index=True,
    )
    t(f"orig {orig.shape} synth {synth.shape}", ts)

    synth_marginal = (
        synth[["Year", "Compound", "PitStop"]]
        .value_counts(normalize=True)
        .reset_index(name="prob")
    )
    synth_drivers = synth["Driver"].unique().tolist()
    synth_stints = synth["Stint"].unique().tolist()

    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)
    replay_full = make_replay(orig, synth_marginal, synth_drivers, synth_stints, n=20_000, seed=0)
    t(f"replay {replay_full.shape}", ts)

    auc_all, imp_all = disc_auc_with_importance(replay_full, synth_disc)
    t(f"disc with all features = {auc_all:.4f}", ts)
    out["all_features"] = {
        "disc_auc": auc_all,
        "importances": dict(sorted(imp_all.items(), key=lambda kv: -kv[1])),
    }

    # Drop features iteratively in order of importance
    sorted_feats = sorted(imp_all, key=lambda k: -imp_all[k])
    out["drop_top_iter"] = []
    drops = []
    for i, feat in enumerate(sorted_feats[:8]):
        drops.append(feat)
        auc, _ = disc_auc_with_importance(replay_full, synth_disc, drop=drops)
        out["drop_top_iter"].append({
            "drop": list(drops),
            "disc_auc": auc,
        })
        t(f"after dropping {drops}: disc_auc = {auc:.4f}", ts)

    fp = ART / "dgp_v3_qG_remaining_gap.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== Feature importance for disc(noisy-orig sigma=0 vs synth) ===")
    for feat, imp in list(out["all_features"]["importances"].items())[:12]:
        print(f"  {feat:30s} importance = {imp:.1f}")

    print("\n=== Iterative-drop disc-AUC ===")
    print(f"  baseline (all features)       {auc_all:.4f}")
    for v in out["drop_top_iter"]:
        print(f"  drop {v['drop']}: disc_auc = {v['disc_auc']:.4f}")


if __name__ == "__main__":
    main()
