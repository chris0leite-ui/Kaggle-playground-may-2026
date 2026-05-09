"""qI — feature importance for the remaining gap after conditional Driver/Stint.

After qH (cond Driver/Stint, sigma=0) gives disc-AUC 0.8323, what features
still discriminate?

Output: scripts/artifacts/dgp_v3_qI_remaining.json
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


def build_cond(synth: pd.DataFrame, col: str) -> dict:
    cond = {}
    for k, g in synth.groupby(["Year", "Compound", "PitStop"]):
        vals = g[col].value_counts(normalize=True)
        cond[k] = {"values": vals.index.tolist(), "p": vals.values.tolist()}
    return cond


def make_replay_conditional(
    orig: pd.DataFrame, synth_marginal: pd.DataFrame,
    synth_cond_driver: dict, synth_cond_stint: dict,
    n: int, seed: int = 0,
) -> pd.DataFrame:
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
    driver_assigned = np.empty(len(df), dtype=object)
    stint_assigned = np.empty(len(df), dtype=int)
    for k, idx in df.groupby(["Year", "Compound", "PitStop"]).groups.items():
        idx_arr = np.array(list(idx))
        if k in synth_cond_driver:
            opts = synth_cond_driver[k]
            driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
        if k in synth_cond_stint:
            opts = synth_cond_stint[k]
            stint_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
    df["Driver"] = driver_assigned
    df["Stint"] = stint_assigned
    return df


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
        .value_counts(normalize=True).reset_index(name="prob")
    )
    synth_cond_driver = build_cond(synth, "Driver")
    synth_cond_stint = build_cond(synth, "Stint")
    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)
    replay = make_replay_conditional(
        orig, synth_marginal, synth_cond_driver, synth_cond_stint,
        n=20_000, seed=0,
    )
    t(f"replay {replay.shape}", ts)

    auc_all, imp_all = disc_auc_with_importance(replay, synth_disc)
    t(f"disc all features = {auc_all:.4f}", ts)
    out["all_features"] = {
        "disc_auc": auc_all,
        "importances": dict(sorted(imp_all.items(), key=lambda kv: -kv[1])),
    }

    # Iterative-drop
    sorted_feats = sorted(imp_all, key=lambda k: -imp_all[k])
    drops = []
    out["drop_top_iter"] = []
    for feat in sorted_feats[:8]:
        drops.append(feat)
        auc, _ = disc_auc_with_importance(replay, synth_disc, drop=drops)
        out["drop_top_iter"].append({"drop": list(drops), "disc_auc": auc})
        t(f"drop {drops}: disc_auc = {auc:.4f}", ts)

    fp = ART / "dgp_v3_qI_remaining.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qI feature importance for cond replay vs synth ===")
    for feat, imp in list(out["all_features"]["importances"].items())[:12]:
        print(f"  {feat:30s} importance = {imp:.1f}")
    print(f"\n=== Iterative drop ===")
    print(f"  baseline (all features)       {auc_all:.4f}")
    for v in out["drop_top_iter"]:
        print(f"  drop {v['drop']}: disc_auc = {v['disc_auc']:.4f}")


if __name__ == "__main__":
    main()
