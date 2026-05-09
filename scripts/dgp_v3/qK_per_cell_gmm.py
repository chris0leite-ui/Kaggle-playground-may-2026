"""qK — per-cell BGMM on continuous columns + cond categorical.

Hypothesis: host's continuous columns are sampled from a per-cell density
(BGMM-fitted from orig) rather than literally copied. Test by:

  1. For each (Year, Compound, PitStop) cell, fit BGMM(K=3) on the
     6 continuous columns of orig.
  2. Sample N rows from the cell BGMM.
  3. Add cell metadata (Year, Compound, PitStop, Driver, Stint, Race,
     LapNumber, Position).
  4. Sample Driver/Stint conditional from synth empirical.
  5. Sample Race/Position/LapNumber per-cell from synth empirical too.

This tests if synth = "per-cell density estimator + categorical sampling".

Output: scripts/artifacts/dgp_v3_qK_gmm.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"

CONT_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "TyreLife", "Position_Change"]


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def disc_auc(replay: pd.DataFrame, synth: pd.DataFrame) -> float:
    common = sorted(set(replay.columns) & set(synth.columns))
    df = pd.concat(
        [replay[common].assign(_lbl=0), synth[common].assign(_lbl=1)],
        ignore_index=True,
    )
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df[c] = pd.Categorical(df[c]).codes
    X = df.drop(columns=["_lbl"]).values
    y = df["_lbl"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, n_jobs=-1, verbosity=-1,
        )
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, oof))


def build_cond(synth: pd.DataFrame, col: str, by: list[str]) -> dict:
    cond = {}
    for k, g in synth.groupby(by):
        vals = g[col].value_counts(normalize=True)
        cond[k] = {"values": vals.index.tolist(), "p": vals.values.tolist()}
    return cond


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

    # Marginal cells from synth
    synth_marginal = (
        synth[["Year", "Compound", "PitStop"]]
        .value_counts(normalize=True).reset_index(name="prob")
    )

    # Build per-cell conditional distributions for categorical cols (from synth)
    cond_cols = ["Driver", "Stint", "Race", "Position", "LapNumber"]
    conds = {c: build_cond(synth, c, ["Year", "Compound", "PitStop"]) for c in cond_cols}
    t("synth conditional distributions built", ts)

    # Build per-cell BGMM on orig continuous columns
    bgmm_per_cell = {}
    for cell, g in orig.groupby(["Year", "Compound", "PitStop"]):
        if len(g) < 30:
            continue
        n_components = min(3, max(1, len(g) // 50))
        try:
            bgmm = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=0,
                max_iter=100,
                n_init=1,
            )
            bgmm.fit(g[CONT_COLS].values)
            bgmm_per_cell[cell] = bgmm
        except Exception:
            pass
    t(f"BGMM fit on {len(bgmm_per_cell)} cells", ts)

    # Sample N rows from synth marginal, drawing continuous from per-cell BGMM
    N = 20_000
    rng = np.random.default_rng(0)
    cells = synth_marginal.sample(N, weights="prob", replace=True, random_state=0)
    rows_yc, rows_cont = [], []
    cell_keys = []
    for cell_yr, cell_cmp, cell_ps, _ in cells.itertuples(index=False):
        key = (cell_yr, cell_cmp, cell_ps)
        if key not in bgmm_per_cell:
            continue
        sample_vals, _ = bgmm_per_cell[key].sample(1)
        rows_yc.append((cell_yr, cell_cmp, cell_ps))
        rows_cont.append(sample_vals[0])
        cell_keys.append(key)
    df = pd.DataFrame(rows_cont, columns=CONT_COLS)
    df[["Year", "Compound", "PitStop"]] = pd.DataFrame(rows_yc, columns=["Year", "Compound", "PitStop"])
    t(f"BGMM-sampled continuous rows: {df.shape}", ts)

    # Sample categorical conditional on cell
    for col in cond_cols:
        col_assigned = np.empty(len(df), dtype=object)
        for k, idx in df.groupby(["Year", "Compound", "PitStop"]).groups.items():
            idx_arr = np.array(list(idx))
            if k in conds[col]:
                opts = conds[col][k]
                col_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
        df[col] = col_assigned
    # Make Stint integer
    if "Stint" in df.columns:
        df["Stint"] = df["Stint"].astype(int)
    if "Position" in df.columns:
        df["Position"] = df["Position"].astype(int)
    if "LapNumber" in df.columns:
        df["LapNumber"] = df["LapNumber"].astype(int)
    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(int)
    if "PitStop" in df.columns:
        df["PitStop"] = df["PitStop"].astype(int)
    t(f"final replay {df.shape}", ts)

    synth_disc = synth.sample(N, random_state=0).reset_index(drop=True)
    auc = disc_auc(df, synth_disc)
    t(f"qK BGMM cell + cond categorical: disc_auc = {auc:.4f}", ts)

    out["disc_auc"] = auc
    out["n_cells_with_bgmm"] = len(bgmm_per_cell)
    fp = ART / "dgp_v3_qK_gmm.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qK summary ===")
    print(f"  reference: qH (cond cat, no noise): 0.8323")
    print(f"  qK (cell BGMM continuous + cond categorical): {auc:.4f}")
    if auc < 0.7:
        print("  *** STRONG HIT — host might be per-cell density ***")
    elif auc < 0.85:
        print("  *** HIT below 0.85 ***")


if __name__ == "__main__":
    main()
