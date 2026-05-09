"""Phase A2-A4 — train CTGAN-on-orig and decompose host's customisation.

Hypothesis (from Q1+Q2 fingerprint): the host's CTGAN customisation is
mostly in the *sampling marginal* (oversampling Year=2023, PitStop=0),
not in the architecture. Test by training off-the-shelf SDV CTGAN on
orig at two cond-vector schemas and sampling four ways:

  Sample 1 (Q3): default SDV CTGAN, sample with default marginal.
                 Reproduces F6's 0.9993.
  Sample 2 (Q5): default SDV CTGAN, sample with SYNTH's empirical
                 (Year, Compound, PitStop) marginal.
                 If disc-AUC drops below ~0.95, the marginal is the
                 dominant host axis (F8 → F6).
  Sample 3 (Q4):  default SDV CTGAN, sample conditioning on PitStop=0
                  rows for 80%, PitStop=1 rows for 20% (matches synth
                  prior). Tests whether forcing the marginal is enough.

Disc-AUC harness: 5-fold LightGBM, AUC averaged.

Output:
  scripts/artifacts/dgp_v3_q3q4q5_disc_auc.json
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
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"

EPOCHS = 10
N_SAMPLE = 20_000
N_SYNTH_DISC = 20_000


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def load_orig() -> pd.DataFrame:
    df = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    df = df.rename(columns={"LapTime (s)": "LapTime"})
    df = df.drop(columns=["Normalized_TyreLife"])
    df = df.dropna()
    return df.reset_index(drop=True)


def load_synth_subsample(n: int = N_SYNTH_DISC) -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    cols = [c for c in train.columns if c != "PitNextLap"]
    df = pd.concat([train[cols], test[cols]], ignore_index=True)
    if "LapTime (s)" in df.columns:
        df = df.rename(columns={"LapTime (s)": "LapTime"})
    return df.sample(n, random_state=0).reset_index(drop=True)


def disc_auc(replay: pd.DataFrame, synth: pd.DataFrame, label: str = "") -> float:
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


def main() -> None:
    out: dict = {"epochs": EPOCHS, "n_sample": N_SAMPLE}
    ts = time.time()

    orig = load_orig()
    t(f"loaded orig {orig.shape}", ts)

    synth_disc = load_synth_subsample()
    t(f"loaded synth disc subsample {synth_disc.shape}", ts)

    # Synth marginal P(Year, Compound, PitStop) — for Q5 conditioning
    synth_full_pool = pd.concat(
        [pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")],
        ignore_index=True,
    )
    if "LapTime (s)" in synth_full_pool.columns:
        synth_full_pool = synth_full_pool.rename(columns={"LapTime (s)": "LapTime"})
    synth_ycp_marginal = (
        synth_full_pool[["Year", "Compound", "PitStop"]]
        .value_counts(normalize=True)
        .reset_index(name="prob")
    )
    out["synth_YCP_top10"] = synth_ycp_marginal.head(10).to_dict(orient="records")
    t(f"synth (Year, Compound, PitStop) marginal: {len(synth_ycp_marginal)} cells", ts)

    # Train CTGAN on orig
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer

    metadata = Metadata.detect_from_dataframe(data=orig)
    # Force PitStop and Year as categorical (they are int but semantically
    # categorical — SDV's auto-detect may treat them as numeric).
    for col in ["PitStop", "Year", "Stint", "Position"]:
        try:
            metadata.update_column(column_name=col, sdtype="categorical")
        except Exception:
            pass
    t("metadata configured", ts)

    model = CTGANSynthesizer(metadata, epochs=EPOCHS, verbose=False, cuda=False)
    model.fit(orig)
    t(f"CTGAN fit done ({EPOCHS} epochs, {len(orig)} rows)", ts)

    # Sample 1 (Q3): default sampling
    s1 = model.sample(N_SAMPLE)
    t(f"S1 default sample: {s1.shape}", ts)
    out["disc_auc_default_sample"] = disc_auc(s1, synth_disc, "Q3-default")
    t(f"S1 disc AUC vs synth = {out['disc_auc_default_sample']:.4f}", ts)

    # Sample 2 (Q5): synth-marginal conditional sampling
    # Draw N_SAMPLE rows with (Year, Compound, PitStop) ~ synth empirical marginal
    rng = np.random.default_rng(0)
    cells = synth_ycp_marginal.sample(N_SAMPLE, weights="prob", replace=True, random_state=0)
    known = cells[["Year", "Compound", "PitStop"]].reset_index(drop=True)
    # SDV API: sample_from_conditions for legacy, sample_remaining_columns for new
    try:
        s2 = model.sample_remaining_columns(known_columns=known)
    except Exception as e:
        print(f"  sample_remaining_columns failed: {e}")
        # Fallback: use sample_from_conditions per-cell
        from sdv.sampling import Condition
        chunks = []
        for cell, n in cells.groupby(["Year", "Compound", "PitStop"]).size().items():
            cond = Condition(num_rows=int(n), column_values=dict(zip(["Year", "Compound", "PitStop"], cell)))
            try:
                chunks.append(model.sample_from_conditions(conditions=[cond]))
            except Exception:
                pass
        s2 = pd.concat(chunks, ignore_index=True) if chunks else None
    if s2 is not None:
        t(f"S2 synth-marginal sample: {s2.shape}", ts)
        out["disc_auc_synth_marginal"] = disc_auc(s2, synth_disc, "Q5-synth-marginal")
        t(f"S2 disc AUC vs synth = {out['disc_auc_synth_marginal']:.4f}", ts)
        # Also compute per-year disc-AUC asymmetry
        out["per_year_disc_auc_synth_marginal"] = {}
        for y in sorted(synth_disc["Year"].unique().tolist()):
            sub_s = s2[s2["Year"] == y]
            sub_t = synth_disc[synth_disc["Year"] == y]
            if len(sub_s) >= 200 and len(sub_t) >= 200:
                a = disc_auc(sub_s, sub_t, f"Q5-{y}")
                out["per_year_disc_auc_synth_marginal"][int(y)] = a
        t(f"per-year disc AUCs: {out['per_year_disc_auc_synth_marginal']}", ts)
    else:
        out["disc_auc_synth_marginal"] = None

    # Save
    fp = ART / "dgp_v3_q3q4q5_disc_auc.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    # (skip saving model to keep memory free)

    print("\n=== Disc-AUC summary ===")
    print(f"  Q3 default-sampling     : {out['disc_auc_default_sample']:.4f}  (host F6: 0.9993)")
    if out.get("disc_auc_synth_marginal") is not None:
        print(f"  Q5 synth-marginal       : {out['disc_auc_synth_marginal']:.4f}")
        delta = out["disc_auc_default_sample"] - out["disc_auc_synth_marginal"]
        print(f"  Q5 - Q3 (closes gap)   : Δ={delta:+.4f}")


if __name__ == "__main__":
    main()
