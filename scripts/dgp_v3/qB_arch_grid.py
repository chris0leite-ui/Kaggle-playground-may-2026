"""Phase B0 — alternative-architecture sweep.

Q3/Q5 falsified the marginal-recovery hypothesis. Sweep across SDV's
non-CTGAN synthesisers (and one TabDDPM-class baseline if installed)
to find a model class with disc-AUC < 0.95 vs host synth.

Each model is trained on full orig (101k × 14 cols, drop Norm_TyreLife)
for a small budget, sampled to 20k rows, scored against a 20k synth
disc subsample by 5-fold LightGBM AUC.

Output: scripts/artifacts/dgp_v3_qB_arch_grid.json (incremental writes
so a partial run is still useful).
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

N_SAMPLE = 20_000
N_SYNTH = 20_000
RESULT_PATH = ART / "dgp_v3_qB_arch_grid.json"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def load_orig() -> pd.DataFrame:
    df = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    df = df.rename(columns={"LapTime (s)": "LapTime"})
    df = df.drop(columns=["Normalized_TyreLife"]).dropna()
    return df.reset_index(drop=True)


def load_synth_disc(n: int) -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    cols = [c for c in train.columns if c != "PitNextLap"]
    df = pd.concat([train[cols], test[cols]], ignore_index=True)
    if "LapTime (s)" in df.columns:
        df = df.rename(columns={"LapTime (s)": "LapTime"})
    return df.sample(n, random_state=0).reset_index(drop=True)


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


def save_results(results: dict) -> None:
    RESULT_PATH.write_text(json.dumps(results, indent=2, default=str))


def make_metadata(orig: pd.DataFrame):
    from sdv.metadata import Metadata
    metadata = Metadata.detect_from_dataframe(data=orig)
    for col in ["PitStop", "Year", "Stint", "Position"]:
        try:
            metadata.update_column(column_name=col, sdtype="categorical")
        except Exception:
            pass
    return metadata


def run_gaussian_copula(orig: pd.DataFrame, synth_disc: pd.DataFrame, ts: float) -> dict:
    from sdv.single_table import GaussianCopulaSynthesizer
    metadata = make_metadata(orig)
    m = GaussianCopulaSynthesizer(metadata)
    m.fit(orig)
    t("GaussianCopula fit done", ts)
    s = m.sample(N_SAMPLE)
    t(f"sample {s.shape}", ts)
    auc = disc_auc(s, synth_disc)
    t(f"GaussianCopula disc AUC = {auc:.4f}", ts)
    return {"name": "GaussianCopula", "disc_auc": auc, "n_sample": int(len(s))}


def run_tvae(orig: pd.DataFrame, synth_disc: pd.DataFrame, ts: float, epochs: int = 10) -> dict:
    from sdv.single_table import TVAESynthesizer
    metadata = make_metadata(orig)
    m = TVAESynthesizer(metadata, epochs=epochs, cuda=False, verbose=False)
    m.fit(orig)
    t(f"TVAE fit done ({epochs} epochs)", ts)
    s = m.sample(N_SAMPLE)
    t(f"sample {s.shape}", ts)
    auc = disc_auc(s, synth_disc)
    t(f"TVAE disc AUC = {auc:.4f}", ts)
    return {"name": "TVAE", "disc_auc": auc, "epochs": epochs, "n_sample": int(len(s))}


def run_copulagan(orig: pd.DataFrame, synth_disc: pd.DataFrame, ts: float, epochs: int = 10) -> dict:
    from sdv.single_table import CopulaGANSynthesizer
    metadata = make_metadata(orig)
    m = CopulaGANSynthesizer(metadata, epochs=epochs, cuda=False, verbose=False)
    m.fit(orig)
    t(f"CopulaGAN fit done ({epochs} epochs)", ts)
    s = m.sample(N_SAMPLE)
    t(f"sample {s.shape}", ts)
    auc = disc_auc(s, synth_disc)
    t(f"CopulaGAN disc AUC = {auc:.4f}", ts)
    return {"name": "CopulaGAN", "disc_auc": auc, "epochs": epochs, "n_sample": int(len(s))}


def main() -> None:
    ts = time.time()
    orig = load_orig()
    synth_disc = load_synth_disc(N_SYNTH)
    t(f"orig {orig.shape} synth_disc {synth_disc.shape}", ts)

    results = {"runs": [], "host_F6": 0.9993, "Q3_default_ctgan": 0.9993}

    # 1. GaussianCopula — fastest, most basic
    try:
        r = run_gaussian_copula(orig, synth_disc, ts)
        results["runs"].append(r)
        save_results(results)
    except Exception as e:
        print(f"  GaussianCopula FAILED: {type(e).__name__}: {e}")
        results["runs"].append({"name": "GaussianCopula", "error": str(e)})
        save_results(results)

    # 2. TVAE — neural autoencoder
    try:
        r = run_tvae(orig, synth_disc, ts, epochs=10)
        results["runs"].append(r)
        save_results(results)
    except Exception as e:
        print(f"  TVAE FAILED: {type(e).__name__}: {e}")
        results["runs"].append({"name": "TVAE", "error": str(e)})
        save_results(results)

    # 3. CopulaGAN — copula transform + GAN
    try:
        r = run_copulagan(orig, synth_disc, ts, epochs=10)
        results["runs"].append(r)
        save_results(results)
    except Exception as e:
        print(f"  CopulaGAN FAILED: {type(e).__name__}: {e}")
        results["runs"].append({"name": "CopulaGAN", "error": str(e)})
        save_results(results)

    save_results(results)
    print("\n=== Summary ===")
    print(f"  reference: host F6 disc-AUC = 0.9993, default CTGAN = {results['Q3_default_ctgan']}")
    for r in results["runs"]:
        if "disc_auc" in r:
            mark = " <- HIT" if r["disc_auc"] < 0.95 else ""
            print(f"  {r['name']:20s} disc_auc = {r['disc_auc']:.4f}{mark}")
        else:
            print(f"  {r['name']:20s} ERROR: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
