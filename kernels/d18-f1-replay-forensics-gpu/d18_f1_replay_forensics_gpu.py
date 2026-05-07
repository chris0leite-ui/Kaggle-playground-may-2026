"""d18 F1 — Synthesizer-architecture identification via multi-arch replay.

Train multiple off-the-shelf synthesizers on aadigupta1601 orig (~99k rows),
sample replays, compare to host_synth via:
  (a) Per-feature KS-divergence of each replay vs host_synth — closest = match.
  (b) Discriminator AUC: train binary LGBM "host_synth vs each replay" on
      stratified samples; replay closest to host has lowest disc AUC.
  (c) Mean log-likelihood of host_synth rows under each fitted synthesizer
      density model (where supported).

Architectures (SDV library):
  - GaussianCopula   (parametric copula)
  - CTGAN            (conditional GAN, mode-specific normalization)
  - TVAE             (variational autoencoder)
  - CopulaGAN        (copula + GAN hybrid)

Skipped: TabDDPM (different package, slow on T4); included if budget permits.

Outputs (under /kaggle/working/):
  d18_f1_arch_id_summary.json  — KS / disc AUC / LL per architecture
  oof_d18_f1_disc_<arch>_strat.npy  — discriminator score per synth row (4 bases)
  test_d18_f1_disc_<arch>_strat.npy
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# CRITICAL: ensure SDV + dependencies installed BEFORE imports
print("[boot] install SDV + deps (full deps)", flush=True)
try:
    # numpy<2 pin avoids the sdv==1.16 incompatibility on Python 3.12 image
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "numpy<2.0.0", "sdv==1.16.*", "sdmetrics>=0.16.0"])
except subprocess.CalledProcessError as e:
    print(f"[boot] pip install failed: {e}", flush=True)
    raise

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

WORK = Path("/kaggle/working")


def find_competition_data():
    """Find playground-series-s6e5 train.csv via rglob (path varies)."""
    base = Path("/kaggle/input")
    matches = list(base.rglob("train.csv"))
    matches = [m for m in matches if "playground-series" in str(m).lower()
               or "s6e5" in str(m).lower()]
    if not matches:
        # fallback any train.csv
        matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv under {base}; "
                           f"contents: {os.listdir(base)}")
    return matches[0].parent


def find_orig_data():
    """Find aadigupta1601 orig csv via rglob."""
    base = Path("/kaggle/input")
    matches = list(base.rglob("f1_strategy_dataset*.csv"))
    if not matches:
        matches = list(base.rglob("*strategy*.csv"))
    if not matches:
        raise RuntimeError(f"no orig f1_strategy*.csv under {base}; "
                           f"contents: {os.listdir(base)}")
    return matches[0]
SEED = 42
N_FOLDS = 5
N_REPLAY = 100_000  # rows to sample from each architecture
TARGET = "PitNextLap"
LAPTIME = "LapTime (s)"

# Cap epochs for GPU-affordability: CTGAN/TVAE on 99k × 16 cols, ~150 ep =
# ~30-45 min each on T4. Budget total ~3h.
EPOCHS_CTGAN = 150
EPOCHS_TVAE = 150
EPOCHS_COPULAGAN = 150

# Features used for discriminator + KS comparison. Skip Driver (ghost
# in synth, real in orig — comparing on Driver is uninformative for
# arch ID; the discriminator can learn the synthesizer's tell quickly
# but it's the well-known ghosting, not the arch).
FEATS_NUM = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             LAPTIME, "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change"]
FEATS_CAT = ["Compound", "Race"]
FEATS_ALL = FEATS_NUM + FEATS_CAT


def ks_per_feature(host: pd.DataFrame, replay: pd.DataFrame, feats):
    out = {}
    for c in feats:
        a = pd.to_numeric(host[c], errors="coerce").dropna().values
        b = pd.to_numeric(replay[c], errors="coerce").dropna().values
        if len(a) == 0 or len(b) == 0:
            continue
        if a.dtype.kind not in "fi" or b.dtype.kind not in "fi":
            continue
        try:
            ks, p = ks_2samp(a, b)
            out[c] = float(ks)
        except Exception:
            continue
    out["mean"] = float(np.mean(list(out.values()))) if out else 1.0
    return out


def disc_auc(host_X, replay_X, n_per=20_000):
    """Train binary LGBM 'host vs replay' via 5-fold; report mean OOF AUC."""
    rng = np.random.RandomState(SEED)
    idx_h = rng.choice(len(host_X), min(n_per, len(host_X)), replace=False)
    idx_r = rng.choice(len(replay_X), min(n_per, len(replay_X)), replace=False)
    X = pd.concat([host_X.iloc[idx_h].reset_index(drop=True),
                   replay_X.iloc[idx_r].reset_index(drop=True)], axis=0,
                  ignore_index=True)
    y = np.concatenate([np.zeros(len(idx_h), dtype=int),
                        np.ones(len(idx_r), dtype=int)])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for tr_i, va_i in skf.split(np.zeros(len(y)), y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=63, min_child_samples=50,
                               verbosity=-1, random_state=SEED)
        m.fit(X.iloc[tr_i], y[tr_i])
        p = m.predict_proba(X.iloc[va_i])[:, 1]
        aucs.append(roc_auc_score(y[va_i], p))
    return float(np.mean(aucs))


def main():
    t0 = time.time()
    print(f"[d18 F1 replay forensics]  sdv version check", flush=True)
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (
        GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer,
        CopulaGANSynthesizer,
    )
    print("[boot] sdv imported OK", flush=True)

    # Load
    DATA = find_competition_data()
    print(f"  found competition data at: {DATA}", flush=True)
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    orig_path = find_orig_data()
    print(f"  found orig csv: {orig_path}", flush=True)
    orig = pd.read_csv(orig_path)
    orig = orig[orig["Compound"].notna()].copy()
    if "Normalized_TyreLife" in orig.columns:
        orig = orig.drop(columns=["Normalized_TyreLife"])
    print(f"  train {tr.shape}  test {te.shape}  orig {orig.shape}", flush=True)

    # Restrict to FEATS_ALL + TARGET for SDV training
    train_orig = orig[FEATS_ALL + [TARGET]].copy()
    train_orig[TARGET] = train_orig[TARGET].astype(int)

    # Build SDV metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_orig)
    # Force categorical detection on Compound/Race
    metadata.update_column("Compound", sdtype="categorical")
    metadata.update_column("Race", sdtype="categorical")
    metadata.update_column(TARGET, sdtype="categorical")
    print(f"  metadata detected: {len(metadata.columns)} cols", flush=True)

    archs_to_train = [
        ("GaussianCopula", lambda: GaussianCopulaSynthesizer(metadata)),
        ("CTGAN",
         lambda: CTGANSynthesizer(metadata, epochs=EPOCHS_CTGAN, batch_size=500,
                                  cuda=True, verbose=True)),
        ("TVAE",
         lambda: TVAESynthesizer(metadata, epochs=EPOCHS_TVAE, batch_size=500,
                                 cuda=True)),
        ("CopulaGAN",
         lambda: CopulaGANSynthesizer(metadata, epochs=EPOCHS_COPULAGAN,
                                      batch_size=500, cuda=True, verbose=True)),
    ]

    results = {}
    fitted_models = {}
    replays = {}

    for name, ctor in archs_to_train:
        t1 = time.time()
        print(f"\n[fit {name}]  start", flush=True)
        try:
            model = ctor()
            model.fit(train_orig)
            print(f"[fit {name}]  done in {time.time()-t1:.0f}s", flush=True)
            fitted_models[name] = model
            replay = model.sample(num_rows=N_REPLAY)
            print(f"[sample {name}]  {replay.shape}  cols {list(replay.columns)[:5]}",
                  flush=True)
            replays[name] = replay
        except Exception as e:
            print(f"[fit {name}]  FAILED: {e}", flush=True)
            results[name] = dict(error=str(e))
            continue

    # KS per architecture vs host_synth (using train, FEATS_NUM)
    host = tr[FEATS_NUM + FEATS_CAT].copy()
    print(f"\n[KS per feature: host_synth vs each replay]", flush=True)
    for name, replay in replays.items():
        ks = ks_per_feature(host, replay, FEATS_NUM)
        results.setdefault(name, {})["ks_vs_host"] = ks
        print(f"  {name:18s}  mean KS = {ks['mean']:.4f}", flush=True)

    # Discriminator AUC per architecture (host vs replay)
    print(f"\n[discriminator AUC: host_synth vs each replay]", flush=True)
    for name, replay in replays.items():
        # Encode cats using the union of host + replay
        h = host.copy(); r = replay[FEATS_NUM + FEATS_CAT].copy()
        for c in FEATS_CAT:
            uni = sorted(set(h[c].astype(str)) | set(r[c].astype(str)))
            cm = {v: i for i, v in enumerate(uni)}
            h[c] = h[c].astype(str).map(cm).astype(int)
            r[c] = r[c].astype(str).map(cm).astype(int)
        for c in FEATS_NUM:
            h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0.0)
            r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0.0)
        auc = disc_auc(h, r)
        results.setdefault(name, {})["disc_auc"] = auc
        print(f"  {name:18s}  disc AUC = {auc:.4f}  "
              f"(closer to 0.5 = better arch match)", flush=True)

    # Disc-features as PER-ROW base: P(host_synth | x) under each arch's
    # disc model, applied to all 627k synth rows.
    # For each architecture:
    #   1. Train disc on (host_train, replay).
    #   2. Score all 627k synth rows (train + test) → P(host).
    #   3. Save as oof_/test_ artifacts (positive class = "host").
    print(f"\n[per-row discriminator features for stack]", flush=True)
    all_synth = pd.concat([tr[FEATS_NUM + FEATS_CAT].assign(_split="train"),
                           te[FEATS_NUM + FEATS_CAT].assign(_split="test")],
                          ignore_index=True)
    for c in FEATS_CAT:
        uni = sorted(set(all_synth[c].astype(str)))
        # Add replay categories so encoder can handle them
        for replay in replays.values():
            uni = sorted(set(uni) | set(replay[c].astype(str)))
        cm = {v: i for i, v in enumerate(uni)}
        all_synth[c] = all_synth[c].astype(str).map(cm).astype(int)

    for name, replay in replays.items():
        # Encode replay with same cm
        r = replay[FEATS_NUM + FEATS_CAT].copy()
        for c in FEATS_CAT:
            uni = sorted(set(all_synth[c].astype(int).astype(str)) |
                         set(replay[c].astype(str)))
            cm2 = {v: i for i, v in enumerate(uni)}
            r[c] = r[c].astype(str).map(cm2).astype(int)
        # Build training set: all 627k synth = label 0; replay = label 1.
        X_t = pd.concat([all_synth[FEATS_NUM + FEATS_CAT],
                         r[FEATS_NUM + FEATS_CAT]], ignore_index=True)
        y_t = np.concatenate([np.zeros(len(all_synth), dtype=int),
                              np.ones(len(r), dtype=int)])
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=63, min_child_samples=50,
                               verbosity=-1, random_state=SEED)
        m.fit(X_t, y_t)
        # Score all 627k synth rows: P(replay|x) under this disc.
        # Higher = looks more like THIS arch's replay.
        p_synth = m.predict_proba(all_synth[FEATS_NUM + FEATS_CAT])[:, 1]
        n_tr = len(tr); oof_pos = p_synth[:n_tr]; te_pos = p_synth[n_tr:]
        np.save(WORK / f"oof_d18_f1_disc_{name.lower()}_strat.npy",
                np.column_stack([1 - oof_pos, oof_pos]))
        np.save(WORK / f"test_d18_f1_disc_{name.lower()}_strat.npy",
                np.column_stack([1 - te_pos, te_pos]))
        results.setdefault(name, {})["disc_pos_mean_train"] = float(oof_pos.mean())
        results.setdefault(name, {})["disc_pos_mean_test"] = float(te_pos.mean())
        print(f"  {name:18s}  saved oof+test  mean(p_replay) tr={oof_pos.mean():.4f}",
              flush=True)

    # Verdict: arch with lowest mean KS + lowest disc AUC ≈ closest match
    print(f"\n[verdict]", flush=True)
    summary = dict(
        n_orig=int(len(orig)), n_train=int(len(tr)), n_test=int(len(te)),
        epochs={"CTGAN": EPOCHS_CTGAN, "TVAE": EPOCHS_TVAE,
                "CopulaGAN": EPOCHS_COPULAGAN},
        archs=results,
        wall_s=time.time() - t0,
    )
    # Rank
    ranked = sorted(
        [(n, r.get("ks_vs_host", {}).get("mean", 1.0),
          r.get("disc_auc", 1.0))
         for n, r in results.items() if "error" not in r],
        key=lambda r: (r[1] + r[2]) / 2,
    )
    for n, ks, auc in ranked:
        print(f"  {n:18s}  mean_KS {ks:.4f}  disc_AUC {auc:.4f}  "
              f"composite {(ks+auc)/2:.4f}", flush=True)
    summary["ranking_by_composite"] = [
        dict(arch=n, mean_ks=ks, disc_auc=auc) for n, ks, auc in ranked
    ]
    (WORK / "d18_f1_arch_id_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n[done]  wall {time.time()-t0:.0f}s  →  d18_f1_arch_id_summary.json",
          flush=True)


if __name__ == "__main__":
    main()
