"""d18 I — Mode-collapse / GAN-bias-factor features.

GANs notoriously over-/under-sample modes during generation. For each
(KS-low-feature, mode-id), compute:
  bias_factor = synth_freq[mode] / orig_freq[mode]
Synth rows in over-sampled modes (bias_factor >> 1) are GAN artifacts;
their target rate likely drifts from orig's. Synth rows in under-sampled
modes are atypical of orig's distribution.

Per-row features:
  bf_<feat>            bias_factor for the row's mode-id on each feature
  bf_geom_mean         geometric mean of 7 bias-factors (composite)
  bf_max_log           max(log bias_factor) — most-extreme over-sampled mode
  bf_min_log           min(log bias_factor) — most-extreme under-sampled mode

This is the FIRST probe to target a known GAN failure mode directly.

EV: K=21+1 +0.3-2 bp (uncertain; novel mechanism).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from d18_g_mode_id_ctgan import (
    KS_LOW_FEATS, NUM_FEATS, N_MODES,
    fit_vgm_per_feature, assign_modes,
)

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
DATA_OUT = Path("data")
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"
LAPTIME = "LapTime (s)"
EPS = 1e-9


def main():
    t0 = time.time()
    print("[I mode-collapse / bias-factor]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()

    p_tr = DATA_OUT / "mode_id_features_train.parquet"
    p_te = DATA_OUT / "mode_id_features_test.parquet"
    if p_tr.exists() and p_te.exists():
        print(f"  loading existing mode-id parquets")
        tr_M = pd.read_parquet(p_tr)
        te_M = pd.read_parquet(p_te)
        models = fit_vgm_per_feature(orig, KS_LOW_FEATS)
        orig_M = assign_modes(models, orig, KS_LOW_FEATS)
    else:
        print(f"  fitting VGM + assigning modes")
        models = fit_vgm_per_feature(orig, KS_LOW_FEATS)
        orig_M = assign_modes(models, orig, KS_LOW_FEATS)
        tr_M = assign_modes(models, tr, KS_LOW_FEATS)
        te_M = assign_modes(models, te, KS_LOW_FEATS)

    # Per-feature mode-frequency normalization
    bias_lookups = {}
    print(f"\n[per-feature mode bias factors (synth_freq / orig_freq)]")
    for f in KS_LOW_FEATS:
        col_safe = f.replace(" ", "_").replace("(", "").replace(")", "")
        mode_col = f"mode_{col_safe}"
        # Frequencies
        synth_modes = tr_M[mode_col].values  # train as proxy for synth (test is similar by AV)
        orig_modes = orig_M[mode_col].values
        unique_modes = sorted(set(np.unique(synth_modes)) | set(np.unique(orig_modes)))
        unique_modes = [m for m in unique_modes if m >= 0]
        synth_n = len(synth_modes); orig_n = len(orig_modes)
        bf = {}
        print(f"  {f:24s}")
        for m in unique_modes:
            sf = max((synth_modes == m).sum() / synth_n, EPS)
            of = max((orig_modes == m).sum() / orig_n, EPS)
            bf[m] = sf / of
            print(f"    mode {m:>2d}  synth_freq={sf:.4f}  orig_freq={of:.4f}  "
                  f"bias_factor={bf[m]:.3f}")
        bias_lookups[f] = bf

    def add_bias_features(df_M):
        n = len(df_M)
        feats = pd.DataFrame(index=df_M.index)
        log_bfs = np.zeros((n, len(KS_LOW_FEATS)), dtype=np.float64)
        for j, f in enumerate(KS_LOW_FEATS):
            col_safe = f.replace(" ", "_").replace("(", "").replace(")", "")
            mode_col = f"mode_{col_safe}"
            modes_v = df_M[mode_col].values
            lookup = bias_lookups[f]
            arr = np.array([lookup.get(int(m), 1.0) if m >= 0 else 1.0
                            for m in modes_v], dtype=np.float64)
            feats[f"bf_{col_safe}"] = arr.astype(np.float32)
            log_bfs[:, j] = np.log(np.clip(arr, EPS, None))
        feats["bf_geom_mean"] = np.exp(log_bfs.mean(axis=1)).astype(np.float32)
        feats["bf_max_log"] = log_bfs.max(axis=1).astype(np.float32)
        feats["bf_min_log"] = log_bfs.min(axis=1).astype(np.float32)
        feats["bf_log_sum"] = log_bfs.sum(axis=1).astype(np.float32)
        return feats

    tr_I = add_bias_features(tr_M)
    te_I = add_bias_features(te_M)

    y = tr[TARGET].astype(int).values
    print(f"\n  per-feature standalone AUC:")
    for c in tr_I.columns:
        a = roc_auc_score(y, tr_I[c].values)
        print(f"    {c:24s}  AUC={a:.5f}")

    cmps = sorted(set(tr["Compound"].astype(str)) | set(te["Compound"].astype(str)))
    cm = {c: i for i, c in enumerate(cmps)}
    races = sorted(set(tr["Race"].astype(str)) | set(te["Race"].astype(str)))
    rm = {r: i for i, r in enumerate(races)}
    raw_cols = ["Compound", "Race"] + NUM_FEATS
    trX = tr[raw_cols].copy()
    teX = te[raw_cols].copy()
    trX["Compound"] = tr["Compound"].astype(str).map(cm).astype(int)
    teX["Compound"] = te["Compound"].astype(str).map(cm).astype(int)
    trX["Race"] = tr["Race"].astype(str).map(rm).astype(int)
    teX["Race"] = te["Race"].astype(str).map(rm).astype(int)
    trX = pd.concat([trX.reset_index(drop=True), tr_I.reset_index(drop=True)], axis=1)
    teX = pd.concat([teX.reset_index(drop=True), te_I.reset_index(drop=True)], axis=1)

    print(f"\n[downstream LGBM raw + 11 bias-factor features]")
    cat_idx = [trX.columns.get_loc(c) for c in ["Compound", "Race"]]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y)); test_avg = np.zeros(len(te))
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=127, min_data_in_leaf=80, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbosity=-1, seed=SEED)
    for fi, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        ds_tr = lgb.Dataset(trX.iloc[tr_i], label=y[tr_i],
                            categorical_feature=cat_idx, free_raw_data=False)
        ds_va = lgb.Dataset(trX.iloc[va_i], label=y[va_i],
                            categorical_feature=cat_idx, reference=ds_tr,
                            free_raw_data=False)
        m = lgb.train(p, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_i] = m.predict(trX.iloc[va_i], num_iteration=m.best_iteration)
        test_avg += m.predict(teX, num_iteration=m.best_iteration) / N_FOLDS
        print(f"  fold {fi}: AUC={roc_auc_score(y[va_i], oof[va_i]):.5f}  "
              f"best_iter={m.best_iteration}")

    auc = float(roc_auc_score(y, oof))
    print(f"\n  OOF AUC = {auc:.5f}")
    np.save(ART / "oof_d18_i_mode_collapse_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d18_i_mode_collapse_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))
    summary = dict(oof_auc=auc, wall_s=time.time() - t0)
    (ART / "d18_i_mode_collapse_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done I]  wall {time.time()-t0:.0f}s  OOF {auc:.5f}")


if __name__ == "__main__":
    main()
