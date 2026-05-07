"""d18 H — Mode-id × (Compound, Stint) trivariate orig class-rate lookup.

Combines G's mode-ids (CTGAN-aware discrete latent) with d15's leak_lookup
pattern. For each KS-low feature's mode-id, compute orig-empirical
P(PitNextLap=1 | (Compound, Stint, mode_id_feat)) → 7 lookup features.

Plus aggregate: mean of 7 lookups, max-min spread.

Leakage-clean: orig labels only; orig mode-id assignment used (the same
VGM that scored synth rows). No synth labels in feature pipeline.

Cost: ~5 min CPU after G has run.
EV: K=21+1 +0.5-2 bp (complements G; G uses tree splits on mode-id;
H uses orig-empirical class rates at the same axis).
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
from sklearn.mixture import BayesianGaussianMixture
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
CAT_OK = ["Compound", "Race"]


def main():
    t0 = time.time()
    print("[H mode-id × (Compound, Stint) lookup]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()

    # Reuse mode-id features from G (or refit if not present)
    p_tr = DATA_OUT / "mode_id_features_train.parquet"
    p_te = DATA_OUT / "mode_id_features_test.parquet"
    if p_tr.exists() and p_te.exists():
        print(f"  loading existing mode-id parquets")
        tr_M = pd.read_parquet(p_tr)
        te_M = pd.read_parquet(p_te)
        # Need orig mode-ids too; refit VGM and assign
        models = fit_vgm_per_feature(orig, KS_LOW_FEATS)
        orig_M = assign_modes(models, orig, KS_LOW_FEATS)
    else:
        print(f"  fitting VGM on orig + assigning modes")
        models = fit_vgm_per_feature(orig, KS_LOW_FEATS)
        orig_M = assign_modes(models, orig, KS_LOW_FEATS)
        tr_M = assign_modes(models, tr, KS_LOW_FEATS)
        te_M = assign_modes(models, te, KS_LOW_FEATS)

    # Build orig lookup table per feature: groupby(Compound, Stint, mode_id_feat).PitNextLap.mean()
    orig_y = orig[TARGET].astype(int).values
    orig_cmp = orig["Compound"].astype(str).values
    orig_stint = orig["Stint"].astype(int).values

    GLOBAL_RATE = float(orig_y.mean())
    print(f"  orig global P(y=1) = {GLOBAL_RATE:.4f}")

    feature_lookups = {}
    for f in KS_LOW_FEATS:
        col_safe = f.replace(" ", "_").replace("(", "").replace(")", "")
        mode_col = f"mode_{col_safe}"
        df_ref = pd.DataFrame({
            "Compound": orig_cmp,
            "Stint": orig_stint,
            "mode": orig_M[mode_col].values,
            "y": orig_y,
        })
        # EB-smoothed: rate = (sum + alpha*global) / (count + alpha)
        ALPHA = 20.0
        gb = df_ref.groupby(["Compound", "Stint", "mode"]).agg(
            n=("y", "size"), s=("y", "sum")
        )
        gb["rate_eb"] = (gb["s"] + ALPHA * GLOBAL_RATE) / (gb["n"] + ALPHA)
        feature_lookups[f] = gb["rate_eb"].to_dict()
        # Print the spread per mode (averaged over Compound, Stint)
        spread = gb["rate_eb"].max() - gb["rate_eb"].min()
        print(f"  {f:24s}  cells={len(gb)}  rate_eb spread {spread:.3f}")

    def add_lookup_features(df, df_M):
        feats = pd.DataFrame(index=df.index)
        cmp_v = df["Compound"].astype(str).values
        stint_v = df["Stint"].astype(int).values
        for f in KS_LOW_FEATS:
            col_safe = f.replace(" ", "_").replace("(", "").replace(")", "")
            mode_col = f"mode_{col_safe}"
            modes_v = df_M[mode_col].values
            lookup = feature_lookups[f]
            n = len(df)
            arr = np.empty(n, dtype=np.float32)
            for i in range(n):
                key = (cmp_v[i], stint_v[i], modes_v[i])
                arr[i] = lookup.get(key, GLOBAL_RATE)
            feats[f"hl_{col_safe}"] = arr
        feats["hl_mean"] = feats.values.mean(axis=1).astype(np.float32)
        feats["hl_max"] = feats.iloc[:, :7].values.max(axis=1).astype(np.float32)
        feats["hl_min"] = feats.iloc[:, :7].values.min(axis=1).astype(np.float32)
        return feats

    print(f"\n[apply lookups → train, test]")
    tr_H = add_lookup_features(tr, tr_M)
    te_H = add_lookup_features(te, te_M)

    print(f"  feature stats (train mean):")
    for c in tr_H.columns:
        print(f"    {c:24s}  mean={tr_H[c].mean():.4f}  std={tr_H[c].std():.4f}")

    # Standalone AUC of each lookup feature
    y = tr[TARGET].astype(int).values
    print(f"\n  per-feature standalone AUC:")
    for c in tr_H.columns:
        a = roc_auc_score(y, tr_H[c].values)
        print(f"    {c:24s}  AUC={a:.5f}")

    # Downstream LGBM: raw + 10 lookup features
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
    trX = pd.concat([trX.reset_index(drop=True), tr_H.reset_index(drop=True)], axis=1)
    teX = pd.concat([teX.reset_index(drop=True), te_H.reset_index(drop=True)], axis=1)

    print(f"\n[downstream LGBM raw + 10 lookup features]")
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
    np.save(ART / "oof_d18_h_mode_lookup_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d18_h_mode_lookup_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))
    summary = dict(oof_auc=auc, ks_low_feats=KS_LOW_FEATS,
                   wall_s=time.time() - t0)
    (ART / "d18_h_mode_lookup_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done H]  wall {time.time()-t0:.0f}s  OOF {auc:.5f}")


if __name__ == "__main__":
    main()
