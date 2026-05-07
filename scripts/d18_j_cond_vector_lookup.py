"""d18 J — Conditional-vector tuple lookup.

CTGAN's conditional vector during training-by-sampling rotates through
discrete columns. The discrete columns in this dataset are:
  PitStop, Compound, Stint, Year, Race
The synth was generated with cond vector sampled from the empirical joint
of these. For each synth row, look up its tuple in orig:
  P(PitNextLap=1 | tuple)  +  log(orig_count + 1) for tuple

EB-smoothed lookup with alpha=20.

Per-row features (5 lookups + composites):
  cv_pcs       P(y=1 | (PitStop, Compound, Stint))             3-way
  cv_pcsy      P(y=1 | (PitStop, Compound, Stint, Year))       4-way
  cv_pcsr      P(y=1 | (PitStop, Compound, Stint, Race))       4-way
  cv_pcsry     P(y=1 | full tuple)                             5-way (sparse)
  cv_count_pcs orig count for the 3-way tuple
  cv_count_pcsry orig count for the 5-way tuple
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
from d18_g_mode_id_ctgan import NUM_FEATS

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"
ALPHA = 20.0


def build_lookup(orig, keys, target_col, alpha=ALPHA):
    glob = float(orig[target_col].astype(int).mean())
    gb = orig.groupby(keys, dropna=False).agg(
        n=(target_col, "size"), s=(target_col, "sum")
    )
    gb["rate_eb"] = (gb["s"] + alpha * glob) / (gb["n"] + alpha)
    return gb["rate_eb"].to_dict(), gb["n"].to_dict(), glob


def lookup_apply(df, lookup, keys, default_val):
    n = len(df)
    arr = np.empty(n, dtype=np.float32)
    cols = [df[k].values for k in keys]
    for i in range(n):
        key = tuple(c[i] for c in cols)
        # Handle scalar keys
        if len(key) == 1:
            key = key[0]
        arr[i] = lookup.get(key, default_val)
    return arr


def main():
    t0 = time.time()
    print("[J cond-vector tuple lookup]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig[TARGET] = orig[TARGET].astype(int)

    print(f"  orig {orig.shape}")

    # Normalize discrete columns (str representation for tuple keys)
    for d in [orig, tr, te]:
        for c in ["Compound", "Race"]:
            d[c] = d[c].astype(str)
        for c in ["PitStop", "Stint", "Year"]:
            d[c] = d[c].astype(int)

    print(f"\n[build orig lookups]")
    lookup_pcs, count_pcs, glob = build_lookup(orig, ["PitStop", "Compound", "Stint"], TARGET)
    lookup_pcsy, count_pcsy, _ = build_lookup(orig, ["PitStop", "Compound", "Stint", "Year"], TARGET)
    lookup_pcsr, count_pcsr, _ = build_lookup(orig, ["PitStop", "Compound", "Stint", "Race"], TARGET)
    lookup_pcsry, count_pcsry, _ = build_lookup(orig, ["PitStop", "Compound", "Stint", "Race", "Year"], TARGET)
    print(f"  3-way (P,C,S): {len(lookup_pcs)} buckets")
    print(f"  4-way (P,C,S,Y): {len(lookup_pcsy)} buckets")
    print(f"  4-way (P,C,S,R): {len(lookup_pcsr)} buckets")
    print(f"  5-way (P,C,S,R,Y): {len(lookup_pcsry)} buckets")
    print(f"  global P(y=1) = {glob:.4f}")

    def add_features(df):
        feats = pd.DataFrame(index=df.index)
        feats["cv_pcs"] = lookup_apply(df, lookup_pcs, ["PitStop", "Compound", "Stint"], glob)
        feats["cv_pcsy"] = lookup_apply(df, lookup_pcsy, ["PitStop", "Compound", "Stint", "Year"], glob)
        feats["cv_pcsr"] = lookup_apply(df, lookup_pcsr, ["PitStop", "Compound", "Stint", "Race"], glob)
        feats["cv_pcsry"] = lookup_apply(df, lookup_pcsry, ["PitStop", "Compound", "Stint", "Race", "Year"], glob)
        feats["cv_log_count_pcs"] = np.log1p(
            lookup_apply(df, count_pcs, ["PitStop", "Compound", "Stint"], 0.0)).astype(np.float32)
        feats["cv_log_count_pcsry"] = np.log1p(
            lookup_apply(df, count_pcsry, ["PitStop", "Compound", "Stint", "Race", "Year"], 0.0)).astype(np.float32)
        return feats

    tr_J = add_features(tr)
    te_J = add_features(te)

    y = tr[TARGET].values
    print(f"\n  per-feature standalone AUC:")
    for c in tr_J.columns:
        a = roc_auc_score(y, tr_J[c].values)
        print(f"    {c:24s}  AUC={a:.5f}")

    cmps = sorted(set(tr["Compound"]) | set(te["Compound"]))
    cm = {c: i for i, c in enumerate(cmps)}
    races = sorted(set(tr["Race"]) | set(te["Race"]))
    rm = {r: i for i, r in enumerate(races)}
    raw_cols = ["Compound", "Race"] + NUM_FEATS
    trX = tr[raw_cols].copy()
    teX = te[raw_cols].copy()
    trX["Compound"] = tr["Compound"].map(cm).astype(int)
    teX["Compound"] = te["Compound"].map(cm).astype(int)
    trX["Race"] = tr["Race"].map(rm).astype(int)
    teX["Race"] = te["Race"].map(rm).astype(int)
    trX = pd.concat([trX.reset_index(drop=True), tr_J.reset_index(drop=True)], axis=1)
    teX = pd.concat([teX.reset_index(drop=True), te_J.reset_index(drop=True)], axis=1)

    print(f"\n[downstream LGBM raw + 6 cond-vec features]")
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
    np.save(ART / "oof_d18_j_cond_vector_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d18_j_cond_vector_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))
    summary = dict(oof_auc=auc, wall_s=time.time() - t0)
    (ART / "d18_j_cond_vector_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done J]  wall {time.time()-t0:.0f}s  OOF {auc:.5f}")


if __name__ == "__main__":
    main()
