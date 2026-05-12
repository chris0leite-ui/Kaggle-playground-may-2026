"""qAZ — K=1 slim-kNN with stint_imputed in cell key (Stint replaced).

Synth Stint is fabricated 8-valued. stint_imputed = LapNumber-TyreLife+1
has 105 distinct in synth, 5119+ in orig. Replacing Stint with
stint_imputed in the 6-axis cell key gives much finer cell granularity
where it fits, falling back via hierarchy when sparse.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"

CONT_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]


def stint_imp(df):
    return (df["LapNumber"] - df["TyreLife"] + 1).astype(np.int32)


def kNN_K1_with_levels(orig, df, cell_levels, cont_cols, metric="euclidean"):
    sc = StandardScaler().fit(orig[cont_cols].values)
    Xo = sc.transform(orig[cont_cols].values)
    Xq = sc.transform(df[cont_cols].values)
    yo_all = orig[TARGET].values
    out_label = np.full(len(df), np.nan, dtype=np.float32)
    out_d = np.full(len(df), np.nan, dtype=np.float32)
    out_level = np.full(len(df), -1, dtype=np.int32)
    for level_idx, (level_name, keys) in enumerate(cell_levels):
        unfilled = np.isnan(out_label)
        n_uf = unfilled.sum()
        if n_uf == 0:
            break
        print(f"    {level_name}: {n_uf} unfilled", flush=True)
        orig_grp = orig.groupby(keys, observed=True).indices
        df_local = df.loc[unfilled, keys].copy()
        df_local["_qidx"] = np.where(unfilled)[0]
        for cell, sub_df in df_local.groupby(keys, observed=True):
            if cell not in orig_grp:
                continue
            o_idx = orig_grp[cell]
            if len(o_idx) < 1:
                continue
            q_idx = sub_df["_qidx"].values
            Xo_c = Xo[o_idx]
            Xq_c = Xq[q_idx]
            yo_c = yo_all[o_idx]
            nn = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=1).fit(Xo_c)
            d, ii = nn.kneighbors(Xq_c)
            out_label[q_idx] = yo_c[ii.ravel()].astype(np.float32)
            out_d[q_idx] = d.ravel().astype(np.float32)
            out_level[q_idx] = level_idx
    global_rate = float(orig[TARGET].mean())
    rem = np.isnan(out_label)
    out_label[rem] = global_rate
    out_d[rem] = 0.0
    out_level[rem] = len(cell_levels)
    return out_label, out_d, out_level


def fit_and_gate(X, X_test, y, name, primary_oof, primary_test, BASES_files):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    lgb_params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                      min_child_samples=80, random_state=SEED, n_jobs=-1, verbosity=-1)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        a = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(a)
    auc = float(roc_auc_score(y, oof))
    rho_oof = float(spearmanr(oof, primary_oof).correlation)
    rho_test = float(spearmanr(test_pred, primary_test).correlation)

    base_oofs = []
    for fn in BASES_files:
        o = np.load(ART / fn)
        if o.ndim == 2: o = o[:, 1]
        base_oofs.append(o)

    def expand(p_list):
        cols = []
        for p in p_list:
            p = np.clip(p, 1e-6, 1 - 1e-6)
            cols += [p, pd.Series(p).rank().values / len(p), np.log(p / (1 - p))]
        return np.column_stack(cols)

    def lr_meta(Xm, y_):
        skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        om = np.zeros(len(y_))
        for tr, va in skf2.split(Xm, y_):
            m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            m.fit(Xm[tr], y_[tr])
            om[va] = m.predict_proba(Xm[va])[:, 1]
        return om

    Xm_K4 = expand(base_oofs)
    Xm_K5 = expand(base_oofs + [oof])
    auc_K4 = float(roc_auc_score(y, lr_meta(Xm_K4, y)))
    auc_K5 = float(roc_auc_score(y, lr_meta(Xm_K5, y)))
    delta = (auc_K5 - auc_K4) * 1e4
    print(f"\n  {name} standalone OOF = {auc:.5f}", flush=True)
    print(f"    rho_test: {rho_test:.5f}, K=4+1 lift: {delta:+.3f} bp", flush=True)
    return oof, test_pred, dict(oof_auc=auc, fold_aucs=fold_aucs,
                                 rho_oof=rho_oof, rho_test=rho_test,
                                 k4plus1_lift_bp=delta)


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv").rename(columns={"LapTime (s)": "LapTime"})
    test = pd.read_csv(DATA / "test.csv").rename(columns={"LapTime (s)": "LapTime"})
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv").rename(
        columns={"LapTime (s)": "LapTime"}).drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)

    train["stint_imp"] = stint_imp(train)
    test["stint_imp"] = stint_imp(test)
    orig["stint_imp"] = stint_imp(orig)

    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")
    if primary_oof.ndim == 2: primary_oof = primary_oof[:, 1]
    if primary_test.ndim == 2: primary_test = primary_test[:, 1]
    BASES_files = ["oof_d17_h1d_yekenot_full_strat.npy",
                   "oof_p1_single_cb_v4_gpu_strat.npy",
                   "oof_f1_hgbc_deep_strat.npy",
                   "oof_d16_orig_continuous_only_strat.npy"]
    y = train[TARGET].values

    # qAZ: stint_imputed in cell key (replacing Stint)
    CELL_LEVELS_AZ = [
        ("L6", ["Year", "Compound", "PitStop", "Race", "stint_imp", "LapNumber"]),
        ("L5", ["Year", "Compound", "PitStop", "Race", "stint_imp"]),
        ("L4", ["Year", "Compound", "PitStop", "Race"]),
        ("L3", ["Year", "Compound", "PitStop"]),
    ]
    print("\n=== qAZ: K=1, stint_imputed cell key ===")
    tr_l, tr_d, tr_lv = kNN_K1_with_levels(orig, train, CELL_LEVELS_AZ, CONT_COLS)
    te_l, te_d, te_lv = kNN_K1_with_levels(orig, test, CELL_LEVELS_AZ, CONT_COLS)
    X_AZ = np.column_stack([tr_l, tr_d, tr_lv]).astype(np.float32)
    Xt_AZ = np.column_stack([te_l, te_d, te_lv]).astype(np.float32)
    oof_AZ, test_AZ, info_AZ = fit_and_gate(X_AZ, Xt_AZ, y, "qAZ", primary_oof, primary_test, BASES_files)
    np.save(ART / "dgp_v3_qAZ_stint_imp_cell_oof.npy", oof_AZ)
    np.save(ART / "dgp_v3_qAZ_stint_imp_cell_test.npy", test_AZ)

    # qBA: K=1 with Manhattan distance
    print("\n=== qBA: K=1, Manhattan distance, original 6-axis cell ===")
    CELL_LEVELS_STD = [
        ("L6", ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]),
        ("L5", ["Year", "Compound", "PitStop", "Race", "Stint"]),
        ("L4", ["Year", "Compound", "PitStop", "Race"]),
        ("L3", ["Year", "Compound", "PitStop"]),
    ]
    tr_l, tr_d, tr_lv = kNN_K1_with_levels(orig, train, CELL_LEVELS_STD, CONT_COLS, metric="manhattan")
    te_l, te_d, te_lv = kNN_K1_with_levels(orig, test, CELL_LEVELS_STD, CONT_COLS, metric="manhattan")
    X_BA = np.column_stack([tr_l, tr_d, tr_lv]).astype(np.float32)
    Xt_BA = np.column_stack([te_l, te_d, te_lv]).astype(np.float32)
    oof_BA, test_BA, info_BA = fit_and_gate(X_BA, Xt_BA, y, "qBA", primary_oof, primary_test, BASES_files)
    np.save(ART / "dgp_v3_qBA_manhattan_oof.npy", oof_BA)
    np.save(ART / "dgp_v3_qBA_manhattan_test.npy", test_BA)

    # qBB: K=1 with TyreLife-only single-feature distance
    print("\n=== qBB: K=1, single-feature TyreLife distance ===")
    tr_l, tr_d, tr_lv = kNN_K1_with_levels(orig, train, CELL_LEVELS_STD, ["TyreLife"])
    te_l, te_d, te_lv = kNN_K1_with_levels(orig, test, CELL_LEVELS_STD, ["TyreLife"])
    X_BB = np.column_stack([tr_l, tr_d, tr_lv]).astype(np.float32)
    Xt_BB = np.column_stack([te_l, te_d, te_lv]).astype(np.float32)
    oof_BB, test_BB, info_BB = fit_and_gate(X_BB, Xt_BB, y, "qBB", primary_oof, primary_test, BASES_files)
    np.save(ART / "dgp_v3_qBB_tyrelife_oof.npy", oof_BB)
    np.save(ART / "dgp_v3_qBB_tyrelife_test.npy", test_BB)

    # qBC: K=1 with Position-only distance
    print("\n=== qBC: K=1, single-feature Position distance ===")
    tr_l, tr_d, tr_lv = kNN_K1_with_levels(orig, train, CELL_LEVELS_STD, ["Position"])
    te_l, te_d, te_lv = kNN_K1_with_levels(orig, test, CELL_LEVELS_STD, ["Position"])
    X_BC = np.column_stack([tr_l, tr_d, tr_lv]).astype(np.float32)
    Xt_BC = np.column_stack([te_l, te_d, te_lv]).astype(np.float32)
    oof_BC, test_BC, info_BC = fit_and_gate(X_BC, Xt_BC, y, "qBC", primary_oof, primary_test, BASES_files)
    np.save(ART / "dgp_v3_qBC_position_oof.npy", oof_BC)
    np.save(ART / "dgp_v3_qBC_position_test.npy", test_BC)

    # qBD: K=1 with Cumulative_Degradation only
    print("\n=== qBD: K=1, single-feature CumDeg distance ===")
    tr_l, tr_d, tr_lv = kNN_K1_with_levels(orig, train, CELL_LEVELS_STD, ["Cumulative_Degradation"])
    te_l, te_d, te_lv = kNN_K1_with_levels(orig, test, CELL_LEVELS_STD, ["Cumulative_Degradation"])
    X_BD = np.column_stack([tr_l, tr_d, tr_lv]).astype(np.float32)
    Xt_BD = np.column_stack([te_l, te_d, te_lv]).astype(np.float32)
    oof_BD, test_BD, info_BD = fit_and_gate(X_BD, Xt_BD, y, "qBD", primary_oof, primary_test, BASES_files)
    np.save(ART / "dgp_v3_qBD_cumdeg_oof.npy", oof_BD)
    np.save(ART / "dgp_v3_qBD_cumdeg_test.npy", test_BD)

    # qBE: K=1 with Driver in cell key (very sparse but precise)
    print("\n=== qBE: K=1 with Driver in cell key (sparse) ===")
    CELL_LEVELS_DRV = [
        ("L7", ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber", "Driver"]),
        ("L6", ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]),
        ("L5", ["Year", "Compound", "PitStop", "Race", "Stint"]),
        ("L4", ["Year", "Compound", "PitStop", "Race"]),
        ("L3", ["Year", "Compound", "PitStop"]),
    ]
    tr_l, tr_d, tr_lv = kNN_K1_with_levels(orig, train, CELL_LEVELS_DRV, CONT_COLS)
    te_l, te_d, te_lv = kNN_K1_with_levels(orig, test, CELL_LEVELS_DRV, CONT_COLS)
    X_BE = np.column_stack([tr_l, tr_d, tr_lv]).astype(np.float32)
    Xt_BE = np.column_stack([te_l, te_d, te_lv]).astype(np.float32)
    oof_BE, test_BE, info_BE = fit_and_gate(X_BE, Xt_BE, y, "qBE", primary_oof, primary_test, BASES_files)
    np.save(ART / "dgp_v3_qBE_driver_oof.npy", oof_BE)
    np.save(ART / "dgp_v3_qBE_driver_test.npy", test_BE)

    out = {"qAZ": info_AZ, "qBA": info_BA, "qBB": info_BB, "qBC": info_BC,
           "qBD": info_BD, "qBE": info_BE}
    fp = ART / "dgp_v3_q_AZ_BE_variants.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[total wall {time.time()-ts:.1f}s]")
    print("\n=== SUMMARY ===")
    for nm, info in out.items():
        print(f"  {nm}: standalone {info['oof_auc']:.5f}, ρ_test {info['rho_test']:.4f}, K=4+1 {info['k4plus1_lift_bp']:+.3f} bp")


if __name__ == "__main__":
    main()
