"""qAV — K=1 strictest match in 7-feature distance space.

Different from qAT's 4-feature distance space (LapTime/Δ/CumDeg/RP).
qAV uses the full 7 KS-low features (Tyre/Position/LapTime/CumDeg/
RaceProgress/Δ/LapNumber) — different similarity → different neighbors.

If qAV is structurally orthogonal to qAT, K=4 + qAT + qAV could lift more.
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

CELL_LEVELS = [
    ("L6", ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]),
    ("L5", ["Year", "Compound", "PitStop", "Race", "Stint"]),
    ("L4", ["Year", "Compound", "PitStop", "Race"]),
    ("L3", ["Year", "Compound", "PitStop"]),
]
CONT_COLS = ["TyreLife", "Position", "LapTime", "Cumulative_Degradation",
             "RaceProgress", "LapTime_Delta", "LapNumber"]


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv").rename(columns={"LapTime (s)": "LapTime"})
    test = pd.read_csv(DATA / "test.csv").rename(columns={"LapTime (s)": "LapTime"})
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv").rename(
        columns={"LapTime (s)": "LapTime"}).drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)
    print(f"orig {orig.shape} train {train.shape} test {test.shape}", flush=True)

    sc = StandardScaler().fit(orig[CONT_COLS].values)
    Xo = sc.transform(orig[CONT_COLS].values)
    yo_all = orig[TARGET].values

    def kNN_K1(df_):
        Xq = sc.transform(df_[CONT_COLS].values)
        out_label = np.full(len(df_), np.nan, dtype=np.float32)
        out_d = np.full(len(df_), np.nan, dtype=np.float32)
        out_level = np.full(len(df_), -1, dtype=np.int32)
        for level_idx, (level_name, keys) in enumerate(CELL_LEVELS):
            unfilled = np.isnan(out_label)
            n_uf = unfilled.sum()
            if n_uf == 0:
                break
            print(f"    {level_name}: {n_uf} unfilled", flush=True)
            orig_grp = orig.groupby(keys, observed=True).indices
            df_local = df_.loc[unfilled, keys].copy()
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
                nn = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(Xo_c)
                d, ii = nn.kneighbors(Xq_c)
                out_label[q_idx] = yo_c[ii.ravel()].astype(np.float32)
                out_d[q_idx] = d.ravel().astype(np.float32)
                out_level[q_idx] = level_idx
        global_rate = float(orig[TARGET].mean())
        rem = np.isnan(out_label)
        out_label[rem] = global_rate
        out_d[rem] = 0.0
        out_level[rem] = len(CELL_LEVELS)
        return out_label, out_d, out_level

    tr_l, tr_d, tr_lv = kNN_K1(train)
    te_l, te_d, te_lv = kNN_K1(test)

    y = train[TARGET].values
    X = np.column_stack([tr_l, tr_d, tr_lv]).astype(np.float32)
    X_test = np.column_stack([te_l, te_d, te_lv]).astype(np.float32)
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
        print(f"  fold {fold+1} AUC = {a:.5f}  best_iter={m.best_iteration_}", flush=True)

    auc = float(roc_auc_score(y, oof))
    print(f"\nqAV standalone OOF = {auc:.5f}", flush=True)

    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")
    if primary_oof.ndim == 2: primary_oof = primary_oof[:, 1]
    if primary_test.ndim == 2: primary_test = primary_test[:, 1]
    rho_oof = float(spearmanr(oof, primary_oof).correlation)
    rho_test = float(spearmanr(test_pred, primary_test).correlation)
    print(f"  rho_oof: {rho_oof:.5f}  rho_test: {rho_test:.5f}", flush=True)

    # rho vs qAT (4-feat K=1)
    qAT_oof = np.load(ART / "dgp_v3_qAT_K1_oof.npy")
    rho_qAT = float(spearmanr(oof, qAT_oof).correlation)
    print(f"  rho_oof vs qAT: {rho_qAT:.5f}", flush=True)

    # Gates
    BASES = [("d17_h1d_yekenot_full", "oof_d17_h1d_yekenot_full_strat.npy"),
             ("p1_single_cb_v4_gpu", "oof_p1_single_cb_v4_gpu_strat.npy"),
             ("f1_hgbc_deep", "oof_f1_hgbc_deep_strat.npy"),
             ("d16_orig_continuous_only", "oof_d16_orig_continuous_only_strat.npy")]
    base_oofs = []
    for nm, fn in BASES:
        o = np.load(ART / fn)
        if o.ndim == 2: o = o[:, 1]
        base_oofs.append(o)

    def expand(p_list):
        cols = []
        for p in p_list:
            p = np.clip(p, 1e-6, 1 - 1e-6)
            cols += [p, pd.Series(p).rank().values / len(p), np.log(p / (1 - p))]
        return np.column_stack(cols)

    def lr_meta_oof(Xm, y_):
        skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        om = np.zeros(len(y_))
        for tr, va in skf2.split(Xm, y_):
            mm = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            mm.fit(Xm[tr], y_[tr])
            om[va] = mm.predict_proba(Xm[va])[:, 1]
        return om

    Xm_K4 = expand(base_oofs)
    Xm_K5 = expand(base_oofs + [oof])
    Xm_K6 = expand(base_oofs + [oof, qAT_oof])
    auc_K4 = float(roc_auc_score(y, lr_meta_oof(Xm_K4, y)))
    auc_K5 = float(roc_auc_score(y, lr_meta_oof(Xm_K5, y)))
    auc_K6 = float(roc_auc_score(y, lr_meta_oof(Xm_K6, y)))
    print(f"  K=4+qAV alone: OOF={auc_K5:.5f} Δ={(auc_K5-auc_K4)*1e4:+.3f} bp", flush=True)
    print(f"  K=4+qAV+qAT:   OOF={auc_K6:.5f} Δ={(auc_K6-auc_K4)*1e4:+.3f} bp", flush=True)

    np.save(ART / "dgp_v3_qAV_K1_7feat_oof.npy", oof)
    np.save(ART / "dgp_v3_qAV_K1_7feat_test.npy", test_pred)
    out = {"oof_auc": auc, "fold_aucs": fold_aucs, "rho_test": rho_test,
           "rho_oof": rho_oof, "rho_oof_vs_qAT": rho_qAT,
           "k4plus1_lift_bp": (auc_K5-auc_K4)*1e4,
           "k4plus2_qAT_qAV_lift_bp": (auc_K6-auc_K4)*1e4}
    fp = ART / "dgp_v3_qAV_K1_7feat.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {fp.name} t={time.time()-ts:.1f}s")


if __name__ == "__main__":
    main()
