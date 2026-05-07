"""scripts/lr_leverage_phaseA.py — Probes 1 and 2.

Probe 1: K=24 + lr_mega min-meta gate (does mega add anything to PRIMARY?)
Probe 2: lr_mega top-30 coefficient distillation (which features carry signal?)

Output: scripts/artifacts/lr_leverage_phaseA.json + console.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from p1_features import (
    make_features_static, fit_fs_a, apply_fs_a, cv_target_encode, TE_CONFIGS,
)
from lr_bank_rich_fe import build_dgp_rule_features

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K24_GBDT_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
    "d16_orig_continuous_only", "p1_single_cb_v3_gpu",
    "d17_h1d_yekenot_full",
]

NUM_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
]
CAT_COLS = ["Driver", "Compound", "Race"]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def _meta_oof(y, F):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


# ============ Probe 1 ============
def probe1_k24_plus_mega_gate(y):
    print("\n=== Probe 1: K=24 + lr_mega min-meta gate ===", flush=True)
    P_gbdt = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in K24_GBDT_BASES])
    mega_oof = _pos(ART / "oof_lr_mega_strat.npy")

    F_gbdt = _expand(P_gbdt)
    t0 = time.time()
    _, auc_gbdt = _meta_oof(y, F_gbdt)
    print(f"  K=24 baseline OOF: {auc_gbdt:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    F_with = _expand(np.column_stack([P_gbdt, mega_oof]))
    t0 = time.time()
    _, auc_with = _meta_oof(y, F_with)
    delta_bp = (auc_with - auc_gbdt) * 1e4
    print(f"  K=24 + lr_mega OOF: {auc_with:.5f}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  Δ vs K=24: {delta_bp:+.3f} bp", flush=True)

    prim_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    rho, _ = spearmanr(mega_oof, prim_oof)
    print(f"  ρ(mega, PRIMARY) = {rho:+.4f}  (highest of any LR)", flush=True)

    return dict(
        k24_auc=auc_gbdt, with_mega_auc=auc_with, delta_bp=float(delta_bp),
        rho_mega_vs_primary=float(rho),
    )


# ============ Probe 2 ============
def build_mega_features_full_train(train, test, y):
    """Reproduce mega's FE matrix for FULL TRAIN (no fold loop, for coef extraction).

    Matches lr_bank_rich_fe.run_lr_mega's per-fold pipeline but using ALL train rows
    for FS_A and DGP rules (since we're fitting on full train for coef inspection).
    Returns (X_full_train, X_test, feat_names).
    """
    state2 = {}
    train_S, state2 = make_features_static(train, fit=True, state=state2)
    test_S, _ = make_features_static(test, fit=False, state=state2)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    rozen_te_oof, rozen_te_test = {}, {}
    for cols, smooth, name in TE_CONFIGS:
        oof_enc, test_enc = cv_target_encode(
            train, test, cols, train[TARGET].astype(int), fold_list, smooth)
        rozen_te_oof[name] = oof_enc
        rozen_te_test[name] = test_enc

    keys_3way = [["Driver", "Race", "Year"], ["Driver", "Race", "Compound"],
                 ["Driver", "Year", "Compound"], ["Race", "Year", "Compound"]]
    smoothings = [1, 5, 20, 100]
    threeway_oof, threeway_test = [], []
    threeway_names = []
    for keys in keys_3way:
        for sm in smoothings:
            oof_enc, test_enc = cv_target_encode(
                train, test, keys, train[TARGET].astype(int), fold_list, sm)
            threeway_oof.append(oof_enc)
            threeway_test.append(test_enc)
            threeway_names.append(f"3wTE_{'_'.join(keys)}_α{sm}")

    num_tr_raw = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te_raw = test[NUM_COLS].fillna(0).values.astype(np.float32)
    kb = KBinsDiscretizer(n_bins=20, encode="onehot", strategy="quantile",
                          subsample=None)
    kb.fit(np.vstack([num_tr_raw, num_te_raw]))
    Bk_tr = kb.transform(num_tr_raw)
    Bk_te = kb.transform(num_te_raw)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    Oc_tr = enc.transform(train[CAT_COLS])
    Oc_te = enc.transform(test[CAT_COLS])

    # FS_A on FULL train (for coef inspection only — leak-aware)
    fs_a = fit_fs_a(train)
    train_A = apply_fs_a(train_S, fs_a)
    test_A = apply_fs_a(test_S, fs_a)

    drop_cols_static = ["Driver", "Race", "Compound", "id", TARGET]
    feat_cols = [c for c in train_A.columns if c not in drop_cols_static
                 and c not in CAT_COLS and train_A[c].dtype.kind in "biufc"]

    Xstatic_tr = train_A[feat_cols].fillna(0).values.astype(np.float32)
    Xstatic_te = test_A[feat_cols].fillna(0).values.astype(np.float32)

    # DGP rules — for "full train fit" we approximate with fold-0 setup
    n = len(y)
    tr0 = np.arange(n)  # full train
    va0 = np.arange(0)  # empty
    # Custom: directly fit lookups on full train
    rule_tr_full, _, rule_te_full = build_dgp_rule_features_full(
        train, test, y)

    # Build feature matrix
    te_arr = np.column_stack(list(rozen_te_oof.values()))
    te_arr_te = np.column_stack(list(rozen_te_test.values()))
    tw_arr = np.column_stack(threeway_oof)
    tw_arr_te = np.column_stack(threeway_test)

    num_tr = np.hstack([Xstatic_tr, te_arr, tw_arr, rule_tr_full])
    num_te = np.hstack([Xstatic_te, te_arr_te, tw_arr_te, rule_te_full])
    sc = StandardScaler()
    num_tr_s = sc.fit_transform(num_tr)
    num_te_s = sc.transform(num_te)

    X_tr = np.hstack([num_tr_s, Bk_tr.toarray().astype(np.float32),
                      Oc_tr.toarray().astype(np.float32)])
    X_te = np.hstack([num_te_s, Bk_te.toarray().astype(np.float32),
                      Oc_te.toarray().astype(np.float32)])

    # Build feature names
    names = (feat_cols
             + [f"rozenTE_{n}" for n in rozen_te_oof.keys()]
             + threeway_names
             + [f"DGP_rule_{i}" for i in range(rule_tr_full.shape[1])]
             + [f"KBins20_{i}" for i in range(Bk_tr.shape[1])]
             + [f"catOHE_{i}" for i in range(Oc_tr.shape[1])])
    return X_tr, X_te, names


def build_dgp_rule_features_full(train_df, test_df, y_train):
    """Same as in lr_bank_rich_fe but fit on FULL train (for coef inspection only)."""
    rules = [("Compound", "Stint"), ("Driver", "Compound"), ("Year", "Race")]
    tr_tl = train_df["TyreLife"].values
    edges = np.quantile(tr_tl, np.linspace(0, 1, 11))
    edges[0] = -np.inf
    edges[-1] = np.inf

    def tyre_decile(arr):
        return np.clip(np.searchsorted(edges, arr, side="right") - 1, 0, 9)

    def make_keys(df):
        keys = {}
        for cols in rules:
            keys[cols] = list(zip(*[df[c].astype(str).values for c in cols]))
        keys[("Compound", "TyreDecile")] = list(zip(
            df["Compound"].astype(str).values,
            tyre_decile(df["TyreLife"].values).astype(str)))
        return keys

    keys_train = make_keys(train_df)
    keys_test = make_keys(test_df)
    glob = float(y_train.mean())
    alphas = [5, 20, 100, 500]

    tr_cols, te_cols = [], []
    for rule_cols, all_keys in keys_train.items():
        keys_te_r = keys_test[rule_cols]
        df = pd.DataFrame({"k": all_keys, "y": y_train})
        g = df.groupby("k", observed=True)["y"]
        counts = g.count(); sums = g.sum()
        for a in alphas:
            sm = (sums + a * glob) / (counts + a)
            mp = sm.to_dict()
            tr_cols.append(np.array([mp.get(k, glob) for k in all_keys],
                                    dtype=np.float32))
            te_cols.append(np.array([mp.get(k, glob) for k in keys_te_r],
                                    dtype=np.float32))
    return np.column_stack(tr_cols), None, np.column_stack(te_cols)


def probe2_mega_coef_distillation(train, test, y):
    print("\n=== Probe 2: lr_mega top-30 coefficient distillation ===", flush=True)
    print("  Building mega FE matrix (full-train for coef inspection)...", flush=True)
    t0 = time.time()
    X_tr, X_te, names = build_mega_features_full_train(train, test, y)
    print(f"  shape Xtr {X_tr.shape}, names {len(names)}  ({time.time()-t0:.1f}s)", flush=True)

    print("  Fitting full-train LR for coef extraction...", flush=True)
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr.fit(X_tr, y)
    coef = lr.coef_.ravel()
    print(f"  fit done ({time.time()-t0:.1f}s)", flush=True)

    # Top by |w|
    order = np.argsort(-np.abs(coef))
    top30 = [(int(i), names[i] if i < len(names) else f"feat_{i}", float(coef[i]))
             for i in order[:30]]
    print(f"\n  Top-30 features by |w|:", flush=True)
    for i, name, w in top30:
        print(f"    [{i:>4}]  w={w:+.4f}   {name}", flush=True)

    # Family-level summary
    family_mass = {"static": 0.0, "rozenTE": 0.0, "3wTE": 0.0, "DGP_rule": 0.0,
                   "KBins20": 0.0, "catOHE": 0.0}
    family_count = {k: 0 for k in family_mass}
    for i, name, w in zip(range(len(names)), names, coef):
        if name.startswith("rozenTE_"):
            k = "rozenTE"
        elif name.startswith("3wTE_"):
            k = "3wTE"
        elif name.startswith("DGP_rule_"):
            k = "DGP_rule"
        elif name.startswith("KBins20_"):
            k = "KBins20"
        elif name.startswith("catOHE_"):
            k = "catOHE"
        else:
            k = "static"
        family_mass[k] += abs(w)
        family_count[k] += 1
    print(f"\n  Family-level |w| mass:", flush=True)
    total_mass = sum(family_mass.values())
    for k, m in sorted(family_mass.items(), key=lambda x: -x[1]):
        cnt = family_count[k]
        print(f"    {k:<10s}  mass={m:.2f} ({100*m/total_mass:.1f}%)  "
              f"n_feats={cnt}  mean_|w|={m/max(cnt,1):.4f}", flush=True)

    return dict(top30=top30, family_mass=family_mass, family_count=family_count,
                n_features=len(names))


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"data: train {train.shape}, test {test.shape}, prior {y.mean():.4f}",
          flush=True)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-probe1", action="store_true")
    args = ap.parse_args()
    out = {}
    if args.skip_probe1:
        # reuse already-computed probe1 result
        out["probe1"] = dict(k24_auc=0.95385, with_mega_auc=0.95387,
                              delta_bp=0.183, rho_mega_vs_primary=0.9030)
    else:
        out["probe1"] = probe1_k24_plus_mega_gate(y)
    out["probe2"] = probe2_mega_coef_distillation(train, test, y)

    out_json = ART / "lr_leverage_phaseA.json"
    out_json.write_text(json.dumps(out, indent=2, default=lambda o: float(o)))
    print(f"\n→ {out_json}", flush=True)


if __name__ == "__main__":
    main()
