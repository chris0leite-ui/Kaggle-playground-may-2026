"""Day-18b — Path-B hier-meta with alternative segmentation crosses on K=24.

Three label-free axes that K=24's bases don't natively route as
*interactions*:
  1. (Driver_cluster, Stint)   — Driver clustered by aggregate behavior
                                 (label-free); 4 clusters × 5 stints = 20 segs
  2. (Race_class, TyreLife_q5) — Race clustered by aggregate; 4 × 5 = 20 segs
  3. (Position_q5, Compound)   — Position quintile-binned; 5 × 5 = 25 segs

Same K=24 pool as d18 (matches current PRIMARY composition).

Wall: ~6 min per axis. ~20 min total.

Spec follows d18; saves OOF + test for each (axis, τ).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MIN_ROWS = 1000
MAX_ITER = 500

K24_BASES = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"), ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"), ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"), ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"), ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    ("realmlp", "realmlp"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
    ("FM_A", "d9f_FM_A"), ("FM_B", "d9f_FM_B"),
    ("d16_cont_only", "d16_orig_continuous_only"),
    ("p1_cb_v3", "p1_single_cb_v3_gpu"),
    ("h1d_yekenot", "d17_h1d_yekenot_full"),
]


def _pos_load(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel().astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def fit_lr_aug(F, y, max_iter=MAX_ITER):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


# -----------------------------------------------------------------------------
# Axis builders — return (seg_train, seg_test, n_seg, label)
# All label-free.
# -----------------------------------------------------------------------------

def axis_driver_cluster_stint(train, test, k_drv=4):
    """Driver clusters from KMeans on per-Driver mean of:
       LapTime, TyreLife, Position, RaceProgress.
       Then × Stint (clipped 0-5).
    """
    full = pd.concat([train, test], axis=0, ignore_index=True)
    drv_agg = full.groupby("Driver")[
        ["LapTime (s)", "TyreLife", "Position", "RaceProgress"]
    ].mean().fillna(0)
    sc = StandardScaler()
    X = sc.fit_transform(drv_agg.values)
    km = KMeans(n_clusters=k_drv, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    drv_to_cluster = dict(zip(drv_agg.index, labels))

    drv_cluster_tr = train["Driver"].map(drv_to_cluster).fillna(0).astype(int).values
    drv_cluster_te = test["Driver"].map(drv_to_cluster).fillna(0).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    n_stint = 6
    seg_train = drv_cluster_tr * n_stint + s_tr
    seg_test = drv_cluster_te * n_stint + s_te
    n_seg = k_drv * n_stint
    return seg_train, seg_test, n_seg, "driver_cluster_x_stint"


def axis_race_class_tyrelife_q5(train, test, k_race=4):
    """Race clusters from KMeans on per-Race aggregates; × TyreLife quintile."""
    full = pd.concat([train, test], axis=0, ignore_index=True)
    race_agg = full.groupby("Race")[
        ["LapTime (s)", "TyreLife", "Position", "RaceProgress"]
    ].mean().fillna(0)
    sc = StandardScaler()
    X = sc.fit_transform(race_agg.values)
    km = KMeans(n_clusters=k_race, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    race_to_cluster = dict(zip(race_agg.index, labels))

    race_cluster_tr = train["Race"].map(race_to_cluster).fillna(0).astype(int).values
    race_cluster_te = test["Race"].map(race_to_cluster).fillna(0).astype(int).values

    # TyreLife quintile (Rule 25 safe, AV-AUC=0.502)
    kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile",
                          subsample=None)
    tl_full = full[["TyreLife"]].fillna(0).values.astype(np.float32)
    kb.fit(tl_full)
    tl_tr_q = kb.transform(train[["TyreLife"]].fillna(0).values.astype(
        np.float32)).ravel().astype(int)
    tl_te_q = kb.transform(test[["TyreLife"]].fillna(0).values.astype(
        np.float32)).ravel().astype(int)
    n_q = 5
    seg_train = race_cluster_tr * n_q + tl_tr_q
    seg_test = race_cluster_te * n_q + tl_te_q
    n_seg = k_race * n_q
    return seg_train, seg_test, n_seg, "race_class_x_tyrelife_q5"


def axis_position_q5_compound(train, test):
    """Position quintile × Compound."""
    full = pd.concat([train, test], axis=0, ignore_index=True)
    kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile",
                          subsample=None)
    pos_full = full[["Position"]].fillna(0).values.astype(np.float32)
    kb.fit(pos_full)
    pos_tr_q = kb.transform(train[["Position"]].fillna(0).values.astype(
        np.float32)).ravel().astype(int)
    pos_te_q = kb.transform(test[["Position"]].fillna(0).values.astype(
        np.float32)).ravel().astype(int)

    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values

    n_cats = len(cats)
    seg_train = pos_tr_q * n_cats + c_tr
    seg_test = pos_te_q * n_cats + c_te
    n_seg = 5 * n_cats
    return seg_train, seg_test, n_seg, "position_q5_x_compound"


AXIS_BUILDERS = {
    "driver_cluster_stint": axis_driver_cluster_stint,
    "race_class_tyrelife_q5": axis_race_class_tyrelife_q5,
    "position_q5_compound": axis_position_q5_compound,
}


def run_axis(axis_name, train, test, y, F_oof, F_test, prim_oof, prim_test):
    """Run Path-B sweep for one axis. Returns dict with all τ results."""
    print(f"\n{'='*70}")
    print(f"AXIS: {axis_name}")
    print('='*70)
    builder = AXIS_BUILDERS[axis_name]
    seg_train, seg_test, n_seg, label = builder(train, test)
    sizes = np.bincount(seg_train, minlength=n_seg)
    populated = int(np.sum(sizes >= MIN_ROWS))
    print(f"n_seg={n_seg}; ≥{MIN_ROWS} rows: {populated}")
    print(f"  segment sizes: min/med/max = "
          f"{sizes[sizes>0].min()}/{int(np.median(sizes[sizes>0]))}/"
          f"{sizes.max()}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    taus = [5000, 20000, 100000, 500000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    print(f"\n--- {axis_name} hier-meta sweep ---")
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = fit_lr_aug(F_oof[tr], y[tr])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        skipped = 0
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                skipped += 1
                continue
            W_local[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            mask[s] = True
        for tau in taus:
            n_local = counts.astype(np.float64)
            alpha = n_local / (n_local + tau)
            W_shrunk = (alpha[:, None] * W_local +
                        (1 - alpha[:, None]) * w_global[None, :])
            for s in np.unique(seg_train[va]):
                idx = np.where(seg_train[va] == s)[0]
                w = W_shrunk[s] if mask[s] else w_global
                oofs[tau][va[idx]] = predict_aug(F_oof[va[idx]], w)
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s "
              f"({int(mask.sum())}/{n_seg} segs fit; {skipped} skipped)")

    # Full-train fit
    w_global_full = fit_lr_aug(F_oof, y)
    W_local_full = np.zeros((n_seg, len(w_global_full)))
    counts_full = np.zeros(n_seg, dtype=np.int64)
    mask_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        counts_full[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_local_full[s] = fit_lr_aug(F_oof[idx], y[idx])
        mask_full[s] = True
    test_preds = {}
    for tau in taus:
        n_local = counts_full.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local_full +
                    (1 - alpha[:, None]) * w_global_full[None, :])
        tp = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx = np.where(seg_test == s)[0]
            w = W_shrunk[s] if mask_full[s] else w_global_full
            tp[idx] = predict_aug(F_test[idx], w)
        test_preds[tau] = tp

    auc_prim_oof = float(roc_auc_score(y, prim_oof))
    rare_thr = float(np.quantile(prim_test, 0.99))
    prim_pos = prim_test >= rare_thr
    print(f"\n  Reference: PRIMARY OOF = {auc_prim_oof:.5f}")
    axis_results = {}
    for tau in taus:
        oof = oofs[tau]
        tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho_prim, _ = spearmanr(tp, prim_test)
        new_pos = tp >= rare_thr
        f_prim_neg = int(np.sum(prim_pos & ~new_pos))
        f_prim_pos = int(np.sum(~prim_pos & new_pos))
        ratio = (min(f_prim_neg, f_prim_pos) /
                 max(f_prim_neg, f_prim_pos)
                 if max(f_prim_neg, f_prim_pos) > 0 else 1.0)
        d_oof = (auc - auc_prim_oof) * 1e4
        print(f"  τ={tau}:  OOF {auc:.5f}  Δ {d_oof:+.2f}bp  "
              f"ρ {rho_prim:.5f}  flips {f_prim_neg}+{f_prim_pos} ratio {ratio:.3f}")
        np.save(ART / f"oof_d18b_{label}_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d18b_{label}_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        axis_results[str(tau)] = dict(
            oof=auc, delta_oof_primary_bp=float(d_oof),
            rho_vs_primary=float(rho_prim),
            flips_neg=f_prim_neg, flips_pos=f_prim_pos, flip_ratio=float(ratio),
        )
    return axis_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+",
                    default=list(AXIS_BUILDERS.keys()))
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    prim_oof = _pos_load(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    prim_test = _pos_load(ART / "test_d17_K24_d18pool_h1d_strat.npy")

    base_oofs, base_tests = [], []
    for label, fname in K24_BASES:
        base_oofs.append(_pos_load(ART / f"oof_{fname}_strat.npy"))
        base_tests.append(_pos_load(ART / f"test_{fname}_strat.npy"))
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"K={len(base_oofs)} bases; F shape {F_oof.shape}")

    summary = dict(primary_oof_ref=float(roc_auc_score(y, prim_oof)),
                   axes={})
    for axis_name in args.axes:
        if axis_name not in AXIS_BUILDERS:
            print(f"  unknown axis: {axis_name}")
            continue
        summary["axes"][axis_name] = run_axis(
            axis_name, train, test, y, F_oof, F_test, prim_oof, prim_test)

    summary["wall_s"] = time.time() - t0
    out_path = ART / "d18b_path_b_alt_axes_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out_path}  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
