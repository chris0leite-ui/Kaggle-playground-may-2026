"""scripts/probe_forest_sweep.py — Random Forest sweep across three angles.

PI directive (2026-05-08): extend the model family with a forest. Three
probes in one script so they share data loading and a common gate.

Angle A — RF base on the yekenot feature recipe.
    Same FE that lifted CatBoost +24 bp standalone (d17 / p1cb_v4).
    Sklearn RandomForestClassifier, 5-fold StratifiedKFold (seed=42).
    Min-meta gate vs K=4+1 LR-meta on the existing pool.
    Hypothesis: if rank-lock holds at the logit level (Day-19 / EXP-NEW),
    this lands WEAK_PASS like d15c ExtraTrees. If yekenot's CV-target-
    encoding interaction features push standalone OOF into 0.94-0.95,
    K=4's 3-D logit ceiling might let it through.

Angle B — RF as the meta-stacker over K=4 [P, rank, logit] (12 feat).
    Direct port of irrigation comp's sklearn RF meta (+35 bp there).
    Compares against LR-meta (the current K=4 PRIMARY meta).
    Hypothesis: Day-20 PCA-meta probe falsified LightGBM-meta by 1-2 bp;
    RF is in the same inductive class. Strong prior for null/regress.

Angle C — RF on combined input (K=4 [P, rank, logit] + raw numerics).
    A non-LR meta-learner that also sees the raw features the bases
    were trained on. Distinct from B because the meta can interact
    base predictions with raw signals. Day-19 K=4-LR-meta + top-5
    raw numerics was null at LR; the inductive-class swap to RF is
    what makes this probe novel.

Each angle saves OOF/test predictions, a results JSON, and prints a
gate-style summary (standalone OOF, ρ vs PRIMARY, K=4+1 LR-meta lift).

Usage:
    python scripts/probe_forest_sweep.py --angles A B C
    python scripts/probe_forest_sweep.py --angles A --smoke   # 1 fold
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)

ID = "id"
TARGET = "PitNextLap"
SEED, N_FOLDS, MAX_ITER = 42, 5, 500

# K=4 forward-greedy pool (matches state/current.md)
K4_BASES = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]

# combo_names from yekenot recipe (cell 6)
IMPORTANT_COMBOS = [("Race", "Compound"), ("Race", "Year")]


# --------------------------------------------------------------------- #
# Yekenot feature engineering — verbatim port of d17_h1d_yekenot_full   #
# (cell 6), minus the orig-data concat (we don't have data/original/).  #
# --------------------------------------------------------------------- #
def yekenot_fe(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
               category_map: dict, fit: bool):
    df = df.copy()
    df["_LapNumber_/_RaceProgress"] = (
        df["LapNumber"] / (df["RaceProgress"] + 1e-6)
    ).astype("float32")
    df["_TyreLife_/_LapNumber"] = (
        df["TyreLife"] / df["LapNumber"].clip(lower=1)
    ).astype("float32")

    for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype(str)

    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = (
            f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        )
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype("int32")

    bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal",
                    strategy="quantile", subsample=None,
                )
                binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                category_map[bin_name] = kb
            else:
                kb = category_map[bin_name]
                binned = kb.transform(df[[col]]).ravel().astype("int32")
            df[bin_name] = binned
            df[bin_name] = df[bin_name].astype(str)

    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        s = df[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(s, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = s.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype(str)

    new_cat = [c for c in df.columns if c.endswith("_")]
    new_num = [c for c in df.columns if c.startswith("_")
               and not c.endswith("_cat_")]
    return df, new_cat, new_num, combo_names


def label_encode(X: pd.DataFrame, X_test: pd.DataFrame, cols: list[str]):
    """Label-encode object/string columns on union of train+test."""
    for c in cols:
        if c not in X.columns:
            continue
        uniq = pd.concat([X[c], X_test[c]], ignore_index=True
                         ).astype(str).unique()
        mp = {v: i for i, v in enumerate(sorted(uniq))}
        X[c] = X[c].astype(str).map(mp).astype(np.int32)
        X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)


def to_float(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        if X[c].dtype == np.int32:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    return X.fillna(-1)


# --------------------------------------------------------------------- #
# Pool helpers: load K=4 OOFs and tests as positive-class probabilities.
# --------------------------------------------------------------------- #
def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def load_k4_pool():
    oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K4_BASES]
    tests = [_pos(ART / f"test_{b}_strat.npy") for b in K4_BASES]
    return np.column_stack(oofs), np.column_stack(tests)


def lr_meta_oof(F: np.ndarray, y: np.ndarray, splits) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof


# --------------------------------------------------------------------- #
# Predicted-LB band (matches probe.py.predicted_lb_delta_bp)            #
# --------------------------------------------------------------------- #
def pred_lb_band(d_oof_bp: float, rho: float) -> float:
    if rho >= 0.99996:
        return d_oof_bp
    if rho >= 0.999:
        return d_oof_bp - 0.5
    if rho >= 0.995:
        return d_oof_bp - 1.5
    if rho >= 0.99:
        return d_oof_bp - 3.0
    return d_oof_bp - 5.0


# --------------------------------------------------------------------- #
# Angle A — RF base on yekenot recipe                                   #
# --------------------------------------------------------------------- #
def angle_a(args, splits, y, k4_oof, k4_test, primary_test_pos):
    print("\n========== ANGLE A: RF base on yekenot recipe ==========")
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X = train.drop([ID, TARGET], axis=1).copy()
    X_test = test.drop([ID], axis=1).copy()

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    cmap: dict = {}
    X, new_cat, new_num, combo_names = yekenot_fe(
        X, num_cols, cat_cols, cmap, fit=True
    )
    X_test, _, _, _ = yekenot_fe(X_test, num_cols, cat_cols, cmap, fit=False)
    cat_cols += new_cat
    num_cols += new_num
    label_encode(X, X_test, cat_cols)
    X = to_float(X)
    X_test = to_float(X_test)
    print(f"  features after FE: train {X.shape}  test {X_test.shape}"
          f"  ({time.time()-t0:.1f}s)")

    n_estimators = 200 if args.smoke else 400
    n_jobs = args.n_jobs

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    fold_walls = []

    folds_to_run = splits[:1] if args.smoke else splits
    for k, (tr, va) in enumerate(folds_to_run):
        t1 = time.time()
        # Per-fold CV target encoding on combo_names (yekenot's #6).
        TE = TargetEncoder(cv=N_FOLDS, smooth="auto", shuffle=True,
                           random_state=SEED)
        tr_te = TE.fit_transform(X[combo_names].iloc[tr], y[tr])
        va_te = TE.transform(X[combo_names].iloc[va])
        tst_te = TE.transform(X_test[combo_names])
        te_names = [f"_{c}TE" for c in combo_names]

        X_tr = X.iloc[tr].drop(columns=combo_names).copy()
        X_va = X.iloc[va].drop(columns=combo_names).copy()
        X_ts = X_test.drop(columns=combo_names).copy()
        X_tr[te_names] = tr_te
        X_va[te_names] = va_te
        X_ts[te_names] = tst_te

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=100,
            max_samples=0.5,
            n_jobs=n_jobs,
            random_state=SEED,
        )
        clf.fit(X_tr.values, y[tr])
        oof[va] = clf.predict_proba(X_va.values)[:, 1]
        test_pred += clf.predict_proba(X_ts.values)[:, 1] / N_FOLDS

        s = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(s)
        fold_walls.append(time.time() - t1)
        print(f"  fold {k+1}/{len(splits)}: AUC={s:.5f}  "
              f"wall={time.time()-t1:.1f}s", flush=True)

    if args.smoke:
        # Only fold 0 done; rescale test and report partial AUC.
        cov_idx = splits[0][1]
        oof_auc = float(roc_auc_score(y[cov_idx], oof[cov_idx]))
        test_pred *= N_FOLDS
        print(f"  [SMOKE] partial OOF AUC: {oof_auc:.5f}  cov="
              f"{len(cov_idx)/len(y)*100:.1f}%")
    else:
        oof_auc = float(roc_auc_score(y, oof))
        print(f"  OOF AUC: {oof_auc:.5f}")

    np.save(ART / "oof_rf_yekenot_strat.npy",
            np.column_stack([1 - oof, oof]).astype(np.float32))
    np.save(ART / "test_rf_yekenot_strat.npy",
            np.column_stack([1 - test_pred, test_pred]).astype(np.float32))

    # Gate vs K=4 LR-meta PRIMARY
    rho, _ = spearmanr(test_pred, primary_test_pos)
    rho = float(rho)

    F_base = expand(k4_oof)
    F_with = expand(np.column_stack([k4_oof, oof]))
    base_oof = lr_meta_oof(F_base, y, splits)
    with_oof = lr_meta_oof(F_with, y, splits)
    auc_base = float(roc_auc_score(y, base_oof))
    auc_with = float(roc_auc_score(y, with_oof))
    delta_bp = (auc_with - auc_base) * 1e4
    pred_lb = pred_lb_band(delta_bp, rho)

    print(f"\n  --- Angle A gate ---")
    print(f"  standalone OOF: {oof_auc:.5f}")
    print(f"  K=4 LR-meta base OOF: {auc_base:.5f}")
    print(f"  K=4+1 LR-meta with-OOF: {auc_with:.5f}")
    print(f"  Δ min-meta: {delta_bp:+.3f} bp")
    print(f"  ρ vs PRIMARY: {rho:.5f}")
    print(f"  predicted LB Δ band: {pred_lb:+.2f} bp")
    print(f"  total wall: {(time.time()-t0)/60:.1f} min")

    return dict(
        angle="A_rf_yekenot_base",
        smoke=args.smoke,
        n_estimators=n_estimators,
        oof_auc=oof_auc,
        fold_aucs=fold_aucs,
        fold_walls_s=fold_walls,
        rho_vs_primary=rho,
        k4_lr_base_oof=auc_base,
        k4_lr_with_oof=auc_with,
        min_meta_delta_bp=float(delta_bp),
        predicted_lb_delta_bp=float(pred_lb),
        wall_min=(time.time() - t0) / 60,
    )


# --------------------------------------------------------------------- #
# Angle B — RF as meta on K=4 [P, rank, logit] (12 feat)                #
# --------------------------------------------------------------------- #
def angle_b(args, splits, y, k4_oof, k4_test, primary_test_pos):
    print("\n========== ANGLE B: RF meta-stacker on K=4 ==========")
    t0 = time.time()
    F_oof = expand(k4_oof)            # (n_train, 12)
    F_test = expand(k4_test)          # (n_test,  12)
    print(f"  meta features: {F_oof.shape[1]}")

    # Baselines
    lr_oof = lr_meta_oof(F_oof, y, splits)
    lr_oof_auc = float(roc_auc_score(y, lr_oof))
    print(f"  K=4 LR-meta baseline OOF: {lr_oof_auc:.5f}")

    n_est = 400 if not args.smoke else 200
    rf_oof = np.zeros(len(y))
    rf_test = np.zeros(len(F_test))
    for k, (tr, va) in enumerate(splits):
        t1 = time.time()
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_features="sqrt",
            min_samples_leaf=200,
            max_samples=0.4,
            n_jobs=args.n_jobs,
            random_state=SEED,
        )
        clf.fit(F_oof[tr], y[tr])
        rf_oof[va] = clf.predict_proba(F_oof[va])[:, 1]
        rf_test += clf.predict_proba(F_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], rf_oof[va]))
        print(f"  fold {k+1}/{N_FOLDS}: AUC={s:.5f}  "
              f"wall={time.time()-t1:.1f}s", flush=True)

    rf_oof_auc = float(roc_auc_score(y, rf_oof))
    delta_bp = (rf_oof_auc - lr_oof_auc) * 1e4
    rho, _ = spearmanr(rf_test, primary_test_pos)
    rho = float(rho)
    pred_lb = pred_lb_band(delta_bp, rho)

    np.save(ART / "oof_rf_meta_K4_strat.npy",
            np.column_stack([1 - rf_oof, rf_oof]).astype(np.float32))
    np.save(ART / "test_rf_meta_K4_strat.npy",
            np.column_stack([1 - rf_test, rf_test]).astype(np.float32))

    print(f"\n  --- Angle B gate ---")
    print(f"  RF-meta OOF: {rf_oof_auc:.5f}  vs LR-meta {lr_oof_auc:.5f}  "
          f"Δ {delta_bp:+.3f} bp")
    print(f"  ρ vs PRIMARY: {rho:.5f}")
    print(f"  predicted LB Δ vs LR-meta: {pred_lb:+.2f} bp")
    print(f"  wall: {(time.time()-t0)/60:.1f} min")

    return dict(
        angle="B_rf_meta_K4",
        smoke=args.smoke,
        n_estimators=n_est,
        rf_meta_oof=rf_oof_auc,
        lr_meta_oof=lr_oof_auc,
        delta_vs_lr_meta_bp=float(delta_bp),
        rho_vs_primary=rho,
        predicted_lb_delta_bp=float(pred_lb),
        wall_min=(time.time() - t0) / 60,
    )


# --------------------------------------------------------------------- #
# Angle C — RF on combined input (K=4 expansion + raw numerics)         #
# --------------------------------------------------------------------- #
def angle_c(args, splits, y, k4_oof, k4_test, primary_test_pos):
    print("\n========== ANGLE C: RF meta on combined input ==========")
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Top numerics (per Day-19 combined-input meta convention).
    raw_cols = ["LapNumber", "TyreLife", "RaceProgress",
                "LapTime (s)", "Position", "Stint"]
    raw_cols = [c for c in raw_cols if c in train.columns]
    raw_train = train[raw_cols].astype(np.float32).fillna(-1).values
    raw_test = test[raw_cols].astype(np.float32).fillna(-1).values

    # Standardize raw features on train (per-fold standardize is overkill
    # for tree-meta; trees are scale-invariant within fold).
    F_oof = np.hstack([expand(k4_oof), raw_train])
    F_test = np.hstack([expand(k4_test), raw_test])
    print(f"  combined-input meta features: {F_oof.shape[1]} "
          f"({F_oof.shape[1] - 12} raw)")

    # Baseline: LR on the SAME combined input
    lr_oof = lr_meta_oof(F_oof, y, splits)
    lr_oof_auc = float(roc_auc_score(y, lr_oof))
    print(f"  LR on combined input baseline OOF: {lr_oof_auc:.5f}")

    n_est = 400 if not args.smoke else 200
    rf_oof = np.zeros(len(y))
    rf_test = np.zeros(len(F_test))
    for k, (tr, va) in enumerate(splits):
        t1 = time.time()
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_features="sqrt",
            min_samples_leaf=200,
            max_samples=0.4,
            n_jobs=args.n_jobs,
            random_state=SEED,
        )
        clf.fit(F_oof[tr], y[tr])
        rf_oof[va] = clf.predict_proba(F_oof[va])[:, 1]
        rf_test += clf.predict_proba(F_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], rf_oof[va]))
        print(f"  fold {k+1}/{N_FOLDS}: AUC={s:.5f}  "
              f"wall={time.time()-t1:.1f}s", flush=True)

    rf_oof_auc = float(roc_auc_score(y, rf_oof))
    delta_vs_lr = (rf_oof_auc - lr_oof_auc) * 1e4
    rho, _ = spearmanr(rf_test, primary_test_pos)
    rho = float(rho)
    pred_lb = pred_lb_band(delta_vs_lr, rho)

    # Comparison vs the original K=4 LR-meta (pure base predictions only)
    F_base = expand(k4_oof)
    base_lr_oof = lr_meta_oof(F_base, y, splits)
    base_lr_oof_auc = float(roc_auc_score(y, base_lr_oof))
    delta_vs_pure_lr = (rf_oof_auc - base_lr_oof_auc) * 1e4

    np.save(ART / "oof_rf_combined_K4_strat.npy",
            np.column_stack([1 - rf_oof, rf_oof]).astype(np.float32))
    np.save(ART / "test_rf_combined_K4_strat.npy",
            np.column_stack([1 - rf_test, rf_test]).astype(np.float32))

    print(f"\n  --- Angle C gate ---")
    print(f"  RF-combined OOF: {rf_oof_auc:.5f}  vs LR-combined "
          f"{lr_oof_auc:.5f}  Δ {delta_vs_lr:+.3f} bp")
    print(f"  ... vs pure-K=4 LR-meta {base_lr_oof_auc:.5f}  "
          f"Δ {delta_vs_pure_lr:+.3f} bp")
    print(f"  ρ vs PRIMARY: {rho:.5f}")
    print(f"  predicted LB Δ band: {pred_lb:+.2f} bp")
    print(f"  wall: {(time.time()-t0)/60:.1f} min")

    return dict(
        angle="C_rf_combined_K4",
        smoke=args.smoke,
        n_estimators=n_est,
        rf_oof=rf_oof_auc,
        lr_combined_oof=lr_oof_auc,
        pure_k4_lr_meta_oof=base_lr_oof_auc,
        delta_vs_lr_combined_bp=float(delta_vs_lr),
        delta_vs_pure_k4_lr_bp=float(delta_vs_pure_lr),
        rho_vs_primary=rho,
        predicted_lb_delta_bp=float(pred_lb),
        wall_min=(time.time() - t0) / 60,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", nargs="+", default=["A", "B", "C"],
                    choices=["A", "B", "C"])
    ap.add_argument("--smoke", action="store_true",
                    help="1-fold smoke for Angle A; smaller n_estimators")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-json",
                    default="scripts/artifacts/probe_forest_sweep.json")
    args = ap.parse_args()

    t_total = time.time()

    train = pd.read_csv("data/train.csv", usecols=[TARGET])
    y = train[TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    k4_oof, k4_test = load_k4_pool()
    print(f"K=4 pool loaded: oof {k4_oof.shape}  test {k4_test.shape}")
    print(f"  bases: {K4_BASES}")

    # PRIMARY-test reference for ρ comparisons. Use Angle B's RF-meta?
    # No — use the LR-meta on K=4 expansion as the canonical PRIMARY-test
    # surface. That's the meta the team's PRIMARY would absorb a new
    # base through.
    F_test = expand(k4_test)
    F_oof = expand(k4_oof)
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F_oof, y)
    primary_test_pos = lr.predict_proba(F_test)[:, 1]
    print(f"  PRIMARY-test ref (LR-on-K=4-expansion) shape "
          f"{primary_test_pos.shape}")

    results = {}
    if "A" in args.angles:
        results["angle_a"] = angle_a(args, splits, y, k4_oof, k4_test,
                                     primary_test_pos)
    if "B" in args.angles:
        results["angle_b"] = angle_b(args, splits, y, k4_oof, k4_test,
                                     primary_test_pos)
    if "C" in args.angles:
        results["angle_c"] = angle_c(args, splits, y, k4_oof, k4_test,
                                     primary_test_pos)

    results["total_wall_min"] = (time.time() - t_total) / 60
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out_json}")
    print(f"Total wall: {results['total_wall_min']:.1f} min")


if __name__ == "__main__":
    main()
