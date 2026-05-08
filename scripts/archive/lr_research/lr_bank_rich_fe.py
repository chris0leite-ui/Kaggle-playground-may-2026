"""scripts/lr_bank_rich_fe.py — RICH feature engineering LR bank.

Critique of lr_bank.py: all 15 variants used the SAME 11 numerics + 3 cats,
varying only the transform. The eff_rank=2 collapse was an artifact of
that pipeline limit, not an LR-class ceiling.

This script tests pure LR with the FULL FE arsenal already in the repo:
  - Rozen kitchen-sink (`p1_features.make_features_static + fit_fs_a`)
  - Yekenot full recipe (`d17_h1d_yekenot_full_recipe.feature_engineering`)
  - DGP rule lookups (Bayesian-smoothed from `d6_multi_rule.fit_lookup`)
  - 3-way target encoding sweep (multiple keys × multiple smoothings)
  - lr_mega: union of ALL above + KBins(20)+OHE-cats — the "pure-LR
    ceiling" probe answering "how good can a simple LR get?"

Each variant: 5-fold StratifiedKFold, fold-safe label-conditional FE.
Saves oof_<NAME>_strat.npy + test_<NAME>_strat.npy in the standard
2-column [P0, P1] format (n_train, 2) and (n_test, 2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    KBinsDiscretizer, OneHotEncoder, StandardScaler,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from p1_features import (
    make_features_static, fit_fs_a, apply_fs_a, cv_target_encode, TE_CONFIGS,
)

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

NUM_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
]
CAT_COLS = ["Driver", "Compound", "Race"]


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    return train, test, y


# -----------------------------------------------------------------------------
# DGP rule lookup: Bayesian-smoothed P(y=1|key) for given key columns + alpha
# -----------------------------------------------------------------------------

def build_dgp_rule_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: np.ndarray,
    tr_idx: np.ndarray, va_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-fold DGP rule features. Fits lookups on tr-rows ONLY.

    Rules: (Compound, Stint), (Driver, Compound), (Year, Race),
    (Compound, TyreLife_decile). 4 rules × 4 alphas = 16 features.

    Returns (tr_features, va_features, te_features) all leak-free.
    """
    rules = [
        ("Compound", "Stint"),
        ("Driver", "Compound"),
        ("Year", "Race"),
    ]

    # Add Compound × TyreLife-decile rule (decile fit on tr-rows only)
    tr_tl = train_df["TyreLife"].iloc[tr_idx].values
    edges = np.quantile(tr_tl, np.linspace(0, 1, 11))
    edges[0] = -np.inf
    edges[-1] = np.inf

    def tyre_decile(arr):
        return np.clip(np.searchsorted(edges, arr, side="right") - 1, 0, 9)

    def make_keys(df):
        keys = {}
        for cols in rules:
            keys[cols] = list(zip(*[df[c].astype(str).values for c in cols]))
        # add compound × tyre_decile
        keys[("Compound", "TyreDecile")] = list(zip(
            df["Compound"].astype(str).values,
            tyre_decile(df["TyreLife"].values).astype(str),
        ))
        return keys

    keys_train = make_keys(train_df)
    keys_test = make_keys(test_df)

    glob = float(y_train[tr_idx].mean())
    alphas = [5, 20, 100, 500]

    tr_cols, va_cols, te_cols = [], [], []
    for rule_cols, all_keys in keys_train.items():
        keys_tr = [all_keys[i] for i in tr_idx]
        keys_va = [all_keys[i] for i in va_idx]
        keys_te = keys_test[rule_cols]

        # Aggregate sum + count per key on tr-rows
        df = pd.DataFrame({"k": keys_tr, "y": y_train[tr_idx]})
        g = df.groupby("k", observed=True)["y"]
        counts = g.count()
        sums = g.sum()

        for alpha in alphas:
            smoothed = (sums + alpha * glob) / (counts + alpha)
            mp = smoothed.to_dict()
            tr_cols.append(np.array([mp.get(k, glob) for k in keys_tr],
                                    dtype=np.float32))
            va_cols.append(np.array([mp.get(k, glob) for k in keys_va],
                                    dtype=np.float32))
            te_cols.append(np.array([mp.get(k, glob) for k in keys_te],
                                    dtype=np.float32))

    return (np.column_stack(tr_cols),
            np.column_stack(va_cols),
            np.column_stack(te_cols))


# -----------------------------------------------------------------------------
# Yekenot recipe FE — fold-safe port of items 1-4+6
# Items: arithmetic, floor-cat, count-enc, KBins(200/RP)+KBins(7/LT), combo-cats,
# CV-TE on (Race, Compound) and (Race, Year) inside each fold.
# Item 5 (orig-aug) skipped — orig dataset not available locally.
# -----------------------------------------------------------------------------

def build_yekenot_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build yekenot's cell 6 features ONCE (fits use full train+test).

    Returns (X_full_train_yek, X_test_yek, num_cols_yek, cat_cols_yek).
    Items 1-4+6 (orig-aug skipped). Item 6 (CV-TE) handled per-fold by
    cv_lr loop; here we just emit the static columns.
    """
    X = train_df.drop(columns=[TARGET, "id"], errors="ignore").copy()
    X_test = test_df.drop(columns=["id"], errors="ignore").copy()

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # 1) Arithmetic
    for df in (X, X_test):
        df["_LapNumber_/_RaceProgress"] = (
            df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
        df["_TyreLife_/_LapNumber"] = (
            df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")

    # 2) Floor-cat
    for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        codes_train, uniques = np.floor(X[col]).factorize()
        code_map = {cat: i for i, cat in enumerate(uniques)}
        codes_test = np.floor(X_test[col]).map(code_map).fillna(-1).astype("int32")
        X[cat_name] = codes_train.astype("int32").astype(str)
        X_test[cat_name] = codes_test.astype("int32").astype(str)

    # 3) Count enc
    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        count_map = X[col].value_counts()  # train counts (label-free)
        X[count_name] = X[col].map(count_map).fillna(0).astype("int32")
        X_test[count_name] = X_test[col].map(count_map).fillna(0).astype("int32")

    # 4) KBins(200/RP) + KBins(7/LT) — fit on combined train+test (Rule 25 safe)
    bin_config = {"RaceProgress": 200, "LapTime (s)": 7}
    for col, n_bins in bin_config.items():
        bin_name = f"{col}_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile",
                              subsample=None)
        combined = np.vstack([X[[col]].values, X_test[[col]].values])
        kb.fit(combined)
        X[bin_name] = kb.transform(X[[col]]).ravel().astype("int32").astype(str)
        X_test[bin_name] = kb.transform(X_test[[col]]).ravel().astype("int32").astype(str)

    # 5) Combo cats (Race×Compound, Race×Year)
    for cols in [("Race", "Compound"), ("Race", "Year")]:
        combo_name = "_".join(cols) + "_"
        combo_train = X[cols[0]].astype(str)
        combo_test = X_test[cols[0]].astype(str)
        for col in cols[1:]:
            combo_train = combo_train + "_" + X[col].astype(str)
            combo_test = combo_test + "_" + X_test[col].astype(str)
        codes, uniques = pd.factorize(combo_train, sort=False)
        code_map = {cat: i for i, cat in enumerate(uniques)}
        X[combo_name] = codes.astype("int32").astype(str)
        X_test[combo_name] = combo_test.map(code_map).fillna(-1).astype("int32").astype(str)

    new_cat = [c for c in X.columns if c.endswith("_")]
    new_num = [c for c in X.columns if c.startswith("_") and not c.endswith("_")]
    return X, X_test, num_cols + new_num, cat_cols + new_cat


# -----------------------------------------------------------------------------
# Rozen kitchen-sink — wraps make_features_static + per-fold fit_fs_a + 6 CV TE
# -----------------------------------------------------------------------------

def build_rozen_static(train_df: pd.DataFrame, test_df: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run make_features_static once on combined (state shared) — label-free."""
    state: dict = {}
    train_S, state = make_features_static(train_df, fit=True, state=state)
    test_S, _ = make_features_static(test_df, fit=False, state=state)
    return train_S, test_S, state


# -----------------------------------------------------------------------------
# Variant runners: each does a 5-fold CV with its own fold-safe FE
# -----------------------------------------------------------------------------

def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns (drop object/category for LR)."""
    return df.select_dtypes(include=[np.number]).fillna(0)


def fit_lr(X_tr, y_tr, X_va, X_te, lr_kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fit LR + return (val_pred, test_pred) probs."""
    lr = LogisticRegression(**lr_kwargs)
    lr.fit(X_tr, y_tr)
    return (lr.predict_proba(X_va)[:, 1].astype(np.float64),
            lr.predict_proba(X_te)[:, 1].astype(np.float64))


def run_lr_rozen_full(name: str, train: pd.DataFrame, test: pd.DataFrame,
                      y: np.ndarray) -> dict:
    """Pure-LR on Rozen kitchen-sink (make_features_static + fit_fs_a + 6 CV TE)."""
    print(f"  [{name}] building static Rozen features...", flush=True)
    t0 = time.time()
    train_S, test_S, _ = build_rozen_static(train, test)

    # Drop heavy-card cat columns; keep their *_cat int codes
    drop_cols = ["Driver", "Race", "Compound", "id", TARGET]
    # We keep id-derived int codes that make_features_static built ("_cat" suffix)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    # Pre-compute the 6 TE features (using TE_CONFIGS); these are fold-safe
    print(f"  [{name}] computing 6 CV TE features...", flush=True)
    te_oof = {}
    te_test = {}
    for cols, smoothing, te_name in TE_CONFIGS:
        oof_enc, test_enc = cv_target_encode(
            train, test, cols, train[TARGET].astype(int), fold_list, smoothing
        )
        te_oof[te_name] = oof_enc
        te_test[te_name] = test_enc

    # Per-fold loop: fit FS_A on tr-rows, apply to all
    print(f"  [{name}] running 5-fold CV with per-fold FS_A...", flush=True)
    n_tr, n_te = len(y), len(test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_pred = np.zeros(n_te, dtype=np.float64)

    for k, (tr, va) in enumerate(fold_list):
        ts = time.time()
        # Fit FS_A on tr-rows of TRAIN (with labels)
        fs_a = fit_fs_a(train.iloc[tr])
        train_A = apply_fs_a(train_S, fs_a)
        test_A = apply_fs_a(test_S, fs_a)

        # Numeric-only matrices (drop raw cat strings; *_cat ints stay)
        feat_cols = [c for c in train_A.columns if c not in drop_cols
                     and c not in CAT_COLS and train_A[c].dtype.kind in "biufc"]
        Xtr = train_A[feat_cols].fillna(0).values.astype(np.float32)
        Xte = test_A[feat_cols].fillna(0).values.astype(np.float32)

        # Append the 6 TE features (already fold-safe via cv_target_encode)
        te_train_arr = np.column_stack([te_oof[k] for k in te_oof])
        te_test_arr = np.column_stack([te_test[k] for k in te_test])
        Xtr = np.hstack([Xtr, te_train_arr])
        Xte = np.hstack([Xte, te_test_arr])

        # StandardScaler fit on tr-rows of this fold
        sc = StandardScaler()
        Xtr_tr = sc.fit_transform(Xtr[tr])
        Xtr_va = sc.transform(Xtr[va])
        Xte_s = sc.transform(Xte)

        val_p, test_p = fit_lr(Xtr_tr, y[tr], Xtr_va, Xte_s,
                               dict(C=1.0, max_iter=2000, solver="lbfgs"))
        oof[va] = val_p
        test_pred += test_p / N_FOLDS
        auc_fold = float(roc_auc_score(y[va], val_p))
        print(f"    fold {k}: AUC {auc_fold:.5f}  ({time.time()-ts:.1f}s, "
              f"{Xtr_tr.shape[1]} feats)", flush=True)

    elapsed = time.time() - t0
    auc = float(roc_auc_score(y, oof))
    print(f"  [{name}] OOF AUC {auc:.5f}  ({elapsed:.1f}s)", flush=True)
    return dict(name=name, auc=auc, time=elapsed,
                n_features=int(Xtr_tr.shape[1]), oof=oof, test=test_pred)


def run_lr_yekenot_full(name: str, train: pd.DataFrame, test: pd.DataFrame,
                        y: np.ndarray) -> dict:
    """Yekenot recipe items 1-4+6 (orig-aug item 5 skipped — no orig data)."""
    from sklearn.preprocessing import TargetEncoder
    print(f"  [{name}] building yekenot features...", flush=True)
    t0 = time.time()
    X_yek, X_test_yek, num_cols_yek, cat_cols_yek = build_yekenot_features(
        train, test, y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_tr, n_te = len(y), len(test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_pred = np.zeros(n_te, dtype=np.float64)

    print(f"  [{name}] running 5-fold CV (yek + per-fold CV TE) ...", flush=True)
    for k, (tr, va) in enumerate(skf.split(np.zeros(n_tr), y)):
        ts = time.time()
        # CV TE on (Race, Compound) and (Race, Year) — fit on tr-rows only
        # Use sklearn TargetEncoder (smooth='auto')
        combo_cols = ["Race_Compound_", "Race_Year_"]
        te = TargetEncoder(cv=5, smooth="auto", target_type="binary",
                           random_state=SEED)
        te_train_va = X_yek.iloc[va][combo_cols].astype(str).values
        te_train_tr = X_yek.iloc[tr][combo_cols].astype(str).values
        te_test_arr = X_test_yek[combo_cols].astype(str).values

        te_oof_train = te.fit_transform(te_train_tr, y[tr])  # fold-safe internal CV
        te_va = te.transform(te_train_va)
        te_te = te.transform(te_test_arr)

        # Build numeric matrix: numeric cols + count cols (numeric) + KBins/floor cats as ordinal
        all_num_cols = (num_cols_yek
                        + [c for c in X_yek.columns if c.startswith("_") and c.endswith("_count")])
        cat_to_int = {c: pd.to_numeric(X_yek[c], errors="coerce").fillna(-1).astype(np.int32)
                      for c in cat_cols_yek}
        cat_to_int_test = {c: pd.to_numeric(X_test_yek[c], errors="coerce").fillna(-1).astype(np.int32)
                           for c in cat_cols_yek}

        feat_train = X_yek[all_num_cols].fillna(0).values.astype(np.float32)
        feat_test = X_test_yek[all_num_cols].fillna(0).values.astype(np.float32)
        # Append cat-as-ordinal
        feat_train = np.hstack([feat_train,
                                np.column_stack([cat_to_int[c].values
                                                 for c in cat_cols_yek])])
        feat_test = np.hstack([feat_test,
                               np.column_stack([cat_to_int_test[c].values
                                                for c in cat_cols_yek])])
        # Append CV TE on combo cats
        feat_train_te = np.zeros((len(y), te_oof_train.shape[1]), dtype=np.float32)
        feat_train_te[tr] = te_oof_train
        feat_train_te[va] = te_va
        feat_train = np.hstack([feat_train, feat_train_te])
        feat_test = np.hstack([feat_test, te_te])

        sc = StandardScaler()
        Xtr_tr = sc.fit_transform(feat_train[tr])
        Xtr_va = sc.transform(feat_train[va])
        Xte_s = sc.transform(feat_test)

        val_p, test_p = fit_lr(Xtr_tr, y[tr], Xtr_va, Xte_s,
                               dict(C=1.0, max_iter=2000, solver="lbfgs"))
        oof[va] = val_p
        test_pred += test_p / N_FOLDS
        auc_fold = float(roc_auc_score(y[va], val_p))
        print(f"    fold {k}: AUC {auc_fold:.5f}  ({time.time()-ts:.1f}s, "
              f"{Xtr_tr.shape[1]} feats)", flush=True)

    elapsed = time.time() - t0
    auc = float(roc_auc_score(y, oof))
    print(f"  [{name}] OOF AUC {auc:.5f}  ({elapsed:.1f}s)", flush=True)
    return dict(name=name, auc=auc, time=elapsed,
                n_features=int(Xtr_tr.shape[1]), oof=oof, test=test_pred)


def run_lr_dgp_rules(name: str, train: pd.DataFrame, test: pd.DataFrame,
                     y: np.ndarray) -> dict:
    """LR on DGP rule lookups (4 rules × 4 alphas = 16 features) + raw + cat OHE."""
    print(f"  [{name}] running 5-fold CV with per-fold rule lookups...", flush=True)
    t0 = time.time()

    num_tr_raw = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te_raw = test[NUM_COLS].fillna(0).values.astype(np.float32)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    Oc_tr = enc.transform(train[CAT_COLS])
    Oc_te = enc.transform(test[CAT_COLS])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_tr, n_te = len(y), len(test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_pred = np.zeros(n_te, dtype=np.float64)

    for k, (tr, va) in enumerate(skf.split(np.zeros(n_tr), y)):
        ts = time.time()
        rule_tr, rule_va, rule_te = build_dgp_rule_features(
            train, test, y, tr, va)

        # Standardize numerics + rule features per fold
        num_combined_tr = np.hstack([num_tr_raw[tr], rule_tr])
        num_combined_va = np.hstack([num_tr_raw[va], rule_va])
        num_combined_te = np.hstack([num_te_raw, rule_te])
        sc = StandardScaler()
        num_combined_tr = sc.fit_transform(num_combined_tr)
        num_combined_va = sc.transform(num_combined_va)
        num_combined_te = sc.transform(num_combined_te)

        Xtr_tr = sparse.hstack([sparse.csr_matrix(num_combined_tr), Oc_tr[tr]],
                               format="csr")
        Xtr_va = sparse.hstack([sparse.csr_matrix(num_combined_va), Oc_tr[va]],
                               format="csr")
        Xte_s = sparse.hstack([sparse.csr_matrix(num_combined_te), Oc_te],
                              format="csr")

        val_p, test_p = fit_lr(Xtr_tr, y[tr], Xtr_va, Xte_s,
                               dict(C=1.0, max_iter=2000, solver="liblinear"))
        oof[va] = val_p
        test_pred += test_p / N_FOLDS
        auc_fold = float(roc_auc_score(y[va], val_p))
        print(f"    fold {k}: AUC {auc_fold:.5f}  ({time.time()-ts:.1f}s, "
              f"{Xtr_tr.shape[1]} feats)", flush=True)

    elapsed = time.time() - t0
    auc = float(roc_auc_score(y, oof))
    print(f"  [{name}] OOF AUC {auc:.5f}  ({elapsed:.1f}s)", flush=True)
    return dict(name=name, auc=auc, time=elapsed,
                n_features=int(Xtr_tr.shape[1]), oof=oof, test=test_pred)


def run_lr_te_3way_sweep(name: str, train: pd.DataFrame, test: pd.DataFrame,
                          y: np.ndarray) -> dict:
    """3-way TE on multiple keys × multiple smoothings."""
    print(f"  [{name}] running 3-way TE sweep...", flush=True)
    t0 = time.time()

    keys_3way = [
        ["Driver", "Race", "Year"],
        ["Driver", "Race", "Compound"],
        ["Driver", "Year", "Compound"],
        ["Race", "Year", "Compound"],
    ]
    smoothings = [1, 5, 20, 100]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    # Pre-compute all (keys × smoothing) TEs (each is fold-safe)
    print(f"  [{name}] computing {len(keys_3way) * len(smoothings)} TE features...",
          flush=True)
    all_oof, all_test = [], []
    for keys in keys_3way:
        for sm in smoothings:
            oof_enc, test_enc = cv_target_encode(
                train, test, keys, train[TARGET].astype(int), fold_list, sm)
            all_oof.append(oof_enc)
            all_test.append(test_enc)
    te_oof_arr = np.column_stack(all_oof).astype(np.float32)
    te_te_arr = np.column_stack(all_test).astype(np.float32)
    print(f"  [{name}] TE matrix: {te_oof_arr.shape}", flush=True)

    num_tr_raw = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te_raw = test[NUM_COLS].fillna(0).values.astype(np.float32)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    Oc_tr = enc.transform(train[CAT_COLS])
    Oc_te = enc.transform(test[CAT_COLS])

    n_tr, n_te = len(y), len(test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_pred = np.zeros(n_te, dtype=np.float64)

    for k, (tr, va) in enumerate(fold_list):
        ts = time.time()
        num_tr_full = np.hstack([num_tr_raw, te_oof_arr])
        num_te_full = np.hstack([num_te_raw, te_te_arr])
        sc = StandardScaler()
        num_tr_s = sc.fit_transform(num_tr_full[tr])
        num_va_s = sc.transform(num_tr_full[va])
        num_te_s = sc.transform(num_te_full)

        Xtr_tr = sparse.hstack([sparse.csr_matrix(num_tr_s), Oc_tr[tr]], format="csr")
        Xtr_va = sparse.hstack([sparse.csr_matrix(num_va_s), Oc_tr[va]], format="csr")
        Xte_s = sparse.hstack([sparse.csr_matrix(num_te_s), Oc_te], format="csr")

        val_p, test_p = fit_lr(Xtr_tr, y[tr], Xtr_va, Xte_s,
                               dict(C=1.0, max_iter=2000, solver="liblinear"))
        oof[va] = val_p
        test_pred += test_p / N_FOLDS
        auc_fold = float(roc_auc_score(y[va], val_p))
        print(f"    fold {k}: AUC {auc_fold:.5f}  ({time.time()-ts:.1f}s)",
              flush=True)

    elapsed = time.time() - t0
    auc = float(roc_auc_score(y, oof))
    print(f"  [{name}] OOF AUC {auc:.5f}  ({elapsed:.1f}s)", flush=True)
    return dict(name=name, auc=auc, time=elapsed,
                n_features=int(Xtr_tr.shape[1]), oof=oof, test=test_pred)


def run_lr_mega(name: str, train: pd.DataFrame, test: pd.DataFrame,
                y: np.ndarray, n_seeds: int = 1) -> dict:
    """The 'pure-LR ceiling' probe: ALL FE families concatenated.

    Rozen static + 6 CV TE + 3-way TE sweep + DGP rule lookups +
    KBins(20)+OHE-cats. Optional bagging across n_seeds for variance reduction
    (n_seeds=5 typically gives +1-3 bp).
    """
    print(f"  [{name}] building Rozen static features...", flush=True)
    t0 = time.time()
    train_S, test_S, _ = build_rozen_static(train, test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    # Pre-compute 6 Rozen TE features
    print(f"  [{name}] computing 6 Rozen CV TE features...", flush=True)
    rozen_te_oof = {}
    rozen_te_test = {}
    for cols, smoothing, te_name in TE_CONFIGS:
        oof_enc, test_enc = cv_target_encode(
            train, test, cols, train[TARGET].astype(int), fold_list, smoothing)
        rozen_te_oof[te_name] = oof_enc
        rozen_te_test[te_name] = test_enc

    # Pre-compute 3-way TE sweep (4 keys × 4 smoothings = 16)
    print(f"  [{name}] computing 16 3-way TE features...", flush=True)
    keys_3way = [["Driver", "Race", "Year"], ["Driver", "Race", "Compound"],
                 ["Driver", "Year", "Compound"], ["Race", "Year", "Compound"]]
    smoothings = [1, 5, 20, 100]
    threeway_oof, threeway_test = [], []
    for keys in keys_3way:
        for sm in smoothings:
            oof_enc, test_enc = cv_target_encode(
                train, test, keys, train[TARGET].astype(int), fold_list, sm)
            threeway_oof.append(oof_enc)
            threeway_test.append(test_enc)

    # KBins(20)+cat OHE
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

    print(f"  [{name}] running 5-fold CV with per-fold FS_A + DGP rules...",
          flush=True)
    drop_cols_static = ["Driver", "Race", "Compound", "id", TARGET]

    n_tr, n_te = len(y), len(test)
    oof_seeds = []
    test_seeds = []

    for seed_idx in range(n_seeds):
        # Reseed only the StratifiedKFold ordering for bagging
        sk = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=SEED + seed_idx)
        fold_list_seed = list(sk.split(np.zeros(len(y)), y))
        oof = np.zeros(n_tr, dtype=np.float64)
        test_pred = np.zeros(n_te, dtype=np.float64)
        for k, (tr, va) in enumerate(fold_list_seed):
            ts = time.time()
            # Per-fold FS_A
            fs_a = fit_fs_a(train.iloc[tr])
            train_A = apply_fs_a(train_S, fs_a)
            test_A = apply_fs_a(test_S, fs_a)

            feat_cols = [c for c in train_A.columns if c not in drop_cols_static
                         and c not in CAT_COLS and train_A[c].dtype.kind in "biufc"]
            Xstatic_tr = train_A[feat_cols].fillna(0).values.astype(np.float32)
            Xstatic_te = test_A[feat_cols].fillna(0).values.astype(np.float32)

            # Per-fold DGP rules
            rule_tr, rule_va, rule_te = build_dgp_rule_features(
                train, test, y, tr, va)

            # Combine all dense numerics
            te_train_arr = np.column_stack(list(rozen_te_oof.values()))
            te_test_arr = np.column_stack(list(rozen_te_test.values()))
            tw_train_arr = np.column_stack(threeway_oof)
            tw_test_arr = np.column_stack(threeway_test)

            num_tr_full = np.hstack([Xstatic_tr[tr], te_train_arr[tr],
                                     tw_train_arr[tr], rule_tr])
            num_va_full = np.hstack([Xstatic_tr[va], te_train_arr[va],
                                     tw_train_arr[va], rule_va])
            num_te_full = np.hstack([Xstatic_te, te_test_arr,
                                     tw_test_arr, rule_te])

            sc = StandardScaler()
            num_tr_s = sc.fit_transform(num_tr_full)
            num_va_s = sc.transform(num_va_full)
            num_te_s = sc.transform(num_te_full)

            # Densify the whole stack so lbfgs (BLAS-multi-threaded) can
            # work — sparse-dense mix kills liblinear/saga on this size.
            # 350k * 461 * 4B = 645 MB; fits in our 16 GB RAM.
            Xtr_tr = np.hstack([num_tr_s, Bk_tr[tr].toarray().astype(np.float32),
                                Oc_tr[tr].toarray().astype(np.float32)])
            Xtr_va = np.hstack([num_va_s, Bk_tr[va].toarray().astype(np.float32),
                                Oc_tr[va].toarray().astype(np.float32)])
            Xte_s = np.hstack([num_te_s, Bk_te.toarray().astype(np.float32),
                               Oc_te.toarray().astype(np.float32)])

            val_p, test_p = fit_lr(Xtr_tr, y[tr], Xtr_va, Xte_s,
                                   dict(C=1.0, max_iter=2000, solver="lbfgs"))
            oof[va] = val_p
            test_pred += test_p / N_FOLDS
            auc_fold = float(roc_auc_score(y[va], val_p))
            print(f"    seed {seed_idx} fold {k}: AUC {auc_fold:.5f}  "
                  f"({time.time()-ts:.1f}s, {Xtr_tr.shape[1]} feats)", flush=True)
        oof_seeds.append(oof)
        test_seeds.append(test_pred)
        auc_seed = float(roc_auc_score(y, oof))
        print(f"  [{name}] seed {seed_idx} OOF AUC {auc_seed:.5f}", flush=True)

    # Bag across seeds (rank average for stability)
    if n_seeds > 1:
        from scipy.stats import rankdata
        n_train = len(y)
        oof_rank = np.zeros(n_train, dtype=np.float64)
        for o in oof_seeds:
            oof_rank += rankdata(o) / n_train
        oof = oof_rank / n_seeds
        test_rank = np.zeros(n_te, dtype=np.float64)
        for t in test_seeds:
            test_rank += rankdata(t) / n_te
        test_pred = test_rank / n_seeds
    else:
        oof = oof_seeds[0]
        test_pred = test_seeds[0]

    elapsed = time.time() - t0
    auc = float(roc_auc_score(y, oof))
    print(f"  [{name}] OOF AUC {auc:.5f}  ({elapsed:.1f}s, n_seeds={n_seeds})",
          flush=True)
    return dict(name=name, auc=auc, time=elapsed,
                n_features=int(Xtr_tr.shape[1]), n_seeds=n_seeds,
                oof=oof, test=test_pred)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

VARIANTS = {
    "lr_rozen_full": run_lr_rozen_full,
    "lr_yekenot_full_recipe": run_lr_yekenot_full,
    "lr_dgp_rules": run_lr_dgp_rules,
    "lr_te_3way_sweep": run_lr_te_3way_sweep,
    "lr_mega": run_lr_mega,  # n_seeds=1 default; pass --mega-seeds for bagging
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=None,
                    help="comma-separated variant names; else --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--mega-seeds", type=int, default=1,
                    help="bag the lr_mega variant across N seeds (default 1)")
    args = ap.parse_args()

    train, test, y = load_data()
    print(f"data: train {train.shape}, test {test.shape}, prior {y.mean():.4f}",
          flush=True)

    if args.names:
        variants = [v.strip() for v in args.names.split(",")]
    else:
        variants = list(VARIANTS.keys())

    summary = []
    for v in variants:
        if v not in VARIANTS:
            print(f"  unknown variant: {v}", flush=True)
            continue
        suffix = "" if (v != "lr_mega" or args.mega_seeds == 1) else f"_bag{args.mega_seeds}"
        out_name = f"{v}{suffix}"
        if args.skip_existing and (ART / f"oof_{out_name}_strat.npy").exists():
            print(f"  skipping {out_name} (exists)", flush=True)
            continue
        print(f"\n=== running {out_name} ===", flush=True)
        try:
            if v == "lr_mega":
                res = VARIANTS[v](out_name, train, test, y, n_seeds=args.mega_seeds)
            else:
                res = VARIANTS[v](out_name, train, test, y)
            oof2 = np.column_stack([1 - res["oof"], res["oof"]])
            test2 = np.column_stack([1 - res["test"], res["test"]])
            np.save(ART / f"oof_{out_name}_strat.npy", oof2)
            np.save(ART / f"test_{out_name}_strat.npy", test2)
            summary.append({k: res[k] for k in res
                            if k not in ("oof", "test")})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            summary.append({"name": out_name, "error": str(e)})

    out_json = ART / "lr_bank_rich_fe_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n→ summary saved: {out_json}", flush=True)


if __name__ == "__main__":
    main()
