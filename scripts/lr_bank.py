"""scripts/lr_bank.py — build a wide bank of logistic-regression bases.

Inspired by Chris Deotte's 2nd-place s6e4 LR-stacker (125-base bank,
GPU PyTorch multinomial LR + class weights + L2 + forward selection).
Goal here is LEARNING: how much diversity can LR (with varied feature
engineering) add to our K=24 GBDT-dominated pool, whose effective rank
is only 2.88 (Arc-A E1)?

Each variant fits a 5-fold StratifiedKFold logistic regression and
saves OOF + test artifacts to scripts/artifacts/oof_<NAME>_strat.npy
and test_<NAME>_strat.npy in the standard 2-column [P0, P1] format.

Variants are organised by tier (A-J). Run subset via --names a,b,c,...
or --all; --tiers A,B,C selects all in those tiers.

Rule 24 (fold-safe label-conditional aggregates): TE done inside fold.
Rule 25 (transductive features need AV check): combined-set transforms
are safe on s6e5 because train/test AV-AUC = 0.502 (per CLAUDE.md U3).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

warnings.filterwarnings("ignore")

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

# All numeric features available; LapTime (s) requires escaping in DataFrame access
NUM_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
]
CAT_COLS = ["Driver", "Compound", "Race"]


# -----------------------------------------------------------------------------
# Loading + simple FE primitives
# -----------------------------------------------------------------------------

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    return train, test, y


def standardize_dense(X_tr: np.ndarray, X_te: np.ndarray):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    return X_tr_s, X_te_s


def freq_encode(train_cat: pd.Series, test_cat: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = train_cat.value_counts()
    return train_cat.map(counts).fillna(0).values, test_cat.map(counts).fillna(0).values


def fit_te_inside_fold(tr_y: np.ndarray, tr_cat: pd.Series, va_cat: pd.Series,
                       prior: float, smoothing: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed CV-safe target encoding. tr arrays are TRAIN ROWS ONLY in this fold.
    Returns (tr_encoded, va_encoded).
    """
    df = pd.DataFrame({"c": tr_cat.values, "y": tr_y})
    g = df.groupby("c")["y"].agg(["mean", "count"])
    smoothed = (g["count"] * g["mean"] + smoothing * prior) / (g["count"] + smoothing)
    tr_enc = tr_cat.map(smoothed).fillna(prior).values.astype(np.float32)
    va_enc = va_cat.map(smoothed).fillna(prior).values.astype(np.float32)
    return tr_enc, va_enc


def kbins_combined(num_tr: np.ndarray, num_te: np.ndarray, n_bins: int, strategy: str):
    """KBins fit on combined train+test (Rule 25 safe per AV-AUC=0.502).
    Returns one-hot sparse (csr) for train and test.
    """
    kb = KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy=strategy, subsample=None)
    combined = np.vstack([num_tr, num_te])
    kb.fit(combined)
    return kb.transform(num_tr), kb.transform(num_te)


def ohe_cats(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list[str]):
    """One-hot encoding fit on combined train+test (handles unseen)."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    combined = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0)
    enc.fit(combined)
    return enc.transform(train_df[cat_cols]), enc.transform(test_df[cat_cols])


def hash_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  combos: list[tuple[str, ...]], n_features: int):
    """Hash interactions (combos = list of tuples of column names) into n_features buckets."""
    def to_token_lists(df):
        out = []
        for _, row in df[CAT_COLS].iterrows():
            tokens = []
            for combo in combos:
                key = "|".join(f"{c}={row[c]}" for c in combo)
                tokens.append(key)
            out.append(tokens)
        return out

    hasher = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)
    return hasher.transform(to_token_lists(train_df)), hasher.transform(to_token_lists(test_df))


def hash_features_vec(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      combos: list[tuple[str, ...]], n_features: int):
    """Faster vectorised version using groupby-style string concat."""
    def df_to_tokens(df):
        cols = []
        for combo in combos:
            cat = df[combo[0]].astype(str)
            for c in combo[1:]:
                cat = cat + "|" + df[c].astype(str)
            cols.append(cat.values)
        # Each row becomes len(combos) tokens
        out = [list(z) for z in zip(*cols)]
        return out

    hasher = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)
    return hasher.transform(df_to_tokens(train_df)), hasher.transform(df_to_tokens(test_df))


# -----------------------------------------------------------------------------
# CV trainer (handles dense, sparse, and TE that needs fold-injection)
# -----------------------------------------------------------------------------

def cv_lr(name: str, X_tr, X_te, y, lr_kwargs: dict,
          inject_te: dict | None = None,
          train_df: pd.DataFrame | None = None,
          test_df: pd.DataFrame | None = None) -> dict:
    """5-fold CV LR with optional fold-safe TE injection.

    inject_te: dict like {'col_name': smoothing}. If set, TE is computed on
    tr-rows of the fold and prepended/appended as additional columns.

    Returns dict with oof, test, auc, time, n_features.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    test_pred = np.zeros(X_te.shape[0], dtype=np.float64)

    is_sparse = sparse.issparse(X_tr)
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
        X_tr_f = X_tr[tr_idx]
        X_va_f = X_tr[va_idx]
        X_te_f = X_te
        y_tr = y[tr_idx]

        if inject_te:
            prior = float(y_tr.mean())
            tr_extra, va_extra, te_extra = [], [], []
            for col, smooth in inject_te.items():
                t_enc, v_enc = fit_te_inside_fold(
                    y_tr, train_df[col].iloc[tr_idx], train_df[col].iloc[va_idx],
                    prior=prior, smoothing=smooth
                )
                # For test, fit on full train_df.iloc[tr_idx] of this fold (use tr_y)
                te_enc_te = test_df[col].map(
                    pd.DataFrame({"c": train_df[col].iloc[tr_idx].values, "y": y_tr})
                    .groupby("c")["y"].agg(["mean", "count"])
                    .pipe(lambda g: (g["count"] * g["mean"] + smooth * prior) / (g["count"] + smooth))
                ).fillna(prior).values.astype(np.float32)
                tr_extra.append(t_enc)
                va_extra.append(v_enc)
                te_extra.append(te_enc_te)
            tr_extra = np.column_stack(tr_extra)
            va_extra = np.column_stack(va_extra)
            te_extra = np.column_stack(te_extra)
            # Standardize the TE columns (leak-free since fit per fold above)
            mu = tr_extra.mean(axis=0)
            sd = tr_extra.std(axis=0) + 1e-9
            tr_extra = (tr_extra - mu) / sd
            va_extra = (va_extra - mu) / sd
            te_extra = (te_extra - mu) / sd
            if is_sparse:
                from scipy.sparse import hstack as sp_hstack, csr_matrix
                X_tr_f = sp_hstack([X_tr_f, csr_matrix(tr_extra)], format="csr")
                X_va_f = sp_hstack([X_va_f, csr_matrix(va_extra)], format="csr")
                X_te_f = sp_hstack([X_te_f, csr_matrix(te_extra)], format="csr")
            else:
                X_tr_f = np.hstack([X_tr_f, tr_extra])
                X_va_f = np.hstack([X_va_f, va_extra])
                X_te_f = np.hstack([X_te_f, te_extra])

        lr = LogisticRegression(**lr_kwargs)
        lr.fit(X_tr_f, y_tr)
        oof[va_idx] = lr.predict_proba(X_va_f)[:, 1]
        test_pred += lr.predict_proba(X_te_f)[:, 1] / N_FOLDS

    elapsed = time.time() - t0
    auc = float(roc_auc_score(y, oof))
    n_feat = int(X_tr.shape[1] + (len(inject_te) if inject_te else 0))
    return dict(name=name, auc=auc, time=elapsed, n_features=n_feat,
                oof=oof, test=test_pred)


# -----------------------------------------------------------------------------
# Variant builders
# -----------------------------------------------------------------------------

def build_variant(variant: str, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray):
    """Return (X_tr, X_te, lr_kwargs, inject_te) for the named variant."""
    num_tr_raw = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te_raw = test[NUM_COLS].fillna(0).values.astype(np.float32)

    LR_DEF = dict(C=1.0, max_iter=2000, solver="lbfgs")
    LR_LIB = dict(C=1.0, max_iter=2000, solver="liblinear")
    # Wide sparse: saga is multi-threaded (n_jobs=-1) — much faster on >500 sparse cols
    LR_SAGA = dict(C=1.0, max_iter=200, solver="saga", n_jobs=-1, tol=1e-3)
    LR_SAGA_L1 = dict(C=1.0, max_iter=200, solver="saga", penalty="l1", n_jobs=-1, tol=1e-3)
    LR_BAL = dict(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")

    # ---- Tier A: vanilla
    if variant == "lr_raw_std":
        Xtr, Xte = standardize_dense(num_tr_raw, num_te_raw)
        return Xtr, Xte, LR_DEF, None
    if variant == "lr_raw_std_balanced":
        Xtr, Xte = standardize_dense(num_tr_raw, num_te_raw)
        return Xtr, Xte, LR_BAL, None

    # ---- Tier B: + cat encodings
    if variant == "lr_raw_freq":
        Xtr, Xte = standardize_dense(num_tr_raw, num_te_raw)
        f_cols_tr, f_cols_te = [], []
        for c in CAT_COLS:
            ftr, fte = freq_encode(train[c], test[c])
            f_cols_tr.append(np.log1p(ftr).astype(np.float32))
            f_cols_te.append(np.log1p(fte).astype(np.float32))
        Xtr = np.hstack([Xtr, np.column_stack(f_cols_tr)])
        Xte = np.hstack([Xte, np.column_stack(f_cols_te)])
        return Xtr, Xte, LR_DEF, None
    if variant == "lr_raw_te":
        Xtr, Xte = standardize_dense(num_tr_raw, num_te_raw)
        return Xtr, Xte, LR_DEF, {"Driver": 20.0, "Race": 20.0, "Compound": 20.0}
    if variant == "lr_raw_ohe":
        Xn_tr, Xn_te = standardize_dense(num_tr_raw, num_te_raw)
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([sparse.csr_matrix(Xn_tr), Oc_tr], format="csr")
        Xte = sparse.hstack([sparse.csr_matrix(Xn_te), Oc_te], format="csr")
        return Xtr, Xte, LR_SAGA, None

    # ---- Tier C: polynomial
    if variant == "lr_poly2_std":
        pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        Xn_tr = pf.fit_transform(num_tr_raw)
        Xn_te = pf.transform(num_te_raw)
        Xtr, Xte = standardize_dense(Xn_tr, Xn_te)
        return Xtr, Xte, LR_DEF, None
    if variant == "lr_poly2_ohe":
        pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        Xn_tr = pf.fit_transform(num_tr_raw)
        Xn_te = pf.transform(num_te_raw)
        Xn_tr, Xn_te = standardize_dense(Xn_tr, Xn_te)
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([sparse.csr_matrix(Xn_tr), Oc_tr], format="csr")
        Xte = sparse.hstack([sparse.csr_matrix(Xn_te), Oc_te], format="csr")
        return Xtr, Xte, LR_SAGA, None
    if variant == "lr_poly3_std":
        pf = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
        Xn_tr = pf.fit_transform(num_tr_raw)
        Xn_te = pf.transform(num_te_raw)
        Xtr, Xte = standardize_dense(Xn_tr, Xn_te)
        return Xtr, Xte, LR_DEF, None

    # ---- Tier D: discretization
    if variant == "lr_kbins5_ohe":
        Xb_tr, Xb_te = kbins_combined(num_tr_raw, num_te_raw, n_bins=5, strategy="quantile")
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
        Xte = sparse.hstack([Xb_te, Oc_te], format="csr")
        return Xtr, Xte, LR_SAGA, None
    if variant == "lr_kbins20_ohe":
        Xb_tr, Xb_te = kbins_combined(num_tr_raw, num_te_raw, n_bins=20, strategy="quantile")
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
        Xte = sparse.hstack([Xb_te, Oc_te], format="csr")
        return Xtr, Xte, LR_SAGA, None
    if variant == "lr_kbins50_uniform":
        Xb_tr, Xb_te = kbins_combined(num_tr_raw, num_te_raw, n_bins=50, strategy="uniform")
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
        Xte = sparse.hstack([Xb_te, Oc_te], format="csr")
        return Xtr, Xte, LR_SAGA, None
    if variant == "lr_kbins_yekenot":
        # yekenot's recipe: KBins(200) on RaceProgress + KBins(7) on LapTime
        idx_rp = NUM_COLS.index("RaceProgress")
        idx_lt = NUM_COLS.index("LapTime (s)")
        rp_tr = num_tr_raw[:, idx_rp:idx_rp + 1]
        rp_te = num_te_raw[:, idx_rp:idx_rp + 1]
        lt_tr = num_tr_raw[:, idx_lt:idx_lt + 1]
        lt_te = num_te_raw[:, idx_lt:idx_lt + 1]
        Brp_tr, Brp_te = kbins_combined(rp_tr, rp_te, n_bins=200, strategy="quantile")
        Blt_tr, Blt_te = kbins_combined(lt_tr, lt_te, n_bins=7, strategy="quantile")
        # Plus standardised raw numerics + cat OHE
        Xn_tr, Xn_te = standardize_dense(num_tr_raw, num_te_raw)
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([sparse.csr_matrix(Xn_tr), Brp_tr, Blt_tr, Oc_tr], format="csr")
        Xte = sparse.hstack([sparse.csr_matrix(Xn_te), Brp_te, Blt_te, Oc_te], format="csr")
        return Xtr, Xte, LR_SAGA, None

    # ---- Tier E: splines
    if variant == "lr_splines_5":
        st = SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False)
        Xs_tr = st.fit_transform(num_tr_raw)
        Xs_te = st.transform(num_te_raw)
        Xs_tr, Xs_te = standardize_dense(Xs_tr, Xs_te)
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([sparse.csr_matrix(Xs_tr), Oc_tr], format="csr")
        Xte = sparse.hstack([sparse.csr_matrix(Xs_te), Oc_te], format="csr")
        return Xtr, Xte, LR_SAGA, None
    if variant == "lr_splines_10":
        st = SplineTransformer(n_knots=10, degree=3, knots="quantile", include_bias=False)
        Xs_tr = st.fit_transform(num_tr_raw)
        Xs_te = st.transform(num_te_raw)
        Xs_tr, Xs_te = standardize_dense(Xs_tr, Xs_te)
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([sparse.csr_matrix(Xs_tr), Oc_tr], format="csr")
        Xte = sparse.hstack([sparse.csr_matrix(Xs_te), Oc_te], format="csr")
        return Xtr, Xte, LR_SAGA, None

    # ---- Tier F: hash trick (high-card combos)
    if variant == "lr_hash_2way_2k":
        combos = [("Driver", "Race"), ("Driver", "Compound"), ("Race", "Compound")]
        Hh_tr, Hh_te = hash_features_vec(train, test, combos, n_features=2048)
        Xn_tr, Xn_te = standardize_dense(num_tr_raw, num_te_raw)
        Xtr = sparse.hstack([sparse.csr_matrix(Xn_tr), Hh_tr], format="csr")
        Xte = sparse.hstack([sparse.csr_matrix(Xn_te), Hh_te], format="csr")
        return Xtr, Xte, LR_SAGA, None
    if variant == "lr_hash_3way_8k":
        combos = [("Driver", "Race", "Compound"), ("Driver", "Race", "Year"),
                  ("Driver", "Compound", "Year"), ("Race", "Compound", "Year")]
        # Year column is in train as int; cast to str through hash function
        train2 = train.copy()
        test2 = test.copy()
        train2["Year"] = train2["Year"].astype(str)
        test2["Year"] = test2["Year"].astype(str)
        Hh_tr, Hh_te = hash_features_vec(train2, test2, combos, n_features=8192)
        Xn_tr, Xn_te = standardize_dense(num_tr_raw, num_te_raw)
        Xtr = sparse.hstack([sparse.csr_matrix(Xn_tr), Hh_tr], format="csr")
        Xte = sparse.hstack([sparse.csr_matrix(Xn_te), Hh_te], format="csr")
        return Xtr, Xte, LR_SAGA, None

    # ---- Tier G: penalty / C
    if variant == "lr_l1_lasso_kbins20":
        Xb_tr, Xb_te = kbins_combined(num_tr_raw, num_te_raw, n_bins=20, strategy="quantile")
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
        Xte = sparse.hstack([Xb_te, Oc_te], format="csr")
        return Xtr, Xte, dict(C=1.0, max_iter=200, solver="saga", penalty="l1",
                              n_jobs=-1, tol=1e-3), None
    if variant == "lr_C_low_kbins20":
        Xb_tr, Xb_te = kbins_combined(num_tr_raw, num_te_raw, n_bins=20, strategy="quantile")
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
        Xte = sparse.hstack([Xb_te, Oc_te], format="csr")
        return Xtr, Xte, dict(C=0.001, max_iter=2000, solver="liblinear"), None
    if variant == "lr_C_high_kbins20":
        Xb_tr, Xb_te = kbins_combined(num_tr_raw, num_te_raw, n_bins=20, strategy="quantile")
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
        Xte = sparse.hstack([Xb_te, Oc_te], format="csr")
        return Xtr, Xte, dict(C=100.0, max_iter=2000, solver="liblinear"), None
    if variant == "lr_balanced_kbins20":
        Xb_tr, Xb_te = kbins_combined(num_tr_raw, num_te_raw, n_bins=20, strategy="quantile")
        Oc_tr, Oc_te = ohe_cats(train, test, CAT_COLS)
        Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
        Xte = sparse.hstack([Xb_te, Oc_te], format="csr")
        return Xtr, Xte, dict(C=1.0, max_iter=2000, solver="liblinear",
                              class_weight="balanced"), None

    # ---- Tier H: per-segment specialists (handled separately by main)
    # ---- Tier I: LR on derived features (handled separately by main)
    # ---- Tier J: GPU PyTorch demo (handled separately by main)

    raise ValueError(f"unknown variant: {variant}")


# -----------------------------------------------------------------------------
# Per-segment specialist: fits separate LR per segment, concatenates predictions
# -----------------------------------------------------------------------------

def run_per_segment(name: str, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
                    seg_col: str) -> dict:
    """Per-segment LR using KBins+OHE features within each segment."""
    seg_tr = train[seg_col].values
    seg_te = test[seg_col].values
    n_tr = len(train)
    n_te = len(test)

    # Use KBins(20)+catOHE features (the strongest LR base typically)
    num_tr_raw = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te_raw = test[NUM_COLS].fillna(0).values.astype(np.float32)
    Xb_tr, Xb_te = kbins_combined(num_tr_raw, num_te_raw, n_bins=20, strategy="quantile")
    cat_other = [c for c in CAT_COLS if c != seg_col]
    Oc_tr, Oc_te = ohe_cats(train, test, cat_other)
    Xtr = sparse.hstack([Xb_tr, Oc_tr], format="csr")
    Xte = sparse.hstack([Xb_te, Oc_te], format="csr")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_pred = np.zeros(n_te, dtype=np.float64)
    test_segs_count = np.zeros(n_te, dtype=np.float64)
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_tr), y)):
        for s in np.unique(seg_tr):
            tr_mask = (seg_tr[tr_idx] == s)
            va_mask = (seg_tr[va_idx] == s)
            te_mask = (seg_te == s)
            if tr_mask.sum() < 100 or len(np.unique(y[tr_idx][tr_mask])) < 2:
                # Fall back to overall prior
                prior = float(y[tr_idx].mean())
                oof[va_idx[va_mask]] = prior
                test_pred[te_mask] += prior / N_FOLDS
                continue
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
            lr.fit(Xtr[tr_idx][tr_mask], y[tr_idx][tr_mask])
            if va_mask.sum() > 0:
                oof[va_idx[va_mask]] = lr.predict_proba(Xtr[va_idx][va_mask])[:, 1]
            if te_mask.sum() > 0:
                test_pred[te_mask] += lr.predict_proba(Xte[te_mask])[:, 1] / N_FOLDS
    elapsed = time.time() - t0
    auc = float(roc_auc_score(y, oof))
    return dict(name=name, auc=auc, time=elapsed, n_features=int(Xtr.shape[1]),
                oof=oof, test=test_pred)


# -----------------------------------------------------------------------------
# LR on derived features: use existing LGBM/CB/etc. OOFs as input
# -----------------------------------------------------------------------------

def run_lr_on_derived(name: str, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
                     base_oof_files: list[str]) -> dict:
    """LR on a small set of derived features (e.g. existing OOFs from other models).
    Fold-safe: re-fits LR per fold but uses already-CV-fitted OOFs as inputs.
    """
    cols_tr = []
    cols_te = []
    for fn in base_oof_files:
        oof_arr = np.load(ART / f"oof_{fn}_strat.npy")
        test_arr = np.load(ART / f"test_{fn}_strat.npy")
        cols_tr.append(oof_arr[:, 1] if oof_arr.ndim == 2 else oof_arr.ravel())
        cols_te.append(test_arr[:, 1] if test_arr.ndim == 2 else test_arr.ravel())
    Pt = np.column_stack(cols_tr)
    Ps = np.column_stack(cols_te)
    # logit transform for LR-input
    Pt_c = np.clip(Pt, 1e-9, 1 - 1e-9)
    Ps_c = np.clip(Ps, 1e-9, 1 - 1e-9)
    Lt = np.log(Pt_c / (1 - Pt_c))
    Ls = np.log(Ps_c / (1 - Ps_c))
    # Add raw numerics (standardised)
    num_tr_raw = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te_raw = test[NUM_COLS].fillna(0).values.astype(np.float32)
    Xn_tr, Xn_te = standardize_dense(num_tr_raw, num_te_raw)
    Xtr = np.hstack([Lt, Xn_tr])
    Xte = np.hstack([Ls, Xn_te])
    return cv_lr(name, Xtr, Xte, y, dict(C=1.0, max_iter=2000, solver="lbfgs"),
                 inject_te=None)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

ALL_VARIANTS = [
    # Tier A
    "lr_raw_std", "lr_raw_std_balanced",
    # Tier B
    "lr_raw_freq", "lr_raw_te", "lr_raw_ohe",
    # Tier C
    "lr_poly2_std", "lr_poly2_ohe", "lr_poly3_std",
    # Tier D
    "lr_kbins5_ohe", "lr_kbins20_ohe", "lr_kbins50_uniform", "lr_kbins_yekenot",
    # Tier E
    "lr_splines_5", "lr_splines_10",
    # Tier F
    "lr_hash_2way_2k", "lr_hash_3way_8k",
    # Tier G
    "lr_l1_lasso_kbins20", "lr_C_low_kbins20", "lr_C_high_kbins20", "lr_balanced_kbins20",
]

PER_SEGMENT_VARIANTS = [
    ("lr_perseg_compound", "Compound"),
    ("lr_perseg_year", "Year"),
]

DERIVED_VARIANTS = [
    # LR on a small set of strongest existing OOFs (LR-of-models)
    ("lr_on_top_models",
     ["d17_h1d_yekenot_full", "p1_single_cb_v4_gpu", "d16_orig_continuous_only",
      "cb_year-cat", "e3_hgbc"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=None,
                    help="comma-separated variant names; else --tiers or --all")
    ap.add_argument("--tiers", default=None,
                    help="comma-separated tier letters A..J")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--per-segment", action="store_true",
                    help="also run per-segment variants")
    ap.add_argument("--derived", action="store_true",
                    help="also run LR-on-derived-feats variants")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip variants whose OOF artifact already exists")
    args = ap.parse_args()

    train, test, y = load_data()
    print(f"data: train {train.shape}, test {test.shape}, prior {y.mean():.4f}")

    if args.names:
        variants = [v.strip() for v in args.names.split(",")]
    elif args.tiers:
        # tier mapping based on positions in ALL_VARIANTS list
        tier_map = {"A": (0, 2), "B": (2, 5), "C": (5, 8), "D": (8, 12),
                    "E": (12, 14), "F": (14, 16), "G": (16, 20)}
        variants = []
        for letter in args.tiers.split(","):
            lo, hi = tier_map[letter.strip().upper()]
            variants.extend(ALL_VARIANTS[lo:hi])
    elif args.all:
        variants = list(ALL_VARIANTS)
    else:
        variants = list(ALL_VARIANTS)

    summary = []
    for v in variants:
        if args.skip_existing and (ART / f"oof_{v}_strat.npy").exists():
            print(f"  skipping {v} (exists)")
            continue
        print(f"\n[{v}]")
        try:
            X_tr, X_te, lr_kw, te_inj = build_variant(v, train, test, y)
            res = cv_lr(v, X_tr, X_te, y, lr_kw,
                        inject_te=te_inj, train_df=train, test_df=test)
            print(f"  AUC {res['auc']:.5f}  ({res['time']:.1f}s, {res['n_features']} feats)")
            oof2 = np.column_stack([1 - res["oof"], res["oof"]])
            test2 = np.column_stack([1 - res["test"], res["test"]])
            np.save(ART / f"oof_{v}_strat.npy", oof2)
            np.save(ART / f"test_{v}_strat.npy", test2)
            summary.append({k: res[k] for k in ("name", "auc", "time", "n_features")})
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            summary.append({"name": v, "error": str(e)})

    if args.per_segment:
        for v, seg in PER_SEGMENT_VARIANTS:
            if args.skip_existing and (ART / f"oof_{v}_strat.npy").exists():
                print(f"  skipping {v} (exists)")
                continue
            print(f"\n[{v}]  segment={seg}")
            try:
                res = run_per_segment(v, train, test, y, seg_col=seg)
                print(f"  AUC {res['auc']:.5f}  ({res['time']:.1f}s)")
                oof2 = np.column_stack([1 - res["oof"], res["oof"]])
                test2 = np.column_stack([1 - res["test"], res["test"]])
                np.save(ART / f"oof_{v}_strat.npy", oof2)
                np.save(ART / f"test_{v}_strat.npy", test2)
                summary.append({k: res[k] for k in ("name", "auc", "time", "n_features")})
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                summary.append({"name": v, "error": str(e)})

    if args.derived:
        for v, files in DERIVED_VARIANTS:
            if args.skip_existing and (ART / f"oof_{v}_strat.npy").exists():
                print(f"  skipping {v} (exists)")
                continue
            print(f"\n[{v}]  derived from {files}")
            try:
                res = run_lr_on_derived(v, train, test, y, files)
                print(f"  AUC {res['auc']:.5f}  ({res['time']:.1f}s)")
                oof2 = np.column_stack([1 - res["oof"], res["oof"]])
                test2 = np.column_stack([1 - res["test"], res["test"]])
                np.save(ART / f"oof_{v}_strat.npy", oof2)
                np.save(ART / f"test_{v}_strat.npy", test2)
                summary.append({k: res[k] for k in ("name", "auc", "time", "n_features")})
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                summary.append({"name": v, "error": str(e)})

    out_json = ART / "lr_bank_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n→ summary saved: {out_json}")


if __name__ == "__main__":
    main()
