"""d18b — Chain-decomposition variants 2 & 3 (Rule-21 family falsification).

Variant grid (passes Rule 21 ≥3-variant requirement):

  v1 (d18):   Gaussian-σ regressor for continuous, multiclass for cat.
              Causal ordering. (Already done.)
  v2 (d18b):  q10-multiclass for continuous, multiclass for cat.
              Same causal ordering. Isolates model-choice axis.
  v3 (d18c):  q10-multiclass for continuous, multiclass for cat.
              REVERSE ordering (LapTime first, decompose backward).
              Isolates ordering axis.

For continuous step k: bin orig values into q10 quantiles → fit multiclass
LGBM → log P(actual bin | upstream) is the chain-LL feature. Drops the
Gaussian-σ approximation (which is suspect when residuals are not normal).

Usage:
  python scripts/d18b_chain_variants.py --variant v2
  python scripts/d18b_chain_variants.py --variant v3

Outputs:
  scripts/artifacts/oof_d18{b,c}_chain_decomp_strat.npy + test
  scripts/artifacts/d18{b,c}_chain_decomp_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
ART = Path("scripts/artifacts")
DATA_OUT = Path("data")
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"
LAPTIME = "LapTime (s)"
EPS = 1e-9
N_BINS = 10  # q10 binning for continuous targets

CAT_OK = ["Compound", "Race"]
NUM_FEATS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             LAPTIME, "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change"]


# Causal ordering (same as d18); reverse is reversed
CAUSAL_CHAIN = [
    ("Year", "marginal_int"),
    ("Race", "multiclass"),
    ("Compound", "multiclass"),
    ("Stint", "multiclass"),
    ("LapNumber", "qbins"),
    ("TyreLife", "qbins"),
    ("RaceProgress", "qbins"),
    ("Position", "qbins"),
    (LAPTIME, "qbins"),
    ("LapTime_Delta", "qbins"),
    ("Cumulative_Degradation", "qbins"),
    ("Position_Change", "qbins"),
    ("PitStop", "binary"),
]


def _encode_cat(df, mappings=None):
    out = df.copy()
    if mappings is None:
        mappings = {}
        for c in CAT_OK:
            levels = sorted(df[c].dropna().astype(str).unique().tolist())
            mappings[c] = {v: i for i, v in enumerate(levels)}
    for c in CAT_OK:
        out[c] = out[c].astype(str).map(mappings[c]).astype("Int64")
    return out, mappings


def _fit_cls(X, y, cat_cols, num_class, n_round=300):
    if num_class == 2:
        p = dict(objective="binary", metric="binary_logloss",
                 learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                 verbosity=-1, seed=SEED)
    else:
        p = dict(objective="multiclass", num_class=int(num_class),
                 metric="multi_logloss", learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=50, verbosity=-1, seed=SEED)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    ds = lgb.Dataset(X, label=y, categorical_feature=cat_idx,
                     free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=n_round)


def build_chain(ordering, all_feats):
    """For each step, conditioning is the union of all features that come
    earlier in 'ordering'. Restrict to features that exist in `all_feats`.
    Skip self in conditioning."""
    seen = []
    chain = []
    for col, kind in ordering:
        feats = [c for c in seen if c in all_feats and c != col]
        chain.append(dict(col=col, kind=kind, feats=feats))
        seen.append(col)
    return chain


def fit_chain(orig: pd.DataFrame, ordering, log=print):
    chain = build_chain(ordering, list(orig.columns))
    fitted = []
    # Year marginal
    y_step = chain[0]
    yvals, ycount = np.unique(orig[y_step["col"]].astype(int).values,
                              return_counts=True)
    ycount = ycount.astype(np.float64) / ycount.sum()
    fitted.append(dict(col=y_step["col"], kind="marginal_int",
                       year_logp={int(v): float(np.log(p))
                                  for v, p in zip(yvals, ycount)}))
    log(f"  step  1/{len(chain)} {y_step['col']:24s} marginal {len(yvals)} levels")

    for i, step in enumerate(chain[1:], start=2):
        col, kind, feats = step["col"], step["kind"], step["feats"]
        if not feats:
            # No conditioning features yet — fall back to marginal log-prob.
            if kind == "qbins":
                vals = pd.to_numeric(orig[col], errors="coerce").dropna()
                edges = np.unique(vals.quantile(np.linspace(0, 1, N_BINS + 1)).values)
                bins = pd.cut(vals, edges, include_lowest=True, duplicates="drop")
                p = bins.value_counts(normalize=True).sort_index()
                logp_marg = np.log(np.clip(p.values, EPS, 1.0))
                fitted.append(dict(col=col, kind="qbins_marg",
                                   edges=edges.tolist(), logp=logp_marg.tolist()))
                log(f"  step {i:>2}/{len(chain)} {col:24s} qbins-marg "
                    f"{len(edges)-1} bins")
                continue
            # else: should not happen for first non-marginal step we use
        t0 = time.time()
        X = orig[feats].copy()
        cats_in = [c for c in feats if c in CAT_OK]

        if kind == "qbins":
            v = pd.to_numeric(orig[col], errors="coerce").astype(float).values
            mask = ~np.isnan(v)
            v_fit = v[mask]
            edges = np.unique(np.quantile(v_fit, np.linspace(0, 1, N_BINS + 1)))
            bins = pd.cut(v_fit, edges, include_lowest=True, duplicates="drop")
            level_ix = np.asarray(bins.codes, dtype=np.int32)
            ok = level_ix >= 0
            X_fit = X.loc[mask].iloc[ok].reset_index(drop=True)
            y_fit = level_ix[ok]
            n_cls = int(level_ix.max()) + 1 if ok.any() else 1
            model = _fit_cls(X_fit, y_fit, cats_in, num_class=n_cls)
            fitted.append(dict(col=col, kind="qbins", feats=feats,
                               cats=cats_in, model=model,
                               edges=edges.tolist(), n_classes=n_cls))
        elif kind == "multiclass":
            y_int = pd.to_numeric(orig[col], errors="coerce").fillna(0).astype(int).values
            uniq = np.sort(np.unique(y_int))
            level_map = {int(v): i for i, v in enumerate(uniq)}
            y_fit = np.array([level_map[int(v)] for v in y_int], dtype=np.int32)
            model = _fit_cls(X, y_fit, cats_in, num_class=len(uniq))
            fitted.append(dict(col=col, kind=kind, feats=feats, cats=cats_in,
                               model=model, levels=uniq.tolist(),
                               level_map=level_map))
        elif kind == "binary":
            y_fit = pd.to_numeric(orig[col], errors="coerce").fillna(0).astype(int).values
            y_fit = np.clip(y_fit, 0, 1)
            model = _fit_cls(X, y_fit, cats_in, num_class=2)
            fitted.append(dict(col=col, kind=kind, feats=feats, cats=cats_in,
                               model=model))
        else:
            raise ValueError(kind)
        log(f"  step {i:>2}/{len(chain)} {col:24s} {kind:10s} "
            f"| {len(feats)} feats | {time.time()-t0:.1f}s")
    return fitted


def apply_chain(steps, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    n = len(df)
    for s in steps:
        col, kind = s["col"], s["kind"]
        safe = col.replace(" ", "_").replace("(", "").replace(")", "")
        if kind == "marginal_int":
            yv = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int).values
            ll = np.array([s["year_logp"].get(int(v), np.log(EPS))
                           for v in yv], dtype=np.float32)
            out[f"chain_ll_{safe}"] = ll
        elif kind == "qbins_marg":
            v = pd.to_numeric(df[col], errors="coerce").astype(float).values
            edges = np.array(s["edges"]); logp = np.array(s["logp"])
            ix = np.clip(np.searchsorted(edges, v, side="right") - 1,
                         0, len(logp) - 1)
            out[f"chain_ll_{safe}"] = logp[ix].astype(np.float32)
        elif kind == "qbins":
            X = df[s["feats"]].copy()
            proba = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            v = pd.to_numeric(df[col], errors="coerce").astype(float).values
            edges = np.array(s["edges"])
            n_cls = s["n_classes"]
            ix = np.clip(np.searchsorted(edges, v, side="right") - 1,
                         0, n_cls - 1)
            ok = ~np.isnan(v)
            ll = np.full(n, np.log(EPS), dtype=np.float32)
            if ok.any():
                row_ix = np.where(ok)[0]
                p_act = np.clip(proba[row_ix, ix[ok]], EPS, 1.0)
                ll[row_ix] = np.log(p_act)
            out[f"chain_ll_{safe}"] = ll
        elif kind == "multiclass":
            X = df[s["feats"]].copy()
            proba = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            actual_raw = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int).values
            actual_idx = np.array([s["level_map"].get(int(v), -1) for v in actual_raw])
            ll = np.full(n, np.log(EPS), dtype=np.float32)
            ok = actual_idx >= 0
            if ok.any():
                p_act = np.clip(proba[ok][np.arange(int(ok.sum())),
                                          actual_idx[ok]], EPS, 1.0)
                ll[ok] = np.log(p_act)
            out[f"chain_ll_{safe}"] = ll
        elif kind == "binary":
            X = df[s["feats"]].copy()
            p1 = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            actual = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).values
            p_act = np.where(actual == 1, np.clip(p1, EPS, 1 - EPS),
                             np.clip(1 - p1, EPS, 1 - EPS))
            out[f"chain_ll_{safe}"] = np.log(p_act).astype(np.float32)
    out["chain_total_ll"] = out.values.sum(axis=1)
    return out


def downstream_lgbm(tr_X, tr_y, te_X, cat_cols):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr_y), dtype=np.float64)
    test_avg = np.zeros(len(te_X), dtype=np.float64)
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=127, min_data_in_leaf=80, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbosity=-1, seed=SEED)
    for fi, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(tr_y)), tr_y), 1):
        cat_idx = [tr_X.columns.get_loc(c) for c in cat_cols if c in tr_X.columns]
        ds_tr = lgb.Dataset(tr_X.iloc[tr_i], label=tr_y[tr_i],
                            categorical_feature=cat_idx, free_raw_data=False)
        ds_va = lgb.Dataset(tr_X.iloc[va_i], label=tr_y[va_i],
                            categorical_feature=cat_idx, reference=ds_tr,
                            free_raw_data=False)
        m = lgb.train(p, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_i] = m.predict(tr_X.iloc[va_i], num_iteration=m.best_iteration)
        test_avg += m.predict(te_X, num_iteration=m.best_iteration) / N_FOLDS
        print(f"    fold {fi}: AUC={roc_auc_score(tr_y[va_i], oof[va_i]):.5f}  "
              f"best_iter={m.best_iteration}")
    print(f"    OOF AUC = {roc_auc_score(tr_y, oof):.5f}")
    return oof, test_avg


def _save_oof_test(name, oof_pos, test_pos):
    oof = np.column_stack([1.0 - oof_pos, oof_pos]).astype(np.float64)
    test = np.column_stack([1.0 - test_pos, test_pos]).astype(np.float64)
    np.save(ART / f"oof_{name}_strat.npy", oof)
    np.save(ART / f"test_{name}_strat.npy", test)
    print(f"  saved oof_{name}_strat.npy / test_{name}_strat.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v2", "v3"], required=True)
    args = ap.parse_args()
    t0 = time.time()
    if args.variant == "v2":
        ordering = CAUSAL_CHAIN
        outname = "d18b_chain_decomp"
        label = "v2 causal+q10"
    else:
        ordering = list(reversed(CAUSAL_CHAIN))
        # Ensure first step is treated as a marginal (can't have empty
        # conditioning otherwise). For PitStop-first (binary) we'll fall
        # back to marginal log-prob via a tiny logit on the empirical rate.
        # For simplicity, force first step to a marginal qbins / cat-marg:
        # just relabel its kind.
        first_col, first_kind = ordering[0]
        # Convert binary first to a marginal (Bernoulli) by switching to
        # 'multiclass' at 2 classes AND adding a synthetic feature.
        # Easiest: keep as binary but in apply_chain it falls to model-fit
        # of upstream feats; since upstream is empty, we handle in build_chain
        # by treating qbins/binary/marginal_int with empty feats as marginal.
        if first_kind == "binary":
            # use marginal_int: 2-level
            ordering = [("PitStop", "marginal_int")] + ordering[1:]
        outname = "d18c_chain_decomp"
        label = "v3 reverse+q10"

    print(f"[{label}]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()

    union = pd.concat([orig[CAT_OK], tr[CAT_OK], te[CAT_OK]], ignore_index=True)
    _, mappings = _encode_cat(union)
    orig_e, _ = _encode_cat(orig, mappings)
    tr_e, _ = _encode_cat(tr, mappings)
    te_e, _ = _encode_cat(te, mappings)

    # For reverse ordering, relabel first step kinds where conditioning would
    # be empty. The build_chain path falls through with no model; we
    # convert to a univariate marginal model via qbins_marg / freq table.
    ordering_for_fit = []
    for i, (col, kind) in enumerate(ordering):
        if i == 0:
            if kind in ("qbins", "binary", "multiclass"):
                # Use marginal_int for binary; qbins_marg for continuous;
                # multiclass-marg via freq for multiclass.
                if kind == "binary":
                    ordering_for_fit.append((col, "marginal_int"))
                elif kind == "qbins":
                    ordering_for_fit.append((col, "qbins"))  # build_chain path with empty feats
                else:
                    ordering_for_fit.append((col, "multiclass"))
            else:
                ordering_for_fit.append((col, kind))
        else:
            ordering_for_fit.append((col, kind))

    print(f"[fit chain v={args.variant}, ordering len={len(ordering_for_fit)}]")
    steps = fit_chain(orig_e, ordering_for_fit)
    print(f"\n[apply → train {len(tr_e)}]")
    tr_chain = apply_chain(steps, tr_e)
    print(f"[apply → test  {len(te_e)}]")
    te_chain = apply_chain(steps, te_e)
    print(f"  features: {len(tr_chain.columns)}")

    # Diagnostic KS y=0 vs y=1
    y = tr_e[TARGET].astype(int).values
    pos = y == 1; neg = y == 0
    ks = []
    for col in tr_chain.columns:
        v = tr_chain[col].values
        stat, p = ks_2samp(v[pos], v[neg])
        ks.append(dict(feature=col, ks=float(stat), p=float(p)))
    ks.sort(key=lambda r: -r["ks"])
    print(f"  top 5 KS y=0 vs y=1: "
          f"{[(r['feature'], round(r['ks'], 3)) for r in ks[:5]]}")

    # Downstream LGBM (raw + chain)
    raw_cols = CAT_OK + NUM_FEATS
    tr_X = pd.concat([tr_e[raw_cols].reset_index(drop=True),
                      tr_chain.reset_index(drop=True)], axis=1)
    te_X = pd.concat([te_e[raw_cols].reset_index(drop=True),
                      te_chain.reset_index(drop=True)], axis=1)
    print(f"\n[downstream LGBM raw + chain ({args.variant})]")
    oof, test = downstream_lgbm(tr_X, y, te_X, CAT_OK)
    _save_oof_test(outname, oof, test)

    summary = dict(variant=args.variant, label=label,
                   n_chain_features=int(len(tr_chain.columns)),
                   oof_auc=float(roc_auc_score(y, oof)),
                   class_conditional_ks=ks)
    (ART / f"{outname}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done {label}]  wall {time.time()-t0:.0f}s  OOF {summary['oof_auc']:.5f}")
    print(f"  → next: python scripts/probe_min_meta.py --candidates {outname}")


if __name__ == "__main__":
    main()
