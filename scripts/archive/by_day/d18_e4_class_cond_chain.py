"""d18 E4 — Class-conditional chain decomposition.

For each class y∈{0,1}, fit a SEPARATE chain on orig[PitNextLap=y] subset
using the same CHAIN_STEPS as d18 v1 (Gaussian for continuous).
For each synth row compute:

  log_ratio_<col> = chain_ll_y1_<col> - chain_ll_y0_<col>   per step
  total_log_ratio = sum log_ratio_<col>

This is the per-step orig-DGP class-conditional log-Bayes-factor.
A row whose features look much more likely under the y=1 sub-population
under the orig DGP gets large positive log_ratio; vice versa.

The key axis: synth rows whose class-conditional likelihood differs
strongly between y=1 and y=0 carry direct class-discriminative
information from the orig DGP. This is the per-step factorisation
of d15_orig_transfer / d16_orig_continuous_only (which use joint
P(y|x) directly).

Cost: ~25 min CPU. 2 chains × 12 LGBM steps × small orig subsets.

Outputs:
  scripts/artifacts/oof_d18_e4_class_cond_strat.npy
  scripts/artifacts/test_d18_e4_class_cond_strat.npy
  data/class_cond_features_{train,test}.parquet (diagnostic)
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

# Reuse machinery from d18_chain_decomp.
sys.path.insert(0, str(Path(__file__).parent))
from d18_chain_decomp import (
    CAT_OK, NUM_FEATS, CHAIN_STEPS, _encode_cat, _gauss_ll, _cls_ll,
    _fit_reg, _fit_cls, _safe_clip_int, downstream_lgbm,
)

warnings.filterwarnings("ignore", category=UserWarning)
ART = Path("scripts/artifacts")
DATA_OUT = Path("data")
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"
EPS = 1e-9


def fit_chain_on_subset(orig_subset: pd.DataFrame, log=print, label=""):
    """Fit chain on a subset of orig (e.g. orig[y=0])."""
    steps = []
    yvals, ycount = np.unique(orig_subset["Year"].astype(int).values,
                              return_counts=True)
    ycount = ycount.astype(np.float64) / ycount.sum()
    yp = {int(y): float(np.log(p)) for y, p in zip(yvals, ycount)}
    steps.append(dict(col="Year", kind="marginal_int", year_logp=yp))

    for i, (col, kind, feats) in enumerate(CHAIN_STEPS[1:], start=2):
        feats_present = [f for f in feats if f in orig_subset.columns]
        X = orig_subset[feats_present].copy()
        y_raw = orig_subset[col]
        cats_in = [c for c in feats_present if c in CAT_OK]
        if kind == "gauss":
            yv = pd.to_numeric(y_raw, errors="coerce").astype(float).values
            mask = ~np.isnan(yv)
            X_fit, y_fit = X.loc[mask], yv[mask]
            model = _fit_reg(X_fit, y_fit, cats_in)
            pred_train = model.predict(X_fit, num_iteration=model.best_iteration)
            sigma = float(np.sqrt(np.mean((y_fit - pred_train) ** 2)))
            steps.append(dict(col=col, kind=kind, feats=feats_present,
                              cats=cats_in, model=model, sigma=sigma))
        elif kind == "multiclass":
            y_int = _safe_clip_int(y_raw, 0, 1_000_000).values
            uniq = np.sort(np.unique(y_int))
            level_map = {int(v): i for i, v in enumerate(uniq)}
            y_fit = np.array([level_map[int(v)] for v in y_int], dtype=np.int32)
            model = _fit_cls(X, y_fit, cats_in, num_class=len(uniq))
            steps.append(dict(col=col, kind=kind, feats=feats_present,
                              cats=cats_in, model=model, levels=uniq.tolist(),
                              level_map=level_map))
        elif kind == "binary":
            y_fit = _safe_clip_int(y_raw, 0, 1).values
            model = _fit_cls(X, y_fit, cats_in, num_class=2)
            steps.append(dict(col=col, kind=kind, feats=feats_present,
                              cats=cats_in, model=model))
    return steps


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
            out[f"ll_{safe}"] = ll
        elif kind == "gauss":
            X = df[s["feats"]].copy()
            pred = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            actual = pd.to_numeric(df[col], errors="coerce").astype(float).values
            mask = ~np.isnan(actual)
            ll = np.zeros(n, dtype=np.float32)
            _, llk = _gauss_ll(actual[mask], pred[mask], s["sigma"])
            ll[mask] = llk
            out[f"ll_{safe}"] = ll
        elif kind == "multiclass":
            X = df[s["feats"]].copy()
            proba = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            actual_raw = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int).values
            actual_idx = np.array([s["level_map"].get(int(v), -1) for v in actual_raw])
            ll = np.full(n, np.log(EPS), dtype=np.float32)
            ok = actual_idx >= 0
            if ok.any():
                ll[ok] = _cls_ll(actual_idx[ok], proba[ok], len(s["levels"]))
            out[f"ll_{safe}"] = ll
        elif kind == "binary":
            X = df[s["feats"]].copy()
            p1 = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            actual = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).values
            ll = _cls_ll(actual, p1, 2)
            out[f"ll_{safe}"] = ll
    return out


def main():
    t0 = time.time()
    print("[E4 class-conditional chain]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()

    orig_y0 = orig[orig[TARGET] == 0].copy()
    orig_y1 = orig[orig[TARGET] == 1].copy()
    print(f"  orig y=0: {len(orig_y0)}, orig y=1: {len(orig_y1)}")

    union = pd.concat([orig[CAT_OK], tr[CAT_OK], te[CAT_OK]], ignore_index=True)
    _, mappings = _encode_cat(union)
    orig_y0_e, _ = _encode_cat(orig_y0, mappings)
    orig_y1_e, _ = _encode_cat(orig_y1, mappings)
    tr_e, _ = _encode_cat(tr, mappings)
    te_e, _ = _encode_cat(te, mappings)

    print(f"\n[fit chain on orig[y=0] n={len(orig_y0_e)}]")
    steps_y0 = fit_chain_on_subset(orig_y0_e, label="y0")
    print(f"\n[fit chain on orig[y=1] n={len(orig_y1_e)}]")
    steps_y1 = fit_chain_on_subset(orig_y1_e, label="y1")

    print(f"\n[apply chains → train {len(tr_e)}]")
    tr_y0 = apply_chain(steps_y0, tr_e)
    tr_y1 = apply_chain(steps_y1, tr_e)
    print(f"[apply chains → test {len(te_e)}]")
    te_y0 = apply_chain(steps_y0, te_e)
    te_y1 = apply_chain(steps_y1, te_e)

    # Per-step log ratio = ll_y1 - ll_y0.
    cols = list(tr_y0.columns)
    tr_ratio = pd.DataFrame({f"logratio{c[2:]}": tr_y1[c].values - tr_y0[c].values
                             for c in cols}, index=tr_e.index)
    te_ratio = pd.DataFrame({f"logratio{c[2:]}": te_y1[c].values - te_y0[c].values
                             for c in cols}, index=te_e.index)
    tr_ratio["total_logratio"] = tr_ratio.values.sum(axis=1)
    te_ratio["total_logratio"] = te_ratio.values.sum(axis=1)

    print(f"  features: {len(tr_ratio.columns)}")
    print(f"  total_logratio stats:")
    print(f"    train mean {tr_ratio['total_logratio'].mean():+.3f}  "
          f"std {tr_ratio['total_logratio'].std():.3f}")
    # Quick AUC of total_logratio alone
    y = tr_e[TARGET].astype(int).values
    auc_alone = roc_auc_score(y, tr_ratio["total_logratio"].values)
    print(f"  total_logratio standalone AUC: {auc_alone:.5f}")

    # Save diagnostic parquets
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    tr_ratio.to_parquet(DATA_OUT / "class_cond_features_train.parquet")
    te_ratio.to_parquet(DATA_OUT / "class_cond_features_test.parquet")

    # Downstream LGBM (raw + log_ratio features)
    raw_cols = CAT_OK + NUM_FEATS
    tr_X = pd.concat([tr_e[raw_cols].reset_index(drop=True),
                      tr_ratio.reset_index(drop=True)], axis=1)
    te_X = pd.concat([te_e[raw_cols].reset_index(drop=True),
                      te_ratio.reset_index(drop=True)], axis=1)
    print(f"\n[downstream LGBM raw + class-cond log-ratio]")
    oof, test = downstream_lgbm(tr_X, y, te_X, CAT_OK)
    np.save(ART / "oof_d18_e4_class_cond_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d18_e4_class_cond_strat.npy",
            np.column_stack([1 - test, test]))
    auc = float(roc_auc_score(y, oof))
    print(f"\n[done E4]  wall {time.time()-t0:.0f}s  OOF {auc:.5f}  "
          f"  total_logratio_alone {auc_alone:.5f}")
    summary = dict(oof_auc=auc, total_logratio_alone_auc=auc_alone,
                   wall_s=time.time() - t0)
    (ART / "d18_e4_class_cond_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
