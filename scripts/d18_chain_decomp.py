"""d18 — Iterative chain-decomposition of P(X) on orig, applied to synth.

Reverse-engineering the data-generating process. The synthesizer learned a
multivariate distribution over the orig dataset (aadigupta1601 F1 strategy)
and sampled. Marginals are near-identical (97.55% LapTime overlap per d15)
but joints are corrupted (5.8% triple survival). d14 DGP-residuals showed
within-row features are conditionally near-independent — but that probe
self-predicted from synth, so it conflated synthesizer corruption with the
orig DGP. This probe instead **conditions on the orig DGP chain**.

Approach.

Pick a domain-causal ordering of features (Year → Race → Compound → Stint
→ LapNumber → TyreLife → RaceProgress → Position → LapTime → LapTime_Delta
→ Cumulative_Degradation → Position_Change → PitStop). For each step k,
fit a small LGBM on **orig only** modelling P(X_k | X_{<k}). For each synth
row compute:

  chain_ll_{col}  : per-step orig-log-likelihood of the synth value
  chain_z_{col}   : per-step (actual - pred_mean) / sigma_orig (continuous)
  chain_anomaly_L1, chain_total_ll : composite scores

Then 5-fold LGBM on (raw 14 features + ~24 chain features) → PitNextLap.
This is the candidate base. The full per-row chain-feature matrix is the
**diagnostic artifact** for E2-E5 (saved as parquet for reuse).

Driver is excluded as both target and feature (orig has 31 historical
codes; synth has 856 ghost D-codes — no overlap), per the d16 KS-driven
rule.

Outputs:
  scripts/artifacts/oof_d18_chain_decomp_strat.npy        # base OOF (n_train, 2)
  scripts/artifacts/test_d18_chain_decomp_strat.npy       # base test pred
  scripts/artifacts/oof_d18_chain_decomp_lgbm_only_strat.npy   # raw-only ablation
  scripts/artifacts/test_d18_chain_decomp_lgbm_only_strat.npy
  data/chain_decomp_features_train.parquet                # diagnostic
  data/chain_decomp_features_test.parquet
  scripts/artifacts/d18_chain_decomp_summary.json         # per-step diagnostics
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
DATA_OUT = Path("data")
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"
LAPTIME = "LapTime (s)"
EPS = 1e-9

# Categorical features in orig and synth that can co-train. Driver excluded.
CAT_OK = ["Compound", "Race"]
NUM_FEATS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             LAPTIME, "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change"]


def _encode_cat(df, mappings=None):
    """Integer-encode CAT_OK columns. Build / reuse mappings dict."""
    out = df.copy()
    if mappings is None:
        mappings = {}
        for c in CAT_OK:
            levels = sorted(df[c].dropna().astype(str).unique().tolist())
            mappings[c] = {v: i for i, v in enumerate(levels)}
    for c in CAT_OK:
        out[c] = out[c].astype(str).map(mappings[c]).astype("Int64")
    return out, mappings


def _lgb_dataset(X, y, cat_cols):
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    return lgb.Dataset(X, label=y, categorical_feature=cat_idx,
                       free_raw_data=False)


def _fit_reg(X, y, cat_cols, n_round=400, smoke=False):
    p = dict(objective="regression", metric="rmse", learning_rate=0.05,
             num_leaves=63, min_data_in_leaf=50, verbosity=-1, seed=SEED)
    if smoke:
        n_round = 80
    return lgb.train(p, _lgb_dataset(X, y, cat_cols), num_boost_round=n_round)


def _fit_cls(X, y, cat_cols, num_class, n_round=300, smoke=False):
    if num_class == 2:
        p = dict(objective="binary", metric="binary_logloss",
                 learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                 verbosity=-1, seed=SEED)
    else:
        p = dict(objective="multiclass", num_class=int(num_class),
                 metric="multi_logloss", learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=50, verbosity=-1, seed=SEED)
    if smoke:
        n_round = 60
    return lgb.train(p, _lgb_dataset(X, y, cat_cols), num_boost_round=n_round)


def _gauss_ll(actual, pred_mean, sigma):
    sigma = max(float(sigma), EPS)
    z = (actual - pred_mean) / sigma
    ll = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * z ** 2
    return z.astype(np.float32), ll.astype(np.float32)


def _cls_ll(actual_int, pred_proba, n_classes):
    n = len(actual_int)
    if pred_proba.ndim == 1:  # binary
        p1 = np.clip(pred_proba, EPS, 1 - EPS)
        p = np.where(actual_int == 1, p1, 1 - p1)
    else:
        idx = np.clip(actual_int, 0, n_classes - 1)
        p = np.clip(pred_proba[np.arange(n), idx], EPS, 1.0)
    return np.log(p).astype(np.float32)


# ---- Chain definition ----------------------------------------------------
# Each step: (target_col, kind, conditioning_features). 'kind' ∈
# {'gauss', 'multiclass', 'binary'}. Year is modelled by its empirical
# marginal (no LGBM); included for completeness.
CHAIN_STEPS = [
    # (target, kind, feats)
    ("Year",                   "marginal_int", []),
    ("Race",                   "multiclass",   ["Year"]),
    ("Compound",               "multiclass",   ["Year", "Race", "Stint"]),
    ("Stint",                  "multiclass",   ["Year", "Race", "Compound"]),
    ("LapNumber",              "gauss",        ["Year", "Race", "Compound", "Stint"]),
    ("TyreLife",               "gauss",        ["Year", "Race", "Compound", "Stint", "LapNumber"]),
    ("RaceProgress",           "gauss",        ["Year", "Race", "LapNumber"]),
    ("Position",               "gauss",        ["Year", "Race", "Compound", "Stint", "LapNumber", "TyreLife"]),
    (LAPTIME,                  "gauss",        ["Year", "Race", "Compound", "Stint", "LapNumber", "TyreLife", "Position"]),
    ("LapTime_Delta",          "gauss",        ["Compound", "TyreLife", LAPTIME, "LapNumber", "Stint"]),
    ("Cumulative_Degradation", "gauss",        ["Compound", "Stint", "TyreLife", "LapTime_Delta", LAPTIME]),
    ("Position_Change",        "gauss",        ["Year", "Race", "LapNumber", "Position"]),
    ("PitStop",                "binary",       ["Year", "Race", "Compound", "Stint", "LapNumber", "TyreLife", "Position", "RaceProgress"]),
]


def _safe_clip_int(s, vmin, vmax):
    s = pd.to_numeric(s, errors="coerce")
    s = s.fillna(vmin)
    return s.clip(vmin, vmax).astype(int)


def _safe_col(df, col):
    """Return numeric series, NaN-filled. Categorical→already int from _encode_cat."""
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) or s.dtype.name in {"Int64", "Int32"}:
        return pd.to_numeric(s, errors="coerce").astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)


def fit_chain(orig: pd.DataFrame, smoke=False, log=print):
    """Fit each chain step on orig. Return list of step dicts and Compound/Race
    integer mappings (built from union of orig+synth before training)."""
    steps = []

    # Year marginal (4-class categorical-ish int)
    yvals, ycount = np.unique(orig["Year"].astype(int).values, return_counts=True)
    ycount = ycount.astype(np.float64) / ycount.sum()
    year_logp = {int(y): float(np.log(p)) for y, p in zip(yvals, ycount)}
    steps.append(dict(col="Year", kind="marginal_int", year_logp=year_logp))
    log(f"  step  1/{len(CHAIN_STEPS)} Year   marginal logp over {len(yvals)} levels")

    # Build mappings using orig+synth-train+synth-test union for cat cols
    # so categorical_feature codes line up at predict time.
    # (deferred to caller — we accept already-encoded df here)

    for i, (col, kind, feats) in enumerate(CHAIN_STEPS[1:], start=2):
        t0 = time.time()
        feats_present = [f for f in feats if f in orig.columns]
        X = orig[feats_present].copy()
        y_raw = orig[col]
        cats_in = [c for c in feats_present if c in CAT_OK]

        if kind == "gauss":
            y = pd.to_numeric(y_raw, errors="coerce").astype(float).values
            mask = ~np.isnan(y)
            X_fit, y_fit = X.loc[mask], y[mask]
            model = _fit_reg(X_fit, y_fit, cats_in, smoke=smoke)
            pred_train = model.predict(X_fit, num_iteration=model.best_iteration)
            sigma = float(np.sqrt(np.mean((y_fit - pred_train) ** 2)))
            steps.append(dict(col=col, kind=kind, feats=feats_present,
                              cats=cats_in, model=model, sigma=sigma))
        elif kind == "multiclass":
            # Map levels (already integer-encoded for CAT_OK; integer-binned
            # for numeric cats like Compound/Stint may need an own mapping).
            y_int = _safe_clip_int(y_raw, 0, 1_000_000).values
            uniq = np.sort(np.unique(y_int))
            level_map = {int(v): i for i, v in enumerate(uniq)}
            y_fit = np.array([level_map[int(v)] for v in y_int], dtype=np.int32)
            model = _fit_cls(X, y_fit, cats_in, num_class=len(uniq), smoke=smoke)
            steps.append(dict(col=col, kind=kind, feats=feats_present,
                              cats=cats_in, model=model, levels=uniq.tolist(),
                              level_map=level_map))
        elif kind == "binary":
            y_fit = _safe_clip_int(y_raw, 0, 1).values
            model = _fit_cls(X, y_fit, cats_in, num_class=2, smoke=smoke)
            steps.append(dict(col=col, kind=kind, feats=feats_present,
                              cats=cats_in, model=model))
        else:
            raise ValueError(kind)

        log(f"  step {i:>2}/{len(CHAIN_STEPS)} {col:24s} "
            f"{kind:10s} | {len(feats_present)} feats | "
            f"σ={steps[-1].get('sigma', '-'):>8} | {time.time()-t0:.1f}s")
    return steps


def apply_chain(steps, df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-step LL + z-score features for the rows of df."""
    out = pd.DataFrame(index=df.index)
    n = len(df)
    for s in steps:
        col, kind = s["col"], s["kind"]
        if kind == "marginal_int":
            yvals = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int).values
            ll = np.array([s["year_logp"].get(int(v), np.log(EPS))
                           for v in yvals], dtype=np.float32)
            out[f"chain_ll_{col}"] = ll
        elif kind == "gauss":
            X = df[s["feats"]].copy()
            pred = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            actual = pd.to_numeric(df[col], errors="coerce").astype(float).values
            mask = ~np.isnan(actual)
            z = np.zeros(n, dtype=np.float32); ll = np.zeros(n, dtype=np.float32)
            zk, llk = _gauss_ll(actual[mask], pred[mask], s["sigma"])
            z[mask] = zk; ll[mask] = llk
            safe = col.replace(" ", "_").replace("(", "").replace(")", "")
            out[f"chain_z_{safe}"] = z
            out[f"chain_ll_{safe}"] = ll
        elif kind == "multiclass":
            X = df[s["feats"]].copy()
            pred_proba = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            actual_raw = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int).values
            actual_idx = np.array([s["level_map"].get(int(v), -1) for v in actual_raw])
            n_cls = len(s["levels"])
            ll = np.empty(n, dtype=np.float32)
            ok = actual_idx >= 0
            if ok.any():
                ll[ok] = _cls_ll(actual_idx[ok], pred_proba[ok], n_cls)
            ll[~ok] = np.log(EPS)
            out[f"chain_ll_{col}"] = ll
        elif kind == "binary":
            X = df[s["feats"]].copy()
            p1 = s["model"].predict(X, num_iteration=s["model"].best_iteration)
            actual = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).values
            ll = _cls_ll(actual, p1, 2)
            out[f"chain_ll_{col}"] = ll
    # Composite anomaly + total log-likelihood
    z_cols = [c for c in out.columns if c.startswith("chain_z_")]
    ll_cols = [c for c in out.columns if c.startswith("chain_ll_")]
    out["chain_anomaly_L1"] = np.abs(out[z_cols].values).sum(axis=1) if z_cols else 0.0
    out["chain_total_ll"] = out[ll_cols].values.sum(axis=1) if ll_cols else 0.0
    return out


def _save_oof_test(name, oof_pos, test_pos):
    n_tr = len(oof_pos); n_te = len(test_pos)
    oof = np.column_stack([1.0 - oof_pos, oof_pos]).astype(np.float64)
    test = np.column_stack([1.0 - test_pos, test_pos]).astype(np.float64)
    np.save(ART / f"oof_{name}_strat.npy", oof)
    np.save(ART / f"test_{name}_strat.npy", test)
    print(f"  saved oof_{name}_strat.npy  ({n_tr},2)  test_{name}_strat.npy ({n_te},2)")


def downstream_lgbm(tr_X, tr_y, te_X, cat_cols, smoke=False):
    """5-fold StratifiedKFold LGBM. Returns (oof_pos, test_pos_avg)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr_y), dtype=np.float64)
    test_avg = np.zeros(len(te_X), dtype=np.float64)
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=127, min_data_in_leaf=80, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbosity=-1, seed=SEED)
    n_round = 800 if not smoke else 80
    for fi, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(tr_y)), tr_y), 1):
        X_tr = tr_X.iloc[tr_i]; y_tr = tr_y[tr_i]
        X_va = tr_X.iloc[va_i]; y_va = tr_y[va_i]
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]
        ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_idx,
                            free_raw_data=False)
        ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_idx,
                            reference=ds_tr, free_raw_data=False)
        m = lgb.train(p, ds_tr, num_boost_round=n_round, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_i] = m.predict(X_va, num_iteration=m.best_iteration)
        test_avg += m.predict(te_X, num_iteration=m.best_iteration) / N_FOLDS
        auc = roc_auc_score(y_va, oof[va_i])
        print(f"    fold {fi}: AUC={auc:.5f}  best_iter={m.best_iteration}")
    print(f"    OOF AUC = {roc_auc_score(tr_y, oof):.5f}")
    return oof, test_avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 fold + tiny rounds")
    ap.add_argument("--n_synth_sample", type=int, default=0,
                    help="if >0, subsample synth-train/test for smoke")
    ap.add_argument("--out_prefix", default="d18_chain_decomp",
                    help="output filename prefix")
    args = ap.parse_args()
    t0 = time.time()
    smoke = args.smoke

    print(f"[d18 chain-decomp{' SMOKE' if smoke else ''}]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    print(f"  train {tr.shape}  test {te.shape}  orig {orig.shape}")

    if args.n_synth_sample > 0:
        tr = tr.sample(n=args.n_synth_sample, random_state=SEED).reset_index(drop=True)
        te = te.sample(n=min(args.n_synth_sample // 2, len(te)),
                       random_state=SEED).reset_index(drop=True)
        print(f"  smoke-sampled: train {tr.shape}  test {te.shape}")

    # Build CAT_OK mapping over union of all three sources so categorical
    # feature codes are consistent across orig training and synth predict.
    union = pd.concat([orig[CAT_OK], tr[CAT_OK], te[CAT_OK]], ignore_index=True)
    _, mappings = _encode_cat(union)
    orig_e, _ = _encode_cat(orig, mappings)
    tr_e,   _ = _encode_cat(tr, mappings)
    te_e,   _ = _encode_cat(te, mappings)

    print(f"\n[fit chain on orig n={len(orig_e)}]")
    steps = fit_chain(orig_e, smoke=smoke)

    print(f"\n[apply chain → train {len(tr_e)}]")
    tr_chain = apply_chain(steps, tr_e)
    print(f"  features: {list(tr_chain.columns)[:5]}... ({len(tr_chain.columns)} total)")

    print(f"[apply chain → test  {len(te_e)}]")
    te_chain = apply_chain(steps, te_e)

    # Save diagnostic parquets (gitignored under data/).
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    diag_train = pd.concat([tr_e[["id"] + CAT_OK + NUM_FEATS + [TARGET]],
                            tr_chain.reset_index(drop=True)], axis=1)
    diag_test = pd.concat([te_e[["id"] + CAT_OK + NUM_FEATS],
                           te_chain.reset_index(drop=True)], axis=1)
    diag_train.to_parquet(DATA_OUT / "chain_decomp_features_train.parquet")
    diag_test.to_parquet(DATA_OUT / "chain_decomp_features_test.parquet")
    print(f"  diagnostic parquets saved under data/ ({len(diag_train)} + {len(diag_test)} rows)")

    # ---- Per-step diagnostic summary --------------------------------
    summary = {"steps": [], "n_orig": int(len(orig_e)),
               "n_train": int(len(tr_e)), "n_test": int(len(te_e))}
    for col in tr_chain.columns:
        if not col.startswith("chain_"):
            continue
        v_tr = tr_chain[col].values
        v_te = te_chain[col].values
        summary["steps"].append(dict(
            feature=col,
            tr_mean=float(np.nanmean(v_tr)), tr_std=float(np.nanstd(v_tr)),
            te_mean=float(np.nanmean(v_te)), te_std=float(np.nanstd(v_te))))
    # Per-step KS-diff between y=0 and y=1 inside synth — the load-bearing
    # diagnostic for "which step's likelihood routes the target".
    from scipy.stats import ks_2samp
    y = tr_e[TARGET].astype(int).values
    pos = (y == 1); neg = (y == 0)
    ks = []
    for col in tr_chain.columns:
        if col.startswith("chain_ll_") or col == "chain_total_ll":
            v = tr_chain[col].values
            stat, p = ks_2samp(v[pos], v[neg])
            ks.append(dict(feature=col, ks_y0_vs_y1=float(stat),
                           pvalue=float(p)))
    ks.sort(key=lambda r: -r["ks_y0_vs_y1"])
    summary["class_conditional_ks"] = ks
    (ART / f"{args.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  per-step KS y=0 vs y=1 (top 5): "
          f"{[(r['feature'], round(r['ks_y0_vs_y1'], 3)) for r in ks[:5]]}")

    # ---- Downstream LGBM head ---------------------------------------
    raw_cols = CAT_OK + NUM_FEATS
    chain_cols = list(tr_chain.columns)

    # Variant A: raw-only LGBM (ablation; matches existing K=21 e3-class)
    print(f"\n[downstream LGBM raw-only ablation]")
    tr_X_raw = tr_e[raw_cols].copy()
    te_X_raw = te_e[raw_cols].copy()
    y = tr_e[TARGET].astype(int).values
    oof_raw, test_raw = downstream_lgbm(tr_X_raw, y, te_X_raw, CAT_OK, smoke=smoke)
    _save_oof_test(f"{args.out_prefix}_lgbm_only", oof_raw, test_raw)

    # Variant B: raw + chain features (the candidate base)
    print(f"\n[downstream LGBM raw + chain features]")
    tr_X = pd.concat([tr_X_raw.reset_index(drop=True),
                      tr_chain.reset_index(drop=True)], axis=1)
    te_X = pd.concat([te_X_raw.reset_index(drop=True),
                      te_chain.reset_index(drop=True)], axis=1)
    oof, test = downstream_lgbm(tr_X, y, te_X, CAT_OK, smoke=smoke)
    _save_oof_test(f"{args.out_prefix}", oof, test)

    summary["oof_auc_raw_only"] = float(roc_auc_score(y, oof_raw))
    summary["oof_auc_with_chain"] = float(roc_auc_score(y, oof))
    summary["delta_bp_chain_vs_raw"] = float(
        (summary["oof_auc_with_chain"] - summary["oof_auc_raw_only"]) * 1e4)
    (ART / f"{args.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[done]  total wall: {time.time() - t0:.1f}s")
    print(f"  raw-only OOF AUC:    {summary['oof_auc_raw_only']:.5f}")
    print(f"  raw+chain OOF AUC:   {summary['oof_auc_with_chain']:.5f}")
    print(f"  Δ (bp) vs raw-only:  {summary['delta_bp_chain_vs_raw']:+.2f}")
    print(f"  → next: python scripts/probe_min_meta.py --candidates d18_chain_decomp")


if __name__ == "__main__":
    main()
