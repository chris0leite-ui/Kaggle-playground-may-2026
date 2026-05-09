"""Sprint C — orig-cell hierarchical label-vote + kNN-on-orig per cell (M4+M5).

For each synth row, compute:
  H1: orig-empirical PitNextLap rate at hierarchical cell-key levels
      L6 = (Y, C, PS, R, S, LapN)
      L5 = (Y, C, PS, R, S)
      L4 = (Y, C, PS, R)
      L3 = (Y, C, PS)
      L2 = (Y, C)
      L1 = (Y)
      L0 = global
      → 7 cell-rate features
      Empirical-Bayes shrinkage between levels: hierarchical Bayesian TE.

  H2: orig-kNN within (Y, C, PS) cell on standardised 4 continuous cols
      K=20 nearest orig neighbours, distance-weighted mean PitNextLap
      → 1 feature (cell-conditional kNN-vote)

  H3: log_density features per (Y, C, PS) cell — Gaussian per-cell log p(x)
      using orig per-cell mean+cov over (LapTime, Δ, CumDeg, RP)
      → 1 feature (cell-conditional log-density)

This implements the M4 + M5 mechanisms from the second-wave research
synthesis. The orig labels are external (not in CV stream) so the
features are inherently fold-safe (Rule 24 doesn't bite).

This base is structurally orthogonal to qAA (which uses synth-only
sequence features) — they should NOT be ρ-correlated.

Output: standalone OOF, ρ vs PRIMARY, K=4+1 LR-meta gate, +
        K=5 (K=4 + qAB) gate, + K=6 (K=4 + qAA + qAB) gate.
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

SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

CELL_KEYS = [
    ("L6", ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]),
    ("L5", ["Year", "Compound", "PitStop", "Race", "Stint"]),
    ("L4", ["Year", "Compound", "PitStop", "Race"]),
    ("L3", ["Year", "Compound", "PitStop"]),
    ("L2", ["Year", "Compound"]),
    ("L1", ["Year"]),
]


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def hierarchical_te(orig: pd.DataFrame, df: pd.DataFrame, k_smooth: float = 30.0) -> pd.DataFrame:
    """Hierarchical Bayesian-shrunk PitNextLap rate at each cell level.

    For each level L, output a column named f"orig_pn_{L}" containing the
    empirical mean of orig.PitNextLap within the row's cell, smoothed
    toward the parent level using empirical-Bayes count-based shrinkage:

        shrunk = (count * mean_cell + k_smooth * mean_parent) / (count + k_smooth)
    """
    out = pd.DataFrame(index=df.index)
    global_rate = float(orig[TARGET].mean())

    parent_rate = pd.Series(global_rate, index=df.index, dtype=np.float32)
    # Iterate levels from most specific to least specific, but in the
    # SHRINKAGE direction we go LEAST → MOST so each level shrinks toward parent.
    levels_for_shrink = list(reversed(CELL_KEYS))  # L1 first, then L2, ..., L6

    for level_name, keys in levels_for_shrink:
        cell_stats = orig.groupby(keys, observed=True)[TARGET].agg(["sum", "count"])
        cell_stats.columns = ["sum", "count"]
        merged = df[keys].merge(cell_stats, left_on=keys, right_index=True, how="left")
        merged["sum"] = merged["sum"].fillna(0.0)
        merged["count"] = merged["count"].fillna(0).astype(np.float32)

        # parent rate at this level — for L1 the parent is global; for L>1 it
        # is the result of the previous (less-specific) iteration
        rate = (merged["sum"] + k_smooth * parent_rate.values) / (merged["count"] + k_smooth)
        out[f"orig_pn_{level_name}"] = rate.astype(np.float32).values
        # And expose count as an auxiliary feature (sparsity diagnostic)
        out[f"orig_n_{level_name}"] = merged["count"].astype(np.float32).values

        # Update parent for next level (more specific shrinks toward this level's rate)
        parent_rate = rate.astype(np.float32)

    return out.reset_index(drop=True)


def cell_knn_vote(
    orig: pd.DataFrame, df: pd.DataFrame, cont_cols: list[str],
    cell_key: list[str], k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """For each row in df, find K nearest orig rows within its cell on
    standardised cont_cols. Return (kNN-distance-weighted PN-rate,
    median NN distance).
    """
    sc = StandardScaler().fit(orig[cont_cols].values)
    Xo_all = sc.transform(orig[cont_cols].values)
    Xq_all = sc.transform(df[cont_cols].values)

    out_pn = np.full(len(df), np.nan, dtype=np.float32)
    out_d  = np.full(len(df), np.nan, dtype=np.float32)

    # Group by cell
    orig_grp = orig.groupby(cell_key, observed=True).indices
    df_grp = df.groupby(cell_key, observed=True).indices

    n_cells_with_match = 0
    n_cells_total = 0
    for cell, q_idx in df_grp.items():
        n_cells_total += 1
        if cell not in orig_grp:
            continue
        o_idx = orig_grp[cell]
        if len(o_idx) < 1:
            continue
        n_cells_with_match += 1
        Xo = Xo_all[o_idx]
        Xq = Xq_all[q_idx]
        yo = orig[TARGET].values[o_idx]

        kk = min(k, len(o_idx))
        nn = NearestNeighbors(n_neighbors=kk, n_jobs=1).fit(Xo)
        d, idx = nn.kneighbors(Xq)

        # Distance-weighted mean (1/(d+eps))
        w = 1.0 / (d + 1e-3)
        wn = w / w.sum(axis=1, keepdims=True)
        votes = (yo[idx] * wn).sum(axis=1)
        out_pn[q_idx] = votes.astype(np.float32)
        out_d[q_idx]  = np.median(d, axis=1).astype(np.float32)

    print(f"    kNN cells matched: {n_cells_with_match}/{n_cells_total}; "
          f"queries assigned: {(~np.isnan(out_pn)).sum()}/{len(df)}", flush=True)
    return out_pn, out_d


def main():
    ts = time.time()
    out: dict = {}

    train = pd.read_csv(DATA / "train.csv").rename(columns={"LapTime (s)": "LapTime"})
    test = pd.read_csv(DATA / "test.csv").rename(columns={"LapTime (s)": "LapTime"})
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv").rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)
    t(f"train {train.shape} test {test.shape} orig {orig.shape}", ts)

    cont_cols = ["LapTime", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]

    # Hierarchical TE on orig
    h_train = hierarchical_te(orig, train)
    h_test = hierarchical_te(orig, test)
    t(f"hierarchical TE: train cols {h_train.shape}, test cols {h_test.shape}", ts)

    # Cell kNN vote on (Y, C, PS) — coarse but reliable
    knn_train_pn, knn_train_d = cell_knn_vote(orig, train, cont_cols,
                                              ["Year", "Compound", "PitStop"], k=20)
    knn_test_pn, knn_test_d = cell_knn_vote(orig, test, cont_cols,
                                            ["Year", "Compound", "PitStop"], k=20)
    t(f"kNN done; train NaN frac: {np.isnan(knn_train_pn).mean():.3f}", ts)

    # Fill NaN kNN with hierarchical TE level-3
    knn_train_pn_fill = np.where(np.isnan(knn_train_pn),
                                 h_train["orig_pn_L3"].values, knn_train_pn)
    knn_test_pn_fill = np.where(np.isnan(knn_test_pn),
                                h_test["orig_pn_L3"].values, knn_test_pn)
    knn_train_d_fill = np.where(np.isnan(knn_train_d), 0.0, knn_train_d)
    knn_test_d_fill = np.where(np.isnan(knn_test_d), 0.0, knn_test_d)

    # Per-cell Gaussian log-density on (Y, C, PS) — captures the "deep in synth
    # manifold" axis that the host's per-cell density generator imposes
    def cell_gauss_logp(orig_, df_, cont_cols, cell_key):
        out_lp = np.full(len(df_), -1e6, dtype=np.float32)
        orig_grp = orig_.groupby(cell_key, observed=True).indices
        df_grp = df_.groupby(cell_key, observed=True).indices
        for cell, q_idx in df_grp.items():
            if cell not in orig_grp:
                continue
            o_idx = orig_grp[cell]
            if len(o_idx) < 10:
                continue
            mu = orig_[cont_cols].values[o_idx].mean(axis=0)
            cov = np.cov(orig_[cont_cols].values[o_idx].T) + 1e-3 * np.eye(len(cont_cols))
            Xq = df_[cont_cols].values[q_idx]
            try:
                inv = np.linalg.inv(cov)
                sign, lod = np.linalg.slogdet(cov)
                if sign <= 0:
                    continue
                diff = Xq - mu
                m = np.einsum("ij,jk,ik->i", diff, inv, diff)
                logp = -0.5 * (m + lod + len(cont_cols) * np.log(2*np.pi))
                out_lp[q_idx] = logp.astype(np.float32)
            except np.linalg.LinAlgError:
                continue
        return out_lp

    lp_train = cell_gauss_logp(orig, train, cont_cols, ["Year", "Compound", "PitStop"])
    lp_test = cell_gauss_logp(orig, test, cont_cols, ["Year", "Compound", "PitStop"])
    t(f"per-cell log-density done; train NaN frac: {(lp_train==-1e6).mean():.3f}", ts)

    # Build feature matrix: ONLY the new orig-derived features (~14 features)
    new_feats = pd.DataFrame()
    for col in h_train.columns:
        new_feats[col] = h_train[col].values
    new_feats["orig_knn_pn"] = knn_train_pn_fill
    new_feats["orig_knn_d_med"] = knn_train_d_fill
    new_feats["cell_log_density"] = lp_train

    new_feats_test = pd.DataFrame()
    for col in h_test.columns:
        new_feats_test[col] = h_test[col].values
    new_feats_test["orig_knn_pn"] = knn_test_pn_fill
    new_feats_test["orig_knn_d_med"] = knn_test_d_fill
    new_feats_test["cell_log_density"] = lp_test

    feat_cols = list(new_feats.columns)
    out["feat_cols"] = feat_cols
    out["n_features"] = len(feat_cols)
    print(f"  feat_cols: {feat_cols}", flush=True)

    # Train LightGBM with ONLY these new features (cleanest test of orthogonality)
    X = new_feats.values
    y = train[TARGET].values
    X_test = new_feats_test.values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs = []
    lgb_params = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        t1 = time.time()
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        val_pred = m.predict_proba(X[va])[:, 1]
        oof[va] = val_pred
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        a = float(roc_auc_score(y[va], val_pred))
        fold_aucs.append(a)
        print(f"  fold {fold+1} AUC = {a:.5f}  ({time.time()-t1:.0f}s)  best_iter={m.best_iteration_}", flush=True)

    auc = float(roc_auc_score(y, oof))
    print(f"\n=== qAB standalone OOF AUC = {auc:.5f} (orig-derived features only) ===", flush=True)
    out["fold_aucs"] = fold_aucs
    out["oof_auc"] = auc

    # rho vs K=4 PRIMARY
    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")
    if primary_oof.ndim == 2:
        primary_oof = primary_oof[:, 1]
    if primary_test.ndim == 2:
        primary_test = primary_test[:, 1]
    rho_oof = float(spearmanr(oof, primary_oof).correlation)
    rho_test = float(spearmanr(test_pred, primary_test).correlation)
    primary_auc = float(roc_auc_score(y, primary_oof))
    print(f"  PRIMARY K=4 OOF AUC: {primary_auc:.5f}; rho_test: {rho_test:.5f}", flush=True)
    out["rho_test_vs_primary"] = rho_test
    out["rho_oof_vs_primary"] = rho_oof

    # K=4+1 gate
    BASES = [
        ("d17_h1d_yekenot_full", "oof_d17_h1d_yekenot_full_strat.npy", "test_d17_h1d_yekenot_full_strat.npy"),
        ("p1_single_cb_v4_gpu", "oof_p1_single_cb_v4_gpu_strat.npy", "test_p1_single_cb_v4_gpu_strat.npy"),
        ("f1_hgbc_deep", "oof_f1_hgbc_deep_strat.npy", "test_f1_hgbc_deep_strat.npy"),
        ("d16_orig_continuous_only", "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ]
    base_oofs, base_tests = [], []
    for name, oof_f, test_f in BASES:
        o = np.load(ART / oof_f)
        te = np.load(ART / test_f)
        if o.ndim == 2: o = o[:, 1]
        if te.ndim == 2: te = te[:, 1]
        base_oofs.append(o); base_tests.append(te)

    def expand(p_list):
        cols = []
        for p in p_list:
            p = np.clip(p, 1e-6, 1 - 1e-6)
            cols += [p, pd.Series(p).rank().values / len(p), np.log(p / (1 - p))]
        return np.column_stack(cols)

    Xm_K4 = expand(base_oofs)
    Xm_K5 = expand(base_oofs + [oof])
    def lr_meta_oof(Xm, y_):
        skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        om = np.zeros(len(y_))
        for tr, va in skf2.split(Xm, y_):
            mlr = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=SEED)
            mlr.fit(Xm[tr], y_[tr])
            om[va] = mlr.predict_proba(Xm[va])[:, 1]
        return om

    auc_K4 = float(roc_auc_score(y, lr_meta_oof(Xm_K4, y)))
    auc_K5 = float(roc_auc_score(y, lr_meta_oof(Xm_K5, y)))
    delta_bp = (auc_K5 - auc_K4) * 1e4
    print(f"\n  K=4 LR-meta: {auc_K4:.5f}", flush=True)
    print(f"  K=5 (K=4+qAB) LR-meta: {auc_K5:.5f}", flush=True)
    print(f"  K=4+1 lift: {delta_bp:+.3f} bp", flush=True)
    print(f"  GATE: {'PASS' if delta_bp >= 0.5 else ('WEAK' if delta_bp > -0.3 else 'FAIL')}", flush=True)
    out["k4plus1_lift_bp"] = delta_bp
    out["gate_verdict"] = "PASS" if delta_bp >= 0.5 else ("WEAK" if delta_bp > -0.3 else "FAIL")

    # Also test K=4 + qAA + qAB (joint) if qAA artifacts exist
    qaa_oof_path = ART / "dgp_v3_qAA_stint_imputed_oof.npy"
    qaa_test_path = ART / "dgp_v3_qAA_stint_imputed_test.npy"
    if qaa_oof_path.exists() and qaa_test_path.exists():
        qaa_oof = np.load(qaa_oof_path)
        qaa_test = np.load(qaa_test_path)
        Xm_K6 = expand(base_oofs + [qaa_oof, oof])
        auc_K6 = float(roc_auc_score(y, lr_meta_oof(Xm_K6, y)))
        delta_K6 = (auc_K6 - auc_K4) * 1e4
        print(f"\n  K=4 + qAA + qAB LR-meta: {auc_K6:.5f}", flush=True)
        print(f"  K=4+2 lift: {delta_K6:+.3f} bp", flush=True)
        out["k4plus2_qaa_qab_lift_bp"] = delta_K6
        # Pairwise rho
        rho_qaa_qab = float(spearmanr(oof, qaa_oof).correlation)
        out["rho_qaa_vs_qab_oof"] = rho_qaa_qab
        print(f"  rho(qAA, qAB) on OOF: {rho_qaa_qab:.5f}", flush=True)

    np.save(ART / "dgp_v3_qAB_orig_cell_oof.npy", oof)
    np.save(ART / "dgp_v3_qAB_orig_cell_test.npy", test_pred)
    fp = ART / "dgp_v3_qAB_orig_cell.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
