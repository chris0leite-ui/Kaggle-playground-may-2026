"""scripts/probe_r4_hmm_seq.py — Round 4 super-model attempt: HMM sequence

PI-approved super-model exploration. Mechanism-ledger flags HMM
transitions as the open sequence-class path. The within-stint
fingerprint LGBM scored +0.15 bp at K=4+1; HMM should 2-3x that.

Approach:
- Per (Year, Race, Driver) sequence, lap-ordered.
- Observations: (Compound_int, TyreLife, RaceProgress, Stint,
  Position_Change, Cumulative_Degradation).
- GaussianHMM with K=8 hidden states; fit on TRAIN sequences only.
- For each row, compute smoothed posterior P(state | full sequence)
  via forward-backward (hmmlearn predict_proba).
- Add 8 state-posterior features + log-likelihood per row.
- Train LightGBM 5-fold Stratified OOF on raw 14 + 9 HMM features.

Rule 24: HMM is unsupervised (no label use); fold-safe by construction.
Rule 25: train-only fit, AV-AUC 0.502 makes transductive borrowing safe
anyway.

Usage:
  python scripts/probe_r4_hmm_seq.py [--smoke] [--n-states 8]
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)

LGB_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    lambda_l1=0.0, lambda_l2=1.0, max_depth=-1, n_jobs=-1,
    verbose=-1, random_state=SEED,
)

HMM_OBS = ["Compound_int", "TyreLife", "RaceProgress", "Stint",
           "Position_Change", "Cumulative_Degradation"]


def build_sequences(df: pd.DataFrame, obs_cols: list[str]):
    """Sort df by (Year, Race, Driver, LapNumber). Return (X, lengths,
    seq_index) where X is concatenated observations, lengths is the
    per-sequence row count, seq_index maps original df-row index ->
    position in X. We keep original ordering."""
    sort_cols = ["Year", "Race", "Driver", "LapNumber"]
    sort_idx = df.sort_values(sort_cols).index.values
    df_sorted = df.loc[sort_idx]
    grp = df_sorted.groupby(["Year", "Race", "Driver"], sort=False)
    lengths = grp.size().values.astype(int)
    X = df_sorted[obs_cols].values.astype(np.float64)
    # inverse permutation: for sorted-position i, get back the original df-index
    inv = np.empty_like(sort_idx)
    inv[np.argsort(sort_idx)] = np.arange(len(sort_idx))
    return X, lengths, inv, sort_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 50k rows, 200 max-rounds")
    ap.add_argument("--name", default="r4_hmm_seq")
    ap.add_argument("--max-rounds", type=int, default=2000)
    ap.add_argument("--n-states", type=int, default=8,
                    help="HMM hidden state count")
    ap.add_argument("--hmm-iter", type=int, default=30,
                    help="Baum-Welch EM iteration cap")
    args = ap.parse_args()

    print(f"=== R4 HMM sequence | smoke={args.smoke} K={args.n_states} ===")
    t0 = time.time()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}")

    # Map Compound to int (consistent train+test)
    compound_map = {"HARD": 0, "MEDIUM": 1, "SOFT": 2,
                    "INTERMEDIATE": 3, "WET": 4}
    train["Compound_int"] = train["Compound"].map(compound_map).astype(np.float32)
    test["Compound_int"] = test["Compound"].map(compound_map).astype(np.float32)

    # Fit HMM on TRAIN sequences only
    t_hmm = time.time()
    X_train_seq, len_train, inv_train, sort_train = build_sequences(train, HMM_OBS)
    print(f"  train sequences: {len(len_train):,}  "
          f"avg-len {len_train.mean():.1f}  total-rows {X_train_seq.shape[0]:,}")

    if args.smoke:
        # Keep only first ~5000 sequences for smoke fit
        n_seq_smoke = min(5000, len(len_train))
        rows_keep = int(len_train[:n_seq_smoke].sum())
        X_train_seq_fit = X_train_seq[:rows_keep]
        len_train_fit = len_train[:n_seq_smoke]
        print(f"  SMOKE -> HMM fit on first {n_seq_smoke} seqs / "
              f"{rows_keep} rows")
    else:
        X_train_seq_fit = X_train_seq
        len_train_fit = len_train

    # Standardize observations for HMM (helps Gaussian)
    obs_mean = X_train_seq_fit.mean(axis=0)
    obs_std = X_train_seq_fit.std(axis=0) + 1e-6
    X_train_seq_fit_s = (X_train_seq_fit - obs_mean) / obs_std
    X_train_seq_s = (X_train_seq - obs_mean) / obs_std

    hmm = GaussianHMM(n_components=args.n_states,
                       covariance_type="diag",
                       n_iter=args.hmm_iter,
                       tol=1e-3,
                       random_state=SEED,
                       init_params="stmc",
                       verbose=False)
    hmm.fit(X_train_seq_fit_s, len_train_fit)
    print(f"  HMM fit done: {time.time()-t_hmm:.1f}s, "
          f"converged at iter {hmm.monitor_.iter}, "
          f"final logL/row {hmm.monitor_.history[-1] / X_train_seq_fit_s.shape[0]:.4f}")

    # Compute smoothed posteriors for train + test
    t_post = time.time()
    train_posterior = hmm.predict_proba(X_train_seq_s, len_train)
    print(f"  train posterior shape {train_posterior.shape}  "
          f"({time.time()-t_post:.1f}s)")
    # Map sorted-order posteriors back to original row order
    train_posterior_orig = train_posterior[inv_train]

    X_test_seq, len_test, inv_test, sort_test = build_sequences(test, HMM_OBS)
    X_test_seq_s = (X_test_seq - obs_mean) / obs_std
    test_posterior = hmm.predict_proba(X_test_seq_s, len_test)
    test_posterior_orig = test_posterior[inv_test]
    print(f"  test posterior shape {test_posterior.shape}  "
          f"({time.time()-t_post:.1f}s)")

    # Also compute per-row log-likelihood (1 feature: log P(obs_t | state))
    # using emission-prob mass: log sum_s P(state=s | hist) * P(obs_t | state=s)
    # Approximate as just the marginal logL contribution at this row,
    # computed via score_samples. (Per-row score is the local LL contribution.)
    # For simplicity we use posterior entropy as a "confidence" feature.
    hmm_entropy_train = -(train_posterior_orig
                           * np.log(train_posterior_orig + 1e-12)).sum(axis=1)
    hmm_entropy_test = -(test_posterior_orig
                          * np.log(test_posterior_orig + 1e-12)).sum(axis=1)

    # Add HMM features back to train/test (in original row order)
    for k in range(args.n_states):
        train[f"hmm_s{k}"] = train_posterior_orig[:, k].astype(np.float32)
        test[f"hmm_s{k}"] = test_posterior_orig[:, k].astype(np.float32)
    train["hmm_entropy"] = hmm_entropy_train.astype(np.float32)
    test["hmm_entropy"] = hmm_entropy_test.astype(np.float32)
    new_cols = [f"hmm_s{k}" for k in range(args.n_states)] + ["hmm_entropy"]

    if args.smoke:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train), size=50_000, replace=False)
        train = train.iloc[np.sort(idx)].reset_index(drop=True)
        y_all = train[TARGET].astype(int).values
        args.max_rounds = min(args.max_rounds, 500)
        print(f"  SMOKE -> downstream LGBM train {train.shape}")

    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        u = pd.concat([train[c], test[c]], ignore_index=True).unique()
        m = {v: i for i, v in enumerate(u)}
        train[c] = train[c].map(m).astype(np.int32)
        test[c] = test[c].map(m).astype(np.int32)

    feat_cols = [c for c in train.columns if c not in {"id", TARGET}]
    print(f"  total features: {len(feat_cols)} "
          f"({len(feat_cols) - len(new_cols)} raw + {len(new_cols)} HMM)")

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_iter = list(skf.split(np.zeros(len(y_all)), y_all))
    if args.smoke:
        fold_iter = fold_iter[:1]

    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []

    for fold, (ti, vi) in enumerate(fold_iter, 1):
        t_fold = time.time()
        X_tr = train.iloc[ti][feat_cols].fillna(0).values
        X_va = train.iloc[vi][feat_cols].fillna(0).values
        X_te = test[feat_cols].fillna(0).values

        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=args.max_rounds)
        m.fit(X_tr, y_all[ti],
              eval_set=[(X_va, y_all[vi])],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

        oof_va = m.predict_proba(X_va)[:, 1]
        oof[vi] = oof_va
        if not args.smoke:
            test_pred += m.predict_proba(X_te)[:, 1] / N_FOLDS

        auc_va = roc_auc_score(y_all[vi], oof_va)
        fold_aucs.append(float(auc_va))
        print(f"  Fold {fold}: AUC={auc_va:.5f} iters={m.best_iteration_} "
              f"wall={time.time()-t_fold:.1f}s")

    auc_full = (fold_aucs[0] if args.smoke
                else float(roc_auc_score(y_all, oof)))
    print(f"\n  OOF AUC{'  (smoke fold-1)' if args.smoke else ' (full)'}: "
          f"{auc_full:.5f}  fold-std={np.std(fold_aucs):.5f}  "
          f"total wall={time.time()-t0:.1f}s")

    if not args.smoke:
        np.save(ART / f"oof_{args.name}_strat.npy",
                np.column_stack([1 - oof, oof]).astype(np.float64))
        np.save(ART / f"test_{args.name}_strat.npy",
                np.column_stack([1 - test_pred, test_pred]).astype(np.float64))
        (ART / f"{args.name}_results.json").write_text(json.dumps(dict(
            name=args.name, oof_auc=auc_full, fold_aucs=fold_aucs,
            n_states=args.n_states, new_cols=new_cols,
            hmm_iter=int(hmm.monitor_.iter),
        ), indent=2))
        print(f"\n  -> oof_{args.name}_strat.npy   test_{args.name}_strat.npy")


if __name__ == "__main__":
    main()
