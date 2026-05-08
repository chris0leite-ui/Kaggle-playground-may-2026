"""scripts/probe_exp2_lambdarank_per_stint.py — EXP-2.

Per-stint LambdaRank base. LightGBM with objective=lambdarank, group-id =
(Race, Driver, Year, Stint). Each stint is a small "query"; within-query,
the model learns which row is the pit row.

Decision rule:
  K=10+1 plain LR-meta delta vs K=10:
  >= +0.5 bp -> 4th-direction candidate
  -0.1..+0.5 -> ambiguous; record ρ for triangulation
  < -0.1 -> NULL

Q6 alignment: training objective is lambdarank, eval is row-AUC.
Lambdarank optimises pairwise rank inversions within groups.
Row-AUC is global pairwise rank inversion. They're correlated but
not identical — that's the point. Different objective may produce
predictions in a structurally different direction.

Cost: ~25 min CPU.
Outputs scripts/artifacts/probe_exp2_lambdarank.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 500

K10_FWD = [
    "d17_h1d_yekenot_full", "p1_single_cb_v4_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]

NUM_COLS = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
CAT_COLS = ["Driver", "Compound", "Race"]

LGB_RANK = dict(
    objective="lambdarank",
    metric="ndcg",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=80,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbose=-1,
    n_jobs=-1,
    seed=SEED,
    label_gain=[0, 1],
)


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def encode_cats(df):
    df = df.copy()
    for c in CAT_COLS:
        df[c + "_cat"] = df[c].astype("category").cat.codes.astype("int32")
    return df


def main():
    t0 = time.time()
    print("Loading train + K=10 OOFs ...")
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    train_enc = encode_cats(train)
    feats = NUM_COLS + [c + "_cat" for c in CAT_COLS]

    # Stint group-id (must be sorted contiguously per fold for lambdarank)
    train_enc["stint_id"] = (
        train_enc["Race"].astype(str) + "_" + train_enc["Driver"].astype(str)
        + "_" + train_enc["Year"].astype(str) + "_"
        + train_enc["Stint"].astype(str)
    )
    print(f"  stints: {train_enc['stint_id'].nunique():,}  "
          f"rows: {len(train_enc):,}  pos: {y.sum():,} "
          f"({y.mean()*100:.2f}%)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    print("\nTraining LightGBM with objective=lambdarank, "
          "group=(Race,Driver,Year,Stint) ...")
    oof = np.zeros(len(y))
    fold_aucs = []
    for fold, (tr, va) in enumerate(splits):
        # For lambdarank, sort training rows by stint_id so groups are
        # contiguous. Compute group sizes.
        tr_sort_order = np.argsort(train_enc["stint_id"].values[tr], kind="stable")
        tr_sorted = tr[tr_sort_order]
        tr_groups = train_enc["stint_id"].values[tr_sorted]
        # Group sizes
        _, counts = np.unique(tr_groups, return_counts=True)
        # Reorder rows to be contiguous by stint_id (preserve unique order)
        uniq_ids, first_idx = np.unique(tr_groups, return_index=True)
        # Use the discovery order; counts is in that order
        order = np.argsort(first_idx)
        # tr_groups is already in discovery order due to stable sort + unique()
        # so counts are aligned
        ds_tr = lgb.Dataset(
            train_enc[feats].iloc[tr_sorted].values,
            label=y[tr_sorted],
            group=counts,
        )
        # Validation: predict on va; eval as row-AUC implicitly (we just compute it)
        booster = lgb.train(
            LGB_RANK, ds_tr,
            num_boost_round=400,
            callbacks=[lgb.log_evaluation(0)],
        )
        oof[va] = booster.predict(train_enc[feats].iloc[va].values)
        fold_aucs.append(roc_auc_score(y[va], oof[va]))
        print(f"  fold {fold}: row-AUC = {fold_aucs[-1]:.5f}  "
              f"({len(uniq_ids):,} groups; mean group size {counts.mean():.2f})")

    auc_lr = float(roc_auc_score(y, oof))
    print(f"\nLambdaRank standalone OOF AUC: {auc_lr:.5f}")

    # Compare to K=10 plain LR-meta and check ρ
    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K10_FWD]
    F_K10 = expand(np.column_stack(base_oofs))
    F_K11 = expand(np.column_stack(base_oofs + [oof]))

    def fit_plain(F):
        out = np.zeros(len(y))
        for tr, va in splits:
            lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            out[va] = lr.predict_proba(F[va])[:, 1]
        return out, float(roc_auc_score(y, out))

    print("\nFitting K=10 plain LR-meta and K=10+lambdarank plain ...")
    oof_K10, auc_K10 = fit_plain(F_K10)
    oof_K11, auc_K11 = fit_plain(F_K11)
    delta_bp = (auc_K11 - auc_K10) * 1e4
    rho_lr_K10 = float(spearmanr(oof, oof_K10)[0])
    print(f"  K=10:           {auc_K10:.5f}")
    print(f"  K=10+λrank:     {auc_K11:.5f}  (Δ {delta_bp:+.3f} bp)")
    print(f"  ρ(λrank, K=10): {rho_lr_K10:.5f}")

    if delta_bp >= 0.5:
        verdict = "PASS — 4th-direction candidate"
    elif delta_bp >= -0.1:
        verdict = "AMBIGUOUS"
    else:
        verdict = "NULL"
    print(f"  Verdict: {verdict}")

    np.save(ART / "oof_exp2_lambdarank_per_stint_strat.npy", oof)
    out = {
        "lambdarank_oof_auc": auc_lr,
        "K10_plain_oof": auc_K10,
        "K10_plus_lambdarank_plain_oof": auc_K11,
        "delta_K10_plus_bp": float(delta_bp),
        "rho_lambdarank_vs_K10": rho_lr_K10,
        "fold_aucs": [float(x) for x in fold_aucs],
        "verdict": verdict,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_exp2_lambdarank.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote scripts/artifacts/probe_exp2_lambdarank.json. "
          f"Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
