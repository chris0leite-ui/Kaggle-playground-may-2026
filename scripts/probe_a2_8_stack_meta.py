"""scripts/probe_a2_8_stack_meta.py — A2-8: LightGBM stack-meta on K=4.

Per hypothesis-board scope amendment, the genuinely untested mechanism
is meta-level interactions between base predictions, not new raw FE.
EXP-NEW falsification covered linear / Path-B partial-pooling stackers
but never a non-linear stacker with pairwise prediction products /
abs-diffs / raw FE side info.

Design:
- Bases: K=4 forward-greedy
  (d17_h1d_yekenot_full, p1_single_cb_v4_gpu, f1_hgbc_deep,
   d16_orig_continuous_only).
- Meta features (per row):
  - 4 raw P
  - 4 logits, 4 ranks (matching `expand()` from PRIMARY)
  - 6 pairwise products P_i * P_j
  - 6 abs-diffs |P_i - P_j|
  - 6 logit diffs (logit_i - logit_j)
  - Raw FE side info: Stint, LapsToGo (RaceProgress * 100 proxy),
    Compound (one-hot), Year, TyreLife, Position, LapNumber,
    Cumulative_Degradation, LapTime_Delta.
- Meta learner: LightGBM AUC, light hyper (depth 4, leaves 31, lr 0.05,
  500 iters max with early stop on val fold).
- 5-fold StratifiedKFold seed=42 (same splits as PRIMARY).

Reference:
- K=4 plain LR-meta OOF: 0.95399
- K=4 Path-B C×S τ=100k OOF: 0.95403 (current PRIMARY)
- PASS = +0.5 bp vs Path-B → ≥0.95408
- WEAK = +0.1–0.5 bp band

Cost ~5–8 min CPU.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K4 = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]
SIDE_NUM = [
    "Stint", "TyreLife", "Position", "LapNumber",
    "Cumulative_Degradation", "LapTime_Delta", "RaceProgress", "Year",
]
SIDE_CAT = ["Compound"]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def make_meta_features(P: np.ndarray, side: pd.DataFrame) -> pd.DataFrame:
    """Build meta-feature matrix from K base preds + raw side info.

    P: (n, K) base probabilities. side: DataFrame of raw cols (numeric +
    one-hot Compound).
    """
    n, K = P.shape
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logits = np.log(Pc / (1 - Pc))
    ranks = np.column_stack([rankdata(c) / n for c in P.T])

    cols: dict[str, np.ndarray] = {}
    for k in range(K):
        cols[f"p{k}"] = P[:, k]
        cols[f"r{k}"] = ranks[:, k]
        cols[f"l{k}"] = logits[:, k]
    for i, j in combinations(range(K), 2):
        cols[f"prod_{i}_{j}"] = P[:, i] * P[:, j]
        cols[f"absdiff_{i}_{j}"] = np.abs(P[:, i] - P[:, j])
        cols[f"logitdiff_{i}_{j}"] = logits[:, i] - logits[:, j]

    base_df = pd.DataFrame(cols)
    return pd.concat([base_df.reset_index(drop=True),
                      side.reset_index(drop=True)], axis=1)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    print(f"  rows: {len(y):,}")

    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4]
    P_train = np.column_stack(base_oofs)

    # Raw FE side info (numeric + one-hot Compound).
    side_num = train[SIDE_NUM].astype(float).fillna(0.0).reset_index(drop=True)
    cmp_oh = pd.get_dummies(train["Compound"].astype(str),
                             prefix="cmp", dtype=float).reset_index(drop=True)
    side_train = pd.concat([side_num, cmp_oh], axis=1)
    print(f"  side info: {side_train.shape[1]} cols "
          f"({len(SIDE_NUM)} num + {cmp_oh.shape[1]} compound oh)")

    F_train = make_meta_features(P_train, side_train)
    print(f"  meta features: {F_train.shape[1]} cols ({F_train.shape[0]:,} rows)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y))
    fold_aucs = []
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=4,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=SEED,
        num_threads=4,
    )

    print("\n5-fold LightGBM stack-meta (depth 4, leaves 31, lr 0.05) ...")
    for fi, (tr, va) in enumerate(splits, 1):
        t1 = time.time()
        dtr = lgb.Dataset(F_train.iloc[tr].values, label=y[tr],
                          feature_name=list(F_train.columns))
        dva = lgb.Dataset(F_train.iloc[va].values, label=y[va],
                          feature_name=list(F_train.columns), reference=dtr)
        m = lgb.train(
            params, dtr, num_boost_round=800,
            valid_sets=[dva], valid_names=["va"],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict(F_train.iloc[va].values,
                            num_iteration=m.best_iteration)
        a = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(a)
        print(f"    Fold {fi}: AUC={a:.5f}  iters={m.best_iteration}  "
              f"wall={time.time()-t1:.1f}s")

    auc_a2_8 = float(roc_auc_score(y, oof))
    print(f"\n  A2-8 stack-meta OOF: {auc_a2_8:.5f}  fold-std={np.std(fold_aucs):.5f}")

    # References (from probe_a2_2_pathb_K4.json).
    ref_path = ART / "probe_a2_2_pathb_K4.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        auc_K4_plain = ref["K4_plain_oof"]
        auc_K4_pb = ref["K4_pathb_oof"]
    else:
        auc_K4_plain = 0.95399
        auc_K4_pb = 0.95403
    delta_vs_plain = (auc_a2_8 - auc_K4_plain) * 1e4
    delta_vs_pb = (auc_a2_8 - auc_K4_pb) * 1e4
    print(f"\n  vs K=4 plain LR-meta ({auc_K4_plain:.5f}):  {delta_vs_plain:+.2f} bp")
    print(f"  vs K=4 Path-B PRIMARY ({auc_K4_pb:.5f}):    {delta_vs_pb:+.2f} bp")

    if delta_vs_pb >= 0.5:
        verdict = "PASS"
    elif delta_vs_pb >= 0.1:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    print(f"\n  A2-8 verdict (≥0.5 bp = PASS): {verdict}")

    # Correlation with PRIMARY (Path-B OOF if we have it cached).
    pathb_oof_path = ART / "oof_K4_pathb_primary.npy"
    if pathb_oof_path.exists():
        pb_oof = _pos(pathb_oof_path)
        rho = float(spearmanr(oof, pb_oof).statistic)
        print(f"  ρ(A2-8 OOF, K=4 Path-B OOF): {rho:.6f}")
    else:
        rho = None

    # Save OOF for downstream stacking experiments.
    np.save(ART / "oof_a2_8_stack_meta_strat.npy",
            np.column_stack([1 - oof, oof]))
    print(f"  saved → oof_a2_8_stack_meta_strat.npy")

    summary = dict(
        candidate="a2_8_stack_meta",
        K4_bases=K4,
        n_meta_features=int(F_train.shape[1]),
        oof_auc=auc_a2_8,
        fold_aucs=fold_aucs,
        K4_plain_oof_ref=auc_K4_plain,
        K4_pathb_oof_ref=auc_K4_pb,
        delta_vs_plain_bp=float(delta_vs_plain),
        delta_vs_pathb_bp=float(delta_vs_pb),
        rho_vs_pathb_primary=rho,
        verdict=verdict,
        wall_s=time.time() - t0,
    )
    out = ART / "probe_a2_8_stack_meta.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  → {out}")
    print(f"  total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
