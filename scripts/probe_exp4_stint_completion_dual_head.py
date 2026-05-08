"""scripts/probe_exp4_stint_completion_dual_head.py — EXP-4.

Stint-completion dual-head decomposition. The synth's temporal
downsampling means PitNextLap[N] = 1 iff:
  (A) row N is the LAST observed row of its stint AND
  (B) the actual next pit happens at lap N+1 specifically (not later)

Decompose into two heads on combined-frame data (Rule 25 safe at row
level per A3):

  Head A: P(stint changes between this row's next OBSERVED row in same
          (Race, Driver, Year)). Target = (Stint of next observed row
          != Stint of current row), or no successor (last row of group).

  Head B: P(pit was on lap N+1 specifically | Head A=1). Target =
          PitNextLap (constrained to rows where target_A=1).

Combined prediction: P(PitNextLap=1) = P(A=1) * P(B=1 | A=1).

The decomposition is structurally different from binary AUC on the
14 raw columns: Head A learns the *downsampling structure*, Head B
learns the *decision shift*. Each may produce predictions in a new
direction.

Cost: ~30 min CPU.
Outputs scripts/artifacts/probe_exp4_dual_head.json.
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

LGB = dict(
    objective="binary", metric="auc", learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=80, feature_fraction=0.9,
    bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
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


def build_target_A(train: pd.DataFrame,
                   test: pd.DataFrame) -> np.ndarray:
    """target_A = (Stint changes between this row and next observed row
    in same (Race, Driver, Year)). Computed combined-frame; only the
    train half is used for labels."""
    n_tr = len(train)
    df = pd.concat([train.assign(_split="tr"),
                    test.assign(_split="te")],
                   ignore_index=True, sort=False)
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["Race", "Driver", "Year", "LapNumber"],
                        kind="stable").reset_index(drop=True)
    g = df.groupby(["Race", "Driver", "Year"], sort=False)
    df["next_stint"] = g["Stint"].shift(-1)
    df["next_lap"] = g["LapNumber"].shift(-1)
    df["target_A"] = (df["next_stint"].fillna(df["Stint"]) !=
                       df["Stint"]).astype(int)
    # Last row of each group has no successor — treat as 0 (don't know)
    df.loc[df["next_lap"].isna(), "target_A"] = 0
    df = df.sort_values("row_id").reset_index(drop=True)
    return df["target_A"].values[:n_tr]


def main():
    t0 = time.time()
    print("Loading + building target_A (stint-changes-by-next-obs) ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    target_A = build_target_A(train, test)
    print(f"  target_A pos rate: {target_A.mean():.4f}  "
          f"(target_PitNextLap pos rate: {y.mean():.4f})")
    # When target_A=1, what fraction of target_PitNextLap=1?
    p_y_given_A = float(y[target_A == 1].mean()) if (target_A == 1).any() else 0.0
    p_y_given_notA = float(y[target_A == 0].mean()) if (target_A == 0).any() else 0.0
    print(f"  P(PitNextLap=1 | target_A=1):  {p_y_given_A:.4f}  "
          f"(n={int((target_A==1).sum()):,})")
    print(f"  P(PitNextLap=1 | target_A=0):  {p_y_given_notA:.4f}  "
          f"(n={int((target_A==0).sum()):,})")

    train_enc = encode_cats(train)
    feats = NUM_COLS + [c + "_cat" for c in CAT_COLS]
    X = train_enc[feats].astype(float).values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y))

    # ------ Head A: predict target_A on full data ------
    print("\nHead A: target_A (stint-changes-by-next-obs) ...")
    oof_A = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        ds_tr = lgb.Dataset(X[tr], label=target_A[tr])
        ds_va = lgb.Dataset(X[va], label=target_A[va], reference=ds_tr)
        booster = lgb.train(
            LGB, ds_tr, num_boost_round=500, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        oof_A[va] = booster.predict(X[va])
    auc_A = float(roc_auc_score(target_A, oof_A))
    print(f"  Head A standalone OOF AUC (vs target_A): {auc_A:.5f}")
    # Head A as a feature for predicting PitNextLap
    auc_A_for_pit = float(roc_auc_score(y, oof_A))
    print(f"  Head A predictions, evaluated against PitNextLap: "
          f"{auc_A_for_pit:.5f}")

    # ------ Head B: predict PitNextLap on rows where target_A=1 ------
    # Need to be careful: we train Head B only on tr rows with target_A=1,
    # but evaluate on va rows. For va rows where target_A=0, P(PitNextLap)
    # is approximated by the (small) baseline P(y=1 | target_A=0).
    print("\nHead B: PitNextLap on rows where target_A=1 ...")
    oof_B = np.full(len(y), p_y_given_notA)
    for fold, (tr, va) in enumerate(splits):
        tr_A = tr[target_A[tr] == 1]
        if len(tr_A) < 100:
            print(f"  fold {fold}: too few target_A=1 rows ({len(tr_A)})")
            continue
        ds_tr = lgb.Dataset(X[tr_A], label=y[tr_A])
        booster = lgb.train(
            LGB, ds_tr, num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)])
        # Predict only on va rows where target_A could plausibly be 1
        oof_B[va] = booster.predict(X[va])
    # Compose
    composed = oof_A * oof_B + (1 - oof_A) * p_y_given_notA
    auc_composed = float(roc_auc_score(y, composed))
    print(f"  Composed P(A)*P(B|A) + (1-P(A))*p0 OOF AUC: "
          f"{auc_composed:.5f}")

    # K=10+1 plain gate using composed prediction
    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K10_FWD]
    F_K10 = expand(np.column_stack(base_oofs))
    F_K11 = expand(np.column_stack(base_oofs + [composed]))

    def fit_plain(F):
        out = np.zeros(len(y))
        for tr, va in splits:
            lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            out[va] = lr.predict_proba(F[va])[:, 1]
        return float(roc_auc_score(y, out))

    auc_K10 = fit_plain(F_K10)
    auc_K11 = fit_plain(F_K11)
    delta_bp = (auc_K11 - auc_K10) * 1e4

    # Also try Head A alone as a meta-add
    F_K10_plus_A = expand(np.column_stack(base_oofs + [oof_A]))
    auc_K11_A = fit_plain(F_K10_plus_A)
    delta_A_bp = (auc_K11_A - auc_K10) * 1e4

    rho_composed = float(spearmanr(composed, oof_A)[0])
    rho_A_K10 = float(spearmanr(oof_A, np.column_stack(base_oofs).mean(axis=1))[0])
    print(f"\nK=10 plain:                  {auc_K10:.5f}")
    print(f"K=10 + composed dual-head:   {auc_K11:.5f}  (Δ {delta_bp:+.3f} bp)")
    print(f"K=10 + Head A alone:         {auc_K11_A:.5f}  (Δ {delta_A_bp:+.3f} bp)")
    print(f"ρ(Head A, K=10 mean):        {rho_A_K10:.5f}")

    best_delta = max(delta_bp, delta_A_bp)
    if best_delta >= 0.5:
        verdict = "PASS — 4th-direction candidate"
    elif best_delta >= -0.1:
        verdict = "AMBIGUOUS"
    else:
        verdict = "NULL"
    print(f"Verdict: {verdict}")

    np.save(ART / "oof_exp4_dual_head_composed_strat.npy", composed)
    np.save(ART / "oof_exp4_head_A_strat.npy", oof_A)

    out = {
        "head_A_target_pos_rate": float(target_A.mean()),
        "P_y_given_A1": p_y_given_A,
        "P_y_given_A0": p_y_given_notA,
        "head_A_auc_vs_target_A": auc_A,
        "head_A_auc_vs_PitNextLap": auc_A_for_pit,
        "composed_oof_auc": auc_composed,
        "K10_plain_oof": auc_K10,
        "K10_plus_composed_oof": auc_K11,
        "delta_K10_plus_composed_bp": float(delta_bp),
        "K10_plus_head_A_oof": auc_K11_A,
        "delta_K10_plus_head_A_bp": float(delta_A_bp),
        "rho_head_A_vs_K10_mean": rho_A_K10,
        "verdict": verdict,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_exp4_dual_head.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote scripts/artifacts/probe_exp4_dual_head.json. "
          f"Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
