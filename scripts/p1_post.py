"""P1 post-run helper — gate vs PRIMARY + ensemble preview + pre-submit diff."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
PRIMARY_NAME = "d15b_path_b_K22_dae_only_tau20000"
PRIMARY_LB = 0.95059


def load_pred(path):
    a = np.load(path)
    return a[:, 1] if (a.ndim == 2 and a.shape[1] == 2) else a.ravel()


def expand(P):
    from scipy.stats import rankdata
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    lo = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, lo])


def main(name="p1_single_lgbm_feA_te"):
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    p_oof = load_pred(ART / f"oof_{PRIMARY_NAME}_strat.npy")
    p_test = load_pred(ART / f"test_{PRIMARY_NAME}_strat.npy")
    c_oof = load_pred(ART / f"oof_{name}_strat.npy")
    c_test = load_pred(ART / f"test_{name}_strat.npy")

    auc_p = float(roc_auc_score(y, p_oof))
    auc_c = float(roc_auc_score(y, c_oof))
    rho = float(spearmanr(c_test, p_test)[0])
    d_oof = (auc_c - auc_p) * 1e4

    # G3 flips
    thr = float(np.quantile(p_test, 0.99))
    pp = p_test >= thr; cp = c_test >= thr
    f_neg = int(np.sum(pp & ~cp)); f_pos = int(np.sum(~pp & cp))
    flip = (min(f_pos, f_neg) / max(f_pos, f_neg)
            if max(f_pos, f_neg) > 0 else 1.0)

    # Predicted LB band
    if rho >= 0.99996: pred = d_oof
    elif rho >= 0.999: pred = d_oof - 0.5
    elif rho >= 0.995: pred = d_oof - 1.5
    elif rho >= 0.99: pred = d_oof - 3.0
    else: pred = d_oof - 5.0

    print(f"\n=== {name} vs PRIMARY ({PRIMARY_NAME}) ===")
    print(f"  PRIMARY OOF: {auc_p:.5f}  PRIMARY LB: {PRIMARY_LB}")
    print(f"  CAND OOF:    {auc_c:.5f}  Δ = {d_oof:+.2f}bp")
    print(f"  ρ:           {rho:.6f}")
    print(f"  predicted LB Δ band: {pred:+.2f}bp  (LB ~{PRIMARY_LB + pred*1e-4:.5f})")
    print(f"  G3 flips: +→− {f_neg}, −→+ {f_pos}, ratio {flip:.3f}")
    print(f"  R7-eligible (>200 flips): {'YES' if max(f_pos, f_neg) > 200 else 'no'}")

    # K=2 LR-meta gate (cand + PRIMARY meta-derivative)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))
    P_oof_with = np.column_stack([p_oof, c_oof])
    F_oof = expand(P_oof_with)
    oof_meta = np.zeros(len(y))
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        oof_meta[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc_meta = float(roc_auc_score(y, oof_meta))
    print(f"\n  K=2 LR-meta(PRIMARY,cand) OOF: {auc_meta:.5f}  Δ vs PRIMARY {(auc_meta-auc_p)*1e4:+.2f}bp")

    # Decision rule
    print(f"\n  --- DECISION ---")
    if d_oof > 30 and rho < 0.999:
        print(f"  ✅ Strong standalone candidate (Δ>+30bp, ρ<0.999) — pursue submit (PI sign-off).")
    elif d_oof > 5 and rho < 0.995:
        print(f"  ✓ Diverse single model (Δ>+5bp, ρ<0.995) — submit candidate or stack-add probe.")
    elif d_oof > 0:
        print(f"  ~ Marginal — consider stack-add probe rather than direct submit.")
    else:
        print(f"  ✗ Standalone OOF below PRIMARY — pivot.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "p1_single_lgbm_feA_te")
