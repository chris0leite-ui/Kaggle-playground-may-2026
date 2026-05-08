"""Move B / d6_two_base_recursive — 2-base [M5q, recursive] blend probe.

Critic-loop §4 Move B: HANDOVER framing — "K=2 OOF stack was −0.2bp
but rank structure is structurally different from K=15. Pre-submit-
diff vs M5q first; if ρ < 0.999, slot."

The K=15 LR-stack was null (ρ=0.99991 — Kaggle 5-decimal tie). The
2-base stack might preserve the recursive base's signal that the
K=15 LR meta washes out. Test 4 K=2 blend variants:

  V1: LR meta over (M5q, recursive) with expand (raw + rank + logit)
  V2: simple probability average      (no fit, no rank-lock)
  V3: rank-mean average               (pure rank space)
  V4: LGBM-shallow GBDT meta over (raw + rank)

For each: Strat OOF + ρ vs M5q test + predicted-gap classification.

Strat-only (R1).  Pre-submit-diff vs M5q on slot-worthy variants.
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
M5Q_S = 0.95057
M5Q_LB = 0.95005
SEED, N_FOLDS = 42, 5
RHO_TIE = 0.999


def load(name):
    oof = np.load(ART / f"oof_{name}_strat.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_strat.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def make_rank_features(P):
    n = len(P)
    return np.hstack([P, np.column_stack([rankdata(c) / n for c in P.T])
                      ]).astype(np.float32)


def fit_lr_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def fit_lgbm_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_p = np.zeros(len(F_test), dtype=np.float64)
    biters = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = lgb.LGBMClassifier(num_leaves=8, max_depth=3, learning_rate=0.05,
                               n_estimators=2000, min_child_samples=200,
                               reg_lambda=1.0, subsample=0.9,
                               colsample_bytree=0.9, random_state=SEED,
                               verbose=-1)
        m.fit(F_oof[tr], y[tr], eval_set=[(F_oof[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        meta_oof[va] = m.predict_proba(F_oof[va])[:, 1]
        test_p += m.predict_proba(F_test)[:, 1] / N_FOLDS
        biters.append(int(m.best_iteration_))
    return meta_oof, test_p, biters


def predicted_gap(rho_vs_m5q):
    """Predicted-gap heuristic per audit §5.1.

    Pool-divergence penalty calibrated on prior submits:
      ρ ≥ 0.999 → tie regime, predicted gap ≈ M5q gap = -5.2bp
      0.995 ≤ ρ < 0.999 → meta-switch regime, +(−1bp)
      0.99  ≤ ρ < 0.995 → moderate divergence,  +(−2 to −3bp)
      ρ < 0.99           → high divergence, recursive band, +(−4bp)
    """
    base = -5.2
    if rho_vs_m5q >= RHO_TIE:
        return base
    if rho_vs_m5q >= 0.995:
        return base - 1.0
    if rho_vs_m5q >= 0.99:
        return base - 2.5
    return base - 4.0


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    m5q_oof, m5q_test = load("m5q")
    rec_oof, rec_test = load("d5_recursive_m5q")

    # Sanity: anchors
    auc_m5q = float(roc_auc_score(y, m5q_oof))
    auc_rec = float(roc_auc_score(y, rec_oof))
    rho_oof, _ = spearmanr(m5q_oof, rec_oof)
    rho_test, _ = spearmanr(m5q_test, rec_test)
    print(f"Anchor: M5q OOF {auc_m5q:.5f}  LB {M5Q_LB:.5f}")
    print(f"        recursive OOF {auc_rec:.5f}")
    print(f"        ρ(OOF)  M5q vs rec = {rho_oof:.5f}")
    print(f"        ρ(test) M5q vs rec = {rho_test:.5f}\n")

    P_oof = np.column_stack([m5q_oof, rec_oof])
    P_test = np.column_stack([m5q_test, rec_test])

    results = {}

    # V1 — LR meta with expand
    t0 = time.time()
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, m5q_test)
    pg = predicted_gap(rho)
    pred_lb = M5Q_LB + (auc - M5Q_S) + (pg - (-5.2)) / 1e4
    print(f"[V1 LR-expand]    Strat {auc:.5f}  Δ M5q {(auc-M5Q_S)*1e4:+.2f}bp  "
          f"ρ {rho:.5f}  pred-gap {pg:.1f}bp  pred-LB {pred_lb:.5f}  "
          f"wall={time.time()-t0:.1f}s")
    print(f"  L1 weights: M5q raw={abs(coef[0]):.3f} rec raw={abs(coef[1]):.3f}  "
          f"M5q rk={abs(coef[2]):.3f} rec rk={abs(coef[3]):.3f}  "
          f"M5q lg={abs(coef[4]):.3f} rec lg={abs(coef[5]):.3f}")
    results["V1_lr_expand"] = save_variant("v1_lr_expand", mo, tp, auc, rho,
                                           pg, pred_lb, sample_sub)

    # V2 — simple probability average
    t0 = time.time()
    mo = (m5q_oof + rec_oof) / 2
    tp = (m5q_test + rec_test) / 2
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, m5q_test)
    pg = predicted_gap(rho)
    pred_lb = M5Q_LB + (auc - M5Q_S) + (pg - (-5.2)) / 1e4
    print(f"[V2 prob-avg]     Strat {auc:.5f}  Δ M5q {(auc-M5Q_S)*1e4:+.2f}bp  "
          f"ρ {rho:.5f}  pred-gap {pg:.1f}bp  pred-LB {pred_lb:.5f}  "
          f"wall={time.time()-t0:.1f}s")
    results["V2_prob_avg"] = save_variant("v2_prob_avg", mo, tp, auc, rho,
                                          pg, pred_lb, sample_sub)

    # V3 — rank-mean
    t0 = time.time()
    n_oof = len(m5q_oof); n_test = len(m5q_test)
    mo = (rankdata(m5q_oof) + rankdata(rec_oof)) / (2 * n_oof)
    tp = (rankdata(m5q_test) + rankdata(rec_test)) / (2 * n_test)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, m5q_test)
    pg = predicted_gap(rho)
    pred_lb = M5Q_LB + (auc - M5Q_S) + (pg - (-5.2)) / 1e4
    print(f"[V3 rank-avg]     Strat {auc:.5f}  Δ M5q {(auc-M5Q_S)*1e4:+.2f}bp  "
          f"ρ {rho:.5f}  pred-gap {pg:.1f}bp  pred-LB {pred_lb:.5f}  "
          f"wall={time.time()-t0:.1f}s")
    results["V3_rank_avg"] = save_variant("v3_rank_avg", mo, tp, auc, rho,
                                          pg, pred_lb, sample_sub)

    # V4 — LGBM shallow GBDT meta
    t0 = time.time()
    F_oof = make_rank_features(P_oof); F_test = make_rank_features(P_test)
    mo, tp, biters = fit_lgbm_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, m5q_test)
    pg = predicted_gap(rho)
    pred_lb = M5Q_LB + (auc - M5Q_S) + (pg - (-5.2)) / 1e4
    print(f"[V4 lgbm-shallow] Strat {auc:.5f}  Δ M5q {(auc-M5Q_S)*1e4:+.2f}bp  "
          f"ρ {rho:.5f}  pred-gap {pg:.1f}bp  pred-LB {pred_lb:.5f}  "
          f"wall={time.time()-t0:.1f}s  best_iters={biters}")
    results["V4_lgbm_shallow"] = save_variant("v4_lgbm_shallow", mo, tp, auc,
                                              rho, pg, pred_lb, sample_sub)

    # Verdict per variant
    print(f"\n=== Slot decision per audit §5.1 (predicted-gap gate) ===")
    print(f"{'variant':<20} {'OOF':>9} {'ρ vs M5q':>10} {'pred-LB':>10} "
          f"{'gate':<8} {'verdict':<30}")
    for label, r in results.items():
        # Slot-worthy iff: OOF >= M5q AND ρ < 0.999 AND pred_lb >= M5q_LB
        if r["spearman_vs_m5q"] >= RHO_TIE:
            v = "tie regime → wasted slot"
        elif r["strat_oof"] < M5Q_S - 0.5/1e4:
            v = "OOF regression"
        elif r["pred_lb"] < M5Q_LB:
            v = "pred-LB regression"
        elif r["pred_gap_bp"] < -7.0:
            v = "REQUIRES PI SIGN-OFF (gap<-7bp)"
        else:
            v = "SLOT-WORTHY ✓"
        print(f"{label:<20} {r['strat_oof']:>9.5f} "
              f"{r['spearman_vs_m5q']:>10.5f} {r['pred_lb']:>10.5f} "
              f"[{r['pred_gap_bp']:>+5.1f}]  {v}")

    (ART / "d6_two_base_recursive_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n→ scripts/artifacts/d6_two_base_recursive_results.json")


def save_variant(slug, mo, tp, auc, rho, pg, pred_lb, sample_sub):
    np.save(ART / f"oof_d6_2base_{slug}_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / f"test_d6_2base_{slug}_strat.npy",
            np.column_stack([1 - tp, tp]))
    sub = sample_sub.copy(); sub[TARGET] = tp
    sub.to_csv(f"submissions/submission_d6_2base_{slug}.csv", index=False)
    return dict(strat_oof=auc, delta_m5q_bp=(auc - M5Q_S) * 1e4,
                spearman_vs_m5q=float(rho), pred_gap_bp=pg, pred_lb=pred_lb)


if __name__ == "__main__":
    main()
