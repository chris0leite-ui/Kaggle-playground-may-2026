"""T1.5 — Deotte L2 stacking with std/mean/range meta-features.

Per audit/2026-05-08-strategic-menu-wider-steps.md Tier-1 #5.

Mechanism: LR-meta over [raw, rank, logit] of 18 bases is rank-locked.
The std-across-bases per row is OUT of LR's hypothesis class — adding
it strictly enlarges the meta. Approach:

  L2a (LR baseline)   = current K=18 LR-meta (anchor; should match
                        d6_k18_multi_rule).
  L2b (LGBM-L2)       = LightGBM(depth=3) on [raw, rank, logit, std,
                        mean, min, max, range, p25, p75].
  L2c (Ridge-L2)      = Ridge on same expanded feature set.
  L3                  = Ridge-weighted average of {L2a, L2b, L2c} OOF.

Decision rule per audit §5:
  - L2b OOF >= L2a + 0.5bp AND ρ(L2b, L2a) < 0.9995 -> SLOT-WORTHY
  - L3 OOF >= L2a + 0.5bp -> SLOT-WORTHY
  - Minimal-meta gate: each of L2b/L2c against PRIMARY d6_k18_multi_rule
    via 2-comp LR — must clear PRIMARY OOF.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
PRIMARY_S = 0.95065  # d6_k18_multi_rule Strat OOF
PRIMARY_LB = 0.95026
M5Q_S = 0.95057
RHO_TIE = 0.9995

POOL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"),
    ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    ("realmlp", "realmlp"),
    ("rule_compound_tyre", "d6_rule_residual"),
    ("rule_compound_stint", "d6_rule_compound_stint"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]


def load_pool():
    print(f"Loading {len(POOL)} bases (K=18) ...")
    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    return np.column_stack(Xs_oof), np.column_stack(Xs_test), names


def expand_logit_rank(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    p = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(p / (1 - p))
    return np.hstack([P, rk, logit])


def disagreement_features(P):
    """std, mean, min, max, range, p25, p75 across bases per row.
    Returns (n_rows, 7)."""
    return np.column_stack([
        P.std(axis=1),
        P.mean(axis=1),
        P.min(axis=1),
        P.max(axis=1),
        P.max(axis=1) - P.min(axis=1),
        np.quantile(P, 0.25, axis=1),
        np.quantile(P, 0.75, axis=1),
    ])


def fit_lr_meta(F_oof, F_test, y, C=1.0):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1]


def fit_ridge_meta(F_oof, F_test, y, alpha=1.0):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = Ridge(alpha=alpha)
        m.fit(F_oof[tr], y[tr].astype(np.float64))
        meta_oof[va] = m.predict(F_oof[va])
    m_full = Ridge(alpha=alpha)
    m_full.fit(F_oof, y.astype(np.float64))
    return meta_oof, m_full.predict(F_test)


def fit_lgbm_meta(F_oof, F_test, y, depth=3, num_leaves=8):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(F_test), dtype=np.float64)
    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.02, num_leaves=num_leaves, max_depth=depth,
        min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=5, lambda_l2=1.0, verbose=-1, seed=SEED,
    )
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        d_tr = lgb.Dataset(F_oof[tr], label=y[tr])
        d_va = lgb.Dataset(F_oof[va], label=y[va])
        m = lgb.train(params, d_tr, num_boost_round=2000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        meta_oof[va] = m.predict(F_oof[va])
        test_pred += m.predict(F_test) / N_FOLDS
    return meta_oof, test_pred


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    # Load reference: PRIMARY = d6_k18_multi_rule
    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy")[:, 1]
    primary_oof = np.load(ART / "oof_d6_k18_multi_rule_strat.npy")[:, 1]
    primary_oof_auc = float(roc_auc_score(y, primary_oof))
    print(f"PRIMARY d6_k18_multi_rule OOF: {primary_oof_auc:.5f}  "
          f"(stored {PRIMARY_S})")

    # Load 18-base pool
    P_oof, P_test, names = load_pool()
    print(f"Pool shape: oof {P_oof.shape}, test {P_test.shape}")

    # Disagreement aux features
    disag_oof = disagreement_features(P_oof)
    disag_test = disagreement_features(P_test)
    print(f"Disagreement aux: shape {disag_oof.shape}; cols=[std,mean,min,max,range,p25,p75]")
    print(f"  std mean: {disag_oof[:,0].mean():.4f}; range mean: {disag_oof[:,4].mean():.4f}")

    # Expanded base feature set: [raw, rank, logit] like d6_k18
    F_oof_base = expand_logit_rank(P_oof)   # (N, 54)
    F_test_base = expand_logit_rank(P_test) # (N_test, 54)

    # Augment with disagreement: [raw, rank, logit, std, mean, min, max, range, p25, p75]
    F_oof_aug = np.hstack([F_oof_base, disag_oof])    # (N, 61)
    F_test_aug = np.hstack([F_test_base, disag_test]) # (N_test, 61)
    print(f"Aug shape: F_oof {F_oof_aug.shape}, F_test {F_test_aug.shape}")

    results = {}

    # L2a: LR baseline (replicate K=18 LR-meta) — sanity check
    print(f"\n--- L2a: LR baseline on [raw, rank, logit] (K=18 anchor) ---")
    t = time.time()
    lr_oof, lr_test = fit_lr_meta(F_oof_base, F_test_base, y)
    auc_lr = float(roc_auc_score(y, lr_oof))
    rho_lr_primary, _ = spearmanr(lr_test, primary_test)
    print(f"  LR baseline OOF: {auc_lr:.5f}  (vs PRIMARY {primary_oof_auc:.5f})")
    print(f"  ρ vs PRIMARY test: {rho_lr_primary:.5f}  wall={time.time()-t:.1f}s")
    results["lr_baseline"] = dict(oof=auc_lr, rho_vs_primary=float(rho_lr_primary))

    # L2b: LGBM on aug input
    print(f"\n--- L2b: LGBM-L2 (depth=3) on [raw, rank, logit + 7 aux] ---")
    t = time.time()
    lgbm_oof, lgbm_test = fit_lgbm_meta(F_oof_aug, F_test_aug, y)
    auc_lgbm = float(roc_auc_score(y, lgbm_oof))
    rho_lgbm_primary, _ = spearmanr(lgbm_test, primary_test)
    rho_lgbm_lr, _ = spearmanr(lgbm_test, lr_test)
    print(f"  LGBM-L2 OOF: {auc_lgbm:.5f}  Δ PRIMARY {(auc_lgbm-primary_oof_auc)*1e4:+.2f}bp")
    print(f"  ρ vs PRIMARY: {rho_lgbm_primary:.5f}  ρ vs LR-base: {rho_lgbm_lr:.5f}")
    print(f"  wall={time.time()-t:.1f}s")
    results["lgbm_l2"] = dict(
        oof=auc_lgbm, delta_primary_bp=(auc_lgbm-primary_oof_auc)*1e4,
        rho_vs_primary=float(rho_lgbm_primary), rho_vs_lr_base=float(rho_lgbm_lr),
    )

    # L2c: Ridge on aug input
    print(f"\n--- L2c: Ridge-L2 on [raw, rank, logit + 7 aux] ---")
    t = time.time()
    ridge_oof, ridge_test = fit_ridge_meta(F_oof_aug, F_test_aug, y, alpha=1.0)
    # Ridge regression on 0/1 → linear; need to AUC-rank
    auc_ridge = float(roc_auc_score(y, ridge_oof))
    rho_ridge_primary, _ = spearmanr(ridge_test, primary_test)
    rho_ridge_lr, _ = spearmanr(ridge_test, lr_test)
    print(f"  Ridge-L2 OOF: {auc_ridge:.5f}  Δ PRIMARY {(auc_ridge-primary_oof_auc)*1e4:+.2f}bp")
    print(f"  ρ vs PRIMARY: {rho_ridge_primary:.5f}  ρ vs LR-base: {rho_ridge_lr:.5f}")
    print(f"  wall={time.time()-t:.1f}s")
    results["ridge_l2"] = dict(
        oof=auc_ridge, delta_primary_bp=(auc_ridge-primary_oof_auc)*1e4,
        rho_vs_primary=float(rho_ridge_primary), rho_vs_lr_base=float(rho_ridge_lr),
    )

    # L3: Ridge-weighted average of {LR-baseline, LGBM-L2, Ridge-L2}
    # Fit Ridge on stacked OOF (3-col) → y to find blend weights
    print(f"\n--- L3: Ridge-weighted avg of {{LR, LGBM-L2, Ridge-L2}} ---")
    t = time.time()
    F3_oof = np.column_stack([lr_oof, lgbm_oof, ridge_oof])
    F3_test = np.column_stack([lr_test, lgbm_test, ridge_test])
    l3_oof, l3_test = fit_ridge_meta(F3_oof, F3_test, y, alpha=0.1)
    auc_l3 = float(roc_auc_score(y, l3_oof))
    rho_l3_primary, _ = spearmanr(l3_test, primary_test)
    print(f"  L3 OOF: {auc_l3:.5f}  Δ PRIMARY {(auc_l3-primary_oof_auc)*1e4:+.2f}bp")
    print(f"  ρ vs PRIMARY test: {rho_l3_primary:.5f}  wall={time.time()-t:.1f}s")
    # Also dump the L3 ridge weights — use a separate fit to inspect
    m3 = Ridge(alpha=0.1)
    m3.fit(F3_oof, y.astype(np.float64))
    print(f"  L3 weights: LR={m3.coef_[0]:.4f} LGBM={m3.coef_[1]:.4f} Ridge={m3.coef_[2]:.4f}")
    results["l3_blend"] = dict(
        oof=auc_l3, delta_primary_bp=(auc_l3-primary_oof_auc)*1e4,
        rho_vs_primary=float(rho_l3_primary),
        weights=dict(lr=float(m3.coef_[0]), lgbm=float(m3.coef_[1]), ridge=float(m3.coef_[2])),
    )

    # Verdict gates
    def gate(label, auc, rho_primary):
        passes_oof = auc >= primary_oof_auc + 0.5/1e4
        passes_rho = rho_primary < RHO_TIE
        slot = passes_oof and passes_rho
        margin_oof = (auc - primary_oof_auc) * 1e4
        return dict(
            label=label, oof_pass=bool(passes_oof), rho_pass=bool(passes_rho),
            slot_worthy=bool(slot), margin_oof_bp=margin_oof,
            rho_vs_primary=float(rho_primary),
        )

    print(f"\n=== Verdict ===")
    for label, oof, rho_p in [
        ("L2a-LR", auc_lr, rho_lr_primary),
        ("L2b-LGBM", auc_lgbm, rho_lgbm_primary),
        ("L2c-Ridge", auc_ridge, rho_ridge_primary),
        ("L3-blend", auc_l3, rho_l3_primary),
    ]:
        g = gate(label, oof, rho_p)
        marker = "  → SLOT" if g["slot_worthy"] else "  → tie/regress"
        print(f"  {label:12s} OOF {oof:.5f} (Δ {g['margin_oof_bp']:+.2f}bp) "
              f"ρ {g['rho_vs_primary']:.5f}{marker}")
        results.setdefault("gates", {})[label] = g

    # Save artifacts
    for name, oof, test_arr in [
        ("d8_lgbm_l2_aug", lgbm_oof, lgbm_test),
        ("d8_ridge_l2_aug", ridge_oof, ridge_test),
        ("d8_l3_blend", l3_oof, l3_test),
    ]:
        np.save(ART / f"oof_{name}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_{name}_strat.npy",
                np.column_stack([1 - test_arr, test_arr]))

    # Save submission for the candidate that passes (or best of)
    candidates = [
        ("d8_lgbm_l2_aug", auc_lgbm, lgbm_test, rho_lgbm_primary),
        ("d8_ridge_l2_aug", auc_ridge, ridge_test, rho_ridge_primary),
        ("d8_l3_blend", auc_l3, l3_test, rho_l3_primary),
    ]
    best = max(candidates, key=lambda x: x[1])
    if best[1] > primary_oof_auc and best[3] < RHO_TIE:
        sub = sample_sub.copy(); sub[TARGET] = best[2]
        sub.to_csv(f"submissions/submission_{best[0]}.csv", index=False)
        print(f"\n→ candidate submission written: submission_{best[0]}.csv "
              f"(OOF {best[1]:.5f}, ρ {best[3]:.5f})")
    else:
        print(f"\n→ no candidate clears PRIMARY+0.5bp gate; no sub written")

    results["primary_oof"] = primary_oof_auc
    results["primary_lb"] = PRIMARY_LB
    results["wall_total_s"] = time.time() - t0
    (ART / "d8_l2_stacking_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n→ scripts/artifacts/d8_l2_stacking_results.json (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
