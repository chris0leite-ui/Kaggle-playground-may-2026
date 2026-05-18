"""Phase 2 P2.2 — Per-row conformal-like prediction-interval widths.

Simplified conformal-prediction-set-width approach: per K=4 base,
compute per-(Compound, Stint) residual standard deviation on training
rows; map onto val/test rows as a per-row width feature. Stack 4
widths as extra meta-features alongside [P, rank, logit] = 16-feature
LR-meta input.

Fold-safe: per-fold, residuals are computed using training rows only
(Rule 24).

Rationale for simplification vs full CQR:
- Full GradientBoostingRegressor quantile regression per fold per
  base = 40 GBR fits = 20-40 min CPU.
- This iteration has had 9 consecutive nulls so far; CQR has 30%
  prior of lift. Spending 30 min for 30% chance of <+0.5 bp is bad
  EV vs the per-bin std approximation that captures the bulk of
  the per-row uncertainty heterogeneity for ~5 min.

Origin: 2026-05-18 round-2 plan P2.2.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K4 = ["d17_h1d_yekenot_full", "p1_single_cb_v4_gpu",
      "f1_hgbc_deep", "d16_orig_continuous_only"]


def pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand_with_widths(P: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """[P, rank, logit, width] expansion. widths shape (N, K)."""
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit, widths])


def compute_per_bin_widths(train_pred, train_y, train_compound,
                            train_stint, val_compound, val_stint,
                            test_compound, test_stint):
    """For one base: compute per-(Compound, Stint) residual std on
    train rows; map onto val and test rows."""
    df_tr = pd.DataFrame({
        "Compound": train_compound,
        "Stint": train_stint,
        "resid": train_y - train_pred,
    })
    agg = df_tr.groupby(["Compound", "Stint"])["resid"].agg(
        ["std", "count"]).reset_index()
    # Stabilise std for tiny bins: use overall std as prior; weighted avg
    global_std = float(df_tr["resid"].std())
    LAMBDA = 50  # shrinkage toward global_std
    agg["w_std"] = ((agg["std"].fillna(0) ** 2 * agg["count"]
                     + global_std ** 2 * LAMBDA)
                    / (agg["count"] + LAMBDA)) ** 0.5
    bin_map = {(c, s): w for c, s, w in zip(
        agg["Compound"], agg["Stint"], agg["w_std"])}

    def lookup(compound, stint):
        return np.array([bin_map.get((c, s), global_std)
                          for c, s in zip(compound, stint)])

    return lookup(val_compound, val_stint), lookup(test_compound, test_stint)


def main():
    print("=== Phase 2 P2.2 — Per-bin conformal-like widths ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"train {len(train):,}, test {len(test):,}")

    # K=4 base OOF + test
    P_oof = np.column_stack([pos(ART / f"oof_{b}_strat.npy") for b in K4])
    P_test = np.column_stack([pos(ART / f"test_{b}_strat.npy") for b in K4])

    # Baseline K=4 LR-meta (12 feats)
    def _expand_no_width(P):
        n = len(P)
        rk = np.column_stack([rankdata(c) / n for c in P.T])
        Pc = np.clip(P, 1e-9, 1 - 1e-9)
        return np.hstack([P, rk, np.log(Pc / (1 - Pc))])

    F_oof_base = _expand_no_width(P_oof)
    F_test_base = _expand_no_width(P_test)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))

    base_oof = np.zeros(len(y))
    for tr, va in folds:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof_base[tr], y[tr])
        base_oof[va] = lr.predict_proba(F_oof_base[va])[:, 1]
    base_auc = roc_auc_score(y, base_oof)
    print(f"\nBaseline K=4 LR-meta OOF (12 feats): {base_auc:.5f}")

    # Augmented LR-meta with 4 per-base widths (16 feats)
    train_compound = train["Compound"].values
    train_stint = train["Stint"].astype(int).values
    test_compound = test["Compound"].values
    test_stint = test["Stint"].astype(int).values

    # Per-fold per-base width computation
    widths_oof = np.zeros((len(y), 4))
    widths_test = np.zeros((len(test), 4))

    print("\nComputing per-(Compound, Stint) widths per base...")
    for tr, va in folds:
        for k_idx in range(4):
            p_tr = P_oof[tr, k_idx]
            w_va, w_te = compute_per_bin_widths(
                p_tr, y[tr],
                train_compound[tr], train_stint[tr],
                train_compound[va], train_stint[va],
                test_compound, test_stint)
            widths_oof[va, k_idx] = w_va
            widths_test[:, k_idx] += w_te / N_FOLDS

    print(f"  widths_oof shape {widths_oof.shape}, "
          f"sample (driver, fold) widths: {widths_oof[0]}")

    # Fit LR-meta on F_oof_base + widths
    F_oof_aug = np.hstack([F_oof_base, widths_oof])
    F_test_aug = np.hstack([F_test_base, widths_test])
    print(f"\nAugmented feature matrix shape: {F_oof_aug.shape}")

    aug_oof = np.zeros(len(y))
    for tr, va in folds:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof_aug[tr], y[tr])
        aug_oof[va] = lr.predict_proba(F_oof_aug[va])[:, 1]
    aug_auc = roc_auc_score(y, aug_oof)
    delta_bp = (aug_auc - base_auc) * 1e4
    print(f"\nAugmented LR-meta OOF (16 feats): {aug_auc:.5f}  "
          f"(Δ {delta_bp:+.3f} bp)")

    # rho vs PRIMARY
    primary_test = pos(ART / "test_d13e_compound_stint_tau20000_strat.npy")
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof_aug, y)
    test_aug = lr_full.predict_proba(F_test_aug)[:, 1]
    rho, _ = spearmanr(test_aug, primary_test)
    print(f"ρ vs PRIMARY (d13e): {rho:.6f}")

    # Inspect LR coefs on the 4 width features
    print(f"\nLR coefs on 4 width features (last 4 of 16):")
    coefs = lr_full.coef_.ravel()
    for k_idx, b in enumerate(K4):
        print(f"  {b:<35s}  w_coef = {coefs[12 + k_idx]:+.4f}")

    # Save
    np.save(ART / "oof_K4_conformal_widths_strat.npy",
            np.column_stack([1 - aug_oof, aug_oof]).astype(np.float64))
    np.save(ART / "test_K4_conformal_widths_strat.npy",
            np.column_stack([1 - test_aug, test_aug]).astype(np.float64))
    (ART / "probe_s5_conformal_cqr_results.json").write_text(
        json.dumps({"baseline_oof": base_auc,
                    "augmented_oof": aug_auc,
                    "delta_bp": delta_bp,
                    "rho_vs_primary": float(rho),
                    "width_coefs": [float(coefs[12 + i]) for i in range(4)]},
                   indent=2))


if __name__ == "__main__":
    main()
