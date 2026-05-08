"""scripts/probe_lane4_nonlr_meta.py — Lane 4 (4th latent construct).

The K=27 logit pool sits in a 3.23-D subspace (A25). EXP-2/3/4 absorbed
because the 30-feature [P, rank, logit] LR meta projection saturates at
those 3 directions (A29/A30). Lane 4 directly attacks A30:

  D4.1 — characterise the 3 latent directions: SVD on K=4 logits;
         correlate top components with raw row features. Tells us WHAT
         the 3 directions are, hence what the missing 4th would be.

  P4.1 — gradient-boosted meta on K=4 [P, rank, logit] (12 features).
         Direct test of EXP-NEW. If LGBM lifts ≥1 bp over LR, the
         ceiling is LR-specific not data-intrinsic.

  P4.2 — small MLP meta on K=4 [P, rank, logit]. Different non-linearity
         from GBDT; 2 hidden layers, dropout 0.2.

  P4.4 — augmented LR meta on [P, rank, logit, raw_row_features].
         Cheapest test that's NOT yet on the EXPERIMENTS-NEXT.md menu.

P4.3 (kNN-on-predictions meta) is sketched as TODO at end-of-file; needs
faiss or a careful sklearn BallTree at 440k×4 — not run by default.

Cost (CPU): ~1.5 hr combined (LGBM is the heaviest at ~30 min).

Outputs:
  scripts/artifacts/probe_lane4_nonlr_meta.json
  scripts/artifacts/oof_lane4_gbm_meta_strat.npy   (P4.1)
  scripts/artifacts/oof_lane4_mlp_meta_strat.npy   (P4.2)
  scripts/artifacts/oof_lane4_aug_lr_meta_strat.npy (P4.4)
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 500

K4_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def main():
    t0 = time.time()
    print("Loading data + K=4 base OOFs ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4_FWD]
    base_tests = [_pos(ART / f"test_{n}_strat.npy") for n in K4_FWD]
    P4 = np.column_stack(base_oofs)
    P4_test = np.column_stack(base_tests)

    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED).split(np.zeros(len(y)), y))

    F4 = _expand(P4)
    F4_test = _expand(P4_test)

    # Baseline LR meta (target for delta comparisons)
    def fit_lr_meta(F, F_test, C=1.0):
        oof_pred = np.zeros(len(y))
        test_acc = np.zeros(F_test.shape[0])
        for tr, va in splits:
            lr = LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            oof_pred[va] = lr.predict_proba(F[va])[:, 1]
            test_acc += lr.predict_proba(F_test)[:, 1] / N_FOLDS
        return oof_pred, test_acc

    oof_lr, _ = fit_lr_meta(F4, F4_test)
    auc_lr = float(roc_auc_score(y, oof_lr))
    print(f"\nBaseline K=4 LR meta OOF: {auc_lr:.5f}")

    # ============ D4.1 — SVD characterisation =======================
    print("\n--- D4.1: SVD on K=4 logits, identify the 3 latent directions")
    # Logit space (where A25's 3.23-D rank was measured)
    logits = np.log(np.clip(P4, 1e-9, 1 - 1e-9) / (1 - np.clip(P4, 1e-9, 1 - 1e-9)))
    logits_centered = logits - logits.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(logits_centered, full_matrices=False)
    var_explained = (S ** 2) / (S ** 2).sum()
    eff_rank_logit = float(np.exp(-(var_explained * np.log(var_explained + 1e-12)).sum()))
    print(f"  Singular values: {[f'{s:.2f}' for s in S]}")
    print(f"  Variance explained: {[f'{v:.3f}' for v in var_explained]}")
    print(f"  Effective rank (entropy): {eff_rank_logit:.3f}")

    # Right singular vectors (each row of Vt is a direction in 4-base space)
    print("  Top 3 right-singular-vector loadings on bases:")
    for i in range(min(3, len(K4_FWD))):
        loadings = {K4_FWD[j]: float(Vt[i, j]) for j in range(len(K4_FWD))}
        print(f"    Component {i+1}: {loadings}")

    # Correlate top-3 components (U columns × S) with raw features
    # Note: these are PER-ROW projections, length n_train
    raw_feats = ["TyreLife", "LapNumber", "Stint", "Position",
                 "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]
    raw_present = [f for f in raw_feats if f in train.columns]
    correlations = {}
    for i in range(min(3, U.shape[1])):
        comp = U[:, i] * S[i]  # data-rotated component
        corrs = {}
        for feat in raw_present:
            v = train[feat].astype(float).values
            if np.isfinite(v).all() and v.std() > 0:
                corrs[feat] = float(np.corrcoef(comp, v)[0, 1])
        # Compound categorical one-hot
        for c in train["Compound"].unique():
            mask = (train["Compound"] == c).astype(float).values
            if mask.std() > 0:
                corrs[f"is_{c}"] = float(np.corrcoef(comp, mask)[0, 1])
        correlations[f"comp{i+1}"] = corrs
    print("  Top correlations (|ρ|>0.1) per component:")
    for comp_name, corrs in correlations.items():
        top = sorted(corrs.items(), key=lambda x: -abs(x[1]))[:5]
        print(f"    {comp_name}: " + ", ".join(f"{k}={v:+.3f}" for k, v in top))

    # ============ P4.1 — gradient-boosted meta ======================
    print("\n--- P4.1: gradient-boosted meta on K=4 [P, rank, logit] (12 feats)")
    LGB = dict(
        objective="binary", metric="auc", learning_rate=0.03,
        num_leaves=15, min_data_in_leaf=200, feature_fraction=0.9,
        lambda_l2=1.0, verbose=-1, n_jobs=-1, seed=SEED,
    )
    oof_gbm = np.zeros(len(y))
    test_gbm = np.zeros(F4_test.shape[0])
    for fold, (tr, va) in enumerate(splits):
        ds_tr = lgb.Dataset(F4[tr], label=y[tr])
        ds_va = lgb.Dataset(F4[va], label=y[va], reference=ds_tr)
        booster = lgb.train(
            LGB, ds_tr, num_boost_round=600, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        oof_gbm[va] = booster.predict(F4[va])
        test_gbm += booster.predict(F4_test) / N_FOLDS
    auc_gbm = float(roc_auc_score(y, oof_gbm))
    delta_p41_bp = (auc_gbm - auc_lr) * 1e4
    print(f"  K=4 LR meta : {auc_lr:.5f}")
    print(f"  K=4 GBM meta: {auc_gbm:.5f}  (Δ {delta_p41_bp:+.3f} bp)")
    np.save(ART / "oof_lane4_gbm_meta_strat.npy", oof_gbm)
    np.save(ART / "test_lane4_gbm_meta_strat.npy",
            np.column_stack([1 - test_gbm, test_gbm]))

    # ============ P4.2 — small MLP meta ============================
    print("\n--- P4.2: small MLP meta on K=4 [P, rank, logit]")
    oof_mlp = np.zeros(len(y))
    test_mlp = np.zeros(F4_test.shape[0])
    scaler = StandardScaler()
    F4s = scaler.fit_transform(F4)
    F4s_test = scaler.transform(F4_test)
    for fold, (tr, va) in enumerate(splits):
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu",
            alpha=1e-3, batch_size=4096, learning_rate="adaptive",
            learning_rate_init=1e-3, max_iter=40, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=5,
            random_state=SEED + fold,
        )
        mlp.fit(F4s[tr], y[tr])
        oof_mlp[va] = mlp.predict_proba(F4s[va])[:, 1]
        test_mlp += mlp.predict_proba(F4s_test)[:, 1] / N_FOLDS
    auc_mlp = float(roc_auc_score(y, oof_mlp))
    delta_p42_bp = (auc_mlp - auc_lr) * 1e4
    print(f"  K=4 LR meta : {auc_lr:.5f}")
    print(f"  K=4 MLP meta: {auc_mlp:.5f}  (Δ {delta_p42_bp:+.3f} bp)")
    np.save(ART / "oof_lane4_mlp_meta_strat.npy", oof_mlp)
    np.save(ART / "test_lane4_mlp_meta_strat.npy",
            np.column_stack([1 - test_mlp, test_mlp]))

    # ============ P4.4 — augmented LR meta with raw row features ===
    print("\n--- P4.4: augmented LR meta on [P, rank, logit, raw_row_features]")
    extra = train[raw_present].astype(float).fillna(-1).values
    extra_te = test[raw_present].astype(float).fillna(-1).values
    # Scale extra to put on similar footing as logits
    sc = StandardScaler()
    extra_s = sc.fit_transform(extra)
    extra_te_s = sc.transform(extra_te)
    F4A = np.hstack([F4, extra_s])
    F4A_test = np.hstack([F4_test, extra_te_s])
    oof_aug, test_aug = fit_lr_meta(F4A, F4A_test, C=0.5)
    auc_aug = float(roc_auc_score(y, oof_aug))
    delta_p44_bp = (auc_aug - auc_lr) * 1e4
    print(f"  K=4 LR plain : {auc_lr:.5f}")
    print(f"  K=4 LR + raw : {auc_aug:.5f}  (Δ {delta_p44_bp:+.3f} bp)")
    np.save(ART / "oof_lane4_aug_lr_meta_strat.npy", oof_aug)
    np.save(ART / "test_lane4_aug_lr_meta_strat.npy",
            np.column_stack([1 - test_aug, test_aug]))

    rho_p41 = float(spearmanr(oof_gbm, oof_lr)[0])
    rho_p42 = float(spearmanr(oof_mlp, oof_lr)[0])
    rho_p44 = float(spearmanr(oof_aug, oof_lr)[0])

    out = {
        "K4_bases": K4_FWD,
        "D4_1_singular_values": [float(s) for s in S],
        "D4_1_variance_explained": [float(v) for v in var_explained],
        "D4_1_effective_rank_entropy": eff_rank_logit,
        "D4_1_top3_loadings_per_base": {
            f"comp{i+1}": {K4_FWD[j]: float(Vt[i, j]) for j in range(len(K4_FWD))}
            for i in range(min(3, Vt.shape[0]))
        },
        "D4_1_top3_correlations": correlations,
        "P4_baseline_LR_meta_oof": auc_lr,
        "P4_1_GBM_meta_oof": auc_gbm,
        "P4_1_delta_bp": float(delta_p41_bp),
        "P4_1_rho_vs_LR": rho_p41,
        "P4_2_MLP_meta_oof": auc_mlp,
        "P4_2_delta_bp": float(delta_p42_bp),
        "P4_2_rho_vs_LR": rho_p42,
        "P4_4_aug_LR_meta_oof": auc_aug,
        "P4_4_delta_bp": float(delta_p44_bp),
        "P4_4_rho_vs_LR": rho_p44,
        "verdict_P4_1": ("PASS" if delta_p41_bp >= 0.5
                         else "AMBIG" if delta_p41_bp >= -0.1
                         else "NULL"),
        "verdict_P4_2": ("PASS" if delta_p42_bp >= 0.5
                         else "AMBIG" if delta_p42_bp >= -0.1
                         else "NULL"),
        "verdict_P4_4": ("PASS" if delta_p44_bp >= 0.5
                         else "AMBIG" if delta_p44_bp >= -0.1
                         else "NULL"),
        "wall_s": time.time() - t0,
    }
    (ART / "probe_lane4_nonlr_meta.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {ART/'probe_lane4_nonlr_meta.json'}. Wall {out['wall_s']:.1f}s")
    print(f"Verdicts: P4.1 {out['verdict_P4_1']} | P4.2 {out['verdict_P4_2']} "
          f"| P4.4 {out['verdict_P4_4']}")

    # TODO P4.3 — kNN-on-predictions meta. For each test row, find K=20
    # nearest neighbours in K=4 logit space (use sklearn BallTree on the
    # 4D logits; ~3 min build, ~5 min query at 440k×4). Output mean
    # of neighbour labels. Write to oof_lane4_knn_meta_strat.npy.
    # If P4.1 NULL this is also expected NULL (both bypass linearity);
    # if P4.1 AMBIG, kNN is the cheap differential test.


if __name__ == "__main__":
    main()
