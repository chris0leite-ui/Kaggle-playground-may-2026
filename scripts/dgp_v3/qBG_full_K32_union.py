"""qBG — TRUE K=32 union: 27 individual K=27-era bases + 5 new slim-kNN bases.

Hiding in plain sight: we've been treating K=27 path-b-meta as a single
SUPER-base. But the 27 individual base OOFs are saved as artifacts.
Building K=32 = (27 individuals) + qAT + qAV + qAO + qAA + qAF + qAK
gives the LR-meta + Path-B C×S access to ALL bases at full resolution,
not collapsed to a 1-D meta-output.

Predicted: K=32 should beat the K=11-with-K27-super-base because the
meta can re-weight each old base individually based on its
correlation with the new slim-kNN bases.
"""
from __future__ import annotations

import json
import time
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
MIN_ROWS = 1000
MAX_ITER = 500


# K=27 base list (from scripts/d18_path_b.py)
POOL_KEEP_16 = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
]
TOP_3_D9 = ["d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound"]
FM_AB = ["d9f_FM_A", "d9f_FM_B"]
K27_EXTRAS = [
    "p1_single_cb_v4_gpu", "d17_h1d_yekenot_full",
    "d16_orig_continuous_only", "d18_chain_decomp",
    "d18_e2_preimage_knn", "d18_f2_constraint",
]
K27_NAMES = POOL_KEEP_16 + TOP_3_D9 + FM_AB + K27_EXTRAS


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def fit_lr_aug(F, y):
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def lr_meta_oof(Xm, y_):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    om = np.zeros(len(y_))
    for tr, va in skf.split(Xm, y_):
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        m.fit(Xm[tr], y_[tr])
        om[va] = m.predict_proba(Xm[va])[:, 1]
    return om


def run_pathb(base_oofs, base_tests, train, test, y, taus=[20000, 100000, 500000]):
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    n_cats = len(cats)
    seg_train = c_tr * 6 + s_tr
    seg_test = c_te * 6 + s_te
    n_seg = n_cats * 6
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    oofs = {tau: np.zeros(len(y)) for tau in taus}
    for fold, (tr, va) in enumerate(splits):
        w_g = fit_lr_aug(F_oof[tr], y[tr])
        W_l = np.zeros((n_seg, len(w_g)))
        cnt = np.zeros(n_seg, dtype=np.int64)
        msk = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            cnt[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_l[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            msk[s] = True
        for tau in taus:
            n_local = cnt.astype(np.float64)
            alpha = n_local / (n_local + tau)
            W_sh = alpha[:, None] * W_l + (1 - alpha[:, None]) * w_g[None, :]
            for s in np.unique(seg_train[va]):
                idx_v = np.where(seg_train[va] == s)[0]
                w = W_sh[s] if msk[s] else w_g
                oofs[tau][va[idx_v]] = predict_aug(F_oof[va[idx_v]], w)
    w_g_full = fit_lr_aug(F_oof, y)
    W_l_full = np.zeros((n_seg, len(w_g_full)))
    cnt_full = np.zeros(n_seg, dtype=np.int64)
    msk_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        cnt_full[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_l_full[s] = fit_lr_aug(F_oof[idx], y[idx])
        msk_full[s] = True
    test_preds = {}
    for tau in taus:
        n_loc = cnt_full.astype(np.float64)
        alpha = n_loc / (n_loc + tau)
        W_sh = alpha[:, None] * W_l_full + (1 - alpha[:, None]) * w_g_full[None, :]
        tp = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx_t = np.where(seg_test == s)[0]
            w = W_sh[s] if msk_full[s] else w_g_full
            tp[idx_t] = predict_aug(F_test[idx_t], w)
        test_preds[tau] = tp
    return oofs, test_preds


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    print(f"Loading 27 K=27-era bases...")
    K27_oofs, K27_tests = [], []
    missing = []
    for nm in K27_NAMES:
        try:
            oo = _pos(ART / f"oof_{nm}_strat.npy")
            te = _pos(ART / f"test_{nm}_strat.npy")
            K27_oofs.append(oo)
            K27_tests.append(te)
        except FileNotFoundError:
            missing.append(nm)
    if missing:
        print(f"  WARN: missing {len(missing)} base(s): {missing}")
    print(f"  loaded {len(K27_oofs)} of {len(K27_NAMES)}")

    # New slim-kNN bases
    NEW_BASES = [
        ("qAT", "dgp_v3_qAT_K1_oof.npy", "dgp_v3_qAT_K1_test.npy"),
        ("qAV", "dgp_v3_qAV_K1_7feat_oof.npy", "dgp_v3_qAV_K1_7feat_test.npy"),
        ("qAO", "dgp_v3_qAO_knn_multi_oof.npy", "dgp_v3_qAO_knn_multi_test.npy"),
        ("qAA", "dgp_v3_qAA_stint_imputed_oof.npy", "dgp_v3_qAA_stint_imputed_test.npy"),
        ("qAF", "dgp_v3_qAF_d16plus_oof.npy", "dgp_v3_qAF_d16plus_test.npy"),
        ("qAK", "dgp_v3_qAK_knn3_oof.npy", "dgp_v3_qAK_knn3_test.npy"),
        ("qBA", "dgp_v3_qBA_manhattan_oof.npy", "dgp_v3_qBA_manhattan_test.npy"),
    ]
    new_oofs, new_tests, new_names = [], [], []
    for nm, oof_f, test_f in NEW_BASES:
        if (ART / oof_f).exists():
            new_oofs.append(_pos(ART / oof_f))
            new_tests.append(_pos(ART / test_f))
            new_names.append(nm)
    print(f"  loaded {len(new_oofs)} new slim-kNN bases: {new_names}")

    all_oofs = K27_oofs + new_oofs
    all_tests = K27_tests + new_tests
    K_total = len(all_oofs)
    print(f"  TOTAL pool: K={K_total}")

    # Plain LR-meta on full K_total
    auc_full_plain = float(roc_auc_score(y, lr_meta_oof(expand(np.column_stack(all_oofs)), y)))
    print(f"\n=== K={K_total} plain LR-meta OOF: {auc_full_plain:.5f} ===")

    PRIMARY_oof = _pos(ART / "oof_K4_fwd_pathb.npy")
    PRIMARY_test = _pos(ART / "test_K4_fwd_pathb.npy")
    primary_K4_auc = float(roc_auc_score(y, PRIMARY_oof))
    delta_plain = (auc_full_plain - primary_K4_auc) * 1e4
    print(f"  ΔvsK4PRIMARY: {delta_plain:+.3f} bp")

    # Path-B amp
    print(f"\n--- Path-B amp on K={K_total} ---")
    oofs, test_preds = run_pathb(all_oofs, all_tests, train, test, y)
    out = {"K_total": K_total, "PRIMARY_K4_oof": primary_K4_auc,
           "plain_lr_meta_oof": auc_full_plain, "plain_delta_bp": delta_plain,
           "pathb": {}}
    for tau, oo in oofs.items():
        auc = float(roc_auc_score(y, oo))
        delta = (auc - primary_K4_auc) * 1e4
        tp = test_preds[tau]
        rho = float(spearmanr(tp, PRIMARY_test).correlation)
        primary_class = (PRIMARY_test >= 0.5).astype(int)
        new_class = (tp >= 0.5).astype(int)
        flips_pos = int(((new_class == 1) & (primary_class == 0)).sum())
        flips_neg = int(((new_class == 0) & (primary_class == 1)).sum())
        flip_ratio = (
            min(flips_pos, flips_neg) / max(flips_pos, flips_neg)
            if max(flips_pos, flips_neg) > 0 else 0.0
        )
        out["pathb"][f"tau_{tau}"] = {
            "oof_auc": auc, "delta_oof_bp": delta,
            "rho_test_vs_PRIMARY_K4": rho,
            "flips_pos_neg": [flips_pos, flips_neg],
            "flip_ratio": flip_ratio,
        }
        print(f"  τ={tau:>6d}: OOF={auc:.5f} Δ={delta:+.3f} bp  ρ={rho:.5f}  flips={flips_pos}/{flips_neg} ({flip_ratio:.3f})")

        np.save(ART / f"dgp_v3_qBG_K{K_total}_pathb_tau{tau}_oof.npy", oo)
        np.save(ART / f"dgp_v3_qBG_K{K_total}_pathb_tau{tau}_test.npy", tp)

        sub = pd.DataFrame({"id": test["id"].values, TARGET: tp})
        sub_path = Path("submissions") / f"submission_qBG_K{K_total}_full_union_pathb_tau{tau}.csv"
        sub_path.parent.mkdir(exist_ok=True)
        sub.to_csv(sub_path, index=False)

    fp = ART / "dgp_v3_qBG_K_total.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {fp.name}; total wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
