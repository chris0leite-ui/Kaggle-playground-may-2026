"""qBI — #1 + #3 combined:
- #1: K=12 = K=11 + qBA Manhattan kNN. Path-B amp on top of K=12.
- #3: K=34 plain LR-meta with tighter regularization (C=0.1, C=0.01).

qBA Manhattan was K=4+1 +1.414 bp standalone — best of new variants — but
was never combined with K=27 ensemble. Adding it on top of K=11 PRIMARY
introduces a new distance metric the ensemble hasn't seen.

For #3: K=34 plain LR with C=1 had OOF +4.198 → LB 0.95373 (REGRESS).
Over-parameterized at C=1. C=0.1 or C=0.01 might fix this.
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

NEW_BASES = [
    ("qAT", "dgp_v3_qAT_K1_oof.npy", "dgp_v3_qAT_K1_test.npy"),
    ("qAV", "dgp_v3_qAV_K1_7feat_oof.npy", "dgp_v3_qAV_K1_7feat_test.npy"),
    ("qAO", "dgp_v3_qAO_knn_multi_oof.npy", "dgp_v3_qAO_knn_multi_test.npy"),
    ("qAA", "dgp_v3_qAA_stint_imputed_oof.npy", "dgp_v3_qAA_stint_imputed_test.npy"),
    ("qAF", "dgp_v3_qAF_d16plus_oof.npy", "dgp_v3_qAF_d16plus_test.npy"),
    ("qAK", "dgp_v3_qAK_knn3_oof.npy", "dgp_v3_qAK_knn3_test.npy"),
    ("qBA", "dgp_v3_qBA_manhattan_oof.npy", "dgp_v3_qBA_manhattan_test.npy"),
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def fit_lr_aug(F, y, C=1.0):
    lr = LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def lr_meta_oof(Xm, y_, C=1.0):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    om = np.zeros(len(y_))
    test_p = None
    for tr, va in skf.split(Xm, y_):
        m = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
        m.fit(Xm[tr], y_[tr])
        om[va] = m.predict_proba(Xm[va])[:, 1]
    return om


def lr_meta_full(Xm_train, Xm_test, y_, C=1.0):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    om = np.zeros(len(y_))
    test_p = np.zeros(Xm_test.shape[0])
    for tr, va in skf.split(Xm_train, y_):
        m = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
        m.fit(Xm_train[tr], y_[tr])
        om[va] = m.predict_proba(Xm_train[va])[:, 1]
        test_p += m.predict_proba(Xm_test)[:, 1] / N_FOLDS
    return om, test_p


def run_pathb(base_oofs, base_tests, train, test, y, taus=[20000, 100000]):
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_train = c_tr * 6 + s_tr
    seg_test = c_te * 6 + s_te
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

    K4 = [
        ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy",     "test_d17_h1d_yekenot_full_strat.npy"),
        ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy",      "test_p1_single_cb_v4_gpu_strat.npy"),
        ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy",             "test_f1_hgbc_deep_strat.npy"),
        ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ]
    K4_oof = [_pos(ART / f) for _, f, _ in K4]
    K4_test = [_pos(ART / f) for _, _, f in K4]

    EXTRAS = {nm: (_pos(ART / o), _pos(ART / t)) for nm, o, t in NEW_BASES}

    K27_100k_oof = _pos(ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")
    K27_100k_test = _pos(ART / "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")

    PRIMARY_oof = _pos(ART / "oof_K4_fwd_pathb.npy")
    PRIMARY_test = _pos(ART / "test_K4_fwd_pathb.npy")
    primary_K4_auc = float(roc_auc_score(y, PRIMARY_oof))
    print(f"K=4 PRIMARY OOF: {primary_K4_auc:.5f}")

    out = {"results": []}

    # ---------- #1: K=12 = K=11 + qBA Manhattan ----------
    print("\n#1: K=12 = K=4 + qAT+qAV+qAO+qAA+qAF+qAK+K27_100k + qBA")
    combo = ["qAT", "qAV", "qAO", "qAA", "qAF", "qAK", "qBA"]
    extras_oofs = [EXTRAS[k][0] for k in combo] + [K27_100k_oof]
    extras_tests = [EXTRAS[k][1] for k in combo] + [K27_100k_test]
    oof_list = K4_oof + extras_oofs
    test_list = K4_test + extras_tests
    K = len(oof_list)
    print(f"  K={K} pool")

    # Plain LR-meta
    om, tp = lr_meta_full(expand(np.column_stack(oof_list)),
                           expand(np.column_stack(test_list)), y, C=1.0)
    auc = float(roc_auc_score(y, om))
    delta = (auc - primary_K4_auc) * 1e4
    rho = float(spearmanr(tp, PRIMARY_test).correlation)
    print(f"  plain LR-meta: OOF {auc:.5f} Δ{delta:+.3f} ρ_vsK4={rho:.5f}")

    # Path-B amp
    oofs_pb, tests_pb = run_pathb(oof_list, test_list, train, test, y)
    for tau, oo in oofs_pb.items():
        a = float(roc_auc_score(y, oo))
        d = (a - primary_K4_auc) * 1e4
        r = float(spearmanr(tests_pb[tau], PRIMARY_test).correlation)
        print(f"  Path-B τ={tau:>6d}: OOF {a:.5f} Δ{d:+.3f} ρ={r:.5f}")
        np.save(ART / f"dgp_v3_qBI_K{K}_pathb_tau{tau}_oof.npy", oo)
        np.save(ART / f"dgp_v3_qBI_K{K}_pathb_tau{tau}_test.npy", tests_pb[tau])
        sub = pd.DataFrame({"id": test["id"].values, TARGET: tests_pb[tau]})
        sub_path = Path(f"submissions/submission_qBI_K{K}_qAT_qAV_qAO_qAA_qAF_qAK_qBA_K27_pathb_tau{tau}.csv")
        sub.to_csv(sub_path, index=False)

    # Save plain LR submission too
    np.save(ART / f"dgp_v3_qBI_K{K}_plain_oof.npy", om)
    np.save(ART / f"dgp_v3_qBI_K{K}_plain_test.npy", tp)
    sub = pd.DataFrame({"id": test["id"].values, TARGET: tp})
    sub.to_csv(f"submissions/submission_qBI_K{K}_plain_lr.csv", index=False)

    out["results"].append({"id": "qBI_K12", "K": K, "plain_oof": auc, "plain_delta_bp": delta})

    # ---------- #3: K=34 with tighter C ----------
    print(f"\n#3: K=34 unrolled with tighter LR-meta regularization")
    K27_oofs = [_pos(ART / f"oof_{nm}_strat.npy") for nm in K27_NAMES]
    K27_tests = [_pos(ART / f"test_{nm}_strat.npy") for nm in K27_NAMES]
    new_oofs = [EXTRAS[nm][0] for nm, _, _ in NEW_BASES]
    new_tests = [EXTRAS[nm][1] for nm, _, _ in NEW_BASES]
    all_oofs = K27_oofs + new_oofs
    all_tests = K27_tests + new_tests
    K_total = len(all_oofs)
    F_oof_all = expand(np.column_stack(all_oofs))
    F_test_all = expand(np.column_stack(all_tests))

    for C_val in [0.1, 0.03, 0.01]:
        om, tp = lr_meta_full(F_oof_all, F_test_all, y, C=C_val)
        auc = float(roc_auc_score(y, om))
        delta = (auc - primary_K4_auc) * 1e4
        rho = float(spearmanr(tp, PRIMARY_test).correlation)
        print(f"  K=34 C={C_val}: OOF {auc:.5f} Δ{delta:+.3f} ρ_vsK4={rho:.5f}")
        np.save(ART / f"dgp_v3_qBI_K{K_total}_C{C_val}_oof.npy", om)
        np.save(ART / f"dgp_v3_qBI_K{K_total}_C{C_val}_test.npy", tp)
        sub = pd.DataFrame({"id": test["id"].values, TARGET: tp})
        sub.to_csv(f"submissions/submission_qBI_K{K_total}_plain_C{C_val}.csv", index=False)
        out["results"].append({"id": f"qBI_K34_C{C_val}", "K": K_total, "C": C_val,
                                "plain_oof": auc, "plain_delta_bp": delta})

    fp = ART / "dgp_v3_qBI_results.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {fp.name}; total wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
