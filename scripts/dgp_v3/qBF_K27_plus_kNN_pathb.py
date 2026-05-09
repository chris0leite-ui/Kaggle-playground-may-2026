"""qBF — K=27 PRIMARY-of-old (Path-B τ=100k) as base + qAT/qAV/qAO + Path-B amp.

K=27 path-b τ=100k OOF 0.95432 (LB 0.95368 historic). Higher OOF than
my K=9 (0.95423, LB 0.95375 today).

Hypothesis: The K=27 result has DGP-class info our K=9 doesn't have
(d18 chain decomp, F2 constraints, E2 preimage-kNN, etc.). Combining
K=27 path-b as a SUPER-BASE with the new slim-kNN bases qAT/qAV/qAO/qAA/qAF
should give a stack that exceeds either alone.

Procedure:
1. Treat K=27 path-b τ=100k as a single base (3-feat expansion)
2. Add to K=4 + qAT + qAV + qAO + qAA + qAF for K=10 pool
3. Apply Path-B C×S τ-sweep
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


def run_pathb(base_oofs, base_tests, train, test, y, taus=[5000, 20000, 100000]):
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

    K4_FILES = [
        ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy",     "test_d17_h1d_yekenot_full_strat.npy"),
        ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy",      "test_p1_single_cb_v4_gpu_strat.npy"),
        ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy",             "test_f1_hgbc_deep_strat.npy"),
        ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ]
    K4_oof = [_pos(ART / o) for _, o, _ in K4_FILES]
    K4_test = [_pos(ART / t) for _, _, t in K4_FILES]
    K4_names = [n for n, _, _ in K4_FILES]

    EXTRAS = {
        "qAT": ("dgp_v3_qAT_K1_oof.npy", "dgp_v3_qAT_K1_test.npy"),
        "qAV": ("dgp_v3_qAV_K1_7feat_oof.npy", "dgp_v3_qAV_K1_7feat_test.npy"),
        "qAO": ("dgp_v3_qAO_knn_multi_oof.npy", "dgp_v3_qAO_knn_multi_test.npy"),
        "qAA": ("dgp_v3_qAA_stint_imputed_oof.npy", "dgp_v3_qAA_stint_imputed_test.npy"),
        "qAF": ("dgp_v3_qAF_d16plus_oof.npy", "dgp_v3_qAF_d16plus_test.npy"),
        "qAK": ("dgp_v3_qAK_knn3_oof.npy", "dgp_v3_qAK_knn3_test.npy"),
        # K=27 PRIMARY-of-old
        "K27_100k": ("oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
                     "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"),
        "K27_20k":  ("oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau20000_strat.npy",
                     "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau20000_strat.npy"),
    }

    PRIMARY_oof_K4 = _pos(ART / "oof_K4_fwd_pathb.npy")
    PRIMARY_test_K4 = _pos(ART / "test_K4_fwd_pathb.npy")
    primary_K4_auc = float(roc_auc_score(y, PRIMARY_oof_K4))
    print(f"PRIMARY K=4 OOF: {primary_K4_auc:.5f}")

    # K=27 path-b reference
    K27_oof = _pos(ART / EXTRAS["K27_100k"][0])
    K27_test = _pos(ART / EXTRAS["K27_100k"][1])
    print(f"K=27 path-b τ=100k OOF: {roc_auc_score(y, K27_oof):.5f}")

    # 1) Plain LR-meta combos with K=27 added
    print("\n--- Plain LR-meta combos ---")
    K4_anchor_auc = float(roc_auc_score(y, lr_meta_oof(expand(np.column_stack(K4_oof)), y)))
    print(f"K=4 plain LR-meta: {K4_anchor_auc:.5f}")

    test_combos = [
        ["K27_100k"],
        ["qAT", "qAV", "K27_100k"],
        ["qAT", "qAV", "qAO", "K27_100k"],
        ["qAT", "qAV", "qAO", "qAA", "qAF", "K27_100k"],
        ["qAT", "qAV", "qAO", "qAA", "qAF", "K27_20k"],
        ["qAT", "qAV", "qAO", "qAA", "qAF", "K27_100k", "K27_20k"],
        ["qAT", "qAV", "qAO", "qAA", "qAF", "qAK", "K27_100k"],
    ]
    out = {"PRIMARY_K4_oof": primary_K4_auc, "K4_anchor": K4_anchor_auc, "results": []}

    for combo in test_combos:
        oof_list = K4_oof + [_pos(ART / EXTRAS[c][0]) for c in combo]
        test_list = K4_test + [_pos(ART / EXTRAS[c][1]) for c in combo]
        Xm = expand(np.column_stack(oof_list))
        auc = float(roc_auc_score(y, lr_meta_oof(Xm, y)))
        delta_p = (auc - primary_K4_auc) * 1e4
        delta_k = (auc - K4_anchor_auc) * 1e4
        print(f"  K={4+len(combo):2d}  {'+'.join(combo):<40s} OOF={auc:.5f} ΔvsPRIMARY={delta_p:+.3f} bp")

    # 2) Path-B amp on top combos
    print("\n--- Path-B amp ---")
    pathb_combos = [
        ["qAT", "qAV", "K27_100k"],                  # 7 bases
        ["qAT", "qAV", "qAO", "qAA", "qAF", "K27_100k"],  # 10 bases
        ["qAT", "qAV", "qAO", "qAA", "qAF", "qAK", "K27_100k"],  # 11 bases
    ]
    for combo in pathb_combos:
        oof_list = K4_oof + [_pos(ART / EXTRAS[c][0]) for c in combo]
        test_list = K4_test + [_pos(ART / EXTRAS[c][1]) for c in combo]
        names = K4_names + combo
        print(f"\n  K={len(names)} pool: {names}")
        oofs, test_preds = run_pathb(oof_list, test_list, train, test, y)
        for tau, oo in oofs.items():
            auc = float(roc_auc_score(y, oo))
            delta = (auc - primary_K4_auc) * 1e4
            tp = test_preds[tau]
            rho = float(spearmanr(tp, PRIMARY_test_K4).correlation)
            primary_class = (PRIMARY_test_K4 >= 0.5).astype(int)
            new_class = (tp >= 0.5).astype(int)
            flips_pos = int(((new_class == 1) & (primary_class == 0)).sum())
            flips_neg = int(((new_class == 0) & (primary_class == 1)).sum())
            flip_ratio = (
                min(flips_pos, flips_neg) / max(flips_pos, flips_neg)
                if max(flips_pos, flips_neg) > 0 else 0.0
            )
            print(f"    τ={tau:>6d}: OOF={auc:.5f} Δ={delta:+.3f} bp  ρ={rho:.5f}  flips={flips_pos}/{flips_neg} ({flip_ratio:.3f})")
            out["results"].append({
                "combo": combo, "K": len(names), "tau": tau,
                "oof_auc": auc, "delta_oof_bp": delta,
                "rho_test_vs_PRIMARY_K4": rho,
                "flips_pos_neg": [flips_pos, flips_neg],
                "flip_ratio": flip_ratio,
            })

            # Save submission
            if delta > 1.5:
                cb = "_".join(combo)
                np.save(ART / f"dgp_v3_qBF_{cb}_pathb_tau{tau}_oof.npy", oo)
                np.save(ART / f"dgp_v3_qBF_{cb}_pathb_tau{tau}_test.npy", tp)
                sub = pd.DataFrame({"id": test["id"].values, TARGET: tp})
                sub_path = Path("submissions") / f"submission_qBF_K{len(names)}_{cb}_pathb_tau{tau}.csv"
                sub_path.parent.mkdir(exist_ok=True)
                sub.to_csv(sub_path, index=False)

    fp = ART / "dgp_v3_qBF_K27_kNN_results.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[total wall {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
