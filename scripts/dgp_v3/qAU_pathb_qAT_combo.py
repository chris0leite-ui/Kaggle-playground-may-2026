"""qAU — final Path-B amp using the qAT K=1 breakthrough.

qAT K=1 standalone gives K=4+1 +1.172 bp at plain LR-meta (vs
qAK +0.717). Test combinations and Path-B amp.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
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


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    BASES = [
        ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy",     "test_d17_h1d_yekenot_full_strat.npy"),
        ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy",      "test_p1_single_cb_v4_gpu_strat.npy"),
        ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy",             "test_f1_hgbc_deep_strat.npy"),
        ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ]
    base_oofs, base_tests, names = [], [], []
    for nm, oof_f, test_f in BASES:
        base_oofs.append(_pos(ART / oof_f))
        base_tests.append(_pos(ART / test_f))
        names.append(nm)

    EXTRAS = [
        ("qAT", "dgp_v3_qAT_K1_oof.npy", "dgp_v3_qAT_K1_test.npy"),
        ("qAK", "dgp_v3_qAK_knn3_oof.npy", "dgp_v3_qAK_knn3_test.npy"),
        ("qAO", "dgp_v3_qAO_knn_multi_oof.npy", "dgp_v3_qAO_knn_multi_test.npy"),
        ("qAA", "dgp_v3_qAA_stint_imputed_oof.npy", "dgp_v3_qAA_stint_imputed_test.npy"),
        ("qAF", "dgp_v3_qAF_d16plus_oof.npy", "dgp_v3_qAF_d16plus_test.npy"),
    ]
    extras_dict = {nm: (_pos(ART / o), _pos(ART / t)) for nm, o, t in EXTRAS}
    print(f"Loaded extras: {list(extras_dict.keys())}")

    K4_oof = lr_meta_oof(expand(np.column_stack(base_oofs)), y)
    K4_auc = float(roc_auc_score(y, K4_oof))
    print(f"\nK=4 anchor: {K4_auc:.5f}")

    # Top combos with qAT
    print("\n--- qAT-anchored combos at plain LR-meta ---")
    combos_results = []
    extras_keys = list(extras_dict.keys())
    for r in range(1, len(extras_keys) + 1):
        for c in combinations(extras_keys, r):
            if "qAT" not in c:
                continue
            oofs = [extras_dict[k][0] for k in c]
            xm = expand(np.column_stack(base_oofs + oofs))
            auc = float(roc_auc_score(y, lr_meta_oof(xm, y)))
            delta = (auc - K4_auc) * 1e4
            combos_results.append((c, auc, delta))

    combos_results.sort(key=lambda x: -x[2])
    print("TOP 10:")
    for c, auc, delta in combos_results[:10]:
        print(f"  K={4+len(c):2d}  {'+'.join(c):<25s} OOF={auc:.5f} Δ={delta:+.3f} bp")

    # Path-B amp on best K=5 (qAT alone), K=6 (qAT+best partner), K=7 (qAT+2 best)
    print("\n--- Path-B amp on top combos ---")

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
    PRIMARY_oof = _pos(ART / "oof_K4_fwd_pathb.npy")
    PRIMARY_test = _pos(ART / "test_K4_fwd_pathb.npy")
    primary_oof_auc = float(roc_auc_score(y, PRIMARY_oof))
    print(f"PRIMARY OOF: {primary_oof_auc:.5f}")

    out = {"K4_anchor": K4_auc, "PRIMARY_oof": primary_oof_auc, "combos": [], "pathb": []}

    # Test top 5 qAT-combos with Path-B τ=20k
    for c, plain_auc, plain_delta in combos_results[:5]:
        oofs_extras = [extras_dict[k][0] for k in c]
        tests_extras = [extras_dict[k][1] for k in c]
        F_oof = expand(np.column_stack(base_oofs + oofs_extras))
        F_test = expand(np.column_stack(base_tests + tests_extras))

        oof_pb = np.zeros(len(y))
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
            tau = 20000
            n_local = cnt.astype(np.float64)
            alpha = n_local / (n_local + tau)
            W_sh = alpha[:, None] * W_l + (1 - alpha[:, None]) * w_g[None, :]
            for s in np.unique(seg_train[va]):
                idx_v = np.where(seg_train[va] == s)[0]
                w = W_sh[s] if msk[s] else w_g
                oof_pb[va[idx_v]] = predict_aug(F_oof[va[idx_v]], w)
        auc_pb = float(roc_auc_score(y, oof_pb))
        delta_pb = (auc_pb - primary_oof_auc) * 1e4
        amp_factor = delta_pb / plain_delta if plain_delta > 0 else 0
        print(f"  K={4+len(c):2d}  {'+'.join(c):<22s} plain Δ={plain_delta:+.3f} → Path-B τ=20k Δ={delta_pb:+.3f} (amp {amp_factor:.2f}×)")

        # Full-train test
        w_g = fit_lr_aug(F_oof, y)
        W_l = np.zeros((n_seg, len(w_g)))
        cnt = np.zeros(n_seg, dtype=np.int64)
        msk = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train == s)[0]
            cnt[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
                continue
            W_l[s] = fit_lr_aug(F_oof[idx], y[idx])
            msk[s] = True
        tau = 20000
        n_local = cnt.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_sh = alpha[:, None] * W_l + (1 - alpha[:, None]) * w_g[None, :]
        tp = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx_t = np.where(seg_test == s)[0]
            w = W_sh[s] if msk[s] else w_g
            tp[idx_t] = predict_aug(F_test[idx_t], w)

        rho = float(spearmanr(tp, PRIMARY_test).correlation)
        out["pathb"].append({
            "combo": list(c), "K": 4+len(c),
            "plain_oof": plain_auc, "plain_lift_bp": plain_delta,
            "pathb_oof": auc_pb, "pathb_lift_bp": delta_pb,
            "pathb_rho_test": rho,
        })

        # Save submission for top 3
        if delta_pb > 1.3:
            sub = pd.DataFrame({"id": test["id"].values, TARGET: tp})
            cb = "_".join(c)
            sub_path = Path("submissions") / f"submission_{cb}_pathb_cs_tau20000.csv"
            sub_path.parent.mkdir(exist_ok=True)
            sub.to_csv(sub_path, index=False)

    fp = ART / "dgp_v3_qAU_qAT_combos.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {fp.name}; total wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
