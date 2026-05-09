"""qAG — Path-B C×S τ=100k on K=5 = K=4 + qAA stint_imputed base.

Tests whether per-segment shrinkage on Compound × Stint amplifies
qAA's K=4+1 +0.143 bp OOF lift to a meta-arch-redesign-class lift.

Per the friction `path-b-cs-absorbs-single-base-orthogonal-additions-below-0.5bp`,
predicted absorption since qAA is at +0.143 OOF (well below 0.5 bp).
But qAA is structurally distinct from RF-yekenot (which absorbed) — qAA's
ρ_test = 0.965 vs RF's 0.9595, but qAA carries new sequence info.

Output: K=5 + Path-B C×S τ=5k/20k/100k OOF and test predictions; gate
report; diff vs PRIMARY (K=4 + Path-B); flip count.
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
PRIMARY_LB = 0.95351


def _pos(p: Path) -> np.ndarray:
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


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # K=4 bases
    BASES = [
        ("yekenot", "oof_d17_h1d_yekenot_full_strat.npy", "test_d17_h1d_yekenot_full_strat.npy"),
        ("cb_v4", "oof_p1_single_cb_v4_gpu_strat.npy", "test_p1_single_cb_v4_gpu_strat.npy"),
        ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy", "test_f1_hgbc_deep_strat.npy"),
        ("d16_orig", "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ]
    base_oofs, base_tests, names = [], [], []
    for nm, oof_f, test_f in BASES:
        base_oofs.append(_pos(ART / oof_f))
        base_tests.append(_pos(ART / test_f))
        names.append(nm)
    # qAA
    base_oofs.append(_pos(ART / "dgp_v3_qAA_stint_imputed_oof.npy"))
    base_tests.append(_pos(ART / "dgp_v3_qAA_stint_imputed_test.npy"))
    names.append("qAA_stint")
    print(f"K=5 pool: {names}")

    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))

    # Compound × Stint (clipping Stint at 5)
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
    sizes = np.bincount(seg_train, minlength=n_seg)
    populated = int(np.sum(sizes >= MIN_ROWS))
    print(f"Compound×Stint: {populated} segments ≥{MIN_ROWS} rows")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    taus = [5000, 20000, 100000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    print("\nFolding hier-meta...")
    for fold, (tr, va) in enumerate(splits):
        t_f = time.time()
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
        print(f"  fold {fold} done {time.time()-t_f:.1f}s")

    # Full-train test prediction
    print("\nFull-train test prediction...")
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

    # Reports
    PRIMARY_oof = _pos(ART / "oof_K4_fwd_pathb.npy")
    PRIMARY_test = _pos(ART / "test_K4_fwd_pathb.npy")
    primary_oof_auc = float(roc_auc_score(y, PRIMARY_oof))
    print(f"\nPRIMARY (K=4 + Path-B C×S τ=100k) OOF AUC: {primary_oof_auc:.5f}")

    out: dict = {
        "PRIMARY_oof_auc": primary_oof_auc,
        "PRIMARY_LB": PRIMARY_LB,
        "K5_path_b_cs": {},
    }

    for tau in taus:
        auc = float(roc_auc_score(y, oofs[tau]))
        delta_oof = (auc - primary_oof_auc) * 1e4
        rho = float(spearmanr(test_preds[tau], PRIMARY_test).correlation)
        # Flip count threshold 0.5
        primary_class = (PRIMARY_test >= 0.5).astype(int)
        new_class = (test_preds[tau] >= 0.5).astype(int)
        flips_pos = int(((new_class == 1) & (primary_class == 0)).sum())
        flips_neg = int(((new_class == 0) & (primary_class == 1)).sum())
        flip_ratio = (
            min(flips_pos, flips_neg) / max(flips_pos, flips_neg)
            if max(flips_pos, flips_neg) > 0 else 0.0
        )
        out["K5_path_b_cs"][f"tau_{tau}"] = {
            "oof_auc": auc, "delta_oof_bp": delta_oof,
            "rho_test_vs_PRIMARY": rho,
            "flips_pos_neg": [flips_pos, flips_neg],
            "flip_ratio": flip_ratio,
        }
        print(f"\n  τ={tau:>6d}: OOF={auc:.5f} Δ={delta_oof:+.3f} bp")
        print(f"             ρ_test vs PRIMARY={rho:.5f}  flips={flips_pos}/{flips_neg} (ratio={flip_ratio:.3f})")

        # Save per-tau OOF/test
        np.save(ART / f"dgp_v3_qAG_K5_qAA_pathb_tau{tau}_oof.npy", oofs[tau])
        np.save(ART / f"dgp_v3_qAG_K5_qAA_pathb_tau{tau}_test.npy", test_preds[tau])

        # Submission file format
        sub = pd.DataFrame({"id": test["id"].values, TARGET: test_preds[tau]})
        sub_path = Path("submissions") / f"submission_K5_qAA_pathb_cs_tau{tau}.csv"
        sub_path.parent.mkdir(exist_ok=True)
        sub.to_csv(sub_path, index=False)

    fp = ART / "dgp_v3_qAG_K5_qAA_pathb.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {fp.name}")
    print(f"  total wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
