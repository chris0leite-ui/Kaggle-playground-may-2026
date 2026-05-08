"""scripts/probe_exp1_gru_retest.py — EXP-1 (partial): GRU re-test at K=10.

The d16 GRU sequence base was tested NULL at K=22+1 (Δ −0.043 bp).
But the K=22 pool had logit eff-rank ~3, so any base in that subspace
absorbed regardless of pool size. K=10 forward-greedy pool also has
~3 effective directions, but with less redundancy in the directions it
*does* have. If GRU passes at K=10+1, the dense pool was hiding signal.

Decision rule:
  * K=10+1 plain LR-meta delta >= +0.5 bp -> 4th-direction candidate
  * 0..+0.5 bp -> ambiguous; rerun field-state and H9 to triangulate
  * < 0 bp -> rank-lock is real, pool-size-independent

The other 3 NULL candidates (field-state, H9 transductive, combined
lead/lag) need their producer scripts re-run to regenerate OOF arrays.
Deferred to follow-on if GRU passes.

Cost: ~2 min CPU.
Outputs scripts/artifacts/probe_exp1_gru_retest.json.
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
SEED, N_FOLDS, MAX_ITER = 42, 5, 500
GRU_OOF = Path("kernels/d16-gru-sequence-gpu/output/oof_d16_gru_seq_strat.npy")

K10_FWD = [
    "d17_h1d_yekenot_full", "p1_single_cb_v4_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def main():
    t0 = time.time()
    print("Loading K=10 OOFs + GRU + labels ...")
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K10_FWD]
    gru = _pos(GRU_OOF)
    print(f"  GRU shape={gru.shape}, train rows={len(y)}")
    assert len(gru) == len(y), f"GRU OOF len mismatch: {len(gru)} vs {len(y)}"

    auc_gru = float(roc_auc_score(y, gru))
    print(f"  GRU standalone OOF AUC: {auc_gru:.5f}")

    # K=10 plain LR-meta (the baseline to beat)
    F_K10 = expand(np.column_stack(base_oofs))
    F_K11 = expand(np.column_stack(base_oofs + [gru]))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    def fit(F):
        oof = np.zeros(len(y))
        for tr, va in splits:
            lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            oof[va] = lr.predict_proba(F[va])[:, 1]
        return oof, float(roc_auc_score(y, oof))

    print("\nFitting K=10 plain LR-meta ...")
    oof_K10, auc_K10 = fit(F_K10)
    print(f"  K=10 plain LR-meta OOF: {auc_K10:.5f}")

    print("Fitting K=10+GRU plain LR-meta ...")
    oof_K11, auc_K11 = fit(F_K11)
    print(f"  K=10+GRU plain LR-meta OOF: {auc_K11:.5f}")

    delta_bp = (auc_K11 - auc_K10) * 1e4
    print(f"\n=== EXP-1 verdict ===")
    print(f"  Δ at K=10+1 (GRU): {delta_bp:+.3f} bp")
    if delta_bp >= 0.5:
        verdict = "PASS — 4th-direction candidate"
    elif delta_bp >= -0.1:
        verdict = "AMBIGUOUS — rerun field-state and H9 to triangulate"
    else:
        verdict = "NULL — rank-lock is real and pool-size-independent"
    print(f"  Verdict: {verdict}")

    # Also report ρ structure
    rho_K10_K11 = float(spearmanr(oof_K10, oof_K11)[0])
    rho_gru_K10 = float(spearmanr(gru, oof_K10)[0])
    print(f"\n  ρ(K=10 OOF, K=10+GRU OOF): {rho_K10_K11:.6f}")
    print(f"  ρ(GRU standalone, K=10 OOF): {rho_gru_K10:.6f}")

    # Bonus: K=10 + GRU + Path-B Compound x Stint, tau=100k.
    test = pd.read_csv("data/test.csv")
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp_map = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp_map).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_train = c_tr * 6 + s_tr
    MIN_ROWS = 1000

    def fit_lr_aug(F, y):
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F, y)
        return np.concatenate([lr.intercept_, lr.coef_.ravel()])

    print("\nPath-B C×S τ=100k for K=10+GRU ...")
    oof_pb = np.zeros(len(y))
    for tr_idx, va_idx in splits:
        w_global = fit_lr_aug(F_K11[tr_idx], y[tr_idx])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr_idx] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr_idx][idx])) < 2:
                continue
            W_local[s] = fit_lr_aug(F_K11[tr_idx][idx], y[tr_idx][idx])
            mask[s] = True
        n_local = counts.astype(np.float64)
        alpha = n_local / (n_local + 100000.0)
        W_shrunk = (alpha[:, None] * W_local +
                    (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg_train[va_idx]):
            idx = np.where(seg_train[va_idx] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            F_aug = np.column_stack([np.ones(len(idx)),
                                     F_K11[va_idx][idx]])
            oof_pb[va_idx[idx]] = 1.0 / (1.0 + np.exp(
                -np.clip(F_aug @ w, -30, 30)))
    auc_pb = float(roc_auc_score(y, oof_pb))
    print(f"  K=10+GRU Path-B OOF: {auc_pb:.5f}  Δ vs K=10 plain: "
          f"{(auc_pb - auc_K10) * 1e4:+.2f} bp")

    out = {
        "candidate": "d16_gru_seq",
        "candidate_oof_auc": auc_gru,
        "K10_plain_oof": auc_K10,
        "K10_plus_gru_plain_oof": auc_K11,
        "delta_K10_plus_gru_plain_bp": float(delta_bp),
        "K10_plus_gru_path_b_oof": auc_pb,
        "rho_K10_vs_K10_plus_gru": rho_K10_K11,
        "rho_gru_vs_K10": rho_gru_K10,
        "verdict": verdict,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_exp1_gru_retest.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote scripts/artifacts/probe_exp1_gru_retest.json. "
          f"Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
