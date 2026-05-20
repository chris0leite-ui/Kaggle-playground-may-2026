"""scripts/probe_r19_aleatoric_K20_meta.py — K=20 LR-meta probe gate.

Quick LR-meta gate: K=18 baseline pool (R18-rejected version) vs
K=19 (+R19_mean), K=19 (+R19_sigma), K=20 (+both). Pre-Path-B gate;
if even LR-meta shows nothing, skip the heavy Path-B build.

Mirrors the R18 K=19 probe pool. R19 candidates are loaded from
scripts/artifacts/oof_R19_aleatoric_{mean,sigma}_strat.npy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K18_FILES = [
    ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy"),
    ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy"),
    ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy"),
    ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy"),
    ("qAT",       "dgp_v3_qAT_K1_oof.npy"),
    ("qAV",       "dgp_v3_qAV_K1_7feat_oof.npy"),
    ("qAO",       "dgp_v3_qAO_knn_multi_oof.npy"),
    ("qAA",       "dgp_v3_qAA_stint_imputed_oof.npy"),
    ("qAF",       "dgp_v3_qAF_d16plus_oof.npy"),
    ("qAK",       "dgp_v3_qAK_knn3_oof.npy"),
    ("K27_100k",  "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"),
    ("seg_fe",    "oof_r4_segment_fe_strat.npy"),
    ("HMM",       "oof_r4_hmm_seq_strat.npy"),
    ("R12_horizon",      "oof_R12_cb_horizon_strat.npy"),
    ("R13_stint_compl",  "oof_R13_cb_stint_completion_strat.npy"),
    ("R14_tabm",         "oof_R14_tabm_strat.npy"),
    ("R15_xendcg_seg",   "oof_R15_xendcg_per_seg_strat.npy"),
    ("R17_listwise",     "oof_R17_listwise_race_lap_strat.npy"),
]

CAND_FILES = [
    ("R19_mean",  "oof_R19_aleatoric_mean_strat.npy"),
    ("R19_sigma", "oof_R19_aleatoric_sigma_strat.npy"),
]


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.astype(np.float64).ravel()


def _expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _meta_oof(y, F):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def main():
    t0 = time.time()
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    print(f"== K=20 LR-meta gate (n={len(y)}) ==")

    base_oofs = []
    for name, fn in K18_FILES:
        a = _pos(ART / fn)
        assert len(a) == len(y), f"{name}: len {len(a)} != {len(y)}"
        base_oofs.append(a)
    P_base = np.column_stack(base_oofs)
    F_base = _expand(P_base)
    oof_b, auc_b = _meta_oof(y, F_base)
    print(f"\nK=18 baseline: OOF {auc_b:.6f}")

    cands = []
    for name, fn in CAND_FILES:
        a = _pos(ART / fn)
        assert len(a) == len(y), f"{name}: len {len(a)} != {len(y)}"
        cands.append((name, a))

    runs = {}
    runs["K=18"] = auc_b

    for name, c in cands:
        P_w = np.column_stack([P_base, c])
        F_w = _expand(P_w)
        oof_w, auc_w = _meta_oof(y, F_w)
        d_bp = (auc_w - auc_b) * 1e4
        print(f"\nK=19 (+{name}): OOF {auc_w:.6f}  Δ {d_bp:+.3f} bp")
        runs[f"K=19_+{name}"] = auc_w

    P_w2 = np.column_stack([P_base, cands[0][1], cands[1][1]])
    F_w2 = _expand(P_w2)
    oof_w2, auc_w2 = _meta_oof(y, F_w2)
    d_bp2 = (auc_w2 - auc_b) * 1e4
    print(f"\nK=20 (+R19_mean+R19_sigma): OOF {auc_w2:.6f}  Δ {d_bp2:+.3f} bp")
    runs["K=20_+both"] = auc_w2

    summary = dict(
        baseline_pool=[n for n, _ in K18_FILES],
        candidate_pool=[n for n, _ in CAND_FILES],
        runs=runs, wall_s=time.time() - t0,
    )
    out = ART / "probe_r19_aleatoric_K20_meta.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwall {time.time()-t0:.1f}s → {out}")


if __name__ == "__main__":
    main()
