"""qAP — final gate including qAO. Find the very best K=N combo."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
ART = ROOT / "scripts/artifacts"
SEED, N_FOLDS = 42, 5


def load(name):
    o = np.load(ART / name)
    return o[:, 1] if o.ndim == 2 else o


def expand(p_list):
    cols = []
    for p in p_list:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        cols += [p, pd.Series(p).rank().values / len(p), np.log(p / (1 - p))]
    return np.column_stack(cols)


def lr_meta_oof(Xm, y_):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    om = np.zeros(len(y_))
    for tr, va in skf.split(Xm, y_):
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        m.fit(Xm[tr], y_[tr])
        om[va] = m.predict_proba(Xm[va])[:, 1]
    return om


def main():
    y = pd.read_csv(ROOT / "data/train.csv")["PitNextLap"].values
    BASES = {
        "yekenot":   "oof_d17_h1d_yekenot_full_strat.npy",
        "cb_v4":     "oof_p1_single_cb_v4_gpu_strat.npy",
        "hgbc_deep": "oof_f1_hgbc_deep_strat.npy",
        "d16_orig":  "oof_d16_orig_continuous_only_strat.npy",
    }
    base_oofs = [load(v) for v in BASES.values()]

    EXTRAS = {
        "qAA":   load("dgp_v3_qAA_stint_imputed_oof.npy"),
        "qAB":   load("dgp_v3_qAB_orig_cell_oof.npy"),
        "qAC":   load("dgp_v3_qAC_joint_oof.npy"),
        "d18g":  load("oof_d18_g_mode_id_strat.npy"),
        "qAF":   load("dgp_v3_qAF_d16plus_oof.npy"),
        "qAJ":   load("dgp_v3_qAJ_stint_orig_drv_oof.npy"),
        "qAK":   load("dgp_v3_qAK_knn3_oof.npy"),
        "qAO":   load("dgp_v3_qAO_knn_multi_oof.npy"),
    }
    print(f"Loaded extras: {list(EXTRAS.keys())}")

    K4_oof = lr_meta_oof(expand(base_oofs), y)
    K4_auc = float(roc_auc_score(y, K4_oof))
    print(f"\nK=4 LR-meta: {K4_auc:.5f}")

    # Pairwise rho between extras
    print("\nPairwise rho on OOF:")
    keys = list(EXTRAS.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            rho = float(spearmanr(EXTRAS[k1], EXTRAS[k2]).correlation)
            if abs(rho) > 0.5:
                print(f"  {k1:>5s} vs {k2:>5s}: ρ={rho:.4f}")

    # Single adds
    print("\nSingle adds (K=5):")
    for ek in keys:
        xm = expand(base_oofs + [EXTRAS[ek]])
        auc = float(roc_auc_score(y, lr_meta_oof(xm, y)))
        print(f"  K=5  {ek:<6s} OOF={auc:.5f} Δ={(auc-K4_auc)*1e4:+.3f} bp")

    # qAO + 1, 2, 3 others (qAK is the original; check if qAO+others beats qAK+others)
    print("\nqAO + other(s):")
    others = [k for k in keys if k != "qAO"]
    for ek in others:
        xm = expand(base_oofs + [EXTRAS["qAO"], EXTRAS[ek]])
        auc = float(roc_auc_score(y, lr_meta_oof(xm, y)))
        print(f"  K=6  qAO+{ek:<5s} OOF={auc:.5f} Δ={(auc-K4_auc)*1e4:+.3f} bp")

    # qAK+qAO together
    print("\nqAK + qAO + other(s):")
    others = [k for k in keys if k not in ("qAK", "qAO")]
    for ek in others:
        xm = expand(base_oofs + [EXTRAS["qAK"], EXTRAS["qAO"], EXTRAS[ek]])
        auc = float(roc_auc_score(y, lr_meta_oof(xm, y)))
        print(f"  K=7  qAK+qAO+{ek:<5s} OOF={auc:.5f} Δ={(auc-K4_auc)*1e4:+.3f} bp")

    # Top combos exhaustive 4-add
    print("\nTop 4-adds incl qAK/qAO:")
    candidates = ["qAK", "qAO", "qAA", "qAF", "qAB", "qAC", "d18g"]
    best_combos = []
    for r in range(1, len(candidates) + 1):
        for c in combinations(candidates, r):
            xm = expand(base_oofs + [EXTRAS[k] for k in c])
            auc = float(roc_auc_score(y, lr_meta_oof(xm, y)))
            best_combos.append((c, auc, (auc-K4_auc)*1e4))

    best_combos.sort(key=lambda x: -x[2])
    print("\nTOP 10 BY OOF Δ:")
    for c, auc, delta in best_combos[:10]:
        print(f"  K={4+len(c):2d}  {'+'.join(c):<35s} OOF={auc:.5f} Δ={delta:+.3f} bp")

    out = {"K4_anchor": K4_auc, "top_combos": [
        {"combo": list(c), "K": 4+len(c), "auc": auc, "lift_bp": delta}
        for c, auc, delta in best_combos[:30]
    ]}
    fp = ART / "dgp_v3_qAP_final_gate.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {fp.name}")


if __name__ == "__main__":
    main()
