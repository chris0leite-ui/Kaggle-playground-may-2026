"""qAE — unified ablation gate for qAA / qAB / qAC / d18_g.

Build a K=4 + N matrix on every subset combo of {qAA, qAB, qAC, d18_g}
and fit plain LR-meta with [P, rank, logit] expansion. Report each
combo's K=4+N OOF AUC and the lift in bp.

This characterises the rank-lock pattern across all 4 candidate
extensions and identifies which combos beat +0.5 bp at the meta gate.

Output: scripts/artifacts/dgp_v3_qAE_gate_table.json
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
ART = ROOT / "scripts/artifacts"
SEED, N_FOLDS = 42, 5


def load(name):
    o = np.load(ART / name)
    if o.ndim == 2:
        o = o[:, 1]
    return o


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
        "yekenot":   ("oof_d17_h1d_yekenot_full_strat.npy",   "test_d17_h1d_yekenot_full_strat.npy"),
        "cb_v4":     ("oof_p1_single_cb_v4_gpu_strat.npy",    "test_p1_single_cb_v4_gpu_strat.npy"),
        "hgbc_deep": ("oof_f1_hgbc_deep_strat.npy",            "test_f1_hgbc_deep_strat.npy"),
        "d16_orig":  ("oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    }
    base_oofs = [load(v[0]) for v in BASES.values()]

    EXTRAS = {}
    for nm, fn in [
        ("qAA",   "dgp_v3_qAA_stint_imputed_oof.npy"),
        ("qAB",   "dgp_v3_qAB_orig_cell_oof.npy"),
        ("qAC",   "dgp_v3_qAC_joint_oof.npy"),
        ("d18g",  "oof_d18_g_mode_id_strat.npy"),
    ]:
        p = ART / fn
        if p.exists():
            EXTRAS[nm] = load(fn)
            print(f"  loaded {nm}: shape {EXTRAS[nm].shape}, OOF AUC {roc_auc_score(y, EXTRAS[nm]):.5f}")

    K4_oof = lr_meta_oof(expand(base_oofs), y)
    K4_auc = float(roc_auc_score(y, K4_oof))
    print(f"\n  K=4 plain LR-meta OOF: {K4_auc:.5f}\n")

    out = {"K4_lr_meta": K4_auc, "extras": list(EXTRAS.keys()), "table": []}

    extras_keys = list(EXTRAS.keys())
    for r in range(1, len(extras_keys) + 1):
        for combo in combinations(extras_keys, r):
            extras_oofs = [EXTRAS[n] for n in combo]
            xm = expand(base_oofs + extras_oofs)
            auc = float(roc_auc_score(y, lr_meta_oof(xm, y)))
            delta = (auc - K4_auc) * 1e4
            row = {"combo": "+".join(combo), "K": 4 + r, "auc": auc, "lift_bp": delta}
            out["table"].append(row)
            print(f"  K={4+r:2d}  {'+'.join(combo):<25s} OOF={auc:.5f} Δ={delta:+.3f} bp")

    fp = ART / "dgp_v3_qAE_gate_table.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {fp.name}")


if __name__ == "__main__":
    main()
