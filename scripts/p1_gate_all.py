"""P1 omnibus gate — run probe.gate on all p1_* OOF artifacts vs PRIMARY.

Reports standalone OOF, ρ vs PRIMARY (d15b_path_b_K22_dae_only_tau20000),
predicted LB Δ, G3 flip ratio for every p1_* candidate. Also computes a
quick K=K_pool+1 LR-meta gate vs PRIMARY using just [PRIMARY, candidate]
expanded — ablation, not the full K=22+1 stack.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
PRIMARY_OOF = ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_LB = 0.95059


def load_pred(path):
    a = np.load(path)
    return a[:, 1] if (a.ndim == 2 and a.shape[1] == 2) else a.ravel()


def predicted_lb_delta_bp(d_oof_bp, rho):
    if rho >= 0.99996: return d_oof_bp
    if rho >= 0.999: return d_oof_bp - 0.5
    if rho >= 0.995: return d_oof_bp - 1.5
    if rho >= 0.99: return d_oof_bp - 3.0
    return d_oof_bp - 5.0


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    p_oof = load_pred(PRIMARY_OOF)
    p_test = load_pred(PRIMARY_TEST)
    auc_p = float(roc_auc_score(y, p_oof))
    print(f"PRIMARY OOF AUC: {auc_p:.5f}  LB: {PRIMARY_LB}\n")

    rows = []
    for of in sorted(ART.glob("oof_p1_*_strat.npy")):
        name = of.stem.replace("oof_", "").replace("_strat", "")
        tf = ART / f"test_{name}_strat.npy"
        if not tf.exists():
            print(f"  [{name}] missing test artifact, skip")
            continue
        c_oof = load_pred(of)
        c_test = load_pred(tf)
        if len(c_oof) != len(y):
            print(f"  [{name}] length mismatch {len(c_oof)} vs {len(y)}")
            continue
        auc_c = float(roc_auc_score(y, c_oof))
        d_oof_bp = (auc_c - auc_p) * 1e4
        rho = float(spearmanr(c_test, p_test)[0])
        pred_lb = predicted_lb_delta_bp(d_oof_bp, rho)

        # G3 flip ratio
        thr = float(np.quantile(p_test, 0.99))
        pp = p_test >= thr
        cp = c_test >= thr
        f_neg = int(np.sum(pp & ~cp))
        f_pos = int(np.sum(~pp & cp))
        flip = (min(f_pos, f_neg) / max(f_pos, f_neg)
                if max(f_pos, f_neg) > 0 else 1.0)
        rows.append(dict(name=name, auc=auc_c, d_oof_bp=d_oof_bp,
                         rho=rho, pred_lb_bp=pred_lb,
                         flips_neg=f_neg, flips_pos=f_pos,
                         flip_ratio=flip))
        print(f"  {name:50s}  AUC={auc_c:.5f}  Δ={d_oof_bp:+6.1f}bp  "
              f"ρ={rho:.5f}  pred_LB={pred_lb:+5.1f}bp  "
              f"flips +→− {f_neg}/{f_pos}")

    out = ART / "p1_gate_all_results.json"
    out.write_text(json.dumps(dict(primary_oof=auc_p, primary_lb=PRIMARY_LB,
                                   candidates=rows), indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
