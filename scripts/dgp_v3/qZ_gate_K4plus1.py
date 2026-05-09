"""qZ gate — K=4+1 LR-meta probe with d16++ added.

Tests if qZ (d16++ trained on orig with cell-key features) adds lift
to the existing K=4 PRIMARY ensemble at the LR-meta level.

K=4 bases:
  - d17_h1d_yekenot_full   (RealMLP yekenot recipe)
  - p1_single_cb_v4_gpu    (CatBoost yekenot)
  - f1_hgbc_deep           (HistGradientBoosting deep)
  - d16_orig_continuous_only (LightGBM on orig)

K=4+1 = K=4 + qZ d16++.

Compute:
  - K=4 OOF AUC
  - K=4+1 OOF AUC
  - Δ in bp
  - Spearman correlation between qZ and each K=4 base + with K=4 LR-meta
    output

Output: scripts/artifacts/dgp_v3_qZ_gate_K4plus1.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main() -> None:
    out: dict = {}
    ts = time.time()

    train = pd.read_csv(DATA / "train.csv")
    y = train["PitNextLap"].values
    t(f"train {train.shape}", ts)

    K4_files = {
        "h1d_yekenot": "oof_d17_h1d_yekenot_full_strat.npy",
        "p1_cb_v4":    "oof_p1_single_cb_v4_gpu_strat.npy",
        "f1_hgbc":     "oof_f1_hgbc_deep_strat.npy",
        "d16_orig":    "oof_d16_orig_continuous_only_strat.npy",
    }
    K4_oof = {}
    for name, fn in K4_files.items():
        try:
            arr = np.load(ART / fn)
            # If 2D, take positive class column (col 1)
            if arr.ndim == 2 and arr.shape[1] == 2:
                arr = arr[:, 1]
            elif arr.ndim == 2:
                arr = arr[:, -1]
            K4_oof[name] = arr
            t(f"loaded {name}: shape {arr.shape}", ts)
        except FileNotFoundError:
            print(f"  MISSING {fn} - aborting")
            return

    # qZ on synth-train
    qZ_train = np.load(ART / "dgp_v3_qZ_train_synth.npy")
    t(f"loaded qZ train: shape {qZ_train.shape}", ts)

    # If shapes match
    n = len(y)
    for name, oof in K4_oof.items():
        if len(oof) != n:
            print(f"  WARNING: {name} OOF length {len(oof)} != train {n}")
    if len(qZ_train) != n:
        print(f"  WARNING: qZ length {len(qZ_train)} != train {n}")

    # Compute Spearman correlations
    out["spearman_qZ_vs_bases"] = {}
    for name, oof in K4_oof.items():
        s, _ = spearmanr(qZ_train, oof)
        out["spearman_qZ_vs_bases"][name] = float(s)
        t(f"spearman qZ vs {name}: {s:.4f}", ts)

    # K=4 LR-meta on [P, rank, logit]
    def build_features(oof_dict, qZ=None):
        cols = []
        for name, oof in oof_dict.items():
            cols.append(oof)
            cols.append(pd.Series(oof).rank(pct=True).values)
            cols.append(safe_logit(oof))
        if qZ is not None:
            cols.append(qZ)
            cols.append(pd.Series(qZ).rank(pct=True).values)
            cols.append(safe_logit(qZ))
        return np.column_stack(cols)

    X_K4 = build_features(K4_oof)
    X_K5 = build_features(K4_oof, qZ_train)
    t(f"K=4 features: {X_K4.shape}; K=5 features: {X_K5.shape}", ts)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_K4 = np.zeros(n)
    oof_K5 = np.zeros(n)
    for tr, va in skf.split(X_K4, y):
        m4 = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
        m4.fit(X_K4[tr], y[tr])
        oof_K4[va] = m4.predict_proba(X_K4[va])[:, 1]
        m5 = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
        m5.fit(X_K5[tr], y[tr])
        oof_K5[va] = m5.predict_proba(X_K5[va])[:, 1]

    auc_K4 = float(roc_auc_score(y, oof_K4))
    auc_K5 = float(roc_auc_score(y, oof_K5))
    delta_bp = (auc_K5 - auc_K4) * 1e4
    out["auc_K4"] = auc_K4
    out["auc_K5"] = auc_K5
    out["delta_bp"] = delta_bp
    t(f"K=4 LR-meta OOF AUC: {auc_K4:.5f}", ts)
    t(f"K=4+qZ LR-meta OOF AUC: {auc_K5:.5f}", ts)
    t(f"Δ vs K=4: {delta_bp:+.3f} bp", ts)

    # Spearman of qZ vs K=4 LR-meta output
    s, _ = spearmanr(qZ_train, oof_K4)
    out["spearman_qZ_vs_K4meta"] = float(s)
    t(f"spearman qZ vs K=4 LR-meta: {s:.4f}", ts)

    fp = ART / "dgp_v3_qZ_gate_K4plus1.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qZ K=4+1 LR-meta gate ===")
    print(f"  K=4 LR-meta OOF AUC:    {auc_K4:.5f}")
    print(f"  K=4+qZ LR-meta OOF AUC: {auc_K5:.5f}")
    print(f"  Δ vs K=4:                {delta_bp:+.3f} bp")
    print(f"\n  ρ qZ vs K=4 LR-meta: {out['spearman_qZ_vs_K4meta']:.4f}")
    print(f"\n  ρ qZ vs each K=4 base:")
    for name, s in out["spearman_qZ_vs_bases"].items():
        print(f"    {name:15s}: {s:.4f}")


if __name__ == "__main__":
    main()
