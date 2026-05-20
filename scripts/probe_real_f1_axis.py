"""Cheap diagnostic: does K=17 PRIMARY underperform on real-F1 drivers
because of cross-stratum miscalibration, or because of within-stratum
information deficit?

Two probes:
  (A) Per-stratum rank-percentile transform (is_real vs synthetic).
      Cheap: if cross-stratum ranks are misaligned, switching to per-
      stratum quantile ranks lifts overall AUC.
  (B) Stack: train a tiny LR meta on [P_K17, P_K17*is_real, is_real]
      using fold-safe TE for Driver target rate. If LR-meta lifts
      overall AUC, axis carries marginal signal.

Both probes preserve fold safety (no label leakage).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def fold_safe_te(driver_arr, y, fold_idx):
    """5-fold safe target-encoding of Driver, smoothed alpha=20.

    fold_idx maps each row to its CV fold (the row's "validation" fold).
    For fold k, the encoding uses ONLY rows whose fold != k.
    """
    te = np.zeros(len(driver_arr), dtype=np.float32)
    for k in range(N_FOLDS):
        is_val = fold_idx == k
        is_tr = ~is_val
        d_tr = driver_arr[is_tr]
        y_tr = y[is_tr]
        global_mean = y_tr.mean()
        df = pd.DataFrame({"d": d_tr, "y": y_tr})
        agg = df.groupby("d")["y"].agg(["sum", "count"])
        alpha = 20.0
        smoothed = (agg["sum"] + alpha * global_mean) / (agg["count"] + alpha)
        mapping = smoothed.to_dict()
        te[is_val] = pd.Series(driver_arr[is_val]).map(mapping).fillna(global_mean).values
    return te


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    p_primary = np.load(ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy")
    driver = train["Driver"].astype(str).values
    is_real = (~pd.Series(driver).str.startswith("D")).values.astype(int)

    print(f"PRIMARY OOF AUC overall:  {roc_auc_score(y, p_primary):.5f}")
    print(f"  Real-F1 (n={is_real.sum()}): "
          f"{roc_auc_score(y[is_real==1], p_primary[is_real==1]):.5f}")
    print(f"  Synth   (n={(1-is_real).sum()}): "
          f"{roc_auc_score(y[is_real==0], p_primary[is_real==0]):.5f}")

    # ----- Probe A: per-stratum rank-percentile alignment ----------
    print("\n=== PROBE A: per-stratum rank realignment ===")
    p_realigned = np.empty_like(p_primary)
    for k in (0, 1):
        m = is_real == k
        ranks = rankdata(p_primary[m])
        p_realigned[m] = ranks / m.sum()
    auc_A = roc_auc_score(y, p_realigned)
    print(f"  Per-stratum-rank AUC:  {auc_A:.5f}  "
          f"(Δ vs PRIMARY: {(auc_A - roc_auc_score(y, p_primary))*1e4:+.2f} bp)")

    # ----- Probe B: LR meta with is_real + Driver_TE ---------------
    print("\n=== PROBE B: LR meta with is_real + Driver TE ===")
    # Build fold index from StratifiedKFold(seed=42)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_idx = np.empty(len(y), dtype=int)
    for k, (_, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_idx[vi] = k
    drv_te = fold_safe_te(driver, y, fold_idx)
    print(f"  Driver TE built (fold-safe). Mean={drv_te.mean():.4f}")

    # Meta features: logit(P), logit(P)*is_real, drv_te, is_real
    p_clip = np.clip(p_primary, 1e-6, 1 - 1e-6)
    L = logit(p_clip)
    X = np.column_stack([L, L * is_real, drv_te, is_real]).astype(np.float64)
    oof_meta = np.zeros(len(y))
    for k in range(N_FOLDS):
        is_val = fold_idx == k
        is_tr = ~is_val
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(X[is_tr], y[is_tr])
        oof_meta[is_val] = lr.predict_proba(X[is_val])[:, 1]
    auc_B = roc_auc_score(y, oof_meta)
    print(f"  LR-meta AUC:          {auc_B:.5f}  "
          f"(Δ vs PRIMARY: {(auc_B - roc_auc_score(y, p_primary))*1e4:+.2f} bp)")

    # Decompose: AUC on real vs synth after meta
    print(f"  Meta on Real-F1:   {roc_auc_score(y[is_real==1], oof_meta[is_real==1]):.5f}")
    print(f"  Meta on Synth:     {roc_auc_score(y[is_real==0], oof_meta[is_real==0]):.5f}")

    # ----- Probe C: just drv_te alone as a feature (sanity) --------
    print("\n=== PROBE C: drv_te alone OOF AUC (sanity) ===")
    auc_C = roc_auc_score(y, drv_te)
    print(f"  Driver-TE-only AUC: {auc_C:.5f}")

    out = dict(
        primary_auc=float(roc_auc_score(y, p_primary)),
        primary_real_auc=float(roc_auc_score(y[is_real==1], p_primary[is_real==1])),
        primary_synth_auc=float(roc_auc_score(y[is_real==0], p_primary[is_real==0])),
        probe_A_per_stratum_rank=float(auc_A),
        probe_B_lr_meta=float(auc_B),
        probe_B_real_auc=float(roc_auc_score(y[is_real==1], oof_meta[is_real==1])),
        probe_B_synth_auc=float(roc_auc_score(y[is_real==0], oof_meta[is_real==0])),
        probe_C_drv_te_only=float(auc_C),
        wall_s=time.time() - t0,
    )
    Path("audit/2026-05-19-real-f1-axis-probe.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nWall: {out['wall_s']:.1f}s")
    print("Wrote audit/2026-05-19-real-f1-axis-probe.json")


if __name__ == "__main__":
    main()
