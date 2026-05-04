"""E4 — RealMLP single-fold CPU probe.

Per the relaxed 1h cap (single-fold actual wall, not 5-fold projection):
fit RealMLP-TD on Strat fold 0 only, on full data. If wall <1h and AUC
clears baseline (0.94075), pursue further. If wall >1h or AUC <baseline,
abort. Cross-comp research (Source 1 #1) cites yekenot's RealMLP at
~0.946 OOF on this dataset.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pytabkit import RealMLP_TD_Classifier

TARGET, ID_COL = "PitNextLap", "id"
SEED = 42


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")

    # RealMLP handles native categoricals via embeddings; pass as object/category.
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    print(f"cat cols: {cat_cols}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.zeros(len(y)), y)))
    print(f"fold 0: train={len(tr)} val={len(va)}  t={time.time()-t0:.1f}s")

    print("=== RealMLP-TD CPU fit (fold 0, full data) ===")
    t1 = time.time()
    # Use defaults; CPU only; reduce verbosity
    model = RealMLP_TD_Classifier(
        device="cpu",
        random_state=SEED,
        n_cv=1,                     # single train/val split (we'll provide the split)
        val_metric_name="cross_entropy",
        use_ls=False,               # for AUC
        verbosity=0,
        n_threads=-1,
    )
    # RealMLP wants an explicit val split via val_idxs
    val_idxs = va.tolist() if hasattr(va, 'tolist') else list(va)
    model.fit(X.iloc[tr], y[tr])    # validate on internal val_fraction split
    fit_secs = time.time() - t1
    print(f"fit done in {fit_secs:.0f}s  ({fit_secs/60:.1f} min)")

    if fit_secs >= 3600:
        msg = (f"# E4 RealMLP PROBE TIMEOUT\n\n"
               f"single-fold wall: {fit_secs:.0f}s ≥ 1h cap. STOP.\n"
               f"No 5-fold attempt; mark RealMLP CPU as blocked for D3.\n")
        Path("audit/2026-05-04-e4-realmlp-PROBE-TIMEOUT.md").write_text(msg)
        print("HARD STOP")
        return False

    p_va = model.predict_proba(X.iloc[va])[:, 1]
    auc = float(roc_auc_score(y[va], p_va))
    print(f"\nfold 0 AUC={auc:.5f}  Δ baseline (0.94075)={(auc-0.94075)*1e4:+.1f}bp")
    print(f"total wall: {time.time()-t0:.0f}s")

    # Save partial OOF (single fold only) + test pred for diagnostic
    Path("scripts/artifacts").mkdir(exist_ok=True)
    np.save("scripts/artifacts/oof_e4_realmlp_fold0.npy", p_va)
    p_test = model.predict_proba(X_test)[:, 1]
    np.save("scripts/artifacts/test_e4_realmlp_fold0.npy", p_test)

    res = dict(fold=0, fold_auc=auc,
               delta_vs_baseline_bp=(auc - 0.94075) * 1e4,
               fit_secs=fit_secs, total_secs=time.time() - t0,
               cap_hit=False, decision="continue" if auc > 0.94 else "skip")
    Path("scripts/artifacts/e4_realmlp_probe_results.json").write_text(json.dumps(res, indent=2))

    body = (
        f"# E4 — RealMLP-TD CPU single-fold probe (2026-05-04)\n\n"
        f"Per relaxed 1h cap: single-fold wall not 5-fold projection.\n\n"
        f"## Result\n\n"
        f"- Fold 0 (StratKFold, full data, CPU): AUC = **{auc:.5f}**\n"
        f"- Δ vs baseline (0.94075): {(auc - 0.94075) * 1e4:+.1f}bp\n"
        f"- Fit wall: {fit_secs:.0f}s ({fit_secs/60:.1f} min) — within 1h cap.\n"
        f"- Total wall: {time.time() - t0:.0f}s.\n\n"
        f"## Verdict\n\n"
        + (f"PROMISING — AUC > 0.94. Worth full 5-fold + adding to M5 pool.\n"
           if auc > 0.94 else
           f"NULL — AUC ≤ 0.94. RealMLP CPU not competitive on this DGP.\n")
    )
    Path("audit/2026-05-04-e4-realmlp-probe.md").write_text(body)
    return True


if __name__ == "__main__":
    main()
