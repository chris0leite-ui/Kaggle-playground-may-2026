"""Strategy-critic sections 1-3: per-segment failure map, calibration,
disagreement localization. Output JSON for downstream audit MD.

Triggered by 4-null plateau on 2026-05-19 (R11-B, R11-C, R12-1,
R15 Phase 4 cb_focal_weighted).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ART = Path("scripts/artifacts")
PRIMARY = "oof_K17_xendcg_pathb_dcs_tau100000.npy"


def ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        total += (mask.sum() / len(p)) * abs(conf - acc)
    return float(total)


def seg_auc(y, p, grp, label):
    rows = []
    levels = sorted(pd.unique(grp))
    for lev in levels:
        m = grp == lev
        n = int(m.sum())
        if n < 200:
            continue
        if y[m].sum() == 0 or y[m].sum() == n:
            continue
        try:
            a = float(roc_auc_score(y[m], p[m]))
        except Exception:
            continue
        rows.append((label, str(lev), n, a))
    return rows


def main():
    train = pd.read_csv("data/train.csv")
    y = train["PitNextLap"].astype(int).values
    p = np.load(ART / PRIMARY)
    print(f"PRIMARY OOF AUC: {roc_auc_score(y, p):.5f}")
    print(f"len={len(y)}, prior={y.mean():.4f}")

    # ----- Section 1: per-segment failure map ---------------------
    seg_rows = []
    for cat_col in ["Compound", "Stint", "Year", "Race", "Driver"]:
        if cat_col not in train.columns:
            continue
        seg_rows += seg_auc(y, p, train[cat_col].values, cat_col)
    # TyreLife decile
    if "TyreLife" in train.columns:
        tl_dec = pd.qcut(train["TyreLife"].fillna(-1), 10,
                          labels=False, duplicates="drop")
        seg_rows += seg_auc(y, p, tl_dec.values, "TyreLife_dec")
    # LapNumber decile
    if "LapNumber" in train.columns:
        ln_dec = pd.qcut(train["LapNumber"].fillna(-1), 10,
                          labels=False, duplicates="drop")
        seg_rows += seg_auc(y, p, ln_dec.values, "LapNumber_dec")

    mean_auc = float(roc_auc_score(y, p))
    seg_df = pd.DataFrame(seg_rows, columns=["seg", "level", "n", "auc"])
    seg_df["delta_bp"] = (seg_df["auc"] - mean_auc) * 1e4
    seg_df = seg_df.sort_values("delta_bp").reset_index(drop=True)
    print("\n=== BOTTOM 8 SEGMENTS ===")
    print(seg_df.head(8).to_string(index=False))
    print("\n=== TOP 5 SEGMENTS ===")
    print(seg_df.tail(5).to_string(index=False))

    # ----- Section 2: calibration ---------------------------------
    brier = float(brier_score_loss(y, p))
    e = ece(y, p)
    bins = np.linspace(0, 1, 11)
    print(f"\n=== CALIBRATION ===")
    print(f"Brier: {brier:.5f}  (worst-case 0.25)")
    print(f"ECE-15: {e:.5f}")
    rel = []
    idx = np.digitize(p, bins[1:-1])
    for b in range(10):
        m = idx == b
        if m.sum() < 50:
            continue
        rel.append(dict(bin=b, n=int(m.sum()),
                        mean_p=float(p[m].mean()),
                        emp_rate=float(y[m].mean())))
    print("Reliability table:")
    for r in rel:
        print(f"  bin{r['bin']}: n={r['n']:>6}  pred={r['mean_p']:.3f}  "
              f"emp={r['emp_rate']:.3f}  gap={r['mean_p']-r['emp_rate']:+.3f}")

    # ----- Section 3: disagreement on candidate bases -------------
    base_files = {
        "K17_xendcg_pathb": PRIMARY,
        "K13_seghmm_tau100000": "oof_K13_seghmm_pathb_tau100000.npy",
        "K14_dae_compound_stint": "oof_K14_dae_pathb_compound_stint_tau100000.npy",
        "K14_dae_driverclass_stint": "oof_K14_dae_pathb_driverclass_stint_tau100000.npy",
        "K14_seghmm_trf_pathb": "oof_K14_seghmm_trf_pathb_tau100000.npy",
        "R15_cb_focal_weighted": "oof_R15_cb_focal_weighted_strat.npy",
    }
    bases = {}
    for k, f in base_files.items():
        p_path = ART / f
        if p_path.exists():
            arr = np.load(p_path)
            if len(arr) == len(y):
                bases[k] = arr
            else:
                print(f"  skip {k}: len mismatch {len(arr)} vs {len(y)}")
        else:
            print(f"  missing {f}")
    print(f"\n=== DISAGREEMENT ({len(bases)} bases) ===")
    # disagreement = stdev of base preds per row
    if len(bases) >= 2:
        stack = np.stack(list(bases.values()), axis=1)
        disagree = stack.std(axis=1)
        # rows where any pair differs > 0.15
        d_top1 = float(np.quantile(disagree, 0.99))
        n_residual = int((disagree > d_top1).sum())
        print(f"99th percentile row-stdev: {d_top1:.4f}")
        print(f"residual-difficulty rows (>p99 stdev): {n_residual}")
        # AUC on residual-difficulty subset for each base
        m = disagree > d_top1
        print(f"On residual-difficulty subset (n={int(m.sum())}, "
              f"prior={y[m].mean():.3f}):")
        for k, arr in bases.items():
            if y[m].sum() == 0 or y[m].sum() == m.sum():
                continue
            print(f"  {k}: AUC {roc_auc_score(y[m], arr[m]):.4f}")

    # Save JSON summary
    out = dict(
        date="2026-05-19",
        trigger="plateau-4-nulls",
        primary_oof_auc=mean_auc,
        bottom_segments=seg_df.head(8).to_dict(orient="records"),
        top_segments=seg_df.tail(5).to_dict(orient="records"),
        brier=brier,
        ece=e,
        reliability=rel,
        bases_loaded=list(bases.keys()),
        residual_difficulty_n=n_residual if len(bases) >= 2 else None,
        residual_difficulty_p99_stdev=d_top1 if len(bases) >= 2 else None,
    )
    Path("audit/2026-05-19-strategy-critique.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("\nWrote audit/2026-05-19-strategy-critique.json")


if __name__ == "__main__":
    main()
