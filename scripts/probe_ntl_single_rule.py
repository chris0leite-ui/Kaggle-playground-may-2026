"""scripts/probe_ntl_single_rule.py — Probe #3 hiding-in-plain-sight.

Hypothesis: host quote in brief.md says Normalized_TyreLife "makes the
prediction trivial". Two NTL reconstructions exist in the codebase but
neither has been measured as a single-feature standalone OOF baseline.

Probes:
  R1.  compound_tyre_norm = TyreLife / COMPOUND_MAX_LIFE_MAP[Compound]
  R2.  thresholded rules: 1[ctn > theta] for theta in {0.7, 0.8, 0.85, 0.9}
  R3.  NTL_stint   = TyreLife / max(TyreLife) within (Driver,Race,Year,Stint)
  R4.  NTL_global  = TyreLife / max(TyreLife) within (Compound, Year)
  R5.  NTL_compound_p99 = TyreLife / p99-TyreLife per Compound (robust)

Runs single-feature 5-fold StratifiedKFold OOF AUC. No model training
beyond `roc_auc_score(y, feature)` per fold (which is just a sort).

Output: scripts/artifacts/probe_ntl_single_rule.json (load-bearing).

Decision-time log: written via probe.py bote in driver, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
OUT = ART / "probe_ntl_single_rule.json"
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"

COMPOUND_MAX_LIFE_MAP = {
    "SOFT": 15, "MEDIUM": 30, "HARD": 50,
    "INTERMEDIATE": 25, "WET": 20,
}


def single_feat_oof_auc(y: np.ndarray, x: np.ndarray) -> dict:
    """5-fold Strat OOF AUC of a single feature (rank ⇒ order, no model)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        # OOF AUC of a static feature is identical per fold to the full-data
        # AUC restricted to the val rows (no fitting), but we report per-fold
        # for stability.
        try:
            a = roc_auc_score(y[va_idx], x[va_idx])
        except Exception:
            a = float("nan")
        fold_aucs.append(float(a))
    full_auc = float(roc_auc_score(y, x))
    return {
        "full_auc": full_auc,
        "fold_aucs": fold_aucs,
        "fold_mean": float(np.nanmean(fold_aucs)),
        "fold_std": float(np.nanstd(fold_aucs)),
    }


def main() -> None:
    print("Loading train...")
    tr = pd.read_csv("data/train.csv")
    print(f"  train: {tr.shape}")
    y = tr[TARGET].astype(int).to_numpy()
    print(f"  pos rate: {y.mean():.4f}")

    out: dict = {}

    # R1 compound_tyre_norm
    cml = tr["Compound"].map(COMPOUND_MAX_LIFE_MAP).fillna(30).astype(float)
    ctn = (tr["TyreLife"].astype(float) / cml).clip(0, 2).to_numpy()
    out["R1_compound_tyre_norm"] = single_feat_oof_auc(y, ctn)

    # R2 thresholded rules
    for theta in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
        rule = (ctn > theta).astype(np.float32)
        out[f"R2_rule_ctn_gt_{theta:.2f}"] = single_feat_oof_auc(y, rule)

    # R3 NTL_stint = TyreLife / max within stint
    g_stint_max = (tr.groupby(["Driver", "Race", "Year", "Stint"])["TyreLife"]
                     .transform("max").clip(lower=1)).astype(float)
    ntl_stint = (tr["TyreLife"].astype(float) / g_stint_max).to_numpy()
    out["R3_ntl_stint"] = single_feat_oof_auc(y, ntl_stint)
    # threshold at 1.0 (last lap of observed stint window)
    out["R3_rule_ntl_stint_eq_1"] = single_feat_oof_auc(
        y, (ntl_stint >= 0.999).astype(np.float32)
    )

    # R4 NTL_global = TyreLife / max within (Compound, Year)
    g_cy_max = (tr.groupby(["Compound", "Year"])["TyreLife"]
                  .transform("max").clip(lower=1)).astype(float)
    ntl_global = (tr["TyreLife"].astype(float) / g_cy_max).to_numpy()
    out["R4_ntl_compound_year_max"] = single_feat_oof_auc(y, ntl_global)

    # R5 NTL with p99 denominator (robust to outliers)
    p99_by_c = tr.groupby("Compound")["TyreLife"].quantile(0.99)
    cml_p99 = tr["Compound"].map(p99_by_c).clip(lower=1).astype(float)
    ntl_p99 = (tr["TyreLife"].astype(float) / cml_p99).clip(0, 2).to_numpy()
    out["R5_ntl_compound_p99"] = single_feat_oof_auc(y, ntl_p99)
    for theta in [0.70, 0.80, 0.85, 0.90, 0.95]:
        out[f"R5_rule_ntl_p99_gt_{theta:.2f}"] = single_feat_oof_auc(
            y, (ntl_p99 > theta).astype(np.float32)
        )

    # R6 reference single features
    out["REF_TyreLife_only"] = single_feat_oof_auc(
        y, tr["TyreLife"].astype(float).to_numpy()
    )
    out["REF_RaceProgress_only"] = single_feat_oof_auc(
        y, tr["RaceProgress"].astype(float).to_numpy()
    )

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}\n")
    # Pretty top-line summary
    print(f"{'feature':40} {'full_auc':>10} {'fold_std':>10}")
    for k, v in out.items():
        print(f"{k:40} {v['full_auc']:>10.5f} {v['fold_std']:>10.5f}")


if __name__ == "__main__":
    main()
