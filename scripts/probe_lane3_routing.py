"""scripts/probe_lane3_routing.py — Lane 3 (compound routing/gating).

Tests whether the optimal LR-meta projection from K=4 [P, rank, logit]
varies by Compound, AND whether per-Compound calibration / blending
recovers the rain-row residual (W1) without losing cross-Compound
transfer (the failure mode of A22 specialist replacement).

Key distinction from previously closed probes:
- A4 / Day-18 per-Compound LR specialists were on RAW FEATURES at the
  base level (absorbed by K=10+1).
- A22 rain specialist was a single-LGBM REPLACEMENT on rain rows
  (lost cross-Compound transfer, −152 bp).
- This script tests per-Compound projections AT THE META LEVEL on K=4
  predictions, retaining full-pool transfer.

Probes:
  D3.1 — per-Compound K=4 standalone AUC and forward-greedy K=1 winner
  D3.2 — per-Compound calibration (ECE) of K=4 PRIMARY
  P3.1 — per-Compound LR meta heads (5 LRs, routed at inference)
  P3.2 — per-Compound flat isotonic recalibration of PRIMARY
  P3.3 — rain-row meta blend (refit K=4 meta on rain only; blend at infer)

Cost (CPU): ~30 min combined. Outputs:
  scripts/artifacts/probe_lane3_routing.json
  scripts/artifacts/oof_lane3_per_compound_lr_strat.npy   (P3.1)
  scripts/artifacts/oof_lane3_per_compound_iso_strat.npy  (P3.2)
  scripts/artifacts/oof_lane3_rain_blend_strat.npy        (P3.3)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 500

K4_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def main():
    t0 = time.time()
    print("Loading data + K=4 base OOFs ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    compound_tr = train["Compound"].values
    compound_te = test["Compound"].values
    compounds = sorted(np.unique(compound_tr))

    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4_FWD]
    base_tests = [_pos(ART / f"test_{n}_strat.npy") for n in K4_FWD]
    P4 = np.column_stack(base_oofs)
    P4_test = np.column_stack(base_tests)

    primary_path = ART / "oof_K4_fwd_pathb_strat.npy"
    primary_test_path = ART / "test_K4_fwd_pathb_strat.npy"
    if primary_path.exists():
        primary_oof = _pos(primary_path)
        primary_test = _pos(primary_test_path)
        primary_kind = "K4_fwd_pathb"
    else:
        print("  (PRIMARY composite OOF not on disk; using plain K=4 LR meta substitute)")
        F4_tmp = _expand(P4)
        F4_test_tmp = _expand(P4_test)
        primary_oof = np.zeros(len(y))
        primary_test = np.zeros(F4_test_tmp.shape[0])
        for tr, va in StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                       random_state=SEED).split(np.zeros(len(y)), y):
            lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F4_tmp[tr], y[tr])
            primary_oof[va] = lr.predict_proba(F4_tmp[va])[:, 1]
            primary_test += lr.predict_proba(F4_test_tmp)[:, 1] / N_FOLDS
        primary_kind = "K4_LR_meta_substitute"

    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED).split(np.zeros(len(y)), y))

    # ============ D3.1 — per-Compound K=4 standalone AUC ============
    print("\n--- D3.1: per-Compound K=4 standalone AUC + best single base")
    d31_rows = []
    for c in compounds:
        m = compound_tr == c
        if m.sum() < 100 or y[m].min() == y[m].max():
            continue
        per_base_auc = {K4_FWD[i]: float(roc_auc_score(y[m], P4[m, i]))
                        for i in range(P4.shape[1])}
        primary_auc_in = float(roc_auc_score(y[m], primary_oof[m]))
        best_base = max(per_base_auc, key=per_base_auc.get)
        d31_rows.append({
            "Compound": c,
            "n": int(m.sum()),
            "p_pit": float(y[m].mean()),
            "PRIMARY_auc": primary_auc_in,
            "best_single_base": best_base,
            "best_single_auc": per_base_auc[best_base],
            "per_base_auc": per_base_auc,
        })
    for r in d31_rows:
        print(f"  {r['Compound']:>14s}  n={r['n']:>7d}  p_pit={r['p_pit']:.3f}  "
              f"PRIMARY_AUC={r['PRIMARY_auc']:.4f}  "
              f"best_single={r['best_single_base']} ({r['best_single_auc']:.4f})")

    # ============ D3.2 — per-Compound calibration (ECE) =============
    print("\n--- D3.2: per-Compound K=4 PRIMARY calibration")
    d32_rows = []
    for c in compounds:
        m = compound_tr == c
        if m.sum() < 100:
            continue
        # 10-bin ECE
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.searchsorted(bins, primary_oof[m]) - 1, 0, 9)
        ece = 0.0
        for b in range(10):
            bm = idx == b
            if bm.sum() < 5:
                continue
            ece += abs(primary_oof[m][bm].mean() - y[m][bm].mean()) * (bm.sum() / m.sum())
        d32_rows.append({
            "Compound": c,
            "n": int(m.sum()),
            "ECE_10bin": float(ece),
            "p_mean": float(primary_oof[m].mean()),
            "y_mean": float(y[m].mean()),
        })
    for r in d32_rows:
        print(f"  {r['Compound']:>14s}  ECE={r['ECE_10bin']:.5f}  "
              f"p̂_mean={r['p_mean']:.4f}  ȳ_mean={r['y_mean']:.4f}")

    # ============ P3.1 — per-Compound LR meta heads =================
    print("\n--- P3.1: per-Compound LR meta heads (routed at inference)")
    F4 = _expand(P4)
    F4_test = _expand(P4_test)
    oof_pcm = np.zeros(len(y))
    test_pcm = np.zeros(F4_test.shape[0])
    for tr, va in splits:
        for c in compounds:
            tr_c = tr[(compound_tr[tr] == c)]
            va_c = va[(compound_tr[va] == c)]
            te_c = np.where(compound_te == c)[0]
            if len(tr_c) < 200:
                # Fall back to global
                lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
                lr.fit(F4[tr], y[tr])
            else:
                lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
                lr.fit(F4[tr_c], y[tr_c])
            if len(va_c) > 0:
                oof_pcm[va_c] = lr.predict_proba(F4[va_c])[:, 1]
            if len(te_c) > 0:
                test_pcm[te_c] += lr.predict_proba(F4_test[te_c])[:, 1] / N_FOLDS

    # global LR for baseline comparison
    oof_g = np.zeros(len(y))
    test_g = np.zeros(F4_test.shape[0])
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F4[tr], y[tr])
        oof_g[va] = lr.predict_proba(F4[va])[:, 1]
        test_g += lr.predict_proba(F4_test)[:, 1] / N_FOLDS

    auc_global = float(roc_auc_score(y, oof_g))
    auc_pcm = float(roc_auc_score(y, oof_pcm))
    delta_p31_bp = (auc_pcm - auc_global) * 1e4
    print(f"  Global LR meta on K=4 [P,rank,logit] : {auc_global:.5f}")
    print(f"  Per-Compound LR meta heads           : {auc_pcm:.5f}  "
          f"(Δ {delta_p31_bp:+.3f} bp)")
    np.save(ART / "oof_lane3_per_compound_lr_strat.npy", oof_pcm)
    np.save(ART / "test_lane3_per_compound_lr_strat.npy",
            np.column_stack([1 - test_pcm, test_pcm]))

    # ============ P3.2 — per-Compound flat isotonic =================
    print("\n--- P3.2: per-Compound isotonic recalibration of PRIMARY")
    fold_assign = np.zeros(len(y), dtype=int)
    for fold, (_, va) in enumerate(splits):
        fold_assign[va] = fold
    primary_iso = primary_oof.copy()
    for c in compounds:
        cm = compound_tr == c
        if cm.sum() < 200 or y[cm].min() == y[cm].max():
            continue
        for fold in range(N_FOLDS):
            tr_m = cm & (fold_assign != fold)
            va_m = cm & (fold_assign == fold)
            if tr_m.sum() < 100 or va_m.sum() < 1:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(primary_oof[tr_m], y[tr_m])
            primary_iso[va_m] = iso.transform(primary_oof[va_m])
    auc_primary = float(roc_auc_score(y, primary_oof))
    auc_iso = float(roc_auc_score(y, primary_iso))
    delta_p32_bp = (auc_iso - auc_primary) * 1e4
    print(f"  PRIMARY plain         : {auc_primary:.5f}")
    print(f"  PRIMARY + per-Compound: {auc_iso:.5f}  (Δ {delta_p32_bp:+.3f} bp)")
    np.save(ART / "oof_lane3_per_compound_iso_strat.npy", primary_iso)

    # ============ P3.3 — rain-row meta blend ========================
    print("\n--- P3.3: rain-row meta blend (refit K=4 meta on INTER+WET only)")
    rain_tr_mask = np.isin(compound_tr, ["INTERMEDIATE", "WET"])
    rain_te_mask = np.isin(compound_te, ["INTERMEDIATE", "WET"])
    print(f"  rain rows: train n={rain_tr_mask.sum()}, test n={rain_te_mask.sum()}")
    # Refit K=4 LR meta on rain rows only, fold-safe
    oof_rain_meta = np.zeros(len(y))
    test_rain_meta = np.zeros(F4_test.shape[0])
    for tr, va in splits:
        rain_tr = tr[rain_tr_mask[tr]]
        if len(rain_tr) < 100:
            continue
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F4[rain_tr], y[rain_tr])
        oof_rain_meta[va] = lr.predict_proba(F4[va])[:, 1]
        test_rain_meta += lr.predict_proba(F4_test)[:, 1] / N_FOLDS
    # Blend: primary + 0.5(rain_meta - primary) on rain rows
    blend_oof = primary_oof.copy()
    blend_oof[rain_tr_mask] = 0.5 * primary_oof[rain_tr_mask] + 0.5 * oof_rain_meta[rain_tr_mask]
    blend_test = primary_test.copy()
    blend_test[rain_te_mask] = 0.5 * primary_test[rain_te_mask] + 0.5 * test_rain_meta[rain_te_mask]
    auc_blend = float(roc_auc_score(y, blend_oof))
    delta_p33_bp = (auc_blend - auc_primary) * 1e4
    # Within-rain delta
    auc_rain_primary = float(roc_auc_score(y[rain_tr_mask], primary_oof[rain_tr_mask])) \
        if y[rain_tr_mask].min() != y[rain_tr_mask].max() else float("nan")
    auc_rain_blend = float(roc_auc_score(y[rain_tr_mask], blend_oof[rain_tr_mask])) \
        if y[rain_tr_mask].min() != y[rain_tr_mask].max() else float("nan")
    print(f"  PRIMARY plain (global) : {auc_primary:.5f}")
    print(f"  Rain-blend (global)    : {auc_blend:.5f}  (Δ {delta_p33_bp:+.3f} bp)")
    print(f"  Within-rain AUC: PRIMARY={auc_rain_primary:.5f} → blend={auc_rain_blend:.5f}")
    np.save(ART / "oof_lane3_rain_blend_strat.npy", blend_oof)
    np.save(ART / "test_lane3_rain_blend_strat.npy",
            np.column_stack([1 - blend_test, blend_test]))

    rho_p31 = float(spearmanr(oof_pcm, primary_oof)[0])
    rho_p33 = float(spearmanr(blend_oof, primary_oof)[0])

    out = {
        "K4_bases": K4_FWD,
        "compounds": compounds,
        "D3_1_per_compound_standalone": d31_rows,
        "D3_2_per_compound_calibration": d32_rows,
        "P3_1_global_LR_meta_oof": auc_global,
        "P3_1_per_compound_LR_oof": auc_pcm,
        "P3_1_delta_bp": float(delta_p31_bp),
        "P3_1_rho_vs_primary": rho_p31,
        "P3_2_PRIMARY_oof": auc_primary,
        "P3_2_per_compound_iso_oof": auc_iso,
        "P3_2_delta_bp": float(delta_p32_bp),
        "P3_3_blend_oof_global": auc_blend,
        "P3_3_delta_bp_global": float(delta_p33_bp),
        "P3_3_within_rain_PRIMARY": auc_rain_primary,
        "P3_3_within_rain_blend": auc_rain_blend,
        "P3_3_rho_vs_primary": rho_p33,
        "verdict_P3_1": ("PASS" if delta_p31_bp >= 0.5
                         else "AMBIG" if delta_p31_bp >= -0.1
                         else "NULL"),
        "verdict_P3_2": ("PASS" if delta_p32_bp >= 0.2
                         else "AMBIG" if delta_p32_bp >= -0.1
                         else "NULL"),
        "verdict_P3_3": ("PASS" if delta_p33_bp >= 0.2
                         else "AMBIG" if delta_p33_bp >= -0.1
                         else "NULL"),
        "wall_s": time.time() - t0,
    }
    (ART / "probe_lane3_routing.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {ART/'probe_lane3_routing.json'}. Wall {out['wall_s']:.1f}s")
    print(f"Verdicts: P3.1 {out['verdict_P3_1']} | P3.2 {out['verdict_P3_2']} "
          f"| P3.3 {out['verdict_P3_3']}")


if __name__ == "__main__":
    main()
