"""scripts/lr_leverage_phaseC.py — Probes 4 and 5.

Probe 4: Random-Subspace bagged-LR on mega FE — does the bag's eff_rank
         exceed the 2.19 ceiling of 20 hand-designed LR variants?
Probe 5: Per-segment mega LR (Compound × Year) — does per-cell LR
         find local linearizations the global LR misses?

Both probes share mega's FE matrix; we compute it once and re-use.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from p1_features import (
    make_features_static, fit_fs_a, apply_fs_a, cv_target_encode, TE_CONFIGS,
)
from lr_bank_rich_fe import build_dgp_rule_features

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

NUM_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
]
CAT_COLS = ["Driver", "Compound", "Race"]


def _eff_rank_entropy(s):
    s2 = s ** 2
    p = s2 / s2.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def build_mega_FE_matrix(train, test, y):
    """Build the mega FE matrix once. Returns dict with per-fold artifacts."""
    print("  Building mega static FE...", flush=True)
    state2 = {}
    train_S, state2 = make_features_static(train, fit=True, state=state2)
    test_S, _ = make_features_static(test, fit=False, state=state2)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    print("  Computing 6 Rozen TE features...", flush=True)
    rozen_te_oof, rozen_te_test = {}, {}
    for cols, smooth, name in TE_CONFIGS:
        oof_enc, test_enc = cv_target_encode(
            train, test, cols, train[TARGET].astype(int), fold_list, smooth)
        rozen_te_oof[name] = oof_enc
        rozen_te_test[name] = test_enc

    print("  Computing 16 3-way TE features...", flush=True)
    keys_3way = [["Driver", "Race", "Year"], ["Driver", "Race", "Compound"],
                 ["Driver", "Year", "Compound"], ["Race", "Year", "Compound"]]
    smoothings = [1, 5, 20, 100]
    threeway_oof, threeway_test = [], []
    for keys in keys_3way:
        for sm in smoothings:
            oof_enc, test_enc = cv_target_encode(
                train, test, keys, train[TARGET].astype(int), fold_list, sm)
            threeway_oof.append(oof_enc)
            threeway_test.append(test_enc)

    num_tr_raw = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te_raw = test[NUM_COLS].fillna(0).values.astype(np.float32)
    kb = KBinsDiscretizer(n_bins=20, encode="onehot", strategy="quantile",
                          subsample=None)
    kb.fit(np.vstack([num_tr_raw, num_te_raw]))
    Bk_tr = kb.transform(num_tr_raw).toarray().astype(np.float32)
    Bk_te = kb.transform(num_te_raw).toarray().astype(np.float32)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    Oc_tr = enc.transform(train[CAT_COLS]).toarray().astype(np.float32)
    Oc_te = enc.transform(test[CAT_COLS]).toarray().astype(np.float32)

    return dict(
        train_S=train_S, test_S=test_S, fold_list=fold_list,
        rozen_te_oof=rozen_te_oof, rozen_te_test=rozen_te_test,
        threeway_oof=threeway_oof, threeway_test=threeway_test,
        Bk_tr=Bk_tr, Bk_te=Bk_te, Oc_tr=Oc_tr, Oc_te=Oc_te,
    )


def assemble_fold_X(fe, train, test, y, tr, va, skip_test=True):
    """For one fold, assemble the dense mega feature matrix.
    For phase-C probes 4+5: skip cat OHE (918 cols, mass 42% but
    distributed weakly per feature; dropping it cuts X from 1202 to
    ~284 cols — fits in RAM with all 5 fold caches)."""
    fs_a = fit_fs_a(train.iloc[tr])
    train_A = apply_fs_a(fe["train_S"], fs_a)
    drop = ["Driver", "Race", "Compound", "id", TARGET]
    feat_cols = [c for c in train_A.columns if c not in drop
                 and c not in CAT_COLS and train_A[c].dtype.kind in "biufc"]
    Xstatic_tr = train_A[feat_cols].fillna(0).values.astype(np.float32)

    rule_tr, rule_va, _ = build_dgp_rule_features(train, test, y, tr, va)

    te_arr = np.column_stack(list(fe["rozen_te_oof"].values()))
    tw_arr = np.column_stack(fe["threeway_oof"])

    num_tr_full = np.hstack([Xstatic_tr[tr], te_arr[tr], tw_arr[tr], rule_tr])
    num_va_full = np.hstack([Xstatic_tr[va], te_arr[va], tw_arr[va], rule_va])
    sc = StandardScaler()
    num_tr_s = sc.fit_transform(num_tr_full)
    num_va_s = sc.transform(num_va_full)

    # Drop fe["Oc_tr"] (cat OHE) for these phase-C probes — saves 4-5x RAM
    Xtr_tr = np.hstack([num_tr_s, fe["Bk_tr"][tr]])
    Xtr_va = np.hstack([num_va_s, fe["Bk_tr"][va]])
    return Xtr_tr, Xtr_va, None


def probe4_random_subspace_bagged_lr(fe, train, test, y, fold_X_cache,
                                      n_bags=20, subset_frac=0.30):
    """Random-subspace bagged LR: each bag is a fold-CV LR on a random
    subset of mega's columns. Question: does the bag have higher eff_rank
    than the 20 hand-designed LR variants (eff_rank=2.19)?

    Uses fold_X_cache (pre-built) so we don't re-assemble per bag.
    """
    print(f"\n=== Probe 4: Random-Subspace bagged-LR ({n_bags} bags, "
          f"{int(subset_frac*100)}% subset) ===", flush=True)

    rng = np.random.default_rng(SEED)
    n_tr = len(y)
    bag_oofs = np.zeros((n_bags, n_tr), dtype=np.float64)

    fold_list = fe["fold_list"]
    n_cols = fold_X_cache[0]["Xtr"].shape[1]
    n_subset = int(n_cols * subset_frac)
    print(f"  mega has {n_cols} columns; each bag uses {n_subset}", flush=True)

    # For each bag: pick random column subset; run fold-CV using cache
    for b in range(n_bags):
        cols = rng.choice(n_cols, size=n_subset, replace=False)
        oof = np.zeros(n_tr)
        t0 = time.time()
        for k, (tr, va) in enumerate(fold_list):
            Xtr_sub = fold_X_cache[k]["Xtr"][:, cols]
            Xva_sub = fold_X_cache[k]["Xva"][:, cols]
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            lr.fit(Xtr_sub, y[tr])
            oof[va] = lr.predict_proba(Xva_sub)[:, 1]
        auc = float(roc_auc_score(y, oof))
        bag_oofs[b] = oof
        print(f"    bag {b:>2}: subset {n_subset} cols  AUC {auc:.5f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # Aggregate: rank-average
    from scipy.stats import rankdata
    bag_ranks = np.array([rankdata(o) / n_tr for o in bag_oofs])
    bag_avg = bag_ranks.mean(axis=0)
    auc_avg = float(roc_auc_score(y, bag_avg))
    print(f"\n  bag rank-average AUC: {auc_avg:.5f}", flush=True)

    # eff_rank of the bag
    L = np.log(np.clip(bag_oofs, 1e-9, 1 - 1e-9) /
               (1 - np.clip(bag_oofs, 1e-9, 1 - 1e-9))).T  # (n_tr, n_bags)
    Lc = (L - L.mean(axis=0)) / (L.std(axis=0) + 1e-12)
    s = np.linalg.svd(Lc, compute_uv=False)
    eff_rank = _eff_rank_entropy(s)
    print(f"  bag-of-LRs logit eff_rank: {eff_rank:.3f}  (vs 20-LR-bank 2.19)",
          flush=True)

    return dict(n_bags=n_bags, subset_frac=subset_frac, n_cols=n_cols,
                n_subset=n_subset, bag_avg_auc=auc_avg, eff_rank=eff_rank)


def probe5_per_segment_mega_lr(fe, train, test, y, fold_X_cache):
    """Per (Compound, Year) segment: does per-cell LR beat global LR
    in that cell? Uses fold_X_cache.
    """
    print(f"\n=== Probe 5: Per-segment LR (Compound × Year) ===", flush=True)

    mega_oof_global = np.load(ART / "oof_lr_mega_strat.npy")[:, 1]

    fold_list = fe["fold_list"]
    n_tr = len(y)
    perseg_oof = np.zeros(n_tr)

    seg = (train["Compound"].astype(str) + "_"
           + train["Year"].astype(str)).values
    cells = sorted(np.unique(seg))
    print(f"  {len(cells)} cells: {cells}", flush=True)

    for k, (tr, va) in enumerate(fold_list):
        t0 = time.time()
        Xtr = fold_X_cache[k]["Xtr"]
        Xva = fold_X_cache[k]["Xva"]
        for c in cells:
            tr_mask = (seg[tr] == c)
            va_mask = (seg[va] == c)
            if tr_mask.sum() < 200 or len(np.unique(y[tr][tr_mask])) < 2:
                if va_mask.any():
                    perseg_oof[va[va_mask]] = mega_oof_global[va[va_mask]]
                continue
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            lr.fit(Xtr[tr_mask], y[tr][tr_mask])
            if va_mask.any():
                perseg_oof[va[va_mask]] = lr.predict_proba(Xva[va_mask])[:, 1]
        print(f"    fold {k}: per-cell fit done ({time.time()-t0:.1f}s)",
              flush=True)

    # Global vs per-segment AUC
    global_auc = float(roc_auc_score(y, mega_oof_global))
    perseg_auc = float(roc_auc_score(y, perseg_oof))
    print(f"\n  Global mega AUC: {global_auc:.5f}", flush=True)
    print(f"  Per-segment mega AUC: {perseg_auc:.5f}  "
          f"Δ {(perseg_auc - global_auc) * 1e4:+.3f} bp", flush=True)

    # Per-cell AUCs
    print(f"\n  Per-cell AUC (global vs per-segment):", flush=True)
    cell_table = []
    for c in cells:
        mask = (seg == c)
        if mask.sum() < 100 or len(np.unique(y[mask])) < 2:
            continue
        a_g = float(roc_auc_score(y[mask], mega_oof_global[mask]))
        a_s = float(roc_auc_score(y[mask], perseg_oof[mask]))
        d_bp = (a_s - a_g) * 1e4
        cell_table.append({"cell": c, "n": int(mask.sum()),
                            "global_auc": a_g, "perseg_auc": a_s, "delta_bp": d_bp})
        print(f"    {c:<25s} n={mask.sum():>6}  global={a_g:.4f}  "
              f"perseg={a_s:.4f}  Δ={d_bp:+.2f}bp", flush=True)

    return dict(global_auc=global_auc, perseg_auc=perseg_auc,
                delta_bp=float((perseg_auc - global_auc) * 1e4),
                cell_table=cell_table)


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"data: train {train.shape}, prior {y.mean():.4f}", flush=True)

    print("\n[setup] building shared mega FE once...", flush=True)
    fe = build_mega_FE_matrix(train, test, y)

    print("\n[setup] caching per-fold Xtr+Xva (skip Xte to save RAM)...",
          flush=True)
    fold_X_cache = {}
    for k, (tr, va) in enumerate(fe["fold_list"]):
        t0 = time.time()
        Xtr, Xva, _ = assemble_fold_X(fe, train, test, y, tr, va)
        fold_X_cache[k] = dict(Xtr=Xtr, Xva=Xva)
        print(f"  fold {k}: cached  ({time.time()-t0:.1f}s, "
              f"Xtr {Xtr.shape})", flush=True)

    out = {}
    out["probe4"] = probe4_random_subspace_bagged_lr(
        fe, train, test, y, fold_X_cache, n_bags=10, subset_frac=0.30)
    out["probe5"] = probe5_per_segment_mega_lr(fe, train, test, y, fold_X_cache)

    out_json = ART / "lr_leverage_phaseC.json"
    out_json.write_text(json.dumps(out, indent=2, default=lambda o: float(o)))
    print(f"\n→ {out_json}", flush=True)


if __name__ == "__main__":
    main()
