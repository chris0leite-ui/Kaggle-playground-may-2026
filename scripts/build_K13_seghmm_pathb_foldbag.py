"""scripts/build_K13_seghmm_pathb_foldbag.py — Round 6 Phase B

Fixes the R5 5-seed bag design flaw. `run_pathb` in
`build_K11_full_pathb.py:116-128` uses a single FULL-train Path-B
fit for test predictions; LR convex fit + per-segment shrinkage is
seed-invariant given the same training data, so multi-seed test
predictions are identical (ρ=1.0 in R5 5-seed bag).

This script uses FOLD-FIT averaging across seeds:
- For each seed s in {42, 43, 44, 45, 46}:
  - Run 5-fold StratifiedKFold(seed=s)
  - For each fold k, fit Path-B per-segment shrunk weights on train_k
  - Apply those weights to TEST → test_pred_(s, k)
- Average all (seed × fold) test predictions

This is TRUE bagging: each (seed, fold) produces a distinct test
prediction. OOF is also averaged across seeds.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from build_K11_full_pathb import (_pos, expand, fit_lr_aug, predict_aug,
                                   MIN_ROWS, MAX_ITER, ART, DATA, TARGET)


K13_FILES = [
    ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy",     "test_d17_h1d_yekenot_full_strat.npy"),
    ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy",      "test_p1_single_cb_v4_gpu_strat.npy"),
    ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy",             "test_f1_hgbc_deep_strat.npy"),
    ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ("qAT",       "dgp_v3_qAT_K1_oof.npy",                  "dgp_v3_qAT_K1_test.npy"),
    ("qAV",       "dgp_v3_qAV_K1_7feat_oof.npy",            "dgp_v3_qAV_K1_7feat_test.npy"),
    ("qAO",       "dgp_v3_qAO_knn_multi_oof.npy",           "dgp_v3_qAO_knn_multi_test.npy"),
    ("qAA",       "dgp_v3_qAA_stint_imputed_oof.npy",       "dgp_v3_qAA_stint_imputed_test.npy"),
    ("qAF",       "dgp_v3_qAF_d16plus_oof.npy",             "dgp_v3_qAF_d16plus_test.npy"),
    ("qAK",       "dgp_v3_qAK_knn3_oof.npy",                "dgp_v3_qAK_knn3_test.npy"),
    ("K27_100k",  "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
                  "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"),
    ("seg_fe",    "oof_r4_segment_fe_strat.npy",            "test_r4_segment_fe_strat.npy"),
    ("HMM",       "oof_r4_hmm_seq_strat.npy",               "test_r4_hmm_seq_strat.npy"),
]

N_FOLDS = 5


def run_pathb_foldfit(base_oofs, base_tests, train, test, y, tau, seed):
    """Run Path-B with FOLD-FIT test predictions (no full-train fit).
    Returns (oof, test_pred) where test_pred is averaged across the 5 folds.
    """
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_train = c_tr * 6 + s_tr
    seg_test = c_te * 6 + s_te

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))

    for fold, (tr, va) in enumerate(splits):
        t0 = time.time()
        w_g = fit_lr_aug(F_oof[tr], y[tr])
        W_l = np.zeros((n_seg, len(w_g)))
        cnt = np.zeros(n_seg, dtype=np.int64)
        msk = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            cnt[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_l[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            msk[s] = True
        n_local = cnt.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_sh = alpha[:, None] * W_l + (1 - alpha[:, None]) * w_g[None, :]
        # OOF on val partition
        for s in np.unique(seg_train[va]):
            idx_v = np.where(seg_train[va] == s)[0]
            w = W_sh[s] if msk[s] else w_g
            oof[va[idx_v]] = predict_aug(F_oof[va[idx_v]], w)
        # TEST predictions using this fold's weights, then average
        test_fold = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx_t = np.where(seg_test == s)[0]
            w = W_sh[s] if msk[s] else w_g
            test_fold[idx_t] = predict_aug(F_test[idx_t], w)
        test_pred += test_fold / N_FOLDS
        print(f"    seed={seed} fold {fold+1}: {time.time()-t0:.1f}s", flush=True)

    return oof, test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=100_000)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--name", default="K13_seghmm_pathb_foldbag")
    args = ap.parse_args()
    tau = args.tau

    t0 = time.time()
    print(f"=== R6 Phase B: fold-fit bagging K=13+Path-B τ={tau} "
          f"× {len(args.seeds)} seeds ===")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values

    oofs = [_pos(ART / o) for _, o, _ in K13_FILES]
    tests = [_pos(ART / t) for _, _, t in K13_FILES]

    seed_oofs, seed_tests = [], []
    for s in args.seeds:
        print(f"\n  -- seed {s} --")
        oof_s, test_s = run_pathb_foldfit(oofs, tests, train, test, y,
                                           tau=tau, seed=s)
        seed_oofs.append(oof_s)
        seed_tests.append(test_s)
        auc = roc_auc_score(y, oof_s)
        print(f"    seed {s} OOF AUC: {auc:.5f}")

    bag_oof = np.mean(seed_oofs, axis=0)
    bag_test = np.mean(seed_tests, axis=0)
    bag_auc = float(roc_auc_score(y, bag_oof))
    print(f"\n  Bagged OOF AUC: {bag_auc:.5f}  ({len(args.seeds)} seeds)")

    # Compare to single-seed reference (R5.2 K=13+Path-B τ=100k OOF)
    ref_oof_path = ART / "K13_seghmm_pathb_tau100000_oof.npy"
    ref_test_path = ART / "K13_seghmm_pathb_tau100000_test.npy"
    if ref_oof_path.exists():
        ref_oof = _pos(ref_oof_path)
        ref_test = _pos(ref_test_path)
        ref_auc = float(roc_auc_score(y, ref_oof))
        delta_bp = (bag_auc - ref_auc) * 1e4
        rho_test, _ = spearmanr(bag_test, ref_test)
        print(f"  R5.2 single-seed reference OOF: {ref_auc:.5f}")
        print(f"  Bag - reference: {delta_bp:+.3f} bp")
        print(f"  ρ_test vs reference: {rho_test:.6f}")
        if abs(rho_test) > 0.99999:
            print(f"  ⚠ ρ ≈ 1 — bag may still be seed-invariant; CHECK")
        else:
            print(f"  ✓ ρ < 1 — bag predictions differ from reference (variance reduction working)")

    np.save(ART / f"oof_{args.name}_strat.npy",
            np.column_stack([1 - bag_oof, bag_oof]).astype(np.float64))
    np.save(ART / f"test_{args.name}_strat.npy",
            np.column_stack([1 - bag_test, bag_test]).astype(np.float64))

    sub = pd.DataFrame({"id": test["id"], "PitNextLap": np.clip(bag_test, 0.001, 0.999)})
    sub_path = Path(f"submissions/submission_{args.name}_tau{tau}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\n  Wrote {sub_path}")
    print(f"  Total wall: {time.time()-t0:.1f}s")

    Path("audit").mkdir(exist_ok=True)
    Path(f"audit/2026-05-18-round-6-phase-b.json").write_text(json.dumps({
        "tau": tau, "seeds": args.seeds, "bag_auc": bag_auc,
        "per_seed_auc": [float(roc_auc_score(y, o)) for o in seed_oofs],
        "reference_auc": ref_auc if ref_oof_path.exists() else None,
        "delta_bp": delta_bp if ref_oof_path.exists() else None,
        "rho_test": float(rho_test) if ref_oof_path.exists() else None,
    }, indent=2))


if __name__ == "__main__":
    main()
