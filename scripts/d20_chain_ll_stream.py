"""scripts/d20_chain_ll_stream.py — Chain-LL-only base (Mechanism 2).

Synthetic-only plan, M2. d18_chain_decomp trains a downstream LGBM on
(raw 14 features + ~24 chain features) and lands +7.365 bp at K=21+1
with ρ=0.9914. This script strips the raw features: downstream LGBM
runs on chain features ONLY. Hypothesis: lower standalone OOF but
ρ-vs-K18 < 0.985 (vs d18's 0.9914 baseline), trading raw predictive
signal for parent-archaeology orthogonality.

Inputs:
  data/train.csv, data/test.csv, data/original/f1_strategy_dataset_v4.csv

Outputs:
  scripts/artifacts/oof_d20_chain_ll_stream_strat.npy    (n_train, 2)
  scripts/artifacts/test_d20_chain_ll_stream_strat.npy   (n_test, 2)
  scripts/artifacts/d20_chain_ll_stream_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from d18_chain_decomp import (CAT_OK, NUM_FEATS, TARGET, SEED, N_FOLDS,
                               _encode_cat, fit_chain, apply_chain,
                               downstream_lgbm)

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
K18_OOF = ART / "oof_K18_pathb_driverclass_stint_tau100000.npy"
K18_TEST = ART / "test_K18_pathb_driverclass_stint_tau100000.npy"


def _save_oof_test(name, oof_pos, test_pos):
    oof = np.column_stack([1.0 - oof_pos, oof_pos]).astype(np.float64)
    test = np.column_stack([1.0 - test_pos, test_pos]).astype(np.float64)
    np.save(ART / f"oof_{name}_strat.npy", oof)
    np.save(ART / f"test_{name}_strat.npy", test)
    print(f"  saved oof_{name}_strat.npy  ({oof.shape})  "
          f"test_{name}_strat.npy ({test.shape})")


def _pos(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, 1].astype(np.float64)
    return arr.astype(np.float64).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold + tiny rounds; ~3 min wall")
    ap.add_argument("--n-synth-sample", type=int, default=0,
                    help="if >0, subsample synth-train/test for smoke")
    ap.add_argument("--out-prefix", default="d20_chain_ll_stream")
    args = ap.parse_args()

    t0 = time.time()
    smoke = args.smoke
    print(f"[d20 chain-LL-only{' SMOKE' if smoke else ''}]  loading data")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test",
                                     "Pre-Season Track Session"])].copy()
    print(f"  train {tr.shape}  test {te.shape}  orig {orig.shape}")

    if args.n_synth_sample > 0:
        tr = tr.sample(n=args.n_synth_sample,
                       random_state=SEED).reset_index(drop=True)
        te = te.sample(n=min(args.n_synth_sample // 2, len(te)),
                       random_state=SEED).reset_index(drop=True)
        print(f"  smoke-sampled: train {tr.shape}  test {te.shape}")

    union = pd.concat([orig[CAT_OK], tr[CAT_OK], te[CAT_OK]],
                      ignore_index=True)
    _, mappings = _encode_cat(union)
    orig_e, _ = _encode_cat(orig, mappings)
    tr_e, _ = _encode_cat(tr, mappings)
    te_e, _ = _encode_cat(te, mappings)

    print(f"\n[fit chain on orig n={len(orig_e)}]")
    t_fit = time.time()
    steps = fit_chain(orig_e, smoke=smoke)
    print(f"  fit_chain wall: {time.time()-t_fit:.1f}s")

    print(f"\n[apply chain → train {len(tr_e)}]")
    t_app = time.time()
    tr_chain = apply_chain(steps, tr_e)
    te_chain = apply_chain(steps, te_e)
    chain_cols = list(tr_chain.columns)
    print(f"  chain features ({len(chain_cols)}): {chain_cols}")
    print(f"  apply_chain wall: {time.time()-t_app:.1f}s")

    y = tr_e[TARGET].astype(int).values

    print(f"\n[downstream LGBM on CHAIN-ONLY features (no raw)]")
    t_lgb = time.time()
    tr_X = tr_chain.reset_index(drop=True)
    te_X = te_chain.reset_index(drop=True)
    oof, test = downstream_lgbm(tr_X, y, te_X, cat_cols=[], smoke=smoke)
    print(f"  downstream_lgbm wall: {time.time()-t_lgb:.1f}s")
    _save_oof_test(args.out_prefix, oof, test)

    standalone_auc = float(roc_auc_score(y, oof))

    summary = dict(
        n_train=int(len(tr_e)),
        n_test=int(len(te_e)),
        n_orig=int(len(orig_e)),
        n_chain_features=len(chain_cols),
        chain_features=chain_cols,
        standalone_oof_auc=standalone_auc,
    )

    # Compare to K=18 PRIMARY if artifacts exist (full run only;
    # smoke subsample breaks length alignment).
    if K18_OOF.exists() and K18_TEST.exists() and args.n_synth_sample == 0:
        k18_oof = _pos(np.load(K18_OOF))
        k18_te = _pos(np.load(K18_TEST))
        if len(k18_oof) == len(oof):
            k18_auc = float(roc_auc_score(y, k18_oof))
            rho_oof = float(spearmanr(oof, k18_oof)[0])
            rho_test = float(spearmanr(test, k18_te)[0])
            gap_bp = (standalone_auc - k18_auc) * 1e4
            summary.update(
                k18_oof_auc=k18_auc,
                gap_vs_k18_bp=gap_bp,
                rho_oof_vs_k18=rho_oof,
                rho_test_vs_k18=rho_test,
            )
            print(f"\n  K=18 PRIMARY OOF AUC  : {k18_auc:.5f}")
            print(f"  d20 standalone OOF AUC : {standalone_auc:.5f}")
            print(f"  gap                    : {gap_bp:+.2f} bp")
            print(f"  ρ_OOF  d20↔K18         : {rho_oof:.5f}")
            print(f"  ρ_test d20↔K18         : {rho_test:.5f}")

    (ART / f"{args.out_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n[done]  total wall: {time.time()-t0:.1f}s")
    print(f"  → next: python scripts/probe.py gate {args.out_prefix} "
          f"--oof {ART}/oof_{args.out_prefix}_strat.npy "
          f"--test {ART}/test_{args.out_prefix}_strat.npy "
          f"--primary-oof {K18_OOF} --primary-test {K18_TEST}")


if __name__ == "__main__":
    main()
