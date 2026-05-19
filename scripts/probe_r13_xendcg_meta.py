"""scripts/probe_r13_xendcg_meta.py — Phase C: LightGBM rank_xendcg meta on K=14.

Replaces Path-B's segmented LR-meta (Logloss) with a LightGBM listwise
ranking loss (`objective='rank_xendcg'`) on the K=14 base columns (K=13
pool + cb_horizon). The plan-agent identified this as the UNTOUCHED
META-CLASS axis: cb_horizon proved loss-class novelty pays off at the
BASE; xendcg tests it at the META.

2026-05-18 plateau-brainstorm S4 (line 204-208): "The single most
likely lift is the loss-class reframe... those introduce a NEW
LOSS-FUNCTION class, not a NEW FEATURE/BASE within the existing
log-loss family." cb_horizon's win at the base level supports this
prediction; xendcg-meta tests it at the meta level.

Fold-safety: 5-fold StratifiedKFold seed=42, same as Path-B. Per-fold
meta refit on ti rows only (no val leakage). Test predictions averaged
across folds.

Features: expand(K14_OOF) → 42 columns (14 raw + 14 rank + 14 logit),
matching Path-B's expand() convention.

Target: PitNextLap binary (0/1) used as relevance score for xendcg.
Group: by (Year, Race) — listwise ranking per race. LightGBM
`rank_xendcg` caps groups at 10000 rows; ~104 (Year, Race) combos
yields ~4200 rows/group on average (well under cap). Per-race
listwise rank-of-PitNextLap is highly correlated with the global
rank, so global AUC should approximate.

Compare to R12-2 PRIMARY OOF 0.954475.

Usage:
  python scripts/probe_r13_xendcg_meta.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from build_K13_pathb_multiseg import K13_FILES
from build_K11_full_pathb import _pos, expand

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

LGB_RANK_PARAMS = dict(
    objective="rank_xendcg",
    metric="ndcg",
    eval_at=[1, 10, 100],
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    lambda_l1=0.0,
    lambda_l2=1.0,
    max_depth=-1,
    n_jobs=-1,
    verbose=-1,
    random_state=SEED,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    print("== R13 Phase C: LightGBM rank_xendcg meta on K=14 ==", flush=True)
    y_all = pd.read_csv("data/train.csv")[TARGET].astype(int).values
    print(f"  y: {len(y_all)} rows, prior {y_all.mean():.4f}", flush=True)

    test_id = pd.read_csv("data/test.csv")["id"].values
    n_test = len(test_id)
    print(f"  test rows: {n_test}", flush=True)

    # Build K=14 base OOF + test matrices (K=13 pool + cb_horizon)
    oof_cols = []
    test_cols = []
    names = []
    for name, oof_file, test_file in K13_FILES:
        oof_cols.append(_pos(ART / oof_file))
        test_cols.append(_pos(ART / test_file))
        names.append(name)
    # cb_horizon as 14th
    cbh_oof = np.load(ART / "oof_R12_cb_horizon_strat.npy").astype(np.float64)
    cbh_test = np.load(ART / "test_R12_cb_horizon_strat.npy").astype(np.float64)
    oof_cols.append(cbh_oof)
    test_cols.append(cbh_test)
    names.append("cb_horizon")
    K14_oof = np.column_stack(oof_cols)
    K14_test = np.column_stack(test_cols)
    print(f"  K=14 OOF: {K14_oof.shape}; test: {K14_test.shape}", flush=True)
    print(f"  bases: {names}", flush=True)

    # expand: [raw | rank/n | logit] → 42 cols
    K14_exp_oof = expand(K14_oof)
    K14_exp_test = expand(K14_test)
    print(f"  expanded OOF: {K14_exp_oof.shape}; test: {K14_exp_test.shape}",
          flush=True)

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(y_all), 50_000,
                                                  replace=False)
        K14_exp_oof = K14_exp_oof[idx]
        y_all = y_all[idx]
        print(f"  SMOKE: subset to {len(y_all)} rows", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_all)), y_all))
    oof_meta = np.zeros(len(y_all), dtype=np.float64)
    test_meta = np.zeros(n_test, dtype=np.float64)
    fold_aucs, walls = [], []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    for k, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
        t_f = time.time()
        X_tr = K14_exp_oof[ti]
        X_va = K14_exp_oof[vi]
        y_tr = y_all[ti]
        y_va = y_all[vi]
        # rank_xendcg expects integer relevance scores >= 0
        ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(int),
                            group=[len(ti)])
        ds_va = lgb.Dataset(X_va, label=y_va.astype(int),
                            group=[len(vi)], reference=ds_tr)

        booster = lgb.train(
            LGB_RANK_PARAMS,
            ds_tr,
            num_boost_round=args.max_rounds,
            valid_sets=[ds_va],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        pred_va = booster.predict(X_va, num_iteration=booster.best_iteration)
        oof_meta[vi] = pred_va
        # Test: use the test-side expanded matrix (full)
        if not args.smoke:
            pred_te = booster.predict(K14_exp_test,
                                       num_iteration=booster.best_iteration)
            test_meta += pred_te / n_eff_folds

        try:
            auc_va = float(roc_auc_score(y_va, pred_va))
        except ValueError:
            auc_va = float("nan")
        wall = time.time() - t_f
        fold_aucs.append(auc_va)
        walls.append(wall)
        print(f"  fold {k}/{n_eff_folds}: best_iter={booster.best_iteration} "
              f"AUC(val)={auc_va:.5f} wall={wall:.0f}s", flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold projection "
              f"~ {(time.time()-t0) * N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y_all, oof_meta))
    print(f"\n  R13 xendcg-meta OOF AUC: {auc_full:.6f}", flush=True)

    # Compare to R12-2 PRIMARY OOF
    r12_oof_path = ART / "oof_K14_pathb_driverclass_stint_tau100000.npy"
    r12_oof = np.load(r12_oof_path)
    auc_r12 = float(roc_auc_score(y_all, r12_oof))
    delta_bp = (auc_full - auc_r12) * 1e4
    rho_oof, _ = spearmanr(oof_meta, r12_oof)
    print(f"  R12-2 PRIMARY OOF:        {auc_r12:.6f}", flush=True)
    print(f"  Δ vs R12-2:               {delta_bp:+.4f} bp", flush=True)
    print(f"  ρ_OOF vs R12-2:           {rho_oof:.6f}", flush=True)

    # Test rho vs R12-2 test
    r12_test = np.load(ART / "test_K14_pathb_driverclass_stint_tau100000.npy")
    rho_test, _ = spearmanr(test_meta, r12_test)
    print(f"  ρ_test vs R12-2:          {rho_test:.6f}", flush=True)

    # Save artifacts (rank-normalize for K=15 add convention, optional)
    np.save(ART / "oof_R13_xendcg_meta_strat.npy", oof_meta.astype(np.float32))
    np.save(ART / "test_R13_xendcg_meta_strat.npy", test_meta.astype(np.float32))
    print(f"  Saved oof_R13_xendcg_meta_strat.npy + test_..._strat.npy",
          flush=True)

    # Submission CSV if gate-clears
    SUBMISSION_GATE_OOF_FLOOR = 0.954475 + 0.02e-4  # +0.02 bp over R12-2
    SUBMISSION_RHO_OK_LOW, SUBMISSION_RHO_OK_HIGH = 0.999, 0.9999
    submission_qualifies = (
        (auc_full >= SUBMISSION_GATE_OOF_FLOOR and
         SUBMISSION_RHO_OK_LOW <= rho_test < SUBMISSION_RHO_OK_HIGH)
        or (delta_bp >= 0.10)
    )
    if submission_qualifies:
        # Need test predictions normalized to (0, 1) -- LightGBM rank outputs
        # are real-valued ranking scores. Rank-normalize.
        from scipy.stats import rankdata
        combined = np.concatenate([oof_meta, test_meta])
        ranks = rankdata(combined)
        eps = 1.0 / (2 * len(ranks))
        uniform = np.clip((ranks - 0.5) / len(ranks), eps, 1 - eps)
        oof_uniform = uniform[:len(oof_meta)]
        test_uniform = uniform[len(oof_meta):]
        sub = pd.DataFrame({"id": test_id,
                            "PitNextLap": np.clip(test_uniform, 0.001, 0.999)})
        Path("submissions").mkdir(exist_ok=True)
        sub_path = ("submissions/submission_R13_xendcg_meta_K14_"
                    "pathb_dcs_tau100000.csv")
        sub.to_csv(sub_path, index=False)
        print(f"\n  SUBMISSION GATE PASSED ({delta_bp:+.4f} bp / ρ_test "
              f"{rho_test:.6f}); wrote {sub_path}", flush=True)
    else:
        print(f"\n  Gate NOT cleared "
              f"(Δ {delta_bp:+.4f} bp, ρ_test {rho_test:.6f})", flush=True)

    summary = dict(
        round="R13_C_xendcg_meta",
        oof_auc=auc_full,
        r12_primary_oof=auc_r12,
        delta_vs_r12_bp=delta_bp,
        rho_oof_vs_r12=float(rho_oof),
        rho_test_vs_r12=float(rho_test),
        fold_aucs=fold_aucs,
        fold_walls=walls,
        submission_qualifies=submission_qualifies,
        wall_total_s=time.time() - t0,
    )
    out_json = Path("audit/2026-05-19-r13-xendcg-meta.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
