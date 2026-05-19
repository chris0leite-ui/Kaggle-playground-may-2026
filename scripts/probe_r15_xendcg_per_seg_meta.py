"""scripts/probe_r15_xendcg_per_seg_meta.py — R15 Phase 1: per-segment xendcg-meta.

META loss-class swap on K=16 PRIMARY pool. Replaces per-segment LR-meta
+ Path-B shrinkage with per-segment LightGBM `rank_xendcg` (listwise
ranking loss). Different gradient profile from Logloss-LR.

Loss-class axis at the META layer has been unchanged since R5 (LR-meta
+ Path-B shrinkage). Brainstorm S4 (line 92-99: "the loss-class
reframe... single most likely lift" per 2026-05-18 plateau-brainstorm).
Path-B segmentation contributes +0.15 bp per R13 Phase B diagnostic;
PER-SEGMENT xendcg preserves segmentation while swapping the loss
class.

Approach: per fold, per segment (DriverClass × Stint, 12 segs), train
a LightGBM rank_xendcg model with per-(Year, Race) groups inside the
segment. Skip Path-B shrinkage in v1 (simpler implementation; the
question is whether loss-class novelty alone helps).

K=16 pool (R14 PRIMARY): K=13 + cb_horizon + cb_stint_completion + TabM.

Usage:
  python scripts/probe_r15_xendcg_per_seg_meta.py [--smoke] [--max-rounds 2000]
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
from build_K13_pathb_multiseg import K13_FILES, _is_named
from build_K11_full_pathb import _pos, expand, MIN_ROWS

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
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


def build_driverclass_stint_seg(train: pd.DataFrame, test: pd.DataFrame):
    """DriverClass × Stint segmentation (12 segments). Same as R7/R14."""
    named_tr = _is_named(train["Driver"]).astype(int).values
    named_te = _is_named(test["Driver"]).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    return named_tr * 6 + s_tr, named_te * 6 + s_te, 12


def make_groups(df_subset: pd.DataFrame) -> list[int]:
    """Return group sizes per (Year, Race) for LightGBM rank_xendcg.

    Input df_subset should already be sorted by (Year, Race, ...).
    Returns list of consecutive group sizes; sum equals len(df_subset).
    """
    keys = (df_subset["Year"].astype(str) + "|" +
            df_subset["Race"].astype(str)).values
    sizes = []
    cur = keys[0]
    cnt = 1
    for k in keys[1:]:
        if k == cur:
            cnt += 1
        else:
            sizes.append(cnt)
            cur = k
            cnt = 1
    sizes.append(cnt)
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    print("== R15 Phase 1: per-segment xendcg-meta on K=16 ==", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    # Build K=16 base OOF + test matrices
    extra_bases = ["R12_cb_horizon", "R13_cb_stint_completion", "R14_tabm"]
    oof_cols = []
    test_cols = []
    names = []
    for name, oof_file, test_file in K13_FILES:
        oof_cols.append(_pos(ART / oof_file))
        test_cols.append(_pos(ART / test_file))
        names.append(name)
    for nm in extra_bases:
        oof_cols.append(_pos(ART / f"oof_{nm}_strat.npy"))
        test_cols.append(_pos(ART / f"test_{nm}_strat.npy"))
        names.append(nm)
    K16_oof = np.column_stack(oof_cols)
    K16_test = np.column_stack(test_cols)
    print(f"  K=16 OOF: {K16_oof.shape}; test: {K16_test.shape}", flush=True)

    # expand: [raw | rank/n | logit] → 48 cols
    F_oof = expand(K16_oof)
    F_test = expand(K16_test)
    print(f"  expanded OOF: {F_oof.shape}; test: {F_test.shape}", flush=True)

    # Segmentation: DriverClass × Stint (same as R14 PRIMARY)
    seg_tr, seg_te, n_seg = build_driverclass_stint_seg(train, test)
    cnts = np.bincount(seg_tr, minlength=n_seg)
    print(f"  DriverClass×Stint: {n_seg} segments; "
          f"{(cnts >= MIN_ROWS).sum()} above MIN_ROWS={MIN_ROWS}", flush=True)

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(y_all), 50_000,
                                                  replace=False)
        F_oof = F_oof[idx]
        seg_tr = seg_tr[idx]
        y_all = y_all[idx]
        train_sub = train.iloc[idx].reset_index(drop=True)
        print(f"  SMOKE: subset to {len(y_all)} rows", flush=True)
    else:
        train_sub = train

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_all)), y_all))
    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(F_test), dtype=np.float64)
    fold_walls = []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
        t_f = time.time()
        n_used = 0
        for s in range(n_seg):
            # ti rows in segment s
            idx_ti = np.where(seg_tr[ti] == s)[0]
            if len(idx_ti) < MIN_ROWS or len(np.unique(y_all[ti][idx_ti])) < 2:
                continue
            # va rows in segment s (for OOF)
            idx_va = np.where(seg_tr[vi] == s)[0]
            if len(idx_va) == 0:
                continue
            # te rows in segment s (for test pred)
            idx_te = np.where(seg_te == s)[0]

            # Build per-(Year, Race) groups within segment for rank_xendcg
            ti_sub_df = train_sub.iloc[ti[idx_ti]].copy()
            # Need sorted by (Year, Race) for grouping
            sort_idx = np.argsort(ti_sub_df[["Year", "Race"]]
                                  .apply(lambda r: f"{r['Year']}_{r['Race']}",
                                         axis=1).values, kind="stable")
            ti_sub_df = ti_sub_df.iloc[sort_idx].reset_index(drop=True)
            X_tr = F_oof[ti[idx_ti]][sort_idx]
            y_tr = y_all[ti[idx_ti]][sort_idx]
            groups_tr = make_groups(ti_sub_df)

            # Train per-segment rank_xendcg
            ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(int),
                                group=groups_tr)
            booster = lgb.train(
                LGB_RANK_PARAMS, ds_tr,
                num_boost_round=args.max_rounds,
                callbacks=[lgb.log_evaluation(period=0)],
            )

            # Predict on val rows of segment s (no group needed for prediction)
            pred_va = booster.predict(F_oof[vi[idx_va]])
            oof[vi[idx_va]] = pred_va

            # Predict on test rows of segment s
            if len(idx_te) > 0:
                pred_te = booster.predict(F_test[idx_te])
                test_pred[idx_te] += pred_te / n_eff_folds
            n_used += 1
        wall = time.time() - t_f
        fold_walls.append(wall)
        print(f"  fold {fold}/{n_eff_folds}: segments used {n_used}/{n_seg}  "
              f"wall {wall:.0f}s", flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold proj "
              f"~{(time.time()-t0)*N_FOLDS:.0f}s", flush=True)
        return

    auc = float(roc_auc_score(y_all, oof))
    print(f"\n  Per-segment xendcg-meta OOF AUC: {auc:.6f}", flush=True)

    # Compare to R14 PRIMARY (K=16 + LR-meta Path-B DCS τ=100k)
    r14_oof = np.load(ART / "oof_K16_tabm_pathb_dcs_tau100000.npy")
    auc_r14 = float(roc_auc_score(y_all, r14_oof))
    delta_bp = (auc - auc_r14) * 1e4
    rho_oof, _ = spearmanr(oof, r14_oof)
    print(f"  R14 PRIMARY OOF: {auc_r14:.6f}", flush=True)
    print(f"  Δ vs R14: {delta_bp:+.4f} bp", flush=True)
    print(f"  ρ_OOF vs R14: {rho_oof:.6f}", flush=True)

    r14_test = np.load(ART / "test_K16_tabm_pathb_dcs_tau100000.npy")
    rho_test, _ = spearmanr(test_pred, r14_test)
    print(f"  ρ_test vs R14: {rho_test:.6f}", flush=True)

    # Save artifacts
    np.save(ART / "oof_R15_xendcg_per_seg_strat.npy",
            oof.astype(np.float32))
    np.save(ART / "test_R15_xendcg_per_seg_strat.npy",
            test_pred.astype(np.float32))
    print(f"  Saved oof_R15_xendcg_per_seg_strat.npy + test_..._strat.npy",
          flush=True)

    # Submission CSV if gate-clears
    eps = 1e-3
    gate_oof = (delta_bp >= 0.02 and 0.999 <= rho_test < 0.9999)
    strict_g2 = (delta_bp >= 0.10)
    if gate_oof or strict_g2:
        # rank-normalize test predictions to (0.001, 0.999) for submission
        from scipy.stats import rankdata
        ranks = rankdata(test_pred)
        unif = np.clip((ranks - 0.5) / len(ranks), 0.001, 0.999)
        sub_id = pd.read_csv("data/test.csv")["id"].values
        sub = pd.DataFrame({"id": sub_id, "PitNextLap": unif})
        Path("submissions").mkdir(exist_ok=True)
        sub.to_csv("submissions/submission_R15_xendcg_per_seg_K16.csv",
                   index=False)
        print(f"\n  GATE PASSED (Δ {delta_bp:+.4f} / ρ_test {rho_test:.6f});"
              f" wrote submissions/submission_R15_xendcg_per_seg_K16.csv",
              flush=True)
    else:
        print(f"\n  Gate NOT cleared (Δ {delta_bp:+.4f} / ρ_test {rho_test:.6f})",
              flush=True)

    summary = dict(
        round="R15_Phase1_xendcg_per_seg",
        oof_auc=auc,
        r14_primary_oof=auc_r14,
        delta_vs_r14_bp=delta_bp,
        rho_oof_vs_r14=float(rho_oof),
        rho_test_vs_r14=float(rho_test),
        fold_walls=fold_walls,
        wall_total_s=time.time() - t0,
        gate_passed=(gate_oof or strict_g2),
    )
    out_json = Path("audit/2026-05-19-r15-xendcg-per-seg.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
