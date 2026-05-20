"""scripts/probe_r17_listwise_race_lap_cohort.py — Round 17 inventor track:
listwise rank_xendcg meta on K=17 with (Year, Race, LapNumber) cohort
grouping.

Single lever changed from R13 / R15: cohort definition.
- R13 xendcg-meta: group = whole fold (~350k rows) — too coarse.
- R15 xendcg-per-seg: per-segment (DriverClass × Stint × Year × Race) —
  ~1200 small groups within 12 segments.
- R17 (this): **per-(Year, Race, LapNumber) global cohort** — ~6,182
  cohorts, median 58 / mean 71 / max 373 rows per cohort. This is the
  natural cohort that competes at the same race instant, the row-AUC
  metric's local structure.

Why this could break the ceiling: K=17 + Path-B uses pointwise Logloss +
per-(DriverClass × Stint) shrinkage. Listwise NDCG on (Year, Race,
LapNumber) cohorts directly optimizes row-AUC structure WITHIN local
competitor cohorts — the metric the comp scores. R15's xendcg-as-base
validated listwise loss is orthogonal to logloss bases at coarser
grouping; R17 tests whether finer cohort definition unlocks more.

Output: oof_R17_listwise_race_lap_strat.npy + test_R17_..._strat.npy.
To be folded back as a base into Path-B K=18 via build_K13_pathb_multiseg.

Usage:
  python scripts/probe_r17_listwise_race_lap_cohort.py [--smoke] [--max-rounds 2000]
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


def make_groups_yrl(df_subset: pd.DataFrame) -> list[int]:
    """Return group sizes per (Year, Race, LapNumber).
    df_subset must already be sorted by (Year, Race, LapNumber).
    """
    keys = (df_subset["Year"].astype(str) + "|" +
            df_subset["Race"].astype(str) + "|" +
            df_subset["LapNumber"].astype(str)).values
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


def build_k17_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """K=17 = K=13 pool + cb_horizon + cb_stint_completion + TabM + R15
    xendcg-per-seg base (same pool R15 PRIMARY uses)."""
    oof_cols, test_cols, names = [], [], []
    for name, oof_file, test_file in K13_FILES:
        oof_cols.append(_pos(ART / oof_file))
        test_cols.append(_pos(ART / test_file))
        names.append(name)
    extras = [
        ("R12_cb_horizon", "oof_R12_cb_horizon_strat.npy",
         "test_R12_cb_horizon_strat.npy"),
        ("R13_cb_stint_completion", "oof_R13_cb_stint_completion_strat.npy",
         "test_R13_cb_stint_completion_strat.npy"),
        ("R14_tabm", "oof_R14_tabm_strat.npy", "test_R14_tabm_strat.npy"),
        ("R15_xendcg_per_seg", "oof_R15_xendcg_per_seg_strat.npy",
         "test_R15_xendcg_per_seg_strat.npy"),
    ]
    for nm, of, tf in extras:
        oof_cols.append(_pos(ART / of))
        test_cols.append(_pos(ART / tf))
        names.append(nm)
    return (np.column_stack(oof_cols), np.column_stack(test_cols), names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    print("== R17 inventor-track: listwise xendcg on (Year, Race, LapNumber) ==",
          flush=True)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    K17_oof, K17_test, base_names = build_k17_matrix(train, test)
    print(f"  K=17 OOF: {K17_oof.shape}; bases: {len(base_names)}",
          flush=True)

    F_oof = expand(K17_oof)
    F_test = expand(K17_test)
    print(f"  expanded OOF: {F_oof.shape}; test: {F_test.shape}", flush=True)

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(y_all), 50_000,
                                                  replace=False)
        F_oof = F_oof[idx]
        y_all = y_all[idx]
        train_sub = train.iloc[idx].reset_index(drop=True)
        print(f"  SMOKE: subset to {len(y_all)} rows", flush=True)
    else:
        train_sub = train

    # Diagnostic: cohort size distribution on full train
    if not args.smoke:
        cohort_keys = (train["Year"].astype(str) + "|" +
                       train["Race"].astype(str) + "|" +
                       train["LapNumber"].astype(str))
        cohort_sizes = cohort_keys.value_counts().values
        print(f"  cohort (Y,R,L) count: {len(cohort_sizes)} "
              f"median={int(np.median(cohort_sizes))} "
              f"mean={cohort_sizes.mean():.1f} "
              f"max={cohort_sizes.max()}", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_all)), y_all))
    oof_meta = np.zeros(len(y_all), dtype=np.float64)
    test_meta = np.zeros(len(F_test), dtype=np.float64)
    fold_aucs, walls = [], []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    for k, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
        t_f = time.time()
        # Sort training rows by (Year, Race, LapNumber) for per-cohort grouping
        ti_df = train_sub.iloc[ti].copy().reset_index(drop=True)
        sort_idx = np.lexsort((
            ti_df["LapNumber"].astype(int).values,
            ti_df["Race"].astype(str).values,
            ti_df["Year"].astype(int).values,
        ))
        ti_df_sorted = ti_df.iloc[sort_idx].reset_index(drop=True)
        X_tr = F_oof[ti][sort_idx]
        y_tr = y_all[ti][sort_idx]
        groups_tr = make_groups_yrl(ti_df_sorted)
        print(f"  fold {k}/{n_eff_folds}: ti={len(ti)} sorted, "
              f"{len(groups_tr)} cohorts "
              f"(min={min(groups_tr)} max={max(groups_tr)} "
              f"median={int(np.median(groups_tr))})", flush=True)

        ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(int), group=groups_tr)
        booster = lgb.train(
            LGB_RANK_PARAMS, ds_tr,
            num_boost_round=args.max_rounds,
            callbacks=[lgb.log_evaluation(period=200)],
        )

        # Predict on val (no group sort needed for prediction)
        pred_va = booster.predict(F_oof[vi])
        oof_meta[vi] = pred_va
        if not args.smoke:
            pred_te = booster.predict(F_test)
            test_meta += pred_te / n_eff_folds

        auc_va = float(roc_auc_score(y_all[vi], pred_va))
        wall = time.time() - t_f
        fold_aucs.append(auc_va)
        walls.append(wall)
        print(f"    iter_final={booster.current_iteration()} "
              f"AUC(val)={auc_va:.5f} wall={wall:.0f}s", flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold proj "
              f"~{(time.time()-t0)*N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y_all, oof_meta))
    print(f"\n  R17 listwise (Y,R,L) meta OOF AUC: {auc_full:.6f}",
          flush=True)

    # Compare to R15 PRIMARY (K=17 + Path-B DCS τ=100k)
    r15_oof = np.load(ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy")
    auc_r15 = float(roc_auc_score(y_all, r15_oof))
    delta_bp = (auc_full - auc_r15) * 1e4
    rho_oof, _ = spearmanr(oof_meta, r15_oof)
    print(f"  R15 PRIMARY OOF: {auc_r15:.6f}", flush=True)
    print(f"  Δ vs R15: {delta_bp:+.4f} bp", flush=True)
    print(f"  ρ_OOF vs R15: {rho_oof:.6f}", flush=True)

    r15_test = np.load(ART / "test_K17_xendcg_pathb_dcs_tau100000.npy")
    rho_test, _ = spearmanr(test_meta, r15_test)
    print(f"  ρ_test vs R15: {rho_test:.6f}", flush=True)
    if rho_test >= 0.9999:
        verdict = "TIE_ZONE"
    elif rho_test < 0.999:
        verdict = "REGRESSION_RISK"
    else:
        verdict = "OK"
    print(f"  Band verdict: {verdict}", flush=True)

    # Save artifacts
    np.save(ART / "oof_R17_listwise_race_lap_strat.npy",
            oof_meta.astype(np.float32))
    np.save(ART / "test_R17_listwise_race_lap_strat.npy",
            test_meta.astype(np.float32))
    print(f"  Saved oof_R17_listwise_race_lap_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R17_inventor_listwise_race_lap_cohort",
        oof_auc=auc_full,
        auc_r15_baseline=auc_r15,
        delta_bp_vs_r15=delta_bp,
        rho_oof_vs_r15=float(rho_oof),
        rho_test_vs_r15=float(rho_test),
        verdict_band=verdict,
        fold_aucs=fold_aucs,
        fold_walls_s=walls,
        wall_total_s=time.time() - t0,
        max_rounds=args.max_rounds,
        bases=base_names,
    )
    out_json = Path("audit/2026-05-20-round-17-listwise.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}  total wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
