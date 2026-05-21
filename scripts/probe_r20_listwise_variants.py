"""scripts/probe_r20_listwise_variants.py — Round 20 inventor extension:
listwise meta with parameterized cohort grain and loss objective.

R17 established the axis: listwise rank-loss meta on (Year, Race, LapNumber)
cohort grouping produces an orthogonal base column (Path-B K=18 +0.092 bp
OOF / +0.01 bp LB). R20 extends the axis on two single-lever sub-axes:

  --cohort {yrl, yrlc, yrls}
    yrl  = (Year, Race, LapNumber)              ~6,182 groups, median 58  (R17 baseline)
    yrlc = (Year, Race, LapNumber, Compound)    finer competitor cohort
    yrls = (Year, Race, LapNumber, StintNum)    finer driver-strategy cohort

  --loss {xendcg, lambdarank}
    xendcg     = listwise cross-entropy NDCG    (R17 baseline)
    lambdarank = pairwise lambda-rank gradient  (different listwise objective)

Output: oof_R20_listwise_<cohort>_<loss>_strat.npy + test_R20_..._strat.npy.
Designed to be added to K=18 pool via build_K13_pathb_multiseg.py --extra-bases.

Usage:
  python scripts/probe_r20_listwise_variants.py --cohort yrlc --loss xendcg [--smoke]
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

COHORT_COLS = {
    "yrl":  ["Year", "Race", "LapNumber"],
    "yrlc": ["Year", "Race", "LapNumber", "Compound"],
    "yrls": ["Year", "Race", "LapNumber", "StintNum"],
}


def lgb_params(loss: str) -> dict:
    base = dict(
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
    if loss == "xendcg":
        base["objective"] = "rank_xendcg"
    elif loss == "lambdarank":
        base["objective"] = "lambdarank"
        base["label_gain"] = [0, 1]  # binary labels
    else:
        raise ValueError(loss)
    return base


def make_groups(df_subset: pd.DataFrame, cols: list[str]) -> list[int]:
    """Return group sizes given pre-sorted dataframe and cohort columns."""
    parts = []
    for c in cols:
        if df_subset[c].dtype.kind in "ifu":
            parts.append(df_subset[c].astype(int).astype(str).values)
        else:
            parts.append(df_subset[c].astype(str).values)
    keys = np.empty(len(df_subset), dtype=object)
    for i in range(len(df_subset)):
        keys[i] = "|".join(p[i] for p in parts)
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
    """K=17 = R15 pool: K=13 + cb_horizon + cb_stint_completion + TabM
    + R15 xendcg-per-seg base."""
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
    ap.add_argument("--cohort", choices=list(COHORT_COLS), required=True)
    ap.add_argument("--loss", choices=["xendcg", "lambdarank"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=2000)
    args = ap.parse_args()

    cols = COHORT_COLS[args.cohort]
    tag = f"{args.cohort}_{args.loss}"

    t0 = time.time()
    print(f"== R20 inventor: listwise {args.loss} on ({'+'.join(cols)}) cohort ==",
          flush=True)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    K17_oof, K17_test, base_names = build_k17_matrix(train, test)
    print(f"  K=17 OOF: {K17_oof.shape}; bases: {len(base_names)}", flush=True)

    F_oof = expand(K17_oof)
    F_test = expand(K17_test)
    print(f"  expanded OOF: {F_oof.shape}; test: {F_test.shape}", flush=True)

    # Diagnostic: cohort size distribution
    if not args.smoke:
        cohort_keys = train[cols[0]].astype(str)
        for c in cols[1:]:
            cohort_keys = cohort_keys + "|" + train[c].astype(str)
        sizes = cohort_keys.value_counts().values
        print(f"  cohort ({','.join(cols)}) count: {len(sizes)} "
              f"median={int(np.median(sizes))} mean={sizes.mean():.1f} "
              f"max={sizes.max()} min={sizes.min()}", flush=True)

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(y_all), 50_000, replace=False)
        F_oof = F_oof[idx]
        y_all = y_all[idx]
        train_sub = train.iloc[idx].reset_index(drop=True)
        print(f"  SMOKE: subset to {len(y_all)} rows", flush=True)
    else:
        train_sub = train

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_all)), y_all))
    oof_meta = np.zeros(len(y_all), dtype=np.float64)
    test_meta = np.zeros(len(F_test), dtype=np.float64)
    fold_aucs, walls = [], []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    for k, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
        t_f = time.time()
        ti_df = train_sub.iloc[ti].copy().reset_index(drop=True)
        # Lexsort: rightmost key is primary
        sort_keys = [ti_df[c].astype(int).values if ti_df[c].dtype.kind in "ifu"
                     else ti_df[c].astype(str).values
                     for c in cols]
        sort_idx = np.lexsort(tuple(reversed(sort_keys)))
        ti_df_sorted = ti_df.iloc[sort_idx].reset_index(drop=True)
        X_tr = F_oof[ti][sort_idx]
        y_tr = y_all[ti][sort_idx]
        groups_tr = make_groups(ti_df_sorted, cols)
        print(f"  fold {k}/{n_eff_folds}: ti={len(ti)} sorted, "
              f"{len(groups_tr)} cohorts "
              f"(min={min(groups_tr)} max={max(groups_tr)} "
              f"median={int(np.median(groups_tr))})", flush=True)

        ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(int), group=groups_tr)
        booster = lgb.train(
            lgb_params(args.loss), ds_tr,
            num_boost_round=args.max_rounds,
            callbacks=[lgb.log_evaluation(period=200)],
        )
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
    print(f"\n  R20 listwise {tag} OOF AUC: {auc_full:.6f}", flush=True)

    r15_oof = np.load(ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy")
    auc_r15 = float(roc_auc_score(y_all, r15_oof))
    delta_bp = (auc_full - auc_r15) * 1e4
    rho_oof, _ = spearmanr(oof_meta, r15_oof)
    print(f"  R15 PRIMARY OOF: {auc_r15:.6f}", flush=True)
    print(f"  Δ vs R15: {delta_bp:+.4f} bp", flush=True)
    print(f"  ρ_OOF vs R15: {rho_oof:.6f}", flush=True)

    # Also compare to R17 listwise direct OOF (sibling)
    r17_oof_path = ART / "oof_R17_listwise_race_lap_strat.npy"
    if r17_oof_path.exists():
        r17_oof = np.load(r17_oof_path)
        rho_r17, _ = spearmanr(oof_meta, r17_oof)
        print(f"  ρ_OOF vs R17: {rho_r17:.6f}", flush=True)

    r15_test = np.load(ART / "test_K17_xendcg_pathb_dcs_tau100000.npy")
    rho_test, _ = spearmanr(test_meta, r15_test)
    print(f"  ρ_test vs R15: {rho_test:.6f}", flush=True)

    oof_out = ART / f"oof_R20_listwise_{tag}_strat.npy"
    test_out = ART / f"test_R20_listwise_{tag}_strat.npy"
    np.save(oof_out, oof_meta.astype(np.float32))
    np.save(test_out, test_meta.astype(np.float32))
    print(f"  Saved {oof_out.name} + {test_out.name}", flush=True)

    summary = dict(
        round=f"R20_inventor_listwise_{tag}",
        cohort_cols=cols, loss=args.loss,
        oof_auc=auc_full,
        auc_r15_baseline=auc_r15,
        delta_bp_vs_r15=delta_bp,
        rho_oof_vs_r15=float(rho_oof),
        rho_test_vs_r15=float(rho_test),
        fold_aucs=fold_aucs,
        fold_walls_s=walls,
        wall_total_s=time.time() - t0,
        max_rounds=args.max_rounds,
        bases=base_names,
    )
    out_json = Path(f"audit/2026-05-21-round-20-listwise-{tag}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}  total wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
