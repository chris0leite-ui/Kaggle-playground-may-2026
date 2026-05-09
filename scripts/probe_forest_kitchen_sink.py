"""scripts/probe_forest_kitchen_sink.py — RF on a wide engineered tableau.

Follow-up to `probe_forest_sweep.py` Angle A (RF on yekenot recipe alone),
which gated +0.26 bp at K=4+1 LR-meta with rho=0.9595 vs PRIMARY.

PI directive: combine every engineered feature family that has produced
a positive single-base lift in this comp and that RF has never seen.
RF benefits most from feature breadth because each tree samples a
random subset; a wider tableau is the regime where it differentiates
from boosted trees.

Substrate (57 features total, no orig CSV / no GPU required):
  - Yekenot recipe (38 feat) — same as Angle A: arithmetic interactions,
    floor-cat numerics, count encoding, KBins, combo cats, CV target
    encoding on (Race, Compound) and (Race, Year).
  - Constraint-violation features (12 feat) — `d18_f2_constraint`'s
    `compute_violations`: 10 physical/logical violation indicators
    (TyreLife monotone, LapNumber monotone, CumDeg drift, etc.) plus
    viol_count and group_size.
  - Inter-stint memory features (7 feat) — `probe_exp3_inter_stint`'s
    `build_inter_stint_features`: prev_stint_length, prev_pit_lap_in_race,
    prev_compound_cat, stints_completed_so_far, cur_stint_lap_idx,
    laps_since_last_pit, stint_length.

NOT included (require external dependencies):
  - Chain-decomposition features (need orig CSV).
  - Pre-image kNN distances (need orig CSV).
  - Conditional-vector tuple lookup (need orig CSV).
  - DAE 768-d latents (need GPU regeneration).

Output:
  - oof_rf_kitchen_sink_strat.npy / test_rf_kitchen_sink_strat.npy
  - probe_forest_kitchen_sink.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

# Reuse helpers from Angle A's sweep script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_forest_sweep import (  # type: ignore
    K4_BASES, IMPORTANT_COMBOS, yekenot_fe, label_encode, to_float,
    load_k4_pool, expand, lr_meta_oof, pred_lb_band, _pos,
)
# d18_f2_constraint defines compute_violations(df)
from d18_f2_constraint import compute_violations  # type: ignore
# probe_exp3_inter_stint_features defines build_inter_stint_features(train, test)
from probe_exp3_inter_stint_features import build_inter_stint_features  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ART = Path("scripts/artifacts")
ID = "id"
TARGET = "PitNextLap"
SEED, N_FOLDS, MAX_ITER = 42, 5, 500


def build_kitchen_sink_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Produce the union of yekenot + constraint + inter-stint features.

    Returns aligned (X_train, X_test, combo_names) where the row order
    matches train/test inputs. combo_names are the yekenot combo cats
    that need per-fold CV target encoding.
    """
    print("[KS] yekenot FE...", flush=True)
    t0 = time.time()
    X_y = train.drop([ID, TARGET], axis=1).copy()
    X_y_test = test.drop([ID], axis=1).copy()
    cat_cols = X_y.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_y.select_dtypes(exclude=["object"]).columns.tolist()
    cmap: dict = {}
    X_y, new_cat, new_num, combo_names = yekenot_fe(
        X_y, num_cols, cat_cols, cmap, fit=True
    )
    X_y_test, _, _, _ = yekenot_fe(
        X_y_test, num_cols, cat_cols, cmap, fit=False
    )
    cat_cols += new_cat
    label_encode(X_y, X_y_test, cat_cols)
    X_y = to_float(X_y).reset_index(drop=True)
    X_y_test = to_float(X_y_test).reset_index(drop=True)
    print(f"  yekenot: train {X_y.shape}  test {X_y_test.shape}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    print("[KS] constraint violations...", flush=True)
    t1 = time.time()
    train_for_viol = train.copy().reset_index(drop=True)
    test_for_viol = test.copy().reset_index(drop=True)
    # Compute laptime stats on TRAIN only, apply same to test (avoids
    # combined-frame transformation; AV-AUC=0.502 makes either safe but
    # train-fit is cleaner per Rule 25).
    viol_train, lt_stats = compute_violations(train_for_viol)
    viol_test, _ = compute_violations(test_for_viol, laptime_stats=lt_stats)
    viol_cols = list(viol_train.columns)
    print(f"  constraint: {len(viol_cols)} feats in {time.time()-t1:.1f}s",
          flush=True)
    print(f"  viol cols: {viol_cols}", flush=True)
    # Reset the index alignment to standard 0..N-1
    viol_train = viol_train.reset_index(drop=True).astype(np.float32)
    viol_test = viol_test.reset_index(drop=True).astype(np.float32)

    print("[KS] inter-stint features...", flush=True)
    t2 = time.time()
    res = build_inter_stint_features(train_for_viol, test_for_viol)
    # build_inter_stint_features returns (df_combined, feat_names) or
    # (X_tr, X_te, feat_names) depending on signature. Check both.
    if len(res) == 2:
        df_comb, feat_names = res
        n_tr = len(train_for_viol)
        is_train = df_comb["_split"].values == "tr" if "_split" in df_comb.columns else None
        # Fallback: first n_tr rows are train (build_inter_stint sorts by row_id)
        is_inter_train = df_comb.iloc[:n_tr][feat_names].astype(np.float32
                                                                ).reset_index(drop=True)
        is_inter_test = df_comb.iloc[n_tr:][feat_names].astype(np.float32
                                                                 ).reset_index(drop=True)
    else:
        is_inter_train, is_inter_test, feat_names = res
        is_inter_train = is_inter_train[feat_names].astype(np.float32
                                                           ).reset_index(drop=True)
        is_inter_test = is_inter_test[feat_names].astype(np.float32
                                                         ).reset_index(drop=True)
    print(f"  inter-stint: {len(feat_names)} feats in {time.time()-t2:.1f}s",
          flush=True)
    print(f"  inter-stint cols: {feat_names}", flush=True)

    # Concatenate (preserving train row order)
    X_full_train = pd.concat(
        [X_y, viol_train.add_prefix("viol_"), is_inter_train.add_prefix("is_")],
        axis=1
    )
    X_full_test = pd.concat(
        [X_y_test, viol_test.add_prefix("viol_"),
         is_inter_test.add_prefix("is_")],
        axis=1
    )
    # The viol_ prefix is already in compute_violations cols; double-prefix
    # gives viol_viol_C1 — fine, just keep names unique. Same for is_.
    print(f"[KS] total tableau: train {X_full_train.shape}  "
          f"test {X_full_test.shape}", flush=True)
    return X_full_train, X_full_test, combo_names


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--min-samples-leaf", type=int, default=100)
    ap.add_argument("--max-samples", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-json",
                    default="scripts/artifacts/probe_forest_kitchen_sink.json")
    args = ap.parse_args()

    t_total = time.time()
    print("[KS] loading data and K=4 pool", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))
    k4_oof, k4_test = load_k4_pool()

    # PRIMARY-test reference
    F_test_k4 = expand(k4_test)
    F_oof_k4 = expand(k4_oof)
    lr_full = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr_full.fit(F_oof_k4, y)
    primary_test_pos = lr_full.predict_proba(F_test_k4)[:, 1]

    # Build kitchen-sink features
    X_train, X_test, combo_names = build_kitchen_sink_features(train, test)

    # The yekenot combo_names (Race_Compound_, Race_Year_) need per-fold
    # CV target encoding. Other features are static across folds.
    print(f"[KS] combo_names for per-fold TE: {combo_names}", flush=True)

    n_estimators = args.n_estimators
    print(f"[KS] RF settings: n_estimators={n_estimators} "
          f"min_samples_leaf={args.min_samples_leaf} "
          f"max_samples={args.max_samples}", flush=True)

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    fold_walls = []

    for k, (tr, va) in enumerate(splits):
        t1 = time.time()
        # Per-fold CV target encoding on combo_names (yekenot's #6)
        TE = TargetEncoder(cv=N_FOLDS, smooth="auto", shuffle=True,
                           random_state=42)
        tr_te = TE.fit_transform(X_train[combo_names].iloc[tr], y[tr])
        va_te = TE.transform(X_train[combo_names].iloc[va])
        tst_te = TE.transform(X_test[combo_names])
        te_names = [f"_{c}TE" for c in combo_names]

        X_tr = X_train.iloc[tr].drop(columns=combo_names).copy()
        X_va = X_train.iloc[va].drop(columns=combo_names).copy()
        X_ts = X_test.drop(columns=combo_names).copy()
        X_tr[te_names] = tr_te
        X_va[te_names] = va_te
        X_ts[te_names] = tst_te

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=args.min_samples_leaf,
            max_samples=args.max_samples,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        clf.fit(X_tr.values, y[tr])
        oof[va] = clf.predict_proba(X_va.values)[:, 1]
        test_pred += clf.predict_proba(X_ts.values)[:, 1] / N_FOLDS

        s = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(s)
        fold_walls.append(time.time() - t1)
        print(f"  fold {k+1}/{N_FOLDS}: AUC={s:.5f}  "
              f"wall={time.time()-t1:.1f}s", flush=True)

    oof_auc = float(roc_auc_score(y, oof))
    print(f"\n[KS] OOF AUC: {oof_auc:.5f}", flush=True)

    name = f"rf_kitchen_sink_seed{args.seed}" if args.seed != 42 else "rf_kitchen_sink"
    np.save(ART / f"oof_{name}_strat.npy",
            np.column_stack([1 - oof, oof]).astype(np.float32))
    np.save(ART / f"test_{name}_strat.npy",
            np.column_stack([1 - test_pred, test_pred]).astype(np.float32))

    # Gate
    rho, _ = spearmanr(test_pred, primary_test_pos)
    rho = float(rho)
    F_base = expand(k4_oof)
    F_with = expand(np.column_stack([k4_oof, oof]))
    base_oof = lr_meta_oof(F_base, y, splits)
    with_oof = lr_meta_oof(F_with, y, splits)
    auc_base = float(roc_auc_score(y, base_oof))
    auc_with = float(roc_auc_score(y, with_oof))
    delta_bp = (auc_with - auc_base) * 1e4
    pred_lb = pred_lb_band(delta_bp, rho)

    print(f"\n[KS] --- gate ---", flush=True)
    print(f"  standalone OOF: {oof_auc:.5f}", flush=True)
    print(f"  K=4 LR-meta base OOF: {auc_base:.5f}", flush=True)
    print(f"  K=4+1 LR-meta with-OOF: {auc_with:.5f}", flush=True)
    print(f"  Δ min-meta: {delta_bp:+.3f} bp", flush=True)
    print(f"  ρ vs PRIMARY: {rho:.5f}", flush=True)
    print(f"  predicted LB Δ band: {pred_lb:+.2f} bp", flush=True)
    print(f"  total wall: {(time.time()-t_total)/60:.1f} min", flush=True)

    # Comparison vs Angle A baseline
    print(f"\n[KS] --- vs Angle A (RF-yekenot, +0.26 bp) ---", flush=True)
    print(f"  Angle A:  standalone 0.94178  ρ 0.9595  Δ +0.262 bp", flush=True)
    print(f"  Kitchen:  standalone {oof_auc:.5f}  ρ {rho:.4f}  "
          f"Δ {delta_bp:+.3f} bp", flush=True)

    res = dict(
        name=name,
        n_estimators=n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_samples=args.max_samples,
        seed=args.seed,
        n_features=int(X_train.shape[1] - len(combo_names) + len(combo_names)),
        feature_families=[
            "yekenot_recipe (~38)",
            "constraint_violations (12)",
            "inter_stint_memory (7)",
        ],
        oof_auc=oof_auc,
        fold_aucs=fold_aucs,
        fold_walls_s=fold_walls,
        rho_vs_primary=rho,
        k4_lr_base_oof=auc_base,
        k4_lr_with_oof=auc_with,
        min_meta_delta_bp=float(delta_bp),
        predicted_lb_delta_bp_conservative_band=float(pred_lb),
        wall_min=(time.time() - t_total) / 60,
        compared_to_angle_a=dict(
            angle_a_oof=0.94178,
            angle_a_rho=0.9595,
            angle_a_delta_bp=0.262,
            kitchen_minus_a_oof_bp=(oof_auc - 0.94178) * 1e4,
            kitchen_minus_a_delta_bp=delta_bp - 0.262,
        ),
    )
    Path(args.out_json).write_text(json.dumps(res, indent=2))
    print(f"\nSaved {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
