"""scripts/probe_forest_optuna.py — Optuna over RF on kitchen-sink features.

Follow-up to `probe_forest_kitchen_sink.py`. The default RF settings
(n_estimators=400, sqrt features, leaf=100, max_samples=0.5) gave a
+0.248 bp K=4+1 LR-meta lift on a 57-feature kitchen-sink (vs +0.262 bp
on yekenot-only at 38 feat). The lift is reproducible across two RF
configs but small. PI directive: tune RF hyperparameters with Optuna
to push past +0.25 bp.

Strategy:
  - Search-time objective is the **K=4+1 LR-meta delta in bp** under a
    single-fold proxy (fold 0 only, reduced n_estimators=200). Each
    trial fits RF on 80% of the data, computes that fold's RF
    predictions for the val rows, replaces fold-0 of the K=4 LR-meta
    OOF with the new RF base added, and reports the AUC delta on
    fold-0 val rows.
  - The search-time proxy is biased toward overfit: a winning trial
    might just be a fold-0 fluke. Validate the top config at full
    5-fold AND with a second seed before declaring a real lift.

Search space:
  - n_estimators: [300, 800]
  - max_features: {"sqrt", "log2", 0.3, 0.5}
  - min_samples_leaf: [30, 50, 100, 200, 400]
  - max_samples: {0.3, 0.5, 0.7, None}
  - max_depth: {None, 10, 15, 20}
  - criterion: {"gini", "entropy"}
  - bootstrap: True (RF default; ExtraTrees-style with bootstrap=False
    is a separate variant tested below as a final pin)

Cost target:
  - 15 trials × ~40s per trial proxy = ~10 min
  - Top config at 5-fold seed=42: ~5 min
  - Same config at 5-fold seed=7 (validation): ~5 min
  - Total: ~20-25 min CPU.

Saves:
  - scripts/artifacts/probe_forest_optuna.json — search results +
    validation
  - scripts/artifacts/oof_rf_optuna_best_strat.npy — winning RF base
  - scripts/artifacts/test_rf_optuna_best_strat.npy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_forest_sweep import (  # type: ignore
    K4_BASES, expand, lr_meta_oof, pred_lb_band,
)
from probe_forest_kitchen_sink import build_kitchen_sink_features  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
N_FOLDS, MAX_ITER = 5, 500


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def load_k4_pool():
    oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K4_BASES]
    tests = [_pos(ART / f"test_{b}_strat.npy") for b in K4_BASES]
    return np.column_stack(oofs), np.column_stack(tests)


def fit_lr_proxy_fold(F_oof, y, splits, fold_idx):
    """Fit LR on the meta-train rows of fold `fold_idx`; predict val rows."""
    tr, va = splits[fold_idx]
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F_oof[tr], y[tr])
    return lr.predict_proba(F_oof[va])[:, 1], va


def make_objective(X_train, X_test, combo_names, y, splits, k4_oof):
    """Return Optuna objective: maximize K=4+1 LR-meta lift on fold 0."""
    fold_idx = 0
    tr_full, va_full = splits[fold_idx]

    # Pre-compute baseline LR on K=4 expansion (no candidate base)
    F_base = expand(k4_oof)
    base_pred, va_idx = fit_lr_proxy_fold(F_base, y, splits, fold_idx)
    base_auc = float(roc_auc_score(y[va_idx], base_pred))

    # Pre-fit per-fold target encoder (same TE for all trials — fast)
    te = TargetEncoder(cv=N_FOLDS, smooth="auto", shuffle=True,
                       random_state=42)
    tr_te = te.fit_transform(X_train[combo_names].iloc[tr_full], y[tr_full])
    va_te = te.transform(X_train[combo_names].iloc[va_full])
    tst_te = te.transform(X_test[combo_names])
    te_names = [f"_{c}TE" for c in combo_names]

    X_tr_arr = X_train.iloc[tr_full].drop(columns=combo_names).copy()
    X_va_arr = X_train.iloc[va_full].drop(columns=combo_names).copy()
    X_ts_arr = X_test.drop(columns=combo_names).copy()
    X_tr_arr[te_names] = tr_te
    X_va_arr[te_names] = va_te
    X_ts_arr[te_names] = tst_te
    X_tr_arr = X_tr_arr.values
    X_va_arr = X_va_arr.values
    X_ts_arr = X_ts_arr.values
    print(f"  proxy setup done: fold-0 LR-base AUC {base_auc:.5f}",
          flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 800, step=100),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5]),
            min_samples_leaf=trial.suggest_categorical(
                "min_samples_leaf", [30, 50, 100, 200, 400]),
            max_samples=trial.suggest_categorical(
                "max_samples", [0.3, 0.5, 0.7, None]),
            max_depth=trial.suggest_categorical(
                "max_depth", [None, 10, 15, 20]),
            criterion=trial.suggest_categorical(
                "criterion", ["gini", "entropy"]),
        )
        clf = RandomForestClassifier(
            **params,
            n_jobs=-1, random_state=42,
        )
        t0 = time.time()
        clf.fit(X_tr_arr, y[tr_full])
        rf_va_pred = clf.predict_proba(X_va_arr)[:, 1]
        wall = time.time() - t0

        # Build K=4+1 fold-0 proxy: replace candidate column for val rows
        # by using the K=4 OOFs (which are honest by construction) plus
        # the RF prediction on val (and a placeholder on train rows;
        # we don't refit LR meta within the proxy because the LR fit
        # uses meta-train features that include K=4 only — the new
        # base column needs to be present at LR fit time, which it
        # isn't unless we cross-fit. So use a simpler proxy:
        # fit LR-meta with the K=4 expansion only, then evaluate the
        # AUC on a blend = α*LR_base + (1-α)*RF_pred over a small grid.
        # The blend AUC ceiling is the "best-case LR meta lift" if
        # the meta knew the new base. This is the upper bound for the
        # min-meta gate at full 5-fold.
        best_auc = base_auc
        for alpha in (0.85, 0.90, 0.95, 0.98):
            blend = alpha * base_pred + (1 - alpha) * rf_va_pred
            a = float(roc_auc_score(y[va_idx], blend))
            if a > best_auc:
                best_auc = a
        delta_bp = (best_auc - base_auc) * 1e4

        trial.set_user_attr("wall_s", wall)
        trial.set_user_attr("rf_va_pred_mean", float(rf_va_pred.mean()))
        return delta_bp

    return objective, base_auc


def fit_rf_5fold(X_train, X_test, combo_names, y, splits, params, seed):
    """Full 5-fold RF fit with given params; returns oof, test_pred."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        t1 = time.time()
        te = TargetEncoder(cv=N_FOLDS, smooth="auto", shuffle=True,
                           random_state=seed)
        tr_te = te.fit_transform(X_train[combo_names].iloc[tr], y[tr])
        va_te = te.transform(X_train[combo_names].iloc[va])
        tst_te = te.transform(X_test[combo_names])
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr = X_train.iloc[tr].drop(columns=combo_names).copy()
        X_va = X_train.iloc[va].drop(columns=combo_names).copy()
        X_ts = X_test.drop(columns=combo_names).copy()
        X_tr[te_names] = tr_te
        X_va[te_names] = va_te
        X_ts[te_names] = tst_te
        clf = RandomForestClassifier(
            **params, n_jobs=-1, random_state=seed,
        )
        clf.fit(X_tr.values, y[tr])
        oof[va] = clf.predict_proba(X_va.values)[:, 1]
        test_pred += clf.predict_proba(X_ts.values)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(s)
        print(f"    fold {k+1}/{N_FOLDS}: AUC={s:.5f}  "
              f"wall={time.time()-t1:.1f}s", flush=True)
    return oof, test_pred, fold_aucs


def gate_at_k4(k4_oof, k4_test, oof, test_pred, y, splits):
    """K=4+1 LR-meta gate. Returns auc_base, auc_with, delta_bp, rho."""
    F_base = expand(k4_oof)
    F_with = expand(np.column_stack([k4_oof, oof]))
    base_oof = lr_meta_oof(F_base, y, splits)
    with_oof = lr_meta_oof(F_with, y, splits)
    auc_base = float(roc_auc_score(y, base_oof))
    auc_with = float(roc_auc_score(y, with_oof))
    delta_bp = (auc_with - auc_base) * 1e4

    # ρ vs PRIMARY-test (LR-on-K=4-expansion fit on all rows)
    F_test_k4 = expand(k4_test)
    F_oof_k4 = expand(k4_oof)
    lr_full = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr_full.fit(F_oof_k4, y)
    primary_test_pos = lr_full.predict_proba(F_test_k4)[:, 1]
    rho, _ = spearmanr(test_pred, primary_test_pos)
    return dict(
        auc_base=auc_base, auc_with=auc_with,
        delta_bp=float(delta_bp),
        rho_vs_primary=float(rho),
        oof_auc_standalone=float(roc_auc_score(y, oof)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--proxy-n-estimators", type=int, default=200,
                    help="n_estimators override during Optuna search")
    ap.add_argument("--seeds-validate", nargs="+", type=int,
                    default=[42, 7])
    ap.add_argument("--out-json",
                    default="scripts/artifacts/probe_forest_optuna.json")
    args = ap.parse_args()

    t_total = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))
    k4_oof, k4_test = load_k4_pool()
    print(f"[OPT] data loaded: train {train.shape}  test {test.shape}",
          flush=True)
    print(f"[OPT] K=4 pool: oof {k4_oof.shape}  test {k4_test.shape}",
          flush=True)

    print("[OPT] building kitchen-sink features ...", flush=True)
    X_train, X_test, combo_names = build_kitchen_sink_features(train, test)
    print(f"[OPT] tableau {X_train.shape}  combo_names {combo_names}",
          flush=True)

    objective, base_auc_fold0 = make_objective(
        X_train, X_test, combo_names, y, splits, k4_oof
    )

    print(f"\n[OPT] starting Optuna study: {args.n_trials} trials, "
          f"proxy n_estimators={args.proxy_n_estimators}", flush=True)
    print("[OPT] objective: best-case fold-0 K=4 LR + RF blend ΔAUC (bp)",
          flush=True)

    # Wrap the objective to override n_estimators with the proxy value
    inner_objective = objective

    def wrapped(trial):
        # Override n_estimators in trial params with proxy value
        # The trial still SUGGESTS n_estimators in [300, 800], but we
        # use the proxy value for the search to keep trials cheap. We
        # log the suggested value so the validation step uses it.
        suggested_n = trial.suggest_int(
            "n_estimators", 300, 800, step=100
        )
        # Re-create RandomForest with proxy n_estimators
        params = dict(
            n_estimators=args.proxy_n_estimators,
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3]),
            min_samples_leaf=trial.suggest_categorical(
                "min_samples_leaf", [50, 100, 200, 400]),
            max_samples=trial.suggest_categorical(
                "max_samples", [0.3, 0.5, 0.7]),
            max_depth=trial.suggest_categorical(
                "max_depth", [10, 12, 15, 18]),
            criterion=trial.suggest_categorical(
                "criterion", ["gini", "entropy"]),
        )
        # Pull pre-computed proxy data via closure
        from sklearn.ensemble import RandomForestClassifier as RFC
        from sklearn.linear_model import LogisticRegression as LR
        tr, va = splits[0]
        # Refit base LR on K=4 fold-0
        F_base = expand(k4_oof)
        lr = LR(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_base[tr], y[tr])
        base_pred = lr.predict_proba(F_base[va])[:, 1]
        base_auc_local = float(roc_auc_score(y[va], base_pred))

        te = TargetEncoder(cv=N_FOLDS, smooth="auto", shuffle=True,
                           random_state=42)
        tr_te = te.fit_transform(X_train[combo_names].iloc[tr], y[tr])
        va_te = te.transform(X_train[combo_names].iloc[va])
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr_arr = X_train.iloc[tr].drop(columns=combo_names).copy()
        X_va_arr = X_train.iloc[va].drop(columns=combo_names).copy()
        X_tr_arr[te_names] = tr_te
        X_va_arr[te_names] = va_te
        X_tr_arr = X_tr_arr.values
        X_va_arr = X_va_arr.values

        clf = RFC(**params, n_jobs=-1, random_state=42)
        t0 = time.time()
        clf.fit(X_tr_arr, y[tr])
        rf_va_pred = clf.predict_proba(X_va_arr)[:, 1]
        wall = time.time() - t0
        # Best-case blend with K=4 LR base on fold-0 val
        best_auc = base_auc_local
        best_alpha = 1.0
        for alpha in (0.80, 0.85, 0.90, 0.95, 0.97, 0.99):
            blend = alpha * base_pred + (1 - alpha) * rf_va_pred
            a = float(roc_auc_score(y[va], blend))
            if a > best_auc:
                best_auc = a
                best_alpha = alpha
        delta_bp = (best_auc - base_auc_local) * 1e4
        trial.set_user_attr("wall_s", wall)
        trial.set_user_attr("best_blend_alpha", best_alpha)
        trial.set_user_attr("suggested_n_estimators_for_validation",
                            suggested_n)
        return delta_bp

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t_search_start = time.time()
    for trial_i in range(args.n_trials):
        trial = study.ask()
        try:
            val = wrapped(trial)
        except Exception as e:
            print(f"  trial {trial_i+1} FAILED: {e}", flush=True)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        study.tell(trial, val)
        wall_s = trial.user_attrs.get("wall_s", 0)
        best_so_far = study.best_value
        print(f"  trial {trial_i+1}/{args.n_trials}: Δ {val:+.3f} bp  "
              f"(best so far {best_so_far:+.3f})  wall={wall_s:.0f}s  "
              f"params {trial.params}",
              flush=True)

    t_search_min = (time.time() - t_search_start) / 60
    print(f"\n[OPT] search done in {t_search_min:.1f} min", flush=True)
    print(f"[OPT] best trial: Δ {study.best_value:+.3f} bp  "
          f"params {study.best_params}", flush=True)

    # Validate top config at full 5-fold + multiple seeds
    best_params = dict(study.best_params)
    # Use the originally-suggested n_estimators for validation (not proxy)
    best_params["n_estimators"] = (
        study.best_trial.user_attrs.get("suggested_n_estimators_for_validation",
                                        best_params.get("n_estimators", 600))
    )
    print(f"\n[OPT] validating top config: {best_params}", flush=True)

    validation_runs = {}
    for seed in args.seeds_validate:
        print(f"\n[OPT] validation seed={seed}", flush=True)
        oof, test_pred, fold_aucs = fit_rf_5fold(
            X_train, X_test, combo_names, y, splits, best_params, seed
        )
        gate = gate_at_k4(k4_oof, k4_test, oof, test_pred, y, splits)
        pred_lb = pred_lb_band(gate["delta_bp"], gate["rho_vs_primary"])
        print(f"[OPT] seed={seed} standalone OOF {gate['oof_auc_standalone']:.5f}",
              flush=True)
        print(f"[OPT] seed={seed} K=4+1 LR-meta Δ {gate['delta_bp']:+.3f} bp  "
              f"ρ {gate['rho_vs_primary']:.4f}  pred LB Δ {pred_lb:+.2f} bp",
              flush=True)
        validation_runs[f"seed_{seed}"] = dict(
            oof_auc_standalone=gate["oof_auc_standalone"],
            fold_aucs=fold_aucs,
            k4_lr_base_oof=gate["auc_base"],
            k4_lr_with_oof=gate["auc_with"],
            min_meta_delta_bp=gate["delta_bp"],
            rho_vs_primary=gate["rho_vs_primary"],
            predicted_lb_delta_bp=float(pred_lb),
        )
        # Save the seed=42 winner as the canonical OOF/test
        if seed == 42:
            np.save(ART / "oof_rf_optuna_best_strat.npy",
                    np.column_stack([1 - oof, oof]).astype(np.float32))
            np.save(ART / "test_rf_optuna_best_strat.npy",
                    np.column_stack([1 - test_pred, test_pred]
                                    ).astype(np.float32))

    # Cross-seed agreement
    if len(args.seeds_validate) >= 2:
        s1, s2 = args.seeds_validate[0], args.seeds_validate[1]
        d1 = validation_runs[f"seed_{s1}"]["min_meta_delta_bp"]
        d2 = validation_runs[f"seed_{s2}"]["min_meta_delta_bp"]
        print(f"\n[OPT] cross-seed agreement on K=4+1 lift: "
              f"seed{s1} {d1:+.3f} bp vs seed{s2} {d2:+.3f} bp  "
              f"|Δ| = {abs(d1-d2):.3f} bp", flush=True)

    # Save
    res = dict(
        n_trials=args.n_trials,
        proxy_n_estimators=args.proxy_n_estimators,
        search_time_min=t_search_min,
        best_proxy_delta_bp=study.best_value,
        best_params=best_params,
        all_trials=[
            dict(
                trial=i,
                params=t.params,
                value=t.value if t.value is not None else None,
                wall_s=t.user_attrs.get("wall_s", 0),
                best_blend_alpha=t.user_attrs.get("best_blend_alpha", 1.0),
            )
            for i, t in enumerate(study.trials)
        ],
        validation=validation_runs,
        comparison_baseline=dict(
            angle_a_oof=0.94178, angle_a_rho=0.9595, angle_a_delta_bp=0.262,
            kitchen_oof=0.94054, kitchen_rho=0.9580, kitchen_delta_bp=0.248,
        ),
        total_wall_min=(time.time() - t_total) / 60,
    )
    Path(args.out_json).write_text(json.dumps(res, indent=2))
    print(f"\n[OPT] Saved {args.out_json}", flush=True)
    print(f"[OPT] total wall: {res['total_wall_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
