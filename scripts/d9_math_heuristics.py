"""Day-9 — 10 simple math/heuristic rule_residual probes.

Mirrors the F1.2 multi-rule template (`scripts/d6_multi_rule.py`):
each approach produces a rule probability vector, a residual HGBC is
fit on raw features, and the standalone OOF + ρ vs PRIMARY + minimal-
meta sanity check are collected for triage.

Strat-only (R1). 5-fold, SEED=42. No submissions written; this is an
EV-gathering pass. Each approach's OOF/test arrays are saved so a
follow-up K=N stack can be built if the min-meta gate passes.

Approaches:
  R5  weibull_compound       — per-Compound Weibull(k, λ) hazard h(TyreLife).
  R6  next_compound          — 1-step lookup p(pit | curr × next × stint_q).
  R7  prev_compound          — 1-step lookup p(pit | prev × curr × laps_in_stint_q).
  R8  position_progress      — Position-decile × RaceProgress-decile lookup.
  R9  laptime_delta_z        — per-Compound z-score sigmoid of LapTime_Delta.
  R10 driver_eb              — per-Driver Beta-Binomial empirical-Bayes pit rate.
  R11 stint_overdue          — (Compound, Stint#) median-stint excess heuristic.
  R12 cumdeg_knee            — per-Compound 2-segment knee on Cumulative_Degradation.
  R13 race_lapbin            — Race × within-race lap-bin lookup.
  R14 hash_lr_3way           — sparse one-hot LR over Driver×Compound×Stint.

R14 has no residual stage; it is a pure simple-ML base.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
M5Q_S, M5Q_LB = 0.95057, 0.95005
PRIMARY_S, PRIMARY_LB = 0.95065, 0.95026  # d6_k18_multi_rule
RHO_TIE = 0.999
ALPHA = 50.0  # Bayesian smoothing

# HGBC residual params — match d6_multi_rule for std-OOF comparability
def make_hgbc_regressor():
    return HistGradientBoostingRegressor(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
        categorical_features="from_dtype",
    )


def encode_features(X, X_test):
    HIGH_CARD = ["Driver"]
    LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True
                             ).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mp).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")
    return X, X_test


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1]


# ---------------------------------------------------------------------
# Approach kernels — each returns (rp_tr, rp_va, rp_te) given
# (train_full, test_full, tr_idx, va_idx, y_tr). Three slices per
# fold so we can fit the residual on tr, evaluate on va, predict te.
# Returned arrays are clipped to (1e-9, 1-1e-9).
# ---------------------------------------------------------------------

def _clip(p):
    return np.clip(p, 1e-9, 1.0 - 1e-9)


def _smoothed_lookup(keys_tr, y_tr, keys_apply_lists, alpha=ALPHA):
    df = pd.DataFrame({"k": keys_tr, "y": y_tr})
    g = df.groupby("k", observed=True)["y"]
    counts = g.count(); means = g.mean()
    glob = float(np.mean(y_tr))
    smoothed = ((means * counts + glob * alpha) / (counts + alpha)).to_dict()
    return [np.array([smoothed.get(k, glob) for k in keys], dtype=np.float64)
            for keys in keys_apply_lists]


def _decile_edges(arr, n_bins=10):
    edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return edges


def _bin(arr, edges):
    return np.clip(np.searchsorted(edges, arr, side="right") - 1,
                   0, len(edges) - 2).astype(np.int32)


def rule_weibull_compound(train, test, tr, va, y_tr):
    """R5: per-Compound Weibull(k, λ) hazard fit on observed pit-event
    TyreLife distributions. h(t|c) = (k/λ)(t/λ)^(k-1). Pure parametric."""
    df_tr = pd.DataFrame({"Compound": train["Compound"].values[tr],
                          "TyreLife": train["TyreLife"].values[tr], "y": y_tr})
    haz_lookup = {}
    glob_k, glob_lam = 1.5, 25.0
    for c, g in df_tr.groupby("Compound", observed=True):
        events = g.loc[g["y"] == 1, "TyreLife"].values
        if len(events) < 50:
            haz_lookup[c] = (glob_k, glob_lam)
            continue
        m1 = float(np.mean(events))
        v1 = float(np.var(events) + 1e-6)
        cv = (np.sqrt(v1) / max(m1, 1e-3))
        k_est = max(0.8, min(4.0, 1.0 / max(cv, 0.2)))
        lam_est = max(5.0, m1 / max(0.886, 1.0 - 0.5 / k_est))
        haz_lookup[c] = (float(k_est), float(lam_est))
    def haz(c_arr, t_arr):
        out = np.zeros(len(c_arr), dtype=np.float64)
        for c in np.unique(c_arr):
            k, lam = haz_lookup.get(c, (glob_k, glob_lam))
            mask = (c_arr == c)
            t = np.maximum(t_arr[mask], 1e-3)
            out[mask] = (k / lam) * (t / lam) ** (k - 1)
        return _clip(1.0 - np.exp(-out))
    return (
        haz(train["Compound"].values[tr], train["TyreLife"].values[tr]),
        haz(train["Compound"].values[va], train["TyreLife"].values[va]),
        haz(test["Compound"].values, test["TyreLife"].values),
    )


def _compute_neighbour_compounds(df):
    """For each row, compute next_compound and prev_compound within the
    same (Year, Race, Driver) ordered by LapNumber. Returns two
    np.array of strings ('UNK' where unavailable). Leak-free: pure
    sequential lookup, no target involvement."""
    sort_idx = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).index
    s = df.loc[sort_idx].copy()
    grp = s.groupby(["Year", "Race", "Driver"], sort=False)
    s["next_compound"] = grp["Compound"].shift(-1).fillna("UNK").astype(str)
    s["prev_compound"] = grp["Compound"].shift(+1).fillna("UNK").astype(str)
    s = s.sort_index()
    return s["next_compound"].values, s["prev_compound"].values


def rule_next_compound(train, test, tr, va, y_tr,
                       next_train, next_test):
    """R6: lookup p(pit | curr_compound × next_compound × stint_quintile)."""
    sq_train = train["Stint"].clip(upper=4).astype(int).values
    sq_test = test["Stint"].clip(upper=4).astype(int).values
    keys_tr = list(zip(train["Compound"].values[tr], next_train[tr], sq_train[tr]))
    keys_va = list(zip(train["Compound"].values[va], next_train[va], sq_train[va]))
    keys_te = list(zip(test["Compound"].values, next_test, sq_test))
    return _smoothed_lookup(keys_tr, y_tr, [keys_tr, keys_va, keys_te])


def rule_prev_compound(train, test, tr, va, y_tr,
                       prev_train, prev_test):
    """R7: lookup p(pit | prev × curr × laps_into_stint_quintile)."""
    edges = _decile_edges(train["TyreLife"].values[tr], n_bins=5)
    lis_train = _bin(train["TyreLife"].values, edges)
    lis_test = _bin(test["TyreLife"].values, edges)
    keys_tr = list(zip(prev_train[tr], train["Compound"].values[tr], lis_train[tr]))
    keys_va = list(zip(prev_train[va], train["Compound"].values[va], lis_train[va]))
    keys_te = list(zip(prev_test, test["Compound"].values, lis_test))
    return _smoothed_lookup(keys_tr, y_tr, [keys_tr, keys_va, keys_te])


def rule_position_progress(train, test, tr, va, y_tr):
    """R8: Position-decile × RaceProgress-decile lookup."""
    pos_edges = _decile_edges(train["Position"].values[tr])
    rp_edges = _decile_edges(train["RaceProgress"].values[tr])
    pos_all = _bin(train["Position"].values, pos_edges)
    rp_all = _bin(train["RaceProgress"].values, rp_edges)
    pos_te = _bin(test["Position"].values, pos_edges)
    rp_te = _bin(test["RaceProgress"].values, rp_edges)
    keys_tr = list(zip(pos_all[tr], rp_all[tr]))
    keys_va = list(zip(pos_all[va], rp_all[va]))
    keys_te = list(zip(pos_te, rp_te))
    return _smoothed_lookup(keys_tr, y_tr, [keys_tr, keys_va, keys_te])


def rule_laptime_delta_z(train, test, tr, va, y_tr):
    """R9: per-Compound z-score sigmoid of LapTime_Delta. Closed-form."""
    df_tr = pd.DataFrame({"Compound": train["Compound"].values[tr],
                          "LapTime_Delta": train["LapTime_Delta"].values[tr]})
    stats = df_tr.groupby("Compound", observed=True)["LapTime_Delta"].agg(["mean", "std"])
    glob_mean = float(df_tr["LapTime_Delta"].mean())
    glob_std = float(df_tr["LapTime_Delta"].std() + 1e-6)
    def score(c_arr, d_arr):
        mu = np.array([stats["mean"].get(ci, glob_mean) for ci in c_arr])
        sd = np.array([stats["std"].get(ci, glob_std) for ci in c_arr]) + 1e-6
        z = (d_arr - mu) / sd
        return _clip(1.0 / (1.0 + np.exp(-(z - 1.0))))
    return (
        score(train["Compound"].values[tr], train["LapTime_Delta"].values[tr]),
        score(train["Compound"].values[va], train["LapTime_Delta"].values[va]),
        score(test["Compound"].values, test["LapTime_Delta"].values),
    )


def rule_driver_eb(train, test, tr, va, y_tr, alpha_eb=20.0):
    """R10: Beta-Binomial empirical-Bayes per-Driver pit rate."""
    df_tr = pd.DataFrame({"Driver": train["Driver"].values[tr], "y": y_tr})
    g = df_tr.groupby("Driver", observed=True)["y"]
    counts = g.count(); sums = g.sum()
    glob = float(np.mean(y_tr))
    eb = ((sums + alpha_eb * glob) / (counts + alpha_eb)).to_dict()
    def score(d_arr):
        return np.array([eb.get(d, glob) for d in d_arr], dtype=np.float64)
    return (
        score(train["Driver"].values[tr]),
        score(train["Driver"].values[va]),
        score(test["Driver"].values),
    )


def rule_stint_overdue(train, test, tr, va, y_tr):
    """R11: (Compound, Stint) median-stint length λ; row score = σ((tyre - λ)/scale)."""
    df_tr = pd.DataFrame({"Compound": train["Compound"].values[tr],
                          "Stint": train["Stint"].values[tr],
                          "TyreLife": train["TyreLife"].values[tr], "y": y_tr})
    events = df_tr[df_tr["y"] == 1]
    lam = events.groupby(["Compound", "Stint"], observed=True)["TyreLife"].median()
    glob_lam = float(events["TyreLife"].median()) if len(events) else 25.0
    def score(c_arr, s_arr, t_arr):
        s_clipped = np.clip(s_arr.astype(int), 1, 5)
        keys = list(zip(c_arr, s_clipped))
        lam_arr = np.array([lam.get(k, glob_lam) for k in keys], dtype=np.float64)
        scale = np.maximum(0.25 * lam_arr, 1.0)
        z = (t_arr - lam_arr) / scale
        return _clip(1.0 / (1.0 + np.exp(-z)))
    return (
        score(train["Compound"].values[tr], train["Stint"].values[tr],
              train["TyreLife"].values[tr]),
        score(train["Compound"].values[va], train["Stint"].values[va],
              train["TyreLife"].values[va]),
        score(test["Compound"].values, test["Stint"].values,
              test["TyreLife"].values),
    )


def rule_cumdeg_knee(train, test, tr, va, y_tr):
    """R12: per-Compound 2-segment piecewise-linear knee on Cumulative_Degradation."""
    df_tr = pd.DataFrame({
        "Compound": train["Compound"].values[tr],
        "cd": train["Cumulative_Degradation"].values[tr], "y": y_tr,
    })
    knees, scales = {}, {}
    glob_cd = float(np.median(df_tr["cd"]))
    glob_scale = float(np.std(df_tr["cd"]) + 1e-6)
    for c, g in df_tr.groupby("Compound", observed=True):
        if len(g) < 200:
            knees[c] = glob_cd; scales[c] = glob_scale; continue
        cd = g["cd"].values; y = g["y"].values
        qs = np.quantile(cd, np.linspace(0.2, 0.9, 15))
        best_auc, best_k = 0.0, glob_cd
        for k in qs:
            ind = (cd > k).astype(np.float64)
            try:
                a = roc_auc_score(y, ind)
                if a > best_auc:
                    best_auc, best_k = a, k
            except ValueError:
                pass
        knees[c] = float(best_k)
        scales[c] = float(np.std(cd) + 1e-6)
    def score(c_arr, cd_arr):
        k = np.array([knees.get(ci, glob_cd) for ci in c_arr])
        s = np.array([scales.get(ci, glob_scale) for ci in c_arr]) + 1e-6
        return _clip(1.0 / (1.0 + np.exp(-((cd_arr - k) / s))))
    return (
        score(train["Compound"].values[tr], train["Cumulative_Degradation"].values[tr]),
        score(train["Compound"].values[va], train["Cumulative_Degradation"].values[va]),
        score(test["Compound"].values, test["Cumulative_Degradation"].values),
    )


def rule_race_lapbin(train, test, tr, va, y_tr):
    """R13: Race × within-race lap-bin (RaceProgress decile) lookup."""
    rp_train = np.clip(np.floor(train["RaceProgress"].values * 10).astype(int), 0, 9)
    rp_test = np.clip(np.floor(test["RaceProgress"].values * 10).astype(int), 0, 9)
    keys_tr = list(zip(train["Race"].values[tr], rp_train[tr]))
    keys_va = list(zip(train["Race"].values[va], rp_train[va]))
    keys_te = list(zip(test["Race"].values, rp_test))
    return _smoothed_lookup(keys_tr, y_tr, [keys_tr, keys_va, keys_te])


def rule_hash_lr_3way(train, test, tr, va, y_tr,
                      hash_train=None, hash_test=None):
    """R14: sparse one-hot LR over Driver × Compound × Stint. No residual."""
    lr = LogisticRegression(C=1.0, max_iter=200, solver="liblinear", n_jobs=1)
    lr.fit(hash_train[tr], y_tr)
    rp_tr = lr.predict_proba(hash_train[tr])[:, 1]
    rp_va = lr.predict_proba(hash_train[va])[:, 1]
    rp_te = lr.predict_proba(hash_test)[:, 1]
    return _clip(rp_tr), _clip(rp_va), _clip(rp_te)


def _make_hash_features(df):
    d = df["Driver"].astype(str).values
    c = df["Compound"].astype(str).values
    s = df["Stint"].clip(upper=5).astype(int).astype(str).values
    rows = [[f"D={di}", f"C={ci}", f"S={si}",
             f"DC={di}|{ci}", f"CS={ci}|{si}", f"DS={di}|{si}",
             f"DCS={di}|{ci}|{si}"]
            for di, ci, si in zip(d, c, s)]
    h = FeatureHasher(n_features=2**16, input_type="string", alternate_sign=False)
    return h.transform(rows)


# ---------------------------------------------------------------------
# Builder loop
# ---------------------------------------------------------------------

def build_with_residual(approach_name, rule_fn, train, test, X_enc, X_test_enc,
                        y, splits, has_residual=True, **rule_kwargs):
    print(f"\n--- {approach_name} (residual={has_residual}) ---")
    n_train, n_test = len(train), len(test)
    oof_full = np.zeros(n_train, dtype=np.float64)
    test_full = np.zeros(n_test, dtype=np.float64)
    rule_aucs, full_aucs, walls = [], [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        rp_tr, rp_va, rp_te = rule_fn(train, test, tr, va, y[tr], **rule_kwargs)
        if has_residual:
            m = make_hgbc_regressor()
            m.fit(X_enc.iloc[tr], y[tr].astype(np.float64) - rp_tr)
            resid_va = m.predict(X_enc.iloc[va])
            resid_te = m.predict(X_test_enc)
            pred_va = _clip(rp_va + resid_va)
            pred_te = _clip(rp_te + resid_te)
        else:
            pred_va = _clip(rp_va)
            pred_te = _clip(rp_te)
        oof_full[va] = pred_va
        test_full += pred_te / N_FOLDS
        s_rule = float(roc_auc_score(y[va], rp_va))
        s_full = float(roc_auc_score(y[va], pred_va))
        wall = time.time() - t0
        rule_aucs.append(s_rule); full_aucs.append(s_full); walls.append(wall)
        print(f"  f{k}: rule={s_rule:.5f}  full={s_full:.5f}  wall={wall:.1f}s")
    auc_full = float(roc_auc_score(y, oof_full))
    print(f"  → standalone OOF: {auc_full:.5f}  (rule-mean fold AUC {np.mean(rule_aucs):.5f})  "
          f"total wall={sum(walls):.1f}s")
    return oof_full, test_full, auc_full


def main():
    t_total = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # Pre-compute next/prev compounds (no target leakage; sequential lookup)
    print("Computing next/prev compound on train+test ...")
    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    nx, pv = _compute_neighbour_compounds(full)
    next_train = nx[full["__src"].values == "tr"]
    next_test = nx[full["__src"].values == "te"]
    prev_train = pv[full["__src"].values == "tr"]
    prev_test = pv[full["__src"].values == "te"]
    cov_next_test = float(np.mean(next_test != "UNK"))
    cov_prev_test = float(np.mean(prev_test != "UNK"))
    print(f"  next_compound test coverage: {cov_next_test:.3f}")
    print(f"  prev_compound test coverage: {cov_prev_test:.3f}")

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    X_test = test.drop(columns=[ID_COL], errors="ignore").copy()
    X_enc, X_test_enc = encode_features(X.copy(), X_test.copy())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # PRIMARY artifacts for ρ + minimal-meta
    m5q_oof = np.load(ART / "oof_m5q_strat.npy")[:, 1].astype(np.float64)
    m5q_test = np.load(ART / "test_m5q_strat.npy")[:, 1].astype(np.float64)
    primary_oof = np.load(ART / "oof_d6_k18_multi_rule_strat.npy"
                          )[:, 1].astype(np.float64)
    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy"
                           )[:, 1].astype(np.float64)

    print("Pre-computing R14 sparse hash matrix ...")
    hash_train = _make_hash_features(train)
    hash_test = _make_hash_features(test)

    # Approach registry
    approaches = [
        ("R5_weibull_compound", rule_weibull_compound, True, {}),
        ("R6_next_compound", rule_next_compound, True,
         dict(next_train=next_train, next_test=next_test)),
        ("R7_prev_compound", rule_prev_compound, True,
         dict(prev_train=prev_train, prev_test=prev_test)),
        ("R8_position_progress", rule_position_progress, True, {}),
        ("R9_laptime_delta_z", rule_laptime_delta_z, True, {}),
        ("R10_driver_eb", rule_driver_eb, True, {}),
        ("R11_stint_overdue", rule_stint_overdue, True, {}),
        ("R12_cumdeg_knee", rule_cumdeg_knee, True, {}),
        ("R13_race_lapbin", rule_race_lapbin, True, {}),
        ("R14_hash_lr_3way", rule_hash_lr_3way, False,
         dict(hash_train=hash_train, hash_test=hash_test)),
    ]

    results = {}
    coverages = dict(next_compound=cov_next_test, prev_compound=cov_prev_test)
    for name, fn, has_resid, kwargs in approaches:
        oof, tp, auc = build_with_residual(
            name, fn, train, test, X_enc, X_test_enc, y, splits,
            has_residual=has_resid, **kwargs,
        )
        rho_test_m5q, _ = spearmanr(tp, m5q_test)
        rho_test_primary, _ = spearmanr(tp, primary_test)
        # Minimal-meta vs PRIMARY (K=2 LR)
        F_min = expand(np.column_stack([primary_oof, oof]))
        F_min_t = expand(np.column_stack([primary_test, tp]))
        mo_min, _ = fit_lr_meta(F_min, F_min_t, y)
        auc_min = float(roc_auc_score(y, mo_min))
        # Save artifacts
        np.save(ART / f"oof_d9_{name}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d9_{name}_strat.npy",
                np.column_stack([1 - tp, tp]))
        results[name] = dict(
            standalone_oof=auc,
            rho_vs_m5q_test=float(rho_test_m5q),
            rho_vs_primary_test=float(rho_test_primary),
            minimal_meta_oof=auc_min,
            minimal_meta_delta_primary_bp=(auc_min - PRIMARY_S) * 1e4,
            minimal_meta_pass=bool(auc_min >= PRIMARY_S),
            has_residual=has_resid,
        )
        r = results[name]
        print(f"  ρ vs M5q test: {r['rho_vs_m5q_test']:.5f}   "
              f"ρ vs PRIMARY test: {r['rho_vs_primary_test']:.5f}")
        print(f"  Minimal-meta OOF: {auc_min:.5f}  Δ PRIMARY "
              f"{r['minimal_meta_delta_primary_bp']:+.2f}bp  "
              f"{'PASS ✓' if r['minimal_meta_pass'] else 'FAIL ✗'}")

    final = dict(
        approaches=results,
        coverages=coverages,
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        m5q=dict(strat_oof=M5Q_S, lb=M5Q_LB),
        total_wall_s=time.time() - t_total,
    )
    (ART / "d9_math_heuristics_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9_math_heuristics_results.json  "
          f"(total wall {time.time()-t_total:.0f}s)")

    # Triage report
    print("\n" + "=" * 78)
    print(f"{'approach':<24s} {'std_OOF':>8s}  {'ρ_PRIM':>7s}  {'Δprim_bp':>9s}  verdict")
    print("-" * 78)
    for name, r in results.items():
        verdict = "PASS" if r["minimal_meta_pass"] else "FAIL"
        if r["minimal_meta_pass"] and r["rho_vs_primary_test"] < RHO_TIE:
            verdict = "PASS+DIVERSE"
        print(f"{name:<24s} {r['standalone_oof']:>8.5f}  "
              f"{r['rho_vs_primary_test']:>7.5f}  "
              f"{r['minimal_meta_delta_primary_bp']:>+8.2f}  {verdict}")


if __name__ == "__main__":
    main()
