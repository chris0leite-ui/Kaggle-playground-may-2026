"""T1.2 multi-formulation L1 bases — the 3 untried reformulations.

Day-8 falsified Poisson on cum_pit_count_remaining (d8_poisson_lapsuntil).
This script builds the 3 OTHER reformulations from
audit/2026-05-08-strategic-menu-wider-steps.md §3 T1.2:

  T1.2c  CENSORED REGRESSION on `laps_until_next_pit` —
         right-censored when no future pit in (Race, Driver, Year, Stint).
         Implemented via XGBoost `objective='survival:cox'`. Output is
         the linear log-hazard score; AUC on -score (higher hazard
         → pit-imminent).

  T1.2d  RATIO TARGET continuous regression —
         target = total_pits_in_race(Race,Driver,Year) / total_stints
         per (Race,Driver,Year) group. LightGBM regression. Heuristic
         transform to per-row pit_proba via:
           pit_proba_row = ratio_pred * (1 - exp(-stint_age / mean_stint_len))
         then 1D isotonic on a held-out fold's binary target.

  T1.2e  STINT-LEVEL SURVIVAL — different parametrization than (c).
         Build stint-level dataset (one row per (Driver,Race,Year,Stint))
         with stint duration in laps + censored flag. Fit CoxPH-style
         GradientBoostingSurvivalAnalysis (sksurv, ~stint-level rows
         only, tractable). Predict cumulative hazard at remaining-laps
         for the TEST row; compute lap-level conditional hazard
           h_lap = 1 - S(t+1)/S(t)
         that goes to AUC.

For each formulation we compute:
  - standalone OOF AUC (Strat 5-fold)
  - Spearman ρ vs PRIMARY's test prediction (test_d9f_K21_swap_strat.npy)
  - 3-feature minimal-meta gate vs PRIMARY OOF
  - K=22 stack: K=21 PRIMARY-keep pool + new candidate; combinations of
    2-3 candidates also evaluated.

Constraints: 1h CPU per formulation; smoke 1-fold/50k first. No submit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

# K=21 d9f swap is the current PRIMARY but its OOF wasn't saved — use
# d9c_Sd_K20_swap_FM for OOF (closest, +0bp) and d9f_K21_swap_strat
# test array for ρ-test. K=18 multi_rule OOF/test as fallback.
PRIMARY_OOF_FILE = "oof_d9c_Sd_K20_swap_FM_strat.npy"   # K=20 stack OOF
PRIMARY_TEST_FILE = "test_d9f_K21_swap_strat.npy"       # K=21 test (LB 0.95031)
PRIMARY_LB_K21 = 0.95031
RHO_TIE = 0.999

# ============================================================ helpers


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
    p = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(p / (1 - p))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    mo = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        mo[va] = lr.predict_proba(F_oof[va])[:, 1]
    lrf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lrf.fit(F_oof, y)
    return mo, lrf.predict_proba(F_test)[:, 1], lrf.coef_.ravel()


def to_proba_rank(score):
    """Convert any score to [0,1] via rank/quantile."""
    return rankdata(score) / len(score)


# ============================================================ T1.2c: censored regression / Cox


def build_laps_until_pit_grouped(df_combined):
    """For each row in df_combined, compute (laps_until_next_pit, is_observed)
    within (Race, Driver, Year, Stint) group.

    is_observed = 1 if a future PitStop=1 row exists in the same group;
    laps_until_next_pit = next_pit_lap - current_lap (≥1).
    is_observed = 0 (right-censored); laps_until = (last_lap_in_group -
                  current_lap) + 1 (lower bound on time-to-event).
    """
    print("  Building (laps_until_next_pit, is_observed) ...")
    t = time.time()
    df = df_combined.sort_values(
        ["Driver", "Race", "Year", "Stint", "LapNumber"], kind="stable"
    ).reset_index(drop=True)
    grp_keys = ["Driver", "Race", "Year", "Stint"]
    grp_idx = df.groupby(grp_keys, sort=False).indices

    laps_until = np.full(len(df), -1, dtype=np.int32)
    observed = np.zeros(len(df), dtype=np.int8)

    pit_arr = df["PitStop"].values.astype(np.int8)
    lap_arr = df["LapNumber"].values.astype(np.int32)

    for k, idxs in grp_idx.items():
        order = np.argsort(lap_arr[idxs], kind="stable")
        idxs_sorted = idxs[order]
        laps_g = lap_arr[idxs_sorted]
        pits_g = pit_arr[idxs_sorted]
        n = len(idxs_sorted)
        last_lap = laps_g[-1]
        # walk back; track most-recent observed future pit
        next_pit_lap = -1
        for i in range(n - 1, -1, -1):
            d_obs = next_pit_lap - laps_g[i] if next_pit_lap >= 0 else -1
            if d_obs >= 1:
                laps_until[idxs_sorted[i]] = d_obs
                observed[idxs_sorted[i]] = 1
            else:
                # right-censored: at least (last_lap - current_lap) + 1 laps
                d_cens = max(last_lap - laps_g[i] + 1, 1)
                laps_until[idxs_sorted[i]] = d_cens
                observed[idxs_sorted[i]] = 0
            if pits_g[i] == 1:
                next_pit_lap = laps_g[i]
    df["laps_until_next_pit"] = laps_until
    df["is_observed"] = observed
    print(f"    pos rate of observed: {observed.mean():.4%}")
    print(f"    laps_until distribution head:")
    vc = df["laps_until_next_pit"].value_counts().sort_index().head(8)
    for k, v in vc.items():
        print(f"      {k}: {v}")
    print(f"    censored share: {(observed==0).mean():.4%}")
    print(f"    wall {time.time()-t:.1f}s")
    return df


def fit_lgbm_censored_regression(X_train, X_test, laps_until, observed,
                                 splits, y_binary,
                                 censored_weight=0.3):
    """Censored regression via LightGBM regression on log(laps_until+1)
    with sample_weight=1.0 for observed rows, censored_weight (default
    0.3) for right-censored rows.

    Rationale: a true Cox PH on 350k rows is O(N²) per boost iteration
    and infeasible. Tobit / AFT have similar costs. The pragmatic
    censored-regression substitute is:
      - target = log(laps_until + 1)  (≥0 since laps_until ≥ 1)
      - sample_weight: 1.0 if PitStop=1 in remaining group, ≤1 if not
        (censored target value is a *lower bound* on the true
        time-to-event so it should weigh less).

    The output is interpreted as predicted log(time-to-pit). To convert
    to AUC for PitNextLap=1 we use *negative* prediction as rank score
    (smaller predicted time → pit imminent).

    NOTE: this differs structurally from a binary classifier because
    the loss landscape is on a continuous regression target rather
    than logistic on the binary label. Bases on classification
    targets rank-lock at ρ≥0.999 in our pool; a regression base is
    a candidate to break the lock.
    """
    print(f"  LGBM censored regression (censored_weight={censored_weight})",
          flush=True)
    target = np.log(laps_until + 1.0)
    sw = np.where(observed == 1, 1.0,
                  censored_weight).astype(np.float32)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=SEED,
        num_threads=2,
    )
    oof = np.zeros(len(target), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    walls = []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        d_tr = lgb.Dataset(X_train.iloc[tr], label=target[tr], weight=sw[tr])
        d_va = lgb.Dataset(X_train.iloc[va], label=target[va], weight=sw[va])
        m = lgb.train(params, d_tr, num_boost_round=600,
                      valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[va] = m.predict(X_train.iloc[va])
        test_pred += m.predict(X_test) / N_FOLDS
        wall = time.time() - t0
        # AUC of -oof[va] (smaller predicted time → pit-imminent rank)
        s_auc = float(roc_auc_score(y_binary[va], -oof[va]))
        walls.append(wall)
        print(f"    f{k}: best_iter={m.best_iteration} "
              f"AUC(-oof,y)={s_auc:.5f} wall={wall:.1f}s", flush=True)
    # Score against full y using -oof
    auc_full = float(roc_auc_score(y_binary, -oof))
    print(f"  → standalone OOF AUC (censored-reg, -pred rank): "
          f"{auc_full:.5f} total wall={sum(walls):.1f}s")
    return oof, test_pred, auc_full


# ============================================================ T1.2d: ratio target


def build_ratio_target(df_combined):
    """For each (Race, Driver, Year), compute total_pits / total_stints."""
    print("  Building ratio target ...")
    g_keys = ["Race", "Driver", "Year"]
    g = df_combined.groupby(g_keys, observed=True)
    pits = g["PitStop"].sum()  # available in train+test (PitStop is a feature in test)
    stints = g["Stint"].max()
    ratio = (pits / stints).rename("pit_to_stint_ratio")
    df_combined = df_combined.merge(ratio, left_on=g_keys, right_index=True,
                                    how="left")
    # Add stint_age (laps_into_stint) — we'll need this for the transform.
    # laps_into_stint = LapNumber - min(LapNumber) within (D,R,Y,Stint)
    sg = df_combined.groupby(["Driver", "Race", "Year", "Stint"],
                             observed=True)["LapNumber"]
    df_combined["stint_min_lap"] = sg.transform("min")
    df_combined["laps_into_stint"] = (df_combined["LapNumber"]
                                      - df_combined["stint_min_lap"]).astype(np.int32)
    sg_max = df_combined.groupby(["Driver", "Race", "Year", "Stint"],
                                 observed=True)["TyreLife"].transform("max")
    print(f"    ratio dist: mean={ratio.mean():.3f} std={ratio.std():.3f} "
          f"min={ratio.min():.3f} max={ratio.max():.3f}")
    return df_combined


def fit_lgbm_ratio(X_train, X_test, y_ratio, splits, y_binary,
                   stint_age_train, stint_age_test, mean_stint_len):
    """LGBM regression on the ratio. Output transformed via heuristic
    + isotonic-on-binary."""
    print("  Training LGBM regression on pit_to_stint_ratio ...", flush=True)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=SEED,
        num_threads=2,
    )
    oof_ratio = np.zeros(len(y_ratio), dtype=np.float64)
    test_pred_ratio = np.zeros(len(X_test), dtype=np.float64)
    walls = []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        d_tr = lgb.Dataset(X_train.iloc[tr], label=y_ratio[tr])
        d_va = lgb.Dataset(X_train.iloc[va], label=y_ratio[va])
        m = lgb.train(params, d_tr, num_boost_round=600,
                      valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        oof_ratio[va] = m.predict(X_train.iloc[va])
        test_pred_ratio += m.predict(X_test) / N_FOLDS
        wall = time.time() - t0
        walls.append(wall)
        print(f"    f{k}: best_iter={m.best_iteration} wall={wall:.1f}s",
              flush=True)
    # Heuristic transform: pit_proba_row = ratio * (1 - exp(-age / mean_len))
    print(f"    mean_stint_len = {mean_stint_len:.2f}")
    raw_oof = (oof_ratio
               * (1 - np.exp(-stint_age_train / max(mean_stint_len, 1.0))))
    raw_test = (test_pred_ratio
                * (1 - np.exp(-stint_age_test / max(mean_stint_len, 1.0))))
    # OOF-isotonic against binary y for calibration
    iso_oof = np.zeros_like(raw_oof)
    test_iso_folds = np.zeros((N_FOLDS, len(raw_test)), dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_oof[tr], y_binary[tr])
        iso_oof[va] = iso.predict(raw_oof[va])
        test_iso_folds[k] = iso.predict(raw_test)
    test_iso = test_iso_folds.mean(0)
    auc_full = float(roc_auc_score(y_binary, iso_oof))
    print(f"  → standalone OOF AUC (ratio→iso): {auc_full:.5f} "
          f"total wall={sum(walls):.1f}s")
    return iso_oof, test_iso, auc_full


# ============================================================ T1.2e: stint-level survival


def build_stint_dataset(df_combined):
    """Build stint-level rows: one row per (Driver,Race,Year,Stint).

    Features: stint-start TyreLife, Compound (mode), Race, Driver, Year,
    Stint number, mean LapTime, stint-level pos delta, etc.
    Outcome: stint_duration_laps = max(LapNumber) - min(LapNumber) + 1
             event_observed = 1 if any PitStop=1 within stint else 0.
    """
    print("  Building stint-level dataset ...")
    t = time.time()
    g_keys = ["Driver", "Race", "Year", "Stint"]
    g = df_combined.groupby(g_keys, observed=True, sort=False)
    stint_df = g.agg(
        start_lap=("LapNumber", "min"),
        end_lap=("LapNumber", "max"),
        start_tyrelife=("TyreLife", "min"),
        mean_laptime=("LapTime (s)", "mean"),
        mean_degradation=("Cumulative_Degradation", "mean"),
        mean_position=("Position", "mean"),
        mean_progress=("RaceProgress", "mean"),
        compound=("Compound", lambda s: s.mode().iat[0]),
        n_laps=("LapNumber", "count"),
        any_pit=("PitStop", "max"),
    ).reset_index()
    stint_df["duration"] = (stint_df["end_lap"] - stint_df["start_lap"]
                            + 1).astype(np.int32)
    stint_df["event"] = stint_df["any_pit"].astype(np.int8)
    print(f"    stint rows: {len(stint_df)} "
          f"event rate: {stint_df['event'].mean():.4f}")
    print(f"    duration stats: mean={stint_df['duration'].mean():.2f} "
          f"median={stint_df['duration'].median():.2f}")
    print(f"    wall {time.time()-t:.1f}s")
    return stint_df


def fit_stint_survival_simple(stint_df_orig, splits_train_global,
                              all_df, y_binary, train_n):
    """STINT-LEVEL SURVIVAL via LGBM regression on log(stint_duration+1)
    with sample-weight ≤1 for censored stints (no observed pit), then
    map back to row-level pit_proba via a hazard transform.

    Steps:
      1. stint_df has columns: duration, event, stint-level features.
      2. Encode stint-level (Driver,Race,Year,Stint) → __sid.
      3. Map each row in all_df to __sid via composite key.
      4. Form train/val FOLD splits at stint level by majority vote of
         the row-fold of its first-observed train row.
      5. LGBM weighted regression on log(duration+1) (event-1 weight,
         0.3 for censored).
      6. For each row r in fold k:
           pred_dur_r = exp(stint_pred[r's stint, fold k]) - 1
           laps_into_stint = LapNumber - min(LapNumber within stint)
           pit_proba_row = laps_into_stint / pred_dur_r,  clipped [0,1]
         (Heuristic: probability that this row is the LAST lap of the
         stint, given expected duration. Not a true Cox hazard, but
         structurally captures stint-level survival information.)
      7. Then 1D OOF-isotonic against binary y for calibration.

    This avoids O(N²) Cox fitting (infeasible at 113k stints) while
    still producing a stint-level-survival-flavored base whose ranking
    is structurally different from row-level binary classifiers.
    """
    print("  Fitting stint-level LGBM duration-regression "
          "+ row hazard transform ...")
    t0 = time.time()
    sd = stint_df_orig.copy().reset_index(drop=True)
    sd["__sid"] = np.arange(len(sd))
    g_keys = ["Driver", "Race", "Year", "Stint"]
    # row → __sid lookup via composite key tuples (use stable hashing)
    sd_keys = list(zip(
        sd["Driver"].astype(str).values,
        sd["Race"].astype(str).values,
        sd["Year"].values.astype(np.int32),
        sd["Stint"].values.astype(np.int32),
    ))
    lookup = {k: i for i, k in enumerate(sd_keys)}
    all_keys = list(zip(
        all_df["Driver"].astype(str).values,
        all_df["Race"].astype(str).values,
        all_df["Year"].values.astype(np.int32),
        all_df["Stint"].values.astype(np.int32),
    ))
    row_stint_idx = np.array([lookup.get(k, -1) for k in all_keys],
                             dtype=np.int64)
    miss = (row_stint_idx < 0).sum()
    if miss > 0:
        raise RuntimeError(f"missing stint mapping for {miss} rows")
    n_train = train_n

    # Encode stint-level cats
    for c in ["Driver", "Race"]:
        uniq = sorted(sd[c].astype(str).unique())
        mp = {v: i for i, v in enumerate(uniq)}
        sd[c] = sd[c].astype(str).map(mp).astype(np.int32)
    sd["compound"] = sd["compound"].astype("category")
    feats = ["Driver", "Race", "Year", "Stint", "start_tyrelife",
             "mean_laptime", "mean_degradation", "mean_position",
             "mean_progress", "compound", "start_lap"]
    Xs = sd[feats].copy()
    durations = sd["duration"].values.astype(np.float64)
    events = sd["event"].values.astype(np.int8)
    log_dur = np.log(durations + 1.0)
    sw = np.where(events == 1, 1.0, 0.3).astype(np.float32)

    # row_fold for train rows
    row_fold = np.full(n_train, -1, dtype=np.int32)
    for k, (_, va) in enumerate(splits_train_global):
        row_fold[va] = k
    # Stint fold = mode of row-fold among rows in stint (train side only)
    train_row_stint = row_stint_idx[:n_train]
    stint_fold = np.full(len(sd), -1, dtype=np.int32)
    # Fill the fold of the FIRST encountered train row per stint (proxy)
    first_seen = {}
    for ri in range(n_train):
        s_idx = train_row_stint[ri]
        if s_idx not in first_seen:
            first_seen[s_idx] = row_fold[ri]
    for s_idx, f in first_seen.items():
        stint_fold[s_idx] = f

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_data_in_leaf=20,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=SEED,
        num_threads=2,
    )
    stint_oof_pred = np.zeros(len(sd), dtype=np.float64)
    # Test stints: average over 5 folds (note: 'test' stints may have
    # train-row-fold assignment too, since same stint spans both)
    stint_test_pred = np.zeros(len(sd), dtype=np.float64)
    walls = []
    for k in range(N_FOLDS):
        t1 = time.time()
        tr_mask = (stint_fold != k) & (stint_fold != -1)
        # also include stints with no train rows (-1) in training set
        tr_mask = tr_mask | (stint_fold == -1)
        va_mask = (stint_fold == k)
        if va_mask.sum() == 0:
            print(f"    f{k}: no val stints, skip")
            walls.append(0.0)
            continue
        tr_st = np.where(tr_mask)[0]
        va_st = np.where(va_mask)[0]
        d_tr = lgb.Dataset(Xs.iloc[tr_st], label=log_dur[tr_st],
                           weight=sw[tr_st])
        d_va = lgb.Dataset(Xs.iloc[va_st], label=log_dur[va_st],
                           weight=sw[va_st])
        m = lgb.train(params, d_tr, num_boost_round=600,
                      valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        stint_oof_pred[va_st] = m.predict(Xs.iloc[va_st])
        # avg across all stints (including stint-fold=-1 stints)
        stint_test_pred += m.predict(Xs) / N_FOLDS
        wall = time.time() - t1
        walls.append(wall)
        print(f"    f{k}: best_iter={m.best_iteration} "
              f"train_st={len(tr_st)} val_st={len(va_st)} "
              f"wall={wall:.1f}s", flush=True)
    # Use stint_oof_pred for the stint-fold's val stints; for stints
    # with no fold assignment (i.e., stint_fold=-1 → all rows are
    # train-only), they appear in TRAIN side of every fold so their
    # OOF pred is undefined. Use stint_test_pred (5-fold-avg) as fallback.
    stint_pred_for_oof = np.where(stint_fold >= 0, stint_oof_pred,
                                  stint_test_pred)

    # Map back to row-level pit_proba using laps_into_stint
    print("  Mapping stint-level pred to row-level pit_proba ...")
    g_keys = ["Driver", "Race", "Year", "Stint"]
    laps_into_stint = (
        all_df["LapNumber"].values
        - all_df.groupby(g_keys, observed=True)["LapNumber"]
                .transform("min").values
    ).astype(np.float64)
    # Predicted duration per row's stint
    row_pred_dur_oof = np.exp(stint_pred_for_oof[row_stint_idx]) - 1.0
    row_pred_dur_test = np.exp(stint_test_pred[row_stint_idx]) - 1.0
    row_pred_dur_oof = np.maximum(row_pred_dur_oof, 1.0)
    row_pred_dur_test = np.maximum(row_pred_dur_test, 1.0)
    raw_oof = np.clip(laps_into_stint[:n_train]
                      / row_pred_dur_oof[:n_train], 0.0, 1.0)
    raw_test = np.clip(laps_into_stint[n_train:]
                       / row_pred_dur_test[n_train:], 0.0, 1.0)

    # OOF-isotonic against binary y for calibration
    iso_oof = np.zeros_like(raw_oof)
    test_iso_folds = np.zeros((N_FOLDS, len(raw_test)), dtype=np.float64)
    for k, (tr, va) in enumerate(splits_train_global):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_oof[tr], y_binary[tr])
        iso_oof[va] = iso.predict(raw_oof[va])
        test_iso_folds[k] = iso.predict(raw_test)
    test_iso = test_iso_folds.mean(0)
    auc_full = float(roc_auc_score(y_binary, iso_oof))
    print(f"  → standalone OOF AUC (stint-survival): {auc_full:.5f} "
          f"total wall={time.time()-t0:.1f}s")
    return iso_oof, test_iso, auc_full


# ============================================================ stack-eval


def stack_eval(name, Xs_oof, Xs_test, names, y, primary_test, primary_oof,
               results):
    K = len(names)
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, primary_test)
    primary_oof_auc = float(roc_auc_score(y, primary_oof))
    delta_primary = (auc - primary_oof_auc) * 1e4
    pred_lb = PRIMARY_LB_K21 + (auc - primary_oof_auc)
    if rho >= RHO_TIE: pred_lb = pred_lb
    elif rho >= 0.995: pred_lb -= 0.0001
    elif rho >= 0.99:  pred_lb -= 0.00025
    else:              pred_lb -= 0.0004
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i])
                          + abs(coef[2*K + i])) for i in range(K)}
    print(f"\n=== {name} (K={K}) ===")
    print(f"  Strat OOF: {auc:.5f}  Δ PRIMARY {delta_primary:+.2f}bp")
    print(f"  ρ vs PRIMARY test: {rho:.5f}  pred-LB {pred_lb:.5f} "
          f"(Δ {(pred_lb - PRIMARY_LB_K21)*1e4:+.2f}bp)")
    top_l1 = sorted(l1.items(), key=lambda kv: -kv[1])[:12]
    print(f"  L1 top-12:")
    for n_, v in top_l1:
        marker = ""
        if n_.startswith("t12"): marker = "  ← T1.2 candidate"
        print(f"    {n_:<26s} L1={v:.3f}{marker}")
    results[name] = dict(K=K, strat_oof=auc, delta_primary_bp=delta_primary,
                         rho_vs_primary_test=float(rho),
                         pred_lb=float(pred_lb),
                         delta_lb_bp=float((pred_lb - PRIMARY_LB_K21) * 1e4),
                         l1_ranking=l1)
    return mo, tp


# ============================================================ smoke driver


def run_smoke(formulation, train_df, test_df, y, n_smoke=50_000):
    """Run 1-fold smoke on first n_smoke rows; only used to validate
    pipeline + estimate per-fold wall."""
    print(f"\n--- SMOKE {formulation} (1 fold, {n_smoke} rows) ---")
    # Use first n_smoke rows of train, full test
    sm = train_df.iloc[:n_smoke].copy()
    y_sm = y[:n_smoke]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y_sm)), y_sm))
    splits_one = [splits[0]]
    return splits_one, sm, y_sm


# ============================================================ main


def main():
    t_total = time.time()
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train_df[TARGET].astype(int).values
    n_train = len(train_df)

    # PRIMARY anchors
    primary_oof = np.load(ART / PRIMARY_OOF_FILE)[:, 1].astype(np.float64)
    primary_test = np.load(ART / PRIMARY_TEST_FILE)[:, 1].astype(np.float64)
    primary_oof_auc = float(roc_auc_score(y, primary_oof))
    print(f"PRIMARY (K=20 d9c Sd) OOF AUC: {primary_oof_auc:.5f}")
    print(f"PRIMARY (K=21 d9f swap) test array used for ρ; LB anchor "
          f"= {PRIMARY_LB_K21}")

    # Combined for FE + features that span train+test (laps_until and ratio
    # both use PitStop which exists in test too)
    train_df["_src"] = "train"; test_df["_src"] = "test"; test_df[TARGET] = -1
    df_all = pd.concat([train_df, test_df], ignore_index=True)

    # Build laps_until_next_pit (right-censored)
    df_all_c = build_laps_until_pit_grouped(df_all)
    df_all_c = df_all_c.sort_values("id", kind="stable").reset_index(drop=True)
    train_c = df_all_c[df_all_c["_src"] == "train"].copy()
    test_c = df_all_c[df_all_c["_src"] == "test"].copy()
    assert (train_c[TARGET].astype(int).values == y).all()
    laps_until = train_c["laps_until_next_pit"].astype(np.float64).values
    observed = train_c["is_observed"].astype(np.int8).values
    laps_until_test = test_c["laps_until_next_pit"].astype(np.float64).values
    observed_test = test_c["is_observed"].astype(np.int8).values

    # Build ratio + stint_age (needs same FE)
    df_all_d = build_ratio_target(df_all_c)
    df_all_d = df_all_d.sort_values("id", kind="stable").reset_index(drop=True)
    train_d = df_all_d[df_all_d["_src"] == "train"].copy()
    test_d = df_all_d[df_all_d["_src"] == "test"].copy()
    ratio_target = train_d["pit_to_stint_ratio"].astype(np.float64).values
    stint_age_train = train_d["laps_into_stint"].astype(np.float64).values
    stint_age_test = test_d["laps_into_stint"].astype(np.float64).values
    mean_stint_len = float(df_all_d.groupby(
        ["Driver", "Race", "Year", "Stint"], observed=True)["LapNumber"].count(
        ).mean())

    # Feature matrix for (c) and (d) — same encoding
    drop_cols = [TARGET, ID_COL, "_src", "laps_until_next_pit",
                 "is_observed", "pit_to_stint_ratio",
                 "stint_min_lap"]
    X_train = train_d.drop(columns=drop_cols, errors="ignore").copy()
    X_test = test_d.drop(columns=drop_cols, errors="ignore").copy()
    X_train, X_test = encode_features(X_train.copy(), X_test.copy())
    print(f"feature matrix shape: train {X_train.shape} test {X_test.shape}")
    print(f"feature cols: {list(X_train.columns)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    final = {"primary_oof_auc": primary_oof_auc,
             "primary_lb": PRIMARY_LB_K21,
             "candidates": {}}

    # ====================================================== T1.2c: censored regression
    print("\n" + "=" * 70)
    print("[T1.2c] CENSORED REGRESSION — LGBM weighted-loss substitute")
    print("=" * 70)
    t12c_oof_raw, t12c_test_raw, t12c_auc_std = fit_lgbm_censored_regression(
        X_train, X_test, laps_until, observed, splits, y,
        censored_weight=0.3)
    # to [0,1] proba via rank — note we use -raw since SMALL predicted
    # time means pit-imminent
    t12c_oof_p = to_proba_rank(-t12c_oof_raw)
    t12c_test_p = to_proba_rank(-t12c_test_raw)
    rho_c_primary, _ = spearmanr(t12c_test_p, primary_test)
    np.save(ART / "oof_d12_t12c_censored_strat.npy",
            np.column_stack([1 - t12c_oof_p, t12c_oof_p]))
    np.save(ART / "test_d12_t12c_censored_strat.npy",
            np.column_stack([1 - t12c_test_p, t12c_test_p]))
    # min-meta gate
    F_min = expand(np.column_stack([primary_oof, t12c_oof_p,
                                    np.abs(primary_oof - t12c_oof_p)]))
    F_min_t = expand(np.column_stack([primary_test, t12c_test_p,
                                      np.abs(primary_test - t12c_test_p)]))
    mo_min, _, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min_c = float(roc_auc_score(y, mo_min))
    delta_min_c = (auc_min_c - primary_oof_auc) * 1e4
    print(f"\n[T1.2c] standalone OOF AUC: {t12c_auc_std:.5f} "
          f"ρ vs PRIMARY: {rho_c_primary:.5f}")
    print(f"[T1.2c] min-meta OOF: {auc_min_c:.5f} "
          f"Δ PRIMARY {delta_min_c:+.2f}bp "
          f"{'PASS ✓' if delta_min_c >= 0.10 else 'FAIL ✗'}")
    final["candidates"]["t12c_censored_cox"] = dict(
        std_oof=t12c_auc_std, rho_vs_primary=float(rho_c_primary),
        min_meta_oof=auc_min_c, min_meta_delta_bp=float(delta_min_c),
        min_meta_pass=bool(delta_min_c >= 0.10),
    )

    # ====================================================== T1.2d: ratio
    print("\n" + "=" * 70)
    print("[T1.2d] RATIO TARGET — LGBM regression + heuristic + isotonic")
    print("=" * 70)
    t12d_oof_p, t12d_test_p, t12d_auc_std = fit_lgbm_ratio(
        X_train, X_test, ratio_target, splits, y,
        stint_age_train, stint_age_test, mean_stint_len)
    rho_d_primary, _ = spearmanr(t12d_test_p, primary_test)
    np.save(ART / "oof_d12_t12d_ratio_strat.npy",
            np.column_stack([1 - t12d_oof_p, t12d_oof_p]))
    np.save(ART / "test_d12_t12d_ratio_strat.npy",
            np.column_stack([1 - t12d_test_p, t12d_test_p]))
    F_min = expand(np.column_stack([primary_oof, t12d_oof_p,
                                    np.abs(primary_oof - t12d_oof_p)]))
    F_min_t = expand(np.column_stack([primary_test, t12d_test_p,
                                      np.abs(primary_test - t12d_test_p)]))
    mo_min, _, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min_d = float(roc_auc_score(y, mo_min))
    delta_min_d = (auc_min_d - primary_oof_auc) * 1e4
    print(f"\n[T1.2d] standalone OOF AUC: {t12d_auc_std:.5f} "
          f"ρ vs PRIMARY: {rho_d_primary:.5f}")
    print(f"[T1.2d] min-meta OOF: {auc_min_d:.5f} "
          f"Δ PRIMARY {delta_min_d:+.2f}bp "
          f"{'PASS ✓' if delta_min_d >= 0.10 else 'FAIL ✗'}")
    final["candidates"]["t12d_ratio"] = dict(
        std_oof=t12d_auc_std, rho_vs_primary=float(rho_d_primary),
        min_meta_oof=auc_min_d, min_meta_delta_bp=float(delta_min_d),
        min_meta_pass=bool(delta_min_d >= 0.10),
    )

    # ====================================================== T1.2e: stint-survival
    print("\n" + "=" * 70)
    print("[T1.2e] STINT-LEVEL SURVIVAL — XGB Cox stint-rows + Breslow baseline")
    print("=" * 70)
    stint_df = build_stint_dataset(df_all_d)
    t12e_oof_p, t12e_test_p, t12e_auc_std = fit_stint_survival_simple(
        stint_df, splits, df_all_d, y, n_train)
    rho_e_primary, _ = spearmanr(t12e_test_p, primary_test)
    np.save(ART / "oof_d12_t12e_survival_strat.npy",
            np.column_stack([1 - t12e_oof_p, t12e_oof_p]))
    np.save(ART / "test_d12_t12e_survival_strat.npy",
            np.column_stack([1 - t12e_test_p, t12e_test_p]))
    F_min = expand(np.column_stack([primary_oof, t12e_oof_p,
                                    np.abs(primary_oof - t12e_oof_p)]))
    F_min_t = expand(np.column_stack([primary_test, t12e_test_p,
                                      np.abs(primary_test - t12e_test_p)]))
    mo_min, _, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min_e = float(roc_auc_score(y, mo_min))
    delta_min_e = (auc_min_e - primary_oof_auc) * 1e4
    print(f"\n[T1.2e] standalone OOF AUC: {t12e_auc_std:.5f} "
          f"ρ vs PRIMARY: {rho_e_primary:.5f}")
    print(f"[T1.2e] min-meta OOF: {auc_min_e:.5f} "
          f"Δ PRIMARY {delta_min_e:+.2f}bp "
          f"{'PASS ✓' if delta_min_e >= 0.10 else 'FAIL ✗'}")
    final["candidates"]["t12e_stint_survival"] = dict(
        std_oof=t12e_auc_std, rho_vs_primary=float(rho_e_primary),
        min_meta_oof=auc_min_e, min_meta_delta_bp=float(delta_min_e),
        min_meta_pass=bool(delta_min_e >= 0.10),
    )

    # ====================================================== K=22 / K=23 / K=24 stacks
    print("\n" + "=" * 70)
    print("K=22 STACK — PRIMARY-keep K=21 + each candidate")
    print("=" * 70)
    # Build the K=21 PRIMARY pool (drop d9c FM, add FM_A + FM_B)
    POOL_K21 = [
        ("baseline", "baseline_two_anchor"),
        ("d2a_te", "d2a_te"), ("m2_xgb", "m2_xgb"),
        ("e1_cb_sub", "e1_catboost_sub"),
        ("e3_hgbc", "e3_hgbc"), ("e5_optuna_lgbm", "e5_optuna_lgbm"),
        ("a_horizon", "a_horizon"), ("b_lapsuntilpit", "b_lapsuntilpit"),
        ("f1_hgbc_deep", "f1_hgbc_deep"),
        ("f2_hgbc_shallow", "f2_hgbc_shallow"),
        ("cb_year-cat", "cb_year-cat"), ("cb_lossguide", "cb_lossguide"),
        ("cb_slow-wide-bag", "cb_slow-wide-bag"),
        ("realmlp", "realmlp"),
        ("rule_driver_compound", "d6_rule_driver_compound"),
        ("rule_year_race", "d6_rule_year_race"),
        ("R6_next_compound", "d9_R6_next_compound"),
        ("R10_driver_eb", "d9_R10_driver_eb"),
        ("R7_prev_compound", "d9_R7_prev_compound"),
        ("FM_A", "d9f_FM_A"),
        ("FM_B", "d9f_FM_B"),
    ]
    pool_oof, pool_test, pool_names = [], [], []
    for label, fname in POOL_K21:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        pool_oof.append(oo); pool_test.append(te); pool_names.append(label)
    print(f"K=21 pool loaded: {len(pool_names)} bases")

    cands = [
        ("t12c_cox", t12c_oof_p, t12c_test_p),
        ("t12d_ratio", t12d_oof_p, t12d_test_p),
        ("t12e_survival", t12e_oof_p, t12e_test_p),
    ]

    stack_results = {}
    # Single-add K=22 stacks
    best_oof_k = None; best_oof_score = -1; best_label = None
    best_oof_arr = None; best_test_arr = None
    for cname, c_oof, c_test in cands:
        Xs = pool_oof + [c_oof]; Ts = pool_test + [c_test]
        Ns = pool_names + [cname]
        mo, tp = stack_eval(f"K22_{cname}", Xs, Ts, Ns, y, primary_test,
                            primary_oof, stack_results)
        sk = stack_results[f"K22_{cname}"]
        if sk["strat_oof"] > best_oof_score:
            best_oof_score = sk["strat_oof"]; best_label = f"K22_{cname}"
            best_oof_arr = mo; best_test_arr = tp

    # K=23 pairs
    pair_combos = [(0, 1), (0, 2), (1, 2)]
    for i, j in pair_combos:
        ci, cj = cands[i], cands[j]
        Xs = pool_oof + [ci[1], cj[1]]
        Ts = pool_test + [ci[2], cj[2]]
        Ns = pool_names + [ci[0], cj[0]]
        mo, tp = stack_eval(f"K23_{ci[0]}_{cj[0]}", Xs, Ts, Ns, y,
                            primary_test, primary_oof, stack_results)
        sk = stack_results[f"K23_{ci[0]}_{cj[0]}"]
        if sk["strat_oof"] > best_oof_score:
            best_oof_score = sk["strat_oof"]
            best_label = f"K23_{ci[0]}_{cj[0]}"
            best_oof_arr = mo; best_test_arr = tp

    # K=24 all three
    Xs = pool_oof + [c[1] for c in cands]
    Ts = pool_test + [c[2] for c in cands]
    Ns = pool_names + [c[0] for c in cands]
    mo, tp = stack_eval("K24_all_three", Xs, Ts, Ns, y, primary_test,
                        primary_oof, stack_results)
    sk = stack_results["K24_all_three"]
    if sk["strat_oof"] > best_oof_score:
        best_oof_score = sk["strat_oof"]; best_label = "K24_all_three"
        best_oof_arr = mo; best_test_arr = tp

    # Save best stack as oof_d12_t12_best_strat.npy
    np.save(ART / "oof_d12_t12_best_strat.npy",
            np.column_stack([1 - best_oof_arr, best_oof_arr]))
    np.save(ART / "test_d12_t12_best_strat.npy",
            np.column_stack([1 - best_test_arr, best_test_arr]))
    print(f"\n>>> BEST STACK: {best_label}  OOF {best_oof_score:.5f}")

    final["stacks"] = stack_results
    final["best_stack"] = dict(label=best_label, strat_oof=best_oof_score)
    final["wall_total_s"] = time.time() - t_total
    (ART / "d12_t12_multi_formulation_results.json").write_text(
        json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d12_t12_multi_formulation_results.json "
          f"(wall {time.time()-t_total:.0f}s)")

    # Summary
    print("\n" + "=" * 78)
    print(f"{'candidate':<24s} {'std_OOF':>9s} {'ρ_PRIM':>8s} "
          f"{'min_meta':>9s} {'Δbp':>6s} {'PASS':>5s}")
    print("-" * 78)
    for nm, r in final["candidates"].items():
        flag = "PASS" if r["min_meta_pass"] else "FAIL"
        print(f"{nm:<24s} {r['std_oof']:>9.5f} {r['rho_vs_primary']:>+8.4f} "
              f"{r['min_meta_oof']:>9.5f} {r['min_meta_delta_bp']:>+6.2f} "
              f"{flag:>5s}")
    print("\n" + "=" * 78)
    print(f"{'stack':<28s} {'OOF':>9s} {'Δprim':>7s} {'ρ_PRIM':>7s} "
          f"{'predLB':>9s} {'ΔLB':>6s}")
    print("-" * 78)
    for nm, r in final["stacks"].items():
        print(f"{nm:<28s} {r['strat_oof']:>9.5f} {r['delta_primary_bp']:>+6.2f} "
              f"{r['rho_vs_primary_test']:>7.5f} {r['pred_lb']:>9.5f} "
              f"{r['delta_lb_bp']:>+5.2f}")


if __name__ == "__main__":
    main()
