"""Sprint A — stint_imputed-anchored LightGBM base (M7+M8+M15+M17).

Hypothesis: the synth `Stint` label is a fabricated 8-valued categorical.
The recovered identity `stint_imputed = LapNumber - TyreLife + 1` has 106
distinct values per row and 2.1x richer (Race, Year, Driver) partition.
Every prior probe that used `Stint` as a within-stint context variable
(Day-15 GRU sequence-fingerprint, A3-1 RankSortedGaps, EXP-NEW Phase 1
sequence picks) was operating on a 13x-degenerate identity. With the
recovered identity:

  M7  stint_imputed                          (cardinality 106 vs 8)
  M15 CumulativeTimeStint                    (Frontiers F1 top feature)
  M17 prev_lap_delta vs prev lap WITHIN STINT (verified our LapTime_Delta is NOT this; corr=0.16)
  M8  stint_lap_idx, stint_lap_frac, stint_size, prev_compound,
      position_change_in_stint, compound_changes, position_at_stint_start

This base trains a LightGBM on 14 raw cols + 9 new stint-anchored
features = 23 features. Cheap (~5-10 min CPU). Reports:
  - standalone 5-fold StratifiedKF OOF AUC
  - rho_test vs K=4 PRIMARY
  - K=4+1 LR-meta gate (single-add lift in bp)
  - per-fold AUCs to track variance

Output:
  scripts/artifacts/dgp_v3_qAA_stint_imputed_oof.npy
  scripts/artifacts/dgp_v3_qAA_stint_imputed_test.npy
  scripts/artifacts/dgp_v3_qAA_stint_imputed.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"

SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def build_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 9 stint_imputed-anchored features to df. Idempotent.

    Critical: this is a deterministic transform of (LapNumber, TyreLife,
    Compound, Position, LapTime, Race, Year, Driver) — no labels touched.
    Therefore Rule 24 (fold-safe label-conditional aggregates) does NOT
    apply: features are independent of fold split.
    """
    df = df.copy()
    df["stint_imputed"] = (df["LapNumber"] - df["TyreLife"] + 1).astype(np.int32)

    # Sort by lap within driver-race so cumsum / lag respects time order
    sort_keys = ["Race", "Year", "Driver", "stint_imputed", "LapNumber"]
    df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    g_full = df.groupby(["Race", "Year", "Driver"], sort=False)
    g_stint = df.groupby(["Race", "Year", "Driver", "stint_imputed"], sort=False)

    # M15: CumulativeTimeStint
    df["CumulativeTimeStint"] = g_stint["LapTime (s)"].cumsum().astype(np.float32)

    # M17: prev_lap_delta within stint_imputed (true Frontiers feature)
    df["prev_lap_delta_stint"] = (
        df["LapTime (s)"] - g_stint["LapTime (s)"].shift(1)
    ).astype(np.float32)
    # also a prev-lap delta at the (Race,Year,Driver) level (regardless of stint)
    df["prev_lap_delta_drv"] = (
        df["LapTime (s)"] - g_full["LapTime (s)"].shift(1)
    ).astype(np.float32)

    # M8: sequence-on-stint-imputed structural features
    df["stint_lap_idx"] = g_stint.cumcount().astype(np.int32)
    stint_size = g_stint["LapNumber"].transform("size").astype(np.int32)
    df["stint_size"] = stint_size
    df["stint_lap_frac"] = (df["stint_lap_idx"] / stint_size.replace(0, 1)).astype(np.float32)

    # prev_compound within driver-race (regardless of stint, captures stint changes)
    df["prev_compound"] = g_full["Compound"].shift(1)

    # compound_changes = count of compound switches up to this lap (vectorized)
    is_change = (df["Compound"] != df["prev_compound"]).astype(np.int32)
    is_change.iloc[0] = 0  # NaN prev → 0
    df["compound_changes"] = g_full.cumcount().astype(np.int32)  # placeholder
    # Use cumsum of is_change within each (Race, Year, Driver) group
    df["compound_changes"] = (
        is_change.groupby([df["Race"], df["Year"], df["Driver"]]).cumsum() - is_change
    ).fillna(0).astype(np.int32)

    # position_change_in_stint: Position - first_Position_in_stint
    first_pos = g_stint["Position"].transform("first").astype(np.float32)
    df["position_at_stint_start"] = first_pos
    df["position_change_in_stint"] = (df["Position"].astype(np.float32) - first_pos).astype(np.float32)

    # Restore original row order
    df = df.sort_values("_orig_idx", kind="mergesort").drop(columns=["_orig_idx"]).reset_index(drop=True)
    return df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list[str]):
    """Label-encode categoricals using union vocab from train+test."""
    out_train = train_df.copy()
    out_test = test_df.copy()
    for c in cat_cols:
        all_vals = pd.concat([train_df[c], test_df[c]], ignore_index=True)
        cats = pd.Categorical(all_vals).categories
        out_train[c] = pd.Categorical(train_df[c], categories=cats).codes.astype(np.int32)
        out_test[c] = pd.Categorical(test_df[c], categories=cats).codes.astype(np.int32)
    return out_train, out_test


def main():
    ts = time.time()
    out: dict = {}

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    t(f"train {train.shape} test {test.shape}", ts)

    train_fe = build_stint_features(train)
    test_fe = build_stint_features(test)
    t(f"feats built; train new cols: {set(train_fe.columns) - set(train.columns)}", ts)

    # Confirm stint_imputed cardinality jump
    print(f"\n  Stint cardinality: {train.Stint.nunique()}; stint_imputed: {train_fe.stint_imputed.nunique()}")
    print(f"  prev_lap_delta_stint mean={train_fe.prev_lap_delta_stint.mean():.3f}, "
          f"std={train_fe.prev_lap_delta_stint.std():.3f}, "
          f"valid_frac={train_fe.prev_lap_delta_stint.notna().mean():.3f}")

    # Feature columns for LightGBM
    base_num = ["LapNumber", "TyreLife", "Position", "LapTime (s)",
                "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
                "Position_Change", "PitStop", "Stint", "Year"]
    base_cat = ["Driver", "Compound", "Race"]
    new_num = ["stint_imputed", "CumulativeTimeStint", "prev_lap_delta_stint",
               "prev_lap_delta_drv", "stint_lap_idx", "stint_size",
               "stint_lap_frac", "compound_changes", "position_at_stint_start",
               "position_change_in_stint"]
    new_cat = ["prev_compound"]

    feat_cols = base_num + base_cat + new_num + new_cat
    cat_cols = base_cat + new_cat
    out["feat_cols"] = feat_cols
    out["cat_cols"] = cat_cols
    out["n_features"] = len(feat_cols)

    train_enc, test_enc = encode_categoricals(train_fe, test_fe, cat_cols)

    X = train_enc[feat_cols].values
    y = train_enc[TARGET].values
    X_test = test_enc[feat_cols].values

    # Cast to float32 to save memory; keep cat indices as int via LightGBM
    cat_idx = [feat_cols.index(c) for c in cat_cols]

    # 5-fold StratifiedKFold (LB proxy per Rule R1, comp-context)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs = []

    lgb_params = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        t1 = time.time()
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X[tr], y[tr], categorical_feature=cat_idx,
              eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        val_pred = m.predict_proba(X[va])[:, 1]
        oof[va] = val_pred
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        a = float(roc_auc_score(y[va], val_pred))
        fold_aucs.append(a)
        print(f"  fold {fold+1} AUC = {a:.5f}  ({time.time()-t1:.0f}s)  best_iter={m.best_iteration_}")

    auc = float(roc_auc_score(y, oof))
    print(f"\n=== qAA standalone OOF AUC = {auc:.5f} ===")
    print(f"  fold std: {np.std(fold_aucs):.5f}")

    out["fold_aucs"] = fold_aucs
    out["oof_auc"] = auc

    # ---- rho vs K=4 PRIMARY ------------------------------------------------
    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")
    if primary_oof.ndim == 2:
        primary_oof = primary_oof[:, 1]
    if primary_test.ndim == 2:
        primary_test = primary_test[:, 1]
    rho_oof = float(spearmanr(oof, primary_oof).correlation)
    rho_test = float(spearmanr(test_pred, primary_test).correlation)
    primary_auc = float(roc_auc_score(y, primary_oof))
    print(f"\n  PRIMARY K=4 OOF AUC: {primary_auc:.5f}")
    print(f"  rho_oof vs PRIMARY: {rho_oof:.5f}")
    print(f"  rho_test vs PRIMARY: {rho_test:.5f}")
    out["primary_oof_auc"] = primary_auc
    out["rho_oof_vs_primary"] = rho_oof
    out["rho_test_vs_primary"] = rho_test

    # ---- K=4+1 LR-meta gate ------------------------------------------------
    # Load K=4 base OOFs + tests
    BASES = [
        ("d17_h1d_yekenot_full", "oof_d17_h1d_yekenot_full_strat.npy", "test_d17_h1d_yekenot_full_strat.npy"),
        ("p1_single_cb_v4_gpu", "oof_p1_single_cb_v4_gpu_strat.npy", "test_p1_single_cb_v4_gpu_strat.npy"),
        ("f1_hgbc_deep", "oof_f1_hgbc_deep_strat.npy", "test_f1_hgbc_deep_strat.npy"),
        ("d16_orig_continuous_only", "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ]
    base_oofs = []
    base_tests = []
    for name, oof_f, test_f in BASES:
        o = np.load(ART / oof_f)
        te = np.load(ART / test_f)
        if o.ndim == 2:
            o = o[:, 1]
        if te.ndim == 2:
            te = te[:, 1]
        base_oofs.append(o)
        base_tests.append(te)

    # Plain K=4 LR-meta with [P, rank, logit] expansion
    def expand_features(p_list):
        cols = []
        for p in p_list:
            p = np.clip(p, 1e-6, 1 - 1e-6)
            rank = pd.Series(p).rank().values / len(p)
            logit = np.log(p / (1 - p))
            cols += [p, rank, logit]
        return np.column_stack(cols)

    Xm_K4 = expand_features(base_oofs)
    Xm_K4_test = expand_features(base_tests)
    Xm_K5 = expand_features(base_oofs + [oof])
    Xm_K5_test = expand_features(base_tests + [test_pred])

    # 5-fold OOF on the meta layer
    def lr_meta_oof(Xm, y_):
        skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_m = np.zeros(len(y_))
        for tr, va in skf2.split(Xm, y_):
            mlr = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=SEED)
            mlr.fit(Xm[tr], y_[tr])
            oof_m[va] = mlr.predict_proba(Xm[va])[:, 1]
        return oof_m

    oof_K4 = lr_meta_oof(Xm_K4, y)
    oof_K5 = lr_meta_oof(Xm_K5, y)
    auc_K4 = float(roc_auc_score(y, oof_K4))
    auc_K5 = float(roc_auc_score(y, oof_K5))
    delta_bp = (auc_K5 - auc_K4) * 1e4
    print(f"\n  K=4 plain LR-meta OOF: {auc_K4:.5f}")
    print(f"  K=5 (K=4+qAA) plain LR-meta OOF: {auc_K5:.5f}")
    print(f"  K=4+1 lift: {delta_bp:+.3f} bp")
    print(f"  GATE: {'PASS (>=+0.5 bp)' if delta_bp >= 0.5 else ('WEAK (in (-0.3, +0.5))' if delta_bp > -0.3 else 'FAIL')}")
    out["k4_plain_lr_meta_oof"] = auc_K4
    out["k5_plain_lr_meta_oof"] = auc_K5
    out["k4plus1_lift_bp"] = delta_bp
    out["gate_verdict"] = "PASS" if delta_bp >= 0.5 else ("WEAK" if delta_bp > -0.3 else "FAIL")

    # Save
    np.save(ART / "dgp_v3_qAA_stint_imputed_oof.npy", oof)
    np.save(ART / "dgp_v3_qAA_stint_imputed_test.npy", test_pred)
    fp = ART / "dgp_v3_qAA_stint_imputed.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
