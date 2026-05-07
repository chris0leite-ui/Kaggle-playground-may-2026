"""scripts/d17_h2_fastf1_join.py — Day-17 H2 external-data probe.

Pull FastF1 lap-by-lap telemetry for the (Year, Race) combos in
train+test, key-merge on (Driver, Race, Year, LapNumber) for the
real-driver-TLA subset (3-letter codes), compute Frontiers-AI-2025
Bi-LSTM-paper features (DriverAheadPit, TrackStatus, CumulativeTimeStint),
fold-safe 5-fold LGBM, gate as K=22 stack-add candidate.

Pre-flight: 80/20 holdout AV check on the merged subset to detect
DGP-leak (aadigupta1601 source dataset may have been seeded by FastF1).

Outputs:
    scripts/artifacts/oof_d17_h2_fastf1_strat.npy
    scripts/artifacts/test_d17_h2_fastf1_strat.npy
    scripts/artifacts/d17_h2_fastf1_results.json
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import fastf1
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("/root/.cache/fastf1")
# Hush the FastF1 INFO logs (noisy during 100+ race pulls).
import logging
for nm in ("fastf1", "req", "_api", "core"):
    logging.getLogger(nm).setLevel(logging.WARNING)

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)
TARGET = "PitNextLap"
SEED = 42
N_FOLDS = 5

# Races that didn't happen in real life or are not real F1 sessions.
SKIP_COMBOS = {
    ("Pre-Season Testing", 2022), ("Pre-Season Testing", 2023),
    ("Pre-Season Testing", 2024), ("Pre-Season Testing", 2025),
    ("Chinese Grand Prix", 2022), ("Chinese Grand Prix", 2023),
    ("French Grand Prix", 2022), ("French Grand Prix", 2023),
    ("French Grand Prix", 2024), ("French Grand Prix", 2025),
    ("Russian Grand Prix", 2022), ("Russian Grand Prix", 2023),
    ("Russian Grand Prix", 2024), ("Russian Grand Prix", 2025),
}


def load_session_safe(year: int, race: str):
    try:
        s = fastf1.get_session(year, race, "R")
        s.load(laps=True, telemetry=False, weather=False, messages=False)
        # Validate that laps actually loaded — when livetiming API
        # returns 403, FastF1 logs warnings but the laps DataFrame is
        # empty / accessing it raises DataNotLoadedError.
        try:
            n = len(s.laps)
            if n == 0:
                return None
        except Exception:
            return None
        return s
    except Exception as e:
        print(f"  [skip] {year} {race}: {e}")
        return None


def build_features_for_session(year: int, race: str, laps: pd.DataFrame) -> pd.DataFrame:
    """Compute Frontiers-AI features per (Driver, LapNumber)."""
    laps = laps.copy()
    laps["LapNumber"] = laps["LapNumber"].astype(int, errors="ignore")
    # Drop laps with nan LapNumber
    laps = laps[laps["LapNumber"].notna()].copy()
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    laps["PittedThisLap"] = laps["PitInTime"].notna().astype(int)
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    # CumulativeTimeStint: cumulative LapTime within current Stint, per Driver.
    laps = laps.sort_values(["Driver", "LapNumber"]).copy()
    laps["CumulativeTimeStint"] = (
        laps.groupby(["Driver", "Stint"])["LapTimeSec"]
        .cumsum()
        .fillna(method="ffill")
    )

    # DriverAheadPit: at lap N, what was the AheadPittedThisLap on lap N-1?
    # Step 1: per lap, sort by Position, find driver-ahead PittedThisLap.
    rows = []
    for ln, g in laps.groupby("LapNumber"):
        g = g.dropna(subset=["Position"]).copy()
        if len(g) == 0:
            continue
        g["Position"] = g["Position"].astype(int)
        g = g.sort_values("Position")
        pos2driver = dict(zip(g["Position"], g["Driver"]))
        pos2pit = dict(zip(g["Position"], g["PittedThisLap"]))
        for _, row in g.iterrows():
            p = row["Position"]
            ap = pos2pit.get(p - 1, 0)
            rows.append({
                "Driver": row["Driver"],
                "LapNumber": ln,
                "AheadPittedThisLap": int(ap),
            })
    ahead_df = pd.DataFrame(rows)
    if len(ahead_df) == 0:
        return pd.DataFrame()
    # Shift by +1 so that AheadPittedThisLap on lap N-1 lands at LapNumber N.
    ahead_df["LapNumber"] = ahead_df["LapNumber"] + 1
    ahead_df = ahead_df.rename(columns={"AheadPittedThisLap": "DriverAheadPitLastLap"})

    # TrackStatus: take lap row's TrackStatus.
    laps["TrackStatus"] = (
        laps["TrackStatus"].astype(str).str.replace(",", "", regex=False)
    )
    # Convert to int code (use first digit if compound, e.g. "12" -> 1).
    laps["TrackStatusCode"] = (
        laps["TrackStatus"].str[0].astype(int, errors="ignore")
    )

    # delta_laptime = lap time minus last-lap time, per (Driver, Stint).
    laps["delta_laptime"] = laps.groupby(["Driver", "Stint"])["LapTimeSec"].diff()

    feat = laps[
        ["Driver", "LapNumber", "TrackStatusCode",
         "CumulativeTimeStint", "delta_laptime", "LapTimeSec"]
    ].drop_duplicates(subset=["Driver", "LapNumber"], keep="first")
    feat = feat.merge(ahead_df, on=["Driver", "LapNumber"], how="left")
    feat["DriverAheadPitLastLap"] = feat["DriverAheadPitLastLap"].fillna(0).astype(int)
    feat["Year"] = year
    feat["Race"] = race
    return feat


def main():
    t_total = time.time()
    print("=" * 60)
    print("Day-17 H2 — FastF1 external-data join")
    print("=" * 60)

    # === Load train/test ===
    t0 = time.time()
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    print(f"train: {len(tr)}, test: {len(te)}  ({time.time()-t0:.1f}s)")

    # Real-driver subset (3-letter alpha code).
    real_re = re.compile(r"^[A-Z]{3}$")
    tr["is_real"] = tr["Driver"].map(lambda d: bool(real_re.match(d))).astype(int)
    te["is_real"] = te["Driver"].map(lambda d: bool(real_re.match(d))).astype(int)
    print(f"real-driver rows: train {tr['is_real'].sum()}, test {te['is_real'].sum()}")

    combos = (
        pd.concat([tr[["Year", "Race"]], te[["Year", "Race"]]])
        .drop_duplicates().sort_values(["Year", "Race"])
    )
    combos = [
        (int(r.Year), str(r.Race))
        for r in combos.itertuples()
        if (r.Race, int(r.Year)) not in SKIP_COMBOS
    ]
    print(f"combos to pull: {len(combos)} (after skip list)")

    # === Pull all sessions ===
    t_pull = time.time()
    feats = []
    for i, (yr, race) in enumerate(combos):
        t1 = time.time()
        s = load_session_safe(yr, race)
        if s is None:
            continue
        try:
            f = build_features_for_session(yr, race, s.laps)
            if len(f) > 0:
                feats.append(f)
            print(f"  [{i+1}/{len(combos)}] {yr} {race}: {len(f)} rows ({time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(combos)}] {yr} {race}: build failed: {e}")
    pull_secs = time.time() - t_pull
    print(f"\n[fastf1 pull] total: {pull_secs:.1f}s")

    if len(feats) == 0:
        print("ABORT: no FastF1 features pulled")
        return

    feat_all = pd.concat(feats, ignore_index=True)
    print(f"[fastf1] feature rows: {len(feat_all)}")

    # === Merge to train/test ===
    t_merge = time.time()
    keys = ["Driver", "LapNumber", "Race", "Year"]
    feat_cols = [
        "DriverAheadPitLastLap", "TrackStatusCode",
        "CumulativeTimeStint", "delta_laptime", "LapTimeSec",
    ]
    tr_m = tr.merge(feat_all[keys + feat_cols], on=keys, how="left", indicator="_merge_tr")
    te_m = te.merge(feat_all[keys + feat_cols], on=keys, how="left", indicator="_merge_te")
    tr_m["matched"] = (tr_m["_merge_tr"] == "both").astype(int)
    te_m["matched"] = (te_m["_merge_te"] == "both").astype(int)
    print(f"[merge] train match-rate: {tr_m['matched'].mean():.4f}, test: {te_m['matched'].mean():.4f}  ({time.time()-t_merge:.1f}s)")
    print(f"[merge] train matched rows: {tr_m['matched'].sum()}/{len(tr_m)}")
    print(f"[merge] test matched rows: {te_m['matched'].sum()}/{len(te_m)}")

    # Fill NaN feats: median of matched rows for cont; 0 for binary.
    def fill_feats(df, ref):
        for c in feat_cols:
            if c in ("DriverAheadPitLastLap", "TrackStatusCode"):
                df[c] = df[c].fillna(0).astype(int)
            else:
                df[c] = df[c].fillna(ref[c].median())
        return df

    tr_m = fill_feats(tr_m, tr_m[tr_m["matched"] == 1])
    te_m = fill_feats(te_m, tr_m[tr_m["matched"] == 1])

    # === DGP-leak AV check ===
    # Train classifier predicting matched (1) vs unmatched (0); if AV-AUC ~0.5, OK.
    print("\n[DGP-leak AV] training matched-vs-unmatched classifier...")
    t_av = time.time()
    base_feats = ["LapNumber", "TyreLife", "Position", "LapTime (s)",
                  "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
                  "Position_Change", "Stint", "Year", "PitStop"]
    X_av = tr_m[base_feats].fillna(-1).values
    y_av = tr_m["matched"].values
    skf_av = StratifiedKFold(n_splits=3, shuffle=True, random_state=99)
    av_aucs = []
    for trii, tei in skf_av.split(X_av, y_av):
        m = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            n_jobs=2, verbose=-1, random_state=SEED,
        )
        m.fit(X_av[trii], y_av[trii])
        p = m.predict_proba(X_av[tei])[:, 1]
        av_aucs.append(roc_auc_score(y_av[tei], p))
    av_auc = float(np.mean(av_aucs))
    print(f"[DGP-leak AV] AV-AUC (matched-vs-unmatched): {av_auc:.4f}  ({time.time()-t_av:.1f}s)")
    pit_rate_matched = float(tr_m.loc[tr_m["matched"] == 1, TARGET].mean())
    pit_rate_unmatch = float(tr_m.loc[tr_m["matched"] == 0, TARGET].mean())
    print(f"[DGP-leak AV] PitNextLap mean — matched: {pit_rate_matched:.4f}, unmatched: {pit_rate_unmatch:.4f}")

    # === 80/20 honest holdout ===
    print("\n[holdout 80/20 seed=99] honest test...")
    t_hold = time.time()
    use_feats = base_feats + feat_cols
    # Compound and Race as label-encoded
    cat_maps = {}
    for cat in ["Compound", "Race", "Driver"]:
        u = pd.concat([tr_m[cat], te_m[cat]]).astype(str).unique()
        cat_maps[cat] = {v: i for i, v in enumerate(u)}
        tr_m[cat + "_le"] = tr_m[cat].astype(str).map(cat_maps[cat])
        te_m[cat + "_le"] = te_m[cat].astype(str).map(cat_maps[cat])
    use_feats = use_feats + ["Compound_le", "Race_le", "Driver_le"]

    X_full = tr_m[use_feats].values
    y_full = tr_m[TARGET].astype(int).values

    Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(
        X_full, y_full, test_size=0.2, random_state=99, stratify=y_full
    )
    m_h = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        min_child_samples=200, n_jobs=2, verbose=-1, random_state=SEED,
    )
    m_h.fit(Xh_tr, yh_tr, eval_set=[(Xh_te, yh_te)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    p_h = m_h.predict_proba(Xh_te)[:, 1]
    holdout_auc = float(roc_auc_score(yh_te, p_h))
    print(f"[holdout] AUC: {holdout_auc:.5f}  ({time.time()-t_hold:.1f}s)")

    # === 5-fold OOF ===
    print("\n[5-fold StratifiedKFold seed=42] OOF...")
    t_cv = time.time()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr_m), dtype=np.float64)
    test_preds = np.zeros(len(te_m), dtype=np.float64)
    fold_aucs = []
    Xtest = te_m[use_feats].values
    booster_imps = None
    for k, (trii, vai) in enumerate(skf.split(X_full, y_full)):
        m = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, n_jobs=2, verbose=-1, random_state=SEED + k,
        )
        m.fit(X_full[trii], y_full[trii],
              eval_set=[(X_full[vai], y_full[vai])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[vai] = m.predict_proba(X_full[vai])[:, 1]
        test_preds += m.predict_proba(Xtest)[:, 1] / N_FOLDS
        fa = roc_auc_score(y_full[vai], oof[vai])
        fold_aucs.append(fa)
        print(f"  fold {k}: {fa:.5f}")
        if booster_imps is None:
            booster_imps = m.booster_.feature_importance(importance_type="gain")
        else:
            booster_imps = booster_imps + m.booster_.feature_importance(importance_type="gain")
    oof_auc = float(roc_auc_score(y_full, oof))
    print(f"[5-fold] OOF AUC: {oof_auc:.5f}  (mean fold {np.mean(fold_aucs):.5f}; {time.time()-t_cv:.1f}s)")

    imp = list(zip(use_feats, booster_imps.tolist()))
    imp.sort(key=lambda x: -x[1])
    print("\n[importance top 10]")
    for n, v in imp[:10]:
        print(f"  {n:<35s} {v:.0f}")

    # Save artifacts: oof / test as (n,2) [neg, pos]
    np.save(ART / "oof_d17_h2_fastf1_strat.npy", np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d17_h2_fastf1_strat.npy", np.column_stack([1 - test_preds, test_preds]))

    summary = dict(
        timing=dict(
            total_secs=time.time() - t_total,
            fastf1_pull_secs=pull_secs,
        ),
        match_rate=dict(
            train=float(tr_m["matched"].mean()),
            test=float(te_m["matched"].mean()),
        ),
        match_counts=dict(
            train_matched=int(tr_m["matched"].sum()),
            train_total=int(len(tr_m)),
            test_matched=int(te_m["matched"].sum()),
            test_total=int(len(te_m)),
        ),
        dgp_leak_av=dict(
            av_auc=av_auc,
            pit_rate_matched=pit_rate_matched,
            pit_rate_unmatched=pit_rate_unmatch,
        ),
        holdout=dict(
            auc=holdout_auc,
        ),
        cv=dict(
            oof_auc=oof_auc,
            fold_aucs=[float(x) for x in fold_aucs],
        ),
        importance_top10=[(n, float(v)) for n, v in imp[:10]],
    )
    (ART / "d17_h2_fastf1_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] total {time.time()-t_total:.1f}s; results -> "
          f"{ART}/d17_h2_fastf1_results.json")


if __name__ == "__main__":
    main()
