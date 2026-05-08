"""scripts/probe_lane1_downsampling.py — Lane 1 (downsampling / censoring).

Diagnostic + active probes for the censoring assumption: each row is a
sparse sample of a longer trajectory; PitNextLap means "the next observed
row's stint started fresh", which mixes horizons of length 1, 2, ..., k
laps depending on the gap to the next observed row.

Hypothesis: explicitly modelling the gap-to-next-observed-row recovers a
4th logit direction that the K=4 pool (which treats rows i.i.d.) cannot.

Probes (each writes a row to the JSON output):
  D1.1 — gap_to_next_obs distribution and conditional P(PitNextLap | gap)
  D1.2 — per-gap calibration of K=4 PRIMARY (deciles per gap-bucket)
  P1.1 — gap features added as meta inputs to K=4 LR meta
  P1.3 — per-gap-bucket isotonic recalibration of K=4 PRIMARY

P1.2 (discrete-time hazard) is a separate, larger script — sketched at
end-of-file as `# TODO P1.2`.

Cost (CPU): ~30 min combined. All probes share gap features; load once.

Outputs:
  scripts/artifacts/probe_lane1_downsampling.json
  scripts/artifacts/oof_lane1_K4plus_gap_meta_strat.npy  (P1.1 OOF)
  scripts/artifacts/test_lane1_K4plus_gap_meta_strat.npy
  scripts/artifacts/oof_lane1_per_gap_isotonic_strat.npy (P1.3)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 500

K4_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def build_gap_features(train: pd.DataFrame,
                       test: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Combined-frame gap features. Per (Race, Driver, Year, Stint),
    sort by LapNumber, compute lap distance to next/prev observed row.

    AV-safe per A3 (row-iid train/test). Fold-safe per Rule 24 (no
    label use; only LapNumber and group keys).
    """
    n_tr = len(train)
    df = pd.concat([train.assign(_split="tr"),
                    test.assign(_split="te")],
                   ignore_index=True, sort=False)
    df["row_id"] = np.arange(len(df))
    keys = ["Race", "Driver", "Year", "Stint"]
    df = df.sort_values(keys + ["LapNumber"], kind="stable").reset_index(drop=True)
    g = df.groupby(keys, sort=False)
    df["next_lap"] = g["LapNumber"].shift(-1)
    df["prev_lap"] = g["LapNumber"].shift(1)
    df["gap_to_next_obs"] = (df["next_lap"] - df["LapNumber"]).fillna(-1).astype(float)
    df["gap_to_prev_obs"] = (df["LapNumber"] - df["prev_lap"]).fillna(-1).astype(float)
    df["stint_size"] = g["LapNumber"].transform("size")
    df["stint_lap_idx"] = g["LapNumber"].rank("dense").astype(int) - 1
    df["is_last_in_stint"] = (df["next_lap"].isna()).astype(int)
    # Within-stint, what fraction of unrecorded laps appear before this row?
    df["stint_min_lap"] = g["LapNumber"].transform("min")
    df["stint_max_lap"] = g["LapNumber"].transform("max")
    df["stint_span"] = df["stint_max_lap"] - df["stint_min_lap"] + 1
    df["stint_density"] = df["stint_size"] / df["stint_span"].clip(lower=1)
    feats = ["gap_to_next_obs", "gap_to_prev_obs", "stint_lap_idx",
             "is_last_in_stint", "stint_density", "stint_size"]
    df = df.sort_values("row_id").reset_index(drop=True)
    return (df.iloc[:n_tr].reset_index(drop=True),
            df.iloc[n_tr:].reset_index(drop=True),
            feats)


def main():
    t0 = time.time()
    print("Loading data + K=4 base OOFs ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4_FWD]
    base_tests = [_pos(ART / f"test_{n}_strat.npy") for n in K4_FWD]
    P4 = np.column_stack(base_oofs)
    P4_test = np.column_stack(base_tests)

    primary_oof = _pos(ART / "oof_K4_fwd_pathb_strat.npy")  # K=4 + Path-B
    primary_test = _pos(ART / "test_K4_fwd_pathb_strat.npy")

    print("Building gap features ...")
    tr_x, te_x, gap_feats = build_gap_features(train, test)

    # ============ D1.1 — gap distribution + P(target | gap) =========
    print("\n--- D1.1: gap_to_next_obs distribution and conditional P(target|gap)")
    gap_tr = tr_x["gap_to_next_obs"].astype(int).values
    gap_buckets = pd.cut(gap_tr, bins=[-1.5, 0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 1000],
                         labels=["last", "1", "2", "3", "4-5", "6-10", "11+"])
    gap_table = pd.DataFrame({
        "gap_bucket": gap_buckets,
        "y": y,
    }).groupby("gap_bucket", observed=True).agg(
        n=("y", "size"),
        p_pit=("y", "mean"),
    ).reset_index()
    print(gap_table.to_string(index=False))

    # ============ D1.2 — per-gap calibration of K=4 PRIMARY ==========
    print("\n--- D1.2: per-gap calibration of K=4 PRIMARY")
    cal_rows = []
    for bucket in gap_table["gap_bucket"]:
        mask = (gap_buckets == bucket).values
        if mask.sum() < 100:
            continue
        p_mean = primary_oof[mask].mean()
        y_mean = y[mask].mean()
        ece = abs(p_mean - y_mean)
        # AUC within bucket if both classes present
        sub_auc = float("nan")
        if y[mask].min() != y[mask].max():
            sub_auc = float(roc_auc_score(y[mask], primary_oof[mask]))
        cal_rows.append({
            "gap_bucket": str(bucket),
            "n": int(mask.sum()),
            "PRIMARY_mean_p": float(p_mean),
            "y_mean": float(y_mean),
            "ECE": float(ece),
            "PRIMARY_AUC_within_bucket": sub_auc,
        })
    for r in cal_rows:
        print(f"  {r['gap_bucket']:>5s}  n={r['n']:>7d}  "
              f"p̂={r['PRIMARY_mean_p']:.4f}  ȳ={r['y_mean']:.4f}  "
              f"ECE={r['ECE']:.4f}  AUC={r['PRIMARY_AUC_within_bucket']:.4f}")

    # ============ P1.1 — K=4 + gap features as meta input ===========
    print("\n--- P1.1: gap features as meta input alongside K=4 [P, rank, logit]")
    F4 = _expand(P4)  # 12 features
    G_tr = tr_x[gap_feats].astype(float).values
    G_te = te_x[gap_feats].astype(float).values
    F4G = np.hstack([F4, G_tr])
    F4_test = _expand(P4_test)
    F4G_test = np.hstack([F4_test, G_te])

    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED).split(np.zeros(len(y)), y))

    def fit_lr_meta(F, F_test):
        oof_pred = np.zeros(len(y))
        test_acc = np.zeros(F_test.shape[0])
        for tr, va in splits:
            lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            oof_pred[va] = lr.predict_proba(F[va])[:, 1]
            test_acc += lr.predict_proba(F_test)[:, 1] / N_FOLDS
        return oof_pred, test_acc

    oof_K4, _ = fit_lr_meta(F4, F4_test)
    oof_K4G, test_K4G = fit_lr_meta(F4G, F4G_test)
    auc_K4 = float(roc_auc_score(y, oof_K4))
    auc_K4G = float(roc_auc_score(y, oof_K4G))
    delta_p11_bp = (auc_K4G - auc_K4) * 1e4
    print(f"  K=4 LR meta plain     : {auc_K4:.5f}")
    print(f"  K=4 + gap meta input  : {auc_K4G:.5f}  (Δ {delta_p11_bp:+.3f} bp)")
    np.save(ART / "oof_lane1_K4plus_gap_meta_strat.npy", oof_K4G)
    np.save(ART / "test_lane1_K4plus_gap_meta_strat.npy",
            np.column_stack([1 - test_K4G, test_K4G]))

    # ============ P1.3 — per-gap-bucket isotonic recalibration ======
    print("\n--- P1.3: per-gap-bucket isotonic recalibration of PRIMARY")
    primary_recal = primary_oof.copy()
    fold_assign = np.zeros(len(y), dtype=int)
    for fold, (_, va) in enumerate(splits):
        fold_assign[va] = fold

    bucket_codes = pd.Categorical(gap_buckets).codes
    n_buckets = bucket_codes.max() + 1
    for b in range(n_buckets):
        bucket_mask = (bucket_codes == b)
        if bucket_mask.sum() < 200:
            continue
        # Per-fold isotonic, fit on train-fold rows of this bucket
        for fold in range(N_FOLDS):
            tr_mask = bucket_mask & (fold_assign != fold)
            va_mask = bucket_mask & (fold_assign == fold)
            if tr_mask.sum() < 100 or va_mask.sum() < 1:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(primary_oof[tr_mask], y[tr_mask])
            primary_recal[va_mask] = iso.transform(primary_oof[va_mask])

    auc_primary = float(roc_auc_score(y, primary_oof))
    auc_recal = float(roc_auc_score(y, primary_recal))
    delta_p13_bp = (auc_recal - auc_primary) * 1e4
    print(f"  PRIMARY plain          : {auc_primary:.5f}")
    print(f"  PRIMARY + per-gap iso  : {auc_recal:.5f}  (Δ {delta_p13_bp:+.3f} bp)")
    np.save(ART / "oof_lane1_per_gap_isotonic_strat.npy", primary_recal)

    # ρ vs PRIMARY
    rho_p11 = float(spearmanr(oof_K4G, primary_oof)[0])
    rho_p13 = float(spearmanr(primary_recal, primary_oof)[0])

    out = {
        "K4_bases": K4_FWD,
        "gap_features": gap_feats,
        "D1_1_gap_distribution": gap_table.to_dict(orient="records"),
        "D1_2_per_gap_calibration": cal_rows,
        "P1_1_K4_LR_meta_plain_oof": auc_K4,
        "P1_1_K4_plus_gap_meta_oof": auc_K4G,
        "P1_1_delta_bp": float(delta_p11_bp),
        "P1_1_rho_vs_primary": rho_p11,
        "P1_3_PRIMARY_oof": auc_primary,
        "P1_3_PRIMARY_per_gap_iso_oof": auc_recal,
        "P1_3_delta_bp": float(delta_p13_bp),
        "P1_3_rho_vs_primary": rho_p13,
        "verdict_P1_1": ("PASS" if delta_p11_bp >= 0.5
                         else "AMBIG" if delta_p11_bp >= -0.1
                         else "NULL"),
        "verdict_P1_3": ("PASS" if delta_p13_bp >= 0.5
                         else "AMBIG" if delta_p13_bp >= -0.1
                         else "NULL"),
        "wall_s": time.time() - t0,
    }
    (ART / "probe_lane1_downsampling.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {ART/'probe_lane1_downsampling.json'}. Wall {out['wall_s']:.1f}s")
    print(f"Verdicts: P1.1 {out['verdict_P1_1']} | P1.3 {out['verdict_P1_3']}")

    # TODO P1.2 — discrete-time hazard.
    # For each row, train binary heads {pit_in_next_k_laps for k in
    # [1,2,3,5,10]}; marginalise to PitNextLap by reading off the
    # row's actual gap_to_next_obs. K=4+1 gate as a single base.
    # Cost: ~45 min CPU. Implement as scripts/probe_lane1_hazard.py
    # if P1.1 fires (PASS), otherwise skip.


if __name__ == "__main__":
    main()
