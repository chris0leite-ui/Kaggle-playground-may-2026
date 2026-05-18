"""scripts/probe_r10_multi_alt_stack_blend.py — R10 multi-constituent alt-stack

PI directive: build alt-stack from 3-4 CPU weak learners, blend their meta with
R7.1 PRIMARY. Hypothesis: individual constituents too weak (LambdaRank 0.855)
but their LR-meta or rank-mean → ~0.93 with retained structural diversity.

Constituents:
  C1. LambdaRank stint-grouped  (existing, OOF 0.85512, ρ_test 0.745 to R7.1)
  C2. LambdaRank race-grouped   (new)
  C3. Rolling-features-only LGBM, NO single-row features (new)
  C4. Kernel-hazard estimate by (Compound, Stint, TyreLife)  (new)

Combiners:
  CM-A. Rank-mean of 4
  CM-B. 5-fold LR-meta of 4 (proper stacking)

Rank-blend: pick combiner with higher OOF AUC, blend with R7.1 at weights
{0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50}, find best in ρ-band.

Gates: best blend Δ ≥ +0.05 bp AND ρ_test ∈ [0.99, 0.9999] → CANDIDATE.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

NUM_COLS = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
CAT_COLS = ["Driver", "Compound", "Race"]

LGB_RANK = dict(
    objective="lambdarank", metric="ndcg",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=80,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    verbose=-1, n_jobs=-1, seed=SEED, label_gain=[0, 1],
)

LGB_BINARY = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=80,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    verbose=-1, n_jobs=-1, seed=SEED,
)


def encode_cats(train, test):
    tr, te = train.copy(), test.copy()
    for c in CAT_COLS:
        u = pd.concat([tr[c], te[c]], ignore_index=True).astype("category")
        m = {v: i for i, v in enumerate(u.cat.categories)}
        tr[c + "_cat"] = tr[c].map(m).astype("int32")
        te[c + "_cat"] = te[c].map(m).astype("int32")
    return tr, te


def lrank_fold_oof_test(X_tr, y, X_te, sid, splits, n_boost=400):
    """5-fold OOF on train + full-fit test prediction with lambdarank."""
    oof = np.zeros(len(y))
    for tr, va in splits:
        order = np.argsort(sid[tr], kind="stable")
        tr_s = tr[order]
        _, counts = np.unique(sid[tr_s], return_counts=True)
        ds = lgb.Dataset(X_tr[tr_s], label=y[tr_s], group=counts)
        bst = lgb.train(LGB_RANK, ds, num_boost_round=n_boost,
                        callbacks=[lgb.log_evaluation(0)])
        oof[va] = bst.predict(X_tr[va])
    order_all = np.argsort(sid, kind="stable")
    _, counts_all = np.unique(sid[order_all], return_counts=True)
    ds_all = lgb.Dataset(X_tr[order_all], label=y[order_all], group=counts_all)
    bst_full = lgb.train(LGB_RANK, ds_all, num_boost_round=n_boost,
                         callbacks=[lgb.log_evaluation(0)])
    test_pred = bst_full.predict(X_te)
    return oof, test_pred


def lgbm_binary_oof_test(X_tr, y, X_te, splits, n_boost=600):
    oof = np.zeros(len(y))
    for tr, va in splits:
        ds = lgb.Dataset(X_tr[tr], label=y[tr])
        dv = lgb.Dataset(X_tr[va], label=y[va], reference=ds)
        bst = lgb.train(LGB_BINARY, ds, num_boost_round=n_boost,
                        valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = bst.predict(X_tr[va])
    ds_all = lgb.Dataset(X_tr, label=y)
    bst_full = lgb.train(LGB_BINARY, ds_all, num_boost_round=n_boost)
    test_pred = bst_full.predict(X_te)
    return oof, test_pred


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(Driver, Race, Year) rolling-window aggregates. NO single-row features.
    Returns a DataFrame with the original row order preserved."""
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df))
    df = df.sort_values(["Driver", "Race", "Year", "LapNumber"],
                        kind="stable").reset_index(drop=True)
    g = df.groupby(["Driver", "Race", "Year"], sort=False)
    out = pd.DataFrame(index=df.index)
    out["_orig_idx"] = df["_orig_idx"].values

    # Cumulative pit count (past pits, exclusive of current lap)
    out["cum_pit_excl"] = g["PitStop"].cumsum().values - df["PitStop"].values
    # Laps since last pit
    pit_lap = df["LapNumber"].where(df["PitStop"] == 1)
    out["lap_since_pit"] = (df["LapNumber"]
                            - pit_lap.groupby([df["Driver"], df["Race"],
                                               df["Year"]]).ffill()).fillna(df["LapNumber"]).values

    for col in ["LapTime (s)", "LapTime_Delta", "Position",
                "TyreLife", "Cumulative_Degradation"]:
        for w in [3, 5]:
            out[f"roll{w}_mean_{col}"] = g[col].transform(
                lambda s: s.rolling(w, min_periods=1).mean()).values
            out[f"roll{w}_std_{col}"] = g[col].transform(
                lambda s: s.rolling(w, min_periods=1).std()).fillna(0).values
        out[f"lag1_{col}"] = g[col].shift(1).fillna(df[col]).values
        out[f"lag2_{col}"] = g[col].shift(2).fillna(df[col]).values
        out[f"delta1_{col}"] = (df[col].values - out[f"lag1_{col}"].values)

    # Restore original row order
    out = out.sort_values("_orig_idx", kind="stable").reset_index(drop=True)
    out = out.drop(columns=["_orig_idx"])
    return out


def kernel_hazard_oof_test(train, test, y, splits, bw=2):
    """Fold-safe kernel hazard h(Compound, Stint, TyreLife).
    For each TyreLife t in window [t-bw, t+bw] within (Compound, Stint), use
    training-rows-only mean of y. Predicts hazard probability."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))

    def build_tbl(tr_df, tr_y):
        # Build a (Compound, Stint, TyreLife) → smoothed-rate table on training rows
        tmp = tr_df.copy()
        tmp["_y"] = tr_y
        agg = tmp.groupby(["Compound", "Stint", "TyreLife"]).agg(
            sum_y=("_y", "sum"), n=("_y", "size")).reset_index()
        # Kernel-smooth over TyreLife within (Compound, Stint)
        agg = agg.sort_values(["Compound", "Stint", "TyreLife"])
        rows = []
        for (c, s), grp in agg.groupby(["Compound", "Stint"]):
            tl = grp["TyreLife"].values
            sy = grp["sum_y"].values
            n = grp["n"].values
            for i in range(len(tl)):
                mask = np.abs(tl - tl[i]) <= bw
                num = sy[mask].sum() + 5.0 * tr_y.mean()
                den = n[mask].sum() + 5.0
                rows.append((c, s, tl[i], num / den))
        return pd.DataFrame(rows, columns=["Compound", "Stint", "TyreLife",
                                            "haz"])

    print("    fold ", end="", flush=True)
    for fold, (tr, va) in enumerate(splits):
        tbl = build_tbl(train.iloc[tr], y[tr])
        merged = train.iloc[va].merge(tbl, on=["Compound", "Stint", "TyreLife"],
                                      how="left")
        oof[va] = merged["haz"].fillna(y[tr].mean()).values
        print(f"{fold+1} ", end="", flush=True)
    print("done.")
    tbl_full = build_tbl(train, y)
    merged_te = test.merge(tbl_full, on=["Compound", "Stint", "TyreLife"],
                           how="left")
    test_pred = merged_te["haz"].fillna(y.mean()).values
    return oof, test_pred


def main() -> None:
    t0 = time.time()
    print("== R10 multi-constituent alt-stack + rank-blend with R7.1 PRIMARY ==")
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    train_e, test_e = encode_cats(train, test)
    feats_row = NUM_COLS + [c + "_cat" for c in CAT_COLS]
    print(f"  rows train: {len(train):,}  test: {len(test):,}  "
          f"pos: {y.sum():,} ({y.mean()*100:.2f}%)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # ----- C1: existing LambdaRank stint-grouped -----
    print("\n[C1] Loading existing LambdaRank stint-grouped ...")
    c1_oof = np.load(ART / "oof_R10_lambdarank_strat.npy")
    c1_test = np.load(ART / "test_R10_lambdarank_strat.npy")
    auc_c1 = float(roc_auc_score(y, c1_oof))
    print(f"  C1 OOF AUC: {auc_c1:.5f}")

    # ----- C2: LambdaRank race-grouped -----
    print("\n[C2] Training LambdaRank race-grouped (5-fold + full-fit) ...")
    t_c2 = time.time()
    sid_race = (train_e["Race"].astype(str) + "_" +
                train_e["Year"].astype(str)).values
    X_tr = train_e[feats_row].values
    X_te = test_e[feats_row].values
    c2_oof, c2_test = lrank_fold_oof_test(X_tr, y, X_te, sid_race, splits)
    auc_c2 = float(roc_auc_score(y, c2_oof))
    print(f"  C2 OOF AUC: {auc_c2:.5f}  wall={time.time()-t_c2:.1f}s")
    np.save(ART / "oof_R10_C2_lrank_race_strat.npy", c2_oof)
    np.save(ART / "test_R10_C2_lrank_race_strat.npy", c2_test)

    # ----- C3: Rolling-features LGBM (no single-row features) -----
    print("\n[C3] Building rolling features + LGBM ...")
    t_c3 = time.time()
    roll_train = build_rolling_features(train_e)
    roll_test  = build_rolling_features(test_e)
    print(f"  rolling features: {roll_train.shape[1]}  built wall={time.time()-t_c3:.1f}s")
    roll_cols = roll_train.columns.tolist()
    X_tr_roll = roll_train[roll_cols].values.astype(np.float32)
    X_te_roll = roll_test[roll_cols].values.astype(np.float32)
    c3_oof, c3_test = lgbm_binary_oof_test(X_tr_roll, y, X_te_roll, splits)
    auc_c3 = float(roc_auc_score(y, c3_oof))
    print(f"  C3 OOF AUC: {auc_c3:.5f}  wall={time.time()-t_c3:.1f}s")
    np.save(ART / "oof_R10_C3_rolling_strat.npy", c3_oof)
    np.save(ART / "test_R10_C3_rolling_strat.npy", c3_test)

    # ----- C4: Kernel hazard -----
    print("\n[C4] Kernel hazard by (Compound, Stint, TyreLife) ...")
    t_c4 = time.time()
    c4_oof, c4_test = kernel_hazard_oof_test(train, test, y, splits, bw=2)
    auc_c4 = float(roc_auc_score(y, c4_oof))
    print(f"  C4 OOF AUC: {auc_c4:.5f}  wall={time.time()-t_c4:.1f}s")
    np.save(ART / "oof_R10_C4_kernel_haz_strat.npy", c4_oof)
    np.save(ART / "test_R10_C4_kernel_haz_strat.npy", c4_test)

    # ----- Combiners -----
    print("\n[Combiners] Build alt-stack from C1..C4 ...")
    n_tr, n_te = len(y), len(c1_test)
    constituents_oof = [c1_oof, c2_oof, c3_oof, c4_oof]
    constituents_test = [c1_test, c2_test, c3_test, c4_test]
    names = ["C1_lrank_stint", "C2_lrank_race", "C3_rolling", "C4_kernel_haz"]
    print("  Constituent OOF AUC and ρ_OOF vs R7.1:")

    r71_oof = np.load(ART / "oof_K13_pathb_driverclass_stint_tau100000.npy")
    r71_test = np.load(ART / "test_K13_pathb_driverclass_stint_tau100000.npy")
    auc_r71 = float(roc_auc_score(y, r71_oof))
    print(f"  R7.1 PRIMARY OOF: {auc_r71:.5f}")
    for nm, p_oof, p_te in zip(names, constituents_oof, constituents_test):
        rho_oof = float(spearmanr(p_oof, r71_oof)[0])
        rho_te = float(spearmanr(p_te, r71_test)[0])
        auc = float(roc_auc_score(y, p_oof))
        print(f"    {nm:18s}  OOF={auc:.5f}  ρ_OOF={rho_oof:+.5f}  ρ_test={rho_te:+.5f}")

    # CM-A: rank-mean (over OOF ranks; test ranks)
    rk_oof = np.column_stack([rankdata(p) / n_tr for p in constituents_oof])
    rk_te  = np.column_stack([rankdata(p) / n_te for p in constituents_test])
    cm_a_oof = rk_oof.mean(axis=1)
    cm_a_te  = rk_te.mean(axis=1)
    auc_cm_a = float(roc_auc_score(y, cm_a_oof))
    print(f"\n  CM-A rank-mean OOF AUC: {auc_cm_a:.5f}")

    # CM-B: 5-fold LR-meta on rank+logit features
    F_oof = np.column_stack([rk_oof,
                              np.column_stack([np.clip(p, 1e-9, 1 - 1e-9)
                                               for p in constituents_oof])])
    F_te  = np.column_stack([rk_te,
                              np.column_stack([np.clip(p, 1e-9, 1 - 1e-9)
                                               for p in constituents_test])])
    cm_b_oof = np.zeros(len(y))
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        cm_b_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    lr_full.fit(F_oof, y)
    cm_b_te = lr_full.predict_proba(F_te)[:, 1]
    auc_cm_b = float(roc_auc_score(y, cm_b_oof))
    print(f"  CM-B LR-meta  OOF AUC:  {auc_cm_b:.5f}")
    print(f"  LR coefs: {dict(zip(names + [n+'_p' for n in names], lr_full.coef_.flatten().round(3).tolist()))}")

    # Pick stronger combiner
    if auc_cm_b >= auc_cm_a:
        cm_oof, cm_te, cm_name, auc_cm = cm_b_oof, cm_b_te, "CM-B_LR_meta", auc_cm_b
    else:
        cm_oof, cm_te, cm_name, auc_cm = cm_a_oof, cm_a_te, "CM-A_rank_mean", auc_cm_a
    rho_cm_oof = float(spearmanr(cm_oof, r71_oof)[0])
    rho_cm_te  = float(spearmanr(cm_te, r71_test)[0])
    print(f"\n  Picked combiner: {cm_name}  OOF={auc_cm:.5f}  "
          f"ρ_OOF={rho_cm_oof:+.5f}  ρ_test={rho_cm_te:+.5f}")

    np.save(ART / f"oof_R10_alt_stack_{cm_name}.npy", cm_oof)
    np.save(ART / f"test_R10_alt_stack_{cm_name}.npy", cm_te)

    # ----- Rank-blend with R7.1 -----
    print(f"\n[Blend] Rank-blend R7.1 ({auc_r71:.5f}) with alt-stack ({auc_cm:.5f}) ...")
    rk_oof_r71 = rankdata(r71_oof) / n_tr
    rk_oof_cm  = rankdata(cm_oof) / n_tr
    rk_te_r71  = rankdata(r71_test) / n_te
    rk_te_cm   = rankdata(cm_te)   / n_te

    weights = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50]
    print(f"\n  {'w(R7.1)':>8} {'w(alt)':>7} {'blend OOF':>10} {'Δ vs R7.1':>13} "
          f"{'ρ_test vs R7.1':>16}")
    results = []
    for w in weights:
        blend_oof = w * rk_oof_r71 + (1 - w) * rk_oof_cm
        blend_te  = w * rk_te_r71  + (1 - w) * rk_te_cm
        auc_b = float(roc_auc_score(y, blend_oof))
        delta = (auc_b - auc_r71) * 1e4
        rho_b = float(spearmanr(blend_te, r71_test)[0])
        results.append(dict(w_r71=w, w_alt=1 - w, auc=auc_b,
                            delta_bp=delta, rho_test_vs_r71=rho_b))
        m = " ★" if delta >= 0.05 else (" =" if delta >= -0.02 else " ↓")
        print(f"  {w:>8.2f} {1-w:>7.2f} {auc_b:>10.5f} {delta:>+10.3f} bp "
              f"{rho_b:>16.6f}{m}")

    eligible = [r for r in results if 0.99 <= r["rho_test_vs_r71"] <= 0.9999]
    if eligible:
        best = max(eligible, key=lambda r: r["delta_bp"])
        verdict = "CANDIDATE" if best["delta_bp"] >= 0.05 else (
                  "MARGINAL"  if best["delta_bp"] >= -0.02 else "NULL")
        print(f"\n  Best in ρ-band: w_R71={best['w_r71']:.2f}  Δ {best['delta_bp']:+.3f} bp"
              f"  ρ_test {best['rho_test_vs_r71']:.6f}  → {verdict}")
        if verdict == "CANDIDATE":
            Path("submissions").mkdir(exist_ok=True)
            w = best["w_r71"]
            blend_te = w * rk_te_r71 + (1 - w) * rk_te_cm
            sub = pd.DataFrame({"id": test["id"].values,
                                TARGET: np.clip(blend_te, 0.001, 0.999)})
            suffix = f"{int(w*100):02d}_{int((1-w)*100):02d}"
            p = f"submissions/submission_R10_blend_r71_altstack_{suffix}.csv"
            sub.to_csv(p, index=False)
            print(f"  Wrote: {p}")
    else:
        best = None
        verdict = "NULL"
        print(f"\n  No weight in ρ-band [0.99, 0.9999]  → NULL")

    out = {
        "constituent_aucs": {nm: float(roc_auc_score(y, p))
                              for nm, p in zip(names, constituents_oof)},
        "rho_constituents_vs_r71_oof": {nm: float(spearmanr(p, r71_oof)[0])
                                         for nm, p in zip(names, constituents_oof)},
        "rho_constituents_vs_r71_test": {nm: float(spearmanr(p, r71_test)[0])
                                          for nm, p in zip(names, constituents_test)},
        "cm_a_auc": auc_cm_a, "cm_b_auc": auc_cm_b,
        "combiner_picked": cm_name, "combiner_auc": auc_cm,
        "rho_alt_vs_r71_oof": rho_cm_oof,
        "rho_alt_vs_r71_test": rho_cm_te,
        "primary_oof": auc_r71,
        "blend_results": results,
        "best_in_band": best,
        "verdict": verdict,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_r10_multi_alt_stack_blend.json").write_text(json.dumps(out, indent=2))
    print(f"\nWall {time.time()-t0:.1f}s. Wrote probe_r10_multi_alt_stack_blend.json")


if __name__ == "__main__":
    main()
