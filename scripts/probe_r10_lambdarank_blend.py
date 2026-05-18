"""scripts/probe_r10_lambdarank_blend.py — Round 10 alt-stack mode

Pool-structural rank-lock at K=13+Path-B confirmed (R10 op-vs-pool
check). Pivot: build STANDALONE alt-mechanism predictions and
RANK-BLEND with R7.1 PRIMARY — bypassing the meta layer entirely.

Mechanism: LambdaRank LGBM with per-(Race, Driver, Year, Stint)
group queries. Training objective = lambdarank (pairwise inversions
within group); eval = row-AUC. Inductive bias is fundamentally
different from row-LGBM log-loss; prior K=21 era recorded ρ=0.942
to LR-meta (FAR from typical 0.999+ base ρ). HIGH structural
diversity is exactly what rank-blending needs.

Gates:
  G1 standalone OOF ≥ 0.92  (diverse, not random)
  G2 rank-blend OOF beats R7.1 PRIMARY by ≥ +0.05 bp at any tested
     weight AND ρ_test ∈ [0.99, 0.9999]  → CANDIDATE
  G3 Rule 27 ρ-band  TIE_ZONE → hedge only; OK → submit-eligible
  KILL: no blend weight achieves Δ ≥ +0.02 bp.

Cost: ~25-30 min CPU (5-fold OOF + 1 full-train fit + blend sweep).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
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


def encode_cats(train: pd.DataFrame, test: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy(); te = test.copy()
    for c in CAT_COLS:
        u = pd.concat([tr[c], te[c]], ignore_index=True).astype("category")
        cat_map = {v: i for i, v in enumerate(u.cat.categories)}
        tr[c + "_cat"] = tr[c].map(cat_map).astype("int32")
        te[c + "_cat"] = te[c].map(cat_map).astype("int32")
    return tr, te


def main() -> None:
    t0 = time.time()
    print("== R10 LambdaRank standalone + rank-blend with R7.1 PRIMARY ==")
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    train_e, test_e = encode_cats(train, test)
    feats = NUM_COLS + [c + "_cat" for c in CAT_COLS]

    sid = (train_e["Race"].astype(str) + "_" +
           train_e["Driver"].astype(str) + "_" +
           train_e["Year"].astype(str) + "_" +
           train_e["Stint"].astype(str)).values
    print(f"  rows: {len(train_e):,}  test: {len(test_e):,}  "
          f"stints: {len(set(sid)):,}  pos: {y.sum():,} ({y.mean()*100:.2f}%)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # ---- 5-fold OOF LambdaRank ----
    print("\n[1/3] Training LambdaRank 5-fold ...")
    oof = np.zeros(len(y))
    fold_aucs = []
    X_tr_all = train_e[feats].values
    X_te = test_e[feats].values
    for fold, (tr, va) in enumerate(splits):
        t_f = time.time()
        order = np.argsort(sid[tr], kind="stable")
        tr_s = tr[order]
        _, counts = np.unique(sid[tr_s], return_counts=True)
        ds_tr = lgb.Dataset(X_tr_all[tr_s], label=y[tr_s], group=counts)
        booster = lgb.train(LGB_RANK, ds_tr, num_boost_round=400,
                            callbacks=[lgb.log_evaluation(0)])
        oof[va] = booster.predict(X_tr_all[va])
        auc_f = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(auc_f)
        print(f"  fold {fold+1}: row-AUC={auc_f:.5f}  wall={time.time()-t_f:.1f}s")
    auc_oof = float(roc_auc_score(y, oof))
    print(f"  Standalone OOF AUC: {auc_oof:.5f}  fold-std={np.std(fold_aucs):.5f}")

    # ---- Full-train fit → test predictions ----
    print("\n[2/3] Full-train fit → test predictions ...")
    t_f = time.time()
    order_all = np.argsort(sid, kind="stable")
    _, counts_all = np.unique(sid[order_all], return_counts=True)
    ds_all = lgb.Dataset(X_tr_all[order_all], label=y[order_all],
                         group=counts_all)
    booster_full = lgb.train(LGB_RANK, ds_all, num_boost_round=400,
                             callbacks=[lgb.log_evaluation(0)])
    test_pred = booster_full.predict(X_te)
    print(f"  test pred range: [{test_pred.min():.3f}, {test_pred.max():.3f}]"
          f"  mean={test_pred.mean():.3f}  wall={time.time()-t_f:.1f}s")

    np.save(ART / "oof_R10_lambdarank_strat.npy", oof.astype(np.float64))
    np.save(ART / "test_R10_lambdarank_strat.npy", test_pred.astype(np.float64))
    print(f"  Saved: oof_R10_lambdarank_strat.npy + test_R10_lambdarank_strat.npy")

    # ---- Rank-blend with R7.1 PRIMARY ----
    print("\n[3/3] Rank-blend with R7.1 PRIMARY ...")
    r71_oof  = np.load(ART / "oof_K13_pathb_driverclass_stint_tau100000.npy")
    r71_test = np.load(ART / "test_K13_pathb_driverclass_stint_tau100000.npy")
    auc_r71 = float(roc_auc_score(y, r71_oof))
    rho_lr_oof  = float(spearmanr(oof, r71_oof)[0])
    rho_lr_test = float(spearmanr(test_pred, r71_test)[0])
    print(f"  R7.1 PRIMARY K=13+Path-B OOF: {auc_r71:.5f}")
    print(f"  ρ_OOF(λrank, R7.1):  {rho_lr_oof:.5f}")
    print(f"  ρ_test(λrank, R7.1): {rho_lr_test:.5f}")

    n_tr = len(y); n_te = len(test_pred)
    rk_oof_lr  = rankdata(oof)  / n_tr
    rk_oof_r71 = rankdata(r71_oof) / n_tr
    rk_te_lr   = rankdata(test_pred) / n_te
    rk_te_r71  = rankdata(r71_test) / n_te

    weights = [0.99, 0.97, 0.95, 0.90, 0.80, 0.70, 0.50]
    print(f"\n  {'w(R7.1)':>8} {'w(λrank)':>9} {'blend OOF':>10} {'Δ vs R7.1':>11} "
          f"{'ρ_test vs R7.1':>15}")
    results = []
    for w in weights:
        blend_oof = w * rk_oof_r71 + (1 - w) * rk_oof_lr
        blend_te  = w * rk_te_r71  + (1 - w) * rk_te_lr
        auc_b = float(roc_auc_score(y, blend_oof))
        delta = (auc_b - auc_r71) * 1e4
        rho_b = float(spearmanr(blend_te, r71_test)[0])
        results.append(dict(w_r71=w, w_lrank=1 - w, auc=auc_b,
                            delta_bp=delta, rho_test_vs_r71=rho_b))
        marker = " ★" if delta >= 0.05 else ("  " if delta >= 0 else " ↓")
        print(f"  {w:>8.2f} {1-w:>9.2f} {auc_b:>10.5f} {delta:>+10.3f} bp "
              f"{rho_b:>15.6f}{marker}")

    # Pick best blend by OOF AUC subject to ρ_test ∈ [0.99, 0.9999]
    eligible = [r for r in results
                if 0.99 <= r["rho_test_vs_r71"] <= 0.9999]
    if eligible:
        best = max(eligible, key=lambda r: r["delta_bp"])
        verdict = "CANDIDATE" if best["delta_bp"] >= 0.05 else "MARGINAL"
        print(f"\n  Best in ρ-band [0.99, 0.9999]: w_R71={best['w_r71']:.2f}, "
              f"Δ {best['delta_bp']:+.3f} bp, ρ_test {best['rho_test_vs_r71']:.6f}")
        print(f"  Verdict: {verdict}")
        # Save best-blend submission CSV
        Path("submissions").mkdir(exist_ok=True)
        w = best["w_r71"]
        blend_te = w * rk_te_r71 + (1 - w) * rk_te_lr
        sub = pd.DataFrame({"id": test["id"].values,
                            TARGET: np.clip(blend_te, 0.001, 0.999)})
        suffix = f"{int(w*100):02d}_{int((1-w)*100):02d}"
        sub_path = f"submissions/submission_R10_blend_r71_lrank_{suffix}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"  Wrote: {sub_path}")
    else:
        best = None
        print(f"\n  No weight lands in ρ-band [0.99, 0.9999]")
        print(f"  Verdict: NULL")

    # Save full results JSON
    out = {
        "lambdarank_oof_auc": auc_oof,
        "fold_aucs": fold_aucs,
        "rho_oof_lambdarank_r71": rho_lr_oof,
        "rho_test_lambdarank_r71": rho_lr_test,
        "primary_oof_auc": auc_r71,
        "blend_results": results,
        "best_in_band": best,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_r10_lambdarank_blend.json").write_text(json.dumps(out, indent=2))
    print(f"\nWall {time.time()-t0:.1f}s. Wrote probe_r10_lambdarank_blend.json")


if __name__ == "__main__":
    main()
