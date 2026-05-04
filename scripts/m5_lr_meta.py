"""M5 — LR meta-stacker on [raw, rank, logit] over the base-model pool.

Pool: baseline, d2a_te, m2_xgb, m3_catboost, m4_relstate.
Per analyticaobscura recipe (cross-comp research Source 1 #2):
LogisticRegression on three representations of each base OOF.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
BASE_S, BASE_G = 0.94075, 0.92059
N_FOLDS, SEED = 5, 42

POOL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("m3_catboost", "m3_catboost"),
    ("m4_relstate", "m4_relstate"),
]


def load(name: str, suffix: str):
    """Load OOF + test [:, 1] (positive-class probability)."""
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1]
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1]
    return oof.astype(np.float64), test.astype(np.float64)


def expand_three_views(probas: np.ndarray) -> np.ndarray:
    """Stack [raw, rank-norm, logit] columnwise. probas: (n, K)."""
    raw = probas
    n, K = probas.shape
    rank = np.zeros_like(raw)
    for k in range(K):
        rank[:, k] = rankdata(raw[:, k]) / n  # rank-normalised in [0, 1]
    logit = np.log(np.clip(raw, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(raw, 1e-9, 1 - 1e-9)))
    return np.hstack([raw, rank, logit])


def run_meta(suffix: str, base_auc: float):
    """Train LR meta on OOFs of `suffix` anchor; return (meta_oof, meta_test, auc, weights)."""
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    Xs_oof, Xs_test, names = [], [], []
    for label, name in POOL:
        oo, te = load(name, suffix)
        if len(oo) != len(y):
            print(f"  SKIP {label}: OOF len {len(oo)} != y len {len(y)}")
            continue
        Xs_oof.append(oo)
        Xs_test.append(te)
        names.append(label)
    P_oof = np.column_stack(Xs_oof)   # (n, K)
    P_test = np.column_stack(Xs_test)
    K = P_oof.shape[1]
    print(f"  pool ({suffix}): {names}  K={K}")

    F_oof = expand_three_views(P_oof)
    F_test = expand_three_views(P_test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    weights_per_fold = []
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
        weights_per_fold.append(lr.coef_.ravel())
    auc = float(roc_auc_score(y, meta_oof))
    delta = (auc - base_auc) * 1e4

    # Final fit on all OOF -> apply to test
    lr_final = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_final.fit(F_oof, y)
    meta_test = lr_final.predict_proba(F_test)[:, 1]

    coef = lr_final.coef_.ravel()
    return meta_oof, meta_test, auc, delta, names, coef


def main():
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    print("=== M5 LR meta — Strat OOFs ===")
    oof_s, test_s, auc_s, d_s, names, coef_s = run_meta("strat", BASE_S)
    print(f"meta_strat OOF: {auc_s:.5f}  Δ={d_s:+.1f}bp")

    print("\n=== M5 LR meta — GroupKF OOFs ===")
    oof_g, test_g, auc_g, d_g, _, coef_g = run_meta("groupkf", BASE_G)
    print(f"meta_groupkf OOF: {auc_g:.5f}  Δ={d_g:+.1f}bp")

    # Also report top-bp single model in pool for comparison
    print("\nPool single-model Strat OOFs:")
    for label, name in POOL:
        try:
            oo, _ = load(name, "strat")
            a = roc_auc_score(y, oo)
            print(f"  {label}: {a:.5f}  Δ={(a - BASE_S)*1e4:+.1f}bp")
        except FileNotFoundError:
            pass

    # Save
    np.save(ART / "oof_m5_lr_meta_strat.npy",
            np.column_stack([1 - oof_s, oof_s]))
    np.save(ART / "test_m5_lr_meta_strat.npy",
            np.column_stack([1 - test_s, test_s]))
    np.save(ART / "oof_m5_lr_meta_groupkf.npy",
            np.column_stack([1 - oof_g, oof_g]))
    np.save(ART / "test_m5_lr_meta_groupkf.npy",
            np.column_stack([1 - test_g, test_g]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv("submissions/submission_m5_lr_meta.csv", index=False)

    res = dict(
        oof_strat=auc_s, oof_groupkf=auc_g,
        delta_strat_bp=d_s, delta_groupkf_bp=d_g,
        pool_models=names,
        coef_strat=coef_s.tolist(), coef_groupkf=coef_g.tolist(),
        feature_names=([f"raw_{n}" for n in names] +
                       [f"rank_{n}" for n in names] +
                       [f"logit_{n}" for n in names]),
    )
    (ART / "m5_lr_meta_results.json").write_text(json.dumps(res, indent=2))

    # Audit
    bp_s = lambda x, base: f"{(x - base) * 1e4:+.1f}bp"
    body = (
        f"# M5 — LR meta-stacker [raw, rank, logit] (2026-05-04)\n\n"
        f"Pool: {names}  (K={len(names)})\n\n"
        f"Recipe (analyticaobscura, cross-comp research Source 1 #2): for each base\n"
        f"model OOF, expand into 3 representations (raw probability, rank-normalised,\n"
        f"logit). Stack horizontally → 3K features. LR(C=1.0) 5-fold over OOFs to\n"
        f"compute meta-OOF AUC. Final fit on full OOFs → meta-test.\n\n"
        f"## Two-anchor results\n\n"
        f"| anchor | meta OOF | base | Δ (bp) |\n|---|---:|---:|---:|\n"
        f"| Strat | **{auc_s:.5f}** | {BASE_S} | {d_s:+.1f} |\n"
        f"| GroupKF | **{auc_g:.5f}** | {BASE_G} | {d_g:+.1f} |\n\n"
        f"## LR coefficients (final fit on all Strat OOFs)\n\n"
        f"| feature | coef |\n|---|---:|\n"
        + "".join(
            f"| {nm} | {c:+.3f} |\n"
            for nm, c in zip(
                [f"raw_{n}" for n in names] +
                [f"rank_{n}" for n in names] +
                [f"logit_{n}" for n in names], coef_s)
        )
    )
    out = Path("audit/2026-05-04-m5-lr-meta.md")
    out.write_text(body)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
