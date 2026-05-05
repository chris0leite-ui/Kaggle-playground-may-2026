"""M5d — LR meta with M5c pool + CatBoost variant winners.

Adds the new CatBoost variant OOFs (cb_<variant>_strat / cb_<variant>_groupkf)
to the M5c pool and refits the LR meta-stacker.

Variants to add are passed via --variants flag, comma-separated, e.g.:
    --variants year-cat,slow-wide
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
BASE_S, BASE_G = 0.94075, 0.92059
M5C_S, M5C_G = 0.95000, 0.92963
SEED, N_FOLDS = 42, 5

M5C_POOL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("m3_catboost", "m3_catboost"),
    ("m4_relstate", "m4_relstate"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
]


def load(name, suffix):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) / (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rank, logit])


def run(suffix, base_auc, pool):
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool:
        try:
            oo, te = load(name, suffix)
            Xs_oof.append(oo); Xs_test.append(te); names.append(label)
            print(f"  + {label} ({suffix}): OOF AUC {roc_auc_score(y, oo):.5f}")
        except FileNotFoundError:
            print(f"  SKIP {label} ({suffix}): file not found")
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    print(f"  M5d OOF: {auc:.5f}  Δbase={(auc - base_auc) * 1e4:+.1f}bp  K={len(names)}")
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc, names, lr_full.coef_.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True,
                    help="comma-separated cb variant names, e.g. year-cat,slow-wide")
    ap.add_argument("--tag", default="m5d", help="output tag")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    pool = list(M5C_POOL) + [(f"cb_{v}", f"cb_{v}") for v in variants]
    print(f"Pool ({len(pool)}): {[p[0] for p in pool]}")

    print(f"\n=== {args.tag} — Strat ===")
    oof_s, test_s, auc_s, names_s, coef_s = run("strat", BASE_S, pool)
    print(f"\n=== {args.tag} — GroupKF ===")
    oof_g, test_g, auc_g, _, coef_g = run("groupkf", BASE_G, pool)

    np.save(ART / f"oof_{args.tag}_strat.npy", np.column_stack([1 - oof_s, oof_s]))
    np.save(ART / f"test_{args.tag}_strat.npy", np.column_stack([1 - test_s, test_s]))
    np.save(ART / f"oof_{args.tag}_groupkf.npy",
            np.column_stack([1 - oof_g, oof_g]))
    np.save(ART / f"test_{args.tag}_groupkf.npy",
            np.column_stack([1 - test_g, test_g]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv(f"submissions/submission_{args.tag}_lr_meta.csv", index=False)

    feat_names = ([f"raw_{n}" for n in names_s] + [f"rank_{n}" for n in names_s]
                  + [f"logit_{n}" for n in names_s])
    res = dict(strat=dict(oof=auc_s, delta_base_bp=(auc_s - BASE_S) * 1e4,
                          delta_m5c_bp=(auc_s - M5C_S) * 1e4,
                          coefs={n: float(c) for n, c in zip(feat_names, coef_s)}),
               groupkf=dict(oof=auc_g, delta_base_bp=(auc_g - BASE_G) * 1e4,
                            delta_m5c_bp=(auc_g - M5C_G) * 1e4),
               pool=names_s, variants=variants)
    (ART / f"{args.tag}_lr_meta_results.json").write_text(json.dumps(res, indent=2))

    body = (
        f"# {args.tag} — LR meta with M5c + CB winners (2026-05-04)\n\n"
        f"Pool ({len(names_s)}): {names_s}\n\n"
        f"New CB variants added: {variants}\n\n"
        f"## Two-anchor results vs M5c\n\n"
        f"| anchor | {args.tag} | M5c | Δ vs M5c | Δ vs base |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| Strat | **{auc_s:.5f}** | {M5C_S:.5f} | "
        f"{(auc_s - M5C_S) * 1e4:+.1f}bp | {(auc_s - BASE_S) * 1e4:+.1f}bp |\n"
        f"| GroupKF | **{auc_g:.5f}** | {M5C_G:.5f} | "
        f"{(auc_g - M5C_G) * 1e4:+.1f}bp | {(auc_g - BASE_G) * 1e4:+.1f}bp |\n\n"
        f"Submission: submissions/submission_{args.tag}_lr_meta.csv (held).\n"
    )
    Path(f"audit/2026-05-04-{args.tag}-lr-meta.md").write_text(body)
    print("audit written")


if __name__ == "__main__":
    main()
