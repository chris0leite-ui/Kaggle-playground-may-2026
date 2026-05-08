"""E2 — L1-regularized meta-stacker.

Tests if M5's −4.4bp OOF→LB overshoot was driven by stacker
over-flexibility on the 15-feature [raw, rank, logit] expansion.
Sweep L1 strength; pick best-OOF; report sparsity (which meta
features L1 zeros out).
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
TARGET = "PitNextLap"
BASE_S, BASE_G = 0.94075, 0.92059
SEED, N_FOLDS = 42, 5
POOL = ["baseline_two_anchor", "d2a_te", "m2_xgb", "m3_catboost", "m4_relstate"]
LABELS = ["baseline", "d2a_te", "m2_xgb", "m3_catboost", "m4_relstate"]


def load_pool(suffix):
    oofs, tests = [], []
    for name in POOL:
        oofs.append(np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64))
        tests.append(np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64))
    return np.column_stack(oofs), np.column_stack(tests)


def expand(P):
    n, K = P.shape
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) / (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rank, logit])


def cv_meta_auc(F, y, C, penalty):
    """5-fold OOF AUC for an LR meta with given C and penalty."""
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, va in skf.split(np.zeros(len(y)), y):
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        lr = LogisticRegression(C=C, max_iter=2000, solver=solver, penalty=penalty)
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def run(suffix, base_auc):
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    P_oof, P_test = load_pool(suffix)
    F_oof = expand(P_oof)
    F_test = expand(P_test)
    feat_names = [f"raw_{n}" for n in LABELS] + [f"rank_{n}" for n in LABELS] + [f"logit_{n}" for n in LABELS]

    print(f"\n=== {suffix} (15 meta features) ===")
    results = []
    # Baseline: L2, C=1.0 (the M5 default)
    auc, _ = cv_meta_auc(F_oof, y, C=1.0, penalty="l2")
    print(f"  L2 C=1.0   (M5 baseline):  OOF={auc:.5f}  Δ={(auc-base_auc)*1e4:+.1f}bp")
    results.append(("l2", 1.0, auc))

    # L1 sweep
    for C in [10.0, 1.0, 0.3, 0.1, 0.03, 0.01]:
        auc, _ = cv_meta_auc(F_oof, y, C=C, penalty="l1")
        print(f"  L1 C={C:<6} OOF={auc:.5f}  Δ={(auc-base_auc)*1e4:+.1f}bp")
        results.append(("l1", C, auc))

    # Best
    results.sort(key=lambda r: -r[2])
    pen, C_best, auc_best = results[0]
    print(f"  best: {pen} C={C_best} → OOF {auc_best:.5f}")

    # Refit best on full OOF, get coefs + sparsity
    solver = "liblinear" if pen == "l1" else "lbfgs"
    lr_full = LogisticRegression(C=C_best, max_iter=2000, solver=solver, penalty=pen)
    lr_full.fit(F_oof, y)
    coef = lr_full.coef_.ravel()
    nonzero = int((np.abs(coef) > 1e-8).sum())
    print(f"  best fit: {nonzero}/{len(coef)} non-zero coefs")
    nz_feats = [(feat_names[i], float(coef[i])) for i in range(len(coef)) if abs(coef[i]) > 1e-8]
    nz_feats.sort(key=lambda t: -abs(t[1]))
    for nm, c in nz_feats[:8]:
        print(f"    {nm}: {c:+.3f}")

    # 5-fold OOF + final test predictions
    auc_final, oof_final = cv_meta_auc(F_oof, y, C=C_best, penalty=pen)
    test_final = lr_full.predict_proba(F_test)[:, 1]

    return dict(suffix=suffix, penalty=pen, C=C_best, auc_oof=auc_best,
                delta_bp=(auc_best - base_auc) * 1e4,
                nonzero_coefs=nonzero, total_coefs=len(coef),
                top_nonzero=nz_feats[:10],
                results=results), oof_final, test_final


def main():
    out_s, oof_s, test_s = run("strat", BASE_S)
    out_g, oof_g, test_g = run("groupkf", BASE_G)

    np.save(ART / "oof_e2_l1_meta_strat.npy", np.column_stack([1 - oof_s, oof_s]))
    np.save(ART / "test_e2_l1_meta_strat.npy", np.column_stack([1 - test_s, test_s]))
    np.save(ART / "oof_e2_l1_meta_groupkf.npy", np.column_stack([1 - oof_g, oof_g]))
    np.save(ART / "test_e2_l1_meta_groupkf.npy", np.column_stack([1 - test_g, test_g]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv("submissions/submission_e2_l1_meta.csv", index=False)
    (ART / "e2_l1_meta_results.json").write_text(json.dumps(
        dict(strat=out_s, groupkf=out_g), indent=2))

    body = (
        f"# E2 — L1-regularized meta-stacker (2026-05-04)\n\n"
        f"Tests M5 OOF→LB overshoot (−4.4bp). Sweep penalty/C; report sparsity.\n\n"
        f"## Sweep results\n\n"
        f"| anchor | best penalty | C | OOF | Δ vs base | nonzero/total |\n"
        f"|---|---|---:|---:|---:|---:|\n"
        f"| Strat | {out_s['penalty']} | {out_s['C']} | {out_s['auc_oof']:.5f} | "
        f"{out_s['delta_bp']:+.1f}bp | {out_s['nonzero_coefs']}/{out_s['total_coefs']} |\n"
        f"| GroupKF | {out_g['penalty']} | {out_g['C']} | {out_g['auc_oof']:.5f} | "
        f"{out_g['delta_bp']:+.1f}bp | {out_g['nonzero_coefs']}/{out_g['total_coefs']} |\n\n"
        f"## Strat best — top non-zero coefs\n\n"
        + "".join(f"- `{nm}`: {c:+.3f}\n" for nm, c in out_s['top_nonzero'][:8])
        + "\n## Verdict\n\n"
        f"Compare against M5 OOF (Strat 0.94737, GroupKF 0.92483). If L1 OOF is\n"
        f"within ±2bp of M5 with sparser fit, expected LB→OOF gap should\n"
        f"narrow vs M5's −4.4bp overshoot. Submission held for D3 PI confirm.\n"
    )
    Path("audit/2026-05-04-e2-l1-meta.md").write_text(body)


if __name__ == "__main__":
    main()
