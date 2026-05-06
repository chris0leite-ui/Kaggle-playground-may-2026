"""d16 Phase 5 — Path B with r̂ / log p_orig as cohort axis.

Depends on Phase 2 (r̂) and Phase 3 (log p_orig) outputs. Per
`path-b-amp-only-fires-on-meta-arch-not-base-add`, cohort-axis redesign
is the only place 6-11.6× LB amp can fire.

Three Path-B variants:
  P5.1  segment by r̂(x) quintile (5 segments)        ; tau ∈ {5k, 20k, 100k}
  P5.2  segment by log p_orig(x) quintile (5 segments); tau ∈ {5k, 20k, 100k}
  P5.3  segment by Compound × r̂_q5 (5×5 = 25 cross)  ; tau = 20k

Implementation: emp-Bayes hierarchical LR meta over K=21 pool with per-segment
shrinkage to global LR (Path B). Same machinery as `d13_path_b_*.py`.

OOF: 5-fold StratifiedKFold on synth_train (preserves R1 Strat anchor).

Outputs:
  scripts/artifacts/oof_d16_path_b_rhat_q5_tauX_strat.npy / test_*  (3 tau values)
  scripts/artifacts/oof_d16_path_b_logp_q5_tauX_strat.npy / test_*  (3 tau values)
  scripts/artifacts/oof_d16_path_b_compXrhat_q5_tau20000_strat.npy / test_*
  scripts/artifacts/d16_phase5_summary.json — OOF AUC table + ρ vs PRIMARY
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

# K=21 pool — must match d15b path-b PRIMARY pool composition
# Read pool list from existing K=22 PRIMARY artifact construction. We reconstruct
# from the documented pool: K=21 = list compiled from CLAUDE.md ladder.
K21_NAMES = [
    "baseline_two_anchor", "e3_hgbc", "e5_optuna_lgbm",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year_cat", "cb_lossguide",
    "cb_slow_wide_bag", "m5h_l1coef_top13", "m5q_realmlp_added",
    "d3a_te_unified", "d6_rule_compound_stint", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9c_FM", "d9f_FM_A_driver", "d9f_FM_B_race",
    "d9h_FM_aug12", "d9i_FM_A_aug", "d9i_FM_B_aug", "d13a_FM_A_53",
]


def load_pool():
    """Try to load K=21 pool. If individual base files missing, fall back to
    using the PRIMARY OOF directly (single-base meta probe)."""
    oofs, tests, names_loaded = [], [], []
    for name in K21_NAMES:
        oof_path = ART / f"oof_{name}_strat.npy"
        test_path = ART / f"test_{name}_strat.npy"
        if oof_path.exists() and test_path.exists():
            o = np.load(oof_path)
            t = np.load(test_path)
            if o.ndim == 2:
                o = o[:, 1]
            if t.ndim == 2:
                t = t[:, 1]
            oofs.append(o)
            tests.append(t)
            names_loaded.append(name)
    return oofs, tests, names_loaded


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def expand_features(O):
    """Build [raw, rank, logit] feature stack from a list of OOF arrays."""
    raw = np.column_stack(O)
    rank = np.column_stack([np.argsort(np.argsort(o)) / len(o) for o in O])
    lgt = np.column_stack([_logit(o) for o in O])
    return np.hstack([raw, rank, lgt])


def hierarchical_lr(X_tr, y_tr, X_va, segments_tr, segments_va, tau=20000.0):
    """Empirical-Bayes hierarchical LR: per-segment LR shrunk to global LR.
    α(n) = n / (n + tau).
    """
    # Global LR
    g = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    g.fit(X_tr, y_tr)
    g_coef = np.concatenate([g.intercept_, g.coef_[0]])

    # Per-segment LR + shrinkage
    pred_va = g.predict_proba(X_va)[:, 1].copy()
    seg_set = sorted(set(np.unique(segments_tr)) | set(np.unique(segments_va)))
    for s in seg_set:
        mask_tr = segments_tr == s
        mask_va = segments_va == s
        if mask_tr.sum() < 50:
            continue
        lr_s = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr_s.fit(X_tr[mask_tr], y_tr[mask_tr])
        lr_s_coef = np.concatenate([lr_s.intercept_, lr_s.coef_[0]])
        n = mask_tr.sum()
        alpha = n / (n + tau)
        shrunk = alpha * lr_s_coef + (1 - alpha) * g_coef
        intercept = shrunk[0]
        coef = shrunk[1:]
        if mask_va.sum() > 0:
            z = X_va[mask_va] @ coef + intercept
            pred_va[mask_va] = 1.0 / (1.0 + np.exp(-z))
    return pred_va


def run_path_b(oofs, tests, segments_tr, segments_te, y, tau, name):
    X_tr = expand_features(oofs)
    X_te = expand_features(tests)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    pred_te = np.zeros(len(X_te))
    for fi, (tri, vai) in enumerate(skf.split(X_tr, y)):
        oof[vai] = hierarchical_lr(
            X_tr[tri], y[tri], X_tr[vai], segments_tr[tri], segments_tr[vai], tau=tau
        )
        # test: use full training set; segment_te assigns to segments using its own scheme
        pred_te_fold = hierarchical_lr(
            X_tr[tri], y[tri], X_te, segments_tr[tri], segments_te, tau=tau
        )
        pred_te += pred_te_fold / N_FOLDS
    auc = roc_auc_score(y, oof)
    np.save(ART / f"oof_d16_path_b_{name}_strat.npy", oof)
    np.save(ART / f"test_d16_path_b_{name}_strat.npy", pred_te)
    return float(auc), oof, pred_te


def quantile_segments(values, k=5, edges=None):
    if edges is None:
        edges = np.unique(np.quantile(values, np.linspace(0, 1, k + 1)))
    seg = np.searchsorted(edges[1:-1], values, side="right")
    return seg, edges


def main():
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}")

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].astype(int).values

    step("loading K=21 pool")
    oofs, tests, names_loaded = load_pool()
    step(f"  loaded {len(oofs)}/{len(K21_NAMES)} bases: {names_loaded}")
    if len(oofs) < 10:
        # Fallback: use ALL available oof_*_strat.npy files matching common names
        # and compose a working pool.
        step("  FALLBACK: scanning all oof_*_strat.npy")
        for p in sorted(ART.glob("oof_*_strat.npy")):
            name = p.stem.replace("oof_", "").replace("_strat", "")
            test_path = ART / f"test_{name}_strat.npy"
            if not test_path.exists():
                continue
            if name in names_loaded or "PRIMARY" in name or "K22" in name or "K23" in name or "K24" in name:
                continue
            if "path_b" in name or "blend" in name or "_meta" in name or "alpha_calib" in name:
                continue
            try:
                o = np.load(p)
                t = np.load(test_path)
                if o.ndim == 2:
                    o = o[:, 1]
                if t.ndim == 2:
                    t = t[:, 1]
                if o.shape[0] != len(y) or t.shape[0] != len(te):
                    continue
                # standalone OOF AUC sanity
                a = roc_auc_score(y, o)
                if a < 0.85:
                    continue
                oofs.append(o)
                tests.append(t)
                names_loaded.append(name)
                if len(oofs) >= 21:
                    break
            except Exception:
                continue
        step(f"  after fallback: {len(oofs)} bases")

    if len(oofs) < 8:
        raise SystemExit(f"Pool too small ({len(oofs)}); aborting Phase 5.")

    # Load r̂ from Phase 2
    rhat_tr = np.load(ART / "d16_rhat_synth_train.npy") if (ART / "d16_rhat_synth_train.npy").exists() else None
    rhat_te = np.load(ART / "d16_rhat_synth_test.npy") if (ART / "d16_rhat_synth_test.npy").exists() else None
    logp_tr = np.load(ART / "d16_logp_orig_gmm_synth_train.npy") if (ART / "d16_logp_orig_gmm_synth_train.npy").exists() else None
    logp_te = np.load(ART / "d16_logp_orig_gmm_synth_test.npy") if (ART / "d16_logp_orig_gmm_synth_test.npy").exists() else None

    if rhat_tr is None or rhat_te is None:
        step("  WARNING: r̂ not yet available; cannot run P5.1 or P5.3")
    if logp_tr is None or logp_te is None:
        step("  WARNING: log p_orig not yet available; cannot run P5.2")

    summary = dict(pool_names=names_loaded, pool_size=len(oofs),
                    runtime_s_load=time.time() - t0, results=[])

    primary_oof = np.load(ART / "oof_PRIMARY_K22_strat.npy")
    if primary_oof.ndim == 2:
        primary_oof = primary_oof[:, 1]
    primary_test = np.load(ART / "test_PRIMARY_K22_strat.npy")
    if primary_test.ndim == 2:
        primary_test = primary_test[:, 1]
    primary_auc = roc_auc_score(y, primary_oof)
    step(f"  PRIMARY OOF AUC reference: {primary_auc:.5f}")

    # P5.1 r̂ quintile
    if rhat_tr is not None:
        seg_tr, edges = quantile_segments(np.log1p(rhat_tr), k=5)
        seg_te, _ = quantile_segments(np.log1p(rhat_te), k=5, edges=edges)
        for tau in [5000, 20000, 100000]:
            step(f"P5.1 r̂_q5 tau={tau}")
            auc, oof, pte = run_path_b(oofs, tests, seg_tr, seg_te, y, tau,
                                        f"rhat_q5_tau{tau}")
            rho = float(np.corrcoef(pte, primary_test)[0, 1])
            step(f"  OOF AUC {auc:.5f}  Δ vs PRIMARY {(auc - primary_auc) * 1e4:+.2f} bp  ρ {rho:.5f}")
            summary["results"].append(dict(name=f"rhat_q5_tau{tau}", oof_auc=float(auc),
                                            delta_bp=float((auc - primary_auc) * 1e4),
                                            rho_test=rho))

    # P5.2 log p_orig quintile
    if logp_tr is not None:
        seg_tr2, edges2 = quantile_segments(logp_tr, k=5)
        seg_te2, _ = quantile_segments(logp_te, k=5, edges=edges2)
        for tau in [5000, 20000, 100000]:
            step(f"P5.2 logp_q5 tau={tau}")
            auc, oof, pte = run_path_b(oofs, tests, seg_tr2, seg_te2, y, tau,
                                        f"logp_q5_tau{tau}")
            rho = float(np.corrcoef(pte, primary_test)[0, 1])
            step(f"  OOF AUC {auc:.5f}  Δ vs PRIMARY {(auc - primary_auc) * 1e4:+.2f} bp  ρ {rho:.5f}")
            summary["results"].append(dict(name=f"logp_q5_tau{tau}", oof_auc=float(auc),
                                            delta_bp=float((auc - primary_auc) * 1e4),
                                            rho_test=rho))

    # P5.3 Compound × r̂_q5 cross
    if rhat_tr is not None:
        compound_tr = tr["Compound"].astype(str).values
        compound_te = te["Compound"].astype(str).values
        seg_tr_r, edges_r = quantile_segments(np.log1p(rhat_tr), k=5)
        seg_te_r, _ = quantile_segments(np.log1p(rhat_te), k=5, edges=edges_r)
        seg_cr_tr = np.array([f"{c}|{r}" for c, r in zip(compound_tr, seg_tr_r)])
        seg_cr_te = np.array([f"{c}|{r}" for c, r in zip(compound_te, seg_te_r)])
        # encode strings to ints for hierarchical_lr
        levels = sorted(set(seg_cr_tr) | set(seg_cr_te))
        l2i = {l: i for i, l in enumerate(levels)}
        seg_cr_tr_i = np.array([l2i[s] for s in seg_cr_tr])
        seg_cr_te_i = np.array([l2i[s] for s in seg_cr_te])
        for tau in [20000]:
            step(f"P5.3 Compound × r̂_q5 ({len(levels)} seg) tau={tau}")
            auc, oof, pte = run_path_b(oofs, tests, seg_cr_tr_i, seg_cr_te_i, y, tau,
                                        f"compXrhat_q5_tau{tau}")
            rho = float(np.corrcoef(pte, primary_test)[0, 1])
            step(f"  OOF AUC {auc:.5f}  Δ vs PRIMARY {(auc - primary_auc) * 1e4:+.2f} bp  ρ {rho:.5f}")
            summary["results"].append(dict(name=f"compXrhat_q5_tau{tau}",
                                            n_segments=len(levels),
                                            oof_auc=float(auc),
                                            delta_bp=float((auc - primary_auc) * 1e4),
                                            rho_test=rho))

    summary["runtime_s_total"] = time.time() - t0
    summary["primary_oof_auc"] = float(primary_auc)
    with open(ART / "d16_phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step("DONE")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
