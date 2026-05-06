"""scripts/probe.py — Ablation harness for cheap probes.

Embeds the experimentation culture: BOTE before code, structured gate
report after artifacts exist. Designed for "many small things, low cost".

Two callables for the experimentation loop:

    bote(name, family, cost_min, *, predicted_std_oof_lift_bp=None,
         override_pessimistic=None, override_optimistic=None,
         prob_useful_override=None) -> dict
        Back-of-envelope EV. Run BEFORE writing any code.
        Returns expected_lb_bp + verdict (PURSUE / DEFER / SKIP).

    gate(name, oof_path, test_path, *, primary_oof_path=None,
         primary_test_path=None, train_csv="data/train.csv",
         min_meta=False) -> dict
        Standardized gate report after artifacts exist. Returns
        standalone_oof, rho_vs_primary, predicted_lb_delta_bp,
        flip_ratio (G3), and a verdict band.

CLI:
    python scripts/probe.py bote NAME --family FAMILY --cost_min N \\
        [--std_oof_lift_bp X] [--prob_useful U]

    python scripts/probe.py gate NAME --oof PATH --test PATH \\
        [--primary-oof PATH] [--primary-test PATH] [--min-meta]

The PRIMARY defaults track current state (d13e Compound×Stint τ=20k);
when PRIMARY changes, update the constants below or pass explicit paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

# ---- PRIMARY defaults (update when PRIMARY changes) -----------------
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_LB = 0.95049

# ---- BOTE family priors ---------------------------------------------
# (P(useful signal at meta-gate), (pessimistic, median, optimistic) bp)
# Calibrated against empirical hit rate (4/24 ≈ 17%) and observed
# OOF→LB amplification ratios from CLAUDE.md ladder.
FAMILY_PRIORS = {
    "new_model_class":          (0.40, (3.0,  8.0, 15.0)),
    "meta_arch_redesign":       (0.30, (1.0,  4.0,  8.0)),
    "per_field_fm_isolated":    (0.25, (0.5,  2.0,  5.0)),
    "code_fix_calibration":     (0.30, (0.3,  1.0,  3.0)),
    "ensembling_blend":         (0.20, (0.0,  0.5,  2.0)),
    "tuning_existing":          (0.20, (0.0,  0.5,  1.5)),
    "external_data_aggregate":  (0.20, (0.0,  1.0,  4.0)),
    "pseudo_label_cheap":       (0.15, (0.0,  1.0,  3.0)),
    "single_base_fe_addition":  (0.05, (0.0,  0.5,  2.0)),  # 4-of-4 NULL
    "pool_addition_redundant":  (0.10, (0.0,  0.3,  1.0)),
    "process_or_infrastructure": (1.00, (0.0,  0.0,  0.0)),  # not bp; utility
}

# ---- ρ → predicted-LB-delta band ------------------------------------
def predicted_lb_delta_bp(d_oof_bp: float, rho: float) -> float:
    """Conservative band based on prior-comp empirics. Rho near 1 ties."""
    if rho >= 0.99996:
        return d_oof_bp           # tied; LB Δ ≈ OOF Δ
    if rho >= 0.999:
        return d_oof_bp - 0.5
    if rho >= 0.995:
        return d_oof_bp - 1.5
    if rho >= 0.99:
        return d_oof_bp - 3.0
    return d_oof_bp - 5.0


def bote(name: str,
         family: str,
         cost_min: float,
         *,
         predicted_std_oof_lift_bp: float | None = None,
         override_pessimistic: float | None = None,
         override_optimistic: float | None = None,
         prob_useful_override: float | None = None,
         note: str = "") -> dict:
    """Back-of-envelope EV calc. Run BEFORE any compute on a candidate.

    Args:
      name: candidate name.
      family: must match a FAMILY_PRIORS key.
      cost_min: estimated CPU minutes (1-fold smoke + gate, not full sweep).
      predicted_std_oof_lift_bp: if you have a prediction for standalone
        OOF lift, pass it here; otherwise the family median is used.
      override_*: replace the default (P, M, O) with custom values.
      prob_useful_override: replace family prior P with custom.
      note: free-text rationale to print.
    """
    if family not in FAMILY_PRIORS:
        raise KeyError(f"unknown family {family}; choices: "
                       f"{sorted(FAMILY_PRIORS)}")
    p, (pess_def, med_def, opt_def) = FAMILY_PRIORS[family]
    pess = override_pessimistic if override_pessimistic is not None else pess_def
    med  = predicted_std_oof_lift_bp if predicted_std_oof_lift_bp is not None else med_def
    opt  = override_optimistic if override_optimistic is not None else opt_def
    p    = prob_useful_override if prob_useful_override is not None else p

    expected_bp = p * med
    cost_efficiency = expected_bp / max(cost_min, 1.0)

    if cost_efficiency >= 0.05:
        verdict = "PURSUE"
    elif cost_efficiency >= 0.01:
        verdict = "DEFER"
    else:
        verdict = "SKIP"

    res = dict(
        name=name, family=family,
        prob_useful=float(p),
        bp_pessimistic=float(pess),
        bp_median=float(med),
        bp_optimistic=float(opt),
        cost_min=float(cost_min),
        expected_lb_bp=float(expected_bp),
        cost_efficiency_bp_per_min=float(cost_efficiency),
        verdict=verdict,
    )
    print(f"\n=== BOTE: {name} ===")
    if note:
        print(f"  note: {note}")
    print(f"  family: {family}  P(useful): {p:.2f}")
    print(f"  bp band (P/M/O): {pess:.1f} / {med:.1f} / {opt:.1f}")
    print(f"  cost: {cost_min:.0f} min  →  expected LB: {expected_bp:+.2f} bp")
    print(f"  cost-efficiency: {cost_efficiency:.3f} bp/min")
    print(f"  verdict: {verdict}")
    return res


def _load_pred(path: Path | str) -> np.ndarray:
    """Load (n, 2) OOF/test artifact and return positive-class column."""
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1].astype(np.float64)
    return arr.astype(np.float64).ravel()


def _load_y(train_csv: str) -> np.ndarray | None:
    """Load y from train_csv if it exists; else try common npy fallbacks;
    else return None (gate degrades gracefully to ρ-only)."""
    if Path(train_csv).exists():
        return pd.read_csv(train_csv, usecols=[TARGET])[TARGET].astype(int).values
    for fallback in [ART / "y_train.npy", Path("data/y_train.npy")]:
        if fallback.exists():
            return np.load(fallback).astype(int)
    return None


def gate(name: str,
         oof_path: Path | str,
         test_path: Path | str,
         *,
         primary_oof_path: Path | str = PRIMARY_OOF,
         primary_test_path: Path | str = PRIMARY_TEST,
         train_csv: str = "data/train.csv",
         min_meta: bool = False,
         min_meta_pool_oofs: list[Path | str] | None = None,
         min_meta_pool_tests: list[Path | str] | None = None,
         ) -> dict:
    """Standardized gate report.

    Computes (when y is available):
      - standalone_oof = roc_auc_score(y, candidate_oof)
      - delta_oof_vs_primary_bp
      - if min_meta: K=K_pool+1 LR meta OOF lift over PRIMARY OOF

    Always computes:
      - rho_vs_primary = Spearman(candidate_test, primary_test)
      - predicted_lb_delta_bp (band-based on rho)
      - flip_ratio (G3 rare-class flip ratio at top-1% threshold)

    Returns a dict; also prints a structured summary.
    """
    y = _load_y(train_csv)
    cand_oof = _load_pred(oof_path)
    cand_test = _load_pred(test_path)
    primary_oof = _load_pred(primary_oof_path)
    primary_test = _load_pred(primary_test_path)

    if y is not None:
        auc_cand = float(roc_auc_score(y, cand_oof))
        auc_primary = float(roc_auc_score(y, primary_oof))
        d_oof_bp = (auc_cand - auc_primary) * 1e4
    else:
        auc_cand = None
        auc_primary = None
        d_oof_bp = 0.0   # unknown; default for predicted_lb_delta calc
    rho, _ = spearmanr(cand_test, primary_test)
    rho = float(rho)
    pred_lb = predicted_lb_delta_bp(d_oof_bp, rho)

    # G3 rare-class flip ratio at top-1% threshold.
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    cand_pos = cand_test >= rare_thr
    flips_to_neg = int(np.sum(primary_pos & ~cand_pos))
    flips_to_pos = int(np.sum(~primary_pos & cand_pos))
    if max(flips_to_pos, flips_to_neg) > 0:
        flip_ratio = float(min(flips_to_pos, flips_to_neg) /
                           max(flips_to_pos, flips_to_neg))
    else:
        flip_ratio = 1.0

    # Verdict (degrades to ρ-only if y missing)
    if rho >= 0.999:
        verdict = "TIE_EXPECTED"
    elif y is None:
        verdict = "RHO_ONLY"
    elif d_oof_bp >= 0.3 and rho >= 0.99:
        verdict = "PASS"
    elif d_oof_bp >= 0:
        verdict = "WEAK_PASS"
    else:
        verdict = "FAIL"

    res = dict(
        name=name,
        n_train=int(len(y)) if y is not None else None,
        standalone_oof=auc_cand,
        primary_oof=auc_primary,
        delta_oof_bp=float(d_oof_bp),
        rho_vs_primary=rho,
        predicted_lb_delta_bp=float(pred_lb),
        rare_thr=rare_thr,
        flips_primary_to_neg=flips_to_neg,
        flips_primary_to_pos=flips_to_pos,
        g3_flip_ratio=flip_ratio,
        verdict=verdict,
        y_available=y is not None,
    )

    print(f"\n=== GATE: {name} ===")
    if auc_cand is not None:
        print(f"  candidate OOF: {auc_cand:.5f}  vs PRIMARY {auc_primary:.5f}  "
              f"Δ {d_oof_bp:+.2f}bp")
    else:
        print(f"  [y not loadable — skipping standalone OOF; ρ-only mode]")
    print(f"  ρ vs PRIMARY:  {rho:.6f}")
    print(f"  pred LB Δ:     {pred_lb:+.2f}bp"
          f"{' (ρ-only; OOF Δ unknown)' if auc_cand is None else ''}")
    print(f"  G3 flip ratio (top-1%): {flip_ratio:.3f}  "
          f"(+→−: {flips_to_neg}, −→+: {flips_to_pos})")
    print(f"  verdict: {verdict}")

    if min_meta:
        # Optional K=K_pool + 1 min-meta gate
        if min_meta_pool_oofs is None or min_meta_pool_tests is None:
            print("  [min_meta requested but pool paths not provided; skipping]")
        else:
            mm = _min_meta_gate(y, cand_oof, cand_test,
                                min_meta_pool_oofs, min_meta_pool_tests,
                                primary_oof, primary_test)
            res.update(mm)
            print(f"  K={len(min_meta_pool_oofs)+1} min-meta OOF: "
                  f"{mm['min_meta_oof']:.5f}  Δ vs K=K_pool {mm['min_meta_delta_bp']:+.2f}bp")

    return res


def _expand(P: np.ndarray) -> np.ndarray:
    """Stack [raw, rank, logit] features (matches K=21 LR meta convention)."""
    from scipy.stats import rankdata
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    P_clip = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(P_clip / (1 - P_clip))
    return np.hstack([P, rk, logit])


def _min_meta_gate(y, cand_oof, cand_test, pool_oofs_paths, pool_tests_paths,
                   primary_oof, primary_test) -> dict:
    pool_oofs = [_load_pred(p) for p in pool_oofs_paths]
    pool_tests = [_load_pred(p) for p in pool_tests_paths]
    P_oof_base = np.column_stack(pool_oofs)
    P_test_base = np.column_stack(pool_tests)
    P_oof_with = np.column_stack(pool_oofs + [cand_oof])
    P_test_with = np.column_stack(pool_tests + [cand_test])
    F_oof_base = _expand(P_oof_base)
    F_oof_with = _expand(P_oof_with)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    def _fit(F):
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, va in splits:
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            oof[va] = lr.predict_proba(F[va])[:, 1]
        return oof

    oof_base = _fit(F_oof_base)
    oof_with = _fit(F_oof_with)
    auc_base = float(roc_auc_score(y, oof_base))
    auc_with = float(roc_auc_score(y, oof_with))
    delta_bp = (auc_with - auc_base) * 1e4
    return dict(
        min_meta_oof=auc_with,
        min_meta_oof_base=auc_base,
        min_meta_delta_bp=float(delta_bp),
    )


# ---- CLI -------------------------------------------------------------
def _cli_bote(args):
    bote(
        name=args.name,
        family=args.family,
        cost_min=args.cost_min,
        predicted_std_oof_lift_bp=args.std_oof_lift_bp,
        override_pessimistic=args.bp_pessimistic,
        override_optimistic=args.bp_optimistic,
        prob_useful_override=args.prob_useful,
        note=args.note or "",
    )


def _cli_gate(args):
    res = gate(
        name=args.name,
        oof_path=args.oof,
        test_path=args.test,
        primary_oof_path=args.primary_oof or PRIMARY_OOF,
        primary_test_path=args.primary_test or PRIMARY_TEST,
        train_csv=args.train_csv,
        min_meta=args.min_meta,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2))
        print(f"\n→ {args.json_out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bote", help="back-of-envelope EV calc")
    b.add_argument("name")
    b.add_argument("--family", required=True, choices=sorted(FAMILY_PRIORS))
    b.add_argument("--cost_min", type=float, required=True)
    b.add_argument("--std_oof_lift_bp", type=float, default=None)
    b.add_argument("--bp_pessimistic", type=float, default=None)
    b.add_argument("--bp_optimistic", type=float, default=None)
    b.add_argument("--prob_useful", type=float, default=None)
    b.add_argument("--note", type=str, default=None)
    b.set_defaults(func=_cli_bote)

    g = sub.add_parser("gate", help="standardized gate report on saved OOF/test")
    g.add_argument("name")
    g.add_argument("--oof", required=True)
    g.add_argument("--test", required=True)
    g.add_argument("--primary-oof", default=None)
    g.add_argument("--primary-test", default=None)
    g.add_argument("--train-csv", default="data/train.csv")
    g.add_argument("--min-meta", action="store_true")
    g.add_argument("--json-out", default=None)
    g.set_defaults(func=_cli_gate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
