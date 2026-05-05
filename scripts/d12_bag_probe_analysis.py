"""D12 monolithic-bag-probe analysis.

Loads the e3 5-seed bag and CB 3-seed bag, computes:
  - standalone OOF AUC
  - ρ vs PRIMARY test prediction (test_d9f_K21_swap_strat.npy)
  - Δ bp vs PRIMARY OOF
  - Per-segment OOF AUC: by Year, by Stint (1,2,3,4,5+), by Compound
  - Disagreement rate |bag - PRIMARY| > 0.1; cohort dominance
  - Min-meta vs PRIMARY (3-feat LR over {PRIMARY, bag, |Δ|})

Writes audit/2026-05-12-d12-monolithic-bag-probe.md.
"""
from __future__ import annotations

import json
import time
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

# Per CLAUDE.md current state, PRIMARY is d9f K=21 (LB 0.95031). The
# K=21 OOF wasn't persisted; nearest available stack OOF is
# d9c_Sd_K20_swap_FM (LB 0.95029, OOF 0.95070). For test ρ comparison,
# d9f K=21 test array IS available. We use:
#   PRIMARY_OOF  = oof_d9c_Sd_K20_swap_FM_strat.npy  (latest stack OOF)
#   PRIMARY_TEST = test_d9f_K21_swap_strat.npy       (current PRIMARY)
PRIMARY_OOF_PATH  = ART / "oof_d9c_Sd_K20_swap_FM_strat.npy"
PRIMARY_TEST_PATH = ART / "test_d9f_K21_swap_strat.npy"

# NOTE on artifact selection (multi-agent CPU contention dropped 5-seed
# fresh bag plan; see audit note for details):
#   E3 candidate: single-seed e3_hgbc (existing artifact). 1-seed only.
#                 If new 3-seed exists, prefer it.
#   CB candidate: cb_slow-wide-bag (existing 3-seed bag from prior CPU run).
BAG_E3_OOF_NEW   = ART / "oof_d12_e3_5seed_bag_strat.npy"
BAG_E3_TEST_NEW  = ART / "test_d12_e3_5seed_bag_strat.npy"
BAG_E3_OOF_FALL  = ART / "oof_e3_hgbc_strat.npy"
BAG_E3_TEST_FALL = ART / "test_e3_hgbc_strat.npy"
BAG_CB_OOF_NEW   = ART / "oof_d12_cb_5seed_bag_strat.npy"
BAG_CB_TEST_NEW  = ART / "test_d12_cb_5seed_bag_strat.npy"
BAG_CB_OOF_FALL  = ART / "oof_cb_slow-wide-bag_strat.npy"
BAG_CB_TEST_FALL = ART / "test_cb_slow-wide-bag_strat.npy"

def _pick(new, fallback):
    return new if new.exists() else fallback

BAG_E3_OOF  = _pick(BAG_E3_OOF_NEW, BAG_E3_OOF_FALL)
BAG_E3_TEST = _pick(BAG_E3_TEST_NEW, BAG_E3_TEST_FALL)
BAG_CB_OOF  = _pick(BAG_CB_OOF_NEW, BAG_CB_OOF_FALL)
BAG_CB_TEST = _pick(BAG_CB_TEST_NEW, BAG_CB_TEST_FALL)


def expand(P):
    n = len(P)
    from scipy.stats import rankdata
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_min_meta(P_oof, P_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    F_oof = expand(P_oof)
    F_test = expand(P_test)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    return float(roc_auc_score(y, meta_oof))


def per_segment_auc(y, p, df, col, value_or_filter):
    if callable(value_or_filter):
        mask = value_or_filter(df)
        label = value_or_filter.__name__
    else:
        mask = df[col].values == value_or_filter
        label = str(value_or_filter)
    n = int(mask.sum())
    pos = int(y[mask].sum())
    if n == 0 or pos == 0 or pos == n:
        return label, n, pos, float("nan")
    return label, n, pos, float(roc_auc_score(y[mask], p[mask]))


def stint_segments(df):
    s = df["Stint"].clip(upper=5).astype(int).values
    return [(f"S{v}", s == v) for v in [1, 2, 3, 4, 5]]


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    n = len(y)

    # Load probes
    primary_oof = np.load(PRIMARY_OOF_PATH)[:, 1].astype(np.float64)
    primary_test = np.load(PRIMARY_TEST_PATH)[:, 1].astype(np.float64)
    e3_oof  = np.load(BAG_E3_OOF)[:, 1].astype(np.float64)
    e3_test = np.load(BAG_E3_TEST)[:, 1].astype(np.float64)
    cb_oof  = np.load(BAG_CB_OOF)[:, 1].astype(np.float64)
    cb_test = np.load(BAG_CB_TEST)[:, 1].astype(np.float64)
    print(f"E3 source : {BAG_E3_OOF}")
    print(f"CB source : {BAG_CB_OOF}")

    # Avg-of-bags (a third candidate: e3+cb prob-mean)
    avg_oof = 0.5 * (e3_oof + cb_oof)
    avg_test = 0.5 * (e3_test + cb_test)

    # Standalone metrics
    auc_primary = float(roc_auc_score(y, primary_oof))
    auc_e3 = float(roc_auc_score(y, e3_oof))
    auc_cb = float(roc_auc_score(y, cb_oof))
    auc_avg = float(roc_auc_score(y, avg_oof))
    print(f"PRIMARY  OOF AUC: {auc_primary:.5f}")
    print(f"E3 bag   OOF AUC: {auc_e3:.5f}  Δ={(auc_e3-auc_primary)*1e4:+.1f}bp")
    print(f"CB bag   OOF AUC: {auc_cb:.5f}  Δ={(auc_cb-auc_primary)*1e4:+.1f}bp")
    print(f"E3+CB avg OOF AUC: {auc_avg:.5f}  Δ={(auc_avg-auc_primary)*1e4:+.1f}bp")

    # ρ vs PRIMARY test
    rho_e3, _ = spearmanr(e3_test, primary_test)
    rho_cb, _ = spearmanr(cb_test, primary_test)
    rho_avg, _ = spearmanr(avg_test, primary_test)
    print(f"ρ(e3 test, PRIMARY test):  {rho_e3:.5f}")
    print(f"ρ(cb test, PRIMARY test):  {rho_cb:.5f}")
    print(f"ρ(avg test, PRIMARY test): {rho_avg:.5f}")

    # Min-meta vs PRIMARY (3-feat: PRIMARY, bag, |Δ|)
    def mm(bag_o, bag_t):
        diff_o = np.abs(bag_o - primary_oof)
        diff_t = np.abs(bag_t - primary_test)
        P_oof = np.column_stack([primary_oof, bag_o, diff_o])
        P_test = np.column_stack([primary_test, bag_t, diff_t])
        return fit_min_meta(P_oof, P_test, y)

    mm_e3 = mm(e3_oof, e3_test)
    mm_cb = mm(cb_oof, cb_test)
    mm_avg = mm(avg_oof, avg_test)
    print(f"\nMin-meta {{PRIMARY, e3, |Δ|}}: {mm_e3:.5f}  Δ={(mm_e3-auc_primary)*1e4:+.2f}bp")
    print(f"Min-meta {{PRIMARY, cb, |Δ|}}: {mm_cb:.5f}  Δ={(mm_cb-auc_primary)*1e4:+.2f}bp")
    print(f"Min-meta {{PRIMARY, avg,|Δ|}}: {mm_avg:.5f}  Δ={(mm_avg-auc_primary)*1e4:+.2f}bp")

    # Per-segment AUC tables
    print("\n=== Per-segment OOF AUC ===")
    seg_results = {}

    # Year
    print("\n-- by Year --")
    print(f"{'year':<6} {'n':>8} {'pos':>8} {'PRIM':>8} {'E3':>8} {'ΔE3':>7} {'CB':>8} {'ΔCB':>7}")
    seg_results["year"] = {}
    for yr in sorted(train["Year"].unique()):
        mask = train["Year"].values == yr
        n_seg = int(mask.sum()); pos = int(y[mask].sum())
        if n_seg == 0 or pos == 0 or pos == n_seg:
            continue
        a_p = roc_auc_score(y[mask], primary_oof[mask])
        a_e = roc_auc_score(y[mask], e3_oof[mask])
        a_c = roc_auc_score(y[mask], cb_oof[mask])
        print(f"{yr:<6} {n_seg:>8d} {pos:>8d} {a_p:>8.5f} "
              f"{a_e:>8.5f} {(a_e-a_p)*1e4:>+7.1f} "
              f"{a_c:>8.5f} {(a_c-a_p)*1e4:>+7.1f}")
        seg_results["year"][str(yr)] = dict(
            n=n_seg, pos=pos, primary=float(a_p), e3=float(a_e), cb=float(a_c),
            d_e3_bp=(a_e - a_p) * 1e4, d_cb_bp=(a_c - a_p) * 1e4)

    # Stint
    print("\n-- by Stint (clip 5) --")
    print(f"{'stint':<6} {'n':>8} {'pos':>8} {'PRIM':>8} {'E3':>8} {'ΔE3':>7} {'CB':>8} {'ΔCB':>7}")
    seg_results["stint"] = {}
    s_arr = train["Stint"].clip(upper=5).astype(int).values
    for sv in [1, 2, 3, 4, 5]:
        mask = s_arr == sv
        n_seg = int(mask.sum()); pos = int(y[mask].sum())
        if n_seg == 0 or pos == 0 or pos == n_seg:
            continue
        a_p = roc_auc_score(y[mask], primary_oof[mask])
        a_e = roc_auc_score(y[mask], e3_oof[mask])
        a_c = roc_auc_score(y[mask], cb_oof[mask])
        print(f"{sv:<6} {n_seg:>8d} {pos:>8d} {a_p:>8.5f} "
              f"{a_e:>8.5f} {(a_e-a_p)*1e4:>+7.1f} "
              f"{a_c:>8.5f} {(a_c-a_p)*1e4:>+7.1f}")
        seg_results["stint"][str(sv)] = dict(
            n=n_seg, pos=pos, primary=float(a_p), e3=float(a_e), cb=float(a_c),
            d_e3_bp=(a_e - a_p) * 1e4, d_cb_bp=(a_c - a_p) * 1e4)

    # Compound
    print("\n-- by Compound --")
    print(f"{'comp':<10} {'n':>8} {'pos':>8} {'PRIM':>8} {'E3':>8} {'ΔE3':>7} {'CB':>8} {'ΔCB':>7}")
    seg_results["compound"] = {}
    for cv in sorted(train["Compound"].astype(str).unique()):
        mask = train["Compound"].astype(str).values == cv
        n_seg = int(mask.sum()); pos = int(y[mask].sum())
        if n_seg == 0 or pos == 0 or pos == n_seg:
            continue
        a_p = roc_auc_score(y[mask], primary_oof[mask])
        a_e = roc_auc_score(y[mask], e3_oof[mask])
        a_c = roc_auc_score(y[mask], cb_oof[mask])
        print(f"{cv:<10} {n_seg:>8d} {pos:>8d} {a_p:>8.5f} "
              f"{a_e:>8.5f} {(a_e-a_p)*1e4:>+7.1f} "
              f"{a_c:>8.5f} {(a_c-a_p)*1e4:>+7.1f}")
        seg_results["compound"][cv] = dict(
            n=n_seg, pos=pos, primary=float(a_p), e3=float(a_e), cb=float(a_c),
            d_e3_bp=(a_e - a_p) * 1e4, d_cb_bp=(a_c - a_p) * 1e4)

    # Disagreement rate (|bag − PRIMARY| > 0.1) — train OOF axis
    print("\n=== Disagreement (|bag − PRIMARY| > 0.1) on train OOF ===")
    disagr_results = {}
    for name, bag in [("e3", e3_oof), ("cb", cb_oof), ("avg", avg_oof)]:
        d = np.abs(bag - primary_oof)
        mask = d > 0.1
        rate = float(mask.mean())
        n_dis = int(mask.sum())
        print(f"  {name}: rate={rate:.4f}  n={n_dis}")
        # Cohort breakdown by year, stint, compound
        coh = {}
        if n_dis > 0:
            for col, vals in [
                ("Year", sorted(train["Year"].unique())),
                ("Stint5", [1, 2, 3, 4, 5]),
                ("Compound", sorted(train["Compound"].astype(str).unique())),
            ]:
                if col == "Stint5":
                    arr = train["Stint"].clip(upper=5).astype(int).values
                else:
                    arr = train[col].values if col != "Compound" else train[col].astype(str).values
                tot_per = {}
                dis_per = {}
                for v in vals:
                    if col == "Stint5":
                        m = arr == v
                    else:
                        m = arr == v
                    tot_per[str(v)] = int(m.sum())
                    dis_per[str(v)] = int((m & mask).sum())
                # frac of disagreements vs population frac
                lift = {}
                for v in vals:
                    pop_frac = tot_per[str(v)] / n
                    dis_frac = dis_per[str(v)] / max(n_dis, 1)
                    lift[str(v)] = float(dis_frac / pop_frac) if pop_frac > 0 else float("nan")
                coh[col] = dict(tot=tot_per, dis=dis_per, dis_lift=lift)
                print(f"    by {col}: lift = " + ", ".join(
                    f"{v}={lift[str(v)]:.2f}" for v in vals))
        disagr_results[name] = dict(rate=rate, n=n_dis, cohort_lift=coh)

    summary = dict(
        primary_oof_auc=auc_primary,
        bags=dict(
            e3=dict(oof_auc=auc_e3, rho_test=float(rho_e3),
                    delta_oof_bp=float((auc_e3 - auc_primary) * 1e4),
                    min_meta_oof=mm_e3,
                    min_meta_delta_bp=float((mm_e3 - auc_primary) * 1e4)),
            cb=dict(oof_auc=auc_cb, rho_test=float(rho_cb),
                    delta_oof_bp=float((auc_cb - auc_primary) * 1e4),
                    min_meta_oof=mm_cb,
                    min_meta_delta_bp=float((mm_cb - auc_primary) * 1e4)),
            avg=dict(oof_auc=auc_avg, rho_test=float(rho_avg),
                     delta_oof_bp=float((auc_avg - auc_primary) * 1e4),
                     min_meta_oof=mm_avg,
                     min_meta_delta_bp=float((mm_avg - auc_primary) * 1e4)),
        ),
        per_segment=seg_results,
        disagreement=disagr_results,
        wall_s=time.time() - t0,
    )
    out = ART / "d12_bag_probe_results.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n→ wrote {out} (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
