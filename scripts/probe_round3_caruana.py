"""Round 3 P1.2 — Caruana forward-selection-with-replacement on
available LB-confirmed PRIMARYs + R5 hedge candidates.

Caruana's ensemble selection (2004): start empty; at each step add
the candidate that most improves held-out OOF AUC when averaged in.
Selection-with-replacement means a candidate can be picked multiple
times (its weight grows). Stop when AUC stops improving or max_steps.

Pool of candidates in this iteration's snapshot:
  - oof_K4_fwd_pathb (LB 0.95351, K=4 + Path-B)
  - oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat (LB 0.95368, K=27 + Path-B)
  - oof_d17_path_b_K23_v4_h1d_tau100000_strat (LB 0.95354, 23-base era)
  - oof_d16_path_b_K22_continuous_only_tau100000_strat (LB 0.95089-ish)
  - oof_d15b_path_b_K22_dae_only_tau100000_strat (Day-15 PRIMARY runner-up)
  - oof_d13e_compound_stint_tau20000_strat (d13e PRIMARY)
  Plus 11 Round-2 candidate OOFs (mostly null at meta but might add
  diversity in arithmetic mean).

Origin: 2026-05-18 round-3 plan P1.2.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

# (name, oof_file, test_file_or_None, LB_or_None)
CANDIDATES = [
    ("K4_pathb", "oof_K4_fwd_pathb.npy",
     "test_K4_fwd_pathb.npy", 0.95351),
    ("K27_pathb", "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
     "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy", 0.95368),
    ("K23_v4h1d_pathb", "oof_d17_path_b_K23_v4_h1d_tau100000_strat.npy",
     "test_d17_path_b_K23_v4_h1d_tau100000_strat.npy", 0.95354),
    ("K22_d16_continuous_only", "oof_d16_path_b_K22_continuous_only_tau100000_strat.npy",
     "test_d16_path_b_K22_continuous_only_tau100000_strat.npy", None),
    ("K22_dae_only", "oof_d15b_path_b_K22_dae_only_tau100000_strat.npy",
     "test_d15b_path_b_K22_dae_only_tau100000_strat.npy", None),
    ("d13e_C_x_S", "oof_d13e_compound_stint_tau20000_strat.npy",
     "test_d13e_compound_stint_tau20000_strat.npy", 0.95049),
    # Round-2 candidates (all null at meta; may add diversity)
    ("R2_conformal", "oof_K4_conformal_widths_strat.npy",
     "test_K4_conformal_widths_strat.npy", None),
    ("R2_rrf_k60", "oof_K4_rrf_k60_strat.npy",
     "test_K4_rrf_k60_strat.npy", None),
    ("R2_lgbm_rank", "oof_K4_meta_lgbm_rank_strat.npy",
     "test_K4_meta_lgbm_rank_strat.npy", None),
    ("R2_trimmed", "oof_K4_trimmed_rank_t1_1_strat.npy",
     "test_K4_trimmed_rank_t1_1_strat.npy", None),
    ("R2_glmm", "oof_K4_glmm_best_strat.npy",
     "test_K4_glmm_best_strat.npy", None),
]

MAX_STEPS = 50


def pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def main():
    y = pd.read_csv("data/train.csv")[TARGET].astype(int).values

    # Load all candidates
    oofs = {}
    tests = {}
    for name, oof_file, test_file, _ in CANDIDATES:
        path = ART / oof_file
        if not path.exists():
            print(f"  ⚠ missing {oof_file}; skip")
            continue
        oofs[name] = pos(path)
        if test_file and (ART / test_file).exists():
            tests[name] = pos(ART / test_file)
    print(f"Loaded {len(oofs)} candidates: {list(oofs.keys())}")

    # Standalone OOF AUCs
    print(f"\n{'name':<25s}  {'OOF AUC':>9s}  {'LB':>9s}")
    print("-" * 55)
    for name, oof_file, _, lb in CANDIDATES:
        if name not in oofs:
            continue
        auc = roc_auc_score(y, oofs[name])
        lb_str = f"{lb:.5f}" if lb else "n/a"
        print(f"{name:<25s}  {auc:.5f}  {lb_str:>9s}")

    # Use a normalised-rank-mean variant for Caruana (the team's blend
    # harness operates on rank-mean ensembles; this matches that family).
    def rank_norm(p):
        return rankdata(p, method="average") / len(p)

    oof_ranks = {n: rank_norm(o) for n, o in oofs.items()}
    test_ranks = {n: rank_norm(t) for n, t in tests.items()
                  if n in tests}

    # Caruana forward-selection-with-replacement on arithmetic mean of
    # normalized ranks. Optimize OOF AUC.
    names = list(oofs.keys())
    selected = []
    history = []
    current_sum_oof = np.zeros(len(y))
    current_sum_test = np.zeros(len(next(iter(test_ranks.values()))))
    best_auc = -np.inf

    print(f"\n=== Caruana hill-climb (with replacement, max {MAX_STEPS} steps) ===")
    for step in range(MAX_STEPS):
        best_pick = None
        best_step_auc = -np.inf
        for name in names:
            trial_sum = current_sum_oof + oof_ranks[name]
            trial_avg = trial_sum / (step + 1)
            auc = roc_auc_score(y, trial_avg)
            if auc > best_step_auc:
                best_step_auc = auc
                best_pick = name
        # Greedy: pick the best-step regardless of whether it improves
        # over best (Caruana w/ replacement can plateau then improve).
        # But stop early if no improvement for 3 consecutive steps.
        if best_step_auc <= best_auc - 1e-7 and step > 5:
            # No improvement on this step (could be local max)
            pass
        if best_step_auc > best_auc:
            best_auc = best_step_auc
        selected.append(best_pick)
        current_sum_oof = current_sum_oof + oof_ranks[best_pick]
        if best_pick in test_ranks:
            current_sum_test = current_sum_test + test_ranks[best_pick]
        history.append((step + 1, best_pick, best_step_auc))
        print(f"  step {step+1:>2d}: pick {best_pick:<25s}  "
              f"OOF AUC = {best_step_auc:.5f}")
        # Stop if last 5 picks made <0.01 bp improvement total
        if step >= 5:
            recent = [h[2] for h in history[-5:]]
            if max(recent) - min(recent) < 1e-7:
                print(f"  → plateaued at step {step+1}; stop")
                break

    # Final blend = arithmetic mean of selected ranks
    n_steps = len(selected)
    blend_oof = current_sum_oof / n_steps
    blend_test = current_sum_test / n_steps if n_steps > 0 else None
    final_auc = roc_auc_score(y, blend_oof)

    # Counts per candidate (weights)
    weights = {n: selected.count(n) / n_steps for n in set(selected)}
    print(f"\n=== Final Caruana blend ===")
    print(f"  steps: {n_steps}")
    print(f"  OOF AUC: {final_auc:.5f}")
    print(f"  weights:")
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<25s}  {w:.3f}")

    # Compare to best single anchor
    best_single = max(oofs.items(), key=lambda kv: roc_auc_score(y, kv[1]))
    best_single_auc = roc_auc_score(y, best_single[1])
    delta = (final_auc - best_single_auc) * 1e4
    print(f"\n  Best single anchor: {best_single[0]}  OOF {best_single_auc:.5f}")
    print(f"  Caruana Δ vs best single: {delta:+.3f} bp")

    # rho vs PRIMARY (d13e Compound × Stint τ=20k)
    if "d13e_C_x_S" in tests:
        primary_test = tests["d13e_C_x_S"]
        if blend_test is not None:
            rho, _ = spearmanr(blend_test, primary_test)
            print(f"  ρ vs d13e PRIMARY: {rho:.6f}")

    # Save artifacts
    np.save(ART / "oof_caruana_blend_round3_strat.npy",
            np.column_stack([1 - blend_oof, blend_oof]).astype(np.float64))
    if blend_test is not None:
        np.save(ART / "test_caruana_blend_round3_strat.npy",
                np.column_stack([1 - blend_test, blend_test]).astype(np.float64))
        Path("submissions").mkdir(exist_ok=True)
        sub = pd.read_csv("data/sample_submission.csv")
        sub[TARGET] = np.clip(blend_test, 0.001, 0.999)
        sub.to_csv("submissions/submission_caruana_round3.csv", index=False)
        print(f"  → submissions/submission_caruana_round3.csv")

    (ART / "probe_round3_caruana_results.json").write_text(json.dumps(dict(
        steps=n_steps, oof_auc=final_auc, best_single=best_single[0],
        best_single_auc=best_single_auc, delta_bp=delta,
        weights=weights, history=[(s, n, a) for s, n, a in history]),
        indent=2))


if __name__ == "__main__":
    main()
