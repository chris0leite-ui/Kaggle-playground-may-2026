"""Phase 1 — AUC-direct meta loss variants vs LR-meta on K=4.

Baseline: sklearn LogisticRegression on [P, rank, logit] expansion of
K=4 OOFs. Anchor OOF AUC = 0.95399.

Variants:
  - lgbm_rank_xendcg: LightGBM with objective='rank_xendcg', single
    group containing all rows (pairwise XENDCG ranking loss).
  - sgd_hinge_pairwise: sklearn SGDClassifier hinge loss on pairwise
    difference features (positive-row minus negative-row).
  - torch_auc_surrogate: small MLP with smooth-AUC surrogate loss
    (sigmoid(s_neg - s_pos)).

Each variant is wrapped as a 5-fold CV meta-OOF function; same fold
seed/structure as probe_min_meta.py so OOF AUCs are directly
comparable.

Origin: 2026-05-18 round-2 plan P1.1/P1.2/P1.3.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K4 = ["d17_h1d_yekenot_full", "p1_single_cb_v4_gpu",
      "f1_hgbc_deep", "d16_orig_continuous_only"]


def pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def meta_oof_lr(y, F):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def _bucket_groups(n: int, group_size: int = 5000):
    """LightGBM caps group size at 10000. Bucket into groups of
    group_size rows (last group may be smaller)."""
    sizes = [group_size] * (n // group_size)
    rem = n - sum(sizes)
    if rem > 0:
        sizes.append(rem)
    return sizes


def meta_oof_lgbm_rank(y, F, params=None, group_size=5000):
    """LightGBM Ranker with rank_xendcg; bucketed groups of
    `group_size` rows (LightGBM caps single-group size at 10k).
    Each group is a separate ranking task; the model learns to
    distinguish positives from negatives within each bucket."""
    if params is None:
        params = dict(objective="rank_xendcg", n_estimators=300,
                      num_leaves=15, learning_rate=0.05,
                      min_data_in_leaf=200, lambda_l2=1.0,
                      verbose=-1, n_jobs=-1, random_state=SEED)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    for tr, va in skf.split(np.zeros(len(y)), y):
        # Shuffle to randomize which rows end up in which group
        # (preserves stratification of positives across groups)
        perm = rng.permutation(len(tr))
        tr_perm = tr[perm]
        groups_tr = _bucket_groups(len(tr_perm), group_size=group_size)
        groups_va = _bucket_groups(len(va), group_size=group_size)
        m = lgb.LGBMRanker(**params)
        m.fit(F[tr_perm], y[tr_perm], group=groups_tr,
              eval_set=[(F[va], y[va])], eval_group=[groups_va],
              eval_at=[10, 100],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(0)])
        oof[va] = m.predict(F[va])
    return oof, float(roc_auc_score(y, oof))


def meta_oof_sgd_pairwise(y, F, n_pairs_per_fold=100_000):
    """SGDClassifier hinge loss on pairwise (pos, neg) differences.

    Construct features = F[pos_idx] - F[neg_idx] for sampled pairs;
    label = +1 (since pos should outrank neg). At inference, score
    is the raw decision_function on F (one-sided).
    """
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    for tr, va in skf.split(np.zeros(len(y)), y):
        pos_idx = np.where(y[tr] == 1)[0]
        neg_idx = np.where(y[tr] == 0)[0]
        pos_idx_tr = tr[pos_idx]
        neg_idx_tr = tr[neg_idx]
        # Sample pairs
        ip = rng.choice(pos_idx_tr, size=n_pairs_per_fold, replace=True)
        in_ = rng.choice(neg_idx_tr, size=n_pairs_per_fold, replace=True)
        X_pairs = F[ip] - F[in_]
        # Label all +1 (pos minus neg → positive class)
        y_pairs = np.ones(n_pairs_per_fold, dtype=np.int32)
        # SGD on hinge with no class_weight tweak (data is balanced by construction)
        clf = SGDClassifier(loss="hinge", alpha=1e-5, max_iter=20,
                            random_state=SEED, fit_intercept=False)
        # Add an opposite class to make sklearn happy:
        X_pairs_full = np.vstack([X_pairs, -X_pairs])
        y_pairs_full = np.hstack([y_pairs, np.zeros_like(y_pairs)])
        clf.fit(X_pairs_full, y_pairs_full)
        # Score = decision_function on F[va] (the model learned a w that
        # ranks positives above negatives)
        oof[va] = clf.decision_function(F[va])
    return oof, float(roc_auc_score(y, oof))


def meta_oof_torch_auc(y, F, n_epochs=30, hidden=32, batch_size=2048,
                       lr=1e-3, neg_per_pos=4):
    """Tiny MLP with smooth-AUC surrogate: mean over (pos, neg) pairs
    of sigmoid(score_neg - score_pos). Lower is better."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    in_dim = F.shape[1]
    rng = np.random.default_rng(SEED)

    for tr, va in skf.split(np.zeros(len(y)), y):
        torch.manual_seed(SEED)
        net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        F_tr = torch.tensor(F[tr], dtype=torch.float32)
        y_tr = torch.tensor(y[tr], dtype=torch.float32)
        F_va = torch.tensor(F[va], dtype=torch.float32)
        pos_mask = (y_tr == 1).numpy()
        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(~pos_mask)[0]

        for epoch in range(n_epochs):
            n_batches = max(1, len(pos_idx) // (batch_size // (1 + neg_per_pos)))
            losses = []
            for _ in range(n_batches):
                bp = batch_size // (1 + neg_per_pos)
                bn = bp * neg_per_pos
                pi = rng.choice(pos_idx, size=bp, replace=True)
                ni = rng.choice(neg_idx, size=bn, replace=True)
                s_pos = net(F_tr[pi]).squeeze(-1)  # (bp,)
                s_neg = net(F_tr[ni]).squeeze(-1)  # (bn,)
                # Pairwise: for each (pos, neg) pair, sigmoid(s_neg - s_pos)
                # Repeat pos so each pos vs neg_per_pos negs
                s_pos_rep = s_pos.repeat_interleave(neg_per_pos)
                loss = torch.sigmoid(s_neg - s_pos_rep).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        with torch.no_grad():
            oof[va] = net(F_va).squeeze(-1).numpy()
    return oof, float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["lr", "lgbm_rank", "sgd_pairwise", "torch_auc"],
                    help="Which meta variants to run")
    args = ap.parse_args()

    print("=== Phase 1 — AUC-direct meta loss variants vs LR on K=4 ===")
    print(f"Variants: {args.variants}")
    y = pd.read_csv("data/train.csv")[TARGET].astype(int).values
    P_oof = np.column_stack([pos(ART / f"oof_{b}_strat.npy") for b in K4])
    P_test = np.column_stack([pos(ART / f"test_{b}_strat.npy") for b in K4])
    F_oof = expand(P_oof)
    F_test = expand(P_test)
    print(f"\ny shape {y.shape}; F_oof shape {F_oof.shape} ({F_oof.shape[1]} feats)")

    primary_test = pos(ART / "test_d13e_compound_stint_tau20000_strat.npy")

    results = {}
    for variant in args.variants:
        print(f"\n--- variant: {variant} ---")
        t0 = time.time()
        if variant == "lr":
            oof, auc = meta_oof_lr(y, F_oof)
        elif variant == "lgbm_rank":
            oof, auc = meta_oof_lgbm_rank(y, F_oof)
        elif variant == "sgd_pairwise":
            oof, auc = meta_oof_sgd_pairwise(y, F_oof)
        elif variant == "torch_auc":
            oof, auc = meta_oof_torch_auc(y, F_oof)
        else:
            print(f"  unknown variant {variant}, skip")
            continue
        wall = time.time() - t0
        # Also need test-set OOF→test mapping; for AUC variants without
        # natural probability output, do a full-train refit for test pred
        # (mirrors probe_min_meta._meta_full_test).
        if variant == "lr":
            lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            lr_full.fit(F_oof, y)
            test_pred = lr_full.predict_proba(F_test)[:, 1]
        elif variant == "lgbm_rank":
            m = lgb.LGBMRanker(objective="rank_xendcg", n_estimators=300,
                               num_leaves=15, learning_rate=0.05,
                               min_data_in_leaf=200, lambda_l2=1.0,
                               verbose=-1, n_jobs=-1, random_state=SEED)
            rng_full = np.random.default_rng(SEED)
            perm = rng_full.permutation(len(y))
            groups_full = _bucket_groups(len(y), group_size=5000)
            m.fit(F_oof[perm], y[perm], group=groups_full)
            test_pred = m.predict(F_test)
        elif variant == "sgd_pairwise":
            # Refit with all data for test
            rng = np.random.default_rng(SEED)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            ip = rng.choice(pos_idx, size=200_000, replace=True)
            in_ = rng.choice(neg_idx, size=200_000, replace=True)
            X_pairs = F_oof[ip] - F_oof[in_]
            X_pairs_full = np.vstack([X_pairs, -X_pairs])
            y_pairs_full = np.hstack([np.ones(len(X_pairs)),
                                       np.zeros(len(X_pairs))])
            clf = SGDClassifier(loss="hinge", alpha=1e-5, max_iter=20,
                                random_state=SEED, fit_intercept=False)
            clf.fit(X_pairs_full, y_pairs_full)
            test_pred = clf.decision_function(F_test)
        elif variant == "torch_auc":
            import torch
            import torch.nn as nn
            import torch.optim as optim
            torch.manual_seed(SEED)
            in_dim = F_oof.shape[1]
            net = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
                nn.Linear(32, 1))
            opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            F_tr = torch.tensor(F_oof, dtype=torch.float32)
            y_tr = torch.tensor(y, dtype=torch.float32)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            for epoch in range(30):
                n_batches = max(1, len(pos_idx) // 410)
                rng_e = np.random.default_rng(SEED + epoch)
                for _ in range(n_batches):
                    pi = rng_e.choice(pos_idx, size=410, replace=True)
                    ni_arr = rng_e.choice(neg_idx, size=1640, replace=True)
                    s_pos = net(F_tr[pi]).squeeze(-1)
                    s_neg = net(F_tr[ni_arr]).squeeze(-1)
                    s_pos_rep = s_pos.repeat_interleave(4)
                    loss = torch.sigmoid(s_neg - s_pos_rep).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                test_pred = net(torch.tensor(F_test, dtype=torch.float32)
                                ).squeeze(-1).numpy()

        # Save normalized to [0, 1] range for downstream consistency
        # (Some variants produce real-valued scores, not probabilities)
        if variant in ("lgbm_rank", "sgd_pairwise", "torch_auc"):
            # Min-max normalize OOF + test consistently (use OOF stats)
            o_min, o_max = oof.min(), oof.max()
            range_ = o_max - o_min + 1e-12
            oof_n = (oof - o_min) / range_
            test_n = (test_pred - o_min) / range_
            test_n = np.clip(test_n, 0.0, 1.0)
            oof_n = np.clip(oof_n, 0.0, 1.0)
        else:
            oof_n, test_n = oof, test_pred

        rho, _ = spearmanr(test_n, primary_test)
        delta = (auc - 0.95399) * 1e4
        results[variant] = {"oof_auc": auc, "delta_bp_vs_lr": delta,
                            "rho_vs_primary": float(rho),
                            "wall_seconds": wall}
        print(f"  OOF AUC: {auc:.5f}  Δ vs LR baseline: {delta:+.3f} bp  "
              f"ρ vs PRIMARY: {rho:.6f}  wall: {wall:.1f}s")
        # Save artifacts
        name = f"K4_meta_{variant}"
        np.save(ART / f"oof_{name}_strat.npy",
                np.column_stack([1 - oof_n, oof_n]).astype(np.float64))
        np.save(ART / f"test_{name}_strat.npy",
                np.column_stack([1 - test_n, test_n]).astype(np.float64))
        print(f"  → oof_{name}_strat.npy / test_{name}_strat.npy")

    print("\n=== Summary ===")
    print(f"{'variant':<20s}  {'OOF AUC':>9s}  {'Δ bp':>10s}  {'ρ':>10s}")
    for v, r in results.items():
        print(f"  {v:<20s}  {r['oof_auc']:.5f}  {r['delta_bp_vs_lr']:+10.3f}  "
              f"{r['rho_vs_primary']:.6f}")

    (ART / "probe_meta_loss_variants_results.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
