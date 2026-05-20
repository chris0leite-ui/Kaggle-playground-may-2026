"""scripts/probe_r18_multitask_joint.py — Round 18 inventor track:
multi-task joint NN with 4 task heads on raw + TE features.

Hypothesis: stacking is post-hoc combination of independently-trained
representations (R12-2 cb_horizon, R13 cb_stint_completion, R14
cb_next_compound, K=13 PitNextLap bases each added independent base
lift). Joint multi-task training creates a SHARED representation
simultaneously consistent with all four targets — richer than
independent stacking can extract.

Architecture: PyTorch MLP, shared trunk (3 hidden × 256 GELU+LN+dropout),
4 task heads:
- PitNextLap (1u, BCE loss, weight 1.0) — primary target
- LapsUntilPit (1u, RMSE on log+1 capped 30, weight 0.3)
- StintCompletion (1u, RMSE on [0,1], weight 0.3)
- NextCompound (5u, CE loss, weight 0.3; masked for rows where next
  compound is undefined)

Per-fold strict target rebuild (Rule 24). Train on union (438k rows).
Output PitNextLap head OOF/test → save as base for Path-B K=18 add.

Reuses:
- p1_features.make_features_static + feature_columns_for_lgbm + TE_CONFIGS
- p1_single_cb.fold_safe_te_for_fold
- b_laps_until_pit.build_laps_until_pit
- probe_r13_cb_stint_completion.build_stint_completion
- probe_r14_cb_next_compound.build_next_compound_target + COMPOUND_VOCAB

Usage:
  python scripts/probe_r18_multitask_joint.py [--smoke] [--epochs 30]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from p1_features import (
    TE_CONFIGS, apply_fs_a, feature_columns_for_lgbm,
    fit_fs_a, make_features_static,
)
from p1_single_cb import fold_safe_te_for_fold
from b_laps_until_pit import build_laps_until_pit
from probe_r13_cb_stint_completion import build_stint_completion
from probe_r14_cb_next_compound import build_next_compound_target, COMPOUND_VOCAB

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
N_COMPOUND = 5  # SOFT/MEDIUM/HARD/INTER/WET (per COMPOUND_VOCAB)

REF_R15_OOF = ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy"
REF_R15_TEST = ART / "test_K17_xendcg_pathb_dcs_tau100000.npy"


class MultiTaskMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
        )
        self.head_pit = nn.Linear(hidden, 1)
        self.head_laps = nn.Linear(hidden, 1)
        self.head_stint = nn.Linear(hidden, 1)
        self.head_compound = nn.Linear(hidden, N_COMPOUND)

    def forward(self, x):
        h = self.trunk(x)
        return (self.head_pit(h).squeeze(-1),
                self.head_laps(h).squeeze(-1),
                self.head_stint(h).squeeze(-1),
                self.head_compound(h))


def train_one_fold(X_tr, X_va, X_te, y_pit_tr, y_pit_va, y_laps_tr,
                   y_stint_tr, y_comp_tr, comp_mask_tr, epochs, lr,
                   weight_decay, batch_size, fold, total_folds):
    device = torch.device("cpu")
    n_in = X_tr.shape[1]
    model = MultiTaskMLP(n_in).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss(reduction="none")

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    y_pit_t = torch.tensor(y_pit_tr, dtype=torch.float32)
    y_laps_t = torch.tensor(y_laps_tr, dtype=torch.float32)
    y_stint_t = torch.tensor(y_stint_tr, dtype=torch.float32)
    y_comp_t = torch.tensor(y_comp_tr, dtype=torch.long)
    mask_t = torch.tensor(comp_mask_tr, dtype=torch.float32)
    ds = TensorDataset(Xt, y_pit_t, y_laps_t, y_stint_t, y_comp_t, mask_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    for ep in range(epochs):
        model.train()
        tot = 0.0
        n = 0
        for batch in loader:
            xb, yp, yl, ys, yc, m = batch
            xb = xb.to(device)
            opt.zero_grad()
            p_pit, p_laps, p_stint, p_comp = model(xb)
            L_pit = bce(p_pit, yp.to(device))
            L_laps = mse(p_laps, yl.to(device))
            L_stint = mse(p_stint, ys.to(device))
            ce_per = ce(p_comp, yc.to(device))  # (B,)
            mask_d = m.to(device)
            L_comp = (ce_per * mask_d).sum() / (mask_d.sum() + 1e-6)
            loss = L_pit + 0.3 * (L_laps + L_stint + L_comp)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            # Quick val AUC on PitNextLap
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
                p_pit_va, _, _, _ = model(Xv)
                p_pit_va = torch.sigmoid(p_pit_va).cpu().numpy()
            try:
                auc = float(roc_auc_score(y_pit_va, p_pit_va))
            except ValueError:
                auc = float("nan")
            print(f"  fold {fold}/{total_folds} ep {ep+1}/{epochs} "
                  f"loss {tot/n:.5f} val AUC {auc:.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
        p_pit_va, _, _, _ = model(Xv)
        p_pit_va_pr = torch.sigmoid(p_pit_va).cpu().numpy()
        Xte_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        p_pit_te, _, _, _ = model(Xte_t)
        p_pit_te_pr = torch.sigmoid(p_pit_te).cpu().numpy()
    return p_pit_va_pr, p_pit_te_pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    t0 = time.time()
    print("== R18 inventor: multi-task joint NN ==", flush=True)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    y_all = train[TARGET].astype(int).values

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(train), 50_000,
                                                  replace=False)
        train = train.iloc[idx].reset_index(drop=True)
        y_all = train[TARGET].astype(int).values
        print(f"  SMOKE: subset to {train.shape}", flush=True)

    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True).values

    sorted_ids = train_S[ID_COL].values
    orig_train_ids = train[ID_COL].values
    id_to_sorted_pos = {tid: i for i, tid in enumerate(sorted_ids)}
    test_sorted_ids = test_S[ID_COL].values
    test_orig_ids = test[ID_COL].values
    test_id_to_sorted_pos = {tid: i for i, tid in enumerate(test_sorted_ids)}

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    sample_ti = fold_list[0][0]
    sample_fs_a = fit_fs_a(train_S.iloc[sample_ti])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in feats and c not in cat_cols:
            cat_cols.append(c)
    feats = feats + [n for _, _, n in TE_CONFIGS]
    print(f"  feats: {len(feats)}  cat: {len(cat_cols)}", flush=True)

    oof_pit = np.zeros(len(y), dtype=np.float64)
    test_pit = np.zeros(len(test_S), dtype=np.float64)
    fold_aucs, fold_walls = [], []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
        t_f = time.time()
        print(f"\n  --- Fold {fold}/{n_eff_folds} | ti={len(ti)} va={len(vi)} ---",
              flush=True)

        fs_a = fit_fs_a(train_S.iloc[ti])
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)

        y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
        fold_safe_te_for_fold(train_ti, train_va, test_fold, y_ti, fold, N_FOLDS)

        # Build aux targets per-fold strict
        ti_df = train_S.iloc[ti].reset_index(drop=True)
        y_laps_ti = build_laps_until_pit(ti_df).astype(np.float32)
        y_stint_ti = build_stint_completion(ti_df).astype(np.float32)
        y_comp_ti_raw = build_next_compound_target(ti_df)
        comp_mask_ti = (y_comp_ti_raw >= 0).astype(np.float32)
        y_comp_ti = np.where(y_comp_ti_raw >= 0, y_comp_ti_raw, 0).astype(np.int64)
        # Standardize aux targets to zero-mean / unit-var for stable training
        laps_mu, laps_sd = float(y_laps_ti.mean()), float(y_laps_ti.std() + 1e-6)
        y_laps_ti = (y_laps_ti - laps_mu) / laps_sd
        stint_mu, stint_sd = float(y_stint_ti.mean()), float(y_stint_ti.std() + 1e-6)
        y_stint_ti = (y_stint_ti - stint_mu) / stint_sd

        # Build feature matrices, fillna(0)
        X_tr = train_ti.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        X_va = train_va.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        X_te = test_fold.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        # Standardize features (use ti stats)
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-6
        X_tr = (X_tr - mu) / sd
        X_va = (X_va - mu) / sd
        X_te = (X_te - mu) / sd

        y_pit_ti = train_ti[TARGET].astype(np.float32).values
        y_pit_va = train_va[TARGET].astype(int).values

        p_pit_va, p_pit_te = train_one_fold(
            X_tr, X_va, X_te,
            y_pit_ti, y_pit_va,
            y_laps_ti, y_stint_ti, y_comp_ti, comp_mask_ti,
            epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, batch_size=args.batch,
            fold=fold, total_folds=n_eff_folds,
        )
        oof_pit[vi] = p_pit_va
        if not args.smoke:
            test_pit += p_pit_te / n_eff_folds

        auc_va = float(roc_auc_score(y_pit_va, p_pit_va))
        fold_aucs.append(auc_va)
        wall = time.time() - t_f
        fold_walls.append(wall)
        print(f"    fold {fold} wall {wall:.0f}s AUC(val PitNextLap)={auc_va:.5f}",
              flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold proj "
              f"~{(time.time()-t0)*N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y, oof_pit))
    print(f"\n  R18 multi-task PitNextLap-head OOF AUC: {auc_full:.6f}",
          flush=True)

    # ρ vs R15 PRIMARY (with id-order alignment per friction cb-focal-rho-misalign-bug)
    full_train_ids = pd.read_csv("data/train.csv")[ID_COL].values
    id_to_r15_pos = {tid: i for i, tid in enumerate(full_train_ids)}
    r15_oof_full = np.load(REF_R15_OOF)
    r15_oof_sorted = np.array([r15_oof_full[id_to_r15_pos[t]] for t in sorted_ids])
    auc_r15 = float(roc_auc_score(y, r15_oof_sorted))
    delta_bp = (auc_full - auc_r15) * 1e4
    rho_oof, _ = spearmanr(oof_pit, r15_oof_sorted)
    print(f"  R15 PRIMARY OOF: {auc_r15:.6f}", flush=True)
    print(f"  Δ vs R15: {delta_bp:+.4f} bp", flush=True)
    print(f"  ρ_OOF vs R15: {rho_oof:.6f}", flush=True)

    r15_test_full = np.load(REF_R15_TEST)
    r15_test_sorted = np.array([r15_test_full[test_id_to_sorted_pos[t]]
                                 for t in test_orig_ids])
    # Note: r15_test_full is already in test.csv original order, so
    # we align it to sorted (S) order for ρ on test.
    test_orig_to_sorted = np.array([test_id_to_sorted_pos[t] for t in test_orig_ids])
    r15_test_in_sort_order = np.empty_like(r15_test_full)
    r15_test_in_sort_order[test_orig_to_sorted] = r15_test_full
    rho_test, _ = spearmanr(test_pit, r15_test_in_sort_order)
    print(f"  ρ_test vs R15: {rho_test:.6f}", flush=True)
    if rho_test >= 0.9999:
        verdict = "TIE_ZONE"
    elif rho_test < 0.999:
        verdict = "REGRESSION_RISK"
    else:
        verdict = "OK"
    print(f"  Band verdict: {verdict}", flush=True)

    # Save in original train.csv order
    order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
    order_back_test = np.array([test_id_to_sorted_pos[t] for t in test_orig_ids])
    np.save(ART / "oof_R18_multitask_pit_head_strat.npy",
            oof_pit[order_back_train].astype(np.float32))
    np.save(ART / "test_R18_multitask_pit_head_strat.npy",
            test_pit[order_back_test].astype(np.float32))
    print(f"  Saved oof_R18_multitask_pit_head_strat.npy + test variant",
          flush=True)

    summary = dict(
        round="R18_multitask_joint_NN",
        oof_auc=auc_full,
        auc_r15_baseline=auc_r15,
        delta_bp_vs_r15=delta_bp,
        rho_oof_vs_r15=float(rho_oof),
        rho_test_vs_r15=float(rho_test),
        verdict_band=verdict,
        fold_aucs=fold_aucs,
        fold_walls_s=fold_walls,
        wall_total_s=time.time() - t0,
        epochs=args.epochs,
        n_feats=len(feats),
    )
    out_json = Path("audit/2026-05-20-round-18-multitask.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}  total wall {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
