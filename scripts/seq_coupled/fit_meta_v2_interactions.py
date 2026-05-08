"""V2: Explicit interaction terms + residual-LR + LightGBM-meta.

Hypothesis from V1 (fit_meta.py): the seq-coupled features escape the
row-local SPAN (R² = 0.487) but not the LOGIT DIRECTION the target
sits in — LR-full was +0.03 bp only. Linear meta can't pick up
conditional structure like "when row-local P is uncertain AND
look-ahead is high → push P up."

Three sharpenings tested here:

  V2.1  Explicit interaction features. Hand-build 8-10 interaction
        cols of the form [seq_feat × row-local-uncertainty-flag],
        feed to LR.

  V2.2  Two-stage residual LR. Fit LR row-local first; compute logit
        residuals; fit a SECOND LR on seq-coupled features predicting
        the residuals (gradient-boosting at the meta layer).

  V2.3  LightGBM meta on the 36-feature input (12 row-local + 24
        seq-coupled). Tree depth lets the model learn the
        conditional interactions LR can't.

If any of V2.1/V2.2/V2.3 lifts ≥+0.5 bp over LR row-local
(which was 0.95400 / matches A26b−2.93 bp gap), we have a real
mechanism. If all weak, the rank-lock is at the conditional-
discrimination level and the structural-mechanism search must
look elsewhere (new BASE class, new external data).

Outputs: scripts/artifacts/probe_seq_coupled_meta_v2.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ART = Path('scripts/artifacts')
SEED = 42
N_FOLDS = 5


def pos(p):
    a = np.load(p)
    return a[:, 1] if a.ndim == 2 else a


def expand_K4(oofs):
    parts = []
    for v in oofs.values():
        parts.append(v)
        parts.append(rankdata(v) / len(v))
        parts.append(np.log(np.clip(v, 1e-7, 1-1e-7) /
                            np.clip(1-v, 1e-7, 1-1e-7)))
    return np.column_stack(parts).astype(np.float32)


def build_interactions(X_local, X_seq, k4_pred):
    """Hand-built interaction terms: seq × row-local-uncertainty.

    Row-local uncertainty: low when k4_pred is near 0 or 1, high near 0.2-0.4.
    Captures "the K=4 meta is unsure about this row" — exactly where
    sequence-coupled context could help most.
    """
    p = k4_pred
    # uncertainty proxy: bell-shaped near prevalence
    unc = np.exp(-((p - 0.20) / 0.15) ** 2).astype(np.float32)
    # pick salient seq features by index in X_seq:
    #   0  look_ahead_pred
    #   2  look_behind_pred
    #   4  session_max
    #   5  session_mean
    #  10  delta_to_session_max
    #  11  delta_to_lookahead
    #  13  base_var
    #  15  base_spread
    sel = [0, 2, 4, 5, 10, 11, 13, 15]
    seq_sel = X_seq[:, sel]
    inter = seq_sel * unc[:, None]            # (n, 8)
    # square-deviation features (catch quadratic structure)
    sq = (X_seq[:, [0, 2, 4]] - p[:, None]) ** 2
    return np.concatenate([inter, sq], axis=1).astype(np.float32)  # (n, 11)


def fit_lr(X_tr, y_tr, X_va):
    sc = StandardScaler().fit(X_tr)
    Xt, Xv = sc.transform(X_tr), sc.transform(X_va)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    lr.fit(Xt, y_tr)
    return lr.predict_proba(Xv)[:, 1]


def fit_lgb(X_tr, y_tr, X_va):
    """LightGBM meta — depth 4, 200 trees, l2=2.0. Conservative to
    avoid overfit on the small effective sample of meta-information."""
    params = dict(
        objective='binary', metric='auc',
        num_leaves=15, max_depth=4,
        learning_rate=0.05, n_estimators=300,
        reg_alpha=1.0, reg_lambda=2.0,
        bagging_fraction=0.85, bagging_freq=5,
        feature_fraction=0.85, min_data_in_leaf=200,
        verbose=-1, seed=SEED,
    )
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, [0])] if False else None,  # no val tracking here
          callbacks=[lgb.early_stopping(0, verbose=False)] if False else None)
    return m.predict_proba(X_va)[:, 1]


def main():
    tr = pd.read_csv('data/train.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)
    oofs = {
        'h1d':  pos(ART / 'oof_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'oof_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'oof_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'oof_d16_orig_continuous_only_strat.npy'),
    }
    k4_oof = pos(ART / 'oof_K4_fwd_pathb.npy')
    X_local = expand_K4(oofs)                                # 12
    X_seq = np.load(ART / 'seq_coupled_X_train.npy')         # 24
    X_inter = build_interactions(X_local, X_seq, k4_oof)     # 11
    X_full = np.concatenate([X_local, X_seq], axis=1)        # 36
    X_full_inter = np.concatenate([X_local, X_seq, X_inter], axis=1)  # 47
    print(f'X_local: {X_local.shape}; X_seq: {X_seq.shape}; '
          f'X_inter: {X_inter.shape}; X_full+inter: {X_full_inter.shape}')

    auc_anchor = float(roc_auc_score(y, k4_oof))
    print(f'\nANCHOR K=4 PRIMARY OOF AUC: {auc_anchor:.5f}')
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    # V2.1: LR with explicit interaction features
    print('\n=== V2.1: LR on row-local + seq + interactions (47 feat) ===')
    oof_v21 = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()
        oof_v21[vai] = fit_lr(X_full_inter[tri], y[tri], X_full_inter[vai])
        print(f'  fold {k}: {roc_auc_score(y[vai], oof_v21[vai]):.5f} '
              f'({time.time()-t0:.1f}s)')
    auc_v21 = float(roc_auc_score(y, oof_v21))
    print(f'  → AUC = {auc_v21:.5f}  Δ vs PRIMARY = {(auc_v21-auc_anchor)*1e4:+.2f} bp')

    # V2.2: Two-stage residual LR
    print('\n=== V2.2: Stage-1 LR row-local; Stage-2 LR seq on residuals ===')
    oof_v22 = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()
        # Stage 1: LR on row-local
        s1 = fit_lr(X_local[tri], y[tri], X_local[tri])
        s1_va = fit_lr(X_local[tri], y[tri], X_local[vai])
        # Residual = y - predicted_p (logit-residual via target − prob)
        resid = y[tri].astype(np.float32) - s1
        # Stage 2: LR on seq features predicting residual sign (binary
        # threshold for tractability) — actually use linear regression on resid
        from sklearn.linear_model import Ridge
        sc = StandardScaler().fit(X_seq[tri])
        ridge = Ridge(alpha=10.0).fit(sc.transform(X_seq[tri]), resid)
        # Combine: final = s1_va + ridge_correction
        correction = ridge.predict(sc.transform(X_seq[vai]))
        final = np.clip(s1_va + correction, 1e-6, 1-1e-6)
        oof_v22[vai] = final
        print(f'  fold {k}: {roc_auc_score(y[vai], oof_v22[vai]):.5f} '
              f'({time.time()-t0:.1f}s)')
    auc_v22 = float(roc_auc_score(y, oof_v22))
    print(f'  → AUC = {auc_v22:.5f}  Δ vs PRIMARY = {(auc_v22-auc_anchor)*1e4:+.2f} bp')

    # V2.3: LightGBM meta on the 36-feature input
    print('\n=== V2.3: LightGBM meta on row-local + seq (36 feat) ===')
    oof_v23 = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()
        oof_v23[vai] = fit_lgb(X_full[tri], y[tri], X_full[vai])
        print(f'  fold {k}: {roc_auc_score(y[vai], oof_v23[vai]):.5f} '
              f'({time.time()-t0:.1f}s)')
    auc_v23 = float(roc_auc_score(y, oof_v23))
    print(f'  → AUC = {auc_v23:.5f}  Δ vs PRIMARY = {(auc_v23-auc_anchor)*1e4:+.2f} bp')

    out = {
        'anchor_K4_PRIMARY_oof_auc': auc_anchor,
        'v2_1_lr_with_interactions':  auc_v21,
        'v2_2_residual_lr':           auc_v22,
        'v2_3_lightgbm_meta_full':    auc_v23,
        'delta_v21_vs_primary_bp':    (auc_v21 - auc_anchor) * 1e4,
        'delta_v22_vs_primary_bp':    (auc_v22 - auc_anchor) * 1e4,
        'delta_v23_vs_primary_bp':    (auc_v23 - auc_anchor) * 1e4,
    }
    (ART / 'probe_seq_coupled_meta_v2.json').write_text(json.dumps(out, indent=2))
    print('\n=== V2 SUMMARY ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
