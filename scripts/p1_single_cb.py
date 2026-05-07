"""P1 v3 single-CatBoost — fold-safe FS_A + maximum-effort recipe.

Goal: push single-model OOF beyond the v3 LGBM ceiling (0.94563) with
CatBoost native categorical handling, per-CLAUDE.md `audit/2026-05-04-
catboost-research.md` lever map. Optionally inject precomputed
orthogonal-axis base OOFs (DAE-LGBM, d16 cont_only, FM_aug12,
leak_lookup, KNN-LGBM, multi-rule) as features — stacking-as-FE in a
single forward-pass model.

Fold safety (Rule 24 / Day-17 P1 lesson):
- FS_A label-conditional aggregates: fit per-fold on ti rows only.
- CV TE: per-fold inner stats; va/test get full-ti stats (never va labels).
- Base-OOF features: per-CV-row OOFs from prior (separate) fold splits.
  Same StratifiedKFold(seed=42) — so each row's "feature" was generated
  by a model that did not see that row's label. (Identical pattern to
  K=22 LR-meta stacking; OK as long as we're consistent about the fold
  split.)

Lever map (catboost-research.md priors):
  #1 Year ∈ cat_features       (+5-15 bp Strat)
  #2 one_hot_max_size=10       (+5-15 bp)
  #3 max_ctr_complexity=6      (+5-10 bp)
  #4 simple_ctr variety        (+5-10 bp)
  #5 slow+wide lr=0.03 it=8k   (+5-10 bp)
  #6 MVS bootstrap subsample=0.7, mvs_reg=0.1  (+5-15 bp GroupKF)
  #7 depth=10 deep             (zeta fold-0 0.94992 proof)
  #8 3-seed bag                (+1-3 bp)

Setup (one-time, run on a host with kaggle creds + GPU):
  uv pip install --system lightgbm catboost numpy pandas scikit-learn
  kaggle competitions download -c playground-series-s6e5 -p data/ && \
      cd data && unzip -o '*.zip' && cd ..
  # External datasets only needed for FS_A historical priors:
  kaggle datasets download -d debashish311601/formula-1-official-data-19502022 \
      -p external/f1_official_1950_2022/ --unzip

Smoke (fast):
  python scripts/p1_single_cb.py --smoke              # 1 fold, 50k rows, CPU/GPU auto

5-fold full:
  python scripts/p1_single_cb.py --name p1_single_cb_v3                      # vanilla
  python scripts/p1_single_cb.py --name p1_single_cb_v3_extras --with-base-oofs --with-knn
  python scripts/p1_single_cb.py --name p1_single_cb_v3_bag3 --n-seeds 3     # bagged

Honest holdout (mandatory before LB submit per Rule 24):
  python scripts/p1_holdout.py --model cb --variant feA_te
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from p1_features import (
    TE_CONFIGS, apply_fs_a, cv_target_encode, feature_columns_for_lgbm,
    fit_fs_a, make_features_static,
)

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

# Orthogonal-axis base OOFs (single probability column → feature)
BASE_OOFS = [
    ("oof_d15b_lgbm_dae_only_strat",        "test_d15b_lgbm_dae_only_strat",       "base_dae"),
    ("oof_d16_orig_continuous_only_strat",  "test_d16_orig_continuous_only_strat", "base_d16cont"),
    ("oof_d9h_FM_aug12_strat",              "test_d9h_FM_aug12_strat",             "base_fm_aug12"),
    ("oof_d15_leak_lookup_strat",           "test_d15_leak_lookup_strat",          "base_leaklook"),
    ("oof_d15d_lgbm_knn_strat",             "test_d15d_lgbm_knn_strat",            "base_knnlgb"),
    ("oof_d6_k18_multi_rule_strat",         "test_d6_k18_multi_rule_strat",        "base_multirule"),
]


def detect_gpu() -> bool:
    """Return True iff a CUDA device is exposed AND CatBoost was built
    with GPU support."""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=3)
        if out.returncode != 0 or not out.stdout.strip():
            return False
    except Exception:
        return False
    # CatBoost GPU is wheel-dependent; probe a tiny train
    try:
        m = cb.CatBoostClassifier(iterations=1, task_type="GPU",
                                  devices="0", verbose=0)
        m.fit([[0.0], [1.0]], [0, 1])
        return True
    except Exception:
        return False


def cb_params(use_gpu: bool, max_iters: int, seed: int, depth: int = 10) -> dict:
    """Research-backed CatBoost recipe (single source of truth).

    Synthesis of:
      - audit/2026-05-04-catboost-research.md lever map
      - audit/2026-05-04-irrigation-water-postmortem.md (LEARNINGS.md)
      - CatBoost official param-tuning docs (catboost.ai/docs/.../parameter-tuning)
      - Garkavenko, "Categorical features parameters in CatBoost"

    Notable research-driven choices:
      - DON'T set `simple_ctr` / `combinations_ctr` — defaults
        (Borders+Counter, Uniform target borders) are documented best
        for BINARY classification. "Useless to increase TargetBorderCount
        for binary class" (Garkavenko).
      - DON'T set `max_ctr_complexity=6` — 6.4× model size for marginal
        lift (Garkavenko). Stick with default 4.
      - Bernoulli + subsample=0.8 (irrigation-water proven; works
        identically on CPU/GPU, no `mvs_reg` GPU-strip dance).
      - rsm=0.8 column subsampling — never tested on s6e5; irrigation
        included this in their 0.98150 PRIMARY recipe.
      - min_data_in_leaf=20 — bigger than irrigation's 2 (s6e5 has 351k
        train rows; min_data_in_leaf=2 too aggressive at this scale).
      - border_count=254 on GPU per docs ("set 254 for GPU if best
        possible quality is required").
      - depth=10 default per zeta fold-0 0.94992 proof + docs 6-10 range.
    """
    p = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        # slow + wide (M3 hit iter-cap on 4/5 folds at 800 iters)
        iterations=max_iters,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=8.0,
        # categorical handling — CatBoost defaults handle Borders+Counter
        # correctly for binary class. Just set OHE threshold high enough
        # to one-hot Compound(5) + Year(4) + Stint(<=5).
        one_hot_max_size=10,
        # row + column subsampling (Bernoulli is GPU/CPU symmetric)
        bootstrap_type="Bernoulli",
        subsample=0.8,
        rsm=0.8,
        # leaf regularization
        min_data_in_leaf=20,
        # ES
        od_type="Iter",
        od_wait=200,
        # plumbing
        random_seed=seed,
        verbose=200,
        allow_writing_files=False,
    )
    if use_gpu:
        p["task_type"] = "GPU"
        p["devices"] = "0:1"
        p["border_count"] = 254  # docs: "set 254 for GPU max quality"
    else:
        p["task_type"] = "CPU"
        p["thread_count"] = -1
        # CPU default border_count=128 is fine.
    return p


def _key_fn(cols):
    def _k(df):
        s = df[cols[0]].fillna("MISSING").astype(str)
        for c in cols[1:]:
            s = s + "__" + df[c].fillna("MISSING").astype(str)
        return s.reset_index(drop=True)
    return _k


def fold_safe_te_for_fold(train_ti, train_va, test_fold, y_ti, fold, n_folds):
    """Compute Rozen's 6 CV TE columns for one outer fold, fold-safely.
    Adds te_* columns in-place to all three frames."""
    inner_skf = StratifiedKFold(n_folds, shuffle=True,
                                random_state=SEED + fold)
    inner_folds = list(inner_skf.split(np.zeros(len(y_ti)), y_ti))
    for cols, smooth, te_name in TE_CONFIGS:
        if not all(c in train_ti.columns for c in cols):
            continue
        ti_enc, _ = cv_target_encode(
            train_ti, train_va, cols, y_ti, inner_folds, smoothing=smooth)
        train_ti[te_name] = ti_enc
        # va + test: use FULL ti stats (only ti labels — no va leakage)
        kfn = _key_fn(cols)
        gm = float(y_ti.mean())
        k_ti = kfn(train_ti)
        stats = (pd.DataFrame({"key": k_ti.values, "target": y_ti.values})
                 .groupby("key")["target"].agg(["sum", "count"]))
        stats["enc"] = ((stats["sum"] + smooth * gm)
                        / (stats["count"] + smooth))
        m = stats["enc"].to_dict()
        train_va[te_name] = kfn(train_va).map(m).fillna(gm).values
        test_fold[te_name] = kfn(test_fold).map(m).fillna(gm).values


def load_base_oofs(orig_train_ids: np.ndarray, orig_test_ids: np.ndarray,
                   want_knn_raw: bool):
    """Load precomputed orthogonal-axis base OOFs from scripts/artifacts/.
    Returns (train_extras_df, test_extras_df). Skips bases not on disk."""
    train_extras = pd.DataFrame(index=orig_train_ids)
    test_extras  = pd.DataFrame(index=orig_test_ids)
    found = []
    for oof_name, test_name, feat_name in BASE_OOFS:
        oof_path  = ART / f"{oof_name}.npy"
        test_path = ART / f"{test_name}.npy"
        if not (oof_path.exists() and test_path.exists()):
            continue
        oof  = np.load(oof_path)
        tst  = np.load(test_path)
        # 2-col [1-p, p] vs 1-col p
        oof_p = oof[:, 1] if oof.ndim == 2 else oof
        tst_p = tst[:, 1] if tst.ndim == 2 else tst
        # OOFs are saved aligned to the original train.csv id order
        # (per p1_single_lgbm_v3.py write path).
        train_extras[feat_name] = oof_p
        test_extras[feat_name]  = tst_p
        found.append(feat_name)
    if want_knn_raw:
        kx_tr = ART / "d15d_knn_X_train.npy"
        kx_te = ART / "d15d_knn_X_test.npy"
        if kx_tr.exists() and kx_te.exists():
            arr_tr = np.load(kx_tr)
            arr_te = np.load(kx_te)
            for j in range(arr_tr.shape[1]):
                train_extras[f"knn_d{j}"] = arr_tr[:, j]
                test_extras[f"knn_d{j}"]  = arr_te[:, j]
            found.append(f"knn_d0..{arr_tr.shape[1]-1}")
    return train_extras, test_extras, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="p1_single_cb_v3_feA_te")
    ap.add_argument("--max-rounds", type=int, default=8000)
    ap.add_argument("--no-te", action="store_true",
                    help="Drop CV TE features for ablation")
    ap.add_argument("--with-base-oofs", action="store_true",
                    help="Inject 6 orthogonal-axis base OOFs as features")
    ap.add_argument("--with-knn", action="store_true",
                    help="Inject d15d 10-d KNN distance features")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="Bag this many CB seeds (predictions averaged)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold + 50k row subsample for sanity check")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--depth", type=int, default=10,
                    help="CB depth (docs recommend 6-10; zeta proved 10 on s6e5)")
    ap.add_argument("--with-orig-data", action="store_true",
                    help="Append aadigupta1601 original rows to train with "
                         "row weight 0.5 (irrigation-water synthetic-DGP trick)")
    args = ap.parse_args()

    use_gpu = (not args.force_cpu) and detect_gpu()
    print(f"=== P1 single-CB v3 | name={args.name} | "
          f"GPU={use_gpu} | seeds={args.n_seeds} | smoke={args.smoke} ===")
    t0_total = time.time()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")
    print(f"  train {train.shape}  test {test.shape}")

    if args.smoke:
        train = train.sample(50_000, random_state=SEED).reset_index(drop=True)
        print(f"  SMOKE: subsampled train → {train.shape}")

    # --- 1) Static features (label-independent) ---
    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    # --- 2) Determine canonical feature list via a sample fold-1 ---
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    sample_ti = fold_list[0][0]
    sample_fs_a = fit_fs_a(train_S.iloc[sample_ti])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    feats, cat_cols = feature_columns_for_lgbm(sample_train)

    # CatBoost lever #1: Year ∈ cat_features. Stint also benefits.
    # Add to cat_cols if present and not already there.
    for c in ("Year", "Stint"):
        if c in feats and c not in cat_cols:
            cat_cols.append(c)
    if not args.no_te:
        feats = feats + [n for _, _, n in TE_CONFIGS]

    # --- 3) Optionally load base-OOF / KNN extras (aligned to orig ids) ---
    extras_train_idx = test_idx = None
    extras_train = extras_test = None
    extras_found: list[str] = []
    if args.with_base_oofs or args.with_knn:
        orig_tr_ids = pd.read_csv("data/train.csv", usecols=[ID_COL])[ID_COL].values
        orig_te_ids = pd.read_csv("data/test.csv",  usecols=[ID_COL])[ID_COL].values
        extras_train, extras_test, extras_found = load_base_oofs(
            orig_tr_ids, orig_te_ids, want_knn_raw=args.with_knn)
        # Restrict to currently-loaded train rows (smoke or full)
        kept = train[ID_COL].values
        extras_train = extras_train.loc[kept].reset_index(drop=True)
        feats = feats + list(extras_train.columns)
        print(f"  + extras: {extras_found}  total feats {len(feats)}")
    print(f"  feats: {len(feats)}  cat: {len(cat_cols)}")

    # --- 4) 5-fold loop with per-fold FS_A, per-fold CV TE, GBDT-on-extras ---
    n_train, n_test = len(y), len(test_S)
    oof = np.zeros(n_train, dtype=np.float64)
    test_pred = np.zeros(n_test, dtype=np.float64)
    fold_aucs, walls, iters_per_fold = [], [], []
    n_eff_folds = 1 if args.smoke else N_FOLDS
    seeds = [SEED + i * 31 for i in range(args.n_seeds)]

    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
        t0 = time.time()
        print(f"\n  --- Fold {fold} | ti={len(ti)} va={len(vi)} ---")

        # 1. fit FS_A on ti rows ONLY
        fs_a = fit_fs_a(train_S.iloc[ti])

        # 2. apply FS_A
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)

        # 3. CV TE
        if not args.no_te:
            y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
            fold_safe_te_for_fold(train_ti, train_va, test_fold,
                                  y_ti, fold, N_FOLDS)

        # 4. inject base-OOF / KNN extras (aligned by original id)
        if extras_train is not None:
            id_to_extras_pos = {tid: i for i, tid in
                                enumerate(train[ID_COL].values)}
            tr_ids = train_S.iloc[ti][ID_COL].values
            va_ids = train_S.iloc[vi][ID_COL].values
            tr_pos = np.array([id_to_extras_pos[t] for t in tr_ids])
            va_pos = np.array([id_to_extras_pos[t] for t in va_ids])
            for c in extras_train.columns:
                train_ti[c] = extras_train[c].values[tr_pos]
                train_va[c] = extras_train[c].values[va_pos]
                test_fold[c] = extras_test[c].values  # test row-aligned
        # 5. assemble matrices
        X_tr = train_ti.reindex(columns=feats, fill_value=0).copy()
        X_va = train_va.reindex(columns=feats, fill_value=0).copy()
        X_te = test_fold.reindex(columns=feats, fill_value=0).copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("int32")
            X_va[c] = X_va[c].astype("int32")
            X_te[c] = X_te[c].astype("int32")
        num_cols = [c for c in feats if c not in cat_cols]
        for X in [X_tr, X_va, X_te]:
            X[num_cols] = X[num_cols].fillna(0).astype(np.float32)
        cat_idx = [feats.index(c) for c in cat_cols]

        # 5b. Optionally append aadigupta1601 original rows with weight
        # 0.5 (irrigation-water synthetic-DGP trick: train sees both
        # the host's synthesizer output AND the underlying real DGP).
        # 97.55% of synth LapTime values exist in original (d15 KS-div
        # diagnostic) — the synth corrupted joint structure but kept
        # marginals, so original rows give CB clean per-row signal.
        train_ti_y = train_ti[TARGET].astype(int).values
        ti_weights = np.ones(len(X_tr), dtype=np.float32)
        if args.with_orig_data:
            orig_path = Path("external/aadigupta_orig/f1_strategy_dataset_v4.csv")
            if orig_path.exists():
                orig_raw = pd.read_csv(orig_path)
                # drop columns synth doesn't have to keep schema aligned
                drop_extra = [c for c in
                              ("Normalized_TyreLife", "Position_Change")
                              if c in orig_raw.columns]
                orig_raw = orig_raw.drop(columns=drop_extra)
                # synth has 'id' col, orig doesn't — synthesize ids past
                # max(train_id) so they don't collide with real ids
                next_id = int(train["id"].max()) + 1
                orig_raw["id"] = np.arange(next_id, next_id + len(orig_raw))
                # full FE chain using ti-fitted state + fold's fs_a
                orig_S, _ = make_features_static(orig_raw, fit=False, state=state)
                orig_FS = apply_fs_a(orig_S, fs_a)
                # CV TE: use full-ti stats (same as va/test branch)
                if not args.no_te:
                    for cols, smooth, te_name in TE_CONFIGS:
                        if not all(c in orig_FS.columns for c in cols):
                            continue
                        kfn = _key_fn(cols)
                        gm = float(y_ti.mean())
                        k_ti = kfn(train_ti)
                        stats = (pd.DataFrame({"key": k_ti.values,
                                               "target": y_ti.values})
                                 .groupby("key")["target"].agg(["sum", "count"]))
                        stats["enc"] = ((stats["sum"] + smooth * gm)
                                        / (stats["count"] + smooth))
                        m = stats["enc"].to_dict()
                        orig_FS[te_name] = kfn(orig_FS).map(m).fillna(gm).values
                # base-OOF / KNN extras: orig rows don't have these — fill
                # with marginal mean (CB will treat as constant; doesn't help
                # but doesn't hurt). Skip if no extras requested.
                if extras_train is not None:
                    for c in extras_train.columns:
                        orig_FS[c] = float(extras_train[c].mean())
                X_orig = orig_FS.reindex(columns=feats, fill_value=0).copy()
                for c in cat_cols:
                    X_orig[c] = X_orig[c].astype("int32")
                num_cols_local = [c for c in feats if c not in cat_cols]
                X_orig[num_cols_local] = (X_orig[num_cols_local].fillna(0)
                                          .astype(np.float32))
                y_orig = orig_FS[TARGET].astype(int).values
                X_tr = pd.concat([X_tr, X_orig], ignore_index=True)
                train_ti_y = np.concatenate([train_ti_y, y_orig])
                ti_weights = np.concatenate([
                    ti_weights,
                    np.full(len(X_orig), 0.5, dtype=np.float32),
                ])
                print(f"    + appended {len(X_orig)} orig rows "
                      f"(weight=0.5; pos rate {y_orig.mean():.4f})")
            else:
                print(f"    !! --with-orig-data set but {orig_path} missing; skip")

        # 6. CatBoost (n_seeds bag)
        oof_va_seed = np.zeros(len(vi), dtype=np.float64)
        test_seed = np.zeros(len(test_S), dtype=np.float64)
        seed_iters = []
        for s in seeds:
            params = cb_params(use_gpu, args.max_rounds, s, depth=args.depth)
            m = cb.CatBoostClassifier(**params)
            fit_kw = dict(eval_set=(X_va, train_va[TARGET].astype(int)),
                          cat_features=cat_idx, use_best_model=True)
            if args.with_orig_data:
                fit_kw["sample_weight"] = ti_weights
            m.fit(X_tr, train_ti_y, **fit_kw)
            oof_va_seed += m.predict_proba(X_va)[:, 1] / len(seeds)
            test_seed += m.predict_proba(X_te)[:, 1] / len(seeds)
            seed_iters.append(int(m.tree_count_))
        oof_va = oof_va_seed
        sorted_vi = train_S.iloc[vi].index.values
        oof[sorted_vi] = oof_va
        test_pred += test_seed / n_eff_folds

        fold_aucs.append(float(roc_auc_score(
            train_va[TARGET].astype(int).values, oof_va)))
        walls.append(time.time() - t0)
        iters_per_fold.append(seed_iters)
        print(f"    Fold {fold}: AUC={fold_aucs[-1]:.5f}  "
              f"iters={seed_iters}  wall={walls[-1]:.1f}s")

    if args.smoke:
        # In smoke mode we only filled fold-1 OOF rows; report fold AUC.
        print(f"\n  SMOKE fold-1 AUC: {fold_aucs[0]:.5f}  "
              f"total wall={time.time()-t0_total:.1f}s")
        # Estimate 5-fold wall projection
        proj = walls[0] * N_FOLDS / (50_000 / 439_140)
        print(f"  Projected 5-fold wall on full data: ~{proj/60:.1f} min")
        return

    auc_full = float(roc_auc_score(y, oof))
    print(f"\n  OOF AUC (full): {auc_full:.5f}  "
          f"fold-std={np.std(fold_aucs):.5f}  "
          f"total wall={time.time()-t0_total:.1f}s")

    # Map back to original train.csv id order (train_S is sorted)
    order = train_S["id"].values
    sort_back = np.argsort(order)
    oof_aligned = oof[sort_back]
    order_te = test_S["id"].values
    id_to_pos = {tid: i for i, tid in enumerate(order_te)}
    orig_te = pd.read_csv("data/test.csv", usecols=[ID_COL])[ID_COL].values
    test_aligned = np.array([test_pred[id_to_pos[t]] for t in orig_te])

    np.save(ART / f"oof_{args.name}_strat.npy",
            np.column_stack([1 - oof_aligned, oof_aligned]).astype(np.float64))
    np.save(ART / f"test_{args.name}_strat.npy",
            np.column_stack([1 - test_aligned, test_aligned]).astype(np.float64))
    sub_out = sub[[ID_COL]].copy()
    sub_out[TARGET] = np.clip(test_aligned, 0.001, 0.999)
    Path("submissions").mkdir(exist_ok=True)
    sub_out.to_csv(f"submissions/submission_{args.name}.csv", index=False)
    (ART / f"{args.name}_results.json").write_text(json.dumps(dict(
        name=args.name,
        no_te=args.no_te, with_base_oofs=args.with_base_oofs,
        with_knn=args.with_knn, n_seeds=args.n_seeds, use_gpu=use_gpu,
        oof_auc_full=auc_full, fold_aucs=fold_aucs,
        fold_iters=iters_per_fold, fold_walls=walls,
        n_feats=len(feats), n_cat=len(cat_cols),
        extras_found=extras_found,
    ), indent=2, default=str))
    print(f"  → oof_{args.name}_strat.npy   test_..._strat.npy   "
          f"submission_..._strat.csv")


if __name__ == "__main__":
    main()
