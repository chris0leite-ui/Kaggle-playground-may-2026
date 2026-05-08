"""Shared CV / OOF / metric utilities for a Kaggle tabular comp.

Generic across competitions. The kickoff agent edits CLASSES and TASK_TYPE
in `comp-context.md` and (if needed) updates this file's defaults.

Pinned conventions (matches every committed OOF / test artifact):
    SEED = 42
    N_FOLDS = 5
    StratifiedKFold(shuffle=True, random_state=SEED) for classification
    KFold(shuffle=True, random_state=SEED) for regression
    OOF shape: (n_train, n_class) for classification (rows sum to 1)
    test shape: (n_test, n_class) (averaged across folds, rows sum to 1)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

SEED = 42
N_FOLDS = 5


def _resolve_artifact_dir() -> Path:
    """Pick scripts/artifacts/ (local) or /kaggle/input/<slug>-artifacts/
    (Kaggle notebook with the dataset attached). Local wins if both exist.

    The Kaggle path slug comes from the ARTIFACT_DATASET env var (set by
    bootstrap.sh from .comp.env), or falls back to scanning /kaggle/input/.
    """
    local = Path("scripts/artifacts")
    if local.exists() and any(local.iterdir()):
        return local

    # Try the env-named slug first
    slug = os.environ.get("ARTIFACT_DATASET", "").split("/", 1)[-1]
    if slug:
        kaggle = Path(f"/kaggle/input/{slug}")
        if kaggle.exists():
            return kaggle

    # Fall back: scan /kaggle/input/ for any *-artifacts dir
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for child in kaggle_root.iterdir():
            if child.is_dir() and child.name.endswith("-artifacts"):
                return child

    local.mkdir(parents=True, exist_ok=True)
    return local


ART = _resolve_artifact_dir()


def folds(y, task: str = "classification"):
    """Yield (fold_idx, train_idx, val_idx) for the pinned 5-fold split."""
    if task == "classification":
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield k, tr, va
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (tr, va) in enumerate(kf.split(np.zeros(len(y)))):
            yield k, tr, va


# Add competition-specific helpers below this line as the project grows.
# Keep this file generic — comp-specific logic belongs in
# scripts/<comp-slug>/ or scripts/<probe>.py.
