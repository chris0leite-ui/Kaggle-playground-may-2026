#!/usr/bin/env bash
# bootstrap.sh — set up a fresh container or laptop for this competition.
# Slug-agnostic; reads COMP and ARTIFACT_DATASET from .comp.env.
#
# What it does:
#   1. Source .comp.env (COMP, ARTIFACT_DATASET).
#   2. Resolve Kaggle credentials (three accepted forms).
#   3. Create ~/.kaggle/kaggle.json from KAGGLE_USERNAME + KAGGLE_KEY if needed.
#   4. Install Python requirements.
#   5. Download competition data into data/ if not present.
#   6. Pull the artifact dataset into scripts/artifacts/ if it exists.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Step 0 — load per-comp config
# ---------------------------------------------------------------------------
if [[ -f .comp.env ]]; then
    # shellcheck disable=SC1091
    source .comp.env
elif [[ -f .comp.env.template ]]; then
    echo "ERROR: .comp.env not found. Copy .comp.env.template, edit, and try again."
    echo "  cp .comp.env.template .comp.env && \$EDITOR .comp.env"
    exit 2
else
    echo "ERROR: no .comp.env or .comp.env.template found in $(pwd)"
    exit 2
fi

: "${COMP:?COMP must be set in .comp.env}"
: "${ARTIFACT_DATASET:?ARTIFACT_DATASET must be set in .comp.env}"

echo "--- comp: $COMP"
echo "--- artifacts: $ARTIFACT_DATASET"

# ---------------------------------------------------------------------------
# Step 1 — credentials
# ---------------------------------------------------------------------------
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"

if [[ -z "${KAGGLE_API_TOKEN:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    export KAGGLE_API_TOKEN="$KAGGLE_KEY"
fi

if [[ -f "$KAGGLE_JSON" ]]; then
    echo "--- credentials: $KAGGLE_JSON found, using it ---"
elif [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    echo "--- credentials: writing $KAGGLE_JSON from KAGGLE_USERNAME + KAGGLE_KEY ---"
    mkdir -p "$(dirname "$KAGGLE_JSON")"
    umask 077
    printf '{"username":"%s","key":"%s"}\n' \
        "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$KAGGLE_JSON"
    chmod 600 "$KAGGLE_JSON"
elif [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "--- credentials: KAGGLE_API_TOKEN set; works with the harness CLI ---"
else
    echo "--- credentials: none found; prompting ---"
    read -rsp "Kaggle API token: " KAGGLE_API_TOKEN
    echo
    export KAGGLE_API_TOKEN
fi

# ---------------------------------------------------------------------------
# Step 2 — Python deps
# ---------------------------------------------------------------------------
if [[ -f requirements.txt ]]; then
    echo "--- installing requirements ---"
    pip install -q -r requirements.txt
else
    echo "--- (no requirements.txt; skipping pip install) ---"
fi

# ---------------------------------------------------------------------------
# Step 3 — competition data
# ---------------------------------------------------------------------------
mkdir -p data
if [[ -f data/train.csv && -f data/test.csv ]]; then
    echo "--- data: already present, skipping download ---"
else
    echo "--- data: downloading $COMP ---"
    kaggle competitions download -c "$COMP" -p data/
    unzip -qo "data/${COMP}.zip" -d data/
    rm -f "data/${COMP}.zip"
fi
ls -lh data/*.csv 2>/dev/null | head -3

# ---------------------------------------------------------------------------
# Step 4 — artifact dataset (OOF / test predictions for stacking)
# ---------------------------------------------------------------------------
ARTIFACT_DIR="scripts/artifacts"
mkdir -p "$ARTIFACT_DIR"

ARTIFACT_COUNT=$(find "$ARTIFACT_DIR" -maxdepth 1 -type f \
    \( -name '*.npy' -o -name '*.json' \) 2>/dev/null | wc -l)

if [[ "$ARTIFACT_COUNT" -gt 0 ]]; then
    echo "--- artifacts: $ARTIFACT_COUNT files in $ARTIFACT_DIR, skipping pull ---"
elif kaggle datasets list -m 2>/dev/null | grep -q "$ARTIFACT_DATASET"; then
    echo "--- artifacts: pulling $ARTIFACT_DATASET ---"
    kaggle datasets download "$ARTIFACT_DATASET" -p "$ARTIFACT_DIR" --unzip
else
    echo "--- artifacts: $ARTIFACT_DATASET not on Kaggle yet ---"
    echo "    Run scripts/setup_artifact_dataset.sh after your first OOF lands."
fi

echo "--- bootstrap done ---"
