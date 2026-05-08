#!/usr/bin/env bash
# bootstrap.sh — set up a fresh container or laptop for this competition.
#
# What it does, in order:
#   1. Resolve Kaggle credentials from the environment (any of the three
#      common forms).
#   2. Create ~/.kaggle/kaggle.json if KAGGLE_USERNAME + KAGGLE_KEY are
#      available — this makes `kaggle datasets create/version` work
#      everywhere, not just on the harness's patched CLI.
#   3. Install Python requirements.
#   4. Download competition data into data/ if not already present.
#   5. Pull the artifact dataset (private OOF / test predictions) into
#      scripts/artifacts/ if it's set up and the local cache is empty.
#
# Run from anywhere: `bash bootstrap.sh`.

set -euo pipefail
cd "$(dirname "$0")"

COMP="playground-series-s6e5"
ARTIFACT_DATASET="${ARTIFACT_DATASET:-chrisleitescha/s6e5-artifacts}"

# ---------------------------------------------------------------------------
# Step 1 — resolve Kaggle credentials
# ---------------------------------------------------------------------------
# Three accepted forms, checked in order:
#   (a) ~/.kaggle/kaggle.json already exists — done.
#   (b) KAGGLE_USERNAME + KAGGLE_KEY env vars (standard).
#   (c) KAGGLE_API_TOKEN env var (the harness's patched CLI form, or the
#       KGAT_-prefixed sandbox secret). Falls back from KAGGLE_KEY alone.

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
    echo "    (note: 'kaggle datasets create/version' commands need a"
    echo "     standard ~/.kaggle/kaggle.json on a non-harness machine)"
else
    echo "--- credentials: none found; prompting ---"
    read -rsp "Kaggle API token (KGAT_... or your kaggle.json key): " KAGGLE_API_TOKEN
    echo
    export KAGGLE_API_TOKEN
fi

# ---------------------------------------------------------------------------
# Step 2 — install Python requirements
# ---------------------------------------------------------------------------
echo "--- installing requirements ---"
pip install -q -r requirements.txt

# ---------------------------------------------------------------------------
# Step 3 — competition data
# ---------------------------------------------------------------------------
if [[ -f data/train.csv && -f data/test.csv ]]; then
    echo "--- data: already present, skipping download ---"
    ls -lh data/*.csv
else
    echo "--- data: downloading $COMP ---"
    kaggle competitions download -c "$COMP" -p data/
    unzip -qo "data/${COMP}.zip" -d data/
    rm -f "data/${COMP}.zip"
    ls -lh data/*.csv
fi

# ---------------------------------------------------------------------------
# Step 4 — artifact dataset (OOF / test predictions for stacking)
# ---------------------------------------------------------------------------
ARTIFACT_DIR="scripts/artifacts"
mkdir -p "$ARTIFACT_DIR"

# Count files excluding the .gitkeep placeholder.
ARTIFACT_COUNT=$(find "$ARTIFACT_DIR" -maxdepth 1 -type f \
    \( -name '*.npy' -o -name '*.json' \) 2>/dev/null | wc -l)

if [[ "$ARTIFACT_COUNT" -gt 0 ]]; then
    echo "--- artifacts: $ARTIFACT_COUNT files in $ARTIFACT_DIR, skipping download ---"
elif kaggle datasets list -m | grep -q "$ARTIFACT_DATASET" 2>/dev/null; then
    echo "--- artifacts: pulling $ARTIFACT_DATASET into $ARTIFACT_DIR ---"
    kaggle datasets download "$ARTIFACT_DATASET" -p "$ARTIFACT_DIR" --unzip
    echo "    pulled $(find "$ARTIFACT_DIR" -maxdepth 1 -type f | wc -l) files."
else
    echo "--- artifacts: $ARTIFACT_DATASET not found on Kaggle ---"
    echo "    Run scripts/setup_artifact_dataset.sh to scaffold and push the"
    echo "    initial dataset (first-time setup only)."
fi

echo "--- bootstrap done ---"
