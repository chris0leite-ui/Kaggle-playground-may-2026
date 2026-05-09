#!/usr/bin/env bash
# bootstrap.sh — set up a fresh container or laptop for Orbit Wars.
# Reads COMP from .comp.env (defaults to "orbit-wars" if absent).
#
# What it does:
#   1. Source .comp.env (COMP).
#   2. Resolve Kaggle credentials (three accepted forms).
#   3. Create ~/.kaggle/kaggle.json from KAGGLE_USERNAME + KAGGLE_KEY if needed.
#   4. Install Python requirements.
#   5. Download competition data into data/ if not present.
#   6. (Optional) Pull reference notebooks into external/kernels/.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Step 0 — load per-comp config
# ---------------------------------------------------------------------------
if [[ -f .comp.env ]]; then
    # shellcheck disable=SC1091
    source .comp.env
fi

# Default for Orbit Wars; .comp.env can override.
COMP="${COMP:-orbit-wars}"

echo "--- comp: $COMP"

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
    echo "ERROR: no Kaggle credentials found."
    echo "  Set up ~/.kaggle/kaggle.json (mode 600) — see SETUP.md."
    exit 2
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
if compgen -G "data/*" > /dev/null 2>&1 && [[ "$(ls -A data 2>/dev/null | grep -v '^\.gitkeep$' | wc -l)" -gt 0 ]]; then
    echo "--- data: already present, skipping download ---"
else
    echo "--- data: downloading $COMP ---"
    kaggle competitions download -c "$COMP" -p data/ || {
        echo "WARNING: kaggle competitions download failed."
        echo "  Possible causes: comp slug wrong, you haven't accepted the rules"
        echo "  on the comp page yet, or your token doesn't have permission."
    }
    if [[ -f "data/${COMP}.zip" ]]; then
        unzip -qo "data/${COMP}.zip" -d data/
        rm -f "data/${COMP}.zip"
    fi
fi
ls -lh data/ 2>/dev/null | head -10

# ---------------------------------------------------------------------------
# Step 4 — reference notebooks (code-comp aid; optional)
# ---------------------------------------------------------------------------
# Day-1 agent fills exact slugs after running:
#   kaggle kernels list -s orbit-wars --sort-by voteCount
# and pulling the top 3 with:
#   kaggle kernels pull <user>/<slug> -p external/kernels/<slug>/
# Skipping by default — see SETUP.md Step 3.

echo "--- bootstrap done ---"
