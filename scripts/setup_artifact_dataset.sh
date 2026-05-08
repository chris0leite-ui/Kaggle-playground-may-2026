#!/usr/bin/env bash
# scripts/setup_artifact_dataset.sh — first-time setup of the private
# Kaggle Dataset that holds OOF and test-prediction artifacts.
#
# Run this once at the start of a new competition, after you have at
# least one model trained and at least one .npy artifact in
# scripts/artifacts/.
#
# It will:
#   1. Edit .kaggle-artifacts/dataset-metadata.json to use your Kaggle
#      username and the chosen dataset slug.
#   2. Hard-link every artifact in scripts/artifacts/ into
#      .kaggle-artifacts/ (no disk doubling).
#   3. Create the dataset on Kaggle (private, CC0).
#
# Usage:
#   bash scripts/setup_artifact_dataset.sh \
#       --username YOUR_KAGGLE_USERNAME \
#       --slug s6e5-artifacts
#
# After it runs, the daily ritual is documented in
# .kaggle-artifacts/README.md.

set -euo pipefail
cd "$(dirname "$0")/.."

USERNAME=""
SLUG=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --username) USERNAME="$2"; shift 2 ;;
        --slug)     SLUG="$2";     shift 2 ;;
        --dry-run)  DRY_RUN=1;     shift   ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

# Defaults if user didn't pass them but we can read kaggle.json
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [[ -z "$USERNAME" && -f "$KAGGLE_JSON" ]]; then
    USERNAME=$(python3 -c "import json,sys; print(json.load(open('$KAGGLE_JSON'))['username'])" 2>/dev/null || true)
fi

if [[ -z "$USERNAME" ]]; then
    echo "ERROR: --username required (or put it in ~/.kaggle/kaggle.json)"
    exit 2
fi

if [[ -z "$SLUG" ]]; then
    SLUG=$(grep -oE 'COMP="[^"]+"' bootstrap.sh | head -1 | sed -e 's/COMP="//' -e 's/"$//' -e 's/playground-series-//')
    if [[ -z "$SLUG" ]]; then
        echo "ERROR: --slug required (could not infer from bootstrap.sh)"
        exit 2
    fi
    SLUG="${SLUG}-artifacts"
    echo "(inferred slug: $SLUG)"
fi

DATASET_ID="$USERNAME/$SLUG"

echo "=== artifact-dataset setup ==="
echo "Dataset: $DATASET_ID"
echo "Dry-run: $DRY_RUN"
echo

# --- step 1: scaffold .kaggle-artifacts/ if missing ---
if [[ ! -d .kaggle-artifacts ]]; then
    echo "--- step 1: creating .kaggle-artifacts/ ---"
    mkdir -p .kaggle-artifacts
    cat > .kaggle-artifacts/dataset-metadata.json <<EOF
{
  "title": "$SLUG — OOF / test artifacts",
  "id": "$DATASET_ID",
  "subtitle": "Stacking pool predictions (private)",
  "description": "Derived numpy artifacts (OOF + test predictions) for the $SLUG project. Files map 1:1 onto scripts/artifacts/*.npy in the repo. Comp data NOT included per Kaggle TOS.",
  "isPrivate": true,
  "licenses": [{"name": "CC0-1.0"}],
  "keywords": ["kaggle-internal", "stacking", "oof"],
  "collaborators": [],
  "data": []
}
EOF
else
    echo "--- step 1: .kaggle-artifacts/ exists; updating dataset id ---"
    python3 -c "
import json, pathlib
p = pathlib.Path('.kaggle-artifacts/dataset-metadata.json')
d = json.loads(p.read_text())
d['id'] = '$DATASET_ID'
p.write_text(json.dumps(d, indent=2) + '\n')
"
fi

# --- step 2: hard-link artifacts ---
echo "--- step 2: hard-linking artifacts ---"
ARTIFACT_FILES=$(find scripts/artifacts -maxdepth 1 -type f \
    \( -name "*.npy" -o -name "*.json" \) 2>/dev/null | wc -l)

if [[ "$ARTIFACT_FILES" -eq 0 ]]; then
    echo "    no artifacts in scripts/artifacts/ yet — skipping link step."
    echo "    Run this script again once you have at least one .npy file."
    exit 0
fi

(
    cd .kaggle-artifacts
    find ../scripts/artifacts -maxdepth 1 -type f \
        \( -name "*.npy" -o -name "*.json" \) -exec ln -f {} . \;
)
LINKED=$(find .kaggle-artifacts -maxdepth 1 -type f \
    \( -name "*.npy" -o -name "*.json" \) | wc -l)
echo "    linked $LINKED files."

# --- step 3: push to Kaggle ---
echo "--- step 3: pushing to Kaggle ---"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    DRY RUN — would: kaggle datasets create -p .kaggle-artifacts --dir-mode zip"
    echo "    DRY RUN — would: verify download round-trip"
    exit 0
fi

(cd .kaggle-artifacts && kaggle datasets create -p . --dir-mode zip)

echo "--- done ---"
echo "Dataset is now at: https://www.kaggle.com/datasets/$DATASET_ID"
echo "Daily versioning ritual lives in .kaggle-artifacts/README.md."
