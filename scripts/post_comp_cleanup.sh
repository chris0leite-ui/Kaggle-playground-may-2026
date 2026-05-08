#!/usr/bin/env bash
# scripts/post_comp_cleanup.sh — release artifacts + free local disk after comp
#
# Run this after the playground-series-s6e5 final-window selection, when
# you no longer need round-trip access to the OOF / test artifacts on
# Kaggle and locally.
#
# WHAT IT DOES (in order, with prompts):
#   1. Confirm comp is over.
#   2. Tag the final state of the Kaggle Dataset (creates an immutable
#      "final" version note for archaeology).
#   3. Optionally delete the Kaggle Dataset entirely (frees the 100 GB
#      private quota slot).
#   4. Clear local working-tree caches: scripts/artifacts/*, submissions/*,
#      .kaggle-artifacts/* (hardlinks).
#   5. Run git gc to reclaim any local pack waste.
#
# Run with --dry-run to preview, --execute to actually do it.

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="chrisleitescha/s6e5-artifacts"
COMP="playground-series-s6e5"
EXECUTE=0
DELETE_DATASET=0

for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE=1 ;;
        --delete-dataset) DELETE_DATASET=1 ;;
        --dry-run|"") EXECUTE=0 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

echo "=== post-comp cleanup ==="
echo "Dataset: $DATASET"
echo "Comp:    $COMP"
echo "Execute: $EXECUTE  (dry-run otherwise)"
echo "Delete:  $DELETE_DATASET (only if --delete-dataset)"
echo

# Step 1 — confirm
if [ "$EXECUTE" -eq 1 ]; then
    read -rp "Comp $COMP final-window selection complete? [y/N] " yn
    [[ "$yn" =~ ^[Yy]$ ]] || { echo "abort"; exit 1; }
fi

# Step 2 — tag final version on Kaggle Dataset (idempotent metadata-only push)
echo "--- Step 2: tag final dataset version ---"
if [ "$EXECUTE" -eq 1 ]; then
    cd .kaggle-artifacts
    # No new files; just bump version with "final" note for archaeology.
    kaggle datasets version -p . -m "post-comp final ($(date +%Y-%m-%d))" \
        --dir-mode zip || echo "  (skip: no changes since last version)"
    cd ..
else
    echo "  would: kaggle datasets version -p .kaggle-artifacts -m 'post-comp final ...'"
fi

# Step 3 — optionally delete the dataset
if [ "$DELETE_DATASET" -eq 1 ]; then
    echo "--- Step 3: delete Kaggle Dataset (frees private quota slot) ---"
    if [ "$EXECUTE" -eq 1 ]; then
        read -rp "Really delete $DATASET ? This cannot be undone. [y/N] " yn
        [[ "$yn" =~ ^[Yy]$ ]] || { echo "skip delete"; DELETE_DATASET=0; }
    fi
    if [ "$EXECUTE" -eq 1 ] && [ "$DELETE_DATASET" -eq 1 ]; then
        kaggle datasets delete -d "$DATASET" -y || echo "  (delete may need manual confirm at kaggle.com)"
    else
        echo "  would: kaggle datasets delete -d $DATASET"
    fi
fi

# Step 4 — clear local caches
echo "--- Step 4: clear local caches ---"
SIZE_BEFORE=$(du -sb scripts/artifacts submissions .kaggle-artifacts 2>/dev/null | awk '{s+=$1} END {print int(s/1024/1024) "MB"}')
echo "  before: $SIZE_BEFORE"
if [ "$EXECUTE" -eq 1 ]; then
    find scripts/artifacts -type f ! -name '.gitkeep' -delete 2>/dev/null
    find submissions     -type f ! -name '.gitkeep' -delete 2>/dev/null
    find .kaggle-artifacts -type f \
         ! -name 'dataset-metadata.json' \
         ! -name 'README.md' \
         -delete 2>/dev/null
    SIZE_AFTER=$(du -sb scripts/artifacts submissions .kaggle-artifacts 2>/dev/null | awk '{s+=$1} END {print int(s/1024/1024) "MB"}')
    echo "  after:  $SIZE_AFTER"
else
    echo "  would: rm scripts/artifacts/* submissions/* .kaggle-artifacts/* (preserve .gitkeep + metadata)"
fi

# Step 5 — git gc (reclaim local pack waste)
echo "--- Step 5: git gc ---"
if [ "$EXECUTE" -eq 1 ]; then
    git gc --aggressive --prune=now
    du -sh .git
else
    echo "  would: git gc --aggressive --prune=now"
fi

echo
echo "=== done ==="
[ "$EXECUTE" -eq 0 ] && echo "(dry-run; pass --execute to actually run)"
