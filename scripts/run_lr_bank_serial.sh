#!/usr/bin/env bash
# Run remaining LR-bank variants one at a time in foreground.
# Each variant gets its own Python process; failure of one doesn't kill the loop.

set -u
cd "$(dirname "$0")/.."

VARIANTS=(
    "lr_C_low_kbins20"
    "lr_C_high_kbins20"
    "lr_balanced_kbins20"
    "lr_l1_lasso_kbins20"
    "lr_splines_5"
    "lr_splines_10"
    "lr_hash_2way_2k"
    "lr_hash_3way_8k"
)

for v in "${VARIANTS[@]}"; do
    if [[ -f "scripts/artifacts/oof_${v}_strat.npy" ]]; then
        echo "[skip] ${v} (exists)"
        continue
    fi
    echo "==== running ${v} ===="
    if timeout 900 python3 -u scripts/lr_bank.py --names "${v}" --skip-existing 2>&1; then
        echo "[ok] ${v}"
    else
        echo "[fail] ${v} (exit $?)"
    fi
done

echo ""
echo "==== running per-segment + derived ===="
timeout 1500 python3 -u scripts/lr_bank.py --names lr_raw_std --per-segment --derived --skip-existing 2>&1
