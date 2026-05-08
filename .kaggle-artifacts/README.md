# .kaggle-artifacts/ — staging dir for the Kaggle Dataset

Mirror of `scripts/artifacts/` for upload to a private Kaggle Dataset.
Two-step ritual replaces git-tracked binaries.

## One-time setup (you run this; needs `~/.kaggle/kaggle.json`)

```bash
# 1. Replace USERNAME in dataset-metadata.json with your Kaggle handle
#    (look in ~/.kaggle/kaggle.json -> "username" field).
sed -i "s|USERNAME/s6e5-artifacts|<your-username>/s6e5-artifacts|" \
    .kaggle-artifacts/dataset-metadata.json

# 2. Hard-link current artifacts into the staging dir (no disk doubling)
cd .kaggle-artifacts
find ../scripts/artifacts -maxdepth 1 -type f \
     \( -name "*.npy" -o -name "*.json" \) -exec ln -f {} . \;

# 3. Initial dataset push (~675 MB)
kaggle datasets create -p . --dir-mode zip
# Verify on https://www.kaggle.com/datasets/<your-username>/s6e5-artifacts
```

## Daily workflow (after each session)

```bash
cd .kaggle-artifacts
find ../scripts/artifacts -maxdepth 1 -type f \
     \( -name "*.npy" -o -name "*.json" \) -exec ln -f {} . \;
kaggle datasets version -p . \
    -m "$(date +%Y-%m-%d) $(git branch --show-current) +<probe-name>" \
    --dir-mode zip
```

## After successful initial push: stop git-tracking the originals

Run ONLY after you've verified the dataset uploaded and is downloadable.
This is the irreversible-on-remote step (history retains binaries on
origin; force-push not required since deletion is a normal commit).

```bash
git rm -r --cached scripts/artifacts/ submissions/
cat >> .gitignore <<'EOF'

# Artifacts now live in private Kaggle Dataset s6e5-artifacts (2026-05-08)
scripts/artifacts/
submissions/
!scripts/artifacts/.gitkeep
!submissions/.gitkeep
EOF
mkdir -p scripts/artifacts submissions
touch scripts/artifacts/.gitkeep submissions/.gitkeep
git add .gitignore scripts/artifacts/.gitkeep submissions/.gitkeep
git commit -m "Move scripts/artifacts/ + submissions/ to Kaggle Dataset"
```

## Loading artifacts in code

`scripts/common.py::artifact_path()` resolves the right location:
- Locally: `scripts/artifacts/oof_X_strat.npy` (downloaded via
  `kaggle datasets download` or hard-linked from `.kaggle-artifacts/`)
- On Kaggle notebooks: `/kaggle/input/s6e5-artifacts/oof_X_strat.npy`
  (after `Add Data → s6e5-artifacts`)

## What's NOT in the dataset

- `data/train.csv`, `data/test.csv` — competition data; redistribution
  forbidden by TOS. Stays in `data/` (gitignored).
- `kernels/` GPU notebook scaffolds — keep in repo.
- `audit/`, `*.md` — prose; stays in git.
