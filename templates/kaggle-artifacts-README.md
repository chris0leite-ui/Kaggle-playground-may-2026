# .kaggle-artifacts/ — staging dir for the Kaggle Dataset

Mirror of `scripts/artifacts/` for upload to a private Kaggle Dataset.
Replaces git-tracked binaries.

## One-time setup

```bash
bash scripts/setup_artifact_dataset.sh \
    --username <your-kaggle-username> \
    --slug <comp-slug>-artifacts
```

That script edits `dataset-metadata.json`, hard-links artifacts in,
and pushes the dataset.

## Daily versioning ritual (after each session)

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

```bash
git rm -r --cached scripts/artifacts/ submissions/
mkdir -p scripts/artifacts submissions
touch scripts/artifacts/.gitkeep submissions/.gitkeep
git add .gitignore scripts/artifacts/.gitkeep submissions/.gitkeep
git commit -m "Migrate artifacts to private Kaggle Dataset"
```

## Loading artifacts in code

`scripts/common.py::ART` resolves the right location automatically:

- **Locally:** `scripts/artifacts/`.
- **On Kaggle notebooks:** `/kaggle/input/<slug>-artifacts/` (after
  Add Data → \<your-username\>/\<slug\>-artifacts in the notebook editor).

Existing scripts use `from common import ART; np.load(ART / "oof_X.npy")`.
The resolver picks the right path; you don't change your code.

## What's NOT in the dataset

- `data/train.csv`, `data/test.csv` — competition data; redistribution
  forbidden by Kaggle TOS. Stays in `data/` (gitignored).
- `kernels/` GPU notebook scaffolds — keep in repo.
- `audit/`, `*.md` — prose; stays in git.
