# SETUP — starting a new competition

This is the onboarding checklist for a fresh container, fresh laptop,
or a brand-new competition repo. **Read this once, top to bottom, on
day 1.** It composes `bootstrap.sh`, `.kaggle-artifacts/`, and
`scripts/common.py::ART` into a single flow.

For day-to-day rules and process, see `CLAUDE.md`. For session-start
read order on an *existing* competition, see `HANDOVER.md`.

## What you need before starting

1. **A Kaggle account** with API access enabled
   (https://www.kaggle.com/settings → "Create New Token" downloads
   `kaggle.json`).
2. **One of these credential forms** in your environment:
   - `~/.kaggle/kaggle.json` already in place (the standard path), OR
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` env vars (the bootstrap will
     turn these into a `kaggle.json` for you), OR
   - `KAGGLE_API_TOKEN` env var with a `KGAT_…` token (the harness
     form).
3. **Python 3.10+** with `pip`.
4. **The competition slug**, e.g. `playground-series-s6e5`.

## Day 1 — first 15 minutes

### Step 1. Clone and bootstrap

```bash
git clone <repo-url>
cd <repo>
bash bootstrap.sh
```

What `bootstrap.sh` does, in order:

1. Resolves Kaggle credentials. If `KAGGLE_USERNAME` + `KAGGLE_KEY` are
   set, it materialises `~/.kaggle/kaggle.json` so the standard
   `kaggle datasets …` commands work everywhere.
2. Installs Python requirements from `requirements.txt`.
3. Downloads the competition data into `data/` if not already present.
4. Pulls the artifact dataset into `scripts/artifacts/` if it exists.

If you're starting a *new* competition, edit `bootstrap.sh` and change
the `COMP="…"` line at the top to the new slug. The
`ARTIFACT_DATASET="…"` line should also change to
`<your-username>/<comp-slug>-artifacts`.

### Step 2. Confirm the data downloaded

```bash
ls -lh data/
# expect: train.csv, test.csv, sample_submission.csv
```

If the download fails with an authentication error, `kaggle.json` is
not in place or your token expired. Re-create it from
https://www.kaggle.com/settings.

## Day 1 (or whenever) — set up artifact storage

This step is needed once you start producing OOF / test prediction
artifacts. **Skip until you have at least one `.npy` file in
`scripts/artifacts/`.**

The artifacts pile up fast (roughly 1-3 MB per base model × the number
of bases). Tracking them in git bloats the `.git` directory; the cure
is a private Kaggle Dataset that mirrors `scripts/artifacts/`.

```bash
# First-time setup — pushes the initial dataset.
bash scripts/setup_artifact_dataset.sh \
    --username <your-kaggle-username> \
    --slug <comp-slug>-artifacts
```

What that script does:

1. Edits `.kaggle-artifacts/dataset-metadata.json` to use your username
   and the slug.
2. Hard-links every `*.npy` and `*.json` from `scripts/artifacts/` into
   `.kaggle-artifacts/` (no disk doubling — they're hard links).
3. Calls `kaggle datasets create` to push the dataset (private, CC0).

After it succeeds, the dataset lives at
`https://www.kaggle.com/datasets/<username>/<slug>`.

### Daily versioning ritual (after each session)

```bash
cd .kaggle-artifacts
find ../scripts/artifacts -maxdepth 1 -type f \
     \( -name "*.npy" -o -name "*.json" \) -exec ln -f {} . \;
kaggle datasets version -p . \
    -m "$(date +%Y-%m-%d) $(git branch --show-current) +<probe-name>" \
    --dir-mode zip
```

Or use the script in `.kaggle-artifacts/README.md`.

### Stop tracking artifacts in git (after the first push works)

```bash
git rm -r --cached scripts/artifacts/ submissions/
mkdir -p scripts/artifacts submissions
touch scripts/artifacts/.gitkeep submissions/.gitkeep
git add .gitignore scripts/artifacts/.gitkeep submissions/.gitkeep
git commit -m "Migrate artifacts to private Kaggle Dataset"
```

`.gitignore` already excludes `scripts/artifacts/*` and `submissions/*`
in this repo. Inherit those lines if you copy this template.

## How the ART resolver works (so you don't have to think about it)

`scripts/common.py` exposes a Path called `ART` that auto-resolves to
the right artifact location:

- **Locally:** `scripts/artifacts/` (downloaded via `kaggle datasets
  download` or hard-linked from `.kaggle-artifacts/`).
- **On Kaggle notebooks:** `/kaggle/input/<comp-slug>-artifacts/`
  (after you click "Add Data → \<your-username\>/\<slug\>" in the
  notebook editor).

Existing scripts use `from common import ART; np.load(ART /
"oof_X.npy")`. The resolver picks the right path; you don't change
your code.

## Where the PI's voice goes

The PI uses voice-to-text and dumps thoughts freely. Those go in
`knowledge-base/thoughts/`, dated, one file per session or per topic:

```
knowledge-base/thoughts/2026-05-08-cleanup-priorities.md
```

The agent transcribes lightly, never overwrites, and links related
entries. **This folder is permanent** — do not archive or compress it
during cleanup. See `knowledge-base/README.md` for the full PI second-
brain layout (`thoughts/`, `concepts/`, `friction/`, `questions/`,
`flags/`).

## Sanity check before declaring "ready to work"

Run this smoke test (it's idempotent — works as a regression test for
the artifact migration too):

```bash
python scripts/smoke_kaggle_artifacts.py
```

Expected: 6 steps PASS in under 10 seconds. If it fails on step 1
("ART resolver"), the dataset isn't downloaded. If it fails on step 2
("y from comp data"), `data/train.csv` isn't downloaded. The error
messages tell you which.

## Next: read these in order

1. `CLAUDE.md` — rules + pointers.
2. `HANDOVER.md` (if mid-comp) or `comp-context.md` (if day 1).
3. `state/current.md` — current PRIMARY and what axes are open.
4. `audit/friction.md` — recent recurring snags.
