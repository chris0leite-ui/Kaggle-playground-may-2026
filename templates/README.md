# templates/ — starter files for a new competition

Drop-in starter copies of the load-bearing infra files. When you start a
new competition repo:

```bash
git clone <new-comp-repo>
cd <new-comp-repo>
cp -r ../<this-repo>/templates/. .
mv .comp.env.template .comp.env
$EDITOR .comp.env       # set COMP, ARTIFACT_DATASET
bash bootstrap.sh
```

What's here:

| File | Purpose |
|------|---------|
| `.comp.env.template` | Sets `COMP=<slug>` and `ARTIFACT_DATASET=<user>/<slug>-artifacts`. Sourced by bootstrap. |
| `bootstrap.sh` | Slug-agnostic bootstrap. Reads `.comp.env`. |
| `comp-context.md` | Settled-once-facts template; kickoff agent fills it on day 1. |
| `gitignore` | Standard ignore patterns (rename to `.gitignore`). |
| `kaggle-artifacts-README.md` | Goes to `.kaggle-artifacts/README.md`. |
| `kaggle-artifacts-metadata.json` | Goes to `.kaggle-artifacts/dataset-metadata.json`. |
| `common.py` | `scripts/common.py` with the `ART` resolver. |
| `knowledge-base-README.md` | Goes to `knowledge-base/README.md`. |

Files this template does NOT include (because they're built fresh per
comp): `CLAUDE.md`, `HANDOVER.md`, `state/`, audit notes, kernels,
the actual scripts/probes.

For the full onboarding flow, see `SETUP.md` in the project root.
