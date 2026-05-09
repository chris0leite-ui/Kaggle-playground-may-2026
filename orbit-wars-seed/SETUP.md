# SETUP — starting Orbit Wars

This is the onboarding checklist for a fresh container, fresh laptop,
or this brand-new repo. **Read this once, top to bottom, on day 1.**

For day-to-day rules and process, see `CLAUDE.md`. For session-start
read order on an *existing* competition, see `HANDOVER.md`.

> **Orbit Wars note (code/agent comp).** This SETUP was originally
> written for tabular comps. The artifact-Kaggle-Dataset section that
> used to live here is dropped — code/agent comps don't produce OOF
> arrays in the tabular sense. The reference-notebook and
> simulator-probe steps replace it.

## What you need before starting

1. **A Kaggle account** with API access enabled
   (https://www.kaggle.com/settings → "Create New Token" downloads
   `kaggle.json`).
2. **`~/.kaggle/kaggle.json` in place, mode 600.** That is the
   PI-recommended path for this repo. Two env-var fallbacks
   (`KAGGLE_USERNAME` + `KAGGLE_KEY`, or `KAGGLE_API_TOKEN`) also work
   via `bootstrap.sh`, but the `~/.kaggle/kaggle.json` path is simpler
   and more portable.
3. **Python 3.10+** with `pip`.
4. **The competition slug**: `orbit-wars`.

## Day 1 — first 15 minutes

### Step 1. Clone and bootstrap

```bash
git clone <repo-url>
cd <repo>
cp .comp.env.template .comp.env  # already pre-filled with COMP="orbit-wars"
bash bootstrap.sh
```

What `bootstrap.sh` does, in order:

1. Resolves Kaggle credentials. If `KAGGLE_USERNAME` + `KAGGLE_KEY` are
   set, it materialises `~/.kaggle/kaggle.json` so the standard
   `kaggle …` commands work everywhere.
2. Installs Python requirements from `requirements.txt`.
3. Downloads the competition data into `data/` if not already present.
4. (Code-comp addendum.) Pulls the top reference notebooks into
   `external/kernels/` for cross-reference. The exact slugs are
   filled in by the day-1 agent — see Step 3 below.

### Step 2. Confirm the data downloaded

```bash
ls -lh data/
```

For a code/agent comp expect a small bundle: a sample-replay JSON, an
env-spec / brief, possibly a sample-agent Python file. There will not
be a `train.csv` / `test.csv`. Inventory whatever appears.

If the download fails with an authentication error, `~/.kaggle/kaggle.json`
is not in place or your token expired. Re-create it from
https://www.kaggle.com/settings.

### Step 3. Pull the reference notebooks

```bash
kaggle kernels list -s orbit-wars --sort-by voteCount | head -10
# pick the top 3 and pull each:
kaggle kernels pull <user>/<slug> -p external/kernels/<slug>/
```

Known starters worth trying first:
- Bovard — "Getting Started" (Kaggle staff baseline).
- Kashiwaba — "Reinforcement Learning Tutorial".
- Sangram Patil — "[API] Download Replay".

For each, read once and write a 2–3-sentence summary into
`audit/<today>-day-1-data-inventory.md` describing the agent class
(heuristic / search / IL / RL / hybrid).

### Step 4. Local simulator probe

```bash
python -c "import kaggle_environments; \
  env = kaggle_environments.make('orbit_wars'); \
  env.run(['random','random'])"
```

If the env name is wrong or the env doesn't ship offline yet, log it
as friction in `audit/friction.md` and fall back to reading a sample
replay JSON to learn the observation/action shapes by hand.

## Where the PI's voice goes

The PI uses voice-to-text and dumps thoughts freely. Those go in
`knowledge-base/thoughts/`, dated, one file per session or per topic:

```
knowledge-base/thoughts/2026-05-09-kickoff-priorities.md
```

The agent transcribes lightly, never overwrites, and links related
entries. **This folder is permanent** — do not archive or compress
it during cleanup. See `knowledge-base/README.md` for the full
PI second-brain layout.

## Sanity check before declaring "ready to work"

Day-1 agent verifies:

- [ ] `kaggle competitions view orbit-wars` returns success.
- [ ] `data/` has the comp bundle unzipped.
- [ ] `external/kernels/` has at least 1 reference notebook.
- [ ] `comp-context.md` is filled in (slug, deadline, daily submission
      cap, agent IO spec, episode length, opponent pool, replay format).
- [ ] `ISSUES.md` has the first problem-tree (5 children).
- [ ] `HANDOVER.md` has the 4-section day-1 brief.

## Next: read these in order

1. `CLAUDE.md` — rules + pointers.
2. `comp-context.md` — settled-once facts for Orbit Wars.
3. `state/current.md` — current submitted agent and what axes are open.
4. `audit/friction.md` — recent recurring snags.
