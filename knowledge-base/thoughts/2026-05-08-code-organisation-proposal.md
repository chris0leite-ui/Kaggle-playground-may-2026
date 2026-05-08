# 2026-05-08 — code organisation proposal (PI question)

PI mused at end of session: "we have standalone codes all over the
place. But building a whole Python package, say, would be another
story in quite complicated maybe and not necessary. Maybe."

This note is the agent's recommendation, not a decision. Decision is
PI's at next-session pickup.

## What's in `scripts/` today

The `scripts/` directory has accumulated ~150 standalone Python files
across 19 days of competition. They fall into four buckets:

1. **Harness scripts** — generic across comps. Currently:
   `probe.py` (BOTE + gate), `probe_min_meta.py` (stack-add gate),
   `pre_submit_diff.py` (mandatory before submit), `research_seed.py`
   (family kickoff stub), `smoke_kaggle_artifacts.py` (Kaggle Dataset
   regression test), `setup_artifact_dataset.sh`, and `common.py`
   (the `ART` resolver + folds + metric helpers).
2. **Comp-specific probe scripts** — one per experiment. Names like
   `probe_field_state.py`, `probe_target_reform_strict_oof.py`,
   `probe_path_b_K22_dae.py`, etc. Each reads from `data/` and
   `scripts/artifacts/`, writes a new OOF/test pair.
3. **LR diagnostic battery** — 10 scripts (`lr_diag_e1_svd.py`, etc.)
   already promoted to a skill recipe.
4. **Archived** — `scripts/archive/` from the audit-ml-repo cleanup
   (117 files moved). Not active.

## Three options, ranked by leverage / cost

### Option A — Curated reusable layer (recommended)

Keep `scripts/` as the working dir for comp-specific probes. Promote
the **harness layer** into a small module that other comps can import
from a sibling location. Specifically:

```
~/.claude/skills/kaggle-comp/lib/
    __init__.py
    harness/
        probe.py           # BOTE + gate
        probe_min_meta.py
        pre_submit_diff.py
        research_seed.py
    common.py              # ART resolver + folds + metrics
    smoke.py               # smoke_kaggle_artifacts core (param'd by slug)
    setup/
        bootstrap.sh
        setup_artifact_dataset.sh
```

The next comp's `scripts/` directory has a small bootstrap shim:

```python
# scripts/__init__.py (new comp)
import sys
sys.path.insert(0, str(Path.home() / ".claude/skills/kaggle-comp/lib"))
from common import ART, folds  # noqa
```

Or, simpler: each new comp's `scripts/common.py` is just a one-line
re-export:

```python
from kaggle_comp_lib.common import *  # noqa
```

**Cost:** ~2 hours one-time refactor, mostly moving files and
parameterising the comp slug. No new package management; just
`sys.path` insertion.

**Benefit:** Bug fixes and improvements to the harness propagate to
every future comp automatically. The five harness scripts have been
edited 30+ times across this comp; that churn would have benefited
every prior comp if they shared a layer.

**Risk:** Drift between comps. If comp A patches `probe.py` for a
metric-specific quirk, comp B inherits it. Mitigation: keep the
harness layer minimal and metric-agnostic; comp-specific behaviour
stays in the comp's own `scripts/`.

### Option B — Full Python package

Make a proper `pip install -e .`-able package. Each comp depends on
it via `requirements.txt`. Setup includes `pyproject.toml`,
versioning, optional CI for the package itself.

**Cost:** ~1 day to build, plus per-comp setup overhead (pin a
version, decide on update strategy).

**Benefit:** Clean dependency management; `import kaggle_comp` from
anywhere; testable in isolation.

**Risk:** PI accurately flagged this. Maintenance overhead exceeds
the value at this scale. Five harness scripts × 2 comps doesn't
justify a package; would need 10+ scripts × 5+ comps before the
overhead pays back.

### Option C — Status quo

Each comp's `scripts/` is freestanding. Manually copy harness scripts
when starting a new comp (the `templates/` directory we just shipped
makes this two `cp` commands).

**Cost:** zero now.

**Benefit:** Maximum freedom per comp; no inheritance bugs.

**Risk:** Drift the other direction — fix a bug in s6e5's `probe.py`
and the next comp will hit the same bug because we forgot to copy the
fix. Friction tag `lesson-not-applied` already documents this pattern.

## Recommendation

**Option A.** Specifically:

1. **Now (defer if PI declines):** Move the 7 harness files to
   `~/.claude/skills/kaggle-comp/lib/`. Update this comp's `scripts/`
   to re-export from there.
2. **Templates update:** `templates/common.py` (already shipped) gets
   a one-line "import from lib" form as a recommended starting point.
3. **No package.** Skip `pyproject.toml`, skip versioning. The lib is
   on `sys.path` via `~/.claude/skills/kaggle-comp/lib/`; if a future
   comp wants to pin a snapshot, it copies the lib into its own
   `scripts/` and diverges from there.

## What stays in each comp's `scripts/`

- All comp-specific probes (the 100+ `probe_*.py` files).
- The comp's own `common.py` — but it's a one-line re-export of the
  shared lib's `common.py`.
- Anything that touches `data/train.csv` or comp-specific feature
  engineering.

## Open question for the PI

**How much pinning?** Two answers possible:

- "Always live with the latest." Each comp's `common.py` re-exports
  whatever `~/.claude/skills/kaggle-comp/lib/common.py` is today. Fixes
  propagate. Risk: a future fix could break this comp's reproducibility.
- "Snapshot at comp start." Each comp copies the lib at kickoff into
  its own `scripts/lib/`. Reproducibility is preserved; cross-comp
  fixes need manual port.

I'd recommend "always live with the latest" while the comp is open;
**snapshot at end-of-comp** so the comp's final repo is reproducible
forever after the comp closes. That's a third option neither of the
above states cleanly. Worth a 5-minute decision when the PI is ready.
