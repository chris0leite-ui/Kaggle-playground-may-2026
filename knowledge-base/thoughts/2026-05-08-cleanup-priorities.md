# 2026-05-08 — cleanup priorities

PI directives during this session, paraphrased and transcribed by the
agent. The day was a documentation + infrastructure cleanup, not a
modelling day.

## Directives in order received

### 1. Pick up the audit-ml branch and check status

The branch `claude/audit-ml-repo-B6hlL` had been doing repo cleanup —
moving stacking artifacts into a private Kaggle Dataset and rewriting
git history to drop binary blobs. The agent reported back on what the
branch did (8 cleanup commits, history rewrite, .git 3.9 GB → 31 MB),
ran the smoke test from a worktree, and confirmed the migration
reproduced on round-trip.

### 2. Push the cleanup branch into main

PI authorised the force-push after being briefed on the parallel
history (the branch shares no merge base with origin/main because of
the filter-repo rewrite). Agent first pushed a backup branch
(`backup/pre-cleanup-main-2026-05-08`) as a rollback path, then
force-pushed the cleanup to main. PI later deleted the backup branch.

### 3. Take the lens of an AI engineer

PI asked the agent to read the friction archive (1,450 lines) and the
broader documentation, then explain in plain English what's there.
Agent grouped findings into six themes: leakage discoveries, stacking
saturation / rank-lock, the per-segment shrinkage trick, framework
behaviour rewarding wrong things, recurring process snags, and
calibration. PI's two requirements: no jargon, no letter-number
experiment codes (E1, F2, K=27, etc.) in chat.

### 4. Restructure the documentation

PI's instructions, paraphrased:

- CLAUDE.md must be short — rules and pointers only, no story.
- The story / war stories / recurring-friction logs go in dedicated
  files that agents pull on demand, not read by default each session.
- Frictions that haven't become rules need to become rules. In a
  concise manner; the story behind them goes elsewhere.
- The PI shouldn't have to look up a glossary mid-conversation.
  Plain-English communication every time.

What landed:

- CLAUDE.md trimmed from 608 to 150 lines.
- New Rule 0: plain-English communication; no abbreviations the PI
  hasn't used; no letter-number experiment codes in chat.
- New Rules 27-34: pre-submit prediction diff, sub-agent dispatch
  limits, same-session friction application, GPU kernel template gate,
  concurrent-compute cap, session-start git fetch, inner-CV-validate
  post-hoc OOF transformations, descriptive experiment names.
- State split into `state/` (current.md, calibration-ladder.md,
  hypothesis-board.md, mechanism-ledger.md).
- New `glossary.md` (210 lines, on-demand) and `rules-history.md`
  (293 lines, on-demand) for everything that no longer belongs in
  CLAUDE.md.
- Friction file: 1,450-line stream-of-consciousness moved to
  `audit/friction-archive.md`; a 128-line concise weekly-summary
  version replaced it at `audit/friction.md`.

### 5. Diary folder is permanent — never archive

PI flagged that `knowledge-base/thoughts/` is the diary and must be
preserved across cleanups. The folder existed but had been dormant
since 2026-05-06.

What landed: new Rule 35 — PI thoughts are append-only; transcribe to
`knowledge-base/thoughts/YYYY-MM-DD-slug.md`; never overwrite, delete,
or archive.

### 6. Setup for the next competition

PI noted that agents in early days didn't know how to bootstrap. For
the next competition repo, the agent should know how to (a) bootstrap
data download, (b) set up the private Kaggle Dataset for artifact
storage.

What landed:

- `bootstrap.sh` rewritten to be portable across three credential forms
  (the standard `~/.kaggle/kaggle.json`, `KAGGLE_USERNAME` +
  `KAGGLE_KEY` env vars, or `KAGGLE_API_TOKEN`). Auto-creates
  `kaggle.json` from env vars when present. Auto-pulls the artifact
  Kaggle Dataset on a fresh clone — verified end-to-end (532 files,
  563 MB pulled in this session).
- `scripts/setup_artifact_dataset.sh` for the first-time setup of a
  new comp's artifact dataset.
- `SETUP.md` in repo root — comprehensive new-comp onboarding flow.
- `~/.claude/skills/kaggle-comp/setup.md` — cross-comp version of the
  same.

### 7. Check for exposed secrets

Agent ran a sweep — no actual tokens or credentials in any tracked
file. The Kaggle username `chrisleitescha` is exposed in 14 places but
that's a public handle, not a secret.

### 8. Take the AI-engineer lens, do one iteration of seven-step

PI asked for an audit of what else needs fixing given everything
learned in the session. Agent's findings (synthesis step):

1. The knowledge base is being protected by Rule 35 but isn't being
   *used*. The diary hasn't received an entry in this entire cleanup
   session despite multiple substantive PI directives. **Writing this
   note is the first action item.**
2. SETUP.md exists but `templates/` doesn't — the next-comp agent has
   to reverse-engineer this repo to find what to copy.
3. `audit/decisions.jsonl` has 25 rows after ~80 probes. Calibration
   undersampled; some families show systematic bias.
4. `ISSUES.md` says "~8 remaining days" — it's 23 days. Strategy-
   critic-loop (Rule 14) was supposed to catch this drift; didn't.
5. The cross-comp `improvements.md` doesn't exist on this machine —
   the postmortem skill silently no-ops on promotions because the
   destination is missing.

PI accepted the recommendations and added two items:

- **Promote the recipes that scale across competitions.** The CatBoost
  yekenot transfer, RealMLP yekenot full-recipe, kitchen-sink Rozen
  LightGBM, the LR-diagnostic suite, the per-segment shrinkage
  stacker — all of these likely transfer to any tabular comp. Move
  them to the cross-comp skill so the next comp inherits them.
- **Code organisation as a future "next move."** The repo has standalone
  scripts everywhere; building a Python package would be cleaner but
  probably overkill. PI is musing — wants the agent to think about
  which code is worth keeping, not necessarily build a package.

## Calibration audit findings (this session)

Ran `python scripts/probe.py calibration`. 25 rows. Visible patterns:

- `single_base_fe_addition` family: agent over-predicts (0.03-0.6 bp
  predicted; 0.0 bp actual on 5 of 5 entries). Friction archive
  already noted this for cross-row aggregates; the family prior in
  `scripts/probe.py` was never updated.
- `meta_arch_redesign` family: agent over-predicts +1.20 vs −0.47
  actual. Single observation; small N.
- `external_data_aggregate`: noisy; the d18 K=23 chain-decomposition
  surprise (predicted +1.26, actual +6.0) drives a +4.74 bp
  underprediction; subsequent probes more conservative.
- `new_model_class`: when it lands, lands big (+19.6 bp on the
  RealMLP recipe transfer); agent's +27 bp prediction was too high
  but the right order of magnitude.

## What this means for next session

The framework optimises HOW to evaluate ideas. The bottleneck is that
the framework's *operation* is uneven — rules logged but not applied,
state files going stale, calibration log undersampled, knowledge base
dormant. The cleanup this session helped readability; it didn't fix
operation.

The single highest-leverage habit change would be: **at session-end,
the agent must add at least one entry to `knowledge-base/thoughts/`,
list any open questions in `knowledge-base/questions/`, and surface
persistent flags in `knowledge-base/flags/`.** The folder being
permanent isn't enough; the writing has to be a habit.

Adding that as Rule 36.
