# Kaggle Playground Series s6e5 — F1 Pit Stop Prediction

Workspace for the May 2026 Playground comp (`PitNextLap` row-level AUC).
Deadline **2026-05-31 23:59 UTC** (23 days remaining as of 2026-05-08).

## Current state

PRIMARY = `d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000`, LB **0.95368**,
rank #98/893 (top 11%). Top-5% boundary 0.95405 (gap −3.7 bp). 39/270
submissions used. Live state in `CLAUDE.md ## Current state`.

## Where things live

| Path | What | Read when |
|------|------|-----------|
| `CLAUDE.md` | Rules + current-state YAML + calibration ladder | Every session start |
| `HANDOVER.md` | Next-session brief (≤150 lines) | "handover" trigger |
| `WRAPUP.md` | Wrap-up + prepare-handover procedure | "wrap up" / "prepare handover" |
| `ISSUES.md` | Live problem decomposition (claim board) | Before any >10 min probe |
| `comp-context.md` | Settled-once facts (schema, LB, decisions) | Never re-asked |
| `audit/` | Per-probe audit notes (`YYYY-MM-DD-<slug>.md`) + decisions log | Citing precedent |
| `audit/friction.md` | One-liner friction tags | Top of every session |
| `audit/decisions.jsonl` | BOTE / outcome calibration log | `probe.py calibration` |
| `knowledge-base/` | PI second-brain (concepts, thoughts, questions) | Reflection / research |
| `scripts/` | Live tools + recent-day probe scripts | Day-13+ on top level |
| `scripts/archive/` | Archived day-tagged research log (Day-3 to Day-12, dead probes) | Reproducing old probes |
| `scripts/artifacts/` | OOF / test `.npy` per base + gate / calibration JSON | Loaded by `probe_min_meta.py` |
| `kernels/` | Kaggle GPU notebook scaffolds | When dispatching to Kaggle |
| `submissions/` | LB-submitted CSVs + held HEDGE candidates | Final-window selection |
| `external/` | Pulled Kaggle datasets (gitignored) | Train / FE merge |
| `data/` | Comp data (gitignored) | Bootstrapped via `bootstrap.sh` |

## Live tools

| Tool | Purpose |
|------|---------|
| `scripts/probe.py` | BOTE pre-flight + post-hoc gate harness (Rule 19) |
| `scripts/probe_min_meta.py` | K=21 + N candidate-base LR-meta gate |
| `scripts/pre_submit_diff.py` | Mandatory ρ-vs-PRIMARY diff before any submit |
| `scripts/research_seed.py` | Generate web-retrieval stub for new family (Rule 19g) |
| `scripts/hypothesis_view.py` | Graph view of `mechanism_families_explored` |
| `scripts/lb_status.py` | Pull leaderboard snapshot |

## Quick commands

```bash
# Bootstrap data + venv
./bootstrap.sh

# Pre-flight a candidate before compute
python scripts/probe.py bote NAME --family X --cost_min N

# Gate after artifacts exist
python scripts/probe_min_meta.py --candidates NAME

# Pre-submit diff (MANDATORY before LB submit)
python scripts/pre_submit_diff.py PATH/TO/submission.csv

# Calibration snapshot (agent expected vs actual LB)
python scripts/probe.py calibration
```

## Branch convention

Develop on `claude/<slug>` per harness assignment. PI consolidates from
`origin/main`. See `WRAPUP.md` for the wrap + handover protocol.
