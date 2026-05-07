# Decision-time logging — design (MVP-first)

> **Status: shipped 2026-05-06.** PI greenlit; MVP merged into
> `scripts/probe.py`. First real `probe.py bote` call creates
> `audit/decisions.jsonl` and appends one row per call.
>
> Closes the F1.6.1 commitment ("really really need to lock decisions
> together with framework state at decision-time") that the
> [postmortem skill v1](../../.claude/skills/postmortem/SKILL.md)
> deliberately did *not* address ([F1.6.2](../questions/2026-05-06-grilling-round-10.md#f162)).
>
> Smoke-test verified: schema well-formed, 13 fields, framework SHA
> stamped from `git rev-parse HEAD` (= last committed state, not
> working tree — correct semantics for "rules in force at decision").

## Substrate: extend `scripts/probe.py bote`

Add an append to `audit/decisions.jsonl` on every BOTE call. Nothing
else. The BOTE call is already mandatory at decision-time per
Rule 19(a), so capture rides on existing discipline rather than
introducing a parallel mandatory step.

Substrates considered and rejected:
- **YAML frontmatter on audits** — inverts audit lifecycle (currently
  written after probes). Adds friction to the rest.
- **Dedicated `scripts/log_decision.py` CLI** — requires remembering
  to call it; weaker enforcement.
- **Manual / status quo** — re-introduces hindsight bias (the trap
  [`decision-quality-vs-outcome-quality.md`](./decision-quality-vs-outcome-quality.md) warns about).

## Schema (one JSONL event per line)

```json
{
  "ts": "2026-05-06T12:34:56Z",
  "decision_id": "tabpfn_v25_5fold",
  "family": "tabpfn_finetune",
  "verdict": "PURSUE",
  "predicted_oof_delta_bp": [-2, 0, 2],
  "predicted_lb_delta_bp": [-1, 1, 3],
  "predicted_rho_vs_primary": 0.95,
  "cost_min_estimate": 600,
  "framework_sha": "abc123",
  "agent_branch": "claude/..."
}
```

Rationale per field:

| Field | Why it matters |
|---|---|
| `ts` | order events; recover decision time vs commit time |
| `decision_id` | join key for later `realized_*` events |
| `family` | family-prior calibration |
| `verdict` | SKIP / DEFER / PURSUE — the decision itself |
| `predicted_oof_delta_bp` | stage-1 calibration (BOTE → OOF) |
| `predicted_lb_delta_bp` | stage-2 calibration (BOTE → LB) |
| `predicted_rho_vs_primary` | redundancy estimate vs current pool |
| `cost_min_estimate` | EV-vs-effort math |
| `framework_sha` | resolves "which rules existed when" exactly |
| `agent_branch` | multi-branch attribution; race-condition triage |

## What MVP gives you

- Every BOTE call → locked, git-committable record **at decision-time**.
- Framework SHA stamped → which rules / thresholds were active is
  recoverable by `git show framework_sha:CLAUDE.md`.
- Append-only JSONL → trivially queryable (`jq`, pandas, polars).
- Multi-agent safe: append-only file. Merge conflicts only on
  near-simultaneous appends; trivial to resolve.

## What MVP does NOT cover (defer until painful)

1. **PI overrides outside BOTE** (e.g. TabPFN-style "test first").
   Patch when needed: postmortem skill writes override events to the
   same JSONL on session-end.
2. **Realized outcomes.** OOF / LB lifts live in audits today. A
   later `event: realized_oof` (or `realized_lb`) linked by
   `decision_id` enables joins for calibration.
3. **Bypass detection.** If an agent skips BOTE, no record exists.
   Useful information itself — diff `decisions.jsonl` against probes
   actually run.

These are extensions, not blockers.

## Implementation sketch

```python
# scripts/probe.py — append to existing bote command
import json, subprocess, datetime as dt, pathlib

DECISIONS_LOG = pathlib.Path("audit/decisions.jsonl")

def _append_decision_record(name, family, cost_min, verdict, predictions):
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "decision_id": name,
        "family": family,
        "verdict": verdict,
        "predicted_oof_delta_bp": predictions["oof_band"],
        "predicted_lb_delta_bp": predictions["lb_band"],
        "predicted_rho_vs_primary": predictions["rho"],
        "cost_min_estimate": cost_min,
        "framework_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "agent_branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip(),
    }
    DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
```

Total diff to `probe.py`: ~30 lines.

## Adjacent

- [`decision-quality-vs-outcome-quality.md`](./decision-quality-vs-outcome-quality.md)
  — the framing this substrate operationalises.
- [`operational-environment.md`](./operational-environment.md) —
  why the artifact must be git-committable (it is — JSONL is text).
- [`postmortem` skill](../../.claude/skills/postmortem/SKILL.md) —
  retrospective half of F1.6.1; this design is the prospective half.
