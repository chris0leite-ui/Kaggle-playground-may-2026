# Knowledge Base

A personal, evolving notebook where the PI (you) thinks out loud and Claude
acts as a writer/synthesizer: capturing thoughts, organizing them, asking
critical questions, clarifying concepts, and linking related entries.

## Purpose

- **Primary goal**: learn how to do *semi-automated research with coding
  agents*. Kaggle competitions are the benchmark/practice ground, not the
  end goal.
- **Role split**:
  - PI = strategist / orchestrator-of-orchestrators. Instructs and checks;
    does not write code.
  - Claude (coding agent) = orchestrator + executor.
- **Comparison anchor**: real-world job is electricity-price time-series
  forecasting under constant distribution shift, feedback loops from
  trading signal, and non-stationary features. Kaggle is the laboratory
  setting; the gap between lab and production is itself a topic of study.

## Structure (initial — will evolve)

- `README.md` — this file. Purpose, structure, how to add entries.
- `thoughts/` — raw thought-dumps, dated. PI talks, Claude transcribes
  and lightly organizes. Source material.
- `concepts/` — clarified concepts, lessons, mental models. Distilled
  from `thoughts/` over time.
- `friction/` — observed friction in the multi-agent / multi-branch /
  multi-machine workflow. Symptoms, hypotheses, attempted fixes.
- `questions/` — open questions Claude has raised back to the PI for
  reflection. Cross-linked to the entry that prompted them.

## How to use

1. PI talks freely (voice/text dump, unstructured).
2. Claude appends to or creates the right file(s) in `thoughts/`.
3. Claude may extract concept sketches into `concepts/` or friction
   notes into `friction/` when a theme repeats.
4. Claude asks clarifying / critical questions in `questions/` and
   surfaces them back at appropriate moments.
5. PI rereads and refines whenever they want.

## Conventions

- Markdown only.
- Date-prefixed filenames (`YYYY-MM-DD-slug.md`) inside `thoughts/`.
- Cross-link aggressively with relative paths.
- Keep individual files small (≤150 lines); split when bloated.
- Capture the PI's voice; do not over-edit. Synthesis goes in separate
  derived files, not on top of the raw thought.
