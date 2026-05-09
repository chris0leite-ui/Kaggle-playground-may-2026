# ISSUES.md — problem decomposition / claim board

> Live problem-tree per Rule 18. A leaf must be claimed before any
> probe ≥10 min CPU/GPU. Status values: `open`, `wip`, `done`, `null`,
> `parked`. Owner is the branch/agent currently working it.

## Active leaves

> **Day-1 agent:** decompose the agent-design problem. Suggested top-
> level: "Build an Orbit Wars agent that wins ≥X% vs the baseline-opponent
> panel by deadline." Children to seed:
>
> - **Env dynamics** — observation, action, reward, transition. Read 3
>   replays end-to-end. `[owner: unclaimed | status: open]`
> - **Agent class** — heuristic / search / IL / RL / hybrid. Choose
>   the simplest class that beats the random baseline. `[owner: unclaimed | status: open]`
> - **Reward / value signal** — what the agent optimises during training
>   vs how Kaggle evaluates. (Rule 16 Q6.) `[owner: unclaimed | status: open]`
> - **Training-eval infra** — local self-play loop, hold-out opponent
>   eval, replay logging. `[owner: unclaimed | status: open]`
> - **Submission packaging** — kernel build, dependency pinning,
>   compute-quota fit, dry-run via `kaggle_environments.evaluate()`.
>   `[owner: unclaimed | status: open]`

## Falsified or dead

(empty)

## Re-decomposition triggers

- 3 nulls in a row on the same leaf → re-decompose that subtree.
- 50% of comp budget elapsed → review tree against current LB shape.
- Plateau ≥2 days on PRIMARY rank → research-loop + re-decompose.
