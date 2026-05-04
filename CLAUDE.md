# CLAUDE.md — playground-series-s6e5

Running log + ⚠️ rules. Cap ≤50k tokens; archive when bloated.

## ⚠️ Reference branch — check `origin/main` FIRST every session

Truth lives on `origin/main`. Feature branches (`claude/*`) often start
behind. Before planning, executing, or answering "what's next?" — run:

```
git fetch origin && git log --oneline HEAD..origin/main
```

If non-empty, fast-forward (`git merge origin/main --ff-only`) before
reading `CLAUDE.md`, `audit/`, or `scripts/`. The state below is only
authoritative once you've synced.

## ⚠️ Top-level rules (inherited from kaggle-comp framework)

These eleven invariants are LOAD-BEARING. Do not skip.

1. **Ask-first / no-loop on submissions.** Every `kaggle competitions
   submit` is single-shot, explicitly approved. Never wrap in retry
   / `until` / `while` / `for`. Monitors that POLL are fine; monitors
   that WRITE are forbidden.
2. **Smoke + 1-fold time-probe + 1h GPU cap.** Smoke at 1 fold /
   50k rows. Then 1-fold full-data probe. If 5-fold projection ≥1h,
   shrink. If a kernel is in preprocessing at t+30min with no fold
   output, kill it.
3. **4-gate leakage filter pre-LB-probe.** G1 standalone OOF clears
   anchor; G2 blend lift; G3 net rare-class-flip ratio ≥0.5; G4
   direction asymmetry. Plus minimal-input-meta sanity check.
4. **NEVER-GIVE-UP / saturation-is-bounded / never-lock-and-stop.**
   Saturation evidence proves we tested known levers, not that no
   lever exists. After every null, brainstorm 3 untried mechanisms.
   Locking is for the final 3-day window only.
5. **Keep CLAUDE.md fresh / archive-on-bloat.** Cap ≤50k tokens.
   Subagents load slices, not full files.
6. **Heuristics before heavy compute.** Closed-form rule / threshold
   / hand-coded baseline before Optuna / GPU / 5-fold-bagging.
7. **Research before saturation.** At every plateau (3 nulls / 5
   sat at same LB / 2 days no lift): web search top notebooks,
   read 2 prior-comp writeups, list 5 untried mechanisms with
   citations BEFORE declaring ceiling.
8. **Settled-once facts** live in `comp-context.md`. Never re-ask.
9. **File-size cap ≤150 lines** for any committed doc.
10. **Pull-style updates.** No proactive minute-level chatter; on
    PI pull, 1-2 sentences with the latest fact.
11. **Model routing.** Haiku for routine read-only checks; Sonnet
    default; Opus for hard reasoning. Use the daily 10/day budget.

## ⚠️ Defaults baked in from prior-comp postmortem

These were the R1/R2/R7 changes from irrigation-water postmortem-07.
They override the kickoff-time defaults.

- **R1 — Two-anchor OOF.** Every gated candidate must pass under
  TWO CV schemes: (a) standard 5-fold StratifiedKFold seed=42, AND
  (b) GroupKFold on a row-id hash (or repeated stratified with a
  different seed). Mechanisms that overfit one fold geometry will
  diverge.
- **R2 — Final selection along the public-LB axis.** PRIMARY = best
  public LB. HEDGE = best OOF that *regressed ≤30bp on public*. NOT
  another orthogonal-mechanism hedge. (Last comp: 5 of our subs
  beat PRIMARY on private; all had been rejected for public regress.)
- **R7 — Override-mechanism rules.** Override candidates with flip
  count <200 cannot be PRIMARY (only HEDGE). Override candidates
  with >200 flips need explicit PI sign-off. Override flips on a
  small public split overfit deterministically.
- **R5 — Final OOF-best regression probe.** In the final 3-day
  window, 1 mandatory probe of the OOF-best candidate that was
  rejected for public regression.
- **R8 — End-of-comp**: log final percentile to
  `~/.claude/skills/kaggle-comp/improvements.md`.

## Current state (Bookkeeper updates daily)

```yaml
day: 1
lb_best_today: 0.95435            # leader at kickoff (2026-05-04)
our_lb_best: 0.94113              # baseline_two_anchor (StratKFold), Day-1
submissions_used_today: 1
submissions_used_total: 1
saturation_count: 1               # D2-A null both anchors (2026-05-04)
mechanism_families_explored: [baseline_lgbm_raw_features, oof_target_encoding]
plateau_days: 0
gate_status: cleared              # pre-baseline gate cleared 2026-05-04; see audit/2026-05-04-pre-baseline-gate.md
headroom_to_top5pct: 0.01232      # 0.95345 − 0.94113 = 123bp
```

## Calibration ladder

Updated by the Calibration-loop. Format: mechanism / OOF / LB / gap.

| Mechanism | OOF | LB | Gap | Notes |
|---|---:|---:|---:|---|
| baseline_two_anchor (StratKFold) | 0.94075 | 0.94113 | +3.8bp | calibration ✓ ; anchor A confirmed right proxy |
| baseline_two_anchor (GroupKFold Race) | 0.92059 | n/a | n/a | race-robustness; not LB proxy |
| d2a_te (StratKFold) | 0.93670 | n/a | n/a | NULL G1; −40.5bp; not submitted |
| d2a_te (GroupKFold Race) | 0.91628 | n/a | n/a | NULL G1; −43.1bp; not submitted |

## Hypothesis board

```
- next: blend-G2 — 0.5·baseline + 0.5·d2a_te on existing OOF .npy
        FREE; tests cause #3 of d2a null (TE may only help in stack)
- next: TE-only-replace-raw — drop raw Driver/Race when adding TE
        cheap edit; tests cause #1 of d2a null
- next: TE on (Driver, Race) only (drop Compound + interactions)
        cheap; tests cause #2 of d2a null (high-card noise)
- D2-C (queued): concat external (aadigupta1601) rows; lower priority
                  since probe 1 join missed
- D3+ : RealMLP/PyTabKit (Source 1 #1: standalone ~0.946 on this DGP)
- D3+ : LGBM hyperparam sweep (Optuna, 30 trials) — only after research
```

## Friction log pointer

Friction one-liners go in `audit/friction.md`, NOT here. Distill
weekly per `~/.claude/skills/kaggle-comp/self-improvement.md`.

## Pointers

- `comp-context.md` — settled-once facts.
- `brief.md` — verbatim host material.
- `LEARNINGS.md` — portable patterns from this comp.
- `REPORT.md` — structured work report.
- `audit/` — timestamped per-experiment results.
- `audit/friction.md` — friction one-liners (rotated weekly).
