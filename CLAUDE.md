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
12. **Spend the full 5/day submission budget every day.** Submissions
    are calibration probes — measured OOF→LB gap per mechanism family
    is the load-bearing data, not just rank. Do NOT intentionally
    underspend. Each submit still single-shot + PI-approved (Rule 1).

## ⚠️ Defaults baked in from prior-comp postmortem

These were the R1/R2/R7 changes from irrigation-water postmortem-07.
They override the kickoff-time defaults.

- **R1 — Two-anchor OOF.** Every gated candidate must pass under
  TWO CV schemes: (a) standard 5-fold StratifiedKFold seed=42, AND
  (b) GroupKFold on a row-id hash (or repeated stratified with a
  different seed). Mechanisms that overfit one fold geometry will
  diverge.
  *Qualifier (added 2026-05-04 per s6e5 friction):* if a U3-equivalent
  split-structure probe confirms test is an i.i.d. row split (not
  held-out by group), GroupKF is diagnostic-only, NOT predictive of
  LB. In that case, single-anchor Strat is sufficient for routine
  bases; reserve GroupKF for the final PRIMARY/HEDGE sanity check
  only. **s6e5 status: GroupKF dropped Day-3+ (test confirmed i.i.d.
  by U3, baseline gap +3.8bp confirms Strat = LB proxy).**
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
day: 2                            # CLOSED 2026-05-04 (5/5 used)
lb_best_today: 0.95435            # leader (still); not refreshed since kickoff
our_lb_best: 0.94963              # M5d 12-base stack, Day-2 (+85.0bp over Day-1)
submissions_used_today: 5         # baseline + M5 + M5b + E3 + M5d
submissions_used_total: 5
saturation_count: 1               # D2-A null both anchors
mechanism_families_explored:
  - baseline_lgbm_raw_features
  - oof_target_encoding
  - xgb_native_categorical
  - catboost_native_categorical
  - relative_state_fe
  - lr_meta_stacker_3view
  - dirichlet_random_search
  - hgbc_label_encoded_driver       # E3 -- BEST single-model
  - row_subsample_catboost          # E1 -- dominated by M3
  - l1_meta_sweep                   # E2 -- null on 'fixes gap'
  - realmlp_cpu_singlefold          # E4 -- 0.94722 fold-0, 39.5min, not pursued
  - lr_meta_stacker_expanded        # M5b -- new PRIMARY, LB 0.94891
plateau_days: 0
gate_status: cleared
headroom_to_top5pct: 0.00382      # 0.95345 − 0.94963 = 38.2bp (was 45.4bp)
```

## Calibration ladder

Updated by the Calibration-loop. Format: mechanism / OOF / LB / gap.

| Mechanism | Strat OOF | GroupKF OOF | LB | Notes |
|---|---:|---:|---:|---|
| baseline_two_anchor | 0.94075 | 0.92059 | 0.94113 | LB-proxy ✓ gap +3.8bp |
| d2a_te | 0.93670 | 0.91628 | n/a | NULL G1; +2-4bp in M1 blend |
| m1_blend (50/50 base+te) | 0.94097 | 0.92098 | n/a | best-w 80/20 |
| m2_xgb | 0.94507 | 0.91084 | n/a | Race-overfit |
| m3_catboost | 0.94612 | 0.91645 | n/a | best single before E3; Race-overfit |
| m4_relstate | 0.94244 | 0.92195 | n/a | only B1 lifting both anchors |
| m5_lr_meta | 0.94737 | 0.92483 | 0.94693 | gap −4.4bp |
| m6_dirichlet | 0.94696 | 0.92459 | n/a | held |
| e1_cb_subsample | 0.94596 | 0.91638 | n/a | dominated by M3 |
| e2_l1_meta | 0.94738 | 0.92489 | n/a | null on "L1 fixes gap" |
| e3_hgbc | 0.94876 | 0.92785 | n/a | BEST single, both anchors lift |
| e4_realmlp_cpu_f0 | 0.94722 (f0) | n/a | n/a | not pursued (3.3h proj for 5-fold) |
| m5b_lr_meta_expanded | 0.94926 | 0.92871 | 0.94891 | gap −3.5bp |
| e3_hgbc_standalone | 0.94876 | 0.92785 | 0.94870 | gap −0.6bp (single-model gap≈0) |
| f1_hgbc_deep | 0.94870 | 0.92739 | n/a | β: ~E3 clone |
| f2_hgbc_shallow | 0.94861 | 0.92711 | n/a | β: ~E3 clone |
| e5_optuna_lgbm | 0.94736 | 0.92585 | n/a | tuned hp via Optuna |
| a_horizon_shift | 0.90640 | 0.87474 | n/a | reformulation; stack diversifier |
| b_lapsuntilpit | 0.89840 | 0.86948 | n/a | reformulation; stack diversifier |
| zeta_catboost_deep_f0 | 0.94992 (f0) | n/a | n/a | best single fold; 5-fold not pursued |
| **m5d_lr_meta_expanded** | **0.95023** | **0.92994** | **0.94963** | **D2 PRIMARY; gap −6.0bp (widened)** |

## Hypothesis board (Day 3)

```
- H1: pseudo-labeling on M5d high-conf test rows (~2h, +10-30bp est)
- H2: more reformulations (stint-stratified, residual-from-baseline,
      driver-recent-pit-history) (~3h, +10-25bp est)
- H3: pairwise correlation gate on pool (drop ρ ≥ 0.97) → M5e refit
      (~10min, +2-8bp est, addresses gap-widening from M5d)
- H4: HGBC multi-seed bagging (proper variance reduction, not β
      architectural variants) (~1h, +3-8bp est)
- H5: hill-climb / Ridge meta drop-in (~30min, +0-5bp est, low EV)
- D3+: GPU-only (RealMLP/PyTabKit) — blocked on hardware
- See audit/2026-05-04-day-2-wrap.md for ranked plan + sequence.
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
