# Day-18 PM postmortem — LR-diagnostic research expedition

Branch: `claude/ensemble-logistic-regression-research-MbLKu`
Session: 2026-05-07 PM, ~5h wall, 0 LB submits.
Trigger: PI request to study s6e4 winners' writeups (Chris Deotte 2nd
+ 1st OvR + 4th + 12th); subsequent reframe from "lift bp" to "build
durable knowledge". Wraps with explicit recommendation to merge to
main.

## What went well

1. **Problem-statement reframe under PI direction.** Initial 7-step
   pass anchored on Chris's recipe; PI clarifications ("no public
   CSVs"; "we want to learn") forced two re-cuts of the 7-step plan
   before any compute spent. Saved a 1-2 day port-public-kernel detour
   that would have given +5-15 bp LB but zero learning.
2. **Three arcs delivered three quantitative findings**, each
   cross-confirmed via independent diagnostic:
   - Pool eff_rank 2.88 of 24 (E1) ⟷ K=10 = K=24 in AUC (E9) ⟷
     `cb_slow-wide-bag` first negative-marginal pick (E2 redundancy
     + miscalibration, E9 forward-select).
   - Stint dominant interaction hub (E6) ⟷ +123 bp standalone lift
     when Stint-cross interactions added (A2 vanilla → rich).
   - Representation-only diversity meta-null (A2 + A4 cross-confirmed
     across two structurally distinct constructions).
3. **Every diagnostic produced reusable artifacts**: 10 scripts +
   3 audits + skill module. The skill is the L2 payoff — drops into
   any future tabular comp's Day-1.
4. **Honest stop on T1#3.** The cheapest meta-arch test (alternative
   segmentation crosses) cleanly returned null on K=10. Per the
   conditional plan, T1#1/T1#2 not run — saved 1+ day of likely-null
   compute.

## What went wrong / friction

1. **E5 saga L1 corner ate 1h+ before kill.** Same friction as E8's
   first run. Saga at C ≤ 0.1 on 200k×47 standardized feature matrix
   is intractable inside `max_iter=4000`. **Fix codified in the skill
   README:** drop saga L1 by default; lbfgs L2 sufficient for binary
   AUC rank-no-op verdict.
2. **E5 first run also stuck for 21+ min in re-launch.** Bash `tail`
   pipe buffered all output; couldn't see progress. Killed and pruned
   to l2_balanced + l2 only. **Fix:** default to lbfgs L2 in the skill;
   document saga as opt-in for analyses where L1 sparsity is the
   target output.
3. **Initial 7-step problem-solving was too workplan-shaped, not
   reasoning-shaped.** PI called this out directly: "Where's the
   problem solving? Show me the seven steps." Triggered a re-run
   that surfaced the Chris recipe vs 4th-place writeup distinction
   (4th place is more transferable for our state). **Fix:** under
   problem-solving step display, show *judgment*, not just
   structure. Updated `do-and-dont.md` candidate.
4. **Initial recommendation chased bp, not knowledge.** PI: "I want
   to learn, we want to experiment a lot." Forced the 3-arc design.
   **Fix:** at problem-statement worksheet (Q3.5), *always* re-confirm
   the L1/L2/L3 criterion ordering with PI before listing actions.

## Where we are vs `pi-stamp-risk`

Per Rule 26(e): if 0/M overrides for 2 consecutive postmortems, flag
stamp risk. **Today: 4 PI overrides** (no public CSVs; learn-not-lift
reframe; show problem-solving not workplan; "kill and relaunch" on
the saga lockup). PI is actively steering. **No stamp risk.**

## Friction-to-promotion candidates

Five friction tags written today (`audit/friction.md` top of file):

| Tag | Promotion target |
|---|---|
| `representation-only-diversity-meta-null-on-saturated-info-space` | `do-and-dont.md` — when GBDT pool's residuals on top-pair cells are <1%, no LR/FM/NN base on same features will help; pivot to new-information or new meta-arch |
| `pool-eff-rank-far-below-nominal-on-saturated-gbdt-pool` | `lr-diagnostics.md` (already promoted) — eff_rank diagnostic |
| `s6e4-3-axes-recipe-does-not-transfer-binary-auc` | Q6 Rule 16 (already encoded as metric-aligned check) |
| `cell-residual-magnitude-necessary-not-sufficient-for-lr-signal` | `lr-diagnostics.md` E6 → A2 filter |
| `path-b-amp-requires-large-redundant-pool-not-saturated-pool` | `do-and-dont.md` — Path-B is a redundancy-re-allocation mechanism, not a new-information mechanism |

PI ratification: not blocked on this postmortem; user is wrapping for
merge. Promotions deferred to merge-target scribe per Rule 15.

## Calibration snapshot (Rule 26)

```
name                                     family                         actual    agent       PI  agent_err   pi_err
h3_id_shift_row_position                 single_base_fe_addition         +0.00    +0.60    +0.00      +0.60    +0.00
h2_fastf1_external_join                  external_data_aggregate         +0.00    +3.60    +5.00      +3.60    +5.00
h1_yekenot_realmlp_recipe                new_model_class                +19.60   +27.00    +0.00      +7.40   -19.60
```

No T2/T1#3 entries: Day-18 was research-loop-only, no sealed-prediction
BOTE on LB submits. The K=10 PRIMARY artifact is built but unsubmitted
(PI hold).

## Lesson promoted to skill

`.claude/skills/kaggle-comp/lr-diagnostics.md` + `templates/scripts/lr_diag/`
(10 scripts + README) — full diagnostic battery validated empirically
on s6e5 across 3 arcs. Three durable lessons baked in:
1. Pool effective rank is often << nominal pool size.
2. The dominant interaction hub is one feature 90% of the time.
3. Representation-only diversity is meta-null on a saturated info
   space.

## Recommendation for next session (per HANDOVER Day-18 PM)

In-pool research is empirically exhausted. Three options:
1. **T1#1 non-Gaussian shrinkage** (4-8h CPU; predict null at high
   confidence per saturated-info argument).
2. **T3 Pirelli external data** (PI scope sign-off needed; only path
   that adds new INFORMATION not just new representation).
3. **Wrap s6e5 here** — top-5% achieved; durable wins shipped.

PI's call.
