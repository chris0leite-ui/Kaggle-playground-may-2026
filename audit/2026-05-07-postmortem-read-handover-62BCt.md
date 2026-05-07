# Postmortem — 2026-05-07 read-handover-62BCt

## What went wrong

**1. H1 misdiagnosis (3 wasted variants × ~30-90 min each).** Strategy
critique priced the +69 bp standalone gap (yekenot 0.95273 vs our
default 0.94582) as "hyperparameter pack + orig-merge". V1 (n_ens=2),
V2 (n_ens=3), V3 (n_ens=3 + orig) all NULL because we skipped the
load-bearing FE pipeline (CV TE on 2-way combos inside fold loop +
arithmetic ratios + floor-cat + count enc + KBins(200/7)).

Decision-quality review (per `concepts/decision-quality-vs-outcome-quality.md`):
- **Decision at BOTE time was rational given priors:** the published
  `REALMLP_PARAMS` block was visible in cell 8 of pit-or-stay; the FE
  pipeline was buried in cell 6 (15 KB code) of yekenot's notebook
  which we hadn't deeply read. Subagent task brief said "yekenot params
  + orig-merge" — that was the load-bearing brief miss, not a model
  mistake.
- **Generalisable rule (PROMOTION CANDIDATE):** before BOTE on a
  "replicate published recipe" candidate, agent must read FULL
  notebook source (not just the hyperparameter cell) and itemise EVERY
  FE step. Promote to skill `do-and-dont.md` as `dont-replicate-published-recipe-from-hyperparams-alone`.

**2. H2 FastF1 — predictable cap not pre-flighted.** Match rate 1.42%
because train.csv has 60% synthetic D### driver codes. Agent BOTE +3.6
bp at 30% probability was too generous given the augmented-codes
problem was foreseeable from a single `train.Driver.value_counts()`
EDA before the 22-min FastF1 install + pull + train cycle.

Decision-quality review:
- **Decision was sub-rational:** EDA on driver codes (1-min
  inspection) would have flagged the 60%-synthetic problem and reduced
  the BOTE EV by 80%, possibly to SKIP. Agent BOTE didn't include
  "preflight check the data" sub-step.
- **Generalisable rule:** before any external-join probe, run
  `<key_col>.value_counts()` head/tail on train + test + external; if
  external can only join < 50% of rows, downgrade BOTE one tier.
  Promote to skill.

**3. Subagent shell-children die on subagent exit (~17 min compute lost).**
H1 strong-mode python launched by subagent's bash invocation got
SIGTERM'd when the subagent timed out. **Fix already applied** — this
branch re-launched strong mode from main thread with `nohup ... &`.
Already promoted to friction tag.

## PI-overrides (calibration data)

PI overrode agent direction twice this session:

1. **PI rejected K=24 Path B follow-up + framed as "small probes".**
   When agent finished Phase A composition gate (C1-C7 LR-meta TIE),
   agent recommended C7 + Path B as natural next step. PI: "These
   improvements that we are thinking about here and these steps that
   we are discussing now they are small, not so relevant. Let's try to
   find where to put our focus." This pivot triggered the strategy-
   critic-loop + research-loop that found the yekenot recipe. **Agent
   was about to grind on noise-floor; PI redirected to mechanism-
   level work.** Single most load-bearing PI override of the comp.

2. **PI sealed predictions consistently more conservative than agent:**
   - H1 PI 0 vs agent +27 (PI win on 3-variant phase; later both beat
     by full-recipe +19.6)
   - H2 PI +5 vs agent +3.6 (both overshot ~5 bp)
   - H3 PI 0 vs agent +0.6 (PI win exact)
   - H1d full-recipe FINAL: PI +10 vs agent +15.11 vs actual +19.6
     (both beat conservative; PI by ~10 bp, agent by ~5 bp)

   Net override count today: 1 (the focus pivot). PI/agent prediction
   delta is a calibration data-point, not an override.

## Frictions logged this session

See `audit/friction.md` 2026-05-07 PM section, 7 entries:
- `recipe-gap-misdiagnosis-when-public-author-FE-not-fully-replicated`
- `synthetic-augmented-driver-codes-cap-external-data-coverage`
- `synthetic-id-range-disjoint-but-decorrelated-from-target`
- `path-b-amp-only-fires-on-meta-arch-not-base-add` (6th cross-confirmation)
- `torch-set-num-interop-threads-once-per-process`
- `subagent-shell-children-die-on-subagent-exit`
- `kaggle-data-not-pre-pulled-on-fresh-branch-checkout`

## Promotion candidates (drafted; PI ratification skipped — wrap+merge mode)

### [ ] .claude/skills/kaggle-comp/do-and-dont.md — "Replicating a published recipe"

**Tag:** `recipe-gap-misdiagnosis-when-public-author-FE-not-fully-replicated`

**Where to insert:** new section after existing recipe-class items.

**What to add:**

```markdown
### Replicate the FULL notebook, not just the hyperparameter cell

When BOTE'ing a "replicate published author X's published recipe":
1. Read FULL notebook source (every code cell), not just the cell
   containing model hyperparameters.
2. Itemise EVERY FE step: arithmetic features, discretization,
   encoding (count / target / ordinal), combo cats, fold-level
   target encoding, per-fold data augmentation.
3. If the FE pipeline has 5+ items, treat hyperparameter
   replication alone as a NULL-likely probe. The +50-100 bp
   standalone gap is almost always FE pipeline, not hyperparams.

Origin: s6e5 H1 v1/v2/v3 burned ~3h on hyperparam-only replication;
H1d full-FE replication closed the gap in 35 min.
```

**Why:** s6e5 H1 v1/v2/v3 NULL across 3 variants; H1d full-recipe
+19.6 bp LB lift. ~3h compute waste on the 3 variants pre-diagnosis.

### [ ] .claude/skills/kaggle-comp/do-and-dont.md — "External-join EDA pre-flight"

**Tag:** `synthetic-augmented-driver-codes-cap-external-data-coverage`

**Where to insert:** new section under "before any external-data probe".

**What to add:**

```markdown
### EDA the join key BEFORE running the external-join probe

Before launching any FastF1 / Ergast / Pirelli external-join probe:
1. `train[key_col].value_counts()` head/tail to spot synthetic
   augmentation (D###/A###/etc placeholder codes).
2. If synthetic-augmented codes > 30% of rows, the join is bounded
   by the real-key subset. Either skip OR build for the real-key
   subset only with explicit cohort routing in the meta.
3. Adjust BOTE EV down by `1 - (1 - synth_pct)` (e.g. 60% synth
   → multiply EV by 0.4).

Origin: s6e5 H2 FastF1 hit 1.4% match rate due to 60% synth D###
codes. Agent BOTE was +3.6 bp at p=0.30; pre-flight EDA would have
downgraded to ~0.7 bp at p=0.10 = SKIP verdict.
```

**Why:** s6e5 H2 wasted 22 min compute + Kaggle slot (didn't submit
but ran full pipeline including FastF1 install).

### [ ] .claude/skills/kaggle-comp/examples/fe-recipe-yekenot-realmlp-kitchen-sink.md (DONE)

Already filed this session — full FE recipe with verified outcome
(OOF 0.95257 matched yekenot pub) for cross-comp use.

## Calibration snapshot (per Rule 26)

```
name                                     family                         actual    agent       PI  agent_err   pi_err
h3_id_shift_row_position                 single_base_fe_addition         +0.00    +0.60    +0.00      +0.60    +0.00
h2_fastf1_external_join                  external_data_aggregate         +0.00    +3.60    +5.00      +3.60    +5.00
h1_yekenot_realmlp_recipe                new_model_class                 +0.00   +27.00    +0.00     +27.00    +0.00
h1_yekenot_realmlp_recipe                new_model_class                +19.60   +27.00    +0.00      +7.40   -19.60
```

PI 2/4 by exact match (H2 was off +5 too); agent 0/4 (H1 had double
entry: outcome NULL on first record then UPDATE on full-recipe win).
**No PI stamp risk** — PI overrode agent direction once (focus pivot)
+ committed sealed predictions on every BOTE. Override rate > 0
satisfies anti-rubber-stamp.

## Framework version at session-end

- Commit SHA: `05c63d7` (last commit before this postmortem)
- Active rules: 1-26 (CLAUDE.md `## Top-level rules`)
- Loaded skills this session: `kaggle-comp`, `postmortem`,
  `update-config` (not invoked), `simplify` (not invoked),
  `claude-api` (not invoked).

## Net session outcome

🎯 LB **0.95345** = AT top-5% threshold. PRIMARY went 0.95089 (start
of session) → 0.95345 (end) = **+25.6 bp lift in single session**
(via sibling submits + my K=24 d18pool+h1d). My branch's contribution:
+19.6 bp via the K=24 stack (the BIGGEST single-submit lift of the
competition).

Permanent assets shipped:
- `external/kernels/ps-s6-e5-realmlp-pytabkit/VALIDATED.md` flagged
- `.claude/skills/kaggle-comp/examples/fe-recipe-yekenot-realmlp-kitchen-sink.md`
  filed (cross-comp portable)
- `audit/2026-05-07-d17-strategy-critique.md` + per-hypothesis audits
