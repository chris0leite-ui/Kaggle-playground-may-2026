# Postmortem — 2026-05-08 pca-k25-ensemble-cUKNt

Branch: `claude/pca-k25-ensemble-cUKNt`. Single-commit branch.

## What went wrong

- **Anchor cited from memory, off by 5.7 bp.** The agent cited the
  K=10+1 plain LR-meta OOF anchor as ~0.94850 in a PI-facing
  AskUserQuestion when designing the PCA-meta probe. The probe
  re-measured it at **0.95417** — 5.7 bp off the cited value.
  Strategic framing didn't change (relative deltas drive verdicts,
  not absolute level), but it's a discipline failure. Root cause:
  agent confused the K=10 LR-meta OOF with a different number
  elsewhere in the docs and didn't grep before pasting. Friction:
  `anchor-cited-from-memory-not-measurement`.
- **A25 → A30 inference was sloppy.** A25 measured the K=27 logit
  pool's *variance*-effective-rank as 3.23 (top 5 PCs capture 93% of
  variance). A30 inferred from this that the LR-meta sees a 3-D
  *predictive* ceiling. The PCA-meta probe shows the LR-meta's
  predictive content actually spreads across ~15 PCs (top-3 PCA-LR
  scores 0.95061, top-15 reaches 0.95401, ≈anchor). Original A25
  measurement was correct; the inferential leap from A25 to A30 was
  unjustified. Logged as A25b / A30b / A30c refinements in
  `ASSUMPTIONS.md`. Friction:
  `predictive-eff-rank-not-variance-eff-rank`.

Other items considered and **not** flagged:

- ISSUES.md leaf claim (Rule 18): probe was 8.4 min wall, just under
  the 10-min threshold. No claim required. Not friction.
- LightGBM hyperparameters not tuned. The gap to LR is consistent
  across input representations (PCs and raw expansions, K=10 and
  K=27), arguing against a hyperparameter-only explanation. A
  60-min Optuna sweep *could* shift the verdict by 1-2 bp but is
  unlikely to flip it. Caveat noted in audit, not friction.
- Bash sleep/poll behaviour. Harness blocked correctly when I tried
  `sleep 90 && tail`. Adapted to Monitor on second attempt. Not a
  framework issue.

## PI overrides this session

Zero. PI selected the 4-variant + K=10-anchor framing in the initial
question, then "wrap up" at end. No mid-session corrections, no
sealed prediction overrides.

This is the second consecutive 0/M postmortem (the prior 2026-05-08
session also had no PI overrides). Per WRAPUP.md step 5b, this is
the threshold to flag stamp-risk. **`pi-stamp-risk` flag candidate
for HANDOVER.md `## Where we are`** — but with 0 sealed predictions
this session, the override count isn't a meaningful signal here
(there was nothing for PI to override). Flagging in passing rather
than as a hard signal.

## Frictions logged this session

See `audit/friction.md` "Week of 2026-05-08":

- `anchor-cited-from-memory-not-measurement`
- `predictive-eff-rank-not-variance-eff-rank`

## Promotion candidates (PI ratified: NO)

PI declined promotion (2026-05-08 PM, "no promotion"). Candidate
recorded below for future reference but **not** propagated to
`.claude/skills/kaggle-comp/improvements.md`.

**Candidate 1 (NOT PROMOTED).** Extend Rule 26 with anchor-citation
hygiene.

```markdown
### [ ] CLAUDE.md — Rule 26 (iii): cite anchors, never paste from memory

**Tag:** `anchor-cited-from-memory-not-measurement` (PI-facing prose
discipline)

**Where to insert:** CLAUDE.md Rule 26 (PI interaction protocol),
extending the (i) Q6 + (ii) precedent enumeration with (iii).

**What to add:**
> Every numerical anchor pasted into a BOTE / AskUserQuestion / handover
> prose is either (a) cited by file:line from
> `state/calibration-ladder.md` or `audit/decisions.jsonl`, or
> (b) explicitly labelled "from memory, will re-measure". Never paste
> a bare number you didn't just look up.

**Why:** 2026-05-08 PM PCA-meta probe — agent cited K=10+1 LR-meta OOF
as ~0.94850 (memory) when actual is 0.95417 (5.7 bp off). Didn't
change the verdict but is a recurring class of error worth a hard
rule. The s6e4 postmortem family already had a related "stale-fact"
entry; this would compose with it.
```

No other candidates this session.

## PI additions (from step 4)

PI: "no addiction and no promotion" (2026-05-08 PM). Read as
"no additions, no promotion." No frictions added; promotion
candidate declined.

## Calibration snapshot (Rule 26)

`python scripts/probe.py calibration` output unchanged from prior
session. Today's PCA-meta probe was an exploratory diagnostic with
no sealed prediction, so no row was added. Recent rows:

| Probe | Family | Actual Δ bp | Agent Δ bp | PI Δ bp | agent_err | pi_err |
|---|---|---:|---:|---:|---:|---:|
| h1_yekenot_realmlp_recipe | new_model_class | +19.60 | +27.00 | +0.00 | +7.40 | −19.60 |
| d18_path_b_K23_d16_d18_tau20000 | external_data_aggregate | +6.00 | +1.26 | +3.00 | −4.74 | −3.00 |
| d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau10 | external_data_aggregate | +1.40 | +0.34 | – | −1.06 | – |
| d19_historical_priors_debashish | external_data_aggregate | +0.00 | +0.20 | −1.00 | +0.20 | −1.00 |
| b2_xgb_v4_K27_verify | pool_addition_redundant | +0.14 | +0.03 | – | −0.11 | – |
| a5_lgbm_v4_fs_K27_proxy | single_base_fe_addition | −0.11 | +0.10 | – | +0.21 | – |
| c1_yao_vehtari_path_b_K27 | meta_arch_redesign | −0.47 | +1.20 | – | +1.67 | – |

Pattern: PI is well-calibrated on small-effect families (close to
zero); agent runs +1 to +3 bp optimistic on novel mechanisms.
PCA-meta result (0 bp lift) is consistent with that pattern — agent
should have priced this lower than the implicit "could break the
ceiling" framing implied by EXP-NEW.

## Framework version at session-end

- Commit SHA: `97cf1be` (this branch).
- Active rules: 1..36 (CLAUDE.md). No rule changes this session.
- Loaded skills this session: `kaggle-comp` (passive, not invoked),
  `postmortem` (this artifact).
