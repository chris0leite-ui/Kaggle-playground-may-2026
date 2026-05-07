# Postmortem — 2026-05-07 review-handover-solutions-oE78b

`branch: claude/review-handover-solutions-oE78b`

## What happened

PI prompt: "What are the simple solutions, the simplest way to learn
something about our problem, that is hiding in plain sight?"

Five probes ran across the session, all diagnostic, no LB submits.
PI sealed prediction = 0 bp LB Δ for every probe (Rule 26a). Every
probe returned 0 bp realised LB (no submission attempted).

| # | Probe | Cost | Standalone OOF | K-stack outcome |
|---|---|---|---|---|
| 3 | NTL single-rule | 5 min | best AUC 0.687 (< raw TyreLife 0.699) | n/a (single-feat) |
| 2 | Target structure EDA | 30 min | n/a (EDA only) | n/a |
| 1 | Combined-frame lead/lag | 30 min | +2.18 bp full / +2.55 bp train-only | combined-frame premium -0.36 bp |
| 4 | Field-state cross-row aggregates | 12 min | +15.58 bp full / +13.73 bp strict per-fold | K=24 stack-add Δ -0.015 bp |
| 4-strict | Strict fold-safe re-run of #4 (per PI flag) | 7 min | +13.35 to +13.73 bp | n/a |

Probe 4 is the structural find of the session: cross-row aggregates
over (Race, Year, LapNumber) and (Race, Year, LapNumber, Compound)
computed from train+test combined. PitStop is a feature column not
the label (single-feat AUC vs target = 0.521 ≈ chance per U2). Top
single-feat: `fs_cum_pits` (cumulative race pit count) AUC 0.7972 —
highest single-feat OOF on this comp (raw TyreLife alone 0.6989).

## What went wrong (descriptive, not consequential)

**Bad decisions:** none material. All five probes followed framework
discipline (BOTE before code, sealed-prediction protocol, gate after
artifacts). Probe 1 and 3 hypotheses were falsified — that's a
diagnostic outcome, not a bad decision.

**PI overrides:** 1.
- After probe 4's standalone +15.58 bp result was committed and the
  audit note declared it a "structural find", PI flagged: "isn't this
  a feature we tested already and discarded due to leakage? Check."
  Agent response: distinguish PitStop=feature (single-feat AUC 0.521
  ≈ chance vs PitNextLap per U2) from Day-17 P1 v2 FS_A merge that
  used `df[df.PitNextLap==1].groupby(...).mean()` (label-derived).
  Then ran strict per-fold re-run (`probe_field_state_strict.py`):
  full-train +15.58 bp → strict-per-fold +13.73 bp (12% loss vs
  Day-17 leakage signature 88-100%). Audit cleared. PI flag was
  load-bearing — without it the OOF lift would have been
  over-claimed.

**Rule-bypass failures:** none.

**Rule-gap failures:** the BOTE family prior `single_base_fe_addition`
in `scripts/probe.py` (p=0.05, band [0, 0.5, 2.0] bp) was calibrated
on row-feature NULLs. Cross-row aggregate features fired +15 bp
standalone OOF (30× the optimistic band). The framework correctly
predicted +0 bp at LB transfer (K=24 stack-add was -0.015 bp, 6th
cross-confirmation of `lr-meta-rank-lock-strong-anchor`) but the
standalone-OOF prior was off by 30×. Friction tag captures this;
not promoted per PI direction.

## Frictions logged this session

Cross-links to `audit/friction.md ## 2026-05-07 PM (branch
claude/review-handover-solutions-oE78b)`:

- `cross-row-aggregates-fire-where-own-row-sequence-doesnt`
- `cross-row-aggregates-survive-strict-fold-safe-audit`
- `field-state-mechanism-fires-on-train-only-too-no-combined-premium`
- `family-prior-single-base-fe-addition-mis-calibrated-for-cross-row`
- `lr-meta-rank-lock-strong-anchor` (6th cross-confirmation)
- `host-quote-trivial-refers-to-original-not-reconstructible`
- `pitnextlap-target-cluster-decay-not-shift`
- `combined-frame-leadlag-premium-evaporates-at-gbdt`

## Promotion candidates (PI ratified)

PI direction: "I would be careful with drawing conclusions rather
describe what happened, and then we can later think about what
consequences it has. The rule to avoid leakage is a good one. We
can take that. The rest for now, you can note under friction, but
we will not promote."

- **Promoted:** G18 strict-fold-safe variant before treating
  group-aggregate OOF as honest. Added to
  `.claude/skills/kaggle-comp/improvements.md` after G17.
- **Not promoted (kept in friction only):** family-prior split for
  cross-row aggregates; sealed-prediction weighting at LB-ceiling;
  integrate-into-anchor-not-on-top mechanism rule; cross-row
  aggregates as a generalisable axis check.

## PI additions (step 4)

PI declined to add additional frictions. Direction was to remain
descriptive at this point and revisit consequences later.

## Calibration snapshot

Per `python scripts/probe.py calibration` at session-end:

```
name                                family                          actual    agent       PI
h3_id_shift_row_position            single_base_fe_addition         +0.00    +0.60    +0.00
h2_fastf1_external_join             external_data_aggregate         +0.00    +3.60    +5.00
h1_yekenot_realmlp_recipe (v1-3)    new_model_class                 +0.00   +27.00    +0.00
h1_yekenot_realmlp_recipe (v1d)     new_model_class                +19.60   +27.00    +0.00
probe_combined_lead_lag             single_base_fe_addition         +0.00    +0.03    +0.00
probe_field_state                   single_base_fe_addition         +0.00    +0.03    +0.00
```

PI overrides this session by override-rate (per Rule 26e): 1/5 probe
recommendations corrected (probe 4 leakage flag); 4 alignments. Two
consecutive postmortems with override-rate ≤ 0/M would trigger
`pi-stamp-risk` flag in HANDOVER; current session has 1 override so
not flagged.

## Framework version at session-end

- Commit SHA: `e28db35`
- Active rules: 1..26 (CLAUDE.md `## Top-level rules`)
- Loaded skills this session: `kaggle-comp`, `postmortem`
- Branch: `claude/review-handover-solutions-oE78b` (top of
  `52b00f2 → ca40a76 → 22ffbc0 → 0241693 → c548fda → 1ca661a → e28db35`)
