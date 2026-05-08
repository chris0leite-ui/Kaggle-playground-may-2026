# Postmortem — 2026-05-08 research-feature-engineering-7oCmj

**Branch:** `claude/research-feature-engineering-7oCmj`.
**Scope:** EXP-NEW Phase 1-5b FE/meta funnel campaign on K=4 + Path-B
Compound × Stint τ=100k PRIMARY. Closed `null`.

## What went wrong

**Bad decision (caught pre-launch, cost ≈ 0).**
`research-scan-duplicate-mechanism-claim`. Rule 7 saturation scan
fired correctly; agent ran two web searches + WebFetch on Frontiers
AI 2025 ("Data-driven pit stop decision support for Formula 1 using
deep learning models"), built a 5-mechanism table, ranked
`DriverAheadPit`/`DriverBehindPit` as #1 — without first grepping
the FE pick registry. A3-1 RankSortedGaps (registered in
`fe_picks_a2a3.py`) already implements both `_ahead_pitted_lag1`,
`_behind_pitted_lag1`, gap_to_car_ahead/behind, and tirechange_pursuer.
A3-1 had nulled in Phase 1 smoke roughly 30 minutes earlier in this
same session. Self-corrected before launching the probe via PI
question construction. Cost would have been ~10 min CPU + audit-log
pollution; cost of catch was nil.

**PI overrides:** zero this session. PI sequenced through 3 question
prompts (Path-B amp test → A2-8 stack-meta → research probe → wrap
up), each consistent with the agent's recommendation. No corrective
overrides.

**Rule-bypass failures:** none.
- Rule 19 (calibration log) applied for both new probes via
  `probe.py record-outcome`.
- Rule 7 (saturation triggers research scan) fired correctly at
  3 nulls + 1 weak.
- Rule 18 (claim ISSUES leaves) — leaves 11a-e all closed in this
  wrap-up.
- Rule 32 (session-start git fetch) ran at session start.

**Rule-gap failures.** The research-scan duplicate-claim is a real
new pattern but PI ratified `friction-only` retention; not promoted
to `improvements.md` this session.

## Frictions logged this session

`audit/friction.md` Week-of-2026-05-08 appends:

- `research-scan-duplicate-mechanism-claim` — top of the section.
- `tree-stack-meta-overfits-small-K-pool` — A2-8 LightGBM stack-meta
  on K=4 lost −1.30 bp vs Path-B PRIMARY and −0.96 bp vs plain LR;
  fold-std 0.00080 (vs typical ~0.00050). Generalisable: convex LR
  + Path-B partial-pooling regularize better than gradient boosting
  at small K.

## Promotion candidates (PI ratified: NO)

Drafted: `guardrails.md / Rule-7-amendment — research-scan must
grep FE registry first` (tag
`research-scan-duplicate-mechanism-claim`).

PI ratified `No — keep in friction only`. Rationale: the pattern is
real but hasn't shown up enough across comps to merit promotion to
the cross-comp improvements log yet. Will revisit if it recurs.

## PI additions (from step 4)

None. PI ratified `Nothing to add; proceed`.

## Calibration snapshot (Rule 26)

Per `python3 scripts/probe.py calibration` at session-end. The new
record-outcome entries from this session
(`a2_2_K4_LR_meta`, `a2_2_mandatory_compound_rule_pathb`,
`a2_8_stack_meta_K4`) live in `audit/decisions.jsonl`; they don't
show in the calibration table yet because they weren't logged via
the agent-vs-PI sealed-prediction protocol — they were Phase-4 /
Phase-5b probes whose outcomes were documented but where the
relevant predictions were registered earlier under
`a2_2_K4_LR_meta`. Snapshot below covers historical anchors:

```
name                                     actual   agent     PI    agent_err  pi_err
h3_id_shift_row_position                  +0.00   +0.60   +0.00     +0.60    +0.00
h2_fastf1_external_join                   +0.00   +3.60   +5.00     +3.60    +5.00
h1_yekenot_realmlp_recipe                 +0.00  +27.00   +0.00    +27.00    +0.00
h1_yekenot_realmlp_recipe                +19.60  +27.00   +0.00     +7.40   -19.60
probe_combined_lead_lag                   +0.00   +0.03   +0.00     +0.03    +0.00
probe_field_state                         +0.00   +0.03   +0.00     +0.03    +0.00
probe_field_state                         +0.00   +0.03   +0.00     +0.03    +0.00
d18_path_b_K23_d16_d18_tau20000           +6.00   +1.26   +3.00     -4.74   -3.00
d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau10  +1.40   +0.34      –      -1.06       –
d19_historical_priors_debashish           +0.00   +0.20   -1.00     +0.20   -1.00
b2_xgb_v4_K27_verify                      +0.14   +0.03      –      -0.11       –
a5_lgbm_v4_fs_K27_proxy                   -0.11   +0.10      –      +0.21       –
c1_yao_vehtari_path_b_K27                 -0.47   +1.20      –      +1.67       –
a3_7_uid_smoothing                      -124.00   +0.30      –    +124.30       –
```

Stamp risk: zero PI overrides for this session, but two prior
postmortems also had zero. Per WRAPUP 5b friction tag
`pi-stamp-risk` candidate would surface to HANDOVER.md after a third
consecutive zero-override session — flagging for next session's
postmortem to verify.

## Framework version at session-end

- Commit SHA at start: `cc8205e` (will be superseded by this wrap-up
  commit).
- Active rules: 1..36 per `CLAUDE.md ## Operating rules`.
- Loaded skills this session: `kaggle-comp` (implicit via CLAUDE.md
  rules), `postmortem` (this run).
