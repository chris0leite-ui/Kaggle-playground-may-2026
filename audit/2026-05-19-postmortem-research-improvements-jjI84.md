# Postmortem — 2026-05-19 research-improvements-jjI84

## What went wrong

- **Bad decision — auth misdiagnosis (escalation before investigation).**
  Session-start 401 across every kaggle endpoint. Same priors at
  decision-time: kaggle.json on disk, kaggle CLI 2.1.2, KGAT_-prefixed
  key. I escalated to PI for a token refresh ("paste new kaggle.json
  JSON") instead of source-grepping the kaggle SDK. A 5-min look into
  `/usr/local/lib/python3.11/dist-packages/kaggle/api/kaggle_api_extended.py`
  surfaces `_authenticate_with_access_token` → `get_access_token_from_env`
  and the `KAGGLE_API_TOKEN` env var. With the same priors I would
  retake this decision: investigate the auth mechanism before
  escalating to the user. PI corrected with "the issue is with you" —
  1-turn calibration.

- **Rule-bypass — CSV ρ trusted on first staging despite known artifact.**
  Friction `pre-submit-diff-floor-clip` was logged 2026-05-18. I
  still initially trusted `pre_submit_diff.py`'s CSV-level ρ=0.998
  reading on a rank-uniform blend, leading to a "recalibrate the
  bands" detour before I caught the tie-structure-mismatch artifact.
  Same friction surfaced 2 days running → promotion candidate.

- **Rule 0 slip — letter-number codes in PI-facing chat.**
  First answer to "what would you submit next" used R7.2 / K=27
  shorthand. PI's prior "explain again simply" had already calibrated
  me; I should have led with "the 5-seed bagged PRIMARY + the older
  27-base wide-pool model" from the first sentence. Recovered mid-
  response. Not load-bearing — but the second occurrence of the
  same slip in 2 sessions; deserves logging.

- **Rule-gap — KGAT auth-method detection at session-start.**
  Kickoff-runbook had no step for "kaggle.json `key` starts with
  KGAT_ → that's an access token, not a legacy API key; the SDK
  needs it in `KAGGLE_API_TOKEN` env var via Bearer auth, not
  Basic auth via the `key` field". bootstrap.sh from 2026-05-18
  handles a related case (KGAT_ already in `KAGGLE_API_TOKEN`),
  but not the case where the only credential source is a
  kaggle.json with KGAT_ in the legacy `key` slot. Promotion
  candidate A addresses this.

## Frictions logged this session

Cross-link to `audit/friction.md` 2026-05-19 block:

- `rank-lock-confirmed-four-axes` — R10 multi-constituent alt-stack
  closed (4th rank-lock axis). Origin of hedge-prep pivot.
- `csv-rho-misread-on-floor-clip` — pre_submit_diff CSV ρ misread
  on floor-clipped reference; pre-staging mis-binned 5 of 6 blends
  until .npy ρ recompute. **Promoted as candidate B.**
- `blend-sweep-r72-dominance-tie` — top-by-OOF blend candidates
  were all R7.2-dominated TIE_ZONE; cross-pool diversity only
  surfaced when K=27 included. PI declined promotion (C).
- `kaggle-cli-kgat-auth-misdiagnosed` — KGAT_ tokens need Bearer
  auth via `KAGGLE_API_TOKEN` env var, not legacy `key` field in
  kaggle.json. **Promoted as candidate A.**
- `hedge-3-lb-confirms-ok-band` — calibration data point: OK-band
  lower-boundary .npy ρ=0.999882 → −0.02 bp LB delta. Not a rule;
  recorded in `state/current.md` and the hedge-ladder update log.

## Promotion candidates (PI ratified)

- **A — `kgat-auth-detection`**: ratified `yes`. Entry added to
  `.claude/skills/kaggle-comp/improvements.md` under the new
  "Promoted 2026-05-19" section with status `[ ]` (actionable;
  pending bootstrap.sh + kickoff-runbook implementation edits).
- **B — `pre-submit-diff-rank-normalize`**: ratified `yes`. Entry
  added at the same section with status `[ ]` (actionable; pending
  edit to `scripts/pre_submit_diff.py`).
- **C — `blend-rho-band-first`**: ratified `no` (PI declined).
  Friction `blend-sweep-r72-dominance-tie` remains logged in
  `audit/friction.md` for context but is not promoted to a skill
  rule.

## PI additions (from step 4)

PI did not surface additional frictions or rule-extraction
candidates beyond ratifying A and B (and declining C). No additional
flags logged this turn.

## Submission outcome

Single submit this session: `submission_R10_blend_R72_K27_arith_75_25.csv`
(HEDGE 3, 75/25 arith blend of R7.2 5-seed bag + K=27 wide-pool).
**LB 0.95387** vs PRIMARY (R7.1) 0.95389 = −0.02 bp delta, well
within R2d's 30-bp regression cap. First cross-mechanism diversity
hedge of the comp confirmed at the OK-band rho boundary
(.npy ρ=0.999882). HEDGE-3 slot in `state/hedge-ladder.md` is now
LB-confirmed; HEDGE 2 (TIE_ZONE) and HEDGE 5 (TIE_ZONE) remain
staged-only and deferred to the final-window May 28-31 if needed.

## Calibration snapshot (Rule 26)

`scripts/probe.py calibration` ran clean; last several entries have
agent BOTE recorded but blank PI predictions (`pi_err = –`),
consistent with the post-Day-19 Rule-26a sealed-prediction protocol
removal. Last 3 PI-scored entries: median |pi_err| ≈ 5 bp,
|agent_err| ≈ 4 bp — agent slightly tighter but small N. No two-
consecutive postmortems with 0/M overrides yet → no `pi-stamp-risk`
flag triggered.

## Framework version at session-end

- Commit SHA (state-edits, pre-postmortem-commit):
  `c356b9517471aeaa1e58c7f21816a479d953ec68`
- Active rules: R0..R36 + R1d/R2d/R5d/R7d/R8d (per `CLAUDE.md`
  current top-level rule families).
- Loaded skills this session: `kaggle-comp`, `postmortem`.
- Branch: `claude/research-improvements-jjI84`.
- Submissions used today: 1 / 10 daily cap (9 unused at session
  end, forfeit at next UTC midnight per Rule 12 — PI declined
  spending TIE_ZONE candidates).
