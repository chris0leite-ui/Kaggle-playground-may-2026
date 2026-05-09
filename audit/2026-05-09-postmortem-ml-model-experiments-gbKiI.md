# Postmortem — 2026-05-09 ml-model-experiments-gbKiI

## What went wrong

Nothing structurally bad. The session ran on a PI directive ("show me original work, work all night autonomously, surprise me, do not replicate or copy") and produced one confirmed LB lift (V4 +0.8 bp), one structural finding (the conditional-target-correlation refinement of A29's rank-lock framing), and two clean negative results (V5 / V6 absorption at ρ ≈ 0.99 vs V4).

Two soft-cost items worth naming:

- **BOTE bookkeeping for Prongs B and S before PI axis-check.** The agent ran BOTEs for "per-segment LightGBM head" (Prong B) and "2-layer stack replication" (Prong S) before asking the PI whether replication-class work was in scope under the "be creative, original" directive. PI then ruled out replication. ~5–10 min of BOTE setup wasted; the BOTE notes themselves are at least retained as evidence the prongs were considered. Cheap miss; would have been free to ask first.

- **V-series leaf claim was retroactive.** Per Rule 18, an `open` ISSUES leaf should be claimed before compute ≥10 min. The V1–V6 sequence was multi-hour and never claimed mid-session — leaf 13 was added at wrap-up. Mid-session leaf claim would have improved Rule 18 hygiene without changing outcomes.

PI overrides this session (each is a calibration data-point, not a fault):
- "do not replicate or copy" — redirected the night from prong work to original V-series investigation.
- "do it" — submitted K=5 with Rule 27 override + start external-data work.
- "keep K=4 as primary" — reverted state/current.md after K=5 submission landed.
- "merge to main" — at wrap-up, after V6 closed.

PI was decisive in each case; no friction at the override level.

## Frictions logged this session

In `audit/friction.md` under "Week of 2026-05-08":

- `rank-lock-at-conditional-target-correlation-not-just-logit-direction`
- `transductive-feature-mechanism-one-dimensional-at-K4`
- `rule-27-abort-threshold-empirically-too-strict-for-sub-bp-moves`
- `bote-on-already-falsified-prong-direction-burned-time`

Cross-link: `audit/2026-05-08-night-session-summary.md`,
`audit/2026-05-08-seq-coupled-meta-probe.md`, ISSUES.md leaf 13.

## Promotion candidates (PI ratified: NO)

Three candidates were drafted and presented to PI for promotion to `.claude/skills/kaggle-comp/improvements.md`. **PI declined all three.** They remain logged as session frictions only.

1. **Rank-lock framing refinement — conditional-target-correlation level.** Future "untried mechanism at meta" candidates need to argue NEW partial correlation with y conditional on the existing pool, not just feature-space orthogonality. (Declined.)
2. **Rule 27 sub-bp calibration — ρ 0.999-0.9999 is not auto-tie.** K=5 V4 at ρ_test 0.99989 produced LB +0.8 bp. Override path: PI-authorised + calibration-probe framing + outcome logged. (Declined.)
3. **Pre-BOTE PI axis-check on creative directives.** When the directive is creative/original/free-form, ask the PI about axis-of-permission before running BOTEs on candidates that may fall in axes the agent isn't sure of. (Declined.)

## PI additions (from step 4)

PI: "Nothing to add."

## Framework version at session-end

- Commit SHA: `3432b5bd00540957268465ee209806ccdda9f33a`
- Active rules: 1..36 (per `CLAUDE.md` §"Operating rules — concise")
- Loaded skills this session: `postmortem` (this run); deferred tools used: `Monitor`, `AskUserQuestion`, `TodoWrite`, `ToolSearch`.

## Decision-quality scorecard (Rule 16 spirit)

Going by decision quality given pre-run priors (not outcome):

- **V1.1–V1.3 + V2.1–V2.3:** good decisions. The diagnostic showed sequence-coupled features had R² 0.487 vs row-local span; testing whether that escapes meta absorption was the cleanest argument the agent could construct. The negative result is informative.
- **V3 (kNN at meta):** good decision. Established the absorption baseline that V4 (kNN at base) then broke.
- **V4:** good decision. The standalone-AUC gap to K=4 (124 bp) was below the 300 bp friction threshold; the K=4+1 gate showed +0.24 bp consistent across 5 folds; production Path-B amp +0.20 bp; submit decision was PI-authorised. LB delivered +0.8 bp.
- **V5 (V4 + multi-cell TE):** good decision a priori — TE features ARE structurally different from kNN-target-mean. The ρ 0.989 absorption at meta was an empirical surprise worth knowing.
- **V6 (V4 + MLP-embedding kNN):** good decision a priori — task-learned similarity metric is structurally different from raw feature distance. Same ρ 0.988 absorption pattern. Two strong negative confirmations are worth more than one in establishing the "transductive features are one-dimensional" finding.

No decision in retrospect would be re-taken with the same priors. The session ran clean.
