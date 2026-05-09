# 2026-05-09 — Postmortem: decode-data-process-5uLq3

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-postmortem`
`session window: 2026-05-09 ~09:20 → ~12:30 UTC`

> Streamlined postmortem (PI ratification deferred — direct merge
> requested).

## What I tried to do

Decode the host's data-generating process for s6e5: learn the function
that maps aadigupta1601's 101k-row file to Kaggle's 627k-row synth
file. Per PI: focus on understanding the DGP, not the LB.

## What I did

Ran 23 probes (Q1-Q10 + qB-qZ) over four phases — fingerprint
refinement, architecture exclusion, analytic conditional resampling,
and a free LB bridge:

1. **Q1+Q2 fingerprint refinement** (~10 s CPU): re-validated F1+F5,
   updated F4 (2023 anomaly is in orig itself), and surfaced four new
   findings (F7-F10) about column-class taxonomy, custom marginals,
   PitStop-conditional asymmetry, and synthesised `Cumulative_Degradation`.
2. **Q6+Q7 retraction** (~1 min CPU): proved the prior P1c
   per-row-literal-copy interpretation wrong. Match rate decays from
   0.9755 (K=1) to 0.0000 (K=6). Synth columns are sampled
   near-independently within cells.
3. **Q3+Q5 marginal hypothesis falsified** (~5 min CPU): SDV CTGAN
   on orig with synth's `(Year, Compound, PitStop)` marginal still
   gives disc-AUC 0.9993. The marginal customisation is real (Q10)
   but is not the load-bearing axis of F6.
4. **qB architecture grid** (~10 min CPU): SDV GaussianCopula 0.9988,
   TVAE 0.9991. Combined with CTGAN's 0.9993 and CopulaGAN (killed
   mid-eval), the SDV library is excluded.
5. **qF-qM analytic resample sweep** (~5 min CPU total): from uniform
   scramble (0.9716) to conditional Driver/Stint on `(Y, C, PS,
   Race, Stint, LapNumber)` (0.7160). −28 pp gain.
6. **qO-qY per-cell continuous-density sweeps** (~5 min CPU total):
   BGMM (0.86), KDE (0.74-0.77), cross-cell mixing (0.72-0.98),
   moment-matching (0.99). All worse than literal orig values.
7. **qZ d16++ free LB probe** (~25 s CPU): trained LightGBM on orig
   with the qM cell-key features, standalone synth-train AUC 0.93985
   (+2.5 pp over current d16). At K=4+1 LR-meta gate +0.149 bp; ρ to
   PRIMARY 0.93. Rank-lock cap holds.

## What worked

- **Sequential analytic decomposition.** Each probe isolated one axis
  of the host pipeline. The disc-AUC ladder (0.999 → 0.97 → 0.83 → 0.72)
  with each step pinned to a specific structural insight is the
  cleanest evidence we have for the host's pipeline.
- **Free LB bridge worked at the standalone level.** qZ d16++ proves
  the decoded cell-key features carry information d16 doesn't already
  have (+2.5 pp standalone). The K=4+1 meta cap is rank-lock, not
  feature quality.
- **Falsified frame helps.** The Q6+Q7 retraction of P1c was
  load-bearing — once we stopped looking for per-row literal copies,
  the analytic per-cell pipeline became obvious.
- **Iterative committing.** 22 commits in 3 hours; each commit
  includes the audit JSON + script + audit doc. Easy to reconstruct
  any probe's reasoning from the commit message alone.

## What didn't work

- **Two CTGAN process zombies.** Bash `2>&1 | tail -50` plus
  `run_in_background: true` left orphaned python processes when the
  parent bash exited. Discovered via `ps` after a second run-attempt
  showed a zombie still consuming all 4 cores. **Fix:** use
  `nohup python -u ... > log 2>&1 &` and tail the log file directly
  instead of `| tail -N`. Promoted to friction.
- **qS forced-cat CTGAN was killed at 8 min.** Forcing Driver / Race /
  LapNumber / TyreLife (all > 20 distinct values) as categorical
  causes SDV CTGAN to balloon embedding dim and stall in metadata
  setup. The smaller-scope qS2 (5 cols only) was killed at 8 min
  because the 30-epoch full-run took too long.
- **qY moment-matching catastrophic.** Naive affine rescaling drove
  disc-AUC from 0.7160 to 0.99 — skewness diffs in the qX p90 = 70
  range are non-Gaussian, so affine fixes break the shape. Should
  have predicted this from qX skew columns before launching qY.
- **synthcity install failed.** Six wheels failed to build (nflows,
  arfpy, pykeops, keopscore, autograd-gamma, feather-format). Would
  have unlocked TabDDPM and other generators. **Fix:** add
  synthcity to a requirements lockfile if a future session has
  time to debug.

## Calibration

I did not run `scripts/probe.py calibration` in this session because
the work was decode-focused (not LB-submission-focused). No
submissions were made today; no overrides occurred. Calibration
log update is the next session's concern.

## Friction tags promoted (added to `audit/friction.md` Week of 2026-05-08)

1. `synth-rows-are-not-literal-copies-of-orig-rows` — retract P1c.
2. `host-not-in-sdv-library` — skip SDV variants in future sweeps.
3. `noise-on-continuous-cols-makes-disc-worse-not-better` — start at
   sigma=0 for any future host-match attempt.
4. `cond-driver-stint-on-cell-saves-14pp` — default any analytic
   resample to per-cell empirical for fabricated categoricals.
5. `extending-cond-axes-monotonic-down-to-LapN-then-sparsity-bites` —
   cell key sweet spot is six axes; don't push past LapNumber on this
   dataset.
6. `affine-moment-matching-fails-skewness-non-trivial` — skew-
   sensitive tools (NF, copula on quantile transforms) over
   BGMM/affine for per-cell density.
7. `host-cont-vals-strictly-per-cell-no-cross-cell-mixing` — host's
   continuous-value generator is strictly cell-conditioned.
8. `rank-lock-saturation-puts-cap-on-K4plus1-with-decode-features` —
   future LB lift needs different meta arch, not new bases.

## Promotion candidates for `~/.claude/skills/kaggle-comp/improvements.md`

Deferred — PI ratification not requested in this wrap. Candidate
lessons that survive:

- **"Decode synth-vs-orig disc-AUC ladder is the right diagnostic
  for tabular synthesisers."** Each architectural axis you exclude
  drops the disc-AUC by a measurable bp. SDV exclusion + analytic
  resample-and-cond is the canonical first move.
- **"Per-cell continuous-density gen is the hard residual on
  Playground-class synth."** Even with full categorical structure
  decoded, the per-cell generator residual is ~22 disc-AUC. Plan for
  this in any future Playground decode.
- **"Free-LB probe = orig-trained model with decoded cell-key
  features."** Cheap; produces a standalone uplift that may or may
  not survive rank-lock at the meta layer.

## Mission outcome

**Decode work complete to structural level.** The host's pipeline is
now characterised in five steps (input → custom marginal → per-cell
NN gen → structured Driver/Stint → drop Norm_TyreLife → output) with
disc-AUC 0.7160 (lower bound 0.4944). The remaining 0.22 disc-AUC
gap is the per-cell NN generator that none of our 11 ruled-out
mechanisms reproduce.

**LB position unchanged.** No submissions made; PRIMARY remains K=4 +
Path-B C×S τ=100k at LB 0.95351. qZ artifacts (d16++ OOF + test +
train_synth predictions) saved for next-session stack-add.

## Pointers

- Read order:
  1. `audit/2026-05-09/2026-05-09-EXEC-SUMMARY.md` — PI-readable.
  2. `audit/2026-05-09/2026-05-09-PHASE-B-FINAL-and-plan-v3.md`.
  3. `audit/2026-05-09/2026-05-09-HANDOVER.md`.
- 23 probe scripts: `scripts/dgp_v3/q[1-9]*.py, q[B-Z]*.py`.
- 22 audit-JSON artifacts: `scripts/artifacts/dgp_v3_q*.json`.
- qZ artefacts: `scripts/artifacts/dgp_v3_qZ_oof_strat.npy`,
  `dgp_v3_qZ_test.npy`, `dgp_v3_qZ_train_synth.npy`.
- 12 audit markdown docs: `audit/2026-05-09/`.
