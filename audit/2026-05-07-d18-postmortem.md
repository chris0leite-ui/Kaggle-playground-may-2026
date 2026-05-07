# 2026-05-07 PM postmortem — d18 reverse-engineering DGP arc

`branch: claude/reverse-engineer-data-generation-Hu8EK`
`session: Day-17 PM through Day-18 PM-late (multi-session continuation)`

## Where we landed

- **🎯 NEW PRIMARY: LB 0.95368** (K=27 Path-B Compound×Stint τ=100k).
  +1.4 bp over previous 0.95354 (main's K=23 v4+h1d).
- **Top-5% gap closed**: −5.1 bp → **−3.7 bp** (boundary 0.95405).
- **Submissions used today**: 8 across both branches; total 39/270.

## What went well

1. **DGP-reverse-engineering pivot delivered**: 14 mechanism probes
   (E1-E5, F1-F6, G/H/I/J/K1-K3) characterised the synthesizer
   end-to-end:
   - Architecture identified: CTGAN-class GAN with custom signature.
   - Mode-specific normalization confirmed (97.55% literal LapTime).
   - Class-conditional generator (PitStop in cond-vector) quantified.
   - Within-row features near-conditionally-independent.
2. **Clean idea-board lock-in**: 14 queued probes documented BEFORE
   running, survived through 4 session-cap interruptions.
3. **Combined-with-main strategy worked**: pulled v4+h1d artifacts via
   `git checkout origin/main -- ...`, ran greedy stack-add, identified
   d16/d18/E2/F2 as the 4 marginally-contributing DGP-class bases, ran
   K=25/26/27 Path-B sweep, submitted K=27 → LB 0.95368.
4. **Calibration loop functional**: 2-of-2 submissions in predicted
   band (d18 K=23 +6 actual vs +5 agent / +3 PI; K=27 +1.4 actual vs
   +0.34 agent / unrecorded PI).

## What went wrong / friction

1. **Parallel-LGBM-3-way OOM**: F2/F5/G procs SIGHUP'd mid-fold-2
   under 3-way CPU contention. Wasted ~3 h CPU. Sequential chain
   ran 30× faster. (Friction tag: `parallel-lgbm-3way-contention-oom`)
2. **F1 GPU kernel iterated 4 versions**: sdv `--no-deps` missing
   sdmetrics → numpy pin → `/kaggle/input/...` path bug → P100 sm_60
   torch 2.4 fix → discriminator design correction. Each iteration
   cost ~1 h Kaggle GPU. Pre-flight dependency check would've saved
   2-3 h.
3. **DAE wildcard fizzled**: predicted +1-3 bp solo on top of v4+h1d;
   actual +0.16 bp. RealMLP h1d's representation layer absorbs the
   unsupervised manifold signal that DAE captured.
4. **Mode-id (G/H/I) fully absorbed by CatBoost's CTR**: G +0.07,
   H +0.03, I +0.05 on top of v4+h1d. The CTGAN-aware abstraction
   layer was already covered by v4's combo-cat CTR.

## PI overrides

0/2 PI overrides this session (both submissions PI-authorized after
sealed-prediction recording, 1 of 2). Per Rule 26e: 1 postmortem at
0/M overrides. Not yet stamp-risk; one more at 0/M would trigger flag.

## Calibration snapshot

```
name                                              family                   actual   agent     PI  agent_err  pi_err
d18_path_b_K23_d16_d18_tau20000                   external_data_aggregate   +6.00   +1.26  +3.00     -4.74   -3.00
d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000      external_data_aggregate   +1.40   +0.34    –       -1.06     –
```

Agent error pattern: under-estimating LB lift by 1-5 bp. Family prior
(0.20 P-useful, central +1 bp band) is calibrated against historical
17% hit-rate but the recent submissions are landing in optimistic
half of band. Consider widening optimistic for `external_data_aggregate`
when pool is augmented mid-session.

## Promotion candidates from friction

| Tag | Promotion candidate | PI to ratify? |
|---|---|---|
| `pool-saturation-v4h1d-absorbs-dgp-class` | Generalises main's `pool-saturation-v4h1d-absorbs-d16d18` to all DGP-class bases including CTGAN-aware mode-id. Add to `~/.claude/skills/kaggle-comp/improvements.md` as: *when a strong public-recipe single-model lands (e.g., yekenot RealMLP), it ABSORBS most of the orthogonal-feature-engineering signal you built earlier; budget ≤+1-2 bp from adding earlier bases on top.* | yes if approved |
| `parallel-lgbm-3way-contention-oom` | Already partially in `improvements.md` (cap ≤3 concurrent). Reinforce with concrete CPU-time-multiplier fact: 3-way LGBM contention = 30× slower than sequential. | nice-to-have |
| `sequential-axis-untouched` | Adds a NEW general principle: *if a Kaggle dataset has implicit grouping/sequence structure (e.g., (group_key, time) tuples), GAN-class synthesizers will likely break sequence coherence; targeted sequence-fingerprinting is high-EV regardless of LB outcome.* | yes if approved |

## Process learnings for next session

1. **Pre-flight idea-board** (this session's contribution): write the
   ranked queue BEFORE executing. Survived 4 session-cap interruptions
   intact. Promotion to skill: yes.
2. **Sequential chain via `setsid nohup bash -c 'python a; python b'`**:
   pattern works for chains up to 2 h total; survives session caps if
   each sub-step is <30 min.
3. **Pull-from-main artifacts via `git checkout origin/main -- path`**:
   without merging, brings collaborator outputs to current branch.
   Used twice in this session (F1 v4 fix + v4/h1d pull). Cross-branch
   research workflow.

## What we still don't know about the DGP

Updated idea-board (in HANDOVER):
- 7g sequence-level DGP fingerprinting (HMM on Compound transitions
  + AR(1) on within-stint TyreLife)
- 7h cross-feature joint mode-id (mode-tuple as GAN's discrete latent)
- 7i membership inference / exact-row copy detection
- 7j class-conditional CTGAN replay with explicit cond-vector spec
- 7k per-Year DGP heterogeneity probe

## Pointers (audits added this session)

- `audit/2026-05-07-d18-chain-decomp.md` — E1 d18 chain v1 (the win)
- `audit/2026-05-07-d18-dgp-decomp-batch.md` — E1-E5 batch synthesis
- `audit/2026-05-07-d18-ideaboard.md` — locked queue 14 probes
- `audit/2026-05-07-d18-tier1-ctgan-batch.md` — G/H/I/J/K + F2/F5
- `audit/2026-05-07-d18-postmortem.md` — this file
- `audit/decisions.jsonl` — 4 BOTE entries + 2 outcomes
- `kernels/d18-f1-replay-forensics-gpu/` — F1 replay forensics kernel
