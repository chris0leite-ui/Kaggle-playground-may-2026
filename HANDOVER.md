# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 14 (2026-05-14)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `scripts/probe.py` — entry point. `bote()` for BOTE, `gate()` for uniform gate report
3. `scripts/probe_min_meta.py` — K=21+N stack-add gate
4. `audit/2026-05-06-blend-and-rho-probes.md` — most recent rule-out + ρ inventory
5. `audit/2026-05-06-alpha-asymmetry-verification.md` — Path B α-asymmetry verified
6. `audit/2026-05-13-d13-path-b-hier-meta.md` — Path B mechanism (load-bearing)
7. `audit/2026-05-13-d13d-path-b-gkf-probe.md` — GKF amplification confirms private-robust
8. `audit/2026-05-12-d12-master-synthesis.md` — leakage-robust thesis
9. `scripts/pre_submit_diff.py` — MANDATORY before submit

Open with a 3-bullet read-back of state + first action.

**Harness usage cheatsheet (Rule 19):**
```bash
# BEFORE writing code for a candidate ≥10 min CPU:
python scripts/probe.py bote NAME --family X --cost_min N \
    [--std_oof_lift_bp Y] [--prob_useful U] [--note "rationale"]

# AFTER artifacts exist (under scripts/artifacts/oof_<NAME>_strat.npy):
python scripts/probe.py gate NAME \
    --oof scripts/artifacts/oof_NAME_strat.npy \
    --test scripts/artifacts/test_NAME_strat.npy

# Stack-add probe (K=21 + candidate(s)):
python scripts/probe_min_meta.py --candidates NAME1 NAME2 ...
```
Family priors are in `scripts/probe.py FAMILY_PRIORS`. Rule-out is
a valid result; cheap NULL findings get audit notes too.

## Where we are (Day 14 morning)

- **PRIMARY** = `d13_path_b_stint_tau100000` LB **0.95041** (NEW Day-13; +7bp from prior PRIMARY).
- **HEDGE** = `d9h_K22_add_aug12` / `d9i_S1_K21_swap_aug2way` (TIED) LB 0.95034.
- **Gap to top-5%** (0.95345): **30.4bp** (narrowed from 31.1bp).
- **13 days remaining** (deadline 2026-05-31). 9 slots/day.
- **Submits used**: 23/270 total; Day-13 4/9 (V1 5/3 multi-FM TIE LB 0.95032; d13c Compound τ=100000 LB 0.95033; d13 Stint τ=100000 LB 0.95041 NEW PRIMARY; d13a S3 K=24 TIE LB 0.95032).

## 2026-05-06 PM addendum (branch `claude/ml-handover-alignment-xvUN0`)

PI redirect: experimentation culture; many small probes; BOTE-first;
"the solution is probably simple, maybe a code-quality fix"; no
agent-side calendar tracking ("PI calls the day").

**Built:** experimentation harness — `scripts/probe.py` (`bote` +
`gate`), `scripts/probe_min_meta.py` (K=21+N stack-add gate),
`scripts/probe_blends_K21.py`, `scripts/probe_rho_inventory.py`.
Family priors calibrated against empirical 17% advance hit rate.
Rule 19 added to CLAUDE.md codifying BOTE-first / gate-after.

**Cheap probes (~5–10 min each, all via harness):**

1. **α-asymmetry verification (`audit/…alpha-asymmetry-verification.md`).**
   Audit-agent claim of "+2-5bp from a hier-meta bug" verified
   structurally but **severity overstated** — per-fold uses fold-train
   counts in α=n/(n+τ); test uses full-train counts. This is
   Bayesian-correct shrinkage, NOT a fixable LB cap. Real
   implication: OOF AUC reflects a more-shrunk model than test
   ships → τ chosen on OOF may not be τ-optimal for test. Probe
   surfaced (PURSUE per BOTE): α-calibrated τ-resweep, ~30 min.

2. **K=21 simple-blend probe (`audit/…blend-and-rho-probes.md`).**
   mean / gmean / rank_mean / trimmed all regress 19–32 bp
   standalone vs PRIMARY. **Hypothesis "LR meta is over-weighting
   bases, simple blend would help" RULED OUT.** rank_mean shows
   asymmetric flip pattern (38 demote / 4470 promote) — interesting
   but not hedge-grade.

3. **ρ inventory of 22 held candidates.** Buckets: 2 TIE_EXPECTED
   (any d13e τ variant), 10 near-tie, 10 diverse. Cleanest near-tie
   HEDGE candidate: **`d12_lr_meta`** (OOF 0.95073, ρ=0.996, flip
   ratio 0.297). `d12_groupkf_meta` evaluation-space mismatch
   flagged (constructed under GKF; needs `--cv` flag in harness).

4. **K=21 + d6_rule_compound_stint min-meta.** Δ −0.020 bp NULL.
   30-cell historical-pit-rate signal (Phase B EDA find: SOFT-S1
   4.25× lift) **already absorbed by pool**.

5. **K=21 + 3 (`d12_lr_meta` + `d10d_leak_corrected_meta` +
   `blend_rank_mean_K21`) min-meta.** **Δ +1.298 bp OOF** vs K=21
   baseline (0.95073 → 0.95086). d12_lr_meta dominates (|w|=3.84).
   Single-candidate ablation in flight at session-end. **First
   non-NULL probe of the session.**

**Open candidates (NOT YET RUN, BOTE-graded):**
- α-calibrated τ-resweep on PRIMARY hier-meta (PURSUE; ~30 min).
- Within-Race quantile-rank of LapTime_Delta as FM input (DEFER;
  attacks 922bp single-feature leak; ~30 min; cv_normalize fix
  required per friction tag `H5 z-score leak`).
- Per-Driver historical pit rate (smoothed EB) (DEFER; ~10 min).
- Year×Stint as sparse-LR feature / dedicated FM partition
  (DEFER; round-2 critic surfaced this; ~30 min).
- TE fold-leak audit on d2a/d3a (DEFER; read-only, 8 min).

**State unchanged:** PRIMARY = `d13e_compound_stint_tau20000`
LB 0.95049. No submissions made this session.

## Day-13 PM addendum (branch `claude/review-ml-handover-VTvWw`)

4 probes on FM-partition + Move-C axis; reinforces main's "same-field
partition mined" + adds a **load-bearing falsification**:

- **d13a 5/3 (D,C,S,T,Cd / R,Y,Rp)**: TIE Strat; GKF Δ −41.6 / −2.9bp — both leakage-robust ✓
- **d13b GKF FULL_22 matrix**: **d9c_FM REDUNDANT given d9f+d13a** (FULL_22 0.94607, SWAP_21 0.94606 = −0.01bp)
- **d13c Strat Move C**: T1 drop-d9c K=23 = T0 K=24 (no regress); **T2/T3 drop-GBDT FALSIFIED (−2.5 / −2.6bp)**
- **d13d V2 4/4 CT-axis + V3 6/6 alt**: TIE on Strat; partition saturation across **6 shapes** total
- **d13a S3 K=24 SUBMITTED**: LB **0.95032** TIE (ρ 0.99976; FM-amplification didn't fire)

**Move C revised thesis**: drop d9c (free), do NOT drop GBDT
leak-eaters (they carry row-iid public-LB signal — public LB ≠ GKF).

**Calibration update**: FM-class amplification (d9h 300×, d9i flip)
required NEW INPUT signal (Cd/Ld/Nx/Pv augmentation), not partition
shape alone. pre_submit_diff ρ>0.999 → TIE warning held for S3.

Audits: `audit/2026-05-13-d13{a,b,c,d}-*.md`.

## Day-13 close: Path B hier-meta is a NEW MODEL CLASS

`d13_path_b_stint_tau100000` is the load-bearing finding. Per-segment
empirical-Bayes hierarchical LR meta over the K=21 PRIMARY pool. Stint
segmentation (5 levels), shrinkage τ=100000.

| | Pre-submit | Result |
|---|---|---|
| OOF Δ vs d9f | +0.86bp | (built-in) |
| ρ vs d9f | 0.998 (sub-tie) | (built-in) |
| G3 flip ratio | 0.211 (FAIL) | benign — aligned with public LB |
| **LB** | predicted ~−1bp | **+10bp vs d9f, +7bp vs d9h/d9i** |
| **Upside** | n/a | **11.6× OOF→LB** |

**d13d GKF probe confirms mechanism is leakage-robust:** Strat lift
+0.90bp → GKF lift +2.59bp = **2.9× AMPLIFIED** (stronger than the
FM-class 2.3× amplification in d10b/c). Three independent leak-blocking
probes converge: public-LB +7bp transfers to private at median +4-6bp,
conservative +2-3bp, bull +6-8bp.

**Insight**: hierarchical-meta heuristics are NOT the same as base-class
heuristics. G3/ρ/R7 thresholds derived from d9c/d9f/d9h all failed for
d13 Stint. New mechanism family → recompute the GKF probe BEFORE
applying prior gates. (See `audit/friction.md` tag
`pred-lb-heuristics-broken-for-hier-meta`.)

## Day-14 first-action plan (40bp gap, structural moves only)

PI directive Day-13 evening: **"we want to improve by 40bp not 2."**
+1-3bp tuning probes are off the table for the comp middle. Each
candidate below is a structural model-class lift.

### Move A — push TabPFN-2.5 fine-tune kernel (PI provides token)
**BLOCKED on TABPFN_TOKEN.** Kernel at `kernels/d12-tabpfn-finetune-gpu/`.
PI: license-accept at https://ux.priorlabs.ai → set Kaggle Secret
`TABPFN_TOKEN` → `kaggle kernels push`. T4×2 5-7h. **Only live ≥10bp
single-shot candidate.** EV +5-15bp standalone, +1-5bp stack tail.

### Move B — pseudo-label cascade at K=21+hier-meta level (~3-4h CPU)
d5 partial pseudo got +14bp standalone but only +2.5bp at K=14 stack —
tighter cascade at K=21+hier-meta ceiling could compound:
1. Pseudo-labels from d13 Stint PRIMARY (LB 0.95041)
2. Confidence-filter (top 30% by spread); retrain 5 fastest bases on (train + pseudo)
3. Re-stack K=21 with pseudo bases + hier-meta; Round 2 if ρ < 0.999

EV +5-10bp. Risk: pseudo over-amp (d5 widened gap on m5q).

### Move C — DeepFM-lite (~3-4h CPU)
FM pairwise + 2-layer MLP head on 21-base expanded space. New class
beyond d9c/d9f/d9h. EV +3-8bp standalone, +1-3bp stacked. Risk:
overfit without dropout (d9e FFM precedent — 4× params died).

### Move D — new FM-input feature engineering (parallel agent's Day-13 finding)
V1/V2/V3 sweep proved **same-12-field FM-partition reshuffle is dead**.
New FM lift needs NEW INPUTS: pit-window-since-last-pit, hazard-decay,
compound-pressure, race-stage. Build 4-6 features, train unified FM
on aug12 + new fields, gate vs PRIMARY. EV +2-6bp.

### Move E — Compound × Stint hier-meta SUBMIT (held; only if Move A-D miss)
`d13e_compound_stint_tau{20000,100000}.csv` (+0.82-1.00bp OOF, ρ=0.996+).

### Move E — Path B hier-meta on Move-C-refactored K=23 pool (cheap)
NEW Day-13 PM hypothesis. Path B Stint built on K=21 PRIMARY pool.
Re-run on T1 K=23 pool (drop d9c, keep d9f + d13a partition FMs):
EV +0-2bp; tests whether per-segment hier-meta lift compounds with
Move C minimal refactor. Cost ~30s (LR-meta only). Fastest of all Day-14
moves. Held: `submissions/submission_d13c_T1_drop_d9c.csv` is the K=23
input pool; combine with `scripts/d13_path_b_stint_*.py` driver.

### Research-loop trigger (Rule 7) IF Day-14 yields no ≥+5bp move
Pause submits. Web-search top-5 finishers' writeups from comparable
playground tabular comps. Identify untried mechanism families. Honest
read: **40bp gap may not be tractable from current architecture**.
Top finishers likely have a structural insight (target reformulation /
unique FE / external data) we haven't found.

## Falsified / dead — do NOT retry

- Big-sequence (P1); kNN/TabR/Hopular/TabPFN-ICL (P2)
- RealMLP bagging; broad pseudo-labeling; TabM-D extended 200ep
- Hazard NN (230bp leak; leakfree zero signal)
- d9 10 math heuristic rule_residuals; d9b R14 K=20 swap+L4 (TIE)
- d9d FM sweep+bag; d9e FFM (overfit); d9g 3-way; d9h K=22 add (TIE)
- T1.2 multi-formulation 4-of-4 (Poisson/censored/ratio/survival)
- Year-specialist; AV reweighting (i.i.d.); LambdaRank meta (-86bp)
- AUC-pairwise XGB (-451bp); external real-world priors on synth
- **d10d leak-corrected meta** (G3 flip ratio 0.001; over-credits FM)
- **Day-13 Move B V1/V2/V3 same-field FM partitions** — V1 5/3 SUBMITTED
  LB 0.95032 TIE; V2/V3 held dead. Same-12-field FM-partition vein FULLY
  MINED. Future FM lift needs NEW INPUT FIELDS, not new partitions.
- **Day-13 PM d13a S3 K=24 SUBMITTED** LB 0.95032 TIE (5 FMs in pool;
  ρ 0.99976; FM-amplification didn't fire on same-field reshuffle).
- **Day-13 PM d13d V2 4/4 CT-axis + V3 6/6 alt-split**: 6 partition
  shapes total now confirmed saturated.
- **Day-13 PM Move C drop-GBDT** (e5_optuna_lgbm + cb_slow-wide-bag):
  −2.5 to −2.6bp Strat. GBDT leakage-eaters are LB-load-bearing.
- **Day-13/14 alternative-axis branch (4-of-4 nulls):** G1 within-stint
  LGBM FE (-0.38bp), G2' cross-driver LGBM FE (+0.03), G3 stint-grouped
  LambdaMART (smoke 0.74), H1 FM aug13 CTRq 3-way (-0.13). All low ρ
  (0.92-0.97) but min-meta zero/neg. **Lesson: single-base FE additions
  hit a noise wall regardless of model class** — Path B hier-meta has
  absorbed signal from existing-class new features. **EDA H1/H6 family
  (3-way concat / Year×Stint partition FM) DEMOTED to ~0bp EV.**

## Held submissions (do NOT submit)

- BURNED: `d5_partial_pseudo_m5q.csv` (LB 0.94963), `d9_k19_hazard_nn_stack` (LB 0.94711)
- TIE/DEAD held (full list in CLAUDE.md): d7-d10 variants, d12 t12/year/AV/lambda
  bases, **d12_groupkf_meta** (HEDGE-eligible R5), **d10d** (G3 fail),
  **d13b Stint τ=20000**, **d13c Compound τ=100000** (LB 0.95033 result),
  **d13e Compound×Stint** 4 τ-variants (only submit if no Move A/B/C win)

## Critical operating rules

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **NEW Day-13: ρ/G3/R7 heuristics DO NOT apply to new mechanism
   families.** Confirm via GKF probe before assuming a sub-tie ρ
   candidate will under-perform.
3. **Rule 16 Q6** (ρ vs FULL pool, not just PRIMARY): binding gate.
4. **GroupKF as secondary gate** (Day-12 finding). For new bases:
   pass Strat AND not regress GKF, OR pass GKF if leakage-robust class.
5. **Strat-only Day-3+ (R1)** for primary OOF. Public LB stays the
   truth (U3: i.i.d. row split). GKF is the private-LB proxy probe.
6. **Submit budget** 23/270; 13 days × 9 = 115 remaining. **40bp gap
   means structural moves only**; tuning probes deferred to R5.
7. **Model-class diversification > tuning** (d9c FM, d13 hier-meta both
   confirmed). New base class > new τ on existing meta.

## Pointers

- `audit/2026-05-13-d13-{path-b-hier-meta,d13d-path-b-gkf-probe}.md` — load-bearing
- `audit/2026-05-13-d13-{problem-decomposition,g-results,eda-deep-dive-synthesis}.md` — Day-13/14 alt-axis (4 nulls; H1 family demoted)
- `audit/2026-05-12-d12-{master-synthesis,tabpfn-finetune-prep}.md`; `audit/friction.md`
- `scripts/d13{,b,c,d,e}_path_b_*.py` (Path B); `scripts/d13_g{1,2,3,5}_*.py` + `d14_h1_*.py` (null branch); `kernels/d12-tabpfn-finetune-gpu/`
