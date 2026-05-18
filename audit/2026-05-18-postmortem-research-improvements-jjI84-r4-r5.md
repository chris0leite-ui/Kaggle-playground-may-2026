# Postmortem — 2026-05-18 research-improvements-jjI84 (Rounds 4-5)

Session covers Round 4 (Rule 23 free-form FE + HMM super-model)
and Round 5 (slim-kNN rebuild + multi-class super-stack + Path-B
operator breakthrough). Continues from earlier postmortem
`audit/2026-05-18-postmortem-research-improvements-jjI84.md`
which covered Rounds 1-3.

**Headline outcome**: new PRIMARY LB **0.95387** (R5.2 K=13+Path-B),
+0.01 bp over prior PRIMARY (within tie band, technically above).
First LB lift in 5+ days; 4 submissions used (R4 0.95354, R5.1
0.95382, R5.2 0.95387, R5.3 0.95385). Top-5% gap 1.8 bp; leader 8.9 bp.

## What went wrong

Three decisions where the same priors today would point to a
different choice:

1. **5-seed Path-B bag launched without inspecting `run_pathb`
   test-prediction code path.** `scripts/build_K11_full_pathb.py:116-128`
   does a full-train fit for test predictions (not fold-fit
   averaging). LR's convex fit is seed-invariant given the same
   full training set — multi-seed test predictions are identical
   (ρ=1.0 confirmed empirically). **Cost: ~17 min CPU wasted on
   5 sequential Path-B runs.** Pre-flight check would have caught
   this in <1 min. *Rule-gap.*

2. **Phase F transformer kernel needed 3 push iterations.** Failures:
   (a) wrong data path — used `os.path.exists()` check on hard-coded
   `/kaggle/input/playground-series-s6e5` instead of the rglob
   pattern that's in `kernels/p1-xgb-v4-gpu/p1_xgb_v4_gpu.py`;
   (b) P100 sm_60 vs torch 2.10 incompat — fix already documented
   in `kernels/d15b-dae-gpu/d15b_dae_gpu.py` (torch 2.4 force-
   reinstall); (c) `enable_internet: false` blocked pip install of
   torch 2.4. **Cost: ~30 min operational overhead.** Would have
   been a single push had I grep'd existing `kernels/*/*.py` for
   precedent before writing. *Rule-gap.*

3. **Phase C smoke shipped with non-vectorized aggregator.** Initial
   `compute_pit_pressure_features` used a Python row-loop over 50k
   rows with dict lookups. Wall time 8+ min on CPU-shared box
   (vs 0.04 s when re-implemented with pandas merge). **Cost:
   ~10 min wasted, plus killing the smoke and rerunning.** Decent
   trade-off (write fast, profile, rewrite) — but the row-loop was
   obviously slow at design time.

Three things that went well (decision-quality-positive):

- **Anchor-progression check (K=4 → K=5 → K=11) was disciplined**
  and produced the predictive +0.245 bp value at K=11 before any
  LB submission, avoiding wasted slot probes.
- **K=15+Path-B build → ρ-check → no submit**: built variant,
  confirmed ρ=0.999937 vs R5.2 (TIE_ZONE), held the slot. Rule 27
  applied correctly.
- **PI consultation at decision-relevant moments only**: 2 questions
  asked (super-model choice + scope), aligned with PI's "iterate
  autonomously" earlier directive while preserving load-bearing
  strategic input.

## Frictions logged this session

Appended to `audit/friction.md ## This week (2026-05-12 → 2026-05-18)`:

- `operator-vs-mechanism-axis` — operator class (Path-B vs LR-meta)
  is an axis distinct from mechanism class; +5 bp LB transfer
  swing at same OOF.
- `pathb-bag-seed-invariant-test` — `run_pathb` test path is
  seed-invariant; 5-seed bag wasted 17 min CPU.
- `gpu-kernel-prior-pattern-skip` — Phase F took 3 push iterations
  because I didn't grep prior kernels/ for the rglob + torch 2.4
  patterns first.
- `snapshot-missing-orig-dataset` — slim-kNN builders need
  `data/original/f1_strategy_dataset_v4.csv` which wasn't in the
  snapshot; pulled from Kaggle dataset.
- `cpu-contention-phase-c-starved` — Phase A's qAO/qAF starved
  Phase C smoke despite Rule 30 "≤2 concurrent CPU-heavy" being
  technically met.

Pre-existing 2026-05-18 entries (from rounds 1-3 session, retained
for context):

- `research-loop-overdue`
- `kaggle-pages-recaptcha-gated`
- `tier-a3-menu-stale`

## Promotion candidates (PI ratified)

PI decisions:
- ✅ **Bag harness verification rule** → guardrails.md §13. PROMOTED.
- ✅ **Operator-class retest rule** → experiment-loop.md. PROMOTED.
- ⏭ **Prior-kernel-pattern grep rule** → agent-ops.md. SKIPPED.
- ✅ **Bootstrap.sh dataset validation** → bootstrap.sh. PROMOTED.
- ✅ **Vectorise-before-launch rule** → guardrails.md §14. PROMOTED.

All four ratified promotions applied; the promotion entries are
mirrored in `.claude/skills/kaggle-comp/improvements.md ## Applied
2026-05-18 (round-4+5 wrap; PI-ratified)`.

## Promotion candidates — drafts (now applied)

### [ ] `.claude/skills/kaggle-comp/guardrails.md` — Pre-bag verification

**Tag:** `pathb-bag-seed-invariant-test` (today; 17 min CPU)

**Where to insert:** new bullet under "Bagging / variance reduction"
section (or create the section).

**What to add:**

```markdown
- **Verify the bag harness's test-prediction code path is actually
  seed-variant before launching multi-seed loops.** For Path-B-class
  operators (`run_pathb` in `build_K11_full_pathb.py`), the test
  path fits on FULL training data (convex, seed-invariant) while
  the OOF path fits per-fold (seed-variant). A multi-seed bag of
  this harness produces identical test predictions (ρ=1.0). For
  true bagging, modify the harness to AVERAGE per-fold-fit test
  predictions across seeds, or vary the base OOFs / sub-sample
  training rows. Cost evidence: 2026-05-18 R5 5-seed bag wasted
  ~17 min CPU.
```

**Why:** General to any meta-operator with full-train fit step;
applies beyond Path-B. Promotion-criterion: ≥1h compute waste
(boundary case, ~17 min, but recurring pattern is plausible).

### [ ] `.claude/skills/kaggle-comp/experiment-loop.md` — Operator-class retest

**Tag:** `operator-vs-mechanism-axis` (today; +5 bp LB transfer swing)

**Where to insert:** Step 0 (ledger-grep gate) or new Step 6
"Operator-class verification" appended to the experiment loop.

**What to add:**

```markdown
- **Operator class is an axis distinct from mechanism class.** A
  mechanism that nulls under one meta operator may pass under
  another with the SAME OOF. 2026-05-18 R5: K=11 + seg_fe + HMM
  pool produced OOF 0.95446 under both LR-meta and Path-B operators,
  but LB differed by +5 bp (Path-B better OOF→LB transfer at this
  pool size). Rule: when retesting a null mechanism, sweep
  operators {LR-meta C ∈ [0.01, 100], Path-B C × Stint at one or
  more τ values} before declaring the mechanism dead. Closed
  mechanism-class results are operator-conditional unless otherwise
  verified.
```

**Why:** Rule-gap with concrete +5 bp LB cost evidence. Prevents
premature mechanism closure under a single operator.

### [ ] `.claude/skills/kaggle-comp/agent-ops.md` — Prior-kernel pattern grep

**Tag:** `gpu-kernel-prior-pattern-skip` (today; ~30 min ops cost)

**Where to insert:** under "Kaggle GPU kernel ops" section.

**What to add:**

```markdown
- **Before writing a new Kaggle GPU kernel, grep existing `kernels/*/*.py`
  for the data-path resolution + torch-version-pinning patterns.**
  Recurring fixes: (1) `find_data_dir(name)` via `Path('/kaggle/input').rglob(name)`
  — Kaggle competition mount path is not the same as the slug; (2)
  `pip install --force-reinstall torch==2.4.*` to keep P100 (sm_60)
  working since Kaggle's default torch ≥2.6 drops sm_60; (3)
  `enable_internet: true` in `kernel-metadata.json` for pip to reach
  PyPI. Recurring failure-mode: each pattern caused a kernel push
  iteration. Cost evidence: 2026-05-18 R5 Phase F took 3 pushes
  (~30 min ops). `kernels/p1-xgb-v4-gpu/` has the rglob pattern;
  `kernels/d15b-dae-gpu/` has the torch reinstall.
```

**Why:** Three documented fixes in prior kernels weren't applied
to a new kernel. ≥30 min ops cost. Recurring pattern (P100/torch
issue is the same one as `recurring  kaggle-p100-torch-sm60-incompat`
in friction.md process section).

### [ ] `bootstrap.sh` — Original-dataset validation

**Tag:** `snapshot-missing-orig-dataset` (today; ~5 min recovery cost)

**Where to insert:** After the Kaggle credentials block + competition
data download, add a check for `data/original/f1_strategy_dataset_v4.csv`
specifically (s6e5).

**What to add:**

```bash
# scripts/dgp_v3/*.py and 7 other paths reference the original
# F1 strategy dataset CSV under data/original/. Pull it if absent
# (snapshot 2026-05-08 didn't include it).
ORIG_CSV="data/original/f1_strategy_dataset_v4.csv"
if [[ ! -f "$ORIG_CSV" ]]; then
    echo "--- data: $ORIG_CSV missing; pulling from aadigupta1601/f1-strategy-dataset-pit-stop-prediction ---"
    mkdir -p data/original
    kaggle datasets download -d aadigupta1601/f1-strategy-dataset-pit-stop-prediction \
        -p data/original/ --unzip || echo "WARN: pull failed; some builders may break"
fi
```

**Why:** Slim-kNN rebuild is the single most-valuable next-session
action; missing dataset would block it again. Cost evidence:
2026-05-18 R5 Phase A failed for 5/6 builders before the pull.
Low cost, high reliability.

### [ ] `.claude/skills/kaggle-comp/guardrails.md` — Vectorise-before-launch

**Tag:** `cpu-contention-phase-c-starved` + Phase C smoke slowness

**Where to insert:** under "Performance and resource hygiene".

**What to add:**

```markdown
- **Probe scripts with O(N) Python loops over rows MUST be replaced
  with vectorized pandas / numpy before launching against the full
  dataset.** Smoke a 50k-row probe with the row-loop version first;
  if smoke wall time × (full N / smoke N) > 5 min on a contended
  CPU, vectorize before full. 2026-05-18 R5 Phase C smoke took
  8+ min CPU on row-loop aggregator before profiling; vectorized
  pandas-merge variant ran in 0.04 s on the same data.
```

**Why:** Recurring temptation to "write it quickly, profile later"
on probe scripts. Cost is small per-instance but adds up. Rule-gap
not currently covered by experiment-loop.

## PI additions

PI: "Nothing to add" (step 4 verbatim). No new flags, no additional
frictions to capture, no rule extractions beyond what was drafted.

## Framework version at session-end

- **Commit SHA**: `97a9c6afca84a0e23e3c0aedd7b68bc385dc6c1a`
- **Branch**: `claude/research-improvements-jjI84`
- **Active rules**: 1-36 (CLAUDE.md `## Rules`)
- **Loaded skills this session**: `kaggle-comp` (`SKILL.md`,
  `guardrails.md`, `loops.md`, `experiment-loop.md`,
  `personas.md`, `problem-solving.md`, `agent-ops.md`,
  `strategy-critic.md`, `improvements.md`), `postmortem`
  (this skill).
- **PRIMARY at session-end**: R5.2 K=13 + Path-B τ=100k, LB 0.95387.
- **Submissions used**: 46 / 270 total; 4 today; 6 daily remaining.
