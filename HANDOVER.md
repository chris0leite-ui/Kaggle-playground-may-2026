# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file with the next
> session's brief.

---

## Today's session — Day 3 mid-session (2026-05-04 ~17:50 UTC)

**Read order on session start** (skip the default; this file is the
synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R13, R14)
2. `comp-context.md` — settled-once facts (compute, schema, GPU workflow)
3. `audit/2026-05-04-strategy-critique.md` — what we still DON'T know
4. `audit/2026-05-04-m5h-l1coef-prune.md` — **validated meta-level lever**

Open with a 3-bullet read-back of state + the first mechanism to run.

## Where we are

- **Day 3 in progress**, **6/10 submissions used today, 4 remaining**.
- **`our_lb_best = 0.94991`** (M5h L1coef-pruned 13-base stack, gap −5.2bp).
  Headroom to top-5% (0.95345): **35.4bp**.
- **Single-model ceiling ~0.94876 OOF** (E3 HGBC); pure-stack ceiling on
  current pool is now demonstrably **bounded by orthogonality, not
  pool size or meta-tuning**: M5h has 13 bases (more than M5d's 12) but
  tighter gap because L1-prune dropped two bases with near-zero meta
  weight (m3_catboost, m4_relstate).

## What today landed

- **CatBoost variants explored on a dedicated branch** (Stage A research
  + 7 1-fold probes + 3 anchor runs):
  - **lossguide** (`grow_policy=Lossguide`, depth=8, Year∈CAT_COLS):
    Strat 0.94697 / GroupKF **0.92377** — first CB to clear G1 GroupKF.
  - **slow-wide-bag** (3-seed GPU bag, lr=0.03, iter=4000, l2=8):
    Strat **0.94790** / GroupKF 0.92322 — best CB on Strat.
  - **year-cat** (Year added to CAT_COLS, M3 params): Strat 0.94679 /
    GroupKF 0.91992. Year-as-numeric was driving M3's overfit.
- **Kaggle GPU pipeline working** (after fixing 2 root causes):
  - `kaggle kernels init` template emits string-quoted booleans which
    Kaggle silently treats as `false`. Always edit to bare `true`/`false`.
  - Comp data path varies; use `Path('/kaggle/input').rglob('train.csv')`
    rather than hardcoding `/kaggle/input/<slug>/`.
  - Working reference: `kernels/cb-slow-wide-gpu/`. Pull a known-working
    prior kernel (`kaggle kernels pull <user>/<slug> -m`) for any new GPU
    work.
- **L1-coef prune validated as a meta-level lever**:
  - M5b (7 bases) gap −3.5bp; M5d (12) gap −6.0bp; **M5h (13, L1-pruned)
    gap −5.2bp**. Drops m3_catboost + m4_relstate (lowest L1 sum across
    raw/rank/logit channels). Reuse on every future stack.
- **OOF→LB transfer is sharply diminishing**: M5b 92% transfer
  → M5d 74% → **M5h 24%**. Each additional 20bp of OOF gives <5bp LB.
  Implication: more bases of the same flavor pay less and less.

## PI direction for remaining slots / Day-4

> *"We need to add more models or features with orthogonal signal
> before we dig deeper into ensemble specifics."*

The L1-prune is the last meta-level squeeze that still works. Further
ensemble tuning (Ridge meta, hill-climb, more pool members of the same
GBDT flavor) is **lower EV than adding genuinely orthogonal signal**.

## Day-3 sequence — orthogonal-signal first (DO IN ORDER)

**Step 1 — 2-way TE base** (~30-60 min CPU). Day-1 missed lever flagged
by the strategy critique and analyticaobscura Source 1 #2:

- Driver×Race / Driver×Compound / Race×Lap-bin with α=80 smoothing,
  inner 5-fold per outer fold (no leakage). Single-base probe first;
  if standalone ≥0.946 OOF, add to M5h pool → M5i refit. Apply L1-prune
  before any LB submit (drop bases below median L1 sum). **Submit slot 7.**

**Step 2 — Sequence-FE base** (~30 min CPU). 97.4% of test has
same-(Race, Driver) within-test continuation — this structure is
unexploited:

- `laps_since_last_pitstop`, `cumulative_pitstops_this_race`,
  `rolling_target_rate(window=5)` over (Race, Driver) groups.
- Single-LGBM probe on baseline + these 3 features. If OOF ≥0.945 → add
  as M5j base. **Submit slot 8.**

**Step 3 — RealMLP on Kaggle GPU** (~2-3h roundtrip including push).
yekenot's 56-vote public notebook for *this exact comp* uses RealMLP;
truly orthogonal mechanism family (NN vs GBDTs):

- Use `kernels/cb-slow-wide-gpu/` as the metadata template (booleans
  fixed, rglob in place).
- E4 was killed at fold-0 local CPU only (3.3h projection); GPU should
  fit comfortably under 1h.
- Single-model probe → add to stack as M5k base. **Submit slot 9.**

**Step 4 — HOLD** slot 10 for whichever of the above gave the best
calibration data, or for an R2-hedge candidate.

## Lower priority (only if Step 1-3 underdeliver)

- HGBC multi-seed bag (H4) — *variance* reduction; pool variance is
  not the binding issue (we have 99% correlation between HGBC variants).
- Ridge / hill-climb meta drop-in (H5) — meta-level lever, but
  L1-prune already squeezed most of what's available.
- M5f raw vs M5h calibration probe — informative but not LB-improving.
- M5e CB-only probe — same.

## Anti-patterns (validated)

- **Don't expand pool with correlated bases** — M5d→M5f gap-widening
  pattern. L1-prune is the patch; orthogonal-signal-add is the cure.
- **Don't pseudo-label from the over-fit stacker** — use multi-base
  agreement guard (skill rule).
- **Don't burn slots on calibration probes when LB calibration is
  already established** — Day-3 has 4 slots; spend on net-new lift,
  not A/B confirmation of known mechanisms.
- **Don't trust `kaggle kernels init` output blindly** — fix the
  string-bool defaults before pushing.

## Workflow rules in force

- **R1** Submissions are single-shot + PI-approved.
- **R12** Spend the full 10/day budget — calibration data is
  load-bearing.
- **R13** Kaggle GPU IS available — port any 5-fold > 1h local-CPU.
- **R14** Strategy-critic-loop fires automatically before adding a
  new mechanism family. Steps 1-3 above each trigger it.

## Open questions for PI

- 2-way TE smoothing α=80 OK, or sweep α?
- RealMLP architecture: copy yekenot's exact config or smaller?
- After steps 1-3 land, do we revisit Ridge meta / hill-climb (Day-4)?
