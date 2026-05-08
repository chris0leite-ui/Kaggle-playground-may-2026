# WEAKNESSES.md — known unsolved gaps and weak spots

A parking lot for problems we're aware of but couldn't solve in this
competition's compute budget. **Not a list of falsified ideas** — for
those, see `state/hypothesis-board.md ## Killed`. **Not a list of
open priorities** — for those, see `state/hypothesis-board.md ##
Open priorities` and `ASSUMPTIONS.md`.

This file is for things where:
- We measured a real signal but couldn't recover it.
- We have a sound theoretical reason to believe a class of
  improvement exists, but no compute-affordable path.
- Our diagnostic tools have known blind spots.
- A known load-bearing inference rests on shaky ground but no cheaper
  test exists.

Each entry: **what's weak**, **why we couldn't fix it**, and **what
might break it open later**. Cite artifact paths.

---

## Process rules

1. Every entry must be tagged with a status: `unsolved`, `partial`, or
   `cold` (lost relevance).
2. When a future session attacks an entry, that session updates the
   status here AND copies the entry to `state/mechanism-ledger.md`
   with the verdict.
3. New entries go in chronological order. Don't rewrite history.
4. Re-check at every postmortem; carry forward to the next comp's
   `improvements.md` if still relevant.

---

## W1 — Rain-condition residual is intrinsic given current pool (status: unsolved)

PRIMARY's worst-cell AUC is 0.68–0.86 on (Compound × Stint × position)
slices that are all INTERMEDIATE / WET. Aggregated rain residual is
~13 bp vs PRIMARY global, of which the WET subgroup (n=1,355, AUC 0.845)
contributes most. A single-LGBM rain specialist gets −152 bp on the
segment because it loses cross-Compound transfer. A richer specialist
(full pool retrained on rain only) is theoretically possible but cost-
prohibitive in remaining time.

What might break it open: (a) a global model with rain-row sample
weighting (pursued but bounded — see `decisions.jsonl`); (b) external
weather telemetry (rain probability per (Race, Lap) from FastF1) at
soft-feature resolution, since the 1.4% hard-join cap was for row-level
matching only.

Artifact: `scripts/probe_rain_specialist.py`,
`scripts/artifacts/probe_rain_specialist.json`.

## W2 — OOF→LB sampling-noise CI is wide (status: cold but worth re-checking)

Bootstrapped 95% CI of a random 20% public draw is [0.95309, 0.95550]
around an OOF of 0.95432 — a span of **24 bp**. This means a 4-bp OOF
gain has a substantial chance of regressing on public LB just from the
public-split lottery, and conversely a 4-bp OOF regression can land as
a paper "lift" of a few bp. The team has been tracking OOF→LB deltas
of order ±5 bp as if they were signal; the CI says they're not.

What might break it open: bootstrap a bigger sample by repeated
seed-restarts of OOF generation, or use the published-LB pairs from
identical-architecture submissions to bound the within-architecture
public-split variance. We can also flag this in any submission decision
as "expected uncertainty band ≈ ±12 bp."

Artifact: `scripts/probe_understand_problem.py` Probe B,
`scripts/artifacts/probe_understand_problem.json`.

## W3 — Synthetic data temporally downsamples real stints (status: unsolved)

Probe C: synthetic stints have mean length 3.87 vs original 19.80;
synthetic consecutive-rows-at-gap=1 fraction is 27.98% vs 99.60% in
original. The synthesiser preserves within-stint physical constraints
(Compound, TyreLife monotonicity, LapNumber strictly increasing) at
≥99.99%, but takes a sparse subset of the underlying lap-by-lap
trajectory. **This means our model is trained on lap-level snapshots,
not on dense lap sequences.** Sequence models that need contiguity
(GRU on (Driver, Race, Stint) windows) operate on data that mostly
isn't contiguous.

What might break it open: model the underlying contiguous trajectory
explicitly using FastF1 as a conditioning signal at the (Race, Year,
Driver) level — then evaluate whether re-densified sequence features
add anything at the meta. Cost: 1-2 days. Not pursued.

Artifact: `scripts/probe_understand_problem.py` Probe C.

## W4 — Adversarial validation tested at row level only (status: partial)

AV-AUC = 0.502 measured on 14-column row features (per `comp-context.md`
U3 probe, 2026-05-04). This drives Rule 25 ("transductive features
need adversarial-validation check") — but Rule 25's diagnostic is
linear-AV at row level. Probe C shows synth and orig differ at sequence
level. We have no AV check at sequence aggregates ((Race, Driver,
Stint) features like stint length, gap-1 fraction).

What might break it open: extend Rule 25 to include AV at sequence
aggregates whenever a transductive transform is at sequence resolution.
Add a cheap diagnostic to `scripts/probe.py`. Cost: 1 hour to
implement. Not done this session.

Artifact: `scripts/probe_understand_problem.py` Probe C metrics.

## W5 — K=22 LR-meta rank-lock is INFERRED, not falsified at every variant (status: partial)

A9 in `ASSUMPTIONS.md`. Five cross-confirmations that base-adds get
absorbed: GRU sequence (d16 H1), field-state aggregates, transductive
pseudo-labels, twin-pool 2-meta, mode-id DGP base. The inference is
that *any* standalone OOF-computable base is absorbed by the K=22
[raw, rank, logit] LR-meta with [F_oof, expand] features.

**Day-19 update:** A9b adds two more variants at K=27 — Compound × Stint ×
Year (-2.25 to -0.16 bp) and Compound × RaceProgress-bin (-1.71 to
-0.16 bp). Per-segment-stacker family is now 11+ variants. Importantly,
at τ=100k the PRIMARY's hier-meta lifts only +0.03 bp over a plain global
K=27 LR-meta — the "Path-B amp" is essentially gone at this pool size.

This still doesn't rule out:
- Bases trained on a **different objective** (e.g., focal loss, ranking
  loss) that route through a non-AUC direction at the meta.
- Bases whose prediction is **conditional** on an external state we
  don't have (FastF1 weather), where the conditioning isn't in the
  meta features.
- **Non-LR per-segment heads** (e.g., per-segment XGBoost or LightGBM
  head replacing the per-segment LR). The current Path-B is LR-on-LR;
  a tree-based per-segment head could exploit non-linear interactions
  that the LR meta cannot.
- **Nested hierarchy**: per-Compound, then per-Stint within Compound,
  with two levels of shrinkage. The current Path-B is flat.

What might break it open: (i) a deliberately mis-specified base
(LightGBM trained on a different loss) — cost 1 hour. (ii) a per-segment
LightGBM head for the meta — cost ~2 hours. Neither pursued.

## W6 — The leader (0.95476) gap is unexplained (status: unsolved)

We are at 0.95368 LB, leader at 0.95476 — gap 10.8 bp. Hypothesis A18:
"FastF1 hard-join is the only path." But the FastF1 hard-join is
capped at 1.4% match rate by synthetic driver codes — that's
INSUFFICIENT alone to deliver +10 bp. So either the leader has a
different mechanism, or the 1.4% cap is wrong, or multiple smaller
mechanisms compose to 10 bp.

What might break it open: scrape the public discussion forum for
mechanism hints at LB ~0.954+; look at top public scores' submissions
trajectories; ping the leader's profile for prior-comp patterns. None
attempted.

## W7 — Public-notebook scan can miss top-5%-tier kernels (status: cold)

Probe D this session scanned 8 public kernels; top reported LB was
0.95388 (`ps-s6e5-hb1` v8). Top-5%-tier kernels (≥ 0.95405) may exist
unpublished, or be published after our scan. Rule 22 calls for a
plateau scan but doesn't trigger continuously.

What might break it open: schedule a Rule-22 scan at every plateau.
Cheap and cumulative.

## W8 — `ASSUMPTIONS.md` low-confidence rows have no cheap re-check scheduled

A11 (RealMLP n_ens=24 lift band), A12 (per-Year CatBoost lift band),
A13b (FastF1 soft features closure), A16 (private LB structural risk),
A18 (leader gap mechanism). Each of these is load-bearing for at
least one strategic decision but has no anchor in the calibration
ladder.

What might break it open: write `scripts/probe_assumption_check.py`
that walks `ASSUMPTIONS.md` and flags rows whose `last_checked` is
> 7 days old or whose `Status` is `live (low confidence)`. Schedule
weekly.
