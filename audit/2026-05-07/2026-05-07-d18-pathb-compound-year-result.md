# d18 Path-B Compound × Year on K=24 — RESULT: NULL across all τ

Branch `claude/logistic-regression-ensemble-0PNkA`. PI authorized
execution of the probe spec'd at
`audit/2026-05-07-pathb-compound-year-probe-plan.md`. **Result: clean NULL.**

## TL;DR

| τ | Strat OOF | Δ vs PRIMARY OOF | ρ vs PRIMARY | flip ratio (PRIMARY) |
|---|---:|---:|---:|---:|
| 5000 | 0.95364 | **−2.05 bp** | 0.9966 | 0.692 |
| 20000 | 0.95375 | **−0.97 bp** | 0.9978 | 0.714 |
| 100000 | 0.95383 | **−0.16 bp** | 0.9994 | 0.491 |
| 500000 | 0.95385 | **−0.01 bp** | 0.9999 | 0.182 |
| Global K=24 LR-meta (baseline) | 0.95385 | (baseline) | — | — |

**Monotonic decay**: as τ → ∞, Path-B converges to global (which equals
the K=24 LR-meta baseline 0.95385). All τ values are at-or-below
baseline. **No amp; no submit.**

Total wall: 367s (≈6 min on 4-core CPU).

## Comparison to predicted scenarios (probe spec)

| Scenario | Predicted P | Predicted LB Δ | Actual |
|---|---:|---:|---|
| Best | ~15% | +10 to +20 bp | NO |
| Mid | ~50% | +2 to +5 bp | NO |
| Bad | ~35% | NULL or −2 bp regression | **YES** ★ |

The bad-case scenario was correctly weighted at ~35% but came true.
Path-B amp does NOT fire on Compound × Year for the K=24 pool.

## Why it failed: mechanism finding

The probe-5 LR-class lift (+60.8 bp standalone per-segment vs global
mega) did NOT transfer to meta-class on K=24. Reading:

**The K=24 pool already includes `cb_year-cat`** — a CatBoost base
trained with `Year` as a native categorical feature. Year-conditional
routing is already absorbed at the BASE level in the K=24 pool's OOFs.
Path-B's per-segment LR-meta would just re-learn what the pool already
routes through cb_year-cat.

This is a NEW mechanism finding (friction tag candidate):

**`pathb-amp-dead-when-pool-already-routes-segmentation-variable`** —
Path-B segmentation cross variable X fires amp only when the pool's
bases do NOT natively route by X. K=24 has cb_year-cat (Year), so
Path-B Year is dead. K=24 has no Stint-native base, so Path-B Stint
fires amp (current PRIMARY).

This is a **stronger statement** than the existing
`path-b-amp-only-fires-on-meta-arch-not-base-add` friction. It explains
*which* meta-arch redesigns fire — those that introduce segmentation
NOT already captured at the base level.

## Reconciling with probe 5 (per-segment mega LR +60 bp)

Probe 5: per-(Compound × Year) LR mega gave +60.8 bp over global mega.
That LR pool was just *one base* (mega LR), which has no native
Year-routing. So per-segment mega DID find new structure.

Probe d18: per-(Compound × Year) LR-meta on K=24 GBDT pool gave −0 to
−2 bp. The K=24 pool ALREADY routes Year via cb_year-cat. Per-segment
meta-LR is redundant.

**The reconciliation**: the same segmentation variable fires at
LR-class (no Year-aware base) and is dead at meta-class (Year-aware
base in pool). Both findings are correct; their reconciliation is the
mechanism finding above.

## Why d14 was NULL on K=21 too

d14 sweep ran Year-segmented Path-B on K=21 (no v4, no h1d, no d16).
K=21 includes `cb_year-cat`. Same mechanism: Year-routing already
absorbed at base level → Path-B Year dead.

Both d14 NULL and d18 NULL share the same explanation. The friction
isn't K=21 vs K=24 specific; it's "K-pool-with-cb_year-cat" specific.

## What this closes

- The Day-18+ HANDOVER Item 9 (this probe) → **NULL, no submission**.
- The friction `path-b-amp-only-fires-on-meta-arch-not-base-add` gets
  refined: the segmentation cross must introduce a routing axis the
  pool lacks. Year is already in the pool via cb_year-cat. Year-axis
  Path-B is permanently dead on this pool.

## What this opens (if anything)

If we want Path-B to fire on a year-conditional axis, we must either:
1. **Drop cb_year-cat from the pool** before running Path-B Year — kills
   the base AUC contribution, probably net negative.
2. **Use a year-derivative variable not captured by cb_year-cat** —
   e.g. `Year_minus_Driver_debut_year` or `seasons_since_compound_change`.
   These would need new FE.
3. **Skip Year axis entirely; try other untapped crosses** —
   `(Driver_clustered, Stint)`, `(Race_class, TyreLife_q5)`,
   `(Position_q5, Compound)`. These remain in HANDOVER A4's "Alternative
   seg crosses" list.

## Submissions impact

7/10 used today; 3 slots remain. **No submit from this probe.**
PRIMARY remains `d17_path_b_K23_v4_h1d_tau100000` LB 0.95354.

## Decisions log entry

`audit/decisions.jsonl`-style entry (not auto-logged since user said
"do it now" without sealed prediction):

```json
{
  "ts": "2026-05-07T20:?:?Z",
  "branch": "claude/logistic-regression-ensemble-0PNkA",
  "probe": "d18_path_b_compound_year_K24",
  "family": "meta_arch_redesign",
  "metric_aligned": true,
  "agent_predicted_lb_bp_midpoint": 3,
  "agent_predicted_lb_bp_range": [-1, 3, 10],
  "pi_predicted_lb_bp": null,
  "actual_oof_delta_vs_primary_bp": [-2.05, -0.97, -0.16, -0.01],
  "actual_lb_bp": null,
  "verdict": "NULL — all τ regress vs PRIMARY OOF; not submitted",
  "mechanism_finding": "pathb-amp-dead-when-pool-already-routes-segmentation-variable",
  "reconciles_with": ["probe-5-perseg-mega-lr-60bp", "d14-pathb-year-K21-null"]
}
```

## Files

- `scripts/d18_path_b_compound_year.py` — the probe script
- `scripts/artifacts/d18_path_b_compound_year_results.json` — full sweep
- `scripts/artifacts/oof_d18_path_b_compound_year_tau{5000,20000,100000,500000}_strat.npy`
- `scripts/artifacts/test_d18_path_b_compound_year_tau*_strat.npy`
- `submissions/submission_d18_path_b_compound_year_tau{5000,20000,100000,500000}.csv`
  (saved but **NOT** submitted; held for completeness/HEDGE consideration)

## Friction tags

NEW: **`pathb-amp-dead-when-pool-already-routes-segmentation-variable`**.
Origin: this probe. Confirmed by reconciliation with d14 NULL on K=21.

Strengthens existing tag `path-b-amp-only-fires-on-meta-arch-not-base-add`
from "meta-arch redesign required" to "meta-arch redesign that
introduces routing the pool lacks".
