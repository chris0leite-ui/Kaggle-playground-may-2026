# d18b Path-B Alt Axes on K=24 — RESULT: NULL across all 3 axes

Branch `claude/logistic-regression-ensemble-0PNkA`. Follow-up to d18
NULL on Compound × Year. Tested 3 alternative segmentation axes the
K=24 pool does not natively route, to verify the new mechanism finding
`pathb-amp-dead-when-pool-already-routes-segmentation-variable`.

## TL;DR

| Axis | n_seg fit | best τ | best OOF Δ vs PRIMARY | best ρ | gate |
|---|---:|---:|---:|---:|---|
| (Driver_cluster, Stint) | 13/24 | 100k | **+0.36 bp** | 0.99941 | FAIL (<+0.5) |
| (Race_class, TyreLife_q5) | 20/20 | 500k | **+0.01 bp** | 0.99999 | FAIL (~0) |
| (Position_q5, Compound) | 20/25 | 500k | **+0.00 bp** | 0.99998 | FAIL (~0) |
| Global K=24 LR-meta (PRIMARY ref) | — | — | (baseline 0.95385) | — | — |

**No axis passes the +0.5 bp gate. No submission.**

Wall: 1054s (≈18 min, 4 cores).

## Full sweep (all 4 τ × 3 axes)

### Axis 1: (Driver_cluster, Stint) — k_drv=4, n_stint=6, 24 cells
4-cluster KMeans on Driver-aggregate features (LapTime, TyreLife,
Position, RaceProgress); cross with Stint clipped to [0,5]. 13/24 cells
≥1k rows.

| τ | OOF | Δ bp | ρ | flips (− / +) | flip ratio |
|---|---:|---:|---:|---:|---:|
| 5k | 0.95381 | −0.43 | 0.99678 | 265 / 168 | 0.634 |
| 20k | 0.95386 | +0.11 | 0.99799 | 186 / 131 | 0.704 |
| **100k** | **0.95388** | **+0.36** | **0.99941** | **87 / 63** | **0.724** |
| 500k | 0.95387 | +0.17 | 0.99994 | 25 / 15 | 0.600 |

Best of the 3 axes — but still under the +0.5 bp standalone-OOF gate.
Net positive flip count (+24 of 150 net at τ=100k) suggests a real but
tiny signal on the Driver-cluster axis.

### Axis 2: (Race_class, TyreLife_q5) — k_race=4, q=5, 20 cells
4-cluster KMeans on Race-aggregate features × KBins(5, quantile,
ordinal) on TyreLife. 20/20 cells ≥1k. Largest cells.

| τ | OOF | Δ bp | ρ | flips (− / +) |
|---|---:|---:|---:|---:|
| 5k | 0.95358 | −2.65 | 0.99696 | 167 / 209 |
| 20k | 0.95375 | −0.94 | 0.99865 | 106 / 112 |
| 100k | 0.95384 | −0.07 | 0.99981 | 37 / 28 |
| 500k | 0.95385 | +0.01 | 0.99999 | 5 / 4 |

Monotone decay to global. NULL.

### Axis 3: (Position_q5, Compound) — q=5, 5×5 = 25 cells
KBins(5, quantile, ordinal) on Position × Compound (4 compounds + WET
= 5 levels in train; 1 of 25 cells empty). 20/25 cells ≥1k.

| τ | OOF | Δ bp | ρ | flips (− / +) |
|---|---:|---:|---:|---:|
| 5k | 0.95361 | −2.43 | 0.99692 | 274 / 291 |
| 20k | 0.95375 | −1.00 | 0.99842 | 185 / 213 |
| 100k | 0.95384 | −0.12 | 0.99970 | 59 / 88 |
| 500k | 0.95385 | +0.00 | 0.99998 | 15 / 18 |

Monotone decay to global. NULL.

## Interpretation

Three alternative axes — one with novel Driver clustering (not in any
pool base), one with custom Race clustering, one quintile-binning on
Position — all fail to fire Path-B amp on K=24.

The new friction
`pathb-amp-dead-when-pool-already-routes-segmentation-variable` from
d18 is **strengthened, not weakened**:

- d18 Compound × Year — NULL (cb_year-cat absorbs)
- d18b axis 1 Driver_cluster × Stint — +0.36 bp (best, but <gate)
- d18b axis 2 Race_class × TyreLife_q5 — NULL (cells too pooled)
- d18b axis 3 Position_q5 × Compound — NULL (Position already in
  continuous bases)

The amp-eligible Compound × Stint axis fires (~+8 bp LB lift in d13e,
+9 bp LB lift in d17 PRIMARY) because Stint indexing is genuinely
absent from the GBDT-pool's native routing — there's no "cb_stint-cat"
in the pool. Every other tested axis is either:
1. Already routed by a pool base (Year via cb_year-cat, Position via
   continuous bases d16/v3/v4), or
2. So coarse that the per-segment LR-meta cannot find new structure
   the global LR-meta misses.

Driver_cluster × Stint at τ=100k showing a faint +0.36 bp suggests the
mechanism *does* still exist on a Driver-derivative axis, but at <1 bp
it's lost in the OOF→LB calibration noise (gap −6 bp on PRIMARY).
**Not submission-worthy.**

## Reconciliation with prior probes

| Probe | Pool | Axis | Result |
|---|---|---|---|
| d13 Path-B | K=14 (early) | Stint | LB +7 bp |
| d13e Path-B | K=22 | Compound × Stint τ=20k | LB +8 bp |
| d14 Path-B sweep | K=21 | 9 cohort axes (Year/YxStint/Race × τ) | NULL (all <PRIMARY) |
| d17 Path-B PRIMARY | K=23 v4+h1d | Compound × Stint τ=100k | LB +9 bp |
| **d18 Path-B** | **K=24** | **Compound × Year** | **NULL** |
| **d18b axis 1** | **K=24** | **Driver_cluster × Stint** | **+0.36 bp <gate** |
| **d18b axis 2** | **K=24** | **Race_class × TyreLife_q5** | **NULL** |
| **d18b axis 3** | **K=24** | **Position_q5 × Compound** | **NULL** |

The Path-B amp axis is **saturated on Compound × Stint for this pool**.
4 alternative crosses now confirmed dead or sub-gate. Future Path-B amp
on K=24 would require either:
1. A genuinely novel axis derived from external data (pit-window
   probabilities, Pirelli compound metadata) — HANDOVER A4 path.
2. Dropping a base whose routing absorbs the candidate axis — costly.
3. Building a per-segment **non-LR meta** (e.g. per-segment small GBDT)
   that captures non-linear segment-specific routing structure.

## Submissions impact

7/10 used today; 3 slots remain. **No submit from any d18b axis.**
PRIMARY remains `d17_path_b_K23_v4_h1d_tau100000` LB 0.95354.

## Decisions log entry (sketch)

```json
{
  "ts": "2026-05-07T20:26:?Z",
  "branch": "claude/logistic-regression-ensemble-0PNkA",
  "probe": "d18b_path_b_alt_axes_K24",
  "family": "meta_arch_redesign",
  "metric_aligned": true,
  "agent_predicted_lb_bp_midpoint": 1,
  "agent_predicted_lb_bp_range": [-3, 1, 5],
  "pi_predicted_lb_bp": null,
  "actual_oof_delta_vs_primary_bp": {
    "driver_cluster_stint_tau100000": 0.36,
    "race_class_tyrelife_q5_tau500000": 0.01,
    "position_q5_compound_tau500000": 0.00
  },
  "actual_lb_bp": null,
  "verdict": "NULL — no axis passes +0.5 bp gate; not submitted",
  "mechanism_finding": "pathb-amp-dead-when-pool-already-routes-segmentation-variable (re-confirmed across 3 axes)",
  "reconciles_with": ["d13e-compound-stint-firing", "d18-compound-year-null", "d14-cohort-sweep-null"]
}
```

## Files

- `scripts/d18b_path_b_alt_axes.py` — probe script
- `scripts/artifacts/d18b_path_b_alt_axes_results.json` — full sweep
- `scripts/artifacts/oof_d18b_{driver_cluster_x_stint,race_class_x_tyrelife_q5,position_q5_x_compound}_tau{5000,20000,100000,500000}_strat.npy`
- `scripts/artifacts/test_d18b_*` (matching test predictions)

## Friction tag updates

`pathb-amp-dead-when-pool-already-routes-segmentation-variable` —
**3rd cross-confirmation**. Origin d18 Compound × Year; now confirmed
on (Race_class × TyreLife_q5), (Position_q5 × Compound), and
sub-gate-passing on (Driver_cluster × Stint).

The amp axis is **uniquely Compound × Stint** on this pool. Future
Path-B work should pivot to (a) external-data axes (HANDOVER A4) or
(b) non-LR per-segment meta classes.

## What this closes

The HANDOVER A4 "Alternative seg crosses" list (`(Driver_clustered,
Stint)`, `(Race_class, TyreLife_q5)`, `(Position_q5, Compound)`) is
now empirically tested and **closed null**. Path-B mechanism on K=24
is exhausted at the LR-meta level for label-free crosses.
