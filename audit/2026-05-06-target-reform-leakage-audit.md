# 2026-05-06 — target reformulation OOF-leakage audit

PI: "top of LB ~0.955 with few/single models. I suspect a single
mechanism." This raised flag for me on the +4.87/+7.67/+9.6 bp
target-reform results from this session — checking OOF correctness.

## Bug found in `compute_targets()`

`scripts/probe_target_reform_v2.py` and `scripts/probe_target_reform.py`
both compute regression targets per (Driver, Race, Year) group using
**ALL train labels in the group**, including labels of rows that will
become fold-validation rows.

```python
for keys, grp in df.groupby(["Driver", "Race", "Year"], sort=False):
    ys = grp["_y"].values        # ALL train labels in group
    total_pits = int(ys.sum())   # includes va-row labels
    cum = np.cumsum(ys)           # cumulative on all labels
    reverse_cum[idxs] = (total_pits - cum).clip(0, 10)
```

When the LightGBM trains on `(X[tr], reverse_cum[tr])`, the target
value for tr-row in group G is shaped by va-row labels in the same
group (via `total_pits` and `cumsum`). The model learns to predict a
target that has been informed by val rows.

This is a **target-construction layer leakage** — not a model-training
layer leakage. Same failure mode as `d12_lr_meta` (which produced
+1.348 bp OOF then −4 bp on LB).

## Strict-OOF audit

Built `scripts/probe_target_reform_strict_oof.py` with per-fold target
construction (only `y[tr]` labels used for both tr and va target
computation).

Results (K=21+1 min-meta, comparing original vs strict-OOF):

| candidate | ORIGINAL Δ | STRICT-OOF Δ | Collapse |
|---|---:|---:|---:|
| reverse_cum | +4.867 bp | **−0.005 bp** | **100%** |
| pit_horizon | +3.191 bp | **+0.302 bp** | **90%** |
| inv_laps_until_pit | +1.899 bp | **+0.234 bp** | **88%** |
| Joint (3 strict) | +7.667 bp | **+0.275 bp** | **96%** |

## Implications for held candidates

| candidate | ORIGINAL OOF Δ vs PRIMARY | LEGITIMATE Δ estimate |
|---|---:|---:|
| `path_b_K22_invlaps_tau20000` | +2.03 bp | ~+0.2 bp (inv_laps leaky) |
| `path_b_K23_dae_invlaps_tau20000` | +2.95 bp | ~+0.5-1 bp (DAE real, inv_laps leaky) |
| `path_b_K25_megapool` various τ | +7.9 to +9.6 bp | ~+0.5-1 bp (only DAE component is real) |

**None of the K=23+ candidates are submission-ready.** Only the new
PRIMARY (`d15b_path_b_K22_dae_only_tau20000`, LB 0.95059) used a
properly OOF-constructed feature (DAE doesn't see target labels).

## Lesson — friction tag

`tag: target-construction-layer-leakage` — when computing a regression
target from `y` per-group (e.g., reverse_cum, total_pits, cumcount of
labels), the per-group computation must use ONLY rows in the current
fold's training set. Using all-train labels per group leaks val-row
labels into tr-row targets via the group-level aggregation. Even with
strict (X[tr], y[tr]) → predict X[va] LightGBM training, the TARGET
itself is contaminated.

**Fix:** for any per-group target derived from y, pass a fold-mask to
the target-computation function. For TEST predictions, full-train
target + full-train fit is correct.

## Implications for strategy

PI's "single model at 0.955" framing now fits the data: my
session's stacking-with-target-derived-bases approach was
chasing inflated OOF that doesn't transfer. The genuine signal
in target reformulations is ≤+0.3 bp at K=21+1, well below the
threshold that would project a top-LB lift.

Path forward:
1. **No submission tomorrow** of any K=23/K=25 Path B variant
   — they're leakage-inflated. Hold path_b_K22_dae_only as
   PRIMARY (real LB 0.95059).
2. **Re-investigate "single model" hypothesis** — what could a
   single model with the available features achieve at 0.955?
   Top scorers found something we didn't.
3. **Pirelli external data scrape** (ISSUES leaf 4a) remains open
   and untested.
4. **Strict-OOF target-reformulation** marginal lifts (+0.2-0.3 bp
   per candidate) are too small for slot-burn unless aggregated
   carefully.

## Files
- `scripts/probe_target_reform_strict_oof.py` — strict-OOF probe.
- `scripts/artifacts/oof_target_reform_*_strict_strat.npy` — strict OOFs.
- `scripts/artifacts/probe_target_reform_strict_oof.json` — std OOFs.
- `scripts/artifacts/probe_min_meta__target_reform_*_strict.json` — gate results.
