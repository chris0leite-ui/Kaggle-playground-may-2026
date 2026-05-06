# 2026-05-06 — K=21 blend probe + ρ inventory

Two cheap probes via the new harness. ~5s wall total. Shape of
"experimentation culture": small, parallel, BOTE-graded, and
rule-out is as valuable as rule-in.

## Probe 1 — K=21 alternative blenders

Computed 4 aggregators across the 21 PRIMARY-pool base predictions
(arithmetic mean / geometric mean / per-row rank-mean / trimmed
mean drop top-3+bot-3) and gated each against PRIMARY (d13e
Compound×Stint τ=20k).

| blender | std OOF | Δ vs PRIMARY | ρ vs PRIMARY | flips +→− / −→+ | verdict |
|---|---:|---:|---:|---|---|
| mean       | 0.94767 | −31.55 bp | 0.9332 | 1760 / 0  | FAIL |
| gmean      | 0.94761 | −32.16 bp | 0.9683 | 1853 / 0  | FAIL |
| rank_mean  | 0.94794 | −28.91 bp | 0.9875 | 38 / 4470 | FAIL |
| trimmed    | 0.94895 | −18.84 bp | 0.9765 | 633 / 98  | FAIL |

**Hypothesis ruled out:** that the K=21 LR meta is over-weighting
some bases and a simple aggregator could match or beat it. Every
naive aggregator regresses 19–32 bp standalone. The LR-on-`expand([raw,rank,logit])`
meta is doing real work — its weights are not redundant with any
democratized blend.

Side observation: the rank-mean blend has a deeply asymmetric flip
pattern (38 demoted out of PRIMARY's top-1%, but **4470 promoted
new** rows into top-1%). Rank-mean democratizes the rare-class signal
across all 21 bases; PRIMARY concentrates it via the LR meta. If
private LB has different rare-class structure than public LB, this
is information about the variance band, not a hedge candidate
(too far down OOF).

## Probe 2 — ρ inventory of held candidates vs PRIMARY

Mapped 22 held submissions/OOF artifacts vs PRIMARY by Spearman ρ
and rare-class flip pattern.

Buckets:
- **TIE_EXPECTED (ρ ≥ 0.999):** 2 (`d13e_compound_stint_tau5000`,
  `d13e_compound_stint_tau100000`) — submitting either is a wasted slot.
- **Near-tie (0.995 ≤ ρ < 0.999):** 10 — d13/d13e/d14 Path B variants
  and `d12_lr_meta`. Worth grading by std OOF; most fall within 0.5 bp
  of PRIMARY OOF.
- **Diverse (ρ < 0.995):** 10 — includes 4 simple blends, d10d
  leak-corrected meta, d12 GroupKF meta, d14 H1 (failed FM aug15).

Notable diverse rows that are *not* dead:

| name | ρ | std OOF | ΔOOF | flip ratio | note |
|---|---:|---:|---:|---:|---|
| `d12_lr_meta` (clean K=20 GKF) | 0.9958 | 0.95073 | −1.01 bp | 0.297 | borderline hedge candidate |
| `d12_groupkf_meta` | 0.9874 | 0.94776 | −30.75 bp | 0.170 | fit on GKF-OOFs; eval space ≠ Strat-space; private-LB hedge re-eval needed |
| `blend_rank_mean_K21` | 0.9875 | 0.94794 | −28.91 bp | 0.009 | ruled out as primary-replacement; structurally diverse |

**Two takeaways for HEDGE staging:**
1. `d12_lr_meta` is the cleanest near-tie hedge candidate
   (−1.0 bp public, ρ 0.996, flips 84/283) — flip ratio 0.297 puts it
   in HEDGE-eligible band per R7 (<200 flips with single-direction
   imbalance, public-LB defendable).
2. `d12_groupkf_meta`'s standalone OOF here is computed on the same
   y but the predictions come from a GKF-fit meta, so the Strat-OOF
   AUC is NOT what we'd see on a row-iid public LB. The −30 bp gap is
   evaluation-space mismatch, not a real public-LB regression of that
   magnitude. **For private LB hedge evaluation, we need a different
   gate** (e.g., compare flip ratios in private-leakage-blocked
   distribution, or run a calibrated GKF probe). Flagged as a future
   harness extension.

## Harness amendments surfaced

These probes uncovered two harness gaps:

1. **Eval-space mismatch:** `gate(d12_groupkf_meta)` reports −30 bp
   but that's because we're scoring its predictions against the *Strat*
   y partition. The candidate was constructed under GKF; the right
   eval is GKF-OOF AUC. Future: optional `--cv` flag to gate against
   a saved GKF fold map.
2. **Flip ratio direction signal:** rank-mean blend's 38/4470 flip
   pattern is qualitatively different from `d12_groupkf_meta`'s
   612/104. Current G3 ratio collapses both to ~0.01 / 0.17 but the
   *sign* of the asymmetry matters for HEDGE eligibility (R7).

## Files
- `scripts/probe_blends_K21.py` — blend probe.
- `scripts/probe_rho_inventory.py` — ρ inventory (22 candidates).
- `scripts/artifacts/probe_blends_K21.json`
- `scripts/artifacts/probe_rho_inventory.json`
- `scripts/artifacts/oof_blend_{mean,gmean,rank_mean,trimmed}_K21_strat.npy`
- `scripts/artifacts/test_blend_{mean,gmean,rank_mean,trimmed}_K21_strat.npy`
