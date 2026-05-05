# 2026-05-08 — Day-8 CPU probes: T1.5, T1.3, T1.2 all FALSIFIED

> Per PI directive (wait for GPU; start other tasks): ran the 3 CPU
> candidates from `audit/2026-05-08-strategic-menu-wider-steps.md`.
> All three failed gates. High-information negative results that
> resharpen the strategic menu.

## Results

| Probe | Std OOF | ρ vs PRIMARY | min-meta Δ | K=N stack | Verdict |
|---|---:|---:|---:|---|---|
| T1.5 Deotte L2 (LGBM-aug) | 0.95057 | 0.99517 | n/a | OOF -0.76bp | FAIL |
| T1.5 Deotte L2 (Ridge-aug) | 0.95015 | 0.93648 | n/a | OOF -4.95bp | FAIL |
| T1.5 L3 blend {LR, LGBM, Ridge} | 0.95065 | 0.98535 | n/a | OOF +0.02bp ρ=0.985 | TIE_EXPECTED |
| T1.3 Q12 forced-pit (Variant A) | 0.94439 | 0.90966 | -0.05bp | K=19 ρ=0.99994 | FAIL |
| T1.3 Q12 forced-pit (Variant B) | 0.94518 | 0.92570 | -0.06bp | K=19 ρ=0.99994 | FAIL |
| T1.2 Poisson laps-until-pit | 0.56951 | 0.36998 | -0.08bp | K=19 ρ=0.99982 | FAIL |

PRIMARY anchor: d6_k18_multi_rule OOF 0.95065, LB 0.95026.
Wall total: ~14 min CPU.

## Mechanism analysis — three falsification modes

### T1.5 Deotte L2 — meta-only changes don't move LR-locked pool
Adding {std, mean, min, max, range, p25, p75} of per-row base
disagreement to LGBM-L2 / Ridge-L2 produced rank shuffle (ρ=0.985-
0.995) but ZERO OOF lift. L3 blend ties exactly. **Confirms d3-
endgame thesis**: the LR-meta is at the information ceiling of the
18-base pool. Different meta-families (LR, LGBM, Ridge) produce
ρ-shifted predictions but no OOF gain because the SIGNAL each can
extract from the 18 base outputs is the same.

**Implication**: any meta-only change is dead. F1.2's +2.1bp came
from new BASES (4 rule_residuals), not new meta. T1.5 is FALSIFIED.

### T1.3 Q12 forced-pit — rule_residual on raw features collapses
Q12 mandatory-2-compound is a HARD CONSTRAINT not in any base.
Rule features (`compounds_used_so_far`, `must_change_compound`,
`forced_pit_pressure`) computable for 91% of size>=10 dry groups.
Standalone OOF 0.94518 with ρ=0.926 vs PRIMARY — legitimately
diverse. But minimal-meta gate FAILED by 0.06bp; K=19 stack
collapsed to ρ=0.99994 with Q12 base getting near-zero L1.

Why: the residual GBDT trained on (raw features ± Q12 features)
predicts the same axis-aligned boundary that the existing 14
GBDT bases already capture. Q12-specific signal applies to only
4.26% of rows (must_change=1 share); for the other 95.7%, the
base's prediction is dominated by the residual GBDT, which is
information-redundant with the pool. The LR meta routes around
the redundant 95.7% subspace and finds no new signal in the 4.26%.

**Implication**: T2.1 (next_compound feature, 68% coverage) and
T2.2 (prev_compound × laps_into_stint) are AT RISK of the same
failure mode. Coverage is higher than Q12's 4.26%, but the
mechanism is identical (rule_residual on raw features →
GBDT-class consensus collapse).

### T1.2 Poisson laps-until-next-pit — REDUNDANT with existing pool
Standalone OOF AUC 0.56951 (basically random on binary target).
Cause-1: 79.77% of rows censored (within (Driver,Race,Year,Stint)
groups, no future pit visible).

But the deeper finding: **`scripts/b_laps_until_pit.py` already
implements laps-until-pit reformulation**, and `a_horizon_shift.py`
already implements horizon-shifted targets. Both bases are in M5q.
The strategic menu's "multi-formulation L1" lift was ALREADY
CAPTURED at Day-2.

**Implication**: T1.2 should be removed from the wider-step menu —
the lever is exhausted. Net "wider step" reformulations not yet
tried:
  - Censored-Cox / nnet-survival with PROPER censoring handling
    (vs my naive sentinel-99 Poisson).
  - Multi-class classification on (laps_until_pit ∈
    {0, 1-2, 3-5, 6+, censored}) — preserves discrete bin info.
  - Multi-task: jointly predict (PitNextLap, will_change_compound,
    Stint_completion) — share representations.

These are still wider but each requires new infrastructure.

## What this collectively implies (load-bearing)

1. **Pool calibration is genuinely tight (P10 confirmed).** Any
   new base trained on the SAME input data with a GBDT-class
   inductive bias and a target that's a monotone-equivalent of
   binary-AUC will collapse in LR-meta. Three independent probes
   today confirm this regime.

2. **F1.2 lift came from rule_residual where the rule itself
   captured genuinely new pairwise signal (Compound×TyreLife,
   Compound×Stint, Driver×Compound, Year×Race) that GBDTs split
   in a different order than the rules' Bayesian-smoothed table.**
   Q12 failed because the rule's high-coverage prediction
   (`compounds_used_so_far` is mostly correlated with Stint) didn't
   add new pairwise information.

3. **The remaining wider-step CPU bets that are NOT redundant:**
   - **External-data lookups** (Q1 Pirelli pit windows, Q3 SC
     probability). NEW STRUCTURE not derivable from raw features.
     Race-marginal preserved under CTGAN. EV +2-5bp each.
   - **EmbMLP on PyTorch CPU.** Different inductive bias for
     high-cardinality Driver (887 → 16-32 dim trainable embedding).
     Distinct from RealMLP-TD's embedding scheme. EV +1-3bp.
   - **Bayesian hierarchical stacking (T2.5).** Different meta
     STRUCTURE (Gaussian partial-pooling); could resist the
     LR-rank-lock if implemented properly. EV +1-6bp.

4. **The remaining wider-step GPU bets (PRIORITY when GPU returns):**
   - **TabM** (T1.1). NOT yet falsified. Different optimization
     landscape, K=32 internal heads. EV +2-8bp.
   - **Hazard-rate NN** (T1.4). Different problem structure (K=20
     hazard buckets, NLL loss over full curve). EV +1-7bp.
   - **SCARF/VIME pretrain on aadigupta1601 unlabeled** (T2.6).
     Different unlabeled corpus avoids d5 partial-pseudo amp.
     EV +1-6bp on top of NN base.

## Updated menu ranking (post-falsification)

Removed/demoted:
- **T1.5 Deotte L2** → DEMOTED to "tried and falsified".
- **T1.3 Q12 single-rule** → DEMOTED. Variant of high-coverage
  rule_residuals (T2.1/T2.2) might still pass; single-rule is dead.
- **T1.2 Poisson reformulation** → REMOVED (redundant with existing
  `a_horizon`, `b_lapsuntilpit`).

Promoted by elimination:
- **T1.1 TabM** (GPU) — sole remaining unfalsified Tier-1.
- **External-data rules (Q1, Q3)** — promote from Tier-2 to top
  of CPU queue.
- **T2.5 Hierarchical Bayesian** — promote; different meta
  STRUCTURE (not just family) may resist rank-lock.

## Artifacts saved

- `scripts/d8_l2_stacking.py` + `oof_d8_l3_blend_strat.npy` etc.
- `scripts/d8_q12_atomicity_check.py` (probe; 91% atomicity
  confirmed for size>=10 dry groups)
- `scripts/d8_q12_forced_pit.py` + `oof_d8_q12_v_b_strat.npy`,
  `oof_d8_k19_q12_strat.npy` (held; do not submit)
- `scripts/d8_poisson_lapsuntil.py` + `oof_d8_poisson_lapsuntil_strat.npy`,
  `oof_d8_k19_poisson_strat.npy` (held; do not submit)
- `scripts/artifacts/d8_*_results.json` — numerical detail

## Held submissions (do NOT submit)

- `submissions/submission_d8_l3_blend.csv` — TIE OOF, ρ=0.985
  rank-shifted but no OOF lift; mechanism-class-equivalent to LR
  baseline per d3-endgame.

End — high-info day; menu sharpened.
