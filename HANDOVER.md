# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file with the next
> session's brief.

---

## Today's session — Day 6 (2026-05-07)

**Read order on session start** (skip the default; this file is the
synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R14)
2. `audit/2026-05-06-d5-slot1-partial-pseudo-result.md` — gap-widening falsification
3. `audit/2026-05-06-d5-path-b-phase2.md` — Path B build that produced slot-1 candidate
4. `audit/2026-05-06-d5-gbdt-meta-k15.md` — meta-add ceiling confirmed
5. `audit/2026-05-06-d5-tabnet-smoke-fail.md` — TabNet parked
6. `audit/2026-05-06-d5-path-c-recursive.md` — recursive K=15 nulls
7. `scripts/pre_submit_diff.py` — MANDATORY before every submit

Open with a 3-bullet read-back of state + first action.

## Where we are

- **Day 6, 0/10 used today.** Day-5 closed with 1/10 used (slot 1
  partial-pseudo K=14 LB 0.94963 = −4.2bp from M5q, GAP WIDENED).
- **PRIMARY** = M5q LB 0.95005 (unchanged; partial-pseudo regressed).
- **Headroom to top-5%** (0.95345): **34.0bp**.
- **21 days remaining** (deadline 2026-05-31). 8 slots/day available.
- **CRITIC-LOOP TRIGGER FIRED.** Rule 14: OOF→LB gap drift ≥2bp on
  consecutive submits. d4 slot-2 (gap +1bp vs M5q's −5bp = +6bp drift)
  + d5 slot-1 (gap −12bp vs M5q's −5bp = −7bp drift). MANDATORY audit
  before adding any new mechanism family. Output:
  `audit/2026-05-07-d6-critic-loop.md` BEFORE first probe.

## Day-5 outcomes (single-paragraph synthesis)

Day 5 stress-tested the M5q pool ceiling and found two: a hard meta
ceiling (six probes vs M5q pool — recursive K=15 LR/GBDT-meta sweeps,
recursive standalone 2-base, TabNet smoke — all NULL or FAIL) and a
hard PSEUDO-CHANNEL ceiling. Path B Phase 1+2 cleared OOF gates
strongly: e3_hgbc rebuild +4.1bp, all 5 fast-CPU bases lift +2-19bp,
partial-pseudo K=14 stack OOF 0.95082 (+2.54bp vs M5q anchor),
ρ=0.99836 REAL_DELTA. **But slot-1 LB came back at 0.94963 (−4.2bp)
with the OOF→LB gap WIDENING from −5.2bp to −12.0bp** — the pseudo
channel amplifies M5q's systematic biases (OOF rewards, LB
penalizes). L1 reshuffling confirms: pseudo-rebuilt bases got
promoted (e3 +116%, m2_xgb +130%) at the cost of original anchor
bases (cb_slow-wide-bag −72%, a_horizon −85%); the meta optimized
for OOF and chose wrong for LB. **Path B at the broad gate (180k/188k
rows) is FALSIFIED.** Phase 3 queue (CB rebuilds, d2a_te, RealMLP
GPU rebuild) HELD pending re-design.

## Day-6 plan — critic-loop FIRST, then re-prioritize

Per Rule 14, BEFORE any new mechanism family: write
`audit/2026-05-07-d6-critic-loop.md` covering:
- 2-submit consecutive negative LB delta (slot-2 d4 + slot-1 d5)
- OOF→LB gap drift quantification across all 13 LB submits
- Whether the gap behavior is systematic or stochastic
- 5 untried mechanism families with citations (Rule 7 "research
  before saturation"; this is null #2 in a row)

### Re-rankable next moves (post critic-loop)

Sorted by expected EV per slot/hour:

1. **2-base [M5q, recursive] standalone LB probe** (~5min build, 1
   slot). The K=2 OOF stack was −0.2bp but rank structure is
   structurally different from K=15. Pre-submit-diff vs M5q first;
   if ρ < 0.999, slot. Recursive standalone OOF was 0.94994 (best
   single GBDT to date); the structural difference may transfer
   without the K=15 LR rank-lock.

2. **Tighter pseudo gate Path B retry.** M5q ∈ [0.99, 1.0] ∪ [0,
   0.01] AND (intersection, not union) ≥12/13 multi-base agreement.
   Expected pseudo set: ~30-50k rows (vs 180k today). Hypothesis:
   keeps highest-confidence LB-relevant signal, drops over-amp.
   Cost: 1-2h CPU rebuild of 6 bases + slot probe. Risk: still
   over-amp-prone given d5 evidence.

3. **Sample-weight pseudo.** Train bases on full real-train ∪
   pseudo-test with pseudo `sample_weight ∈ {0.1, 0.3, 0.5}`.
   Reduces OOF-gaming. Faster than retry-with-tighter-gate. Two
   slot probes needed to find the LB-optimal weight.

4. **Multi-seed RealMLP bag (HANDOVER A.1).** Known +1-3bp prior on
   a base that gave 10× OOF→LB amplification originally. ~6h Kaggle
   GPU overnight. Lower variance than NN-family-multiplication.

5. **Override-mechanism probe (Day-3 R7 rule).** Per-row override
   of M5q's predictions where M5q is uncertain ([0.4, 0.6]) AND
   recursive disagrees. Flip count ~200-500 expected. Per Rule R7,
   <200 → HEDGE only; >200 needs explicit PI sign-off.

### Explicitly NOT recommended

- **Push RealMLP pseudo-rebuild kernel as planned in fork-3 part 2.**
  Slot-1 result said this would burn 6h GPU chasing the same gap-
  widening. Remains HELD until Path B retry direction is settled.
- **Phase 3 CatBoost CPU rebuilds at the broad pseudo gate.** Same
  EV-negative pattern.

## Falsified Day-5 (extended)

- TabNet at default pytorch-tabnet config (n_d=32, cat_emb_dim=4,
  120 epochs) — fold-0 0.93532, FAIL gate. Under-trained, not
  under-priced. Re-test only after Path B's ceiling is mapped.
- Recursive GBDT (HGBC + M5q_oof_proba feature) at standalone +92bp
  baseline; null at 2-base, K=15 LR-stack, K=15 GBDT-meta. 3rd
  independent rank-lock confirmation.
- GBDT-meta over K=15 (recursive in pool) — all 3 variants worse
  than d4 K=14 by ~1bp uniformly. Meta divergence ceiling fixed.
- Pool composition variants (drop e3 / drop f1+f2 + recursive) — null.
- **Path B at broad pseudo gate (180k/188k rows, union of M5q
  ∈ [0.95,1] ∪ [0,0.05] and ≥10/13 vote)** — OOF +2.54bp, LB −4.2bp.
  Pseudo channel over-amplifies M5q's biases at this scale.

## Calibration ladder snapshot (Day 6 morning)

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| baseline_two_anchor | 0.94075 | 0.94113 | LB-proxy ✓ gap +3.8bp |
| e3_hgbc | 0.94876 | 0.94870 | best single GBDT pre-CB |
| m5h | 0.95043 | 0.94991 | gap −5.2bp |
| **m5q (M5h + RealMLP, K=14)** | **0.95057** | **0.95005** | **PRIMARY**; +14bp |
| m5_meta_lgbm_shallow (slot 2 d4) | 0.95048 | 0.95001 | -4bp; meta-switch costs |
| d5_recursive_m5q (HGBC + M5q feat) | 0.94994 | n/a | std-alone +92bp; K=15 stacks NULL |
| d5_e3_pseudo (Phase 1 MVP) | 0.94917 | n/a | +4.1bp anchor; ρ=0.996 |
| d5_partial_pseudo_m5q (K=14) | 0.95082 | **0.94963** | **slot-1 −4.2bp**; gap WIDENED |

## OOF→LB gap ledger (load-bearing for critic-loop)

| Submit | OOF | LB | Gap | Δ vs M5q gap (-5.2bp) |
|---|---:|---:|---:|---:|
| baseline | 0.94075 | 0.94113 | +3.8bp | — |
| m5b | 0.94926 | 0.94891 | −3.5bp | — |
| m5d | 0.95023 | 0.94963 | −6.0bp | — |
| m5h L1pruned | 0.95043 | 0.94991 | −5.2bp | 0 |
| m5h2 / m5j (tied) | 0.95044 | 0.94991 | −5.3bp | −0.1bp |
| m5p / m5n_3b | 0.94808–.94839 | 0.94700–.94754 | −10.8 to −10.6bp | −5.4 to −5.6bp |
| **m5q (PRIMARY)** | 0.95057 | 0.95005 | −5.2bp | 0 |
| m5_meta_lgbm_shallow | 0.95048 | 0.95001 | −4.7bp | +0.5bp |
| **d5_partial_pseudo (slot-1)** | 0.95082 | 0.94963 | **−12.0bp** | **−6.8bp** |

**Two consecutive negative drifts**: d4 slot-2 (+0.5 then) → d5 slot-1
(−6.8bp). Critic-loop fires per Rule 14.

## Held submissions (do not submit blindly)

- (carried forward) `submission_m5x_yetirank.csv` / `m5z_yetirank_nb.csv` — TIE_EXPECTED
- `submission_m5_meta_lgbm_medium.csv` / `m5_meta_hgbc.csv` — meta variants
- `submission_d5_meta_k15_*.csv` × 3 — K=15 GBDT-meta NULLs
- `submission_m5_k15a/b/c.csv` — K=15 LR stack NULLs
- `submission_d5_partial_pseudo_m5q.csv` — burned at LB 0.94963 (do not resubmit)

## Critical operating rules (FRESHLY VIOLATED — READ THESE)

1. **Pre-submit-diff before EVERY submit.** Run
   `python3 scripts/pre_submit_diff.py <candidate.csv>`. ρ ≥ 0.999 → tie.
2. **1-fold smoke before any GPU 5-fold.** Day-5 TabNet smoke saved
   the 5-fold cost (failed at fold 0).
3. **Strat-only Day-3+** (Rule R1).
4. **Don't drop bases purely on L1/diversity grounds.** Minimal-
   basis falsified Day-3.
5. **Bigger-moves rule: weight candidates by EV_bp / day_invested.**
6. **Strategy review before propose: grep audit/ for prior probes.**
7. **OOF lift ≠ LB lift.** s6e5's gap behavior is non-monotonic in
   pseudo-aug intensity; Day-5 slot 1 was the smoking gun. Always
   check the OOF→LB gap ledger before treating an OOF lift as
   slot-worthy.
8. **Critic-loop on 2 consecutive gap drifts ≥2bp** (Rule 14).
   Fired today; first action Day 6 must be the audit, not a probe.

## Pointers

- `audit/2026-05-06-d5-slot1-partial-pseudo-result.md` — falsification
- `audit/2026-05-06-d5-path-b-phase2.md` — Path B build
- `audit/2026-05-06-d5-gbdt-meta-k15.md` — meta-add ceiling
- `audit/2026-05-06-d5-tabnet-smoke-fail.md` — TabNet parked
- `audit/2026-05-06-d5-path-c-recursive.md` — recursive K=15 nulls
- `audit/2026-05-05-d4-gbdt-meta-breakthrough.md` — d4 slot-2 envelope
- `audit/friction.md` — logged failure modes
- `scripts/pre_submit_diff.py` — MANDATORY pre-submit gate
- `scripts/d5_pseudo_label_mvp.py` — Phase 1 MVP (broad gate)
- `scripts/d5_pseudo_phase2_rebuild.py` — Phase 2 driver (broad gate)
- `scripts/d5_recursive_m5q_gbdt.py` — Path C base build
- `scripts/d5_recursive_stack_k15.py` — K=15 LR sweep
- `scripts/d5_gbdt_meta_k15.py` — K=15 GBDT-meta sweep
- `kernels/tabnet-smoke-gpu/` — failed smoke (do not promote)
- `kernels/realmlp-pseudo-gpu/` — NOT BUILT; held pending Path B retry direction
