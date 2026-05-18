# 2026-05-18 — Round 3 execution: ceiling validated; snapshot gap identified

Triggered by: PI request "go through another round, according to our
skill" after Round 2's 11/11 null. Skill prescription at plateau:
problem-solving.md step-1 re-entry + personas.md rotation + strategy-
critic.md section-5 headroom math.

**Net result: row-feature ceiling claim VALIDATED. K=27 + Path-B is
the best blend reachable in the current snapshot at OOF 0.95433 / LB
0.95368 — 1.8 bp BELOW the actual K=11+K=9 PRIMARY (LB 0.95386).
The snapshot is missing the kNN diversity layer that drives the
PRIMARY's lift over K=27.**

## Phase 0 — Proxy-substitution audit (CRITICAL VALIDATION)

The Senior ML Engineer persona surfaced a load-bearing concern:
the Round-2 11/11 null was gated against K=4 (homogeneous tree
pool); the actual PRIMARY is K=11+K=9 with slim-kNN diversity.
If candidates revive under a more diverse anchor, the ceiling
claim is anchor-conditional.

**Test 1 — Multi-anchor retest** (3 candidates × 3 anchors):

| Candidate | K=4+1 Δ bp | K=21+1 Δ bp | K=4+K27super+1 Δ bp |
|---|---:|---:|---:|
| K4_conformal_widths | -0.065 | +34.361 | +0.000 |
| K4_rrf_k60 | +0.023 | +14.559 | +0.026 |
| K4_meta_lgbm_rank | -0.050 | +29.168 | +0.007 |

The K=21 lift (+14 to +34 bp) looks dramatic but is **misleading**:
K=21 base OOF is only 0.95073 — far below K=4's 0.95399. The
candidates recover K=4's level when added to K=21, but don't
push past it. K=4+K27super pool (5-base; 4 K=4 anchor + K=27
super-base) achieves 0.95429 OOF and is null against all 3
candidates.

**Test 2 — LR-meta C-sweep on K=4+1** (Rule 21 family-falsification):

| C | base OOF | with OOF | Δ bp |
|---:|---:|---:|---:|
| 0.01 | 0.95399 | 0.95399 | -0.024 |
| 0.10 | 0.95399 | 0.95399 | -0.017 |
| 1.00 | 0.95399 | 0.95399 | -0.050 |
| 10.0 | 0.95399 | 0.95399 | -0.052 |
| 100. | 0.95399 | 0.95399 | -0.052 |

LGBM-rank candidate is null at every C value across 4 orders of
magnitude. The "LR with log-loss is loss-optimal" claim is NOT
C=1.0-specific; it's robust across the L2 regularization range.

**Test 3 — K=4 vs K=27 residual correlation**:

- K=4 LR-meta OOF AUC: 0.95399
- K=27+Path-B OOF AUC: 0.95432
- **Pearson(resid_K4, resid_K27) = 0.99810**
- Spearman(K4_pred, K27_pred) = 0.99546

**KILLER finding.** K=4 (homogeneous CatBoost/tree pool) and K=27
(structurally diverse with FM/MLP/NN bases) produce essentially
the SAME residuals (ρ = 0.998). The senior engineer's "K=4 is
too homogeneous" hypothesis is **REFUTED at the residual level**.
Any mechanism that nulls at K=4+1 will null at K=27+1 too (and
the multi-anchor retest in Test 1 confirms this empirically).

**Phase 0 verdict:** the row-feature ceiling claim is anchor-
robust. Round 2's 11/11 null is NOT a methodological artifact.

## Phase 1 — Caruana hill-climb (P1.2; only Phase-1 pick the snapshot supports)

Forward-selection-with-replacement on rank-mean of 11 available
candidates (LB-confirmed K=4-Path-B / K=27-Path-B / K=23-v4h1d
plus 5 R5 hedge OOFs plus 5 Round-2 null OOFs). Stop at plateau.

**Result:** plateaued at step 33 with OOF AUC 0.95433.

Weights:
- K27_pathb: **0.879** (dominant)
- R2_lgbm_rank: 0.061
- R2_trimmed: 0.030
- K4_pathb: 0.030

**+0.11 bp vs best single anchor (K27_pathb at 0.95432).** Tie at
OOF precision. ρ vs d13e PRIMARY: 0.987 (REGRESSION_RISK band).

The Caruana blend effectively DEGENERATES to K=27+Path-B alone.
The R2-null candidates contribute minimally; K=4_pathb contributes
3%. **The snapshot has no real diversity beyond K=27+Path-B.**

## Phase 1 / Phase 2 — DEFERRED

- **P1.1 C1 OpenF1** (~45 min CPU): deferred. Phase-0-confirmed
  ceiling makes the EV uncertain; would need K=11 anchor to gate
  against current PRIMARY.
- **P1.3 C2 swap-noise DAE** (~2-3 hr T4): deferred. Same anchor
  issue. Reserve for next session with K=11 OOFs rebuilt.
- **P1.4 Multi-seed bag of PRIMARY** (~12 hr concurrent): deferred.
  Cannot bag K=11 pipeline without the 6 missing slim-kNN bases.
- **P2 Hedge prep**: deferred to next session.

## Strategic finding: snapshot completeness is the gate

The Round-3 audit revealed a CRITICAL operational gap: the
2026-05-08 artifact snapshot contains the K=4 anchor and the
K=27+Path-B super-base, but NOT the 6 slim-kNN bases
(dgp_v3_qAT_K1, qAV, qAO, qAA, qAF, qAK) that constitute the K=11
diversity layer of the current PRIMARY.

Without these bases on disk:
- We cannot reconstruct K=11 LR-meta locally.
- We cannot gate new mechanisms at the real PRIMARY level
  (K=11+1) — only at K=4+1 (3.5 bp behind) or K=27+1 (1.8 bp
  behind).
- Caruana blending is capped at K=27 LB ≈ 0.95368, well below the
  PRIMARY LB 0.95386.

**Therefore the highest-priority next-session action is rebuilding
the 6 slim-kNN bases** (each estimated ~30-60 min CPU per
`scripts/build_K11_full_pathb.py:151-156`), pushing them to the
Kaggle private artifact dataset, then properly gating all
remaining queue picks (C1/C2/EXP-9) at K=11+1.

## Skill checklist (Round 3)

- [x] `problem-solving.md` step-1 re-entry: problem restated as
      "is the ceiling claim correct under the K=4 proxy?"
- [x] `personas.md` rotation: Senior ML Engineer (review) +
      Headroom-Math strategist; surfaced load-bearing flaw +
      ranked posture (a) Aggressive lift-seeking as EV-optimal.
- [x] `strategy-critic.md` section 5 headroom math: queue
      midpoint sum 1.40 bp; top-5% gap 1.9 bp; queue alone
      P(reach) 25-35%. Cannot reach leader at 9.0 bp.
- [x] Rule 4 escape clause: "never declare structural ceiling
      without a fresh Research-loop" — Research-loop done
      2026-05-18, multi-anchor retest done. Ceiling claim now
      defensibly stated.
- [x] Rule 21 family-falsification: C-sweep across 0.01-100
      confirms LR-log-loss-optimal verdict is not C=1.0-specific.

## Headroom math reality check

With Phase 0 confirming the row-feature ceiling at OOF 0.95430:
- The K=27 + Path-B IN THE SNAPSHOT gives OOF 0.95432 and LB 0.95368.
- The actual PRIMARY K=11+K=9 gives LB 0.95386.
- The +1.8 bp PRIMARY lift over K=27 comes from kNN diversity not
  in the snapshot.
- Top-5% boundary LB 0.95405; gap to PRIMARY -1.9 bp.

The remaining queue (C1/C2/EXP-9/A3/B2) MIGHT close the gap, but
ONLY IF they're properly gated against K=11 (not the K=4 proxy).
Without rebuilding the kNN layer, future iterations will continue
to misread their lifts.

## Files added today (round 3)

```
scripts/probe_round3_p0_proxy_audit.py     (Phase 0 audit)
scripts/probe_round3_caruana.py            (Phase 1.2 hill-climb)
scripts/artifacts/oof_caruana_blend_round3_strat.npy
scripts/artifacts/test_caruana_blend_round3_strat.npy
submissions/submission_caruana_round3.csv  (NOT submitted; LB ≈ 0.95370 expected)
audit/2026-05-18-round-3-execution.md       (this file)
```

## Friction surfaced today

- `tag: artifact-snapshot-blocks-k11-gating` — 6 slim-kNN bases
  missing from 2026-05-08 snapshot prevents K=11 reconstruction
  and proper PRIMARY-anchor gating. Rebuild + push is the
  highest-priority operational fix for the rest of comp.
- `tag: k4-k27-residual-correlation-0.998` — the two pools we
  have are essentially identical at the residual level. Proxy
  substitution is NOT the flaw; the row-feature ceiling is real.
- `tag: caruana-degenerates-without-diversity` — Caruana on the
  snapshot picks K=27_pathb 88% of the time. Confirms blend
  harness can't generate net new value without kNN diversity.

## Recommended next-session action plan

1. **Rebuild slim-kNN bases.** `scripts/build_K11_full_pathb.py`
   references 6 inputs (qAT/qAV/qAO/qAA/qAF/qAK); each is built
   by an upstream script (likely the dgp_v3 family). ~3-6 hr CPU.
2. **Push K=11+K=9 OOFs to artifact dataset** via
   `kaggle datasets version` once kNN bases are rebuilt.
3. **Re-test ALL 11 Round-2 nulls + 3 Round-3 retests at K=11+1**
   with proper anchor. Confirms or refutes the ceiling at the
   real PRIMARY level.
4. **C2 swap-noise DAE on Kaggle T4** (~2-3 hr GPU). Highest-EV
   remaining mechanism class; needs K=11 anchor for proper gate.
5. **Final-window hedge prep** in Days 10-13 regardless. Final-1
   = PRIMARY; Final-2 = K=27 + Path-B (structurally different,
   safer for private-LB shake-up scenarios).
