# 2026-05-13 — Day-13 EDA + execution postmortem

PI: "document findings learnings and frictions ... what could have been
avoided ... what context next time."

Session: 6-phase EDA + 4 hypothesis tests + 1 LB submit.
Net LB Δ: **−2 bp regress** on 1 token.  Net knowledge Δ: FM-field-
augmentation lever falsified.

## Findings (kept)

1. Cum_Deg ⊥ TyreLife within Compound (HARD ρ=-0.08, INTER ρ=-0.26) —
   independent signal, contradicts redundancy intuition.
2. `LapTime_Delta` single-feature LR has +922 bp Strat→GroupKF gap —
   largest race-specific feature.
3. Year leakage Δ is **negative** (−360 bp) — Year generalizes BETTER
   under GroupKF (Strat fold contaminated by 2023 anomaly).
4. FM compound cosine non-physical: MEDIUM↔HARD = -0.67, WET↔INTER =
   +0.59 — FM learned binary pit-class, not tyre-spectrum.
5. FM field-pair magnitude peaks at Year×Stint (0.386), not
   Driver×Compound — Year is the FM hub.
6. Family ΔAUC (Strat − GKF): GBDT-formulation 303 · CB 261 · GBDT 233
   · Rule 42 · SparseLR 42 · FM 9.  Quantifies leakage-robust ladder.
7. **FM-field-augmentation saturated at 12 fields.**  H1 had strongest
   std OOF (0.9271) and most-diverse ρ (0.909) of any FM attempt and
   still regressed −2 bp LB.  d9c/d9f/d9h/d9i amplification was specific
   to those 4 field types, not structural.

## Frictions — avoidable

1. **Stale PRIMARY-of-record.**  CLAUDE.md said 0.95034 for 6 hours;
   actual LB-best was 0.95041 (Stint Path B, posted by parallel session
   05:34 UTC).  Every EV calc anchored to wrong number.  *Fix:*
   SessionStart hook polling `kaggle competitions submissions`.

2. **H3 in-sample +4 bp without nested-CV check.**  Synthesis ranked
   +1.2 bp midpoint; nested CV showed −1.4 bp regression.  In-sample
   isotonic on a calibrated meta is a known overfit pattern.  *Fix:*
   in-sample probes never produce synthesis-EV without nested-CV
   companion in the same phase.

3. **~7 min wasted on H4/H5 GBDT bases.**  CLAUDE.md
   `mechanism_families_explored` had 30+ GBDT entries; Rule 16 explicitly
   downgrades GBDT-on-binary-target by 0.3×.  I downgraded synthesis EV
   but not time-spend.  *Fix:* Rule 16 preflight as a CLI gate before
   any new training script starts.

4. **H1 combined 3 new fields with no per-field ablation.**  CRT + Cdpl
   + Ldz in one FM_aug15 → -2 bp regression with no attribution.  *Fix:*
   one-mechanism-per-probe; 3 separate aug13 variants would have cost
   ~30 min more wall but produced 3 separable falsifications.

5. **PRIMARY OOF not committed.**  d9h saved only test pred.  Local
   reconstruction first attempt was 1.3 bp off — missed `expand()`
   raw+rank+logit transform in d9h source.  Wasted ~5 min.  *Fix:*
   every LB-submitting script commits BOTH oof and test artifacts; or
   canonical `oof_PRIMARY_current_strat.npy` updated on PRIMARY change.

6. **EDA underweighted segmented-meta candidates.**  Phase D AVP
   (cluster-2 underpredicted 6.4 pp) and Phase F (per-cohort isotonic
   structure) both pointed at segmented meta; synthesis ranked H7
   (cluster-2 specialist) last as "speculative".  Stint Path B lifted
   LB +7 bp same day.  Signal was in EDA; weighting was wrong.  *Fix:*
   segmented-meta candidates promoted to top-2 when AVP/cluster shows
   >5 pp residuals.

7. **H5 z-score leaked across folds.**  Used full-train groupby stats
   applied to fold-val rows.  *Fix:* `cv_normalize(train, fold_idx,
   group_cols, key)` helper that fits stats on outer-train rows only.

## Frictions — inherent (not avoidable)

- d9h amplification WAS a real precedent (4/4 prior FM submits lifted).
  H1 was the right test of generality.  Falsification at -2 bp is high-
  information; could not have been predicted from OOF alone.  Token
  spend is justified.
- Stint Path B +7 bp came from parallel session.  Without inter-agent
  comms there was no way to know.

## Context I'd want next time

1. **Live LB-state file** — auto-refreshed at session start with
   our_lb_best, leader, gap, tokens_used_today, last-N submissions.
2. **Mechanism family ledger** — structured table (family · n_attempts ·
   std_OOF_range · LB_Δ_range · last_explored · saturation_evidence).
   Rule-16 preflight runs in 30 s, not as free-form audit-log read.
3. **Inter-agent state file** (`WIP.md`) — what every other branch is
   working on right now.
4. **Canonical PRIMARY artifact pair** — `oof_PRIMARY_current_*.npy`,
   `test_PRIMARY_current_*.npy`, refreshed on PRIMARY change.
5. **Rule-16 preflight CLI** — `python scripts/preflight.py --family X
   --predicted_oof Y --predicted_rho Z` returns PASS / DOWNGRADE / FAIL
   with cited precedents.
6. **Ablation harness** — `python scripts/ablate.py --base PRIMARY --add
   F1,F2,F3` runs N+1 min-meta gates with tabulated diff.
7. **In-sample / OOF / nested-CV lint** — warn when same array is fit
   and evaluated for any calibration / transformation.
8. **Hypothesis-register format with `if FALSIFIED →` plan** baked in,
   not just EV midpoint.
9. **Canonical `scripts/load_data.py`** returning train/test/y with
   consistent encoding.  Saves 5 min per script.
10. **Daily diff summary** at session start — new OOFs, new submissions,
    new precedents.  Replaces "read 14 audit files to catch up".

## Carries forward + Day-14 next moves

- Files: `audit/2026-05-13-eda-deep-dive-synthesis.md` (Phase A-F facts);
  `audit/2026-05-13-eda-execution-results.md` (H1-H5 result + FM
  precedent table); `audit/2026-05-13-H3-results.md` (nested-CV
  falsification); this file.
- Append to `mechanism_families_explored`: **factorization_machine_aug15**
  (H1 regress -2 bp; lever saturated).
- Day-14:
  1. Refresh CLAUDE.md current_state to 0.95041 PRIMARY (Stint Path B);
     gap_to_top_5pct = 24 bp.
  2. TabPFN-2.5 GPU kernel (Day-12 prep) → push, run.  Only "+10 bp
     shot" remaining.
  3. Audit G3 flip-ratio gate: Stint Path B lifted LB +7 bp despite G3
     FAIL.  Calibrate or remove G3 for per-segment LR-meta.
  4. Extend Stint Path B to other cohort splits (Compound, Year, Race)
     — EDA already has the cohort target-rate tables to seed it.
