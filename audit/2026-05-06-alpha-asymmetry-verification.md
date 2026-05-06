# 2026-05-06 — hier-meta α asymmetry verification + harness

PI redirect: experimentation culture; small probes; BOTE first.
This note: (1) verifies an audit-agent claim about a hier-meta bug;
(2) introduces `scripts/probe.py` as the harness for the next 30+
cheap probes.

## Verified: per-fold vs full-train α asymmetry is real, but it's
##           Bayesian-correct behavior — NOT a free +2-5bp fix

Claim under test (round-1 audit agent): per-fold OOF computation in
`d13_path_b_hier_meta.py:149-150` and `d13e_path_b_compound_stint.py:152`
uses fold-train counts to compute `α = n / (n + τ)`, while the full-train
test path uses full-train counts. Same segment → different α →
"OOF and test live in different effective spaces"; agent claimed
"+2 to +5bp on PRIMARY if fixed".

### Code reads
- `d13_path_b_hier_meta.py:146-159` (per-fold): `counts = bincount(seg_train[tr])` ⇒ fold-train counts drive α.
- `d13_path_b_hier_meta.py:162-171` (full-train): `counts_full = bincount(seg_train)` ⇒ full-train counts drive α.
- `d13e_path_b_compound_stint.py:144-167` and `:174-197`: same pattern.

### Verdict: structural claim correct, severity overstated
The asymmetry exists and is consistent with Bayesian shrinkage:
- Fold-train fit on ~80% of data → local LR is genuinely noisier
  → more shrinkage to global is appropriate.
- Full-train fit on 100% of data → local LR is more reliable
  → less shrinkage to global is appropriate.

Concrete example, τ=20000, segment full-n=10000:
- Full-train α = 10000 / (10000 + 20000) = 0.333
- Fold-train (n≈8000) α = 8000 / (8000 + 20000) = 0.286

We cannot "fix" this and ship a stronger test-time model — the
test-time model is what it is.

### What the asymmetry *does* mean (the real implication)
**OOF AUC ≠ test-time-model AUC at the same τ.** OOF reports a
slightly more-shrunk model than what we ship. Two consequences:

1. The 11.6× OOF→LB amplification observed for d13 Stint Path B
   may be **partially explained** by this miscalibration — OOF
   underestimates the test-time model's strength when local LR
   benefits from full-train data.
2. The τ that maximizes OOF AUC is **not necessarily** the τ that
   maximizes test-time AUC. We may have selected sub-optimal τ
   for the current PRIMARY.

### Testable hypothesis derived from the finding (the actual probe)
Recompute OOF with `α_oof = counts_full / (counts_full + τ)` —
i.e., apply the test-time shrinkage to the OOF prediction loop,
while keeping `W_local` fit on fold-train. This produces an OOF
estimate calibrated to the test-time α. Re-run the τ sweep on
this calibrated OOF — the optimum may sit at a smaller τ.

BOTE (run via the new harness):
```
$ python3 scripts/probe.py bote calibrated_alpha_tau_resweep \
    --family code_fix_calibration --cost_min 30 --std_oof_lift_bp 1.0
verdict: PURSUE  (cost_efficiency ≈ 0.010 bp/min)
```

## Harness: `scripts/probe.py`

Two callables that embed the BOTE-first / gate-after experimentation
loop into code:

- `bote(name, family, cost_min, …)` — returns expected_lb_bp +
  verdict (PURSUE / DEFER / SKIP) using empirical family priors
  calibrated against the 17% advance hit rate.
- `gate(name, oof_path, test_path, …)` — returns standalone OOF Δ
  (when y available), ρ vs PRIMARY, predicted LB Δ, G3 flip ratio,
  and a verdict band. Degrades to ρ-only mode when `data/train.csv`
  is missing.

CLI examples:
```
python3 scripts/probe.py bote h5_isolated \
    --family per_field_fm_isolated --cost_min 60 --std_oof_lift_bp 2.0

python3 scripts/probe.py gate d13e_tau100000_vs_primary \
    --oof scripts/artifacts/oof_d13e_compound_stint_tau100000_strat.npy \
    --test scripts/artifacts/test_d13e_compound_stint_tau100000_strat.npy
```

Smoke-test results (this session):
- `gate self_vs_self` on PRIMARY artifacts: ρ=1.000000, Δ=0bp,
  flips 0/0 → TIE_EXPECTED ✓
- `gate d13e_tau100000_vs_primary`: ρ=0.999080 (just above tie band),
  flips 42/132 (174 total) → TIE_EXPECTED ✓ (matches expectation
  that the held τ=100000 ties PRIMARY within Kaggle quantization)

Family priors initial calibration (`FAMILY_PRIORS` dict in `probe.py`):
- `new_model_class` 0.40 / (3, 8, 15) bp
- `meta_arch_redesign` 0.30 / (1, 4, 8) bp
- `per_field_fm_isolated` 0.25 / (0.5, 2, 5) bp
- `code_fix_calibration` 0.30 / (0.3, 1, 3) bp
- `single_base_fe_addition` 0.05 / (0, 0.5, 2) bp  ← 4-of-4 NULL prior

Cost-efficiency thresholds (current calibration; update as priors
change): ≥0.05 bp/min → PURSUE, ≥0.01 → DEFER, else SKIP.

## What this enables
Going forward each new candidate gets two cheap touches:
1. `bote NAME --family X --cost_min Y` before any code (rule out the
   hopeless ones in 5 seconds).
2. `gate NAME --oof … --test …` after the artifacts exist (uniform
   summary; no script-by-script bespoke gate logic).
