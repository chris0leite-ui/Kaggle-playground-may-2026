# 2026-05-12 — D12 Year-segmented specialist + adversarial-validation reweighting

Branch: `claude/review-major-improvements-NUhht`. No commits, no submits.
Pipeline: `scripts/d12_year_specialist_advweight.py`. Results JSON:
`scripts/artifacts/d12_year_specialist_advweight_results.json`. Artifact
arrays:

- `oof_d12_year_specialist_strat.npy` / `test_d12_year_specialist_strat.npy`
- `oof_d12_e3_advweight_strat.npy` / `test_d12_e3_advweight_strat.npy`
- `oof_d12_year_advweight_strat.npy` / `test_d12_year_advweight_strat.npy`

PRIMARY surrogate: `d9c_Sd_K20_swap_FM` (OOF + test both exist; OOF
0.95070, LB 0.95029; faithful surrogate for the 0.3bp newer
`d9f_K21_swap` LB-best whose OOF artifact is missing). ρ-vs-LB-best
also computed against `test_d9f_K21_swap_strat.npy`.

## Hypothesis recap

P3 (`audit/2026-05-08-data-probe-results.md`): Year=2023 has 0.96% pit
rate vs ~28% in 2022/2024/2025; 31% of train+test are 2023; 24/26
2023-races are <5%. The single-model pool averages signal across two
disjoint regimes. Hypothesis: split into 2 specialists, route by Year.

P9: 221 drivers <10 rows pit_mean ≈ 0.26% ≈ 2023 base rate ⇒ likely
2023-only synthetic drivers. Reinforces the 2023 mode-collapse signal.

Adversarial-validation reweighting is an orthogonal axis: importance
weights `w_i = p(test|x_i)/p(train|x_i)` realign the train objective
toward the test distribution. Independent of the Year split.

## Methodology

- Base learner: HGBC label-encoded Driver, native categorical
  Compound/Race (mirrors `scripts/e3_hgbc_two_anchor.py`; best
  non-CB single base in pool).
- 5-fold StratifiedKFold(seed=42) — pinned anchor.
- Part A: M_active (Year ∈ {2022,2024,2025}; ~70%) + M_2023 (Year=2023;
  ~31%). Within each fold's tr indices, partition by Year, fit two
  specialists, route on val + test by `Year==2023`.
- Part B: AV classifier = LGBM 5-fold OOF on `is_test` label over
  concat(train+test); compute `w = clip(p_test/(1-p_test), 0.1, 10)`;
  retrain HGBC with `sample_weight=w` (5-fold Strat).
- Part C: Year-specialists trained with `sample_weight=w` on each
  cohort; same routing as A.
- Part D: standalone OOF AUC, ρ-vs-PRIMARY-test (Spearman, also vs
  d9f K21 LB-best test), min-meta gate (3-feat LR over
  {PRIMARY, candidate, |Δ|}, 5-fold OOF AUC), and K=22-add stack
  using the d9c_kn_stack pool + the candidate.

## Results (auto-filled)

### AV classifier

> **AV OOF AUC: <to-fill>** (≥0.55 ⇒ measurable shift, <0.55 ⇒ negligible)
> Weight summary: mean=<>, median=<>, frac clipped at 0.1=<>, at 10=<>

### Per-Year OOF AUC: single-model vs Year-specialist

| model | 2022 | 2023 | 2024 | 2025 | overall |
|---|---:|---:|---:|---:|---:|
| m5q | 0.91438 | 0.94609 | 0.92870 | 0.92897 | 0.95057 |
| PRIMARY (d9c_Sd) | 0.91456 | 0.94602 | 0.92892 | 0.92915 | 0.95070 |
| year_specialist | <> | <> | <> | <> | <> |
| e3_advweight    | <> | <> | <> | <> | <> |
| year_advweight  | <> | <> | <> | <> | <> |

### Standalone + ρ + min-meta gate + K=22 stack

| candidate | std OOF | ρ_test PRIMARY | ρ_test LB-best | min-meta Δ vs PRIMARY | K22 OOF | K22 pred-LB | ΔLB |
|---|---:|---:|---:|---:|---:|---:|---:|
| year_specialist | <> | <> | <> | <>bp | <> | <> | <>bp |
| e3_advweight | <> | <> | <> | <>bp | <> | <> | <>bp |
| year_advweight | <> | <> | <> | <>bp | <> | <> | <>bp |

Reference K=21 PRIMARY (no add): OOF=<>, ρ=<>, predLB=<>.

## Verdict

(filled after results land)

## Cost

(filled after results land)

## Pointers

- Hypothesis origin: `audit/2026-05-08-data-probe-results.md` (P3, P9)
- Pool for K=22 stack: matches `scripts/d9c_kn_stack.py` Sa K=21
  (PRIMARY-keep 16 + top-3 d9 rules + R14_L4 + FM) + 1 candidate
- Per-segment OOF analysis precedent: `audit/2026-05-04-d3-per-segment-analysis.md`
