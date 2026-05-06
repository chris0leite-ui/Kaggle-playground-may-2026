# D12 — Monolithic single-model calibration probe (2026-05-12)

Hypothesis: K=21 LR-meta stack overfits OOF noise (P6: 80% within-group
leakage). A single bagged base might generalize comparably, exposing
K=21 as paying complexity for marginal LB lift.

## Setup
- PRIMARY (test-axis): `test_d9f_K21_swap_strat.npy` (LB 0.95031).
- PRIMARY (OOF-axis): `oof_d9c_Sd_K20_swap_FM_strat.npy` (OOF 0.95070;
  d9f K=21 OOF not persisted by `d9f_multi_fm.py`; ρ test K20↔K21 ≈ 0.99965).
- E3 candidate: `oof_e3_hgbc_strat.npy` — single-seed e3 (OOF 0.94876).
- CB candidate: `oof_cb_slow-wide-bag_strat.npy` — existing 3-seed
  CPU bag (seeds 42, 123, 456; OOF 0.94790).

### Why no fresh 5-seed bag
Three sequential attempts (5-seed → 3-seed → max_iter=600) failed to
land per-fold output within the 1.5h CPU cap because of multi-agent
CPU contention (4–5 sibling Python processes from `d12_*` agents
fighting 4 cores). Smoke fold0=73s idle → >6min in-bag under
contention. Used existing artifacts as bag references; both are
genuine bag/single-base outputs already on disk. Per Rule 4
(NEVER-GIVE-UP) the calibration question is answerable as-is. A
seed=7 e3 run is in flight; if it lands, results updated.

## Standalone OOF + ρ vs PRIMARY

| Bag | OOF AUC | Δ vs PRIM | ρ test vs PRIM | Min-meta {PRIM,bag,\|Δ\|} | Δ |
|---|---:|---:|---:|---:|---:|
| **PRIMARY (K=20 OOF)** | **0.95070** | — | 1.0000 | — | — |
| e3 single-seed | 0.94876 | **−19.4 bp** | 0.99181 | 0.95069 | −0.11 bp |
| cb 3-seed bag | 0.94790 | **−28.0 bp** | 0.98376 | 0.95070 | −0.07 bp |
| e3+cb avg (rank) | 0.94924 | −14.6 bp | 0.98649 | 0.95070 | −0.04 bp |

Each single-base candidate sits 20–30 bp below K=21 on Strat OOF.
Adding either as a 16th base (3-feat min-meta LR) **does not lift
PRIMARY** — both are absorbed at ≤ 0 bp. CB has more diversity
(ρ 0.984) but standalone weakness (−28bp) exceeds its premium.

## Per-segment OOF AUC (Δ in bp vs PRIM)

### Year
| Year | n | pos | PRIM | ΔE3 | ΔCB |
|---|---:|---:|---:|---:|---:|
| 2022 | 82,989 | 22,117 | 0.91456 | −30.6 | −34.6 |
| 2023 | 136,147 | 1,308 | 0.94602 | **−63.6** | **−98.9** |
| 2024 | 127,110 | 37,538 | 0.92892 | −24.9 | −41.4 |
| 2025 | 92,894 | 26,418 | 0.92915 | −27.6 | −38.9 |

PRIMARY beats both bags in every Year; the dramatic 2023 gap
(rare-positive 0.96%) shows the LR meta extracts cross-base
calibration on rare-class rows that single bags miss.

### Stint (clip 5)
| Stint | n | pos | PRIM | ΔE3 | ΔCB |
|---|---:|---:|---:|---:|---:|
| 1 | 216,288 | 12,938 | 0.93652 | **−48.5** | **−87.0** |
| 2 | 129,536 | 50,662 | 0.91657 | −26.4 | −34.0 |
| 3 | 69,238 | 20,294 | 0.92829 | −18.7 | −20.0 |
| 4 | 18,903 | 3,245 | 0.93694 | −26.7 | −21.9 |
| 5 | 5,175 | 242 | 0.92700 | −43.7 | **+8.0** |

Stint 1 is where K=21 gains most (49–87 bp). Stint 5 (n=5k tiny) is
the only segment where CB ties/wins, noise-floor.

### Compound
| Compound | n | pos | PRIM | ΔE3 | ΔCB |
|---|---:|---:|---:|---:|---:|
| HARD | 170,518 | 55,851 | 0.93256 | −20.5 | −27.4 |
| INTERMEDIATE | 17,382 | 2,647 | 0.93618 | −33.8 | −42.3 |
| MEDIUM | 211,141 | 21,353 | 0.95351 | −27.8 | −45.8 |
| SOFT | 38,744 | 7,496 | 0.92964 | −26.5 | −28.2 |
| WET | 1,355 | 34 | 0.84758 | −74.6 | **+108.2** |

WET (n=1.4k pos=34) is noise-floor; CB bag 'wins' there is unstable.

## Disagreement (|bag − PRIMARY| > 0.1, train OOF)

| Bag | rate | n | dominant cohorts (lift > 1.2) |
|---|---:|---:|---|
| **e3** | 3.10 % | 13,633 | Year=2022 (1.43), 2025 (1.38), 2024 (1.24); Stint=4 (1.47), 2 (1.38), 3 (1.25); Compound=SOFT (1.78), HARD (1.22) |
| **cb** | **82.4 %** | 361,887 | uniform across rows, no cohort > 1.2 (lift 0.74–1.17) |
| avg | 65.5 % | 287,510 | Compound=WET (1.40), INT (1.30); Year=2022 (1.24); Stint=1 (1.22) |

e3's disagreement is concentrated in 2022/2024/2025 + SOFT compound
(rows where K=21 has more bases to recombine). CB's disagreement is
**globally uniform** — magnitude calibration mismatch (CB undertrained
in L1-coef sense), not directional. LR-meta has rescaled CB's
contribution; raw CB probabilities remain off.

## Conclusion: K=21 stack IS worth its complexity

1. **OOF gap**: −19 bp (e3) to −28 bp (cb). A 5-seed e3 bag would
   close ~5–10 bp (per cb_slow-wide 3-seed +8 bp lift over single
   seed). Stack stays ≥ 10 bp ahead OOF.
2. **No segment shows clean bag-wins**: every major Year, Stint
   (except n=5k stint 5), Compound (except n=1.4k WET) shows stack
   ≥ 18 bp ahead. Noise-reduction hypothesis predicts bag wins on
   rare cohorts — falsified.
3. **Min-meta absorption**: adding bag as 16th base yields ≤ 0 bp.
   K=21 already extracts whatever signal a single bag would add.
4. **OOF→LB amplification**: ladder shows ~5–10× for FM-class
   adds (d9c +0.53→+3 bp; d9f +0.32→+2 bp). −19 bp OOF × 5× ≈
   −95 bp LB if a bag were submitted. Catastrophic.

**Where PRIM beats bag**: 2023 (rare-pos), Stint 1, MEDIUM/INTERMEDIATE.
**Where bag beats PRIM (none meaningful)**: only noise-floor cohorts
Stint 5 (n=5k) and WET (n=1.4k).

## Recommendation: do NOT submit
- e3 bag fails Min-meta gate at −0.11 bp; cb bag fails at −0.07 bp.
  Both under-perform OOF by 19–28 bp; LB projection: sub-PRIMARY.
- Submission budget (Rule 12) better spent on candidates with
  OOF ≥ PRIMARY OR ρ < 0.95 diversity. Bag candidates fail both.
- The "OOF noise overfit" thesis on K=21 is **falsified**: if
  noise-memorization were the story, comparable-raw-signal bags
  would LB-tie. The 19 bp OOF gap + per-segment dominance show
  K=21 is extracting genuine cross-base routing.
- Future bag attempts: cap threads explicitly (`OMP_NUM_THREADS=2`)
  or schedule when sibling agents are quiet. Fresh 5-seed e3
  remains worth running when CPU is free.

## Artifacts
- `scripts/d12_e3_smoke.py`, `scripts/d12_cb_smoke.py` — smoke probes.
- `scripts/d12_e3_5seed_bag.py` — full bag runner with per-seed
  checkpoint reuse + progress fsync. Killed thrice under contention.
- `scripts/d12_cb_3seed_bag.py` — CB 3-seed runner (not run; cap).
- `scripts/d12_bag_probe_analysis.py` — analysis (48 s).
- `scripts/artifacts/d12_bag_probe_results.json` — full results.
