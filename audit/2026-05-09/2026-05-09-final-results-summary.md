# 2026-05-09 — Final results summary: qAA → qAQ autonomous loop

`branch: claude/analyze-synthetic-data-generation-BtmFl`
`session window: 2026-05-09 ~12:00 → ~13:30 UTC`

## Headline number

**Best K=4+1+ combo at OOF: 0.95415 = +1.262 bp vs PRIMARY 0.95403.**
K=7 = K=4 + qAK + qAO + qAF + Path-B Compound × Stint τ=20,000.

PRIMARY LB: 0.95351. If LB transfer is 1×, expected LB ~0.95363. If
V4-anomaly 5×, expected LB ~0.95414 (at top-5% boundary 0.95405).

## Probe ladder

17 probes (qAA → qAQ) executed in autonomous loop. Results:

### Single-base K=4+1 LR-meta lifts (sorted by Δ bp)

| Probe | Standalone OOF | ρ_test | K=4+1 lift bp | Mechanism |
|---|---:|---:|---:|---|
| **qAO** | 0.88997 | 0.794 | **+0.730** | multi-K (3+5+10) kNN, 6-axis cell, hierarchical fallback |
| **qAK** | 0.87320 | 0.755 | **+0.717** | tight K=3 kNN, 6-axis cell, hierarchical fallback |
| qAF | 0.90362 | 0.617 | +0.149 | d16++ trained on orig with stint_imputed |
| qAA | 0.94495 | 0.965 | +0.143 | stint_imputed Frontiers F1 sequence features |
| qAJ | 0.94475 | 0.963 | +0.139 | qAA + orig-driver-rate features |
| d18g | 0.94877 | 0.980 | +0.028 | BGMM mode-id (existing artifact) |
| qAH | 0.54597 | 0.196 | +0.028 | orig-driver-rate only |
| qAB | 0.92059 | 0.899 | +0.017 | orig hier-TE + K=20 kNN coarse |
| qAC | 0.94605 | 0.961 | -0.005 | qAA+qAB joint base |
| qAI | 0.28046 | -0.278 | +0.077 | error-weighted (broken; inverted) |

### Pair / triple combos at K=4+N LR-meta

| Combo | K | OOF | Δ bp |
|---|---:|---:|---:|
| qAK + qAO + qAF | 7 | 0.95410 | **+1.015** (best 3-add LR-meta) |
| qAK + qAA + qAF | 7 | 0.95409 | +0.983 |
| qAK + qAA + d18g | 7 | 0.95409 | +0.942 |
| qAK + qAO + qAB | 7 | 0.95409 | +0.966 |
| qAK + qAA + qAB | 7 | 0.95408 | +0.849 |
| qAK + qAO | 6 | 0.95408 | +0.851 |
| qAK + qAF | 6 | 0.95408 | +0.867 |
| qAO + qAF | 6 | 0.95409 | +0.927 |
| K=12 ALL extras | 12 | 0.95410 | +1.066 |

### Path-B C×S τ-sweep on K=7 = K=4 + qAK + qAO + qAF (BEST)

| τ | OOF | Δ_oof bp | ρ_test | flips_pos/neg | flip_ratio |
|---:|---:|---:|---:|---|---:|
| 5,000 | 0.95414 | +1.122 | 0.99834 | 497/652 | 0.762 |
| **20,000** | **0.95415** | **+1.262** | **0.99897** | **470/613** | **0.767** |
| 100,000 | 0.95415 | +1.169 | 0.99923 | 464/584 | 0.795 |

### Path-B C×S τ-sweep on K=7 = K=4 + qAK + qAA + qAF (alternative)

| τ | OOF | Δ_oof bp | ρ_test | flips_pos/neg | flip_ratio |
|---:|---:|---:|---:|---|---:|
| 5,000 | 0.95414 | +1.091 | 0.99845 | 482/661 | 0.729 |
| **20,000** | **0.95415** | **+1.215** | **0.99904** | **457/629** | **0.727** |
| 100,000 | 0.95414 | +1.104 | 0.99923 | 448/605 | 0.740 |

## The breakthrough mechanism: qAK / qAO

Both qAK and qAO compute orig-kNN at the 6-axis cell key
(Year, Compound, PitStop, Race, Stint, LapNumber) with hierarchical
fallback (L6→L5→L4→L3). qAK uses K=3; qAO multi-K (3+5+10).

For each test row r, we find the K nearest orig rows that share r's
6-axis cell, in standardised continuous space (LapTime, Δ,
Cumulative_Degradation, RaceProgress). The output features are the
distance-weighted PitNextLap mean, std, max, min, median distance, and
level_used indicator.

**Coverage**: 79% of train rows fit at L6 in qAK; 18% at L5; 2% at L4;
1.4% at L3. The hierarchical fallback ensures every row gets a
prediction.

## Why qAK/qAO escape rank-lock where qAB didn't

qAB used K=20 within the (Y, C, PS) coarse cell — a smooth aggregate
that K=4's bases already absorb via Driver/Compound interactions.
qAK/qAO use tight K within the FULL 6-axis cell — captures per-row
identity match against orig rather than per-cell mean.

The 6-axis cell key is exactly the cond-vector schema we decoded for
the host's DGP (per the prior session's qH-qM analysis). Synth rows
are sampled from per-cell continuous densities. For the 79% of rows
where the 6-axis cell has ≥3 orig rows, the K=3 kNN essentially
identifies orig rows at the same race position (same Race, Stint,
LapNumber) and similar continuous dynamics.

This is structurally a **per-row attribution signal** — the analog of
the MIDST 2025 winner's "loss-features-across-noises" pattern, applied
to tabular synthesis. The LR-meta cannot reconstruct this from K=4's
predictions because K=4's bases see the row's continuous tuple but
don't have access to orig labels at the same cell position.

## Submission candidates ready (NOT submitted; awaits PI approval per Rule 1)

`submissions/`:
- submission_K7_qAK_qAA_qAF_pathb_cs_tau20000.csv (qAN, OOF +1.215 bp)
- **submission_K7_qAK_qAO_qAF_pathb_cs_tau20000.csv (qAQ, OOF +1.262 bp; best)**
- submission_K7_qAK_qAA_qAF_pathb_cs_tau{5k,100k,500k}.csv
- submission_K7_qAK_qAO_qAF_pathb_cs_tau{5k,100k}.csv

ρ_test 0.998-0.999 — exceeds Rule 27's 0.999 abort threshold.
Per friction `rule-27-abort-threshold-empirically-too-strict-for-sub-bp-moves`,
override is appropriate when OOF lift is >+0.5 bp; here lift is
+1.21-1.26 bp.

## Implications for the rank-lock framing

The friction `rank-lock-at-conditional-target-correlation-not-just-logit-direction`
(2026-05-09 night, ml-model-experiments-gbKiI) said features must
introduce new partial correlation with y given K=4 to escape rank-lock
at the meta. qAK's success refines this:

**Updated framing**: rank-lock at LR-meta is broken by features that
match each test row to specific orig rows via cell-conditional kNN,
where the cell granularity matches the host's cond-vector schema. The
"loss-features-across-noises" / MIDST pattern is the formal analog. The
key levers are:
- **Cell granularity**: must match the host's DGP cond-vector
  (6 axes for s6e5)
- **Tight K**: preserves per-row variance vs smooth aggregate
- **Variance features**: std/min/max across NN exploit per-row
  agreement
- **Hierarchical fallback**: ensures coverage without losing tightness
  where it matters

This generalizes V4's lesson: tree-internal exploitation of kNN-derived
features beats both meta-level addition and dense aggregation.

## What didn't work (consolidated)

| Mechanism | K=4+1 result | Lesson |
|---|---|---|
| stint_imputed sequence (qAA) | +0.143 | absorbed; row-context features within K=4's reach |
| orig hier-TE smooth (qAB) | +0.017 | smooth aggregate absorbs |
| stint+orig-cell joint base (qAC) | -0.005 | base-level joint feature absorption |
| d16++ on orig+stint (qAF) | +0.149 | structurally orthogonal but small |
| orig-driver-rate (qAH) | +0.028 | distribution shift kills it |
| error-weighted (qAI) | inverted | up-weighting K=4 errors flips signs |
| stint+orig-driver joint (qAJ) | +0.139 | qAA-equivalent |
| BGMM mode-id (d18g) | +0.028 | absorbed at K=4 |

## Commits

- 2 commits this session: stint_imputed sprints (b0eda05) and
  qAK breakthrough sprints (b533054); next commit is the qAO/qAP/qAQ
  + this synthesis.

## Pointers

- `audit/2026-05-09/2026-05-09-qAK-breakthrough.md` — the breakthrough doc
- `scripts/dgp_v3/qA{A..Q}.py` — 17 probe scripts
- `scripts/artifacts/dgp_v3_qA{A..Q}_*.{npy,json}` — artifacts
- `submissions/submission_K7_qAK_qA{A,O}_qAF_pathb_cs_tau*.csv` —
  LB candidates
