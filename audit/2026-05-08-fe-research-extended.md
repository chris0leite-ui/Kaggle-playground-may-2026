# FE research extended — three-agent synthesis (2026-05-08 PM late)

> Branch: `claude/research-feature-engineering-7oCmj`. Companion to
> `audit/2026-05-08-fe-research-survey.md` and
> `audit/2026-05-08-fe-research-code-grounded.md`. PI directive
> 2026-05-08 PM: "add these as our options, research further, similar
> competitions and write ups."
>
> Three parallel research agents covered (i) older TPS / classical
> binary-AUC writeups, (ii) sequential-events FE in adjacent domains,
> (iii) F1 / motorsport practitioner sources. This note distills their
> findings into corrections + new candidates.

## Headline

Three corrections to existing A2 picks; eight new picks added as
Tier-A3 in `EXPERIMENTS-NEXT.md` (EXP-A3-1 through EXP-A3-8).

The most consequential single finding: **our F7 fuel-correction
constant of 0.035 s/lap (from svanikkolli v12) is mass-confounded.**
F1 cars burn 1.5-1.8 kg/lap; published coefficients are 0.030 s/kg
× 1.7 kg/lap ≈ **0.051 s/lap** (post-2019) and **0.090 s/lap**
(pre-2014). Per-track learned slope strictly dominates the constant.
Corrected in EXP-A3-2.

## Corrections to A2 picks

### A2-6 fuel correction — coefficient is wrong

`svanikkolli v12 §F7` uses 0.035 s/lap. Open-source consensus
(f1metrics, Motorsport Metrics) gives 0.030 s/kg × 1.7 kg/lap ≈
0.051 s/lap post-2019 and 0.090 s/lap pre-2014. **Best practice is
to learn the slope per-track per-fold** by regressing mid-stint clean
laps' `LapTime` on `LapNumber`. Replaces A2-6 with EXP-A3-2.

### A2-2 mandatory 2-compound rule — needs dry-race condition

The FIA regulation only applies in dry races. svanikkolli v12's
implementation tests `(n_compounds_used < 2) & (RaceProgress > 0.45)`
without a dry-race gate. The TUM RL `vse_supervised.py` formulation
adds:

```python
df['dry_race'] = 1 - df.groupby('Race')['Compound'].transform(
    lambda s: s.isin(['INTERMEDIATE', 'WET']).any()).astype(int)
df['rule_binding'] = (df['dry_race'] & (df['n_compounds_used'] < 2)
                      & (df['RaceProgress'] > 0.6)).astype(int)
```

Apply the dry-race gate when implementing A2-2.

### A2-1 gap-to-car features — refine to rank-sorted multi-neighbour

NFL Big Data Bowl 2020 1st place "The Zoo" (Singer & Gordeev) showed
that **rank-sort by distance** is the canonical pattern for
permutation-invariant cross-actor-state. Our 887-Driver categorical
is exactly the setting that benefits. Extend A2-1's `gap_to_car_ahead`
+ `gap_to_car_behind` to **`gap_ahead_1`, `gap_ahead_2`,
`gap_behind_1`, `gap_behind_2`** plus the BDB 2023 U Toronto
**pit-pressure scalar** `Σ exp(-gap/τ) · I(rival_TyreLife < 5)`.
EXP-A3-1 supersedes A2-1.

## New patterns added as Tier-A3

| EXP | Source | Pick | Cost | Novelty |
|---|---|---|---:|---|
| A3-1 | NFL BDB 2020+2023 | Rank-sorted multi-neighbour gaps + pit-pressure | 18 min | HIGH |
| A3-2 | f1metrics + Motorsport Metrics | Per-track learned fuel coefficient | 6 min + per-fold | MEDIUM-HIGH (corrects A2-6) |
| A3-3 | TUMFTM RL state + Frontiers AI 2025 | `tirechange_pursuer` + DriverAheadPit lagged 1-3 laps | 10 min | HIGH |
| A3-4 | TUMFTM `calc_tire_degradation.py` | Heilmeier compound-conditional log-curve residual | 8 min + per-fold | HIGH |
| A3-5 | Soccer LEM 2024 + NFL BDB 2025 | Multi-task aux head: BCE + α·Huber(laps_until_pit) | 25 min | HIGH (non-feature) |
| A3-6 | Home Credit 1st + Otto 1st | KNN-target-mean-of-500 on top continuous features | 20 min | MEDIUM-HIGH |
| A3-7 | IEEE-CIS Fraud 1st | UID post-hoc smoothing | 2 min (post-process) | MEDIUM (cheap probe) |
| A3-8 | TPS-S5E2 1st (Deotte) | Quantile + histogram groupby aggregations | 15 min | MEDIUM |

## Deferred candidate menu (lower-EV, not promoted to A3)

These came up in the research but are either redundant with existing
picks or speculative. Listed for reference if A2/A3 picks all NULL
and we need a deeper bench.

- **VSC vs Full-SC explicit threshold buckets** (F1 sources: VSC ratio
  0.55-0.75, Full-SC <0.55). Refines A2-4 if implemented; otherwise
  A2-4's svanikkolli thresholds (1.08/1.30) work.
- **`cold_tyre = (lap_in_stint ≤ 2)` × `LapTime_Delta`** (TUMFTM
  `tireset.py`'s `t_add_coldtires` constant). Cold-tyre penalty proxy.
- **`stint_progress = TyreLife / quantile(TyreLife | Race, Compound,
  0.95)`** (TUMFTM track-aware). Per-track pit-window normaliser.
  Subsumed by EXP-A3-4 if Heilmeier residual fires.
- **Driver × Compound deviation features** (TUMFTM): `dvr_cmp_lt_med`,
  `dvr_cmp_deg_slope`. Captures Hamilton-vs-Verstappen style. Must
  refit per fold (Rule 24).
- **Right-censored `expected_stops`** (BMC 2022 hazard): drop rows
  after last_pit_lap with no future pit from negative-class denominator.
  Refines A2-5.
- **Time-dummy interactions**: `TyreLife × LapNumber`,
  `Compound × RaceProgress` explicitly precomputed. Hazard-FE pattern
  GBDT can split on but only if given the product.
- **Multi-precision rounding ladder** (Deotte S5E2): `round(col, k)`
  for k ∈ {7, 8, 9, 10}. Dual to digit extraction.
- **Original-dataset MSRP-anchor merge as feature column**, not row
  injection (Deotte S5E2). Different from our yekenot pseudo-row merge.
- **Otto's KNN-k ladder**: KNN at k ∈ {2, 4, 8, ..., 1024} as 10
  separate stacking inputs. Subsumed by A3-6.
- **Number-of-zeros per row** (Otto): count-of-zero indicator.
- **t-SNE-3D + KMeans cluster IDs** (Otto): cheap diversity feature.
- **5x duplication + ordered-TE inside CatBoost** (yunsuxiaozi 8th
  S6E4): reduces TE variance. Specific to CatBoost class.
- **Forward feature-selection via Ridge** over big feature pool
  (Home Credit 1st, Deotte S5E4): selects 75 winners from 10000 in
  fast Ridge. Process discipline, not a feature.
- **t-test feature selection** on continuous (MoA 1st).
- **Per-track pit-loss table** from TUMFTM TOML files (`pars_<Track>_<Year>.toml`):
  Singapore ≈ 28 s, Monaco ≈ 21 s, Bahrain ≈ 22 s, Monza ≈ 18-20 s.
  Refines A2-1's hard-coded `PIT_DELTA = 22.0`.

## Honest scope of what was actually read

### Successfully read end-to-end

- TUMFTM `vse_supervised.py` (the RL agent's Tire-Change head feature
  list) and `calc_tire_degradation.py` (Heilmeier's parametric forms).
- NFL Big Data Bowl 1st-place writeups: 2020 (The Zoo), 2023 (U
  Toronto), 2024 (Chang/Dai/Jiang), 2025 (NYU/SumerSports).
- NVIDIA developer blog posts (Deotte): cuDF FE (S5E2), cuML stacking
  (S5E4), 7 Battle-tested techniques, GenAI-assisted coding (S6E3).
- Otto 1st place writeup (Titericz/Semenov, 2015) — full text.
- Home Credit 2nd place (Onodera) GitHub; 1st place via Medium
  digests + Kaggle URL.
- Soccer LEM (Mendes-Neves et al., 2024) abstract + arXiv preprint.
- BlamerX/Kaggle-Playground-Predection-Competition agent logs for
  S6E1, S6E2, S6E3, S6E4.
- F1 fuel-coefficient blogs: f1metrics, Motorsport Metrics, umakschually
  Medium.
- F1 chronicle / F1 briefing / GPFans: undercut formulas, VSC/SC
  ratios, pit-loss numbers.

### Could not access (JS-rendered / reCAPTCHA-walled)

- Kaggle 1st-place writeup HTML for: S4E1, S4E2, S4E3, S4E5, S4E6,
  S4E7, S4E8, S4E9, S4E10, S4E11, S4E12, S5E1, S5E3, S5E5, S5E6,
  S5E7, S5E8, S5E9, S5E11, MoA detailed FE, cat-in-the-dat I+II.
- Pre-2023 monthly TPS (S3, jan-dec 2021/2022) — not in BlamerX
  repo, no agent logs found.
- Per-track pit-loss TOML files in TUMFTM (directory listing
  confirmed — Sakhir, Monaco, Monza, Singapore, Sochi etc. — but
  individual TOML URLs returned 404 to WebFetch).

### Confidence anchor

The recurrence of patterns across **three independent sources**
(NVIDIA blogs, BlamerX agent logs, GitHub mirrors) for the most
consequential picks (bigram TE, digit features, pseudo-labeling,
hill-climb stacking, KNN-target-mean) makes those near-certain
even where we couldn't read the literal Kaggle HTML.

## Suggested execution order if PI authorises probes

Group by EV-per-minute and fold-safety:

**Phase 1 — fold-safe, cheap, refines existing scope** (~25 min total)
1. EXP-A3-7 UID post-hoc smoothing (2 min, pure post-process)
2. EXP-A2-6 / EXP-A3-2 fuel correction with corrected coefficient
   (6 min + per-fold refit)
3. EXP-A2-2 with `rule_binding` dry-race gate (5 min)
4. EXP-A2-7 field-state competitor features F3 (8 min)

**Phase 2 — fold-safe, novel mechanism** (~50 min total)
5. EXP-A3-1 rank-sorted multi-neighbour + pit-pressure (18 min)
6. EXP-A3-3 `tirechange_pursuer` + lagged window (10 min)
7. EXP-A2-4 VSC/Full-SC split (8 min)
8. EXP-A3-4 Heilmeier log-curve residual (8 min + per-fold refit)

**Phase 3 — heavier, structurally novel** (~70 min total)
9. EXP-A3-6 KNN-target-mean-of-500 (20 min, inner-fold)
10. EXP-A2-3 bigram/trigram nested-fold TE sweep (12 min)
11. EXP-A3-5 multi-task auxiliary head (25 min)
12. EXP-A2-8 LightGBM stack-meta on richly-featured matrix (10 min)

**Phase 4 — speculative or cosmetic** (deferred menu)
13. EXP-A3-8 quantile + histogram aggregations
14. A2-5 Heilmeier `remaining_pit_stops_proxy` (subsumed by A3-4 if
    that fires)
15. A2-1 gap-to-car (subsumed by A3-1 if that fires)

Phase 1 is **22 min CPU and entirely fold-safe by construction**.
Recommend PI authorise Phase 1 as a single batch — if any pick lifts
the K=4+1 plain LR-meta gate by ≥+0.5 bp, it's a stack-add candidate
and we know the FE axis is alive. If Phase 1 is all NULL, we know the
field-state mechanism class is fully exhausted on this comp.

## Sources

### Primary

- TUMFTM/race-simulation: `https://github.com/TUMFTM/race-simulation`
  (`vse_supervised.py`, `calc_tire_degradation.py`)
- Heilmeier 2018 ITSC paper, DOI 10.1109/ITSC.2018.8569534
- Heilmeier 2020 Applied Sciences 10(21):7805
- NFL Big Data Bowl 2020 The Zoo: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/writeups/the-zoo-1st-place-solution-the-zoo`
- NFL Big Data Bowl 2024: `https://operations.nfl.com/gameday/analytics/big-data-bowl/2024-big-data-bowl-winner-and-finalists/`
- NFL Big Data Bowl 2025: `https://sumersports.com/the-zone/nyu-team-clinches-2025-nfl-big-data-bowl-victory-using-sumersports-framework/`
- Soccer LEM: `https://link.springer.com/article/10.1007/s10994-024-06606-y`
- Frontiers AI F1 pit-stop: `https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full`
- arXiv 2501.04068 (RSRL TimeSHAP analysis ranking "Gap Ahead" #1)
- BMC Med Res Methodol discrete-time hazard 2022: `https://pmc.ncbi.nlm.nih.gov/articles/PMC9316420/`

### Top-Grandmaster + classical writeups

- NVIDIA cuDF FE (Deotte S5E2): `https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/`
- NVIDIA cuML stacking (Deotte S5E4): `https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/`
- NVIDIA Grandmasters Playbook: `https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/`
- NVIDIA KGMON S6E3: `https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/`
- Otto 1st (Titericz/Semenov): `https://github.com/ageek/kaggle/blob/master/2015-Kaggle/otto-group-product-classification/winners-writeup-etc/1st-place-winner-solution-gilberto-titericz-stanislav-semenov.txt`
- Home Credit 2nd (Onodera): `https://github.com/KazukiOnodera/Home-Credit-Default-Risk`
- BlamerX repo: `https://github.com/BlamerX/Kaggle-Playground-Predection-Competition`
- IEEE-CIS Fraud 1st: `https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2`
- MoA "Hungry for Gold": `https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold`

### F1 practitioner

- f1metrics blog: `https://f1metrics.wordpress.com/tag/lap-times/`
- Motorsport Metrics fuel-load: `https://themotorsportmetrics.com/fuel-load-and-lap-time/`
- F1 Chronicle undercut: `https://f1chronicle.com/how-the-undercut-works-in-f1/`
- F1 Briefing undercut/overcut: `https://f1briefing.com/undercut-vs-overcut-pit-timing-explained/`
- Raceteq tyre degradation 2024: `https://www.raceteq.com/articles/2024/08/the-science-behind-tyre-degradation-in-formula-1`
- FastF1 examples gallery: `https://docs.fastf1.dev/examples_gallery/index.html`
