# 2026-05-13 — EDA Deep-Dive Synthesis

Day-13 deep dive across 6 phases (A–F).  Triggered by Day-13 Move B
same-field FM partition tying PRIMARY (LB 0.95032 = LB 0.95034 within
quantization).  Goal: surface load-bearing data facts for the next
2-10 bp lift.

PRIMARY 0.95034 · gap to top-5% 0.95345 = **31 bp**.

Phase outputs: `plots/eda_deep/{A..F}_*.{png,md}` + this synthesis.

---

## Phase summary (one-liners + load-bearing facts)

**A — Univariate**: 0/11 numeric features show train↔test drift (KS<0.004,
all p>0.1). Year-2023 pit rate 0.96% across **every Compound** (HARD 0.80%,
MEDIUM 0.95%, SOFT 1.53%, INTER 2.02%) → 2023 is a **flat-rate generator**,
not a race/compound shift.  AV-AUC=0.502 corroborated.  Driver tail:
221 drivers with ≤50 rows have pit-rate 0.144 vs 0.199 global.

**B — Pairwise + 2-way**: top numeric MI = RaceProgress (0.095), Stint
(0.079), Year (0.072); top categorical MI = Compound (0.037) ≫ Race
(0.017) ≫ Driver (0.013).  Compound × Stint cells: SOFT-S1 lift 4.25
(rate 25%), MEDIUM-S2 lift 2.25 (rate 45%), HARD-S2 +HARD-prev rate 44%
(61.7k samples, dominates Stint-2 blind spot).

**C — Three-way**: **Cumulative_Degradation has near-zero correlation
with TyreLife within each Compound** (HARD ρ=-0.08, INTER ρ=-0.26) →
they encode independent signals despite naming overlap. Compound ×
TyreLife-decile × RaceProgress-decile cell lifts up to 3.35× (SOFT
TL_d=5 RP_d=2; HARD TL_d=9 RP_d=8).  Year-2023 std across races = 0.0048
vs 0.11-0.16 for other years → 2023 generator ignores race-specific
strategy.

**D — Model-driven**: SHAP top-6 (LGBM) = TyreLife, Year, Stint,
LapTime_Delta, Position_Change, Race.  AVP in-sample meta:
**cb_slow-wide-bag dominates (+23.5 bp)** while ALL 6 FMs sit at
ΔAUC ≈ 0 in-sample — exact leakage-eaten vs leakage-robust signature.
PCA-on-OOF 8 clusters: cluster 2 (48k rows) **under-predicts target by
6.4 pp** (target 0.852 vs mean_pred 0.788, base_std 0.139); cluster 7
(9.5k rows) over-predicts by 6.4 pp (target 0.003 vs 0.067).

**E — FM embeddings (k=8 direct, val AUC 0.903)**: Compound cosine
matrix is **non-physical** — MEDIUM ↔ HARD = -0.67 (anti-correlated)
yet WET ↔ INTERMEDIATE = +0.59 (correctly paired). FM has learned a
binary-pit/no-pit cluster, NOT a tyre-spectrum gradient.  Field-pair
interaction strength: Year×Stint (0.386) > Year×Driver (0.361) >
Year×Race (0.343) ≫ Compound×anything (≤0.252).  FM mean-pred vs
GBDT mean-pred by (Compound, Stint): FM **over-predicts by 3-4 pp on
rare-late-stints** (INTER S4: +3.5 pp; SOFT S4: +3.8 pp), but FM
**matches target better than GBDT on SOFT S1** (target 0.242 vs FM
0.262 vs GBDT 0.239) — the +3 bp LB lift zone.

**F — Leakage asymmetry**: Family ΔAUC (Strat − GroupKF, bp):
GBDT-formulation 303 · CatBoost 261 · GBDT 233 · TargetEnc 222 ·
SeqFE-GBDT 212 · RuleResid 42 · SparseLR 42 · **FM 9** (FM 23–37×
more leakage-robust). Per-feature single-feature LR Δbp: **LapTime_Delta
+922 bp** (max), Position +436 bp, Stint +95 bp, TyreLife +45 bp,
Year **−360 bp** (FLIPS sign — Year generalizes *better* under
GroupKF). Per-cohort isotonic in-sample headroom on M5q: Stint +4.06
bp, Compound +3.28, Year +3.20 → real LB ≈ 1-2 bp.

---

## Hypothesis register (ranked, EV in bp; 5-question pre-flight applied)

| # | Hypothesis | Source | ρ vs PRIMARY | EV bp (P/M/O) | Notes |
|---|---|---|---:|---:|---|
| H1 | **Compound × TL-quintile × RP-quintile FM field**: add a single concatenated key (5×5×5=125 levels) as a 13th FM field → captures the 3.35× lift cells (SOFT-early, HARD-late) | Phase C | 0.998 | 0.5 / 2.0 / 5.0 | Untried interaction class; ρ≈0.998 (FM-class amp); 5-Q clean |
| H2 | **Compound-as-numeric-ordinal feature**: encode SOFT=0, MED=1, HARD=2, INTER=3, WET=4 and add to FM + sparse-LR; FM cosine showed mis-ordering | Phase E | 0.998 | 0.5 / 1.5 / 4.0 | Cheap; one feature add; corrects FM's binary-cluster artifact |
| H3 | **Per-Stint isotonic post-hoc on PRIMARY**: in-sample +4.06 bp; OOF realistic ~+1.5 bp; uses existing OOFs only | Phase F | 0.99996 | 0.3 / 1.2 / 2.5 | Free; no new model; risk-free if OOF-fit per-Stint |
| H4 | **Cumulative_Degradation × TyreLife interaction encoded explicitly** as `cumdeg_per_lap = Cum_Deg / max(TyreLife, 1)`, added to all bases; Cum_Deg vs TL ρ≈-0.08 within Compound proves independent signal | Phase C | 0.997 | 0.5 / 2.0 / 4.5 | New input that ALL bases (FM, GBDT, rules) can use |
| H5 | **LapTime_Delta race-z-score**: replace raw LapTime_Delta with `(x - μ_race) / σ_race` per (Race, Year, Compound); +922 bp single-feature leakage gap proves the raw form leaks | Phase F | 0.997 | 1.0 / 3.0 / 6.0 | Likely highest-EV; targets THE leakiest feature; lifts the leakage-robust bases without hurting GBDT pool |
| H6 | **Year × Stint partition FM** (4×8 = 32 sub-FMs, share embeddings via gating): FM field-pair Year×Stint magnitude 0.386 was highest; current FM mixes them | Phase E | 0.998 | 0.0 / 1.0 / 3.0 | Speculative; could mature into K=23 add candidate |
| H7 | **Cluster-2 specialist** (high-pit cells under-predicted by 6.4 pp): train an HGBC restricted to OOF cluster 2 (target rate 0.85, n=48k) and use as residual-corrector | Phase D | 0.999 | 0.0 / 0.8 / 2.0 | Niche; might saturate via PRIMARY meta absorption |

**Top 3 to attempt next**: H5 (LapTime_Delta z-score) → H1 (3-way FM field) → H3 (per-Stint isotonic).

---

## Cross-references back to mechanism_families_explored

- H1 reuses `factorization_machine_aug12` family but adds a NEW input field type (3-way concatenation), not in current 12 fields.
- H4 is genuinely new (no existing `cumdeg_per_lap` feature in d2a/d3a/d3b/d6/d9).
- H5 is genuinely new — `relative_state_fe` (m4) used position-ratio but not LapTime_Delta normalization.
- H7 overlaps with `d3c_stint2_specialist` (parked) — but that was Stint-2 only; H7 is on full OOF cluster 2 (mixed Stints, by mean-prediction band).

---

## Anti-findings (things to STOP doing)

- **Same-field FM reshuffles** (Day-13 Move B already proved tied at LB 0.95032). Stop on partition geometry; pivot to new fields.
- **GBDT meta-stackers** (d4 +d5 explored). Phase F shows GBDT ΔAUC 233 bp; LR-meta is metric-aligned and the leakage-asymmetry argument is strongest there.
- **3-way FM partitions** (d9g regressed). 2-way is sweet-spot.
- **LambdaRank pairwise meta** (Day-12 −86 bp regression). Confirmed dead.

---

## Pointers

- Plots: `plots/eda_deep/{A_fact_sheet,B_pairwise,C_threeway,D_model,E_fm,F_leakage}/*.png`
- Per-phase summaries: `plots/eda_deep/{A..F}_summary.md`
- Per-base ΔAUC table: `plots/eda_deep/F_leakage/per_base_delta_auc.csv`
- Scripts: `scripts/eda_deep/0{1..6}_*.py` (read-only, no submissions, no overwrites)

Next session: pull this synthesis, queue H5 first (LapTime_Delta race-z-score
across all bases, minimum-meta gate vs PRIMARY, then PI-approve LB submit if
predicted Δ ≥ +0.5 bp).
