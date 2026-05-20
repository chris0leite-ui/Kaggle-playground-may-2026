# Strategy critique — plateau-triggered, post-Phase-4 (2026-05-19 PM)

Triggered by: **4 consecutive nulls** this session (R11-B transformer,
R11-C survival, R12-1 cb_resid, R15 Phase 4 cb_focal_weighted). Per
`loops.md` plateau spec, Strategy-critic fires BEFORE Research-loop.
Section 5 first per plateau ordering. **Delta on** earlier today's
`2026-05-19-strategy-critique.md` (post-cb_horizon-win critique).

---

## 5. Headroom math vs plan (refreshed)

- PRIMARY R15 LB **0.95397** (was 0.95392 at earlier critique; +0.05 bp
  closed via R13 → R14 → R15 PRIMARY swaps).
- Top-5% **0.95405** → **+0.08 bp** to clear (was +1.3 bp).
- Leader **0.95476** → **+0.79 bp** to catch.
- Days remaining: **13**.

Open H-list candidates (midpoints per precedent):

| Candidate | Midpoint (bp) |
|---|---|
| 2a non-Gaussian Path-B shrinkage | +0.02 |
| 2c GroupKF-meta HEDGE | 0 (hedge) |
| 3c remaining target reformulations | +0.02 |
| 4a Pirelli pit-window scrape (NEW info) | +0.05 |
| **4b F1 historical-priors join (NEW info)** | **+0.10** |
| 5b/5c invlaps τ variants (HELD) | 0 (hedge) |
| 8b Single CB Rozen recipe | 0 (weaker standalone) |
| **Real-F1-driver-specialist (NEW from §1)** | **+0.05 to +0.15** |

Σ midpoints ≈ **+0.27 bp**. 50% additivity discount → **+0.14 bp**.

**Verdict**:
- Top-5% (+0.08): **REACHABLE** (1.7× headroom under discount).
- Leader (+0.79): **5.5× SHORTFALL**. Single-track plan does NOT reach.

**Contingency**: leader-catch requires NEW information (4a/4b external
data) — internal-data axes confirmed at Bayes-ceiling by R12-1 cb_resid
(sub-random AUC on residual prediction) and now reinforced by Phase 4
cb_focal_weighted G1 FAIL (-52 bp standalone vs PRIMARY, ρ 0.982).
**Pivot from mechanism-stacking to external-data integration.**

---

## 1. Per-segment failure map — **the headline finding**

PRIMARY OOF AUC 0.95449. Bottom-3 segments:

| seg | level | n | AUC | Δ vs mean (bp) |
|---|---|---|---|---|
| **Compound** | **WET** | 1,355 | 0.8444 | **-1101** |
| **Driver** | **VET** | 359 | 0.8528 | **-1017** |
| **Driver** | **MSC** | 355 | 0.8768 | **-777** |

All bottom-8 segments are real-F1 entities (WET compound + real F1 driver
codes VET/MSC/COL/DEV/ALO/RIC/BEA). Synthetic D### drivers all > 0.978 AUC.

### Driver-class ablation — DOMINANT FAILURE AXIS

| Class | n | % rows | prior | positives | AUC |
|---|---|---|---|---|---|
| **Real-F1 (123 codes)** | 168,324 | **38.3%** | 0.240 | 40,400 (**46%** of pos) | **0.94672** |
| Synthetic (764 D### codes) | 270,816 | 61.7% | 0.174 | 47,025 (54% of pos) | 0.95874 |

**Model is 120 bp worse on real-F1 drivers than synthetic.** Real-F1
contributes 46% of all positive labels — biggest lift surface available.

### Compound ablation
| | n | AUC |
|---|---|---|
| DRY (MED/HARD/SOFT) | 420,403 | 0.95476 |
| INTERMEDIATE | 17,382 | 0.94089 |
| WET | 1,355 | 0.84439 |

### WET-specialist trap
WET is 0.31% of rows / 0.04% of positives. Even lifting WET to overall mean
AUC shifts overall metric ≲0.5 bp. **Do not waste a slot on a WET-only
specialist.** Real-F1 is the productive axis.

---

## 2. Probability calibration

Brier **0.06667**, ECE-15 **0.00120**. Reliability gap ≤ 0.008 in every
decile bin. **Perfectly calibrated.** Isotonic / Platt yield nothing.
Calibration axis CLOSED for good (re-confirms K11 finding).

---

## 3. Model-disagreement localization

Only 2 base OOFs locally (K17 PRIMARY + R15_cb_focal); other 15 bases live
in Kaggle Dataset `chrisleitescha/s6e5-artifacts` — not re-downloaded this
session. From available pair: 99th-percentile row-stdev 0.30; residual-
difficulty subset n=4,392 (prior 0.083); K17 AUC there 0.6750, CB AUC 0.6682.
Headroom on these rows ~28 bp → ~0.3 bp overall. **Sub-dominant vs §1
real-F1 axis.**

ACTION before next deep dive: `kaggle datasets download
chrisleitescha/s6e5-artifacts -p scripts/artifacts/ --unzip` to re-sync
the full base pool, then recompute proper N=17 disagreement.

---

## 4. Unexploited structural signal scout

From `comp-context.md::structural_findings` — no NEW finding since
2026-05-18 critique (lead_pitstop dead, train/test row-iid confirmed,
test_lead_pitstop already exploited via Path-B transductive TE).

NEW from this critique:
- **Real-F1 driver subset (§1) makes ISSUES.md 4b the load-bearing axis**.
  External historical priors (FastF1 / Ergast / debashish 1950-2022) on
  the 123 real-F1 drivers target the exact failure surface. Aggregate-key
  join (Driver / Driver×Circuit / Driver×Year) sidesteps the row-level
  synthetic-augmentation problem that capped FastF1 at 0.55 TyreLife
  correlation in earlier H2 probes.
- **`is_real_driver` indicator + Real-F1 driver TE base** — 5-min probe
  to confirm the axis is exploitable BEFORE committing to the full 4b
  external-data join.

---

## Concrete plan re-rank (delta on earlier critique queue)

| Rank | Candidate | EV (bp) | Effort | Note |
|---|---|---|---|---|
| **1** | **`is_real_driver` + Real-F1 TE base (NEW)** | +0.02 to +0.05 | 20 min | Cheapest diagnostic confirming real-F1 axis is alive before larger 4b commit. **Run first.** |
| **2** | **F1 historical-priors aggregate join (ISSUES 4b)** | +0.10 | 60-90 min | Dominant failure axis; aggregate-key avoids row-level synth perturbation. Highest EV in queue. |
| 3 | Pirelli pit-window scrape (ISSUES 4a) | +0.05 | 60 min | Complements 4b on INTERMEDIATE underperformance too. |
| 4 | Non-Gaussian Path-B shrinkage (2a) | +0.02 | 30 min | Internal-only; low novelty axis at Bayes ceiling. Defer. |
| 5 | Re-sync full base pool for §3 disagreement map | enabling | 5 min | Required before any diversity specialist sub-model. |
| — | WET-specialist | <0.005 | — | **SKIP** — row-share too small to matter. |
| — | cb_v5_xl kitchen-sink (R12-4 parked) | ~0 | 60 min | CB axis closed (4 nulls). **PARK PERMANENTLY.** |

**Strategic posture pivot**: from "mechanism-stacking on row features"
(closed by R12-1 Bayes-ceiling + Phase 4 G1 FAIL) → **"external-data
join on the real-F1 driver subset"** (4b/4a become load-bearing).

**Slot allocation today** (7 unspent): hold until item 1 confirms axis;
then 1 slot for item 1 G2-clear, 1 slot for item 2 G2-clear. Remaining
5 forfeit per PI directive ("G2-clear mechanisms only").

---

Source data: `audit/2026-05-19-strategy-critique.json`,
`scripts/strategy_critic_2026_05_19.py`.
