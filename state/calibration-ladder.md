# Calibration ladder

How much did each meaningful experiment move the cross-validation score
and the leaderboard? Use this to predict what a new experiment of a
similar kind is worth before running it.

OOF = out-of-fold cross-validation. LB = public leaderboard. ρ = rank
correlation between two prediction sets. bp = basis points (one-ten-thousandth
of an AUC point). All scores are AUC.

For pre-Day-13 entries, see `audit/archive-2026-05-06-claude-md-compression.md`.

| Experiment | Cross-validation AUC (Stratified) | Cross-validation AUC (GroupKFold) | Leaderboard | Notes |
|---|---:|---:|---:|---|
| Two-anchor baseline | 0.94075 | 0.92059 | 0.94113 | Stratified is the LB proxy for this comp; gap +3.8 bp. |
| **m5q stack** (LightGBM-pruned + RealMLP, 14 bases) | 0.95057 | n/a | 0.95005 | Old PRIMARY (Day 3); +14 bp on the LB from +1.4 bp OOF (10× upside). |
| **18-base multi-rule stack** | 0.95065 | n/a | 0.95026 | Old PRIMARY (Day 7); +2.1 bp on the LB. |
| **First factorisation-machine base** (d9c FM) | 0.92069 | n/a | n/a | ρ vs PRIMARY 0.899; first FM-class base added; passed the min-meta gate by +0.18 bp. |
| 20-base swap with FM (d9c) | 0.95070 | n/a | 0.95029 | +3 bp on LB; first time the FM class amplified. |
| **Most-diverse-ever FM base** (d9f FM_A driver-dynamics) | 0.82505 | n/a | n/a | ρ vs PRIMARY 0.487 — the most diverse single base since the early baselines. |
| 21-base swap with two-FM partition | 0.95073 | n/a | 0.95031 | +2 bp on LB (6× upside). |
| **22-base add with augmented FM (12 fields)** | 0.95073 | n/a | 0.95034 | +3 bp on LB; predicted +0.01 bp, realised +3 bp (300× upside). |
| 21-base swap with 2-way augmented FM | 0.95071 | n/a | 0.95034 | +3 bp on LB; cross-validation predicted regression but LB went positive. |
| 13-base baseline (under GroupKFold audit) | 0.95043 | 0.92744 | n/a | Gap −229 bp — leakage signature. |
| 15-base with two FM bases (under GroupKFold) | 0.95052 | 0.92764 | n/a | FM class lift +0.87 bp on Stratified became +2.01 bp on GroupKFold — 2.3× amplified. |
| Leak-corrected meta (refit on GroupKFold OOF) | n/a | 0.92764 | n/a | G3 gate failed (flip ratio 0.001); held; predicted LB 0.95001. |
| **First per-segment shrinkage stacker** (Compound, τ=100k) | 0.95076 | n/a | 0.95033 | +2 bp LB on +0.30 bp OOF (6.7× upside). First hier-meta. |
| Per-segment shrinkage on Stint, τ=100k | 0.95082 | 0.94600 | 0.95041 | +7 bp LB (11.6× upside); GroupKFold +2.59 bp = 2.9× amplified. |
| **Per-segment shrinkage on Compound × Stint, τ=20k** | 0.95083 | n/a | 0.95049 | +8 bp on LB; old PRIMARY (Day 13). |
| Same, τ=100k | 0.95081 | n/a | n/a | Held; +0.82 bp OOF; hedge-eligible. |
| Stint, τ=20k | 0.95082 | n/a | n/a | Held; +0.88 bp OOF. |
| Within-stint LightGBM features / cross-driver intra-race features | 0.94194 / 0.94250 | n/a | n/a | LightGBM-class feature engineering null at min-meta. |
| 3-way concat-field augmented FM | 0.92639 | n/a | n/a | ρ=0.917 most diverse; min-meta −0.13 bp null. |
| 16-field augmented FM (Move D) | 0.92741 | n/a | n/a | +20 bp standalone vs 12-field; min-meta −0.07 bp fail (FM-aug saturated at 12 fields). |
| TabPFN v2.5 (150k rows) | 0.94446 | n/a | n/a | Dead; ceiling 0.944; v2.6 ran out of memory on Kaggle's P100. |
| **Denoising-autoencoder + LightGBM base** (768-dim DAE → LGBM) | 0.94007 | n/a | n/a | ρ=0.948 (most diverse since FM_A driver-dynamics); min-meta +0.79 bp. |
| **22-base + DAE + per-segment Compound × Stint, τ=20k** | 0.95090 | n/a | **0.95059** | PRIMARY (Day 15); +1 bp on LB; flips 59 / 53; realised amp 1.4×. |
| ExtraTrees / kNN-LightGBM | 0.92967 / 0.94166 | n/a | n/a | min-meta +0.05-0.06 bp; correlation ≈ 0.996; R5 hedge only. |
| **22-base, original-data continuous-only, τ=20k** | n/a | n/a | **0.95089** | PRIMARY (Day 16); +3 bp on LB; clean per-segment base-add (passed Rule 24 fold-safe audit). |
| Strict fold-safe target-reformulation audit | n/a | n/a | n/a | Day-17 friction `target-construction-layer-leakage`: all inv-laps / pit-horizon / reverse-cum candidates collapsed 88-100% under strict OOF. |
| **Chain-decomposition base** (causal + Gaussian likelihoods on original DGP) | 0.94954 | 0.9914 | n/a | Day-17 PM K=21+1 = +7.37 bp (largest single-base of session); per-step original-DGP log-likelihood. |
| **23-base + d16 + chain-decomp, τ=20k** | 0.95184 | 0.9923 | **0.95149** | PRIMARY (Day 17); +6 bp on LB; PI sealed prediction +3 bp, agent +5 bp, actual +6 bp. |
| Single LightGBM honest fold-safe ceiling (kitchen-sink Rozen recipe) | 0.94563 | n/a | n/a | Day 17 single-model ceiling; −52 bp from PRIMARY OOF; stacking is justified by this gap. |
| **CatBoost yekenot transfer (v4)** | 0.95200 | n/a | n/a | Day 17 PM single-model lift; +24 bp at K=21+1 (double the previous v3); ρ vs PRIMARY 0.972. |
| **RealMLP yekenot full (h1d)** | 0.95257 | n/a | n/a | Day 17 PM single-model; matched yekenot's published OOF within 1.6 bp. |
| **23-base v4+h1d Compound × Stint, τ=100k** | 0.95415 | n/a | **0.95354** | PRIMARY (Day 17 PM); +30 bp OOF; realised gap −6.1 bp. |
| **27-base v4+h1d+DGP-class, τ=100k** | 0.95432 | n/a | **0.95368** | Old PRIMARY (2026-05-07 PM); hedge-eligible per Rule R7 since 2026-05-08 PM PRIMARY swap. |
| **K=10 forward-greedy + Path-B C×S τ=100k** | 0.95420 | n/a | **0.95356** | Sparse-pool calibration probe (2026-05-08 PM). Δ −1.2 bp vs old K=27 PRIMARY; OOF→LB transfer precise to 0.1 bp at this scale. |
| **K=4 forward-greedy + Path-B C×S τ=100k** | 0.95403 | n/a | **0.95351** | **NEW PRIMARY (2026-05-08 PM).** Bases: yekenot-RealMLP, CatBoost-yekenot, f1_hgbc_deep, d16_orig_continuous_only. Δ −1.7 bp on LB vs old K=27 PRIMARY (was Δ −2.93 bp on OOF); LB outperformed OOF prediction by ~1.2 bp. 4 bases capture 99% of the bank's LB value with 15% of the bases. |
| **Random forest base on yekenot recipe** (no orig) | 0.94178 | n/a | n/a | Day-20 PM forest sweep, Angle A. Most-diverse positively-gating base in the K=4 era: ρ=0.9595 vs PRIMARY-test, K=4+1 LR-meta +0.26 bp (largest K=4-era lift on a new base). +12 bp standalone over `d15c_extra_trees` (raw 0.92967). Hedge-eligible per R5. Path-B refit on K=5 is the natural follow-up. |
| **Kitchen-sink RF** (yekenot + constraint-violations + inter-stint = 57 feat) | 0.94054 | n/a | n/a | Day-20 PM kitchen-sink probe. Standalone OOF −1.24 bp vs yekenot-only; K=4+1 LR-meta +0.25 bp (within 0.02 bp of Angle A — first reproducibility check on the forest-base lift). ρ=0.9580. Feature breadth hurts RF on this data (weak features dilute split capacity at random-feature-subset level). PI hypothesis that RF scales with feature breadth refuted. |
| RF as meta-stacker on K=4 expansion | n/a | n/a | n/a | Day-20 PM forest sweep, Angle B FALSIFIED. RF-meta OOF 0.95384 vs LR-meta 0.95399 (Δ −1.54 bp). Bagged-tree variant of Day-20 PCA-meta finding; closes non-LR meta family across boosted + bagged. |
| RF on combined input (K=4 expansion + 6 raw numerics) | n/a | n/a | n/a | Day-20 PM forest sweep, Angle C FALSIFIED. RF 0.95393 vs LR-on-same 0.95400 (Δ −0.70 bp). Adding raw numerics doesn't rescue tree-class meta. |
