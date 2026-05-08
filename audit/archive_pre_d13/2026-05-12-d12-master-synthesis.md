# Day-12 master synthesis — 6 wider-step options run overnight

> Trigger: PI directive "advancing in too small steps; what could
> deliver 10bp+?". Six options launched in parallel as background
> subagents. 5 of 6 falsified; 1 produced a load-bearing structural
> finding (GroupKF rank-lock dissolution).

## Scoreboard

| # | Option | Verdict | Δ (best stack OOF / pred-LB bp) |
|---|---|---|---:|
| 1 | GroupKFold full rebuild | **STRUCTURAL FINDING** | meta ρ-Strat-vs-GKF = 0.9914 (rank-lock dissolves) |
| 2 | TabPFN-2.5 fine-tuned | KERNEL READY (Kaggle GPU) | est. +5–15bp std-alone |
| 3 | T1.2 multi-formulation 3-of-3 | FALSIFIED | +0.16–0.27bp; below quantization |
| 4 | Year-specialist + AV-weights | FALSIFIED | −4.54 to −4.96bp min-meta |
| 5 | LambdaRank meta + AUC-pairwise | FALSIFIED | −86bp (Race-grouped) |
| 9 | Monolithic single-model bag | DIAGNOSTIC | bags −19 to −28bp; K=21 complexity justified |

## The load-bearing finding (Option 1)

Under GroupKFold(Race, Driver, Year, Stint), bases split into TWO
populations:

  - **GBDT bases** (m2_xgb, e1_cb_sub, cb_*, e3_hgbc, f1/f2_hgbc,
    e5_optuna_lgbm, baseline, d2a_te) drop **−200 to −343bp** in OOF
    AUC vs Strat. They eat within-stint leakage.
  - **Rule + sparse-LR + FM bases** drop only **−9 to −43bp**. FM
    drops only −9bp — **23–37× more leakage-robust than every GBDT**.

Under GroupKF the LR-meta re-weights dramatically: cb_slow-wide-bag
−17 ranks, e5_optuna_lgbm −13; **FM jumps +15 ranks (#20→#5)**,
R6_next_compound +11, rule_year_race L1 0.78→2.70.

Meta-test ρ(Strat-vs-GKF) = **0.9914** (K=21) / **0.9856** (K=20
clean) — both well below the 0.999 RHO_TIE threshold.

**The rank-lock at ρ=0.9999 the team has been hitting since Day-7 is
substantially a Strat-leakage artifact.** Under leakage-blocked OOF
the same pool produces meaningfully different predictions and weights.

## How the 5 falsifications cohere with the structural finding

  - **T1.2 multi-formulation FAIL**: censored / ratio / survival
    standalone OOF 0.54–0.67. These are time-to-event LGBMs — same
    GBDT class as the 13 leakage-eaters. Strat OOF flatters them
    weakly, GroupKF-style validation would crush them.
  - **AV reweighting null**: AV-AUC = 0.50191. Train/test are i.i.d.
    row-level (U3). No domain shift to exploit. Confirms our gap is
    NOT distributional shift; it's stratified-fold within-group leak
    cancellation cap.
  - **Year=2023 inverted**: Pool 2023 AUC 0.94602 (highest of any
    Year). The −45bp segment story was about the pool's *aggregate*
    not *2023*. Cohort splitting strips cross-Year regularization
    (specialist 2023-AUC −105bp). Year is not the lever; *base-class
    leakage robustness* is.
  - **LambdaRank meta −86bp**: Race-grouped pairwise loss collapses
    cross-Race calibration that LR's logit channel preserves. The
    LR-meta on `[raw, rank, logit]` is metric-aligned for global AUC
    when bases are well-calibrated — and our bases ARE well-calibrated
    on Strat by construction.
  - **Single-bag −19 to −28bp**: K=21's edge isn't OOF-noise overfit;
    it's *real cross-base routing*. Specifically the routing exploits
    the FM/rule-class differentiation that GroupKF reveals.

**Unifying frame:** the K=21 stack works because LR-meta routes
between bases that span *both* leakage-eating GBDTs (high Strat AUC,
low GroupKF AUC) AND leakage-robust FM/rules. The two populations
together cover the LB row-iid distribution. We've been searching for
"new bases" without recognizing this duality.

## Strategic implication — pivot to FM/rule-class diversification

Option 1 is the only thing that actually advanced the strategic
picture. The implied refactor:

  - **Replace 3 most-leakage-eating GBDTs** (e5_optuna_lgbm,
    cb_slow-wide-bag, e1_cb_sub — all ΔAUC ≥ −215bp) with more
    FM-class bases.
  - **2-way multi-FM is the sweet spot** (d9f Day-10 +2bp); 3-way
    over-fragments. **5/3 or 4/4 feature partitions** untried — cheap
    +1–4bp tail.
  - **DeepFM-lite** (FM + small MLP head) — adds non-linearity over
    FM embedding space without going to TabM/RealMLP density.
  - **Regularised FFM re-attempt** with stronger L2 / dropout —
    Day-10's FFM overfit, but the 23-37× leakage-robustness ceiling
    means even a 0.5× FM-class lift is high-EV.
  - **GroupKF-meta as HEDGE for R5 final-3-day window** — it's
    leakage-robust by construction. Public LB is row-iid (don't submit
    as PRIMARY), but for private-LB hedge it's the highest-diversity
    meta-output we have (ρ vs PRIMARY = 0.9914).

## TabPFN-2.5 (Option 2) is the only live "10bp shot"

Kaggle GPU kernel ready at `kernels/d12-tabpfn-finetune-gpu/`. Its
ρ is unknown but the model class (foundation model with pretrained
priors) sits OUTSIDE both GBDT and FM regimes, so leakage population
is unknown — could be a third population entirely.

EV: +5–15bp standalone, +1–3bp stack median, +3–9bp stack tail.
Single-shot LB submit decision deferred to PI per Rule 1.

## Subagent corrections / dead-list updates

  - **Option 3's recommendation to "prioritize T1.4 hazard NN"** —
    REJECT. Hazard NN is 2× falsified (Day-9 leaky burned −73.5bp;
    Day-10 leak-free architecture zero signal). Stays dead-listed.
  - **Option 9's recommendation to retry 5-seed bag** — defer; even
    if it hits PRIMARY OOF it can't lift LB, and Option 1 makes the
    diagnostic answer ("K=21 complexity justified") stand without a
    re-run.
  - **Add to mechanism_families_explored**:
    `t12_censored_regression`, `t12_ratio_target`,
    `t12_stintlevel_survival`, `year_segmented_specialist`,
    `adversarial_validation_reweight`, `lambdarank_race_meta`,
    `aucpairwise_xgb_base`, `single_bag_e3_5seed`,
    `groupkf_full_pool_meta`.

## What would actually deliver 10bp from here

  1. **TabPFN-2.5 fine-tuned** (pending Kaggle GPU run). Live shot.
  2. **FM-class diversification refactor** (4–6 weeks of options
     compressed: 5/3 multi-FM, 4/4 multi-FM, DeepFM-lite, reg-FFM).
     Each +1–3bp; aggregate +3–8bp realistic.
  3. **GroupKF-meta as hedge in final-3-day window** (0–5bp on
     private LB; +0bp on public).
  4. **Pool refactor with 3 GBDT drops + 3 FM-class adds**, evaluated
     under BOTH Strat AND GroupKF gates. EV +2–6bp.

Compounding: realistic +5–10bp aggregate over remaining 14 days
without a TabPFN tail. With TabPFN tail: +10–20bp possible.

Top-5% (0.95345) gap −31bp; reachable only if TabPFN lands tail.

## Pointers

  - Detailed: `audit/2026-05-12-d12-{groupkf-rebuild,t12-multi-formulation,year-specialist-advweight,lambdarank-meta,monolithic-bag-probe,tabpfn-finetune-prep}.md`
  - Per-option results JSONs in `scripts/artifacts/d12_*_results.json`
  - GroupKF pool artifacts: `scripts/artifacts/{oof,test}_*_groupkf.npy`
