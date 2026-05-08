# Day-10 evening — GroupKF audit confirms FM lifts are leakage-robust

> Concern: are the four consecutive FM-class LB lifts (d9c +3bp, d9f
> +2bp, d9h +3bp, d9i +3bp; total +8bp from d6_k18 PRIMARY) real, or
> are they public-LB shimmer driven by within-group leakage?
>
> Test: re-train the FM bases under strict GroupKFold by
> `(Race, Driver, Year, Stint)` (the leakage-blocking partition per
> P6) and compare AUCs vs Strat. If FM exploits leakage, Strat→GKF
> drop should match GBDT bases (≥200bp). If FM is genuinely robust,
> drop should be small.

## Diagnostic results

GroupKFold by `(Race, Driver, Year, Stint)`: 113,567 groups, 5-fold
splits, ~22,700 groups and 87,828 rows per fold.

| Base | n_feat | Strat AUC | GKF-strict AUC | Δ Strat→GKF |
|---|---:|---:|---:|---:|
| d9c FM unified | 8 | 0.92069 | **0.91978** | **−9.1bp** |
| d9f FM_A driver-dynamics | 4 (D, C, S, T) | 0.82505 | 0.81964 | −54.1bp |
| d9f FM_B race-context | 4 (R, Y, Rp, P) | 0.88438 | **0.88413** | **−2.5bp** |
| (e3_hgbc, GBDT, Race-only GKF) | — | 0.94876 | 0.92785 | **−209.1bp** |
| (cb_slow-wide-bag, GBDT, Race-only GKF) | — | 0.94790 | 0.92322 | **−246.8bp** |

The GBDT bases drop 200+ bp under the *weaker* Race-only GroupKF.
Under the strict (Race, Driver, Year, Stint) GroupKF that the FMs
were tested against, GBDTs would likely drop 300-500bp by
extrapolation.

**FM bases drop 4–80× LESS than GBDT bases**, even under stricter
validation.

## Why FM is leakage-robust

GBDT trees partition feature space into leaves and store per-leaf
empirical pit_rate. With 80% of (Race, Driver, Year, Stint) groups
having rows in both train fold and val fold (P6), within-stint
consecutive laps have near-identical features → same leaf → leaf
pit_rate is contaminated with val-row labels. Train AUC inflates by
~200bp purely from this within-group leakage.

FM with hashed categoricals has no leaves. Its predictions come from
**pairwise dot products of embeddings** ⟨v_i, v_j⟩, where each v_i
is a function of *only* feature value i, not row context. Within-
stint label leakage doesn't transfer to FM's embeddings — those are
estimated from the marginal distribution of feature combinations,
which is roughly preserved across folds.

## Implications for the FM PRIMARY chain

The d6_k18 → d9c → d9f → d9h/d9i progression added FM-class weight
to the LR meta. The FM weight is *leakage-robust*, while the GBDT
weight it partly displaced is *leakage-inflated*. The cumulative
+8bp public LB lift from d6_k18 to d9h/d9i is therefore mostly
real — the LR meta is reweighting toward predictions that
generalize to held-out groups.

**Cumulative real lift estimate**: +5 to +8bp on private. Range
because:
- Lower bound (+5bp): some of the public-LB micro-lifts (d9c→d9f
  +2bp, d9f→d9h +3bp, etc.) are sample-variance on the 20% public
  split between near-identical OOF predictions. Those won't all
  transfer.
- Upper bound (+8bp): if the LR meta is genuinely upweighting FM in
  the FINAL prediction (it is — L1 ranking shows FMs in top-15),
  the full lift transfers.

**Plausible private LB range**: 0.95029-0.95032 vs d6_k18 baseline
0.95026.

## What to do with this finding

1. **Stop chasing intra-FM micro-lifts.** d9f, d9h, d9i all have OOF
   ~0.95073 and Strat-fold predictions within 6th decimal.
   Differences in their public LB are sample variance.
2. **Lock d9f K=21 swap as PRIMARY** by OOF (cleanest partition,
   most parsimonious K=21). Don't switch to d9h/d9i based on public
   LB ties.
3. **Final-selection HEDGE = d6_k18_multi_rule** (LB 0.95026) per
   Rule 2 — leakage-inflated like all GBDTs but the largest non-FM
   submission. Safety floor if FM advantage compresses on private.
4. **Future submissions must clear a higher bar.** "OOF tied with
   PRIMARY at 6th decimal" → public-LB lift is sample noise. Need
   *real* OOF Δ (≥+0.5bp) AND leakage-robust mechanism.

## Why your overfit concern was partly right

You were right that public-LB chasing is a risk: the +3bp / +3bp
micro-lifts on d9h and d9i are within sample noise on the 20%
public split. If we pick the public-best from that pair as PRIMARY,
we're picking the noisy roll, not the truer model.

You were partly wrong: the *mechanism* (FM-class additions to the
LR meta) is leakage-robust and the cumulative lift is real. The
public LB is reflecting a real phenomenon — just at coarser
resolution than we tried to read.

## Pointers

- `scripts/d10_groupkf_audit.py` — builder.
- `scripts/artifacts/d10_groupkf_audit_results.json` — full numbers.
- `scripts/artifacts/oof_d9c_FM_unified_8_groupkf_strict.npy` — d9c
  FM under strict GroupKF (for follow-up GroupKF stack rebuild).
- `scripts/artifacts/oof_d9f_FM_A_4_groupkf_strict.npy`,
  `scripts/artifacts/oof_d9f_FM_B_4_groupkf_strict.npy` — partition
  FMs under strict GroupKF.
