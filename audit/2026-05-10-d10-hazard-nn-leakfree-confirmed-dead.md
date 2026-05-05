# 2026-05-10 — Hazard-NN leak-free diagnostic confirms DEAD

> Day-9 K=19 hazard stack LB: 0.94711 (gap −73.5bp from OOF 0.95446).
> Day-10 leak-free re-run with stint-drop within Strat fold: leak-free
> OOF 0.92013, leak magnitude **230.6bp** vs leaky bag's seed-42 OOF
> 0.94319. Hazard NN architecture itself extracts ~zero useful signal
> beyond the within-stint label leakage.

## Result

| Variant | Strat OOF | Note |
|---|---:|---|
| baseline_two_anchor | 0.94075 | LB-proxy anchor |
| hazard NN bag (LEAKY, Day-9) | 0.94806 | bag rank-avg, +73bp baseline |
| hazard NN seed-42 5-fold (LEAKY) | 0.94319 | direct fold-by-fold reference |
| **hazard NN leak-free (Day-10)** | **0.92013** | **−206bp baseline, −230bp leaky** |
| Per-fold leak-free AUCs | 0.918, 0.919, 0.920, 0.920, 0.920 | tight σ=0.001 |

Wall: 5s/fold (vs leaky 36s) — model converges faster on 40% of data.

## What this proves

1. **Hazard target's `bfill` IS the dominant signal source.** Removing
   it via stint-drop strips ~230bp.
2. **The hazard NN architecture has no real signal beyond the binary pool.**
   At 0.92013 OOF, leak-free hazard is below baseline. It would NOT
   meet any pool-add gate. The K=20 bucket NLL provides no usable
   regularisation for this dataset's scale and sparsity.
3. **Today's submission was essentially a leak detector.** OOF 0.95446 →
   LB 0.94711 (−73.5bp gap) is exactly what we'd expect when 230bp of
   the standalone OOF lift comes from val-leak.

## Pool implications (load-bearing)

Inspection of `scripts/b_laps_until_pit.py`:
```
def build_laps_until_pit(train):
    df = df.sort_values(["Race", "Driver", "LapNumber"]) ...
    # laps until next PitNextLap=1 within (Driver, Race)
```
- Group key: `(Race, Driver)` — even **broader** than hazard's
  `(Race, Driver, Year, Stint)`. Same bfill-style label propagation.
- This base is **in the M5q pool** (K=18 PRIMARY).

Inspection of `scripts/a_horizon_shift.py`: similar pattern (target
is shifted PitNextLap; future-label-dependent within group).

So both `a_horizon` and `b_lapsuntilpit` carry within-group label
leakage. **K=18's LB calibration (gap −3.9bp) is robust because**:
- All ~14 binary classifiers in M5q (no leak in their targets) anchor
  the gap structure. Their dominant L1 weight in the LR meta keeps
  the stack's predictions close to the binary-direct reference.
- The 2 reformulation bases (a_horizon, b_lapsuntilpit) carry their
  leakage but are bounded in L1 weight (a_horizon L1 = 0.674,
  b_lapsuntilpit L1 = 0.573 in K=20 c5; similar in K=18). Their
  contributions are diluted enough that the gap stays calibrated.
- The hazard NN entered K=19 with **L1 = 9.519** — dominating the
  meta. With its 230bp of leak, the gap exploded.

So K=18 stays load-bearing PRIMARY. Don't pivot from it.

## Why the framework didn't catch this in advance

| Gate | Triggered? | Why missed |
|---|---|---|
| pre_submit_diff (ρ < 0.999) | PASS (ρ=0.96) | only diffs predictions, not target construction |
| min-meta OOF lift | PASS (+30bp) | leak inflates OOF lift |
| K=N+1 OOF gate (≥+0.5bp) | PASS (+38bp) | same |
| Q6 ρ-vs-K=18 < 0.999 | PASS (0.96) | ρ-orthogonality is a property of predictions, not targets |
| Predicted-LB gap (cal'd) | wrong (calibrated for ρ ≥ 0.99) | extrapolated −0.0004 instead of true −0.0079 |

**Missing gate (proposed Rule 16 Q7)**: any base whose target/feature
construction depends on FUTURE same-group labels gets ×0.1 EV
downgrade unless its OOF was computed under GroupKFold by the
relevant group. Apply this PRE-FLIGHT, not post-submit.

## Action items for Day-10 (post-this audit)

1. **Hazard NN is DEAD.** Do not rerun. Held submissions
   (`submission_d9_hazard_nn.csv`, `submission_d10_hazard_nn_leakfree.csv`)
   stay held; they would score sub-baseline.
2. **Pivot to TabM v3** with extended training (PI's earlier prompt).
   TabM showed best val cross-entropy at epoch 5 then 20 epochs of
   oscillation — possibly a deeper basin reachable with longer training
   or HPO. Same gates apply. **No leak risk** since TabM uses binary
   PitNextLap target directly (no within-group propagation).
3. **G4 SCARF/VIME pretrain** on aadigupta1601 unlabeled corpus
   (HANDOVER C-tier; different unlabeled corpus avoids d5 amp).
4. **Codify Rule 16 Q7** in CLAUDE.md (with PI go).
5. **Defer**: explicit a_horizon / b_lapsuntilpit leak audit. K=18 LB
   calibration is stable, so disturbing the pool composition is
   high-risk-low-reward unless we have a clean replacement.

## Submissions used

- 1/10 today (Day-9 hazard burned). Day-10 still 0/10.
- Total: 15.

## Artifacts

- `kernels/hazard-nn-leakfree-gpu/` — kernel code
- `scripts/artifacts/d10_hazard_nn_leakfree_results.json` — numerical detail
- `scripts/artifacts/oof_d10_hazard_nn_leakfree_strat.npy`,
  `test_d10_hazard_nn_leakfree_strat.npy` — leak-free artifacts (held)
- `submissions/submission_d10_hazard_nn_leakfree.csv` — **HELD,
  do not submit** (predicted LB ~0.916, well below baseline).

End. Sober closure of the hazard-NN thread.
