# Next-session prompt — paste this when you return

## State at hand-off

- **PRIMARY**: R7.1 K=13 + Path-B DriverClass × Stint τ=100k →
  **LB 0.95389**.
  File: `submissions/submission_K13_pathb_driverclass_stint_tau100000.csv`.
- **Hedge**: R7.2 (LB 0.95389, 5-seed fold-fit bag of R7.1).
  File: `submissions/submission_K13_dcs_pathb_foldbag.csv`.
- **Top-5% gap**: 1.6 bp (target 0.95405). Leader gap: 8.7 bp.
- **Submissions**: 49/270 total; 7 used 2026-05-18; **3 daily
  slots remaining today**, **10 fresh slots tomorrow** at Kaggle
  UTC midnight.
- **Comp-day**: 18 of 31; 13 days remaining.
- **Branch**: `claude/research-improvements-jjI84`. Last commit:
  `4b25269`.

## Paste-ready PI block (copy below)

```
Session continues from R7. PRIMARY is R7.1 LB 0.95389 (K=13 +
Path-B with DriverClass × Stint segmentation τ=100k). Top-5% gap
1.6 bp; 13 days left.

Round 7 found that Path-B SEGMENTATION choice is a new lift axis:
DriverClass × Stint (named-vs-D0XX × Stint = 12 segments) beat the
default Compound × Stint by +0.106 bp OOF / +0.02 bp LB. The
named-driver pit-rate differential (32-43% vs 16-22% for anonymous
D0XX) is captured by driver-class segmentation; default Compound ×
Stint missed it.

Round 7 closed: swap-noise DAE absorbs at meta (-0.09 to -0.15 bp
across 4 K=14+Path-B configs); embedding-class diversity didn't
help K=11 pool.

Priority queue (HANDOVER.md "Next-session first actions" has detail):

1. More Path-B segmentations on K=13 pool via
   `scripts/build_K13_pathb_multiseg.py`. Add segmentation builders
   for: Driver-tier × Stint (pit-rate-quartile × stint), Race-cluster
   × Stint (high-pit-rate races vs low × stint), Year × Stint
   (4 × 6 = 24), Compound × first-pit-window. ~6 min CPU each.
   P(one lifts ≥+0.05 bp LB) ≈ 25%.

2. Multi-segmentation Path-B rank-blend — combine 3+ Path-B variants.
   Different sub-population variance should stack. P ≈ 30%.

3. C1 OpenF1 per-Race scalar join (~45 min CPU; 1.4% match cap
   bounded). P ≈ 15%.

4. DAE v2 on Kaggle T4: deeper bottleneck (64 dim), masked-column
   pretraining (BERT-style), contrastive loss. v1 absorbed at meta.
   P ≈ 20%.

5. Public-notebook scan (Rule 22; 17 days overdue). Use authenticated
   `kaggle kernels list -c playground-series-s6e5 --sort-by voteCount`
   per the kaggle-pages-recaptcha-gated friction.

Session-start protocol (Rule 32):
1. `git fetch origin && git log HEAD..origin/main && git diff
   HEAD..origin/main HANDOVER.md`
2. Load `audit/2026-05-18-round-7-execution.md` for R7 detail,
   `audit/2026-05-18-postmortem-research-improvements-jjI84-r6-r7.md`
   for postmortem context.
3. Load `state/current.md` and `state/mechanism-ledger.md` for
   ladder + closed-mechanism state.

Two frictions logged in R7 but NOT promoted (held for more data):
- two-axis-operator-sweep-missed (R7 segmentation finding)
- fold-bag-quantize-on-public-LB (R6.1 + R7.2 OOF→LB pattern)

Hedges held for final-window R7d:
- R7.2 5-seed fold-bag (LB 0.95389, structurally distinct from R7.1)
- K=27+Path-B τ=100k (LB 0.95368)
- R5.2 K=13+Path-B Compound×Stint (LB 0.95387)

Go.
```

## Friction-fed open questions for next session

- Will more Path-B segmentations also lift, or is DriverClass ×
  Stint a singleton win? (Answers G1 promotion question.)
- Will R7.2's OOF +0.264 bp register on private LB? (Answers G2
  promotion question.) — only knowable at comp-close.

## Quick-start checklist

- [ ] `git fetch origin && git status` (Rule 32)
- [ ] Read `audit/2026-05-18-postmortem-research-improvements-jjI84-r6-r7.md`
- [ ] Read `state/current.md`
- [ ] Confirm 10 fresh submission slots available
- [ ] Pick #1 from priority queue or follow new PI directive
