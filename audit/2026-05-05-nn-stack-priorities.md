# NN-stack priorities — Day-4 (2026-05-05)

PI-noted ideas to fold into HANDOVER on next "prepare handover".
Strategic anchor: M5q's 10× OOF→LB amplification on RealMLP shows
NN-family bases carry LB EV the OOF metric understates. Pursue
NN-family DIVERSITY > NN tuning per se.

## Priority order (highest EV first)

1. **Multi-seed RealMLP bag** — re-run RealMLP-TD with seeds 123 + 456,
   rank-average the 3 OOF/test. Compounds variance-reduction across
   seeds at modest GPU cost. Prior: +1-3bp on top of M5q's +14bp.
   Cheaper and lower-variance than an Optuna sweep at the same
   GPU-hour budget.

2. **TabNet on Kaggle GPU** — a structurally different NN family from
   RealMLP (attention-based feature selection). 1-fold smoke FIRST
   per Rule 2 (Day-3 RealMLP burned 175min by skipping smoke).
   Different inductive bias = orthogonality candidate, not just
   another seed.

3. **NN-with-TE smoke** — single-fold probe of RealMLP fed
   target-encoded versions of (Driver, Race) on top of native
   features. Tests whether NN can extract incremental signal from
   TE that the stacker-level TE bases (d2a, d3a) don't already
   absorb. Smoke-only before committing — risk: double-counts TE
   already in the M5q stack.

## Explicitly NOT recommended at this stage

- **Optuna sweep on RealMLP-TD.** PyTabKit ships well-tuned defaults;
  sweeps typically yield <1bp on top while burning 5-10× the GPU
  hours of a seed bag. Re-evaluate only if seed bag underperforms
  the +1-3bp prior.

- **Hand-crafted FE specifically for the NN branch.** RealMLP-TD's
  internal numerical embeddings and cat embeddings re-derive most
  hand-crafted features. d3a/d3b already showed FE on the GBDT
  pool was absorbed at the stack level (null lift). Likely same
  outcome here.

End — 33 lines.
