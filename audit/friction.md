# Friction log

One-liners. Distilled weekly per `~/.claude/skills/kaggle-comp/self-improvement.md`.

## 2026-05-04

- `tag: stats-error` — Pre-baseline gate audit reported "PitStop ↔
  PitNextLap match rate 0.724 → strong structural relationship".
  Wrong: independent-baseline match rate at priors 0.136 and 0.199
  is 0.719. Observed 0.724 ≈ chance. U2 single-feature OOF AUC for
  `lead_PitStop` is 0.512 (basically random). Correction: don't
  flag a "match rate" as a structural finding without comparing
  against the independent-baseline expectation. Add to
  pre-baseline-gate.md item 2 ("schema check") a step:
  "for any binary-vs-binary correlation claim, report observed vs.
  independent-baseline match rate; the EXCESS is the signal."

- `tag: cv-anchor-context` — Auto R1 verdict ("gap >50bp ⇒ leakage")
  fired on baseline_two_anchor (gap 200bp), but that conclusion was
  wrong given U3 (test is i.i.d. row split). R1's rule needs a
  qualifier: "leakage" interpretation requires that the test set's
  generalisation regime matches anchor B; if test is i.i.d. row
  split (verifiable via U3-style alt-ratio probe), anchor A is the
  LB proxy and the gap is in-stratum signal, not leakage. Fix:
  update metric_notes default in pre-baseline-gate.md to require
  U3-equivalent split-structure check before interpreting R1 gap.
