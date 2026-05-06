# Open questions for the PI

Questions Claude has surfaced while transcribing. Each links back to the
thought entry that prompted it. PI answers when convenient; answers get
folded back into the source entry or into a `concepts/` distillation.

## From [2026-05-06-process-and-framework.md](../thoughts/2026-05-06-process-and-framework.md)

### Q1. "Counter regressions" — transcription unclear
You said *"link regressions or maybe counter regressions, what works
best"* when describing day-job models. Likely candidates:
- **kernel regressions** (Nadaraya-Watson, etc.)
- **quantile regressions** (typical for price/risk forecasting)
- **linear regressions** (you also said "link", maybe "linear")
- something else?

Which family actually works in the electricity-price work? The answer
shapes what transfers to Kaggle and what doesn't.

### Q2. "Trulikki experimentation culture"
You said: *"there's a lot of literature regarding experimentation culture
in Trulikki."* Almost certainly a transcription artifact. Best guesses:
- **Tukey** (John W. Tukey — EDA, multiple-comparison work).
- A specific company / lab name?
- Something else.

Worth nailing because the literature reference dictates which
multiple-testing / false-discovery toolkit we adopt.

### Q3. "Professional or not professional?"
In [kickoff](../thoughts/2026-05-06-kickoff.md) you first said
*"professional in machine learning practitioner"* and then immediately
said *"my task is complex"* (suggesting *not* a pro). I logged the
non-pro framing. Confirm?

## Critical questions (Claude pushing back)

### Q4. Where does the framework actually break today?
You described friction in *communication* and *common ground for
decisions* across parallel branches/machines. But the existing
`CLAUDE.md` already has:
- Rule 15 handover protocol,
- Rule 18 ISSUES.md claim board,
- Rule 14 strategy-critic loop,
- Rule 19 audit-on-null discipline.

**Concrete question:** can you give one or two real recent examples
where two parallel agents either (a) duplicated work, (b) reached
contradictory conclusions, or (c) couldn't trust the other's audit?
Without concrete cases, "friction" stays abstract and the fix stays
abstract.

### Q5. What's the unit of "transfer" you actually want?
For Kaggle → day-job transfer, three plausible units:
1. **Tooling** (probe.py, gating, BOTE harness) → ports as code.
2. **Process** (bulletproof-7, audit discipline, ISSUES.md) → ports as
   convention.
3. **Mental models** (when to trust an OOF, what a leakage signature
   looks like) → ports as PI judgement.

Which of these matters most to you, and why? (This shapes what we
extract into `concepts/` vs. leave in `thoughts/`.)

### Q6. The synthetic-data critique cuts both ways
You said Kaggle synthetic data limits physical intuition, which makes
feature engineering harder. Counter-question: **does that make Kaggle
a better or worse benchmark for your day-job process?**

- **Better**: forces you to lean on systematic search rather than
  intuition you wouldn't have in a new domain anyway.
- **Worse**: trains habits (heavy stacking, leakage-blocking on
  synthetic noise) that don't translate to physical-law-rich settings.

I don't know the answer. You probably have a strong prior — worth
writing it down before more competitions accumulate evidence.
