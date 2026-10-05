# ADR 0005 — Defer the full ablation grid (U10) to future work

- **Date:** 2026-10-01
- **Status:** accepted

## Context

U10 planned four train-and-evaluate ablation axes: wear versus latent degradation, joint
estimation versus frozen pyfing estimators, abstention on/off, and Tversky α = 0.7 versus
symmetric Dice. Since that plan was written, the session-10–12 variant ladder measured
most of what U10 was for, under the same registry discipline: the minutia-aware loss,
pair consistency, confidence blending and the abstention sweep were each evaluated as a
single-variable change on FVC2004, and abstention was additionally swept on the synthetic
test split (U9), answering the abstention axis outright. The remaining semester weeks are
better spent on the write-up of two completed arms, U9's figure, U11's significance
results and U12's benchmark than on retraining runs whose questions are largely answered.

## Decision

The full ablation grid is deferred to next semester, alongside arm three (ADR 0004). The
thesis presents the variant ladder as its component analysis, states this substitution
explicitly, and carries the one genuinely open axis — **wear-model versus latent-style
training degradation, everything else fixed** — as the first experiment of the future-work
plan, since it tests the thesis's founding premise most directly. Until it runs, the
premise is defended by U3 (the wear model measurably closer to real degradation than the
latent model on distributional and classifier tests) and by the arm comparison (corrected
synthesis bounded by its degradation model; session 10's closing diagnosis).

## Consequences

- The claim "the wear model causes the differences we report" may not be made; only the
  weaker, evidenced claims (realism per U3; training-strategy effects per the ladder).
- One examiner-facing gap is accepted knowingly and is named in the limitations chapter
  rather than discovered during the defense.
- `experiments/ablate.py` is not written this semester; `src/fpe/degradation/latent.py`
  remains in place for the deferred run.
