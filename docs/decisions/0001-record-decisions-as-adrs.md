# ADR 0001 — Record decisions as ADRs

- **Date:** 2026-09-10
- **Status:** accepted

## Context

A thesis is judged partly on whether its choices are *justified*, not merely made. Over nine
months the reasons behind a choice — why this loss, why this test set, why this threshold —
decay faster than the code. Reconstructing them during write-up is expensive and produces
weaker prose than writing them down at the moment of decision.

## Decision

Every non-obvious project decision gets a short numbered file in `docs/decisions/`:
context, decision, consequences, alternatives rejected. Kept short — half a page is normal.

A decision qualifies if a supervisor or reviewer could reasonably ask "why did you do it
that way?". Notably required for:

- choice of dataset, split, or evaluation threshold
- loss function and training protocol choices
- any experiment that touches latent (crime-scene) prints, which are out of scope per
  `STRATEGY.md` §6 and need explicit justification to enter

## Consequences

- Small ongoing cost per decision.
- The thesis methodology chapter is assembled from ADRs rather than recalled.
- Superseded ADRs stay in the repo marked `superseded by NNNN` — the record of a changed mind
  is itself thesis material.
