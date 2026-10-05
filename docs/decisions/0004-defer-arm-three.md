# ADR 0004 — Defer arm three (unpaired translation) to future work

- **Date:** 2026-10-01
- **Status:** accepted

## Context

The thesis frames a controlled comparison of the three literature responses to the
synthetic/real degradation mismatch (STRATEGY.md §1): (a) corrected synthesis, (b)
post-hoc domain alignment, (c) unpaired translation. Arms one and two are complete on a
common harness (sessions 10–11): alignment beats corrected synthesis with disjoint 95%
confidence intervals (EER 0.1105 vs 0.1353 on FVC2004), and neither beats the unenhanced
control (0.0991) on this population.

Arm three is the costliest remaining experiment. Unpaired translation requires an
adversarial setup — two generators and two discriminators — which (i) does not fit the
≤10M-parameter, 4 GB-GPU budget without a scope-reducing redesign that would itself need
an ADR, (ii) is the arm the literature review already identifies as most exposed to the
spurious-minutiae failure that arms one and two measured, and (iii) competes for the
remaining semester with U9–U12 — the minutiae-precision protocol, the precision–coverage
curve, ablations, a second matcher, and the full multi-dataset benchmark — which generate
the thesis's own tables, figures and claims from models that already exist.

## Decision

Arm three is deferred to future work (next semester). This semester completes U9–U12 and
the write-up on the two finished arms. The thesis reports the comparison as two of the
three responses measured head-to-head, states the deferral explicitly, and carries arm
three in the future-work chapter alongside the two requirements named in session 10
(pair-consistency across real impressions; degradation models validated against real
cross-impression variation).

## Consequences

- The comparison claim must be worded as "corrected synthesis versus post-hoc alignment",
  not "all three responses" — the gap the literature review names (no three-way
  comparison exists) is narrowed, not fully closed.
- The harness, pseudo-annotation pipeline and paired-training machinery built for arms
  one and two are the infrastructure arm three needs; next semester starts from a working
  benchmark rather than from zero.
- Freed time goes to U9–U12, where every result strengthens claims already in hand.
