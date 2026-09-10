# ADR 0002 — Build the evaluation harness before any model

- **Date:** 2026-09-10
- **Status:** accepted

## Context

The thesis plan states that the evaluation protocol must be fixed before training anything,
but schedules the benchmark itself for weeks 25–30. Those two statements conflict. The late
schedule also concentrates six weeks of measurement work at the point in the calendar where
slippage is least recoverable.

Separately, the literature review finds that inconsistent evaluation is itself a finding of
the field, and that two metrics relevant to the deployment problem — failure-to-acquire
reduction and demographic stratification — are effectively absent. A benchmark on obtainable
data is one of the thesis's stated contributions (Contribution 3). Contributions should not
be scheduled last.

## Decision

The first code written is the evaluation harness:

1. Dataset adapters that produce manifests with checksums.
2. Metric implementations: NFIQ 2 wrapper, minutiae precision/recall/F1 under Cappelli's
   protocol (τ_D = 14 px, τ_θ = π/9, type-exact and type-agnostic), matcher wrappers
   (`bozorth3`, SourceAFIS) reporting EER / TAR@FAR / rank-1, failure-to-acquire rate.
3. An append-only results registry: `results/registry/runs.jsonl`, one row per evaluation,
   recording git SHA, config hash, dataset manifest hash, metrics and timestamp.
4. Table and figure generation from that registry.

No metric is ever reported that a script did not produce. Every experiment from Phase 1
onward writes to the registry.

## Consequences

- Phase 0 is longer and produces no research result — only a working measurement pipeline.
- Contribution 3 accumulates continuously instead of being built under deadline.
- Results chapter becomes a query over the registry.
- Cross-phase comparisons stay valid because the protocol did not drift.
- Any protocol change after Phase 1 requires an ADR and re-running affected rows; the
  registry makes the affected set identifiable.

## Alternatives rejected

- **Follow the plan's ordering** (harness in Phase 4): risks protocol drift across phases and
  a late crunch; earlier numbers become incomparable.
- **Use PSNR/SSIM early as a cheap proxy, matchers later**: rejected. Pixel metrics measure
  fidelity, not identity information, and would silently reward minutiae-destructive
  enhancement for months.
