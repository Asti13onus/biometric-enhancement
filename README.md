# Restoring Worn Fingerprints

Master's thesis project: **enhancement of fingerprints degraded by occupational wear, ageing
and skin damage**, for large-scale civil identity systems.

Most fingerprint enhancement research targets *latent* prints — crime-scene marks where the
ridge signal is present but occluded by background noise. This project targets a physically
different problem: prints where the ridge signal has been **attenuated at the source** by
manual labour, ageing, dryness or scarring. The failure mode is not a missed minutia in a
forensic lab; it is a failure to enrol at a ration shop, and exclusion from a service.

## Contributions

1. **WDM** — a physiologically grounded wear-degradation model, so paired training data can
   be synthesised that actually resembles the target population. Realism is *validated*
   (NFIQ 2 distribution tests, domain classifier, transfer test), not asserted.
2. **A reproducible, matcher-centric benchmark on obtainable data** — the field's principal
   benchmark (NIST SD27) has been withdrawn since 2017 yet is still reported in 2026 papers.
   This protocol uses only data a researcher can actually get, and adds two metrics the
   literature omits: failure-to-acquire reduction, and demographic stratification.
3. **Abstaining enhancement** — a per-pixel confidence head that declines to reconstruct
   ridges where the evidence is insufficient, then filters uncertain minutiae before
   matching. Tests a falsifiable claim: *discarding uncertain minutiae improves EER more than
   hallucinating them.*
4. *(reserve)* **Multi-impression fusion** — operational deployments already recapture up to
   three times; nothing in the literature exploits it.

## Documents

| File | What it is |
|---|---|
| [`STRATEGY.md`](STRATEGY.md) | Engineering strategy: framing, build order, phases, risks |
| [`fingerprint_enhancement_thesis_plan.md`](fingerprint_enhancement_thesis_plan.md) | Research plan: datasets, literature map, contributions |
| [`literature_review_fingerprint_enhancement.md`](literature_review_fingerprint_enhancement.md) | Full literature review, six eras, identified gaps |
| [`fingerprint_enhancement.bib`](fingerprint_enhancement.bib) | Bibliography |
| [`docs/datasets/REGISTRY.md`](docs/datasets/REGISTRY.md) | Dataset access tracker — the critical path |
| [`docs/decisions/`](docs/decisions/) | Architecture decision records |
| [`docs/progress/LOG.md`](docs/progress/LOG.md) | Session log |

## Layout

```
src/fpe/        the fpe package: data, degradation, models, metrics, eval
configs/        one YAML per experiment
experiments/    thin entrypoints
results/        append-only run registry + generated figures
docs/           decisions, dataset registry, literature notes, progress
scripts/        setup, download, table/figure generation
paper/          thesis and paper sources
data/           gitignored — fingerprint images are never committed
```

## Status

Phase 0 — setting up the evaluation harness and unblocking dataset access.
See `STRATEGY.md` §4 for phase gates.

## Data and licensing

No fingerprint imagery is contained in this repository. Several datasets used
(IAB Rubric, NIST, CASIA, FVC "A" subsets, LivDet) are distributed under licences that
forbid redistribution; only manifests, checksums and split definitions are versioned here.
