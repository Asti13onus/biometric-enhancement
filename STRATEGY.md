# Project Strategy

Master's thesis: **restoration of fingerprints degraded by occupational wear, ageing and
skin damage, for large-scale civil identity systems.**

This document is the executable strategy. `fingerprint_enhancement_thesis_plan.md` is the
research plan; the literature review is the evidence base. This file records the
*engineering* decisions: what we build, in what order, and how we avoid the failure modes
that kill master's theses.

---

## 1. Thesis framing (chosen)

The plan proposes four contributions. The literature review's **Gap 1** contains a sharper
framing that we adopt as the thesis spine:

> Three responses to the same problem — the mismatch between synthetic latent-print
> degradation and real worn-fingerprint degradation — exist in the literature:
> **(a) corrected synthesis**, **(b) post-hoc domain alignment** (Joshi et al.),
> **(c) unpaired translation** (Karabulut et al.). They have never been compared on a
> common benchmark.

So the thesis is a **controlled comparison study whose novel arm is a physiologically
grounded degradation model**, evaluated with a precision-first, matcher-centric protocol on
obtainable data.

Why this framing rather than "my model beats SNFEN":

| | "Beat the SOTA" framing | Comparison-study framing (chosen) |
|---|---|---|
| If our model wins | Thesis succeeds | Thesis succeeds, plus a comparison nobody has published |
| If our model loses | Thesis is in trouble | Still a complete, publishable result: the comparison *is* the contribution |
| Reviewer objection | "Cappelli already did this, better" | "This comparison is genuinely absent" — true, and citable from the review |

Cappelli's lab has a 25-year head start and 4.9M-parameter models that already beat every
GAN. Competing on architecture is a bad bet. Competing on **what you train on** and **how
you measure** is an open field.

### Contributions, in priority order

1. **WDM — wear degradation model.** Parametric, physiologically grounded degradation
   (ridge-amplitude attenuation, crease injection, dryness fragmentation, pressure-dependent
   partial contact). Validated for realism, not asserted. *Solves the data problem.*
2. **Benchmark + protocol on obtainable data.** Precision-first, two independent open
   matchers, NFIQ 2 distribution shift, **failure-to-acquire reduction**, stratified by
   severity and demographics. Released with split files and scripts.
3. **Abstaining enhancement.** Per-pixel confidence head; refuse to synthesise ridges where
   evidence is insufficient; propagate confidence to minutiae filtering.
   Falsifiable claim: *discarding uncertain minutiae improves EER more than hallucinating them.*
4. **(Fallback / bonus) Multi-impression fusion.** Cheap, deployable, under-studied. Held in
   reserve — see risk register.

### The thesis-defining figure

A **precision–coverage operating curve**: sweep the abstention threshold, plot minutiae
precision and matcher EER against the fraction of image area the model is willing to
reconstruct. Every existing method is a single point at coverage = 1.0. This figure is the
argument of the whole thesis in one plot, and it exists whether or not we win on EER.

---

## 2. Five strategic changes to the plan

### 2.1 Build the evaluation harness *first*, not in Phase 4

The plan says "fix the evaluation protocol before you train anything" and then schedules
benchmark completion for weeks 25–30. That is contradictory, and the late version is a
six-week crunch at the worst possible time.

Instead: the harness is the **first code written**. Every experiment from week 3 onward
writes a row to an append-only results registry. The benchmark contribution then accrues for
free, and the results chapter becomes a query rather than a re-run.

### 2.2 No phase may block on a licence

The plan's WDM realism validation needs the Rural Indian DB for 2 of its 3 checks (NFIQ 2
distribution comparison; transfer test). The IAB licence needs a **Registrar's signature**
and may take a month, arrive late, or be refused. If Phase 2's validation depends on it, one
email can stall the thesis.

Mitigation — every deliverable has a validation path on free data:

| Need | Primary (licensed) | Free substitute available day 1 |
|---|---|---|
| Real worn/degraded population | Rural Indian DB | CASIA-V5 (workers/waiters), FVC2004 DB1_B (dry/distorted), NIST SD300 (uncooperative captures) |
| Minutiae ground truth | MUST, MOLF | Tsinghua SD27 annotation package; manual annotation of a small set |
| Segmentation ground truth | — | Thai / Huckemann / Gottschlich manual FVC masks (free) |
| Damage labels + severity | — | SOCOFing-Altered (obliteration / z-cut / rotation × 3 levels) |
| Paired clean/degraded | — | ChaLearn (84k pairs), Anguli, PrintsGAN, our own WDM |

Rural Indian DB becomes the **confirmatory** test set, not the load-bearing one. If it
arrives, it is the headline external validation. If it does not, the thesis still stands.

### 2.3 Treat the hardware constraint as a design goal, not a limitation

Available GPU: **GTX 1650, 4 GB**. Cappelli's SNFEN is 4.9M parameters and trains in 25
minutes on a 3080 Ti — comfortably reachable here (expect roughly 2–3 h). A GAN or diffusion
model is not. This is convenient: the literature already concludes that small,
domain-knowledge-conditioned models win, and "runs on the hardware that actually exists in a
village PoS device" is a stronger thesis conclusion than a 0.3% EER delta on an A100.

Hard budget: **<= 10M parameters, <= 500 ms CPU inference, <= 4 GB training footprint.**
Written into every model config, reported in every results table.

### 2.4 Precision-first, everywhere

Cappelli's table shows the ranking is driven by precision (his best method wins at 0.236
precision while rivals have higher recall). Therefore:

- Headline metrics are **precision** and **EER** — never recall alone, never PSNR.
- Every model reports its spurious-minutiae count.
- Loss functions are asymmetric by default (Tversky, alpha = 0.7, penalising false ridges).

### 2.5 Reproducibility as infrastructure, not discipline

A thesis loses more time to "which config produced that number in the September slide?" than
to training. So:

- **Every** metric is produced by a script, never by hand, and lands in
  `results/registry/runs.jsonl` (append-only, one row per evaluation, carrying git SHA,
  config hash, dataset manifest hash and timestamp).
- Datasets are referenced by **manifest + checksum**, never by path. Manifests go in git;
  images never do.
- Decisions get an ADR in `docs/decisions/`. A thesis must *justify* choices in prose;
  writing the justification at decision time is far cheaper than reconstructing it in month 8.
- Thesis tables and figures are **generated** from the registry (`scripts/make_tables.py`).

---

## 3. Repository layout

```
src/fpe/                 the `fpe` package — all reusable code
  data/                  dataset adapters, manifests, splits, loaders
  degradation/           WDM (Contribution 1) + baseline ChaLearn-style degradation
  models/                baselines (Gabor/HWJ, U-Net, SNFEN repro) + ours
  metrics/               minutiae P/R/F1, NFIQ2 wrapper, matcher wrappers, FTA
  eval/                  the harness: protocol definitions + results registry
configs/                 one YAML per experiment; hashed into the registry
experiments/             thin entrypoints; no logic — logic lives in src/fpe
results/
  registry/runs.jsonl    append-only record of every evaluation ever run
  figures/               generated plots (committed — they go in the thesis)
docs/
  decisions/             ADRs — why we chose what we chose
  datasets/REGISTRY.md   every dataset: access status, licence, request date, checksum
  literature/            reading notes keyed to the .bib
  progress/              weekly log; raw material for the thesis narrative
scripts/                 setup, download, table/figure generation
paper/                   thesis + paper LaTeX
data/                    (gitignored) images live here, never in git
```

**Licence discipline:** IAB, NIST, CASIA, FVC-A and LivDet data may not be redistributed.
`.gitignore` blocks image formats by default. Never override it for dataset images.

---

## 4. Phase plan (engineering view)

Phases mirror the thesis plan; the deliverables here are code and registry rows.

| Phase | Weeks | Engineering deliverable | Gate to pass |
|---|---|---|---|
| **0** Unblock | 1–2 | Licence requests sent. NFIQ2 + NBIS + pyfing installed and verified. Free datasets downloaded and manifested. | `python -m experiments.baseline --dataset fvc2004_db1_b` writes an NFIQ2 + bozorth3 EER row to the registry |
| **1** Baselines | 3–8 | Review chapter drafted. Three baselines reproduced: hand-written Hong-Wan-Jain Gabor, ChaLearn U-Net, pyfing SNFEN. | Baseline table generated from the registry, not typed. Supervisor review. |
| **2** WDM | 9–14 | WDM implemented plus its three realism validations (KS test on NFIQ2 distributions, domain-classifier accuracy, transfer test). | The side-by-side figure: real / ChaLearn-degraded / WDM-degraded. Domain classifier near chance. |
| **3** Model | 15–24 | Orientation+frequency conditioned encoder-decoder, confidence head, abstention. Ablations: conditioning, abstention on/off, WDM vs ChaLearn training data, loss. | Precision–coverage curve produced. Beat SNFEN on precision at matched coverage. |
| **4** Benchmark | 25–30 | Full stratified evaluation, two matchers, every dataset obtained. Failure analysis. | Results chapter generated from the registry. |
| **5** Write-up | 31–36 | Thesis and paper (IWBF / IJCB / IET Biometrics). Public release of the WDM generator, splits and eval scripts. | Repo reproduces every number in the thesis from a clean clone plus data. |

Phases 2 and 3 overlap in practice: WDM generation is cheap once written, and the model needs
its output. Phase 1 baselines must be *finished* before Phase 3 starts, or there is nothing
to compare against.

---

## 5. Risk register (engineering)

| Risk | Severity | Mitigation | Trigger to act |
|---|---|---|---|
| IAB licence delayed or refused | High | §2.2 free substitutes; Rural Indian DB is confirmatory only | No reply by week 4 → supervisor-to-supervisor email to Prof. Vatsa / Prof. Singh |
| Cannot beat SNFEN on EER | Medium-high | Comparison-study framing (§1); precision-at-coverage is a different axis | End of week 20 → freeze the model, pivot effort to Contribution 4 (fusion) |
| WDM realism unconvincing (domain classifier well above chance) | Medium | Report it honestly; a *measured* realism failure is a finding, and the three-way comparison survives | Week 13 |
| 4 GB GPU insufficient | Low | Parameter budget (§2.3); 256×256 patches, mixed precision, gradient accumulation | Any config that OOMs is out of scope by definition |
| Windows toolchain friction (NBIS, TF-GPU) | Medium | Verify in Phase 0. Note: TensorFlow dropped native Windows GPU support after 2.10 — if pyfing needs TF-GPU, use WSL2. Our own models: PyTorch. | Week 1 |
| Scope creep into latent fingerprints | High | ADR required before any latent-only experiment. Latents are where the papers are; that is exactly the trap. | Every planning session |
| Numbers lost or unreproducible | Certain without process | §2.5 registry; nothing is reported that a script did not produce | — |

---

## 6. Non-goals

Stated explicitly so they can be declined without re-litigating:

- Latent (crime-scene) fingerprint enhancement as a primary target.
- Novel architectures for their own sake; anything above 10M parameters.
- Spoof / liveness detection, contactless capture pipelines, fingerphoto research.
- Beating VeriFinger. We evaluate with open matchers so results stay reproducible.
- Any use of test-set backgrounds or textures in training (Cappelli's leakage criticism).
