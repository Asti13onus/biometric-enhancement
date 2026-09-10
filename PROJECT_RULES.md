# Project instructions

Master's thesis on **enhancement of fingerprints degraded by occupational wear, ageing and
skin damage**, for large-scale civil identity systems (Aadhaar-style deployment, not forensics).

Read `STRATEGY.md` first. `fingerprint_enhancement_thesis_plan.md` is the research plan and
`literature_review_fingerprint_enhancement.md` is the evidence base; `fingerprint_enhancement.bib`
holds every citation.

## Non-negotiables

1. **Never commit fingerprint images.** IAB, NIST, CASIA, FVC-A and LivDet data may not be
   redistributed. Images live in `data/` (gitignored). Only manifests, checksums and split
   files are versioned. Do not override `.gitignore` for dataset images.
2. **Never report a number a script did not produce.** Every metric lands in
   `results/registry/runs.jsonl` with git SHA, config hash and dataset manifest hash.
   Thesis tables and figures are generated from that registry (`scripts/make_tables.py`).
3. **Precision over recall, matchers over pixels.** Headline metrics are minutiae precision
   and matcher EER. PSNR/SSIM appear only where clean ground truth exists (i.e. synthetic
   data) and never as a headline. Always report NFIQ 2 distribution shift plus at least one
   matcher's EER.
4. **No test-set leakage into training.** Backgrounds, textures and noise patterns must never
   come from test images — Cappelli's explicit criticism of DNUNets and FingerGAN. State
   compliance in the thesis.
5. **Parameter budget:** <= 10M parameters, <= 500 ms CPU inference, <= 4 GB training
   footprint (target GPU is a GTX 1650, 4 GB). Edge deployability is a stated design goal,
   not a limitation.
6. **Worn prints, not latents.** Latent (crime-scene) enhancement is out of scope. Any
   latent-only experiment requires an ADR justifying it first.

## Working conventions

- Logic lives in `src/fpe/`. `experiments/` holds thin entrypoints only.
- One YAML per experiment in `configs/`; the config is hashed into the registry row.
- Decisions worth a "why did you do it that way?" get an ADR in `docs/decisions/`.
- Update `docs/datasets/REGISTRY.md` whenever dataset access status changes — that table is
  the critical path.
- Append to `docs/progress/LOG.md` at the end of each working session. It is the raw material
  for the thesis narrative and the supervisor updates.
- Reading notes go in `docs/literature/`, keyed to the bib entry.

## Environment

- Windows 11, PowerShell primary; Bash tool available.
- Python 3.11 (`C:\Users\91807\AppData\Local\Programs\Python\Python311`), no conda.
- GPU: NVIDIA GTX 1650, 4 GB. Use PyTorch for our own models.
- TensorFlow dropped native Windows GPU support after 2.10 — if `pyfing` requires TF-GPU,
  run it under WSL2 rather than fighting the native install.
- External binaries (NFIQ 2, NBIS) go in `tools/bin/` or `vendor/`, both gitignored.
