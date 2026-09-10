# Dataset Registry

Single source of truth for dataset access. Update the **Status** and **Last action** columns
every time anything happens — this table is what tells us whether the critical path is clear.

Status values: `not started` · `requested <date>` · `chasing` · `granted` · `downloaded` ·
`manifested` · `refused` · `unavailable`

**Images are never committed to this repo.** Data lives under `data/` (gitignored). Only
manifests (`docs/datasets/manifests/*.csv`: relative path, sha256, width, height, dpi) and
split files are versioned.

---

## Tier 0 — Download today, no paperwork

| Dataset | Role | Status | Last action | Notes |
|---|---|---|---|---|
| ChaLearn LAP Track 3 | Primary paired training set (84k pairs, 275×400) | **needs account** | 2026-09-10: download is behind ChaLearn LAP site registration | `chalearnlap.cvc.uab.cat/dataset/32/description/`. Its degradation is a *latent* model — that mismatch is our Gap 1. |
| SOCOFing (+ Altered) | Damage severity benchmark | **needs Kaggle token** | 2026-09-10 | Kaggle `ruizgara/socofing`. **Caveat:** declared 500 dpi but actually ~96×103 px at ~200 dpi. Rescale before feeding 500-dpi models; never compare NFIQ2 across resolutions. |
| FVC2000/2002/2004 "B" subsets | Low-quality real test data | **manifested** | 2026-09-10: 960 imgs, 134 MB, all 12 DBs, dims match spec | `bias.csr.unibo.it/fvc2000/download.asp` (also fvc2002, fvc2004). 10 fingers × 8 per DB. FVC2004 DB1_B is the Phase 0 smoke-test set. |
| Neurotechnology samples | Extra real prints | **manifested** | 2026-09-10: CrossMatch 408 + U.are.U 520 imgs, 102 MB | CrossMatch 51×8, U.are.U 65×8. `neurotechnology.com/download.html#databases` |
| MINEX validation imagery | Extra real prints, permissive licence | **manifested** | 2026-09-10: 801 imgs, 114 MB, **+ per-image quality band and finger position** | GitHub `usnistgov/minex` |
| Tsinghua SD27 annotations | Orientation / frequency / skeleton ground truth | not started | — | `ivg.au.tsinghua.edu.cn/dataset/NIST.php`. Useful even without SD27 images. |
| FVC manual segmentation GT | Segmentation supervision | not started | — | Thai, Huckemann & Gottschlich, PLOS ONE 2016. What Cappelli trained on. |
| PrintsGAN | Synthetic pretraining (525k imgs, 35k fingers) | **agreement needed** | 2026-09-10: signed agreement to Steven Grosz (MSU) — not paperwork-free | Engelsma, Grosz & Jain (MSU) |
| L3-SF | Level-3 detail (pores, scratches) | **manifested** | 2026-09-10: 8,140 imgs, 569 MB; 740 with TSV pore coords. CC BY-NC-SA 4.0 | Wyzykowski et al. |
| Anguli | Unlimited paired synthesis | **downloaded** | 2026-09-10: Win binary + Qt source in `data/external/anguli/` | Open-source SFinGe reimplementation; generated ChaLearn |
| FVS | Small extra real set | **manifested** | 2026-09-10: 168 imgs (21 fingers × 8), 256×256, dpi unstated | `fvs.sourceforge.net`. Expired TLS cert — fetch with curl, not urllib |

## Tier 1 — Licensed, start the paperwork now (long lead time)

| Dataset | Role | Status | Last action | Notes |
|---|---|---|---|---|
| **Rural Indian Fingerprint DB** | ⭐ Confirmatory test set — the target population | not started | — | `databases@iab-rubric.org`. **Needs Registrar / Head-of-Institution signature** — begin internal paperwork day 1. Ask explicitly for the *public* one (a separate private rural DB exists via Prof. Phalguni Gupta, IIT-K). |
| **IIIT-D MOLF** | Cross-sensor Indian population, latent annotations | not started | — | Same IAB licence route. 19,200 images / 100 subjects. WSQ ~548 MB, BMP ~21 GB. |
| **MUST** | Minutiae + mask ground truth (enables F1) | not started | — | Same IAB route. ~21k impressions, manually marked minutiae, PPI, semantic masks. |
| **NIST SD302 (N2N)** | Multi-sensor real capture | not started | — | `nigos.nist.gov/datasets/sd302/request`. Institutional email; manual NIST review. |
| **NIST SD300** | Naturally poor / uncooperative captures | not started | — | Request form. 888 subjects × 10 fingers, ink cards. |
| **CASIA-FingerprintV5** | Free substitute for worn population (workers, waiters) | **needs registration** | 2026-09-10: free account at idealtest.org | `idealtest.org`, registration. Site availability historically unreliable — try early. |
| IIITD MSLFD | Surface-variation latents | not started | — | IAB route; low priority |
| LFIW | Post-SD27 latent replacement | not started | — | Request; low priority |

## Unavailable — do not plan around these

| Dataset | Status | Consequence |
|---|---|---|
| NIST SD27 | **Withdrawn Jan 2017** | The principal benchmark of the entire latent literature, including the 2026 SOTA, is unobtainable. Published numbers cannot be reproduced. This is a citable motivation for our alternative protocol. |
| NIST SD4 / SD9 / SD10 | Discontinued | Historical baselines unreproducible |
| NIST SD14 | Status uncertain — treat as unavailable | Was the 27k–30k training source for DNUNets, FingerGAN, SFP |

---

## Tooling

| Tool | Purpose | Status | Notes |
|---|---|---|---|
| NFIQ 2 | ISO/IEC 29794-4 quality score (0–100) | not installed | `github.com/usnistgov/NFIQ2`, prebuilt binaries |
| NBIS (`mindtct`, `bozorth3`) | Free minutiae extractor + matcher — baseline matcher #1 | not installed | NIST Biometric Image Software |
| SourceAFIS | Independent matcher #2 | not installed | Two independent matchers is a protocol requirement |
| pyfing | Cappelli's segmentation / orientation / frequency / GBFEN / SNFEN — strongest baseline **and** best starting codebase | not installed | `github.com/raffaele-cappelli/pyfing`. **Verify its DL backend in Phase 0**: if TensorFlow, native Windows GPU support ended at TF 2.10 → use WSL2. |
| CVxTz/fingerprint_denoising | ChaLearn-winning U-Net — trivial baseline | not installed | |
| MCC (Minutia Cylinder-Code) | Optional research matcher | not installed | SDK from Bologna, academic use |
