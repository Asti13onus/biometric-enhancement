# Download Plan — Open / Freely Available Datasets

Verified 2026-09-10 against [robertvazan/fingerprint-datasets](https://github.com/robertvazan/fingerprint-datasets)
(the curated community index), the ChaLearn LAP dataset page, and the primary source pages
for each generator. Nothing is downloaded until approved.

Target root: `E:\biometric-enhancement\data\` (476 GB free — ample).
`data/` is gitignored. Only manifests (`docs/datasets/manifests/*.csv`) and split files
are committed.

---

## Group A — Direct download, no account, no agreement

Fully automatable in one script run.

| # | Dataset | Contents | Resolution | Format | Est. size | Licence | Source |
|---|---|---|---|---|---|---|---|
| A1 | **FVC2000 DB1–DB4 "B"** | 4 × (10 fingers × 8) = 320 imgs | 300×300 / 256×364 / 448×478 / 240×320, 500 dpi | TIFF | ~40 MB | Unspecified, free | `bias.csr.unibo.it/fvc2000/Downloads/DB{1..4}_B.zip` |
| A2 | **FVC2002 DB1–DB4 "B"** | 320 imgs | 388×374 / 296×560 / 300×300 / 288×384, 500–569 dpi | TIFF | ~40 MB | Unspecified, free | `bias.csr.unibo.it/fvc2002/Downloads/DB{1..4}_B.zip` |
| A3 | **FVC2004 DB1–DB4 "B"** | 320 imgs | 640×480 / 328×364 / 300×480 / 288×384, 500–512 dpi | TIFF | ~60 MB | Unspecified, free | `bias.csr.unibo.it/fvc2004/Downloads/DB{1..4}_B.zip` |
| A4 | **Neurotechnology CrossMatch** | 51 fingers × 8 = 408 imgs | 504×480, 500 dpi | TIFF | ~100 MB | Unspecified, free | `neurotechnology.com/download/CrossMatch_Sample_DB.zip` |
| A5 | **Neurotechnology U.are.U** | 65 fingers × 8 = 520 imgs | 326×357, 512 dpi | TIFF | ~60 MB | Unspecified, free | `neurotechnology.com/download/UareU_sample_DB.zip` |
| A6 | **FVS** | 21 fingers × 8 = 168 imgs | 256×256, dpi unknown | BMP | ~11 MB | Unspecified, free | `fvs.sourceforge.net/fingerprint_bitmaps.zip` |
| A7 | **MINEX validation imagery** | 50 subjects × 10 fingers × 2 = 1,000 imgs | 500 dpi | raw grey + metadata | ~200 MB | Permissive | GitHub `usnistgov/minex` → `minexiii/validation/validation_imagery_raw` |
| A8 | **L3-SF** (level-3 synthetic) | 5 subsets × 148 identities × 10 = 7,400 imgs; 740 with **sweat-pore annotations** at 1200 dpi | up to 1200 dpi | — | ~5–10 GB | Research use | `github.com/andrewyzy/L3-SF` → Google Drive (password published on the project page) |
| A9 | **Anguli** generator | Unlimited paired synthesis (clean + degraded), C++/Qt/OpenCV | any | — | < 100 MB | Open source | `dsl.cds.iisc.ac.in/projects/Anguli` |

**Group A total: ~6–11 GB**, dominated by L3-SF.

### Roles
- **A1–A3 (FVC "B")** — real low-quality test data. FVC2002/2004 subjects were *instructed* to
  produce difficult impressions (wet/dry fingers, distortion, rotation). FVC2004 DB1_B is the
  Phase 0 smoke-test set.
- **A4–A7** — additional real prints across sensors; cheap cross-sensor variation.
- **A8 L3-SF** — the only free source of level-3 detail (pores, incipient ridges). Directly
  useful for the WDM, since ridge-profile flattening is a level-3 phenomenon.
- **A9 Anguli** — unlimited clean masters to which the WDM will be applied. This is the engine
  behind Contribution 1, and the same tool that generated ChaLearn.

---

## Group B — Free, but requires an account / login

| # | Dataset | Contents | Est. size | Licence | Access |
|---|---|---|---|---|---|
| B1 | **ChaLearn LAP Track 3** ⭐ | 168,000 imgs = 84,000 **paired** (clean, degraded); train 75,600 pairs; synthetic test 8,400; real test 140 fingers × 12 | ~5–15 GB (unconfirmed) | Not stated on the page | `chalearnlap.cvc.uab.cat/dataset/32/description/` — download link is behind site registration; **to confirm at download time** |
| B2 | **SOCOFing** | 6,000 real (600 subjects × 10 fingers) **+ ~17,900 synthetically altered** (obliteration / central rotation / z-cut × Easy/Medium/Hard) | ~0.5–1 GB | **Non-commercial research only** | Kaggle `ruizgara/socofing` — needs a Kaggle account + API token |
| B3 | **CASIA-FingerprintV5** | 500 subjects × 8 fingers × 5 = 20,000 imgs, 328×356 @ 512 dpi, BMP | ~2.5 GB | **Prohibits publishing and redistribution** | `idealtest.org` — free registration. Site availability historically unreliable; try early |

### Roles
- **B1 ChaLearn** — the primary paired training set and the *baseline* degradation model we
  argue against. Non-negotiable: without it there is no comparison arm.
- **B2 SOCOFing-Altered** — the closest public thing to a labelled damaged-fingerprint
  benchmark, with three severity levels. Our severity-stratification axis.
- **B3 CASIA-V5** — the free stand-in for a worn population: the subject pool explicitly
  includes **workers and waiters**, and subjects were asked to give difficult impressions.
  This is what protects Phase 2 if the Rural Indian licence is slow.

---

## Group C — Free, but a signed agreement must be emailed

Researcher/supervisor signature only — no Registrar involved, so days not weeks.

| # | Dataset | Contents | Est. size | Access |
|---|---|---|---|---|
| C1 | **PrintsGAN** | 525,000 imgs — 35,000 distinct fingers × 15 impressions | ~20–40 GB | `biometrics.cse.msu.edu/Publications/Databases/MSU_PrintsGAN/` — print, sign and email the agreement to Steven Grosz |
| C2 | **Tsinghua SD27 annotation package** | Manual orientation fields, period/frequency maps, skeletons, singular points, ROIs **for** SD27 | small | `ivg.au.tsinghua.edu.cn/dataset/NIST.php` |
| C3 | **FVC manual segmentation GT** | Hand-marked segmentation masks for FVC images (Thai, Huckemann & Gottschlich, PLOS ONE 2016) | small | Hosted by the Göttingen group; exact URL to be located — otherwise email the authors |

### Roles
- **C1 PrintsGAN** — multi-impression synthetic pretraining, and the only large source of
  *multiple impressions per identity*, which Contribution 4 (fusion) needs.
- **C2 + C3** — ground-truth orientation, frequency, skeleton and mask supervision. These are
  what Cappelli's SNFEN was actually trained on. Without them we cannot reproduce his
  training setup, only his inference.

---

## Group D — Institutional paperwork (separate track, not this batch)

Tracked in `REGISTRY.md`. Rural Indian DB, IIIT-D MOLF, MUST, MSLFD (IAB — needs
**Registrar's signature**); NIST SD300 / SD301 / SD302 (request form, institutional email);
FVC2006 (2-year licence via ATVS); LivDet (registration).

---

## Tools to install alongside

| Tool | Purpose | Source |
|---|---|---|
| **NFIQ 2** | ISO/IEC 29794-4 quality score 0–100; prebuilt Windows binaries | `github.com/usnistgov/NFIQ2` |
| **NBIS** (`mindtct`, `bozorth3`) | Free minutiae extractor + matcher — baseline matcher #1 | NIST |
| **SourceAFIS** | Independent matcher #2 (protocol requires two) | open source |
| **pyfing** | Cappelli's segmentation / orientation / frequency / GBFEN / SNFEN — strongest baseline and best starting codebase | `github.com/raffaele-cappelli/pyfing` |
| **CVxTz/fingerprint_denoising** | ChaLearn-winning U-Net — trivial baseline to reproduce | GitHub |

---

## Four corrections to the thesis plan

1. **PrintsGAN is not paperwork-free.** The plan lists it as "released publicly"; it needs a
   signed agreement emailed to Steven Grosz. Still fast, but it belongs in Group C, and the
   request should go out with the others rather than being assumed instant.
2. **L3-SF is bigger than stated.** The plan says "740+ high-resolution prints". It is
   actually **7,400 images** (148 identities × 10 samples × 5 subsets); the 740 figure is the
   pore-annotated subset at 1200 dpi.
3. **SOCOFing's value is the Altered subsets, not the 6,000 real prints.** ~17,900 altered
   images across three damage types and three severity levels — that is the severity axis for
   the stratified benchmark. Licence is non-commercial research only.
4. **Exactly reproducing SNFEN's training needs the FVC "A" subsets, which are not free.**
   Cappelli trained on 100 images each from FVC2002 DB2_A, FVC2002 DB3_A and FVC2004 DB1_A,
   plus 60 from the FFE benchmark. The "A" subsets ship with the *Handbook of Fingerprint
   Recognition* (~€140), which also happens to be the standard reference we need anyway.
   **Recommendation: ask the department to buy the book.** Without it we can run pyfing's
   released weights but cannot retrain SNFEN on its original data, which weakens the
   "same-data comparison" claim in Phase 1.

---

## Proposed on-disk layout

```
data/
  raw/                     exactly as downloaded, never modified
    fvc2000/{db1_b,...}/
    fvc2002/...  fvc2004/...
    neurotech/{crossmatch,uareu}/
    fvs/  minex/  l3sf/
    chalearn/{train,val,test}/
    socofing/{real,altered_easy,altered_medium,altered_hard}/
    casia_v5/
    printsgan/
  interim/                 rescaled / normalised / patch caches
  processed/               WDM-generated corpora
  external/                Anguli binaries + generator configs
docs/datasets/manifests/   <dataset>.csv — relative path, sha256, w, h, dpi  (committed)
```

Every download is followed by a manifest pass: sha256 per file, dimensions, declared dpi.
The manifest hash goes into every results-registry row, so a dataset cannot silently change
underneath a published number.

**Resolution warning to encode in the loaders:** declared dpi and actual dpi disagree across
these sets (SOCOFing declares 500 dpi at ~96×103 px, i.e. roughly 200 dpi; FVC2002 DB2_B is
569 dpi; FVC2004 DB2/DB3_B are 512 dpi; L3-SF is 1200 dpi). Never feed mixed dpi to a
500-dpi-trained model, and never compare NFIQ 2 across resolutions.
