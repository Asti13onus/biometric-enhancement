# Fingerprint Datasets — Acquired and Unavailable

**Enhancement of fingerprints degraded by occupational wear, ageing and skin damage**
Master's thesis · Astitva Srivastava · 11 September 2026

A record of which datasets have been obtained and which have not. Image counts and sizes
are measured from per-image manifests (`docs/datasets/manifests/`), not quoted from the
publishers. Live access status is tracked in `docs/datasets/REGISTRY.md`.

**Summary: 10 datasets acquired, 38,973 images, 2.2 GB.** One further set (SOCOFing) was
downloaded and has been excluded — see §3. Eight datasets are pending; five are permanently
unavailable.

---

## 1. Acquired — in use

| Dataset | Images | MB | Image size | dpi | Structure |
|---|---|---|---|---|---|
| Anguli corpus (generated here) | 22,336 | 654 | 275×400 | 500 | 5,584 fingers × 4 |
| L3-SF | 8,140 | 569 | 320×240 / 512×512 | 1200 (partial) | 740 identities |
| DTD textures | 5,640 | 633 | variable | n/a | backgrounds, not prints |
| MINEX validation | 801 | 114 | 167×325 – 406×662 | 500 | 54 subjects |
| Neurotechnology U.are.U | 520 | 62 | 326×357 | 512 | 65 fingers × 8 |
| Neurotechnology CrossMatch | 408 | 39 | 504×480 | 500 | 51 fingers × 8 |
| FVC2004 "B" (4 databases) | 320 | 55 | 288×384 – 640×480 | 500 / 512 | 40 fingers × 8 |
| FVC2002 "B" (4 databases) | 320 | 41 | 300×300 – 296×560 | 500 / 569 | 40 fingers × 8 |
| FVC2000 "B" (4 databases) | 320 | 38 | 240×320 – 448×478 | 500 | 40 fingers × 8 |
| FVS | 168 | 33 | 256×256 | unstated | 21 fingers × 8 |

### FVC2000 / FVC2002 / FVC2004 — "B" subsets

**http://bias.csr.unibo.it/fvc2000/download.asp** (also `/fvc2002/`, `/fvc2004/`)
University of Bologna. Free for research; the larger "A" subsets ship only with the
*Handbook of Fingerprint Recognition*.

Three biennial fingerprint verification competitions, four sensor databases each — optical,
capacitive, thermal sweep, and synthetic. All twelve "B" databases held: 960 images, 10
fingers × 8 impressions each. FVC2002 and FVC2004 deliberately instructed subjects to give
*difficult* impressions, including wet and dry fingers, which makes **FVC2004 DB1 the closest
freely available analogue to occupationally worn prints.** This is the most widely reported
benchmark in the enhancement literature and the primary real-data evaluation set here.

### Neurotechnology sample databases

**https://www.neurotechnology.com/download.html#databases**
Free vendor evaluation data.

Two clean, consistent optical captures: CrossMatch Verifier 300 (51 fingers × 8) and
U.are.U 4000 (65 fingers × 8). Eight impressions per finger gives 1,428 and 1,820 genuine
comparison pairs respectively — the largest genuine-pair counts in the corpus, and therefore
the tightest error-rate estimates. Used as the clean control condition: a method that damages
good images fails regardless of how it performs on poor ones.

### MINEX validation imagery

**https://github.com/usnistgov/minex**
NIST, public domain.

801 raw images from NIST's MINEX III template-conformance programme, 54 subjects. The images
ship without metadata, but the validation driver's source header carries width, height,
**quality band and finger position** for all 801, which we parse. That gives a finger-position
and quality stratification axis — 8 positions, 132 images labelled poor or fair — on
permissively licensed data.

### Anguli-generated paired corpus (generated locally)

**https://dsl.cds.iisc.ac.in/projects/Anguli/** (generator)
Ansari, *Generation of Synthetic Fingerprint Databases*, MSc thesis, IISc Bangalore, 2011.
Generator free for research; the corpus is our own output and carries no restriction.

5,584 synthetic fingers, each with one clean master plus three simulated impressions —
22,336 images at 275×400. **The training set.** Supervised enhancement needs clean/degraded
pairs, and this is the only source of perfectly paired data available; the clean master is
the reconstruction target and our own degradation code produces the input. Requested at
20,000 fingers, interrupted at 5,584 by a memory exhaustion; the directory is named
`anguli_dev_5584` so the size is not mistaken. 5,584 is ample — the 2026 state of the art
was trained on 360 images.

### DTD — Describable Textures Dataset

**https://www.robots.ox.ac.uk/~vgg/data/dtd/**
Cimpoi et al., *Describing Textures in the Wild*, CVPR 2014. Free for research.

5,640 texture photographs in 47 categories. Not fingerprint data — this is the
surface-background bank used to build the *control* degradation model (conventional
latent-style degradation), against which our wear model is compared. Partitioned by category
so no test-set texture can reach training, which answers Cappelli's leakage criticism of
DNUNets and FingerGAN.

### L3-SF — Level-3 Synthetic Fingerprints

**https://github.com/luannd/L3-SF**
Wyzykowski, Segundo & Lemes, ICPR 2020. CC BY-NC-SA 4.0.

8,140 synthetic fingerprints with level-3 detail — sweat pores and scratches — including a
740-image subset with pore coordinate annotations. Secondary: level-3 detail sits above the
500 dpi civil-capture regime this thesis targets. Note the advertised 1200 dpi applies only
to the 740 annotated 512×512 images; the other 7,400 are 320×240 and cannot be 1200 dpi.

### FVS

**http://fvs.sourceforge.net/** — open-source project sample data.
168 images, 21 fingers × 8, 256×256, resolution unstated. Minor supplementary set.

---

## 2. Acquired but excluded

### SOCOFing

**https://www.kaggle.com/datasets/ruizgara/socofing** — Shehu et al., arXiv:1807.10609, 2018.

55,270 images (6,000 real prints from 600 subjects, plus 49,270 synthetically altered
versions with damage-type and severity labels). Downloaded, but **excluded from the project
on two grounds.** First, it is Kaggle-hosted, and supervisor guidance is that Kaggle datasets
are not to be used for training. Second, and independently, our own measurement does not
support its quality claim: the images are 96×103 px and declared 500 dpi, which would
describe a finger 4.9 mm across — the true sampling is nearer 200 dpi, far below civil
capture resolution. Its damage is also synthetic, so it does not represent the target
population regardless.

The severity axis it would have provided is instead obtained by applying our own wear model
at controlled severities to real clean prints, which gives known rather than undocumented
severity labels. The loss is demographic stratification, for which no substitute is currently
held.

---

## 3. Not acquired

### 3.1 Registration or approval pending

| Dataset | Link | What it is | Why not yet |
|---|---|---|---|
| **CASIA-FingerprintV5** | http://biometrics.idealtest.org (id 7) | 500 subjects × 8 fingers × 5 impressions. The population explicitly includes **workers and waiters**, and difficult impressions were requested — the closest free analogue to an occupationally worn population. | Registration plus manual human approval. Confirmed live 10 Sep 2026; walkthrough in `docs/datasets/casia-access.md`. **Now the highest-priority outstanding item.** |
| **CASIA Fingerprint Subject Ageing** | http://biometrics.idealtest.org (id 15) | 3,664 images. Public longitudinal fingerprint data is close to non-existent, and ageing is a named degradation cause in this thesis. | Same registration route. Found via the portal API; it appears in neither the thesis plan nor the literature review. |
| **PrintsGAN** | MSU — Engelsma, Grosz & Jain | 525,000 synthetic images, 35,000 fingers, multiple impressions per identity. Large-scale pretraining. | Signed agreement sent to Steven Grosz, MSU; awaiting reply. |
| **Tsinghua SD27 annotations** | http://ivg.au.tsinghua.edu.cn/dataset/NIST.php | Manually labelled orientation fields, period maps, skeletons and ROIs — the class of supervision Cappelli's orientation and frequency networks were trained on. | Request not yet sent. Annotates SD27 images, which are themselves withdrawn (§3.3). |
| **FVC manual segmentation ground truth** | Thai, Huckemann & Gottschlich, *PLOS ONE*, 2016 | Hand-marked segmentation masks for FVC images — what Cappelli trained his segmentation on. | Request not yet sent. |

### 3.2 Licensed — institutional signature required

All route through IIIT-Delhi's Image Analysis and Biometrics lab
(`databases@iab-rubric.org`) or NIST, and all need a signature above student level.

| Dataset | What it is | Why not yet |
|---|---|---|
| **Rural Indian Fingerprint DB** | Rural and urban Indian subjects with genuine occupational degradation — **the target population of this thesis**, and the benchmark of the Joshi research line. | Requires a **Registrar / Head-of-Institution signature**. Request drafted in `docs/correspondence/`, not yet submitted. Longest lead time of anything here. |
| **MUST** | ~21,000 impressions with **manually marked minutiae**, pore counts and semantic masks. The dataset that would allow true minutiae F1 against expert ground truth. | Same signature route; drafted, not submitted. |
| **IIIT-D MOLF** | 19,200 images, 100 Indian subjects, five capture methods. Used by the Joshi line and MLFGNet. | Same signature route; drafted, not submitted. |
| **NIST SD302 (N2N)** and **SD300** | SD302: multi-sensor live capture, 200 subjects × 10 fingers. SD300: 888 subjects of naturally poor, uncooperative ink-card captures. | Institutional email and manual NIST review. Not submitted. |

Because these are slow and uncertain, the thesis is structured so that no phase depends on
any of them. The Rural Indian DB is treated as a *confirmatory* external test set: valuable
if it arrives, not load-bearing if it does not.

### 3.3 Permanently unavailable

| Dataset | Status | What this means |
|---|---|---|
| **NIST SD27** | **Withdrawn January 2017** | 258 real crime-scene latents with expert-validated minutiae. This is the benchmark of essentially the entire latent-enhancement literature, *including the 2026 state of the art*, which still reports on it because established groups hold legacy copies. Its published numbers cannot be reproduced by anyone without one — a documented problem in the field, and the reason this thesis builds its protocol on obtainable data instead. |
| **ChaLearn LAP Track 3** | Provider broken; unrecoverable | 84,000 paired clean/degraded images, the intended primary training set. Sign-up returns HTTP 500 and password reset fails; several email addresses tried. Replaced by generating clean masters with Anguli and reimplementing ChaLearn's nine degradation artefacts in our own code (ADR 0003). |
| NIST SD4 / SD9 / SD10 | Discontinued | Historical baselines cannot be reproduced. |
| NIST SD14 | Uncertain; treated as unavailable | Was the 27,000–30,000-image training source for DNUNets, FingerGAN and SFP, so those training sets cannot be reconstructed either. |

---

## 4. Licence note

IAB, NIST, CASIA and FVC-A data may not be redistributed. Images are never committed to the
repository — only manifests recording each file's SHA-256, dimensions and declared
resolution. Of the acquired holdings, only the Anguli corpus (our own output) and MINEX
(public domain) could be released alongside the thesis.

---

*Figures measured from committed manifests via `scripts/dataset_dossier_stats.py`.
Access status per `docs/datasets/REGISTRY.md`.*
