# Biometric Fingerprint Enhancement for Degraded / Worn Fingerprints
## Master's Thesis Plan — Datasets, Literature Review, and Research Direction

---

# 0. Framing the problem (read this first)

Your instinct is right, but "fingerprint enhancement" as a topic is 40 years old and extremely crowded. The version of it that is **still open, still publishable, and matches your motivating story** is narrower:

> **Restoration of fingerprints degraded by permanent or semi-permanent skin damage — occupational abrasion, ageing, burns, scars, dryness — in the context of large-scale civil identity systems.**

That is different from the topic almost everyone actually works on, which is **latent fingerprints** (crime-scene marks with heavy background noise). The distinction matters enormously and should be a stated contribution of your thesis:

| | Latent fingerprints (well studied) | Worn / abraded fingerprints (your problem) |
|---|---|---|
| Dominant degradation | Structured **background** noise, partial prints, distortion | **Signal loss at the source** — ridges physically flattened |
| Ridge information | Present but occluded | Genuinely attenuated or destroyed |
| Fix strategy | Separate ridge signal from background | Recover/infer amplitude that is barely there |
| Failure mode | Missed minutiae | Failure-to-acquire, failure-to-enrol |
| Deployment | Forensic lab, offline, expert in the loop | Ration shop / micro-ATM, real time, no expert |

Almost every deep-learning enhancement paper is trained on *synthetically degraded* prints where the synthetic degradation is "blur + scratches + a texture background". That models a latent print. It does **not** model a bricklayer's thumb. This mismatch is your research gap, and it is a real one — the IIT-Delhi group (Joshi et al.) has published explicitly on the domain shift between synthetic training data and real rural Indian fingerprints.

**Positioning sentence for your synopsis:**
> Existing fingerprint enhancement research optimises for forensic latent prints and evaluates with image-similarity metrics; it under-serves the population-scale inclusion failure caused by occupationally and age-degraded ridges, where the correct objective is matcher performance and reduction in failure-to-acquire.

---

# 1. DATASETS — the priority section

## 1.1 Do these four things this week

Ordered by how long the paperwork takes. Start the slow ones first.

| # | Action | Why now |
|---|---|---|
| 1 | Email `databases@iab-rubric.org` requesting the **Rural Indian Fingerprint Database** and **IIIT-D MOLF** | Needs a licence agreement signed by your **Registrar / Head of Institution** — not you, not your supervisor. This can take 2–4 weeks. Start immediately. |
| 2 | Submit the **NIST SD302** request at `https://nigos.nist.gov/datasets/sd302/request` | Requires an institutional email; manual review by a NIST staff member. |
| 3 | Download the **ChaLearn LAP fingerprint denoising set** — no permission needed | This is your training data. 84,000 paired (clean, degraded) images. You can start coding the same day. |
| 4 | Download **SOCOFing** from Kaggle and the **FVC "B" subsets** — no permission needed | Free, instant, enough to build and debug the whole pipeline before the licensed data arrives. |

Do not wait for the licensed data to start work. Build the entire pipeline on the free data first.

---

## 1.2 Tier 1 — Directly on-topic (degraded / worn / real-world Indian prints)

### **Rural Indian Fingerprint Database** ⭐ your single most important dataset
- **What:** Fingerprints from rural and urban Indian populations, collected precisely because the rural subjects' ridges are degraded by manual labour.
- **Why it matters:** This is the *only* public dataset I found that matches your motivating scenario. Used as the benchmark in the Joshi et al. line of work at IIT Delhi.
- **Citation:** C. Puri, K. Narang, A. Tiwari, M. Vatsa, R. Singh, *On Analysis of Rural and Urban Indian Fingerprint Images*, Intl. Conf. on Ethics and Policy of Biometrics and International Data Sharing, 2010.
- **Access:** IAB Lab (originally IIIT-Delhi, group now at IIT Jodhpur) — `iab-rubric.org` → resources → biometric datasets → fingerprint. Signed licence agreement to `databases@iab-rubric.org`.
- **Caveat:** Some papers distinguish a "publicly available rural Indian fingerprint database" from a separate *private* rural database shared by Prof. Phalguni Gupta (IIT Kanpur). Ask explicitly for the public one; if your supervisor has a contact at IIT-K, the private one is worth requesting too.

### **IIIT-D MOLF** (Multi-sensor Optical and Latent Fingerprint)
- 19,200 fingerprints, 100 Indian subjects, 5 capture methods (Lumidigm, Secugen Hamster-IV, CrossMatch L-Scan Patrol, latent, simultaneous latent). Manual latent annotations available.
- Same licence route as above. WSQ ~548 MB, BMP ~21 GB.
- Sankaran, Vatsa, Singh, *Multisensor Optical and Latent Fingerprint Database*, IEEE Access, 2015.
- Gives you cross-sensor variation *and* an Indian population — very useful as a second test set.

### **MUST — Multi-Surface Multi-Technique Latent Fingerprint Database**
- ~21,000 impressions, 120 fingers, 35 acquisition scenarios, **with manually marked minutiae, PPI, and semantic segmentation masks**. The annotations are what make it valuable — you can compute minutiae F1 against ground truth.
- Malhotra, Vatsa, Singh, Morris, Noore, *MUST Latent Fingerprint Database*, IEEE TIFS, 2023. Same IAB licence route.

### **IIITD MSLFD** — 551 latents from 51 subjects lifted from 8 surface types, with mated slap galleries.

### **LFIW — Latent Fingerprint in the Wild**
- 13,180 samples, 1,318 unique finger instances, 132 subjects; includes contact, contactless and smartphone fingerphotos. Built explicitly because NIST SD27 was withdrawn. Worth searching for the current release page.

---

## 1.3 Tier 2 — Free, instant, no paperwork (build your pipeline on these)

### **ChaLearn LAP Fingerprint Denoising & Inpainting (WCCI'18 / ECCV'18)** ⭐ your training set
- `chalearnlap.cvc.uab.cat/dataset/32/description/`
- 84,000 clean/degraded **pairs** at 275×400, generated with Anguli then degraded with blur, brightness, contrast, elastic transform, occlusion, scratches, resolution change, rotation, plus texture backgrounds. Separate validation and test splits.
- This is the de-facto paired training set for supervised fingerprint restoration. Free.
- **Important limitation to state in your thesis:** its degradation model is a *latent-print* model. Extending it toward a *wear* model is one of your proposed contributions (§3).

### **SOCOFing** (Sokoto Coventry Fingerprint Dataset)
- Kaggle: `kaggle.com/datasets/ruizgara/socofing`. 6,000 real prints (600 African subjects × 10 fingers), plus **synthetically altered versions** simulating *obliteration*, *central rotation* and *z-cut* damage at Easy / Medium / Hard levels.
- Directly relevant: it is the closest public thing to a "damaged fingerprint" benchmark with labels.
- **Caveat you must report:** the curated dataset list notes the images are declared 500 dpi but are really ~96×103 px at roughly 200 dpi. Don't feed them to a 500-dpi-trained model without rescaling, and don't compare NFIQ2 scores across resolutions.

### **FVC2000 / FVC2002 / FVC2004 — "B" subsets**
- `bias.csr.unibo.it/fvc2000/download.asp` (and `/fvc2002/`, `/fvc2004/`). Each: 10 fingers × 8 impressions, free TIFF download.
- DB1–DB3 real (optical, capacitive, thermal sweep); DB4 synthetic (SFinGe). FVC2002/2004 subjects were asked to deliberately produce difficult impressions (wet/dry fingers, distortion, rotation) — genuinely useful low-quality data.
- The larger "A" subsets (100 fingers × 8) come bundled with the *Handbook of Fingerprint Recognition* (~€140). Ask your department to buy the book — you need it anyway and it is the standard reference.

### **Neurotechnology sample databases** — CrossMatch (51 fingers × 8) and U.are.U (65 fingers × 8), free at `neurotechnology.com/download.html#databases`.

### **MINEX validation imagery** — 50 subjects × 10 fingers × 2, permissively licensed, in the NIST `usnistgov/minex` GitHub repo.

### **Tsinghua SD27 annotation package** — `ivg.au.tsinghua.edu.cn/dataset/NIST.php`. Manually labelled orientation fields, period maps, skeletons, singular points and ROIs *for* SD27. Useful even without SD27 images, as a source of ground-truth orientation/frequency supervision.

### **Manual FVC segmentation ground truth** — Thai, Huckemann & Gottschlich, *Filter Design and Performance Evaluation for Fingerprint Image Segmentation*, PLOS ONE, 2016. Hand-marked masks for FVC images; used by Cappelli to train his models.

---

## 1.4 Tier 3 — Licensed / request-based

| Dataset | Size | Access | Notes |
|---|---|---|---|
| **NIST SD302 (N2N)** | 200 subjects × 10 fingers × 12–18 impressions, 15 sensor types + SD302E latents | Request form, institutional email | The best real multi-sensor set still distributed. Includes touch-free sensors. |
| **NIST SD301** | 51 subjects, similar structure + SD301B latents | Request form | Smaller sibling of SD302. |
| **NIST SD300** | 888 subjects × 10 fingers, scanned ink cards, rolled + plain, 500/1000/2000 dpi | Request form | Arrested US citizens, some uncooperative captures → naturally poor quality. |
| **CASIA-FingerprintV5** | 500 subjects × 8 fingers × 5 impressions | `idealtest.org`, registration | Population explicitly includes **workers and waiters**, and subjects were asked to produce difficult impressions. Relevant to your occupational angle. Site availability has been unreliable; try early. |
| **FVC2006 DB1–DB4** | 150 fingers × 12 each | 2-year licence via ATVS/UAM | DB1 is 250 dpi, 96×96 px — deliberately hard. |
| **LivDet 2009–2023** | Various | `livdet.org/registration.php` | Spoof-detection sets, but the live subsets are large and varied. |
| **PolyU low-resolution contactless** | 1,466 webcam finger images, 156 subjects, 2 sessions | PolyU request | Relevant if you explore the touchless fallback. |

---

## 1.5 Discontinued — do NOT plan around these

A very large fraction of older papers benchmark on datasets you cannot get:

- **NIST SD4, SD9, SD10** — discontinued, no longer distributed.
- **NIST SD27** (258 real crime-scene latents with expert minutiae) — **withdrawn January 2017** for lacking required documentation. Still the benchmark in papers published in 2025–2026, because authors hold legacy copies. You will not be able to reproduce those numbers.
- **NIST SD14** — status uncertain; several 2023+ papers still cite it as their 29K–30K-image training source. Treat as unavailable unless NIST says otherwise.

**Consequence for your thesis:** state this explicitly in your literature review. "The field's principal benchmark is unobtainable" is a legitimate, citable motivation for proposing an alternative evaluation protocol on obtainable data (§4). Reviewers respect this.

---

## 1.6 Synthetic generators — your way around the data shortage

Because paired (degraded, clean) real data barely exists, generating it is standard practice and fully accepted.

| Tool | What it gives you |
|---|---|
| **Anguli** (open-source C++, reimplements SFinGe) | Unlimited synthetic master prints + noisy variants, always paired with ground truth. This is what generated the ChaLearn set. |
| **SFinGe** (Bologna, Cappelli) | The original; generated FVC2002/2004/2006 DB4. |
| **PrintsGAN** (Engelsma, Grosz, Jain, MSU) | 525K images: 35K distinct fingers × 15 impressions. Released publicly. Pretrain on this, fine-tune on real. |
| **L3-SF / Multiresolution synthetic** (Wyzykowski et al.) | 740+ high-resolution prints with **pores and scratches** — level-3 detail. Public. |
| **Diffusion-based generators** | DiffFinger (2024); Grosz & Jain, *Universal Fingerprint Generation: Controllable Diffusion Model with Multimodal Conditions*, IEEE TPAMI, 2024. Newest generation. |

---

## 1.7 Tools you will need (all free)

| Tool | Use |
|---|---|
| **NFIQ 2** — `github.com/usnistgov/NFIQ2` | ISO/IEC 29794-4 reference quality score, 0–100. The standard way to show "quality improved". Pre-built binaries available. |
| **NBIS** (NIST Biometric Image Software) | `mindtct` minutiae extractor + `bozorth3` matcher. Your free baseline matcher. |
| **pyfing** — `github.com/raffaele-cappelli/pyfing` ⭐ | Cappelli's open-source segmentation, orientation-field estimation, frequency estimation, and the GBFEN/SNFEN enhancers. **This is your strongest baseline and your best starting codebase.** |
| **SourceAFIS** | Open-source matcher, good second opinion alongside bozorth3. |
| **MCC** (Minutia Cylinder-Code) | Standard research matcher; SDK from Bologna for academic use. |
| **CVxTz/fingerprint_denoising** | U-Net that won ChaLearn Track 3. Trivial baseline to reproduce. |

**Rule:** never report only PSNR/SSIM. Always report NFIQ 2 **and** at least one matcher's EER. More on this in §4.

---

# 2. LITERATURE REVIEW

Organised as you should organise the chapter. I've marked ★ for must-read.

## 2.1 The book (read first, chapters 3 and 5)

- ★ Maltoni, Maio, Jain, Feng — **Handbook of Fingerprint Recognition**, 3rd ed., Springer, 2022. Non-negotiable. Chapter on enhancement, and Cappelli's chapter on fingerprint synthesis.

## 2.2 Classical enhancement (pre-deep-learning) — the foundation

Two features dominate everything: **local ridge orientation** and **local ridge frequency**. Every method, old or new, is ultimately a way of estimating these and filtering accordingly.

- O'Gorman & Nickerson, *Matched filter design for fingerprint image enhancement*, ICASSP 1988. — the origin.
- Sherlock, Monro & Millard, *Fingerprint enhancement by directional Fourier filtering*, IEE Proc. VISP, 1994.
- ★ **Hong, Wan & Jain**, *Fingerprint image enhancement: algorithm and performance evaluation*, IEEE TPAMI 20(8), 1998. — the canonical Gabor-filter method. Still the baseline in 2026 papers. Implement it yourself.
- Almansa & Lindeberg, *Fingerprint enhancement by shape adaptation of scale-space operators*, IEEE TIP, 2000.
- Willis & Myers, *A cost-effective fingerprint recognition system for use with low-quality prints and damaged fingertips*, Pattern Recognition 34, 2001. — **read this one for your framing**; explicitly about damaged fingertips.
- Hsieh, Lai & Wang, *Wavelet-transform-based fingerprint enhancement*, Pattern Recognition 36, 2003.
- Yang et al., *A modified Gabor filter design method*, PRL 24, 2003.
- Chikkerur, Govindaraju & Cartwright, *Fingerprint enhancement using STFT analysis*, 2005.
- Wu & Govindaraju, *Singularity preserving fingerprint adaptive filtering*, ICIP 2006.
- Jirachaweng & Areekul, *Fingerprint enhancement based on DCT*, ICB 2007.
- Fronthaler, Kollreider & Bigun, *Local features for enhancement and minutiae extraction*, IEEE TIP 17, 2008.
- Wang et al., *Log-Gabor filter in fingerprint enhancement*, PRL 29, 2008.
- Zhao et al., *Curvature and singularity driven diffusion*, CVPR 2009.
- ★ **Gottschlich**, *Curved-region-based ridge frequency estimation and curved Gabor filters*, IEEE TIP 21, 2012. — handles high-curvature regions; important for worn prints where curvature estimation degrades.
- Gottschlich & Schönlieb, *Oriented diffusion filtering for enhancing low-quality fingerprint images*, IET Biometrics 1(2), 2012.
- Turroni, Cappelli & Maltoni, *Fingerprint enhancement using contextual iterative filtering*, ICB 2012.
- Sutthiwichaiporn & Areekul, *Adaptive boosted spectral filtering for progressive fingerprint enhancement*, Pattern Recognition 46, 2013.

**Dictionary-based (the bridge to learning):**
- ★ Feng, Zhou & Jain, *Orientation field estimation for latent fingerprint enhancement*, IEEE TPAMI 35, 2013. (GlobalDict)
- Yang, Feng & Zhou, *Localized dictionaries based orientation field estimation*, IEEE TPAMI 36, 2014. (LocalDict)
- Cao, Liu & Jain, *Segmentation and enhancement of latent fingerprints: a coarse-to-fine ridge-structure dictionary*, IEEE TPAMI 36, 2014. (RidgeDict)
- Rama & Namboodiri, *Fingerprint enhancement using hierarchical Markov random fields*, IJCB 2011.
- Chaidee, Horapong & Areekul, *Filter design based on spectral dictionary for latent fingerprint pre-enhancement*, ICB 2018.

## 2.3 Deep learning era — the core of your review

**CNN / autoencoder:**
- Schuch, Schulz & Busch, *De-convolutional autoencoder for enhancement of fingerprint samples*, IPTA 2016. — first DL enhancement, trained on synthetic.
- Svoboda, Monti & Bronstein, *Generative convolutional networks for latent fingerprint reconstruction*, IJCB 2017. (code: `github.com/JanSvob/fingerprint_gae`)
- ★ **Tang, Gao, Feng & Liu**, *FingerNet: An unified deep network for fingerprint minutiae extraction*, IJCB 2017. — unifies orientation estimation, enhancement, segmentation and minutiae extraction. Extremely widely cited; a standard baseline.
- Li, Feng & Kuo, *Deep convolutional neural network for latent fingerprint enhancement*, Signal Processing: Image Communication 60, 2018.
- Qian, Li & Liu, *Latent fingerprint enhancement based on DenseUNet*, ICB 2019.
- Wong & Lai, *Multi-task CNN for restoring corrupted fingerprint images*, Pattern Recognition 101, 2020.
- ★ Cao, Nguyen, Tymoszek & Jain, *End-to-end latent fingerprint search*, IEEE TIFS 15, 2020.
- Liu & Qian, *Automatic segmentation and enhancement of latent fingerprints using deep nested U-Nets*, IEEE TIFS 16, 2021. (DNUNets)

**GAN-based:**
- Dabouei et al., *ID-preserving GAN for partial latent fingerprint reconstruction*, BTAS 2018.
- ★ Joshi, Anand, Vatsa, Singh, Roy & Kalra, *Latent fingerprint enhancement using GANs*, WACV 2019.
- Huang, Qian & Liu, *Latent fingerprint image enhancement based on progressive GAN*, CVPRW 2020.
- ★ **Zhu, Yin & Hu**, *FingerGAN: a constrained fingerprint generation scheme for latent fingerprint enhancement*, IEEE TPAMI 45(7), 2023. — optimises minutiae directly via a skeleton + orientation-model constraint. Code released.
- Pramukha, Akhila & Koolagudi, *End-to-end latent fingerprint enhancement using multi-scale GAN*, PRL 184, 2024.

**Frequency-domain / progressive:**
- Horapong, Srisutheenon & Areekul, *Progressive and corrective feedback for latent fingerprint enhancement*, IEEE Access 9, 2021. (PCF)
- Kriangkhajorn, Horapong & Areekul, *Spectral filter predictor for progressive latent fingerprint restoration*, IEEE Access 12, 2024. (SFP)

**Denoising / inpainting (the ChaLearn line — closest to your setup):**
- Mansar, *Deep end-to-end fingerprint denoising and inpainting*, arXiv 1807.11888, 2018. — 1st place ChaLearn Track 3, U-Net, code public.
- Adiga & Sivaswamy, *FPD-M-Net: fingerprint image denoising and inpainting using M-Net based CNNs*, ECCVW 2018. — 3rd place.
- Escobar et al., *ChaLearn Looking at People: Inpainting and Denoising Challenges*, 2021 — the official challenge report; cite this for the dataset.

**★★ The current state of the art and the most important recent paper for you:**
- **Cappelli, R.** — *Unleashing the Power of Simplicity: A Minimalist Strategy for State-of-the-Art Fingerprint Enhancement*, 2025/2026 (arXiv 2603.19004). Introduces **GBFEN** (Gabor contextual filtering) and **SNFEN** (a plain 5-level encoder–decoder with skip connections, 4.9M params, 25-minute training). Both beat FingerGAN, DNUNets, SFP, FingerNet and all dictionary methods on SD27 by minutiae F1, and win an expert visual ranking 80% of the time. Open source (`pyfing`).
  - Supporting pair: Cappelli, *Exploring the Power of Simplicity: A New State-of-the-Art in Fingerprint Orientation Field Estimation*, IEEE Access 12, 2024; and *No Feature Left Behind: Filling the Gap in Fingerprint Frequency Estimation*, IEEE Access 12, 2024.
  - **Why this changes your plan:** the win came from *better orientation and frequency estimation*, not from a bigger generative model. A plain U-Net conditioned on good orientation + frequency beat every GAN. Your thesis should not be "let's try a diffusion model" — it should be "what conditioning signal does a worn print need?"
  - It also criticises prior work for training on latent prints using noise backgrounds taken from the *test* images (test-set leakage). Do not repeat that mistake, and cite the criticism.

## 2.4 The Indian / worn-fingerprint line ★ — your immediate related work

This is a small, coherent body of work you should know cold. Indu Joshi (IIT Delhi → Inria) with P. K. Kalra, plus Vatsa & Singh's group.

- ★ Puri, Narang, Tiwari, Vatsa & Singh, *On Analysis of Rural and Urban Indian Fingerprint Images*, 2010. — the dataset paper.
- ★ Joshi, Utkarsh, Singh, ... Kalra, *On restoration of degraded fingerprints*, Multimedia Tools and Applications, 2022.
- ★ Joshi, Prakash, Jaiswal, Kumar, Kalra, *Context-aware restoration of noisy fingerprints*, 2022. — argues CNNs exploit spatial context only through convolution, and that ridge orientation/spacing/frequency context must be injected explicitly.
- ★ Joshi, Prakash, Kumar, Kalra et al. — **CDC-GAN / cross-domain consistency**: aligns the synthetic training domain with the real degraded domain. Reports EER improving from 7.30 → 6.10 (Bozorth) and 5.96 → 5.31 (MCC) on the rural Indian data. **This is the number you have to beat.**
- ★ Joshi et al., *On estimating uncertainty of fingerprint enhancement models* — channel-refinement GAN (CR-GAN); evaluates on IIITD-MOLF + Rural Indian DB + a private rural DB.
- Joshi et al., *Data uncertainty and model uncertainty in fingerprint preprocessing*, chapter in *Digital Image Enhancement and Reconstruction*, Elsevier, 2023.
- Joshi et al. — sensor-invariant fingerprint ROI segmentation (evaluated across 12 public databases).

Also relevant:
- Zhou, Wu et al. (2024), *Recovery of incomplete fingerprints based on ridge texture and orientation field*, Electronics 13(14), 2873. — IFSR dual-stream network for **mutilated** prints (orientation branch + detail branch), tested on FVC2002/2004 plus an artificially damaged set.
- *Crease detection from fingerprint images and its applications in elderly people*, Pattern Recognition, 2008. — creases are a specific, tractable, under-studied failure mode in aged and worked hands.

## 2.5 Quality assessment, ageing and demographic bias — your motivation chapter

- ★ Tabassi, Olsen, Bausinger, Busch, Fiumara et al., **NFIQ 2**, NISTIR 8382 / ISO-IEC 29794-4. The standard.
- ★ **Galbally, Cepilovs, Blanco-Gonzalo, Ormiston, Miguel-Hurtado, Racz**, *A large-scale operational study of fingerprint quality and demographics*, arXiv 2409.19992, 2024. 15,942 subjects, 159,420 samples, real visa-issuance data, NFIQ-2 scored. Findings you can cite directly:
  - Quality is stable from age 12 to ~45–50, then **declines linearly**; over-65s are a serious operational risk.
  - Women's prints score consistently lower than men's (median ~49 vs ~59) — attributed to ridge density vs. 500 dpi sensor resolution, not to any intrinsic difference.
  - Right hand > left hand; thumb > index > middle > ring > little.
  - Attributes elderly degradation to **skin condition** (collagen loss, dryness, reduced elasticity) interacting badly with touch-based optical platens.
- ★ Galbally, Haraksim & Beslay, *A study of age and ageing in fingerprint biometrics*, IEEE TIFS 14, 2019. ~500K prints.
- Jain, Arora, Cao, Best-Rowden & Bhatnagar, *Fingerprint recognition of young children*, IEEE TIFS 12, 2016.
- Haraksim, Galbally & Beslay, *Fingerprint growth model for mitigating the ageing effect on children's fingerprints matching*, Pattern Recognition, 2019.
- Carmeli, Patish & Coleman, *The aging hand*, J. Gerontology, 2003. — the dermatological basis. Cite this when you justify your wear-degradation model.
- Priesnitz, Rathgeb, Buchmann, Busch & Margraf, *An overview of touchless 2D fingerprint recognition*, EURASIP JIVP, 2021.
- Orandi et al., NIST TN 8315, *Evaluating the operational impact of contactless fingerprint imagery on matcher performance*, 2020.

## 2.6 Deployment / policy evidence (motivation, not method)

Use sparingly — one or two paragraphs in the introduction, no more.

- *Aadhaar Failures: A Tragedy of Errors*, Economic & Political Weekly, 2019 — documented exclusion from PDS rations; manual labourers and elderly named as the vulnerable groups.
- India's Parliamentary **Public Accounts Committee** hearings, July 2025 — cross-party warnings that worn fingerprints and changed iris patterns are blocking PDS and MGNREGS beneficiaries.
- Reetika Khera's work on Aadhaar and welfare exclusion — the standard academic source for field evidence.
- Policy commentary (2025–26) notes face authentication reached ~1.5 crore monthly transactions against ~9.6 crore *daily* biometric authentications — i.e. the fallback is not yet at scale, and it needs a smartphone.

**A caution:** you mentioned a specific case of a woman denied access to her account. Track down a citable source — a court judgment, an EPW article, or a named news report — before putting it in the thesis. Do not paraphrase a half-remembered story into a research document; examiners check.

---

# 3. PROPOSED RESEARCH CONTRIBUTION

Four gaps fall out of the review above. Pick contributions 1 and 3 as your core; add 2 if time permits.

### Gap A — The degradation model is wrong for your population
Every supervised method trains on synthetic degradation that is essentially *latent-print* degradation: blur, scratches, occlusion, texture backgrounds. Occupational wear is physically different: **uniform ridge-amplitude attenuation**, **flattened ridge profiles**, **deep persistent creases**, **dryness causing broken/dotted ridges**, and **pressure-dependent partial contact dropout**. Nobody has built a degradation model grounded in that physiology.

### Gap B — Generative models hallucinate ridges
GANs and diffusion models invent plausible ridge continuations, producing **spurious minutiae**. For forensics an examiner catches this; for automated Aadhaar-style authentication, a hallucinated minutia is a silent identity error. Cappelli's SD27 results show precision, not recall, is the binding constraint (his best method wins on precision at 0.236 while several rivals have higher recall).

### Gap C — Evaluation uses the wrong metrics and an unobtainable benchmark
PSNR/SSIM measure pixel fidelity, not identity information. And the field's main benchmark (SD27) is withdrawn.

### Gap D — Nobody has stratified enhancement performance by demographics
Galbally et al. showed quality is biased by age, sex and finger type. No one has asked whether *enhancement methods* reduce or amplify that bias.

---

## Proposed contributions

### **Contribution 1 — A physiologically grounded wear-degradation model (WDM)**
Build a parametric degradation operator applied to clean prints from FVC, SD302, PrintsGAN and Anguli:
- ridge-amplitude attenuation with spatially varying severity (models abrasion),
- crease injection along anatomically plausible flexion lines,
- dryness dropout: stochastic ridge fragmentation, not additive noise,
- pressure-dependent nonlinear contrast and partial-contact masking,
- optional elastic distortion (skin has lost elasticity).

**Validate the realism** rather than asserting it — three checks:
1. NFIQ 2 score distributions of WDM-degraded prints vs. the Rural Indian DB (compare distributions, e.g. KS test).
2. Train a binary domain classifier (synthetic vs. real-degraded); a *low* accuracy means your synthesis is realistic.
3. Show a model trained on WDM data transfers to real rural data better than the same model trained on ChaLearn data.

This alone, done rigorously, is a publishable contribution and it *solves your data problem* — you can generate unlimited paired training data that actually looks like your target population.

### **Contribution 2 — Uncertainty-aware, abstaining enhancement**
Take the Cappelli lesson seriously: condition on orientation and frequency, keep the network simple, and add a **per-pixel confidence head**. In low-confidence regions, output "unknown" rather than a synthesised ridge. Then propagate that confidence to the minutiae stage: extracted minutiae in low-confidence regions are down-weighted or discarded before matching.

Hypothesis to test: **discarding uncertain minutiae improves matcher EER more than hallucinating them.** This directly attacks Gap B and gives you a clean, falsifiable claim. Note that Joshi et al. have already begun on uncertainty (CR-GAN, uncertainty estimation) — read those first and position yourself as the *abstention* variant, which they do not do.

### **Contribution 3 — A reproducible, matcher-centric, obtainable benchmark**
Define and release an evaluation protocol using only datasets a 2026 researcher can actually get: Rural Indian DB + IIITD-MOLF + SOCOFing-Altered + FVC low-quality subsets. Report NFIQ 2 shift, minutiae precision/recall/F1 where ground truth exists, and EER / TAR@FAR / rank-1 with **two independent open matchers** (bozorth3 and SourceAFIS or MCC). Include a **demographic and severity stratification** (Gap D). Publish the split files and evaluation scripts.

### **Contribution 4 (optional, high practical value) — Multi-impression fusion**
At a ration shop the operator already retries three times. Nobody uses that. Instead of enhancing one bad image, **fuse N bad impressions of the same finger** into one restored template. Cheap to implement, obviously deployable, and under-studied. If your main track stalls, this is a safe fallback contribution.

**Deployment constraint as a selling point:** target a model small enough for a micro-ATM or PoS device. SNFEN at 4.9M parameters and ~300 ms on CPU sets the bar. "Works on the hardware that actually exists in a village" is a much stronger conclusion than "0.3% better EER on a workstation GPU".

---

# 4. EVALUATION PROTOCOL (fix this before you train anything)

| Level | Metric | Notes |
|---|---|---|
| Image | PSNR, SSIM, Dice, Jaccard | Report only where clean ground truth exists (i.e. synthetic). **Never the headline result.** |
| Quality | **NFIQ 2 score shift** (0–100) | Distribution before vs. after enhancement. Report the *distribution*, not just the mean, and the % crossing an operational threshold. |
| Feature | Minutiae **precision / recall / F1** vs. ground truth | Cappelli's protocol: match on position (τ_D ≈ 14 px), direction (τ_θ ≈ π/9), and type — report both type-exact and type-agnostic. Needs annotated data (MUST, MOLF, SD27 annotations). |
| **Matching (headline)** | **EER, TAR@FAR=0.1%/0.01%, rank-1** | Two independent matchers. This is what the thesis is judged on. |
| Operational | **Failure-to-acquire / failure-to-enrol reduction** | The metric that connects to your motivating story. % of prints moving from below-threshold to above-threshold. Almost nobody reports this — make it yours. |
| Fairness | All the above, **stratified** by age band, sex, finger type, severity | Gap D. |

**Baselines to beat, in order:** Hong-Wan-Jain Gabor (1998) → FingerNet → the ChaLearn U-Net → FingerGAN → **GBFEN and SNFEN from `pyfing`**. If you don't beat the last two, your thesis is not finished.

**Two protocol rules:**
1. Never let test-set backgrounds or textures leak into training. Cappelli calls this out explicitly in prior work.
2. Never report only on synthetic test data. The whole point is the real domain.

---

# 5. PHASED PLAN

Assumes ~9 months. Compress or stretch proportionally.

### Phase 0 — Weeks 1–2: Unblock the data
- Send the IAB licence request (Registrar's signature — start the internal paperwork on day 1).
- Submit the NIST SD302 request.
- Download ChaLearn, SOCOFing, FVC "B", Neurotechnology samples.
- Register at `idealtest.org` for CASIA-V5.
- Install NFIQ 2, NBIS, pyfing. Verify all three run end-to-end on one FVC image.
- **Milestone:** you can compute NFIQ 2 and a bozorth3 EER on FVC2004 DB1_B from a script.

### Phase 1 — Weeks 3–8: Literature review + reproduction
- Write the survey chapter using the taxonomy in §2. Structure it around the two invariant features (orientation, frequency) and the shift from hand-designed filters → dictionaries → CNNs → GANs → the recent "simplicity" correction.
- Reproduce three baselines: Hong-Wan-Jain (write it yourself — you'll learn more from 200 lines of Gabor filtering than from any paper), the ChaLearn U-Net, and pyfing's SNFEN.
- **Milestone:** a baseline table on FVC + SOCOFing with NFIQ 2 and EER. First supervisor review.

### Phase 2 — Weeks 9–14: The wear-degradation model
- Implement WDM (Contribution 1).
- Run the three realism validations.
- Generate a large paired training corpus.
- **Milestone:** side-by-side figure — real rural print, ChaLearn-degraded print, WDM-degraded print. If a naive viewer can't tell yours from the real one, you've succeeded. Put this figure in every presentation.

### Phase 3 — Weeks 15–24: The model
- Orientation + frequency conditioned encoder–decoder, per Cappelli's architecture as a starting point.
- Add the uncertainty head and abstention (Contribution 2).
- Ablate: conditioning inputs, abstention on/off, WDM vs. ChaLearn training data, loss function.
- **Milestone:** beat SNFEN on the Rural Indian DB by EER. If you can't, you still have a strong negative result plus Contribution 1 — write it up honestly, that is real science.

### Phase 4 — Weeks 25–30: Full benchmark
- Complete evaluation across all obtained datasets, both matchers, all metrics, stratified.
- Failure analysis: which severity levels and which demographics still fail, and why.
- **Milestone:** complete results chapter.

### Phase 5 — Weeks 31–36: Write-up + paper
- Thesis.
- Target venues for the paper: **IJCB**, **ICB**, **IEEE TIFS**, **IEEE TBIOM**, **IET Biometrics**, **Pattern Recognition Letters**. IWBF (IEEE Intl. Workshop on Biometrics and Forensics) is a friendly, realistic first venue for a master's-level contribution.
- Release code + WDM generator + evaluation splits on GitHub. This substantially raises citation odds and costs you a weekend.

---

# 6. RISKS AND MITIGATIONS

| Risk | Likelihood | Mitigation |
|---|---|---|
| Rural Indian DB licence delayed or refused | Medium | Start week 1. Fall back to MOLF + SOCOFing-Altered + CASIA-V5 (worker population). Ask your supervisor to email Prof. Vatsa or Prof. Singh directly — a supervisor-to-supervisor email moves faster than a form. |
| You can't beat SNFEN | Medium-high | Cappelli is a top lab with a 25-year head start. Reframe: your contribution is the *degradation model* and the *benchmark*, with the model as a demonstration. Negative results on the "bigger model" hypothesis are publishable given his paper argues exactly that. |
| Compute limits | Medium | SNFEN trains in 25 minutes on a single RTX 3080 Ti. Deliberately choose small models — and make that a stated design goal for edge deployment. |
| Scope creep into latent fingerprints | High | Latent-print work is seductive because that's where the papers are. Resist. Your niche is worn prints. Every time you're tempted, reread §0. |
| No paired ground truth for real worn prints | Certain | This is inherent. Handle it by evaluating with matchers and NFIQ 2, which need no clean reference, and by reserving pixel metrics for synthetic data only. Say this explicitly in the thesis — it's a methodological point, not a weakness. |

---

# 7. Three sentences to take to your professor

1. There is a public **Rural Indian Fingerprint Database** plus **IIIT-D MOLF** and **MUST**, all licensable from `iab-rubric.org`, and the licence needs the Registrar's signature — so I need that started this week.
2. Training data is not actually a blocker, because the **ChaLearn 84,000 paired-image set** and the **Anguli/SFinGe/PrintsGAN** generators are free and unrestricted; the real gap is that their degradation models simulate crime-scene noise rather than occupational ridge wear, which is what I propose to fix.
3. The current state of the art (Cappelli 2025/26) shows that a simple orientation- and frequency-conditioned network beats every GAN on this task, so my contribution should be in the **degradation model, the uncertainty handling, and the evaluation protocol** — not in a larger architecture.

---

## Quick reference: links

| Resource | URL |
|---|---|
| Curated fingerprint dataset list | `github.com/robertvazan/fingerprint-datasets` |
| NIST biometric special databases | `nist.gov/itl/iad/btg/resources/biometric-special-databases-and-software` |
| NIST BDbC catalog (searchable) | `tsapps.nist.gov/BDbC/Search` |
| NIST SD302 request | `nigos.nist.gov/datasets/sd302/request` |
| ChaLearn fingerprint denoising set | `chalearnlap.cvc.uab.cat/dataset/32/description/` |
| SOCOFing | `kaggle.com/datasets/ruizgara/socofing` |
| FVC downloads | `bias.csr.unibo.it/fvc2000/download.asp` (also fvc2002, fvc2004) |
| FVC-onGoing (submit algorithms to sequestered benchmarks) | `biolab.csr.unibo.it/fvcongoing` |
| IAB Lab datasets (Rural Indian, MOLF, MUST, MSLFD, ISPFD, MMFV) | `iab-rubric.org/resources/biometric-datasets/fingerprint` |
| NFIQ 2 | `github.com/usnistgov/NFIQ2` |
| pyfing (SOTA enhancement, open source) | `github.com/raffaele-cappelli/pyfing` |
| ChaLearn-winning U-Net denoiser | `github.com/CVxTz/fingerprint_denoising` |
| SD27 annotations (Tsinghua) | `ivg.au.tsinghua.edu.cn/dataset/NIST.php` |
| LivDet | `livdet.org/registration.php` |
