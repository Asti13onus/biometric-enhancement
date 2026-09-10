# Literature Review

## Fingerprint Image Enhancement and the Restoration of Degraded Fingerprints

---

## 1. Scope and Organisation of this Review

This review surveys the published literature on fingerprint image enhancement, covering the period from 1988 to 2026. The objective is threefold. First, to trace the technical evolution of the field from hand designed spatial filters through dictionary learning to deep convolutional networks, generative adversarial networks, and the most recent transformer and diffusion based formulations. Second, to catalogue the datasets, training regimes, and evaluation protocols on which each family of methods was developed and validated. Third, to identify the specific gap that motivates the present work, namely the restoration of fingerprints degraded by permanent or semi permanent skin damage arising from manual labour, ageing, burns, and injury, as distinct from the crime scene latent fingerprints that dominate the existing literature.

The review is organised as follows. Section 2 defines the terminology and the underlying image model. Section 3 presents a taxonomy of enhancement approaches. Sections 4 through 9 examine each era of the literature in chronological order, identifying for each paper the method, the training data, the evaluation data, the metrics reported, and the prior work it builds upon or corrects. Section 10 covers the specific literature on degraded and worn fingerprints, including the Indian rural fingerprint research line. Section 11 covers fingerprint quality assessment and demographic bias. Section 12 covers synthetic fingerprint generation, which is the dominant solution to the training data problem in this field. Section 13 consolidates the datasets used across the literature and comments on their current availability. Section 14 consolidates evaluation practice. Section 15 states the research gaps that follow from the review.

### 1.1 Review Methodology

Papers were identified through IEEE Xplore, ACM Digital Library, SpringerLink, ScienceDirect, and arXiv, using the search terms fingerprint enhancement, latent fingerprint enhancement, fingerprint restoration, fingerprint denoising, fingerprint inpainting, degraded fingerprint, worn fingerprint, fingerprint image quality, and fingerprint orientation field estimation. Backward citation chaining was performed from three anchor sources: the Handbook of Fingerprint Recognition (Maltoni, Maio, Jain and Feng, 2022), the survey of Schuch, Schulz and Busch (2018), and the comparative table published by Cappelli (2026), which summarises approximately forty enhancement methods spanning four decades. Forward citation chaining was performed from FingerNet (2017), FingerGAN (2023), and the works of Joshi and colleagues on rural Indian fingerprints. Inclusion was restricted to peer reviewed conference and journal publications, arXiv preprints from established groups, and standards documents. Purely commercial white papers were excluded.

---

## 2. Terminology and the Fingerprint Image Model

A fingerprint is the impression of the friction ridge skin on the distal phalanx of a finger. The visible structure consists of alternating **ridges**, which contact the sensor platen and appear dark in a conventional optical capture, and **valleys**, which do not contact the platen and appear light.

Fingerprint features are conventionally organised into three levels. **Level 1** features are the global pattern class, that is, arch, tented arch, left loop, right loop, whorl, and double loop, together with singular points known as **cores** and **deltas**. **Level 2** features are the **minutiae**, the local discontinuities in ridge flow. The two minutia types used by essentially all automated systems are the **ridge ending**, where a ridge terminates, and the **bifurcation**, where a ridge divides into two. A minutia is described by its coordinates, its direction, and its type. **Level 3** features are the fine detail, principally sweat **pores**, incipient ridges, and ridge edge contours, and require capture resolutions above 1000 dots per inch to be reliably imaged.

Almost all automated fingerprint identification systems, known as AFIS, match on Level 2 features. The consequence for enhancement research is central and should be stated explicitly. The purpose of enhancement is not to produce a visually pleasing image. The purpose is to allow a minutiae extractor to recover the correct set of minutiae, and no others. An enhancement method that produces a smooth, high contrast, visually convincing image while inserting ridges that were not present in the original will generate **spurious minutiae** and will degrade rather than improve matching accuracy.

Two local image properties govern nearly every enhancement algorithm ever published.

The **local ridge orientation**, usually written as the orientation field and denoted theta, is the dominant direction of the ridge flow in a neighbourhood of a pixel. Because ridges have no head or tail, orientation is defined modulo pi rather than modulo 2pi. This is why orientation is almost always represented in the **double angle** form, that is, as the pair cosine of 2 theta and sine of 2 theta, which removes the discontinuity at the wrap around point and permits linear averaging and smoothing.

The **local ridge frequency**, denoted f, is the reciprocal of the local ridge period, that is, the number of ridges per unit distance measured perpendicular to the ridge direction. Together with orientation it fully parameterises a local sinusoidal model of the ridge and valley pattern.

Given these two quantities, enhancement reduces to **contextual filtering**. Contextual filtering differs from ordinary convolution in that a different filter is selected for each pixel according to features computed in its neighbourhood, rather than a single filter being applied across the whole image. In fingerprint enhancement the selected filter is typically a bandpass filter tuned to the local orientation and the local frequency, which passes the ridge signal and suppresses everything else.

Two further preprocessing terms recur throughout the literature. **Segmentation** is the separation of the foreground, that is the region containing friction ridge detail, from the background, and is usually expressed as a binary mask. **Cartoon texture decomposition**, introduced to this field from the total variation image processing literature, splits an image into a piecewise smooth cartoon component holding large scale intensity variation and background structure, and an oscillatory texture component holding the ridge pattern. Many latent enhancement pipelines discard the cartoon component as structured noise and operate on the texture component alone.

Finally, a terminological distinction that is essential to this thesis. A **latent fingerprint**, also called a fingermark, is an impression left inadvertently on a surface and recovered at a crime scene. Its degradation is dominated by **structured background noise**, that is, the texture or printing of the substrate, by partial capture, by smearing, and by nonlinear distortion. A **worn** or **abraded** fingerprint, by contrast, is a live capture from a finger whose ridge topography has been physically flattened or fragmented. Its degradation is a **loss of signal at the source**. The ridge information is not occluded, it is attenuated or destroyed. Most of the literature reviewed below addresses the former case. Section 10 addresses the small body of work that addresses the latter.

---

## 3. Taxonomy of Enhancement Approaches

The literature can be organised along seven partially independent axes. These axes are not mutually exclusive and most published methods occupy several simultaneously.

**Contextual filtering.** Whether a distinct filter is selected per pixel or per block on the basis of local features. Almost all pre deep learning methods, and several recent ones, are of this type.

**Domain of operation.** Whether the method works in the spatial domain or in the frequency domain. Approximately one quarter of published methods operate in the frequency domain, typically via the short time Fourier transform or the discrete cosine transform.

**Multiresolution analysis.** Whether the method analyses the image at several scales, either explicitly through a wavelet or pyramid decomposition, or implicitly through pooling and strided convolution in a neural network.

**Iterative processing.** Whether the method refines its output over repeated passes, commonly by beginning enhancement in high quality regions and propagating outward into degraded regions.

**Learning.** Whether filter parameters or mappings are learned from data. This subdivides into dictionary and sparse representation methods, and into neural network methods.

**Adversarial training.** Whether a discriminator network is used to shape the output distribution.

**Training data source.** Whether the model is trained on real fingerprints, on synthetically generated and synthetically degraded fingerprints, or on a combination. Because paired clean and degraded real fingerprints are almost impossible to collect, synthetic training data is the norm.

---

## 4. Era One: Classical Contextual Filtering, 1988 to 2013

### 4.1 Foundational Work

The field begins with **O'Gorman and Nickerson (1988)**, who proposed matched filter design for fingerprint enhancement. They introduced the core idea that a filter should be matched to the expected local ridge and valley profile and steered to the local ridge direction. Every contextual filtering method published since is a descendant of this formulation.

**Sherlock, Monro and Millard (1994)** moved the operation into the frequency domain, performing enhancement by directional Fourier filtering. They precomputed a bank of directional filters in the Fourier domain and selected among them according to the local orientation, thereby avoiding the cost of spatial convolution with a large kernel at every pixel. This established the frequency domain branch of the literature.

**Kamei and Mizoguchi (1995)** departed from the standard contextual selection scheme. Rather than selecting a filter according to a separately estimated orientation, they applied all filters from a predefined bank and then selected the response that was most suitable according to an energy based criterion. This is the earliest instance of what might now be called a response selection rather than a parameter estimation strategy, and it reappears in later work by Nakamura and by Turroni.

### 4.2 The Canonical Method

**Hong, Wan and Jain (1998)**, published in IEEE Transactions on Pattern Analysis and Machine Intelligence, is the single most cited fingerprint enhancement paper and remains the standard baseline in papers published in 2026. Their pipeline consists of normalisation to a target mean and variance, local ridge orientation estimation by gradient averaging in the double angle representation, local ridge frequency estimation by counting ridge peaks along a profile perpendicular to the ridge direction, region masking into recoverable and unrecoverable areas, and finally contextual filtering with an even symmetric **Gabor filter** tuned to the estimated orientation and frequency.

The Gabor filter, following Daugman (1985), is a sinusoid modulated by a Gaussian envelope. It is jointly optimal in the space and spatial frequency domains, which is precisely the property required to isolate a locally oriented, locally periodic signal. In the Hong, Wan and Jain formulation the standard deviation of the Gaussian envelope is a fixed constant. This choice is revisited in later work.

Hong, Wan and Jain also established the evaluation practice that dominated the following decade, namely reporting the goodness of extracted minutiae against a manually marked ground truth and reporting matching accuracy before and after enhancement.

### 4.3 Refinements to Gabor Filtering

A long sequence of papers refined the filter itself.

**Erol, Halici and Ongun (1999)** introduced feature selective filtering, adapting the filter according to the reliability of the local features.

**Greenberg et al. (2000)** applied a combination of anisotropic and directional filtering.

**Almansa and Lindeberg (2000)**, in IEEE Transactions on Image Processing, applied scale space theory, performing shape adaptation of scale space operators with automatic scale selection. This was an early iterative and multiresolution method.

**Willis and Myers (2001)**, in Pattern Recognition, is of particular relevance to the present thesis. Their paper is titled a cost effective fingerprint recognition system for use with low quality prints and damaged fingertips, and is one of the very few early papers to name damaged fingertips as the target condition rather than crime scene marks. They operated in the frequency domain on a block basis.

**Hsieh, Lai and Wang (2003)**, in Pattern Recognition, introduced wavelet transform based enhancement, decomposing the image into multiple resolution subbands and enhancing texture and ridge information separately at each scale. This is the earliest explicit multiresolution enhancement method.

**Yang, Liu, Jiang and Fan (2003)**, in Pattern Recognition Letters, proposed a modified Gabor filter with distinct positive and negative peak periods, in order to accommodate the observation that ridge width and valley width are not generally equal. This is the first of the modified Gabor family.

**Nakamura et al. (2004)** proposed a parallel ridge filter, applying the whole filter bank in parallel and selecting the best response, following the Kamei and Mizoguchi strategy rather than the Hong, Wan and Jain strategy.

**Chikkerur, Govindaraju and Cartwright (2005)**, later extended in Pattern Recognition (2007), introduced short time Fourier transform analysis. The image is divided into overlapping windows, the Fourier spectrum of each window is computed, and the dominant orientation, the dominant frequency, and an energy based quality measure are all read directly from the spectrum. Filtering is then performed in the same domain. The attraction of this method is that orientation, frequency, region mask, and quality are obtained simultaneously from a single transform rather than from four separate algorithms.

**Wu and Govindaraju (2006)** addressed a specific failure mode of contextual filtering, namely that in the neighbourhood of singular points the orientation field is not locally smooth and a single steered filter destroys the singularity. They proposed singularity preserving adaptive filtering.

**Jirachaweng and Areekul (2007)** replaced the Fourier transform with the discrete cosine transform, which is cheaper and better suited to block based hardware implementation.

**Fronthaler, Kollreider and Bigun (2008)**, in IEEE Transactions on Image Processing, used local features derived from the structure tensor and a Laplacian style image scale pyramid, combining multiresolution analysis with contextual enhancement and minutiae extraction in one framework.

**Wang, Li, Huang and Feng (2008)**, in Pattern Recognition Letters, replaced the Gabor filter with the **Log Gabor filter**, which has a Gaussian transfer function on a logarithmic frequency axis. The Log Gabor filter has no DC component and can be constructed with arbitrarily large bandwidth, which addresses a known limitation of the standard Gabor filter when ridge frequency estimates are uncertain.

**Zhao, Zhang, Zhang, Huang and Bai (2009)**, at CVPR, introduced curvature and singularity driven diffusion, an iterative partial differential equation approach to enhancing oriented patterns.

**Cappelli, Maio and Maltoni (2009)** proposed semi automatic enhancement of very low quality fingerprints, admitting an operator into the loop where fully automatic estimation fails.

**Gottschlich (2012)**, in IEEE Transactions on Image Processing, introduced **curved region based ridge frequency estimation and curved Gabor filters**. The insight is that the rectangular windows used by all previous frequency estimators are inappropriate in regions of high ridge curvature, because a rectangle straddling a curved ridge contains several orientations and yields an unreliable frequency estimate. Gottschlich instead defines curved regions that follow the ridge flow, obtaining both better frequency estimates and a filter shape that conforms to the ridge. This paper is directly relevant to degraded fingerprints, where curvature estimation is the first thing to fail.

**Gottschlich and Schonlieb (2012)**, in IET Biometrics, proposed oriented diffusion filtering specifically for low quality fingerprint images, an iterative diffusion process constrained to run along the ridge direction.

**Turroni, Cappelli and Maltoni (2012)**, at the International Conference on Biometrics, proposed contextual iterative filtering. The method applies the full filter bank and iterates, propagating reliable enhancement outward from high confidence regions.

**Sutthiwichaiporn and Areekul (2013)**, in Pattern Recognition, proposed adaptive boosted spectral filtering, a progressive scheme that is iterative, multiresolution, and frequency domain simultaneously. The progressive principle, that enhancement should begin where the signal is good and use those results to condition the enhancement of adjacent poor regions, becomes a recurring theme and is carried forward into the deep learning era by the Areekul group.

### 4.4 Summary of Era One

By 2013 the field had converged on a stable architecture. Estimate segmentation, estimate orientation, estimate frequency, filter contextually. The remaining differences among methods concerned the shape of the filter, the domain in which it was applied, whether the process iterated, and how the local features were estimated. The unresolved problem was that all of these methods fail together on severely degraded images, because they all depend on orientation and frequency estimates that cannot be obtained from an image in which the ridge signal is absent.

---

## 5. Era Two: Dictionary and Sparse Representation Methods, 2011 to 2018

The response to the failure mode identified above was to introduce **prior knowledge** about what fingerprint orientation fields look like, so that a plausible orientation field could be inferred in regions where it could not be measured. This is the conceptual bridge between hand designed filtering and learning.

**Rama and Namboodiri (2011)**, at IJCB, formulated enhancement using hierarchical **Markov random fields**, imposing a smoothness and consistency prior over the orientation field.

**Feng, Zhou and Jain (2013)**, in IEEE Transactions on Pattern Analysis and Machine Intelligence, introduced the method commonly referred to in the comparative literature as **GlobalDict**. A dictionary of reference orientation patches is learned from a corpus of good quality fingerprints. Given a noisy latent, an initial and unreliable orientation estimate is computed, then corrected by finding the dictionary elements that best explain it subject to compatibility constraints between neighbouring patches. Contextual Gabor filtering then produces the final enhanced image. This paper also supplied the ground truth segmentation masks for NIST Special Database 27 that were subsequently used by most of the field.

**Yang, Feng and Zhou (2014)**, in the same journal, extended this to **LocalDict**. The observation is that orientation patterns are not spatially stationary. The orientation patch distribution near the core is different from the distribution near the periphery. They therefore learn localised dictionaries indexed by position relative to an estimated fingerprint pose, which yields better reconstruction than a single global dictionary. LocalDict is an explicit improvement upon GlobalDict by the same research line.

**Cao, Liu and Jain (2014)**, in the same journal, proposed **RidgeDict**, a coarse to fine ridge structure dictionary. Rather than learning a dictionary of orientation patches only, they learn a dictionary of ridge structure patches, so that the reconstruction directly produces ridge information rather than only a direction field. They also introduced cartoon texture decomposition as a preprocessing step to remove structured background noise before dictionary reconstruction. Orientation and frequency are then extracted from the reconstructed patches and used to drive contextual Gabor filtering.

**Liu, Chen and Wang (2015)**, in IEEE Transactions on Information Forensics and Security, proposed latent fingerprint enhancement via multi scale patch based sparse representation, applying sparse coding across scales.

**Chaidee, Horapong and Areekul (2018)**, at the International Conference on Biometrics, moved the dictionary into the frequency domain with **SpectralDict**, learning a dictionary of spectral filters rather than spatial patches, again after a cartoon texture decomposition preprocessing step.

The dictionary methods represented a genuine advance on severely degraded images because they could hallucinate a plausible orientation field where none was measurable. They also introduced the risk that defines the rest of the field, namely that a plausible reconstruction is not necessarily the correct reconstruction.

---

## 6. Era Three: Convolutional Neural Networks, 2016 to 2021

### 6.1 First Neural Approaches

**Schuch, Schulz and Busch (2016)**, at IPTA, proposed a de convolutional autoencoder for enhancement of fingerprint samples. This is generally regarded as the first application of a learned end to end mapping to this problem. Critically, they trained on **synthetically generated fingerprints**, because paired clean and degraded real data does not exist at scale. This decision, taken out of necessity, defines the training methodology of nearly every subsequent deep learning paper in the field and is the origin of the domain shift problem discussed in Section 10.

The same authors published **Survey on the impact of fingerprint image enhancement** in IET Biometrics (2018), which remains the most useful quantitative survey of the pre deep learning methods. They evaluated seven typical enhancement methods across fourteen datasets, measured sample quality with NFIQ1 and NFIQ2, extracted minutiae with MINDTCT and FingerJetFX, and compared with BOZORTH3. Their finding, that enhancement does not uniformly improve biometric performance and that the relationship between image quality metrics and matching accuracy is not monotonic, is an important caution for anyone designing an evaluation protocol.

**Svoboda, Monti and Bronstein (2017)**, at IJCB, proposed generative convolutional networks for latent fingerprint reconstruction, using an encoder decoder architecture with a multiresolution structure, trained on synthetic data produced by the Anguli generator with additional blurring and background compositing. Their code and a subset of their synthetic data were released.

### 6.2 FingerNet and the Domain Knowledge Hybrid

**Tang, Gao, Feng and Liu (2017)**, at IJCB, proposed **FingerNet**, which is the most influential deep learning paper in this field. Their design principle is explicitly hybrid. They take the classical pipeline of orientation estimation, segmentation, Gabor enhancement, and minutiae extraction, and re express each classical operation in convolutional form. They then demonstrate that the resulting pipeline is mathematically equivalent to a shallow network with fixed weights. Having established that equivalence, they release the weights to be learned from data and expand the network capacity, while preserving end to end differentiability.

The architecture comprises two convolutional networks, one for orientation estimation and segmentation and one for minutiae extraction, with the Gabor enhancement step embedded between them as a differentiable layer. A significant practical contribution is their scheme for generating **weak labels** for latent fingerprints from the corresponding matched rolled or slap impressions, which enables modular training in the absence of true ground truth for latents.

FingerNet was trained and evaluated on NIST Special Database 27 for latents and FVC2004 for slap prints, and reported improved minutiae extraction over the then state of the art and over commercial software. Code was released publicly. FingerNet is used as a baseline in essentially every subsequent enhancement paper.

The importance of FingerNet for the present thesis is its central claim, that domain knowledge in the form of orientation and frequency should be built into the architecture rather than left for the network to discover. This claim is contested and then vindicated by the later literature, as described in Section 9.

### 6.3 The ChaLearn Denoising and Inpainting Challenge

A significant methodological event was the **ChaLearn Looking at People Inpainting Challenge, Track 3, Fingerprint Denoising and Inpainting**, held at WCCI 2018 and with a satellite event at ECCV 2018. The organisers generated a large paired dataset using the Anguli synthetic fingerprint generator. Ground truth images were produced with only small random transformations, at most five pixels of translation and plus or minus ten degrees of rotation. Degraded counterparts were produced by applying random artefacts, specifically blur, brightness change, contrast change, elastic transformation, occlusion, scratches, resolution reduction, and rotation, and then compositing the fingerprint onto a variety of background textures. The training set comprised 84,000 ground truth images and 84,000 corresponding degraded images at 275 by 400 pixels. Separate validation and test sets were constructed in the same way, with ground truth withheld for the test set.

This dataset transformed the field by making large scale supervised training accessible without any licensing barrier. It remains the largest freely downloadable source of paired clean and degraded fingerprints.

**Mansar (2018)** won Track 3 with a U-Net style fully convolutional network trained in a fully supervised manner on the challenge data, reporting the best results on all three challenge metrics, which were mean squared error, peak signal to noise ratio, and structural similarity index. Code was released.

**Adiga and Sivaswamy (2018)**, at the ECCV workshop, placed third with **FPD-M-Net**, based on the M-Net architecture. Their contribution was to reformulate denoising and inpainting as a foreground segmentation task and to adopt a structural similarity based loss function rather than a purely pixelwise loss, which better preserves ridge structure.

The limitation of this entire line of work, which is important for the present thesis, is that the metrics are image similarity metrics rather than biometric metrics, and the degradation model is a latent print degradation model.

### 6.4 CNN Refinements

**Li, Feng and Kuo (2018)**, in Signal Processing: Image Communication, proposed a deep convolutional network for latent fingerprint enhancement operating at multiple resolutions, learning a direct mapping from the noisy latent to an enhanced ridge pattern.

**Qian, Li and Liu (2019)**, at ICB, proposed **DenseUNet**, applying densely connected convolutional blocks within a U-Net topology and enhancing the fingerprint patchwise through an iterative refinement.

**Liu and Qian (2021)**, in IEEE Transactions on Information Forensics and Security, extended this to **DNUNets**, a deep nested U-Net architecture in which decoders are nested at each encoding level with skip connections originating from both encoding and decoding paths. Their training set comprised 30,000 latent fingerprints synthesised from real prints taken from NIST Special Database 14, degraded by compositing with various noise backgrounds including backgrounds extracted from NIST SD27 images. Training required approximately 24 hours. DNUNets was among the strongest methods on minutiae based metrics prior to 2025. The practice of drawing noise backgrounds from the test set is criticised in Section 9.

**Wong and Lai (2020)**, in Pattern Recognition, proposed a multi task CNN for restoring corrupted fingerprint images, jointly learning enhancement together with auxiliary tasks, and training on synthetic data. This method appears in later comparative tables under the name OFFIENet.

**Cao, Nguyen, Tymoszek and Jain (2020)**, in IEEE Transactions on Information Forensics and Security, published an end to end latent fingerprint search system in which enhancement is one module within a complete identification pipeline. Their enhancement component, referred to in comparative tables as Autoenc, is a convolutional autoencoder applied to the texture component obtained from cartoon texture decomposition. This work is important because it evaluates enhancement by its effect on identification rank rather than by image similarity, which is the correct evaluation philosophy.

---

## 7. Era Four: Generative Adversarial Networks, 2018 to 2024

The motivation for adversarial training in this field is that a pixelwise reconstruction loss, whether L1, L2, or structural similarity, encourages the network to output a blurred average over plausible ridge configurations when the input is ambiguous. A discriminator penalises outputs that do not lie on the manifold of real fingerprints, which forces the generator to commit to a sharp, ridge like output.

This era divides into two branches according to their supervision requirements. The larger branch, covered in Section 7.1, retains a paired reconstruction loss and adds an adversarial term to it, which means it still requires corresponding clean and degraded images and therefore still depends on synthetic degradation. The smaller branch, covered in Section 7.2, abandons pairing altogether and learns a mapping between two unpaired collections of images. The distinction matters because the pairing requirement is the origin of the domain shift problem that runs through the whole field.

### 7.1 Paired and Supervised Adversarial Methods

**Dabouei, Soleymani, Kazemi, Iranmanesh, Dawson and Nasrabadi (2018)**, at BTAS, proposed an **identity preserving generative adversarial network for partial latent fingerprint reconstruction**. Their central concern is the one that recurs throughout this era. A generative model can produce a realistic looking fingerprint that belongs to a different identity. They therefore add an identity preservation constraint to the objective. They evaluated on the IIIT-Delhi MOLF database in both latent to sensor and latent to latent matching protocols, reporting cumulative match characteristic curves and rank 25 and rank 50 accuracy.

**Joshi, Anand, Vatsa, Singh, Dutta Roy and Kalra (2019)**, at WACV, proposed latent fingerprint enhancement using generative adversarial networks, referred to in later comparative tables as JoshiGAN or FP-E-GAN. They trained on synthetic data. This paper is the entry point of the IIT Delhi research line that becomes central in Section 10.

**Liu, Tang, Li and Feng (2019)**, at ICB, proposed **COOGAN**, a cooperative orientation generative adversarial network, in which orientation field generation and image enhancement are trained cooperatively rather than sequentially.

**Huang, Qian and Liu (2020)**, at the CVPR Workshops, proposed a **progressive generative adversarial network** for latent enhancement. The progressive element refers to training the network by gradually increasing the resolution of the training images, a technique imported from the progressive growing GAN literature, combined with iterative refinement at inference.

**Zhu, Yin and Hu (2023)**, in IEEE Transactions on Pattern Analysis and Machine Intelligence, volume 45, issue 7, pages 8358 to 8371, proposed **FingerGAN**, which is the most technically interesting GAN in this literature. Their formulation reframes enhancement as a **constrained fingerprint generation** problem. Rather than requiring the enhanced grey level image to resemble a ground truth grey level image, they require the generated output to be indistinguishable from ground truth in terms of two specific structures. The first is the **fingerprint skeleton map weighted by minutia locations**, and the second is the **orientation field regularised by the FOMFE model**, that is, the fingerprint orientation model based on two dimensional Fourier expansion of Wang, Hu and Phillips.

The justification is direct. Minutiae are the primary matching feature, and minutiae can be read directly off the skeleton map. Therefore, by making the skeleton the output representation and by increasing the reconstruction loss weight in minutia regions, the network optimises minutia information directly rather than optimising a proxy. FingerGAN was trained on 130,000 latent fingerprints synthesised from 13,000 real fingerprints drawn from NIST Special Database 14, with noise backgrounds taken from NIST SD27 images. Enhancement at inference proceeds by cartoon texture decomposition, division of the image into 192 by 192 pixel patches, one network inference per patch, and recombination. Code is available at the repository HubYZ/LatentEnhancement.

FingerGAN is an explicit improvement over the grey level reconstruction objective used by DenseUNet, DNUNets, and the earlier GANs, and it is the strongest baseline for any minutiae oriented method. Its weakness, identified by Cappelli and visible in published comparisons, is that its output is a valley skeleton rather than a ridge image, and that in very poor regions it reconstructs ridge patterns that are unrealistic.

**Pramukha, Akhila and Koolagudi (2024)**, in Pattern Recognition Letters, volume 184, pages 169 to 175, proposed end to end latent fingerprint enhancement using a multi scale generative adversarial network.

### 7.2 Unpaired Translation and the CycleGAN Line

**Karabulut, Tertychnyi, Arslan, Ozcinar, Nasrollahi, Valls, Vilaseca, Moeslund and Anbarjafari (2020)**, in Multimedia Tools and Applications, volume 79, pages 18569 to 18589, proposed cycle consistent generative adversarial neural network based low quality fingerprint enhancement. This paper takes a different route to the training data problem from every other work reviewed so far, and it is the most directly relevant piece of prior work to the population addressed in this thesis.

**Motivation.** The authors frame the problem operationally rather than forensically. Their work originated from a system built for GEYCE Biometrics, a company whose clients include the Spanish and Portuguese Ministries of Foreign Affairs, and which had deployed a tool at consular posts that detected the type of distortion in a low quality fingerprint so that the operator could instruct the applicant on how to fix it and then recapture. The authors identify two failures of that workflow. Recapture is time consuming and not user friendly, and, more fundamentally, recapture cannot compensate for a distortion caused by skin tissue damage or any other permanent cause. Their stated objective is to enhance the fingerprint digitally to a quality sufficient for identification without recapture. This is the same operational framing adopted in the present thesis.

**Database and distortion taxonomy.** The database comprises 11,541 low quality fingerprint images captured at visa application centres in South American countries during real applications, provided by GEYCE Biometrics and labelled by their experts. Most samples carry two or three distortion types simultaneously. The taxonomy is defined in physiological rather than image processing terms, and is reproduced here because it is the most operationally grounded description of live capture degradation available in the literature.

- **Dry.** The finger makes low contact area with the scanner surface, so the ridge pattern is not fully captured because the ridges themselves make poor contact. The informal physical remedy is to rub the finger against the forehead or nose to pick up sweat and grease.
- **Wet.** The finger makes excessive contact area. Because of large amounts of grease and sweat, the valleys contact the platen along with the ridges, collapsing the ridge to valley contrast. The physical remedy is to remove excess grease and sweat.
- **Damaged.** The finger carries scars or problematic tissue such as burns and cuts. The authors state plainly that no instant remedy exists, that a temporary injury can be recaptured after healing, and that if the damage is permanent or congenital then enhancement by recapture is not possible.
- **Dotted.** Distinguishable black dots appear across the image, located around sweat pores, caused by excessive sweating that is frequently attributable to nervousness. Dots can occur independently of dryness.
- **Blurred.** Regions where ridges are not distinguishable, caused either by finger movement during capture or by excessive pressure that brings the valleys into contact with the platen. Where the underlying cause is burned tissue, no instant remedy exists.

The database is not publicly available, which the authors attribute to privacy concerns.

**Method.** The authors apply CycleGAN, the unpaired image to image translation architecture of Zhu, Park, Isola and Efros (2017), training five separate translations from a distorted domain to an undistorted domain, namely dry to not dry, wet to not wet, dotted to not dotted, damaged to not damaged, and blurred to not blurred. The architecture consists of two generators in a cyclic arrangement, one mapping domain A to domain B and one mapping B to A, together with two discriminators. The reverse translation, from undistorted to distorted, is not itself of interest and its results are not shown, but it is necessary in order to compute the cycle consistency loss that trains the forward generator.

The generators use a U-Net based architecture with skip connections, following the reasoning of Isola et al. that a large amount of low level information is shared between input and output in image translation tasks and should be allowed to bypass the bottleneck. Input resolution is 512 by 512. Convolutional layers use a 4 by 4 filter with stride 2 by 2, transposed convolutions likewise, with weights initialised from a normal distribution of mean zero and standard deviation 0.02. Activations are LeakyReLU with negative slope 0.2, ReLU, and tanh. Batch normalisation is used throughout and dropout is applied at a rate of 0.5. The discriminators follow a standard DCGAN architecture, with the final two convolutional layers using stride 1 by 1.

The generator objective is the sum of the two adversarial terms and a weighted cycle consistency term. The adversarial terms use a least squares formulation rather than the original cross entropy formulation. The cycle consistency term is the L1 distance between an image and its reconstruction after a round trip through both generators. Training used vertical flipping and rotation of plus or minus 35 degrees for augmentation, a cycle consistency weight of 10, the Adam optimizer with initial learning rate 0.0002 and momentum decay rates of 0.5 and 0.999 for both generator and discriminator, and a batch size of 1. The authors note that the loss curve is uninformative for GAN training and that they therefore checked convergence by periodically translating images and inspecting the results.

**Evaluation and results.** Evaluation used the VGG16 based low quality fingerprint classifier previously developed by Tertychnyi, Ozcinar and Anbarjafari (2018), trained on a disjoint subset of the same database. Translated test images were passed to this classifier and the metric reported is the percentage of enhanced images that the classifier labelled as undistorted.

| Distortion class | Percentage labelled undistorted after translation |
|---|---|
| Dry | 86 |
| Wet | 94 |
| Dotted | 87 |
| Damaged | 88 |
| Blurred | 64 |

The wet class performs best, which the authors attribute to the generator removing the unwanted pixel density caused by valley contact and thereby restoring ridge to valley contrast. The blurred class performs worst, which the authors attribute candidly to CycleGAN producing smoothed output, so that a blurred input is not necessarily deblurred. For the dry class the generator fills small discontinuities along ridges caused by low contact area. For the damaged class the authors state that scars and cuts are filled by the generator with respect to the original ridge pattern, with small cuts corrected completely and larger cuts and gaps improved.

**Assessment.** Three observations follow, and each bears directly on the direction of the present work.

First, the distortion taxonomy is the most useful artefact in the paper. It is derived from operational data rather than from a synthetic degradation script, it describes causes rather than appearances, and four of its five classes correspond to physical conditions of the fingertip rather than to properties of the substrate or the background. Dry, wet and damaged in particular map onto the occupational and age related wear that this thesis addresses. Any degradation model intended to simulate worn fingerprints should be validated against this taxonomy.

Second, the unpaired formulation is a genuine alternative to synthetic degradation, and the two approaches should be treated as competing solutions to the same problem rather than as unrelated techniques. Where the paired methods of Section 7.1 must invent a degradation process and then inherit the resulting domain shift, CycleGAN requires only two unpaired collections of real images, one distorted and one not. The cost is that the mapping is constrained only by the discriminators and the cycle consistency term, neither of which knows anything about ridges, orientation fields, or minutiae.

Third, and most importantly, the evaluation does not test identity preservation. The sole metric is the judgement of a distortion classifier, which was trained to recognise the visual signature of dryness, wetness and so on. Demonstrating that a translated image no longer looks dry to that classifier is not evidence that the ridge structure of the correct finger has been recovered. No quality score, no minutiae precision or recall, no matcher, and no error rate is reported. The authors' own account of the damaged class makes the concern concrete, since filling scars and cuts with respect to the surrounding ridge pattern is precisely the generative reconstruction that produces spurious minutiae. Cycle consistency does not prevent this, because a round trip can be satisfied by a mapping that encodes and then recovers information without that information corresponding to real ridge structure. This paper therefore restores fingerprints from exactly the operational population of interest and never verifies that the identity survived the restoration. That omission is a directly addressable gap, and it is taken up in Section 15.

The supporting classifier paper is worth noting separately. **Tertychnyi, Ozcinar and Anbarjafari (2018)**, in IET Biometrics, volume 7, issue 6, pages 550 to 556, proposed low quality fingerprint classification using a deep neural network, based on VGG16, achieving highest accuracy on the dry class and lowest on the blurred class. Its relevance beyond its role as an evaluation tool is that it establishes distortion type classification as a viable preprocessing step. A deployed system could triage an incoming capture by degradation type and route it to a treatment appropriate to that type, rather than applying one enhancement model to every input. No enhancement pipeline in the reviewed literature does this.

---

## 8. Era Five: Frequency Domain Learning, Transformers, and Diffusion, 2021 to 2026

### 8.1 Progressive Spectral Methods

The Areekul group at Kasetsart University pursued a distinct line combining the progressive philosophy of Sutthiwichaiporn and Areekul (2013) with learned components.

**Horapong, Srisutheenon and Areekul (2021)**, in IEEE Access volume 9, pages 96288 to 96308, proposed **PCF**, progressive and corrective feedback for latent fingerprint enhancement using boosted spectral filtering and a spectral autoencoder. A cartoon texture decomposition precedes the process. The system enhances iteratively and includes an explicit error detection and correction feedback loop.

**Kriangkhajorn, Horapong and Areekul (2024)**, in IEEE Access volume 12, pages 66773 to 66800, proposed **SFP**, a spectral filter predictor. A deep network predicts the spectral filter to apply, rather than predicting the enhanced image directly. The procedure is iterative and prioritises image patches by quality, using the improved spectra of already enhanced patches to condition the enhancement of poorer neighbours. Training used 29,700 fingerprints from NIST SD14 combined with backgrounds from the Describable Textures Dataset. Training time was approximately 2.5 hours. This is the most sophisticated frequency domain method in the literature.

### 8.2 Transformers

**Jia, Huang, Wang, Fei, Wu and Feng (2024)**, in IEEE Transactions on Information Forensics and Security, volume 19, pages 8860 to 8874, proposed the **Finger Recovery Transformer**, abbreviated FingerRT, for incomplete fingerprint identification. Their contribution is to incorporate the number of minutiae as prior knowledge into the self attention mechanism, so that the transformer is conditioned on how much Level 2 information the print is expected to contain.

**Wang, Qi, Fu and Hu (2025)**, arXiv preprint 2504.15105, proposed **TBSFNet** and **MLFGNet**. Their observation is that different regions of a degraded fingerprint require different enhancement strategies. High quality regions need contrast improvement only. Low quality regions need structural restoration using surrounding context. Background regions need noise suppression. They therefore construct a **Triple Branch Spatial Fusion Network**, in which three architecturally distinct branches process the three region types and are fused by a double branch spatial attention mechanism derived from the Attention Gate. The low quality and background branches include additional downsampling and upsampling to enlarge the receptive field, on the reasoning that restoring a degraded region requires a larger neighbourhood context than enhancing a good one.

They then extend TBSFNet into **MLFGNet**, the Multi Level Feature Guidance Network, by adding an orientation estimation block, which outputs the sine and cosine of twice the ridge angle, and a minutiae guidance block, which predicts minutiae regions in the enhanced output and amplifies the reconstruction loss weight in those regions. The minutiae guidance block is trained separately from the main network and then used to supervise it.

This paper is significant for the present thesis for two reasons. First, it evaluates on the **MOLF and MUST datasets**, which are exactly the datasets obtainable under licence from the IAB laboratory, rather than on the withdrawn NIST SD27. Second, it reports rank based identification accuracy rather than image similarity. The published results are reproduced below because they constitute the most directly usable baseline table available.

| Method | MUST Rank-1 | MUST Rank-20 | MOLF Rank-1 | MOLF Rank-50 |
|---|---|---|---|---|
| Raw image, no enhancement | 0.1083 | 0.1487 | 0.0050 | 0.0211 |
| FingerNet (2017) | 0.1366 | 0.2148 | 0.1256 | 0.2672 |
| LatentAFIS (Cao et al. 2020) | 0.1558 | 0.2406 | 0.2139 | 0.4306 |
| FingerGAN (2023) | 0.2167 | 0.3034 | 0.2553 | 0.4594 |
| JoshiGAN (2019) | 0.1863 | 0.2672 | 0.3156 | 0.5286 |
| COOGAN (2019) | 0.1937 | 0.2907 | 0.3050 | 0.5114 |
| OFFIENet (Wong and Lai 2020) | 0.2032 | 0.2898 | 0.3206 | 0.5317 |
| NestedUNets (Liu and Qian 2021) | 0.1961 | 0.2735 | 0.2847 | 0.4856 |
| TBSFNet (2025) | 0.2156 | 0.3099 | 0.3686 | 0.6006 |
| MLFGNet (2025) | 0.2299 | 0.3292 | 0.3875 | 0.6278 |

Two observations follow. Enhancement of any kind produces an enormous improvement over the raw image on MOLF, from rank 1 accuracy of 0.005 to above 0.30, which establishes that enhancement is worth doing. And the absolute accuracies remain low in absolute terms, below 0.40 at rank 1, which establishes that the problem is far from solved.

### 8.3 Diffusion Models

Diffusion probabilistic models entered this field primarily through synthesis rather than enhancement.

**Grosz and Jain (2024)**, in IEEE Transactions on Pattern Analysis and Machine Intelligence, proposed universal fingerprint generation using a controllable diffusion model with multimodal conditions, allowing generation conditioned on class, impression type, and other attributes.

**Grabovski, Yasur, Hacmon, Nisimov and Nimrod (2024)**, arXiv 2405.04538, proposed **DiffFinger**, applying denoising diffusion probabilistic models to synthetic fingerprint generation.

Work on diffusion based latent fingerprint synthesis has also examined **intra finger variability**, that is, whether a diffusion model generating multiple impressions of the same finger produces realistic within class variation, and has raised the problem of **hallucination in diffusion models**, connecting to the broader literature on mode interpolation.

Diffusion models applied directly to fingerprint enhancement remain comparatively unexplored, which is both an opportunity and a warning, since the hallucination risk that diffusion models exhibit in general image restoration is particularly dangerous when the output is used for identity decisions.

---

## 9. Era Six: The Simplicity Correction, 2023 to 2026

A sequence of three papers by Cappelli constitutes the most important recent development and substantially revises the field's assumptions.

**Cappelli (2023)**, in IEEE Access volume 11, pages 144530 to 144544, proposed two remarkably effective methods for fingerprint segmentation, both very simple.

**Cappelli (2024a)**, in IEEE Access volume 12, pages 55998 to 56018, proposed a new state of the art in **fingerprint orientation field estimation** using a streamlined encoder decoder convolutional network whose head produces explicit angles in the interval zero to pi. This surpassed all prior orientation estimation methods on standard benchmarks.

**Cappelli (2024b)**, in IEEE Access volume 12, pages 153605 to 153617, addressed **fingerprint frequency estimation**, which had been comparatively neglected. The contribution is a technique for producing accurate ground truth labels for local ridge frequency, derived from manually annotated ridge skeletons, which then permits supervised training of a frequency estimation network. The network takes as input the fingerprint image, its segmentation mask, and the estimated orientation field, concatenated into a four channel tensor.

**Cappelli (2026)**, arXiv preprint 2603.19004, builds on the previous two and presents two enhancement methods.

**GBFEN**, Gabor Based Fingerprint ENhancement, is a pure contextual filtering method. Orientation is estimated by the 2024a network, frequency by the 2024b network, and a precomputed bank of 144 Gabor filters, comprising 16 orientations and 9 frequencies, is applied by selecting for each pixel the filter closest to the local orientation and frequency. The single departure from classical practice is that the standard deviation of the Gaussian envelope is set as a function of the local frequency, specifically sigma equal to five twelfths of the reciprocal of f, rather than held constant as in Hong, Wan and Jain. Kernel size is derived from sigma. Each filter is standardised to zero mean and unit norm.

**SNFEN**, Simple Network for Fingerprint ENhancement, replaces the Gabor filtering stage with a plain encoder decoder convolutional network. The stem block converts the orientation field into the double angle representation and concatenates it with the image, the mask, and the frequency map to form a five channel input. The encoder and decoder each have five levels, using five by five convolutions with padding and ReLU activation, batch normalisation, and two by two max pooling or upsampling respectively, with skip connections between corresponding levels. The head operates at full input resolution with sixteen five by five filters followed by a final convolution with sigmoid activation.

The training methodology is notable for its economy. Training data consisted of only 360 fingerprints, specifically 100 images each from FVC2002 DB2 A, FVC2002 DB3 A, and FVC2004 DB1 A, plus 60 from the FFE benchmark, with FVC2002 DB1 A held out for model selection. Ground truth was assembled as follows. Segmentation masks came from the manually marked data of Thai, Huckemann and Gottschlich (2016) for the FVC sets. Orientation fields were manually annotated using a purpose built tool. Frequency maps were derived from manually annotated ridge skeletons. The ground truth enhanced image was generated by applying the GBFEN contextual convolution to the manually annotated skeleton using the ground truth orientation and frequency, yielding a near binary image with white ridges and black valleys.

Data augmentation was fingerprint specific and included random translation of plus or minus five percent, rotation of plus or minus twenty degrees, scale change of plus or minus fifteen percent, horizontal flip, gamma correction, contrast reduction, morphological erosion and dilation to simulate varying ridge and valley thickness, random lines to simulate scratches, and elliptical blobs to simulate abrasions.

The loss function is based on the **Tversky index**, a generalisation of the Dice coefficient that permits asymmetric weighting of false positives and false negatives. In this formulation the terms are true ridge agreement, false ridge agreement, and false valley agreement, all computed within the foreground mask, with the weighting parameter alpha set to 0.7. Optimisation used the Lion optimizer with cosine decay learning rate schedule and warmup, beta one of 0.2 and beta two of 0.5, for a fixed 25 epochs of 300 batches of 16 images. Total training time was approximately 25 minutes on a single NVIDIA GeForce RTX 3080 Ti. The resulting model has 4.9 million parameters.

Evaluation used NIST SD27, with minutiae extracted from the enhanced images by the VeriFinger SDK version 13.1 and compared against the expert validated ground truth minutiae. The metrics were precision, recall, and F1 score, with correspondence defined by a Euclidean distance threshold of 14 pixels, a direction threshold of pi over nine, and both type exact and type agnostic variants. The reported results are as follows.

| Method | Year | Precision | Recall | F1 score |
|---|---|---|---|---|
| GlobalDict | 2013 | 0.150 | 0.380 | 0.215 |
| Autoenc | 2020 | 0.153 | 0.380 | 0.218 |
| FingerGAN | 2023 | 0.165 | 0.395 | 0.233 |
| LocalDict | 2014 | 0.168 | 0.405 | 0.238 |
| FingerNet | 2017 | 0.165 | 0.431 | 0.238 |
| SpectralDict | 2018 | 0.163 | 0.465 | 0.242 |
| RidgeDict | 2014 | 0.175 | 0.395 | 0.242 |
| PCF | 2021 | 0.173 | 0.429 | 0.247 |
| DenseUNet | 2019 | 0.186 | 0.392 | 0.252 |
| DNUNets | 2021 | 0.198 | 0.389 | 0.263 |
| SFP | 2024 | 0.187 | 0.452 | 0.264 |
| GBFEN | 2026 | 0.201 | 0.495 | 0.286 |
| SNFEN | 2026 | 0.236 | 0.436 | 0.306 |

Values are for exact minutia type matching. Best performance appears at the bottom.

A separate expert evaluation, in which two examiners ranked the outputs of all methods on ten images from the Ugly subset of SD27, placed SNFEN first in 80 percent of cases and GBFEN second in 65 percent of cases.

Five conclusions from this work directly shape the present thesis.

First, the ranking is close to the inverse of architectural complexity. FingerGAN, trained on 130,000 images, is placed third from the bottom. SNFEN, trained on 360 images for 25 minutes, is placed first.

Second, the binding constraint is **precision**, not recall. Several methods achieve competitive recall while scoring poorly overall because they generate large numbers of spurious minutiae. Since a spurious minutia is a false identity feature, precision is the operationally important quantity.

Third, the improvement derives from **better estimation of orientation and frequency**, not from a larger or more expressive enhancement model. This vindicates the FingerNet philosophy of embedding domain knowledge, while rejecting the assumption that more capacity helps.

Fourth, Cappelli explicitly criticises the practice, used by DNUNets and FingerGAN among others, of synthesising training latents by compositing onto noise backgrounds **extracted from the test images**, on the grounds that this risks overfitting and information leakage from the test set. Any new work must avoid this and should report that it has done so.

Fifth, absolute performance remains poor. F1 of 0.306 on the full SD27, and considerably lower on the Bad and Ugly subsets, where GBFEN and SNFEN achieve F1 of 0.243 and 0.267 on Bad and 0.195 and 0.209 on Ugly respectively. Cappelli himself identifies severely degraded prints as the persistent open problem.

Both methods are released as open source in the pyfing repository.

---

## 10. Degraded, Damaged, and Worn Fingerprints

This section covers the comparatively small literature that addresses the specific condition motivating the present thesis.

### 10.1 Physical and Demographic Causes

Fingerprint quality is degraded by creases, dryness, shallow or worn ridges, injuries, and dirt. Creases in particular have been studied as an independent failure mode. **Crease detection from fingerprint images and its applications in elderly people**, published in Pattern Recognition in 2008, treats creases as structures to be detected and handled separately rather than as noise to be filtered, and is one of very few papers to name an age group as the target population.

The dermatological basis for age related degradation is established outside the biometrics literature. **Carmeli, Patish and Coleman (2003)**, in the Journals of Gerontology, describe the ageing hand, specifically the progressive loss of skin elasticity, firmness, and hydration attributable to declining collagen levels, which becomes marked from approximately 45 years of age.

Patent literature filed by biometric vendors distinguishes ridge line breakage caused by burns and chemicals, which is localised, from breakage caused by ageing and manual labour, which affects the entire ridge structure of the finger rather than a specific region. This distinction is significant because localised damage can in principle be inpainted from surrounding context, whereas global attenuation cannot.

The most systematic operational account of live capture degradation is the five class taxonomy of Karabulut et al. (2020), described in detail in Section 7.2 and derived from 11,541 expert labelled images captured at visa application centres. Its value for the present work is that it is organised by physical cause at the fingertip rather than by appearance in the image, and that four of its five classes are conditions of the skin rather than properties of the capture surface or the background. Dryness reduces the contact area so that ridges register only intermittently. Wetness increases the contact area so that valleys register alongside ridges and the ridge to valley contrast collapses. Damage in the form of scars, burns and cuts removes ridge structure outright. Dotting arises from sweat pore activity. Only blurring is primarily behavioural, arising from movement or excessive pressure. The authors further observe that most samples exhibit two or three of these conditions simultaneously, which is an important constraint on any simulation of the condition, since a degradation model that applies one effect at a time will not reproduce the joint distribution found in operational data.

This taxonomy overlaps substantially with the occupational and age related wear that motivates this thesis but does not coincide with it. Dryness, damage and their co-occurrence are precisely the mechanisms reported for manual labourers and elderly subjects. What the taxonomy does not separately name, and what the physiological literature indicates is central, is the progressive flattening of the ridge profile through abrasion, which reduces ridge amplitude across the whole fingertip without producing either the intermittent contact signature of transient dryness or the discrete boundaries of a scar. A wear specific degradation model therefore needs the Karabulut categories as a starting point and an amplitude attenuation term in addition to them.

### 10.2 The Indian Rural Fingerprint Research Line

This is the most directly relevant body of work.

**Puri, Narang, Tiwari, Vatsa and Singh (2010)**, presented at the International Conference on Ethics and Policy of Biometrics and International Data Sharing, published **On Analysis of Rural and Urban Indian Fingerprint Images**. This paper introduced the **Rural Indian Fingerprint Database**, comprising images captured from both rural and urban Indian populations, and analysed the quality difference between them. The rural population performs a greater amount of manual physical work, with the consequence that ridges, valleys, bifurcations, and minutiae are of poorer quality and recognition accuracy falls. This is the only publicly obtainable dataset explicitly constructed around occupational fingerprint degradation.

**Vatsa and colleagues (2010)** additionally published an analysis of the fingerprints of the Indian population using image quality, framed as a UIDAI case study, which connects the technical problem to the national identity deployment.

**Sankaran, Vatsa and Singh (2015)**, in IEEE Access volume 3, pages 653 to 665, published the **Multisensor Optical and Latent Fingerprint database**, known as MOLF. It comprises 19,200 fingerprints from 100 subjects captured by five methods, namely a Lumidigm Venus IP65 Shell sensor, a Secugen Hamster IV, a CrossMatch L-Scan Patrol, lifted latent fingerprints, and simultaneous latent fingerprints developed by black powder dusting. Manual latent annotations are provided. The same group published **Latent fingerprint matching: A survey** in IEEE Access volume 2 (2014), pages 982 to 1004, which remains the standard survey of the latent matching problem.

**Sankaran, Agarwal, Keshari, Ghosh, Sharma, Vatsa and Singh (2015)**, at BTAS, published the **IIITD Multi Surface Latent Fingerprint Database**, comprising 551 latent fingerprints from 51 subjects lifted from eight surface types, with mated slap galleries captured at 500 dots per inch.

**Malhotra, Vatsa, Singh, Morris and Noore (2023)**, in IEEE Transactions on Information Forensics and Security, published the **Multi Surface Multi Technique latent fingerprint database**, known as MUST. It contains more than 16,000 latent impressions from 120 unique fingers, rising to approximately 21,000 impressions including exemplar live scan and inked rolled prints and an extended gallery, acquired under 35 distinct scenarios across 39 subsets. Manually marked minutiae, acquisition resolution, and semantic segmentation masks are provided. The provision of minutiae ground truth makes MUST one of the very few datasets on which minutiae precision and recall can be computed without recourse to the withdrawn SD27.

The methodological line led by **Indu Joshi** with **Prem Kumar Kalra** at IIT Delhi, in collaboration with Antitza Dantcheva at Inria and Sumantra Dutta Roy, addresses directly the problem of applying enhancement models to this population. The published sequence is as follows.

**Joshi et al. (2019)**, WACV, latent fingerprint enhancement using generative adversarial networks, as described in Section 7.

**Joshi, Utkarsh, Kothari, Kurmi, Dantcheva, Dutta Roy and Kalra (2021a)**, at IJCNN, **Data Uncertainty Guided Noise-aware Preprocessing of Fingerprints**. This introduces a data uncertainty framework that enables a preprocessing model to quantify the noise present in the input image and to identify regions with background noise and poor ridge clarity. The uncertainty estimation techniques surveyed and applied include Monte Carlo dropout, deep ensembles, maximum softmax probability, and stochastic variational Bayesian inference. The paper explicitly cites Tiwari et al., Vatsa et al., and Puri et al. as establishing that state of the art matching performs poorly on the rural Indian population, and adopts that population as the evaluation target.

**Joshi et al. (2021b)**, at IJCNN, sensor invariant fingerprint region of interest segmentation using recurrent adversarial learning, evaluated across twelve public fingerprint databases.

**Joshi, Utkarsh, Singh, Dantcheva, Dutta Roy and Kalra (2022a)**, in Multimedia Tools and Applications, **On restoration of degraded fingerprints**. The framing of this paper is the most directly aligned with the present thesis, namely that state of the art matching achieves high accuracy on good quality fingerprints but that degraded fingerprints arising from poor skin condition remain unsolved. The acknowledgements record that a private rural Indian fingerprint database was shared by Professor Phalguni Gupta of IIT Kanpur, in addition to the public database.

**Joshi, Dhamija, Kumar and Kalra (2022b)**, in IEEE Sensors Letters, **Cross-Domain Consistent Fingerprint Denoising**, introducing **CDC-GAN**. The premise is that a fingerprint denoising model trained on synthetic data suffers a cross domain shift when applied to real degraded fingerprints. Their remedy is an unsupervised consistency regularisation loss applied in both source and target domains, enforcing that two fingerprint images sharing the same ridge structure but differing in contrast and ridge to valley clarity should produce similar outputs. They state this to be the first application of consistency regularisation in fingerprint denoising. Implementation details, which are useful as a reference point for planning experiments, are a ResNet based generative adversarial network with an encoder decoder generator totalling 14,141,634 trainable parameters, implemented in PyTorch version 1.11.0, trained with the Adam optimizer at a learning rate of 0.0002 with a batch size of 2, on four NVIDIA GTX 1080 Ti cards.

**Joshi, Prakash, Jaiswal, Kumar and Kalra (2022c)**, **Context-Aware Restoration of Noisy Fingerprints**. The argument is that although the classical literature firmly establishes the value of contextual information such as ridge orientation field, ridge spacing, and ridge frequency, most convolutional restoration models exploit spatial context only implicitly through the convolution operation, and that context must therefore be injected explicitly.

**Joshi, Prakash, Kumar, Dantcheva and Kalra (2023)**, in Multimedia Tools and Applications, **Unsupervised domain alignment of fingerprint denoising models using pseudo annotations**. This paper reports the benchmark numbers that any new work on this problem must exceed. After the proposed domain alignment, on the publicly available rural Indian fingerprint database, equal error rate improves from 7.30 to 6.10 using the Bozorth matcher, and from 5.96 to 5.31 using the Minutiae Cylinder Code matcher. Comparable improvements were obtained on IIITD MOLF and on the private rural fingerprint database.

**Joshi et al. (2023)**, chapter in Digital Image Enhancement and Reconstruction, Elsevier, **On estimating uncertainty of fingerprint enhancement models**, introducing a channel refinement unit within a GAN, referred to as CR-GAN. Evaluation used Dice score, Jaccard similarity, structural similarity index, and peak signal to noise ratio against ground truth, together with quality scores from a standard fingerprint quality assessment tool and fingerprint matching performance. An ablation study separately quantified the contribution of channel refinement in the generator and in the discriminator.

The collective position of this research line is that the principal obstacle is not architecture but **domain shift** between synthetic training degradation and real occupational degradation, and that uncertainty quantification is the route to trustworthy enhancement. Both positions are adopted and extended in the present thesis.

### 10.3 Other Work on Damaged and Incomplete Fingerprints

**Zhou, Wu and colleagues (2024)**, in Electronics volume 13, issue 14, article 2873, published **Recovery of Incomplete Fingerprints Based on Ridge Texture and Orientation Field**, proposing a dual stream network named IFSR. The mutilated fingerprint is first decomposed into cartoon and texture components by total variation decomposition. Local quality is then assessed using ridge frequency and directional features in order to identify the region requiring restoration and to generate a mask. The IFSR network comprises an orientation prediction branch guided by the fingerprint orientation field and a detail restoration branch guided by high quality fingerprint texture. Evaluation was on FVC2002 and FVC2004 together with an artificially damaged dataset, reporting equal error rates of 0.10, 0.12, and 0.20 percent on FVC2002 DB1, DB2, and DB4 respectively.

Work on **wet fingerprint denoising** for small area capacitive and optical sensors, exemplified by **PGT-Net**, a progressive guided multi task neural network, addresses a related but distinct degradation mode, namely moisture rather than abrasion, and uses proprietary paired datasets from a sensor manufacturer in which clean and wet impressions of the same finger are aligned.

Work applying convolutional long short term memory networks to the recognition of **deliberately damaged fingerprints** in a forensic context addresses obliteration and mutilation performed intentionally to evade identification, which overlaps technically with occupational wear even though the motivation differs.

---

## 11. Fingerprint Quality Assessment and Demographic Bias

### 11.1 NFIQ

Quality assessment is the mechanism by which enhancement can be evaluated without a clean ground truth reference, and it is therefore central to any evaluation protocol for real degraded fingerprints.

NIST published the first open source fingerprint quality assessment tool, **NFIQ**, in 2004, producing an integer score from 1 to 5 where 1 denotes the best quality.

**NFIQ 2**, developed from 2011 as a collaboration between NIST, the German Federal Office for Information Security, the German Federal Criminal Police Office, MITRE, Fraunhofer IGD, Hochschule Darmstadt, and Secunet, with subsequent contributions coordinated through ISO/IEC JTC 1 Subcommittee 37 Working Group 3, replaced this with a score in the range 0 to 100. The design objective is that the score should predict operational recognition performance rather than merely describe image appearance. A comprehensive literature survey produced 155 candidate quality features, which were pruned to 14 by analysing predictive power and redundancy using Spearman rank correlation and random forest variable importance. A random forest classifier maps the feature values to the unified score. The features are formally standardised in ISO/IEC 29794-4, and the NFIQ 2 implementation is the recognised reference implementation of that standard. NFIQ 2 is trained on flat, that is, not rolled, 500 dots per inch optical and inked fingerprints, a restriction that must be respected when using it. The nomenclature distinguishes NFIQ 2, meaning the general redesign documented in NISTIR 8382, from NFIQ 2.0 through 2.2, which denote specific software releases.

Extensions have been proposed for other capture modalities. **MCLFIQ**, mobile contactless fingerprint image quality, retrains the NFIQ 2 pipeline on a synthetic contactless fingerprint database and is evaluated using error versus discard characteristic curves on three real contactless databases.

### 11.2 Quality, Ageing, and Demographics

**Galbally, Haraksim and Beslay (2019)**, in IEEE Transactions on Information Forensics and Security volume 14, pages 1351 to 1365, published a study of age and ageing in fingerprint biometrics conducted at the European Commission Joint Research Centre on approximately 500,000 fingerprints from individuals aged 0 to 25 and 65 to 98, acquired operationally with 500 pixel per inch optical touch based scanners for national identity card issuance. Their principal findings were that children's fingerprint impressions show better quality than those of the elderly, and that quality rises rapidly between 0 and 12 years, stabilises through adulthood, and begins a linear decline at approximately 40 to 45 years. Because the dataset lacked subjects aged 26 to 64, the behaviour in that range was estimated by linear fitting rather than observed.

**Jain, Arora, Cao, Best-Rowden and Bhatnagar (2016)**, in the same journal, volume 12, pages 1501 to 1514, studied fingerprint recognition of young children on 309 subjects aged 0 to 5, captured with both a standard 500 pixel per inch reader and a custom 1270 pixel per inch reader. They concluded that children's fingerprints contain all the identity information required for accurate recognition provided the images are captured at sufficient quality and resolution.

**Haraksim, Galbally and Beslay (2019)**, in Pattern Recognition, developed a fingerprint growth model for mitigating the ageing effect on children's fingerprint matching, based on approximately 70,000 fingerprint pairs from children aged 5 to 16 captured 1 to 7 years apart. They verified that minutiae displacement due to growth follows an isotropic model, invariant to the distance of the minutia from the fingerprint centre, and used this to construct a compensating transformation.

**Galbally, Cepilovs, Blanco-Gonzalo, Ormiston, Miguel-Hurtado and Racz (2024)**, arXiv 2409.19992, published a large scale operational study of fingerprint quality and demographics. The database comprised 15,942 ten print records, that is 159,420 individual fingerprint samples, from individuals originating in 34 non European Union countries, captured between March and May 2022 at 115 visa issuing locations using a single FBI certified 500 dot per inch touch based optical scanner, the Cross Match Patrol ID, in a 4-4-2 slap sequence with up to three recapture attempts. Quality was measured with NFIQ 2. The findings are as follows.

Quality remains stable from age 12 through the first part of adulthood, then declines linearly from approximately 45 to 50 years, confirming the hypothesis left unverified in the 2019 study. Subjects over 65 present a substantial operational risk.

Female fingerprints score consistently and significantly lower than male fingerprints, with medians of approximately 49 and 59 respectively, and with higher variance. The authors attribute this to higher ridge density in smaller fingerprints interacting badly with a 500 dot per inch capture resolution and with post processing tuned to larger prints, rather than to any intrinsic difference in identity information content.

The right hand consistently produces better quality than the left, which the authors attribute to handedness. Ordered from lowest to highest quality, the fingers are little, ring, middle, index, thumb, consistently across both hands, which the authors attribute to the ergonomics of flat platen slap capture.

The proposed remedies are higher capture resolution, ridge density specific or age specific processing algorithms, ergonomic redesign of platens, and touchless capture for elderly subjects.

This study is the strongest available evidence that fingerprint recognition exhibits systematic performance variation across population segments and that the cause lies in acquisition and processing rather than in the fingerprints themselves. It provides the fairness framing for the present thesis, and it introduces a evaluation dimension, namely demographic stratification, that is entirely absent from the enhancement literature.

### 11.3 Touchless Capture as an Alternative Remedy

**Priesnitz, Rathgeb, Buchmann, Busch and Margraf (2021)**, in the EURASIP Journal on Image and Video Processing, published an overview of touchless two dimensional fingerprint recognition. **Orandi et al. (2020)**, NIST Technical Note 8315, evaluated the operational impact of contactless fingerprint imagery on matcher performance, and **Libert et al. (2020)**, NIST Technical Note 8307, assessed contactless to contact capture interoperability. The consensus is that touchless capture removes the skin to platen interaction problem, which is the dominant cause of poor quality in elderly subjects, but introduces new variability from uncontrolled illumination, scale, and pose, and remains less mature than contact based capture.

---

## 12. Synthetic Fingerprint Generation

Because paired clean and degraded real fingerprints cannot be collected at scale, synthetic generation is not an optional convenience in this field but a structural necessity. The methods below supply the training data for most of the deep learning work reviewed above.

**SFinGe**, developed by Cappelli at the University of Bologna, is the original synthetic fingerprint generator and the reference method. It generates a master fingerprint from a randomly sampled directional map and density map, grows a ridge pattern from randomly placed seeds, and then simulates acquisition by applying displacement, distortion, varying skin condition, noise, and a rendering model. SFinGe generated the DB4 synthetic subsets of FVC2002, FVC2004, and FVC2006. Cappelli's chapter on fingerprint synthesis in the Handbook of Fingerprint Recognition is the definitive description.

**Anguli** is an open source C++ reimplementation of the principal SFinGe algorithms. It generates the clean master print alongside each degraded variant, which means the paired ground truth is available by construction. Its degradation model includes intra ridge noise, scratch noise, background noise, and other random noise, together with elastic transformation, blur, resolution reduction, and contrast change. Anguli generated the ChaLearn dataset and is the practical default for anyone needing paired data.

**Wyzykowski, Segundo and colleagues (2020, 2022)**, in IET Biometrics, proposed a hybrid multiresolution generator. Anguli is first extended to produce ridge maps with randomised ridge flow frequency, with segmented ridges of dynamically varying thickness. Pores and scratches are then added following a distribution learned from real high resolution images, producing what the authors term Level 3 master fingerprints. Acquisition is simulated by randomly cropping according to a displacement distribution learned from a real database, producing seed images. Finally a **CycleGAN** translates the seed images into realistic textured fingerprints. The output database, referred to as **L3-SF**, contains 740 identities at 1200 dots per inch and is publicly released. The motivation was explicitly that public access to existing high resolution databases had been discontinued.

**Engelsma, Grosz and Jain (2022)**, in **PrintsGAN**, addressed a different limitation, namely that existing public datasets contain few identities and few impressions per identity, which prevents the training of deep fixed length embedding models. PrintsGAN generates unique fingerprints together with multiple impressions of each, producing a released database of 525,000 images comprising 35,000 distinct fingers with 15 impressions each. The evaluation demonstrated the utility of the synthetic data rather than merely its realism. A DeepPrint embedding model pretrained on PrintsGAN data and fine tuned on 25,000 real prints from NIST SD302 achieved a true accept rate of 87.03 percent at a false accept rate of 0.01 percent on NIST SD4, compared with 73.37 percent when trained on SD302 alone. The comparison against SFinGe pretraining showed PrintsGAN to be superior on all three test databases. This paper established the pretrain on synthetic, fine tune on real methodology as standard.

**Bahmani, Plesh, Johnson, Schuckers and Swyka (2021)**, at ICIP, examined high fidelity fingerprint generation from the perspectives of quality, uniqueness, and privacy, raising the question of whether synthetic identities are genuinely distinct.

**Joshi, Dabouei, Nasrabadi and Dawson (2023)** proposed synthetic latent fingerprint generation using style transfer, an alternative to compositing onto background textures.

Diffusion based generators are described in Section 8.3.

---

## 13. Datasets

### 13.1 Consolidated Dataset Table

| Dataset | Content | Size | Access status | Used by |
|---|---|---|---|---|
| FVC2000 DB1-DB4 | Plain live scan, optical, capacitive, synthetic | 100 fingers x 8 per DB (A subsets), 10 x 8 (B subsets) | B subsets free; A subsets bundled with the Handbook | Classical era baselines |
| FVC2002 DB1-DB4 | As above, subjects instructed to create difficult impressions | Same structure | Same | Cappelli training set, most classical work |
| FVC2004 DB1-DB4 | As above, includes thermal sweep sensor | Same structure | Same | FingerNet slap evaluation, Cappelli training |
| FVC2006 DB1-DB4 | Includes 250 dpi 96x96 electric field sensor | 150 fingers x 12 | Two year licence via ATVS | Quality studies |
| NIST SD4 | 8 bit grey rolled prints, balanced pattern classes | 2000 pairs | Discontinued | PrintsGAN evaluation, historical work |
| NIST SD9, SD10 | Mated card pairs, supplemental card data | Various | Discontinued | Historical |
| NIST SD14 | Mated fingerprint card pairs 2 | Approximately 27000 pairs | Status uncertain, treat as unavailable | DNUNets, FingerGAN, SFP training |
| NIST SD27 | Real crime scene latents with expert validated minutiae, graded Good, Bad, Ugly | 258 latents, 88 / 85 / 85 | Withdrawn January 2017 | The principal benchmark of the entire latent literature |
| NIST SD300 | Scanned ink cards, rolled and plain | 888 subjects x 10 fingers x 2 | Request form | Quality studies |
| NIST SD301 | Multi sensor live capture plus latents | 51 subjects x 10 fingers x 14-15 | Request form | Recent work |
| NIST SD302 | Multi sensor live capture, 15 sensor types, plus SD302E latents | 200 subjects x 10 fingers x 12-18 | Request form, institutional email | PrintsGAN fine tuning, recent work |
| IIITD MOLF | Five capture methods, Indian subjects, latents with annotation | 19200 images, 100 subjects | Signed licence via IAB | Dabouei, Joshi line, MLFGNet |
| IIITD MSLFD | Latents from eight surface types | 551 latents, 51 subjects | Signed licence via IAB | Surface variation studies |
| MUST | Latents under 35 scenarios with minutiae and mask annotation | Approx 21000 impressions, 120 fingers | Signed licence via IAB | MLFGNet |
| Rural Indian Fingerprint DB | Rural and urban Indian population, occupational degradation | Not stated in accessible sources | Via IAB | Joshi line, the benchmark for this thesis |
| SOCOFing | Real prints plus synthetically altered versions simulating obliteration, central rotation, z cut, at three difficulty levels | 6000 real, 600 subjects x 10 fingers | Free on Kaggle | Damage classification work |
| CASIA-FingerprintV5 | Population includes workers and waiters, difficult impressions requested | 500 subjects x 8 fingers x 5 | Registration at idealtest.org | Various |
| ChaLearn LAP Track 3 | Paired clean and degraded synthetic | 84000 pairs training | Free download | Mansar, FPD-M-Net, denoising literature |
| L3-SF | High resolution synthetic with pores and scratches | 740 identities at 1200 dpi | Public | Pore and Level 3 work |
| PrintsGAN | Synthetic, multiple impressions per identity | 525000 images, 35000 fingers | Public | Embedding pretraining |
| LFIW | Latent in the wild, contact, contactless and fingerphoto | 13180 samples, 1318 instances, 132 subjects | Request | Post SD27 replacement effort |
| GCDB | Spanish Guardia Civil casework latents with extended minutia type ground truth | Not stated | Via ATVS | Extended minutiae work |
| ISPFDv1, ISPFDv2 | Smartphone fingerphotos with live scan mates | v1: 64 subjects, 128 classes; v2: over 17000 finger selfies, 304 fingers | Signed licence via IAB | Touchless work |
| MMFV | Finger video for contactless capture | 3792 videos, 336 classes | Signed licence via IAB | Contactless work |
| PolyU low resolution contactless | Webcam captured contactless | 1466 images, 156 subjects, two sessions | PolyU request | Low resolution contactless |
| FFE benchmark | Frequency estimation benchmark with ground truth | 60 images used in training by Cappelli | Associated with Cappelli 2024b | Frequency estimation |

### 13.2 The Availability Crisis

An observation that must appear in any current review is that the field's principal benchmark has been unavailable for nine years. NIST Special Database 27, containing 258 real crime scene latents with expert validated minutiae, was withdrawn in January 2017 on the grounds that it lacked the documentation required by NIST for distribution. NIST Special Databases 4, 9, and 10 have likewise been discontinued.

Despite this, SD27 remains the evaluation set in papers published in 2024, 2025, and 2026, including the current state of the art. This is possible only because established groups retain legacy copies. The practical consequences are that new entrants to the field cannot reproduce the published numbers, that comparisons are not verifiable, and that any new work must either negotiate access to legacy copies or construct an alternative protocol.

Explicit responses to this situation in the literature include the construction of the GCDB extended minutiae dataset by the ATVS group at Universidad Autonoma de Madrid, who state that they attempted to use SD27, found distribution discontinued, and therefore generated and released a comparable dataset from Spanish Guardia Civil casework with forensic expert ground truth; the construction of the Latent Fingerprint in the Wild database; and the shift by Wang et al. (2025) to evaluating on MOLF and MUST. The present thesis follows the last of these.

---

## 14. Evaluation Practice

Evaluation practice in this literature is inconsistent, and the inconsistency is itself a finding.

**Image similarity metrics**, namely mean squared error, peak signal to noise ratio, structural similarity index, Dice coefficient, and Jaccard similarity, are used wherever a clean ground truth exists, which in practice means synthetic data. These were the official metrics of the ChaLearn challenge. Their weakness is that they measure pixel agreement rather than identity information, and an enhancement that is pixel accurate but minutiae destructive will score well.

**Quality metrics**, principally NFIQ and NFIQ 2, require no reference image and can therefore be applied to real degraded data. Schuch, Schulz and Busch (2018) established that the relationship between quality metrics and matching performance is not straightforward, so quality improvement alone is insufficient evidence.

**Minutiae detection metrics**, namely precision, recall, and F1 score against expert marked ground truth, occupy an intermediate position and are arguably the most informative single measure, since minutiae are what matchers consume. Cappelli's protocol, using a distance threshold of 14 pixels, a direction threshold of pi over nine, and reporting both type exact and type agnostic correspondence, is the most carefully specified version. This protocol requires a dataset with minutiae ground truth, which currently means SD27, MOLF, MUST, or GCDB.

**Matching metrics**, namely equal error rate, true accept rate at a specified false accept rate, and rank N identification accuracy summarised in cumulative match characteristic curves, are the operationally meaningful measures. The matchers used in the academic literature are BOZORTH3 from the NIST Biometric Image Software suite, the Minutiae Cylinder Code, and commercially the VeriFinger SDK. Minutiae extraction is performed by MINDTCT from NBIS, by FingerJetFX, or by VeriFinger.

**Human expert evaluation**, in which forensic examiners rank the outputs of competing methods, is rare but was used by Cappelli as a supplementary validation.

Two evaluation dimensions are effectively absent from the literature and are proposed as contributions of the present thesis. The first is **failure to acquire and failure to enrol rate reduction**, which is the metric that connects directly to the deployment problem of exclusion from services. The second is **demographic stratification** of enhancement performance, which follows from Galbally et al. (2024) but has not been applied to enhancement methods.

---

## 15. Synthesis and Identified Research Gaps

The literature supports the following synthesis.

The estimation of local ridge orientation and local ridge frequency is the invariant core of the problem. Every method from 1988 to 2026, whether hand designed or learned, either estimates these quantities explicitly or learns a representation that encodes them implicitly. The most recent state of the art was achieved by improving these estimates and applying a deliberately simple enhancement stage, which indicates that the estimation problem, not the enhancement problem, is where remaining performance lies.

Architectural sophistication has reached and passed the point of diminishing returns. A generative adversarial network trained on 130,000 images for an unreported duration is outperformed by a five level encoder decoder trained on 360 images for 25 minutes. Future work should be sceptical of proposals whose novelty resides principally in architecture.

Generative reconstruction of missing ridge structure is a double edged capability. It is what allows dictionary methods and GANs to produce plausible output on severely degraded regions, and it is simultaneously the mechanism by which spurious minutiae are introduced. Precision rather than recall is the binding constraint, and no published method offers a principled mechanism for declining to reconstruct where the evidence is insufficient.

The following gaps are identified.

**Gap 1. The degradation model does not match the target population.** Essentially all supervised methods train on synthetic degradation consisting of blur, scratches, occlusion, elastic distortion, and background compositing. This is a model of crime scene latent degradation. Occupational and age related wear is physically different, consisting of ridge amplitude attenuation, ridge profile flattening, persistent flexion creases, dryness induced ridge fragmentation, and pressure dependent partial contact. No published degradation model is grounded in this physiology. The Joshi research line has correctly diagnosed the resulting domain shift and has addressed it by domain alignment applied after the fact, rather than by correcting the degradation model itself. Karabulut et al. avoid the problem by a third route, discarding pairing entirely in favour of unpaired translation, which removes the domain shift at the cost of removing all structural supervision. These three responses to the same underlying problem, namely corrected synthesis, post hoc alignment, and unpaired translation, have not been compared against one another on a common benchmark.

**Gap 2. Identity preservation is asserted rather than measured.** Two distinct concerns recur in the generative literature without being resolved. Dabouei et al. recognised as early as 2018 that a generative model can output a realistic fingerprint belonging to a different identity, and added an explicit identity preservation constraint in response. FingerGAN addressed a related concern by making the minutiae weighted skeleton the output representation, so that the objective optimises minutia information directly. Yet the most operationally relevant recent work, Karabulut et al., evaluates only by asking a distortion classifier whether the output still looks distorted, and reports that scars and cuts are filled with respect to the surrounding ridge pattern without ever testing whether that filling introduces minutiae that do not belong to the subject. Cycle consistency provides no protection here, since a round trip can be satisfied without the intermediate representation corresponding to real ridge structure. The evaluation gap is therefore not merely that better metrics would be desirable, but that a published method restores fingerprints from a real civil identity population using a mechanism known to fabricate structure, and no reported measurement can distinguish successful restoration from convincing fabrication.

**Gap 3. There is no mechanism for principled abstention.** Uncertainty estimation has been introduced to this field by Joshi and colleagues, using Monte Carlo dropout, deep ensembles, and variational approaches, and has been used to quantify noise and to weight the loss. It has not been used to make the enhancement model decline to reconstruct, nor to filter the extracted minutiae before matching. Given that precision is the binding constraint, abstention is the most direct available intervention.

**Gap 4. Evaluation is not aligned with the deployment problem.** The literature reports image similarity metrics, minutiae F1, and matching accuracy on forensic benchmarks. It does not report failure to acquire reduction, and it does not stratify by demographic group, occupation, or severity, despite strong evidence from the quality literature that performance varies systematically across these dimensions.

**Gap 5. The benchmark is unobtainable.** Nine years after the withdrawal of NIST SD27, the field continues to report on it. A reproducible protocol built entirely on obtainable data is needed, and the recent shift to MOLF and MUST by Wang et al. indicates that this transition is beginning.

**Gap 6. Multiple impressions are not exploited.** Operational deployments already capture multiple impressions per finger, since recapture on quality failure is standard practice and was applied up to three times in the Galbally et al. operational study. The enhancement literature treats the problem as single image restoration throughout. Fusion of several degraded impressions of the same finger into one restored template is a natural and largely unexplored direction with an obvious deployment path.

---

## 16. References

Note on links. Where a paper is available as an open access preprint or an author hosted copy, that link is given, since publisher pages are often paywalled.

### Books and Surveys

1. Maltoni, D., Maio, D., Jain, A. K. and Feng, J. (2022). Handbook of Fingerprint Recognition, 3rd edition. Springer. https://link.springer.com/book/10.1007/978-3-030-83624-5

2. Schuch, P., Schulz, S. and Busch, C. (2018). Survey on the impact of fingerprint image enhancement. IET Biometrics, 7(2), 102-115. https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/iet-bmt.2016.0088

3. Sankaran, A., Vatsa, M. and Singh, R. (2014). Latent fingerprint matching: A survey. IEEE Access, 2, 982-1004.

4. Maltoni, D. and Cappelli, R. (2008). Fingerprint recognition. In Handbook of Biometrics, Springer, 23-42.

### Classical Enhancement

5. O'Gorman, L. and Nickerson, J. V. (1988). Matched filter design for fingerprint image enhancement. ICASSP 1988.

6. Sherlock, B. G., Monro, D. M. and Millard, K. (1994). Fingerprint enhancement by directional Fourier filtering. IEE Proceedings Vision, Image and Signal Processing, 141, 87-94.

7. Kamei, T. and Mizoguchi, M. (1995). Image filter design for fingerprint enhancement. International Symposium on Computer Vision.

8. Hong, L., Wan, Y. and Jain, A. K. (1998). Fingerprint image enhancement: algorithm and performance evaluation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(8), 777-789.

9. Erol, A., Halici, U. and Ongun, G. (1999). Feature selective filtering for ridge extraction. In Intelligent Biometric Techniques in Fingerprint and Face Recognition, CRC Press, 193-215.

10. Almansa, A. and Lindeberg, T. (2000). Fingerprint enhancement by shape adaptation of scale-space operators with automatic scale selection. IEEE Transactions on Image Processing, 9, 2027-2042.

11. Greenberg, S., Aladjem, M., Kogan, D. and Dimitrov, I. (2000). Fingerprint image enhancement using filtering techniques. ICPR 2000.

12. Willis, A. J. and Myers, L. (2001). A cost-effective fingerprint recognition system for use with low-quality prints and damaged fingertips. Pattern Recognition, 34, 255-270.

13. Hsieh, C.-T., Lai, E. and Wang, Y.-C. (2003). An effective algorithm for fingerprint image enhancement based on wavelet transform. Pattern Recognition, 36, 303-312.

14. Yang, J., Liu, L., Jiang, T. and Fan, Y. (2003). A modified Gabor filter design method for fingerprint image enhancement. Pattern Recognition Letters, 24, 1805-1817.

15. Nakamura, T., Hirooka, M., Fujiwara, H. and Sumi, K. (2004). Fingerprint image enhancement using a parallel ridge filter. ICPR 2004.

16. Chikkerur, S., Govindaraju, V. and Cartwright, A. N. (2005). Fingerprint image enhancement using STFT analysis. Pattern Recognition and Image Analysis.

17. Chikkerur, S., Cartwright, A. N. and Govindaraju, V. (2007). Fingerprint enhancement using STFT analysis. Pattern Recognition, 40(1), 198-211.

18. Wu, C. and Govindaraju, V. (2006). Singularity preserving fingerprint image adaptive filtering. ICIP 2006.

19. Jirachaweng, S. and Areekul, V. (2007). Fingerprint enhancement based on discrete cosine transform. Advances in Biometrics.

20. Fronthaler, H., Kollreider, K. and Bigun, J. (2008). Local features for enhancement and minutiae extraction in fingerprints. IEEE Transactions on Image Processing, 17, 354-363.

21. Wang, W., Li, J., Huang, F. and Feng, H. (2008). Design and implementation of Log-Gabor filter in fingerprint image enhancement. Pattern Recognition Letters, 29, 301-308.

22. Zhao, Q., Zhang, L., Zhang, D., Huang, W. and Bai, J. (2009). Curvature and singularity driven diffusion for oriented pattern enhancement with singular points. CVPR 2009.

23. Cappelli, R., Maio, D. and Maltoni, D. (2009). Semi-automatic enhancement of very low quality fingerprints. ISPA 2009, 678-683.

24. Yoon, S., Feng, J. and Jain, A. K. (2010). On latent fingerprint enhancement. SPIE Biometric Technology for Human Identification VII, 7667.

25. Yoon, S., Feng, J. and Jain, A. K. (2011). Latent fingerprint enhancement via robust orientation field estimation. IJCB 2011.

26. Gottschlich, C. (2012). Curved-region-based ridge frequency estimation and curved Gabor filters for fingerprint image enhancement. IEEE Transactions on Image Processing, 21, 2220-2227.

27. Gottschlich, C. and Schonlieb, C.-B. (2012). Oriented diffusion filtering for enhancing low-quality fingerprint images. IET Biometrics, 1(2), 105-113.

28. Turroni, F., Cappelli, R. and Maltoni, D. (2012). Fingerprint enhancement using contextual iterative filtering. ICB 2012.

29. Sutthiwichaiporn, P. and Areekul, V. (2013). Adaptive boosted spectral filtering for progressive fingerprint enhancement. Pattern Recognition, 46, 2465-2486.

30. Daugman, J. G. (1985). Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters. Journal of the Optical Society of America A, 2, 1160-1169.

31. Buades, A., Le, T. M., Morel, J.-M. and Vese, L. A. (2010). Fast cartoon plus texture image filters. IEEE Transactions on Image Processing, 19, 1978-1986.

### Dictionary and Sparse Representation

32. Rama, R. K. N. V. and Namboodiri, A. M. (2011). Fingerprint enhancement using hierarchical Markov random fields. IJCB 2011.

33. Feng, J., Zhou, J. and Jain, A. K. (2013). Orientation field estimation for latent fingerprint enhancement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35, 925-940.

34. Yang, X., Feng, J. and Zhou, J. (2014). Localized dictionaries based orientation field estimation for latent fingerprints. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36, 955-969.

35. Cao, K., Liu, E. and Jain, A. K. (2014). Segmentation and enhancement of latent fingerprints: a coarse to fine ridge structure dictionary. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36, 1847-1859.

36. Liu, M., Chen, X. and Wang, X. (2015). Latent fingerprint enhancement via multi-scale patch based sparse representation. IEEE Transactions on Information Forensics and Security, 10(1), 6-15.

37. Chaidee, W., Horapong, K. and Areekul, V. (2018). Filter design based on spectral dictionary for latent fingerprint pre-enhancement. ICB 2018.

### Deep Learning and Generative Methods

38. Schuch, P., Schulz, S. and Busch, C. (2016). De-convolutional autoencoder for enhancement of fingerprint samples. IPTA 2016.

39. Svoboda, J., Monti, F. and Bronstein, M. M. (2017). Generative convolutional networks for latent fingerprint reconstruction. IJCB 2017. Code: https://github.com/JanSvob/fingerprint_gae

40. Tang, Y., Gao, F., Feng, J. and Liu, Y. (2017). FingerNet: An unified deep network for fingerprint minutiae extraction. IJCB 2017, 108-116. https://arxiv.org/abs/1709.02228

41. Dabouei, A., Soleymani, S., Kazemi, H., Iranmanesh, S. M., Dawson, J. and Nasrabadi, N. M. (2018). ID preserving generative adversarial network for partial latent fingerprint reconstruction. BTAS 2018. https://arxiv.org/abs/1808.00035

42. Li, J., Feng, J. and Kuo, C.-C. J. (2018). Deep convolutional neural network for latent fingerprint enhancement. Signal Processing: Image Communication, 60, 52-63.

43. Mansar, Y. (2018). Deep end-to-end fingerprint denoising and inpainting. arXiv 1807.11888. https://arxiv.org/abs/1807.11888 Code: https://github.com/CVxTz/fingerprint_denoising

44. Adiga V., S. and Sivaswamy, J. (2018). FPD-M-Net: Fingerprint image denoising and inpainting using M-Net based convolutional neural networks. ECCV Workshops 2018. https://arxiv.org/abs/1812.10191

45. Escobar, M. et al. (2021). ChaLearn Looking at People: Inpainting and denoising challenges. arXiv 2106.13071. https://arxiv.org/abs/2106.13071

46. Joshi, I., Anand, A., Vatsa, M., Singh, R., Dutta Roy, S. and Kalra, P. (2019). Latent fingerprint enhancement using generative adversarial networks. WACV 2019, 895-903.

47. Qian, P., Li, A. and Liu, M. (2019). Latent fingerprint enhancement based on DenseUNet. ICB 2019.

48. Liu, Y., Tang, Y., Li, R. and Feng, J. (2019). Cooperative orientation generative adversarial network for latent fingerprint enhancement. ICB 2019.

49. Huang, X., Qian, P. and Liu, M. (2020). Latent fingerprint image enhancement based on progressive generative adversarial network. CVPR Workshops 2020.

50. Wong, W. J. and Lai, S.-H. (2020). Multi-task CNN for restoring corrupted fingerprint images. Pattern Recognition, 101, 107203.

51. Cao, K., Nguyen, D.-L., Tymoszek, C. and Jain, A. K. (2020). End-to-end latent fingerprint search. IEEE Transactions on Information Forensics and Security, 15, 880-894.

52. Horapong, K., Srisutheenon, K. and Areekul, V. (2021). Progressive and corrective feedback for latent fingerprint enhancement using boosted spectral filtering and spectral autoencoder. IEEE Access, 9, 96288-96308.

53. Liu, M. and Qian, P. (2021). Automatic segmentation and enhancement of latent fingerprints using deep nested U-Nets. IEEE Transactions on Information Forensics and Security, 16, 1709-1719.

54. Zhu, Y., Yin, X. and Hu, J. (2023). FingerGAN: A constrained fingerprint generation scheme for latent fingerprint enhancement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7), 8358-8371. https://arxiv.org/abs/2206.12885 Code: https://github.com/HubYZ/LatentEnhancement

55. Kriangkhajorn, S., Horapong, K. and Areekul, V. (2024). Spectral filter predictor for progressive latent fingerprint restoration. IEEE Access, 12, 66773-66800.

56. Pramukha, R. N., Akhila, P. and Koolagudi, S. G. (2024). End-to-end latent fingerprint enhancement using multi-scale generative adversarial network. Pattern Recognition Letters, 184, 169-175.

57. Karabulut, D., Tertychnyi, P., Arslan, H. S., Ozcinar, C., Nasrollahi, K., Valls, J., Vilaseca, J., Moeslund, T. B. and Anbarjafari, G. (2020). Cycle-consistent generative adversarial neural networks based low quality fingerprint enhancement. Multimedia Tools and Applications, 79, 18569-18589. https://doi.org/10.1007/s11042-020-08750-8

58. Tertychnyi, P., Ozcinar, C. and Anbarjafari, G. (2018). Low-quality fingerprint classification using deep neural network. IET Biometrics, 7(6), 550-556.

59. Jia, Z., Huang, C., Wang, Z., Fei, H., Wu, S. and Feng, J. (2024). Finger recovery transformer: Towards better incomplete fingerprint identification. IEEE Transactions on Information Forensics and Security, 19, 8860-8874.

60. Wang, Y., Qi, Z., Fu, S. and Hu, M. (2025). A triple-branch network for latent fingerprint enhancement guided by orientation fields and minutiae. arXiv 2504.15105. https://arxiv.org/abs/2504.15105

### The Simplicity Line

61. Cappelli, R. (2023). Unveiling the power of simplicity: Two remarkably effective methods for fingerprint segmentation. IEEE Access, 11, 144530-144544.

62. Cappelli, R. (2024). Exploring the power of simplicity: A new state-of-the-art in fingerprint orientation field estimation. IEEE Access, 12, 55998-56018.

63. Cappelli, R. (2024). No feature left behind: Filling the gap in fingerprint frequency estimation. IEEE Access, 12, 153605-153617.

64. Cappelli, R. (2026). Unleashing the power of simplicity: A minimalist strategy for state-of-the-art fingerprint enhancement. arXiv 2603.19004. https://arxiv.org/abs/2603.19004 Code: https://github.com/raffaele-cappelli/pyfing

### Degraded, Rural and Indian Fingerprints

65. Puri, C., Narang, K., Tiwari, A., Vatsa, M. and Singh, R. (2010). On analysis of rural and urban Indian fingerprint images. International Conference on Ethics and Policy of Biometrics and International Data Sharing.

66. Sankaran, A., Vatsa, M. and Singh, R. (2015). Multisensor optical and latent fingerprint database. IEEE Access, 3, 653-665. Dataset: https://iab-rubric.org/resources/biometric-datasets/fingerprint

67. Sankaran, A., Agarwal, A., Keshari, R., Ghosh, S., Sharma, A., Vatsa, M. and Singh, R. (2015). Latent fingerprint from multiple surfaces: Database and quality analysis. BTAS 2015.

68. Malhotra, A., Vatsa, M., Singh, R., Morris, K. B. and Noore, A. (2023). Multi-Surface Multi-Technique (MUST) latent fingerprint database. IEEE Transactions on Information Forensics and Security.

69. Joshi, I., Utkarsh, A., Kothari, R., Kurmi, V. K., Dantcheva, A., Dutta Roy, S. and Kalra, P. K. (2021). Data uncertainty guided noise-aware preprocessing of fingerprints. IJCNN 2021, 1-8.

70. Joshi, I., Utkarsh, A., Kothari, R., Kurmi, V. K., Dantcheva, A., Dutta Roy, S. and Kalra, P. K. (2021). Sensor-invariant fingerprint ROI segmentation using recurrent adversarial learning. IJCNN 2021, 1-8.

71. Joshi, I., Utkarsh, A., Singh, P., Dantcheva, A., Dutta Roy, S. and Kalra, P. K. (2022). On restoration of degraded fingerprints. Multimedia Tools and Applications. https://link.springer.com/article/10.1007/s11042-021-11863-3

72. Joshi, I., Dhamija, T., Kumar, R. and Kalra, P. K. (2022). Cross-domain consistent fingerprint denoising. IEEE Sensors Letters. https://www.cse.iitd.ac.in/~sumantra/publications/sensorsl22_cross_domain.pdf

73. Joshi, I., Prakash, T., Jaiswal, B. S., Kumar, R. and Kalra, P. K. (2022). Context-aware restoration of noisy fingerprints.

74. Joshi, I., Prakash, T., Kumar, R., Dantcheva, A. and Kalra, P. K. (2023). Unsupervised domain alignment of fingerprint denoising models using pseudo annotations. Multimedia Tools and Applications. https://link.springer.com/article/10.1007/s11042-023-15513-8

75. Joshi, I., Utkarsh, A., Kothari, R., Kurmi, V. K., Dantcheva, A., Dutta Roy, S. and Kalra, P. K. (2023). On estimating uncertainty of fingerprint enhancement models. In Digital Image Enhancement and Reconstruction, Elsevier.

76. Zhou, et al. (2024). Recovery of incomplete fingerprints based on ridge texture and orientation field. Electronics, 13(14), 2873. https://www.mdpi.com/2079-9292/13/14/2873

77. Crease detection from fingerprint images and its applications in elderly people (2008). Pattern Recognition. https://www.sciencedirect.com/science/article/abs/pii/S0031320308004020

78. PGT-Net: Progressive guided multi-task neural network for small-area wet fingerprint denoising and recognition. arXiv 2308.07024. https://arxiv.org/abs/2308.07024

79. Damaged fingerprint recognition by convolutional long short-term memory networks for forensic purposes. arXiv 2012.15041. https://arxiv.org/abs/2012.15041

### Quality, Ageing and Demographics

80. Tabassi, E., Olsen, M., Bausinger, O., Busch, C., Figlarz, A., Fiumara, G., Henniger, O., Merkle, J., Ruhland, T., Schiel, C. and Schwaiger, M. NFIQ 2: NIST Fingerprint Image Quality. NISTIR 8382. https://www.nist.gov/services-resources/software/nfiq-2 Code: https://github.com/usnistgov/NFIQ2

81. ISO/IEC 29794-4. Information technology, Biometric sample quality, Part 4: Finger image data.

82. Galbally, J., Haraksim, R. and Beslay, L. (2019). A study of age and ageing in fingerprint biometrics. IEEE Transactions on Information Forensics and Security, 14, 1351-1365.

83. Haraksim, R., Galbally, J. and Beslay, L. (2019). Fingerprint growth model for mitigating the ageing effect on children's fingerprints matching. Pattern Recognition, 18, 614-628.

84. Jain, A. K., Arora, S. S., Cao, K., Best-Rowden, L. and Bhatnagar, A. (2016). Fingerprint recognition of young children. IEEE Transactions on Information Forensics and Security, 12, 1501-1514.

85. Galbally, J., Cepilovs, A., Blanco-Gonzalo, R., Ormiston, G., Miguel-Hurtado, O. and Racz, I. S. (2024). A large-scale operational study of fingerprint quality and demographics. arXiv 2409.19992. https://arxiv.org/abs/2409.19992

86. Galbally, J., Cepilovs, A., Blanco-Gonzalo, R., Ormiston, G., Miguel-Hurtado, O. and Racz, I. S. (2023). Fingerprint quality per individual finger type: A large-scale study on real operational data. IWBF 2023.

87. Gafurov, D., Bours, P., Yang, B. and Busch, C. (2010). Impact of finger type in fingerprint authentication. International Conference on Security Technology, Disaster Recovery and Business Continuity.

88. Kukula, E. P., Blomeke, C. R., Modi, S. K. and Elliott, S. J. (2009). Effect of human-biometric sensor interaction on fingerprint matching performance, image quality and minutiae count. International Journal of Computer Applications in Technology, 34, 270-277.

89. Carmeli, E., Patish, H. and Coleman, R. (2003). The aging hand. Journals of Gerontology Series A, 58A(2), M146-M152.

90. Priesnitz, J., Rathgeb, C., Buchmann, N., Busch, C. and Margraf, M. (2021). An overview of touchless 2D fingerprint recognition. EURASIP Journal on Image and Video Processing.

91. Orandi, S., Libert, J., Bandini, B., Ko, K., Grantham, J. D. and Watson, C. I. (2020). Evaluating the operational impact of contactless fingerprint imagery on matcher performance. NIST Technical Note 8315.

92. MCLFIQ: Mobile contactless fingerprint image quality. arXiv 2304.14123. https://arxiv.org/abs/2304.14123

### Synthetic Generation

93. Cappelli, R. (2022). Fingerprint synthesis. In Handbook of Fingerprint Recognition, Springer, 385-426.

94. Cappelli, R. SFinGe. University of Bologna Biometric System Laboratory. http://biolab.csr.unibo.it/

95. Ansari, A. H. Anguli: Synthetic fingerprint generator. Open source implementation of SFinGe.

96. Engelsma, J. J., Grosz, S. A. and Jain, A. K. (2022). PrintsGAN: Synthetic fingerprint generator. arXiv 2201.03674. https://arxiv.org/abs/2201.03674

97. Wyzykowski, A. B. V., Segundo, M. P. and Lemes, R. P. (2022). Multiresolution synthetic fingerprint generation. IET Biometrics. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/bme2.12083 and arXiv 2002.03809

98. Bahmani, K., Plesh, R., Johnson, P., Schuckers, S. and Swyka, T. (2021). High fidelity fingerprint generation: Quality, uniqueness, and privacy. ICIP 2021, 3018-3022.

99. Grosz, S. A. and Jain, A. K. (2024). Universal fingerprint generation: Controllable diffusion model with multimodal conditions. IEEE Transactions on Pattern Analysis and Machine Intelligence.

100. Grabovski, F., Yasur, L., Hacmon, Y., Nisimov, L. and Nimrod, S. (2024). DiffFinger: Advancing synthetic fingerprint generation through denoising diffusion probabilistic models. arXiv 2405.04538. https://arxiv.org/abs/2405.04538

101. Joshi, A. S., Dabouei, A., Nasrabadi, N. M. and Dawson, J. (2023). Synthetic latent fingerprint generation using style transfer.

### Datasets, Software and Benchmarks

102. Maio, D., Maltoni, D., Cappelli, R., Wayman, J. L. and Jain, A. K. (2002). FVC2002: Second fingerprint verification competition. ICPR 2002. http://bias.csr.unibo.it/fvc2002/

103. Cappelli, R., Maio, D., Maltoni, D., Wayman, J. L. and Jain, A. K. (2006). Performance evaluation of fingerprint verification systems. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28, 3-18.

104. Garris, M. D. and McCabe, R. M. (2000). NIST Special Database 27: Fingerprint minutiae from latent and matching tenprint images. NISTIR 6534. Withdrawn January 2017. https://www.nist.gov/publications/nist-special-database-27-fingerprint-minutiae-latent-and-matching-tenprint-images

105. Fiumara, G., Flanagan, P., Grantham, J. et al. (2018). NIST Special Database 300: Uncompressed plain and rolled images from fingerprint cards. NIST Technical Note 1993.

106. NIST Special Database 302, Nail to Nail Fingerprint Challenge. https://www.nist.gov/itl/iad/image-group/nist-special-database-302

107. NIST biometric special databases and software. https://www.nist.gov/itl/iad/btg/resources/biometric-special-databases-and-software

108. Shehu, Y. I., Ruiz-Garcia, A., Palade, V. and James, A. (2018). Sokoto Coventry Fingerprint Dataset. arXiv 1807.10609. https://arxiv.org/abs/1807.10609 Dataset: https://www.kaggle.com/datasets/ruizgara/socofing

109. ChaLearn LAP fingerprint denoising and inpainting dataset. https://chalearnlap.cvc.uab.cat/dataset/32/description/

110. Thai, D. H., Huckemann, S. and Gottschlich, C. (2016). Filter design and performance evaluation for fingerprint image segmentation. PLOS ONE, 11(5).

111. Vazan, R. Fingerprint dataset collection. https://github.com/robertvazan/fingerprint-datasets

112. FVC-onGoing online evaluation of fingerprint recognition algorithms. https://biolab.csr.unibo.it/fvcongoing

113. A latent fingerprint in the wild database. arXiv 2304.00979. https://arxiv.org/abs/2304.00979

114. Improving automated latent fingerprint identification using extended minutia types. arXiv 1810.09801. https://arxiv.org/abs/1810.09801 Dataset: https://atvs.ii.uam.es/atvs/gcdb_features.html

115. Tsinghua NIST SD27 annotations and results. https://ivg.au.tsinghua.edu.cn/dataset/NIST.php

116. NIST Biometric Image Software (NBIS), including MINDTCT and BOZORTH3.

117. Cappelli, R., Ferrara, M. and Maltoni, D. Minutia Cylinder-Code. University of Bologna.

### Deployment Context

118. Economic and Political Weekly (2019). Aadhaar failures: A tragedy of errors. https://www.epw.in/engage/article/aadhaar-failures-food-services-welfare

119. Analyzing fingerprints of Indian population using image quality: A UIDAI case study (2010).
