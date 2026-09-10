# Draft — Tsinghua SD27 annotation package

**Page:** `http://ivg.au.tsinghua.edu.cn/dataset/NIST.php`
**Group:** Institute of Information Processing / Intelligent Vision Group, Tsinghua University
(the Feng Jianjiang / Zhou Jie line of fingerprint work)
**Send from:** institutional address
**Signature required:** none known — the page may offer a direct download or a request form;
check it first and only send the email if there is no download link.
**Status:** not sent

**What it is:** manually labelled **orientation fields, ridge-period (frequency) maps,
skeletons, singular points and ROIs** for the NIST SD27 latent images.

**Why we want it even though SD27 itself is withdrawn:** the annotations are ground truth
for exactly the two quantities the literature identifies as the invariant core of the
problem — local ridge orientation and local ridge frequency. Cappelli's SNFEN is trained on
manually annotated orientation fields, frequency maps derived from annotated skeletons, and
manual segmentation masks. Without comparable supervision we can run his released weights
but cannot retrain his method, which weakens the same-data comparison in Phase 1. The
annotations are useful as supervision and as an evaluation reference *independently* of
whether we ever hold the SD27 imagery.

**Note on scope:** SD27 is latent (crime-scene) data, which is out of scope as a target
(`STRATEGY.md` §6). We want the *annotations* as orientation/frequency supervision, not to
start doing latent enhancement. Per ADR 0001, a latent-only experiment would need its own ADR.

---

> **Subject:** Request for the SD27 fingerprint annotation package (orientation, frequency, skeletons)
>
> Dear Professor / colleagues,
>
> I am a master's student at [institution], supervised by [supervisor, title], working on
> the restoration of fingerprints degraded by occupational wear and ageing.
>
> I would like to request the manual annotation package your group has released for the NIST
> SD27 fingerprints — the labelled orientation fields, ridge period maps, skeletons,
> singular points and regions of interest — described at
> `ivg.au.tsinghua.edu.cn/dataset/NIST.php`.
>
> My interest is in the annotations rather than the imagery. Recent work has shown that the
> remaining performance in fingerprint enhancement lies in accurate estimation of local
> ridge orientation and local ridge frequency rather than in more expressive enhancement
> models, and that training such estimators requires manually annotated ground truth of
> precisely the kind your package provides. I would use them as supervision and as an
> evaluation reference for orientation and frequency estimation.
>
> I am aware that NIST SD27 itself was withdrawn from distribution in 2017 and I am not
> requesting the images. If the annotations can only be released to holders of the imagery,
> I would be grateful if you could tell me — that constraint is itself relevant to my
> review, which discusses the consequences of the benchmark's unavailability.
>
> I will cite the associated publications and will not redistribute the data.
>
> Thank you for your time.
>
> With best regards,
> [name]
> [programme], [department]
> [institution]
> [institutional email]

---

## Follow-up
- Check the page for a direct download before emailing — this may need no correspondence.
- Log the outcome in `REGISTRY.md` either way.
- If the annotations require SD27 imagery we cannot obtain, record that: it is a concrete
  instance of the availability crisis the literature review documents (§13.2), and
  strengthens the motivation for Contribution 3.
- Fallback for orientation/frequency supervision: the FVC manual segmentation ground truth
  (see `fvc-segmentation-gt.md`), L3-SF's pore annotations, and Anguli's `-meta` output
  (pattern class plus singular-point coordinates), which we already hold.
