# Draft — FVC manual segmentation ground truth (Göttingen)

**Paper:** Thai, Huckemann and Gottschlich, *Filter Design and Performance Evaluation for
Fingerprint Image Segmentation*, PLOS ONE 11(5): e0154160, 2016.
**DOI:** `10.1371/journal.pone.0154160`
**Status:** not sent
**Signature required:** none
**Realistic wait:** days

**What it is:** hand-marked segmentation masks for FVC fingerprint images.

**Why it matters:** these are the masks **Cappelli used to train SNFEN** — the current state
of the art, and the baseline the thesis must beat. Reproducing his training setup rather than
only running his released weights depends on this plus the orientation and frequency ground
truth. Small download, high leverage.

---

## Try these first, in order

1. **The PLOS ONE article's own Supporting Information.** PLOS mandates data availability,
   so the masks may be attached to the article or named in its Data Availability statement.
   Open the DOI above and read that statement before anything else.
2. **The Göttingen group's pages** — Institute for Mathematical Stochastics, University of
   Göttingen (Gottschlich's fingerprint software and benchmark pages historically hosted
   downloads). Also check Carsten Gottschlich's personal and GitHub pages.
3. **Only then email**, using the draft below.

---

> **Subject:** Request for the manual FVC segmentation ground truth from your 2016 PLOS ONE paper
>
> Dear Dr Gottschlich / Prof. Huckemann,
>
> I am a master's student at [institution], supervised by [supervisor, title], working on
> the restoration of fingerprints degraded by occupational wear and ageing.
>
> I am writing to ask whether the manually marked segmentation ground truth from *Filter
> Design and Performance Evaluation for Fingerprint Image Segmentation* (PLOS ONE, 2016) is
> still available. I could not locate a download link, and I would rather ask than assume.
>
> The reason I need it specifically: the current state of the art in fingerprint enhancement
> (Cappelli, 2026) is trained using your hand-marked masks together with manually annotated
> orientation and frequency ground truth. My thesis compares against that method, and having
> the same segmentation ground truth is what would let me retrain it on matched data rather
> than only evaluating its released weights — which matters for the comparison to be a fair
> one.
>
> I would cite your paper, use the data only for this research, and not redistribute it.
>
> Thank you for your time, and for having released the annotations in the first place —
> hand-marked ground truth is scarce and disproportionately useful.
>
> With best regards,
> [name]
> [programme], [department]
> [institution]
> [institutional email]

---

## Follow-up
- Record where you found it (or that you had to ask) in `REGISTRY.md` — the availability
  trail is itself material for the review's §13.2.
- On arrival, save terms to `docs/datasets/licences/` and note which FVC databases the masks
  cover. We hold the FVC **"B"** subsets; Cappelli trained on the **"A"** subsets, which are
  not free (they ship with the *Handbook of Fingerprint Recognition*, ~EUR 140 — worth asking
  the department to buy, since it is the standard reference anyway).
