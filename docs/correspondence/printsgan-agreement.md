# Draft — PrintsGAN dataset agreement (MSU)

**To:** Steven Grosz — `groszste@gmail.com`
**Cc:** your supervisor
**Send from:** institutional address
**Signature required:** you (and supervisor if the form asks)
**Status:** not sent
**Realistic wait:** days

**525,000 images — 35,000 distinct fingers × 15 impressions.** Correcting the thesis plan,
which lists PrintsGAN as "released publicly": it is public, but requires a signed agreement
emailed to the author. Agreement form and details at
`https://biometrics.cse.msu.edu/Publications/Databases/MSU_PrintsGAN/`.

**Why we want it:** it is the only large source of **many impressions per identity**, which
Contribution 4 (multi-impression fusion) needs and which ChaLearn does not provide. Also
useful as synthetic pretraining before fine-tuning on real data. Note that Anguli's `-ni`
flag gives us multi-impression data too, so this is valuable rather than essential —
send the request, don't block on it.

---

## Step 1 — get and sign the form

Download the agreement from the dataset page, fill it in, sign it, and have your supervisor
sign if the form provides a line for a faculty sponsor. Send it as a PDF attachment.

## Step 2 — the email

> **Subject:** PrintsGAN dataset agreement — [institution]
>
> Dear Dr Grosz,
>
> I am a master's student at [institution], supervised by [supervisor, title], and I would
> like to request access to the MSU PrintsGAN dataset. The signed agreement is attached.
>
> My thesis concerns the restoration of fingerprints degraded by occupational wear and
> ageing, in the context of large-scale civil identity systems. PrintsGAN is of particular
> interest because it provides many impressions per synthetic identity: one strand of the
> work fuses several poor-quality impressions of the same finger into a single restored
> template, on the observation that operational deployments already recapture on quality
> failure but no published enhancement method exploits it. Large-scale synthetic
> pretraining before fine-tuning on real data is a second intended use.
>
> I will cite the PrintsGAN paper (Engelsma, Grosz and Jain, IEEE TPAMI) in any resulting
> publication, keep the data on institutional machines, and not redistribute it.
>
> Thank you for releasing the dataset.
>
> With best regards,
> [name]
> [programme], [department]
> [institution]
> [institutional email]

---

## On arrival

- ~20–40 GB expected. Check free space first (E: had 476 GB at project start).
- Save the agreement and terms to `docs/datasets/licences/`.
- Add a `printsgan` entry to `DECLARED_DPI` in `scripts/make_manifest.py` before
  manifesting — PrintsGAN's resolution should be taken from its paper, not assumed to be
  500 dpi.
- Update `REGISTRY.md`.
