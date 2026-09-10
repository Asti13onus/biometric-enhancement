# Draft — NIST Special Database requests (SD302, SD300)

Both are **web request forms**, not emails. No institutional signature — but they are
reviewed by a NIST staff member, so an institutional email address matters.

**Status:** not sent
**Signature required:** none
**Realistic wait:** days to weeks (manual review)

| Dataset | Request URL | Why we want it |
|---|---|---|
| **SD302 (N2N)** | `https://nigos.nist.gov/datasets/sd302/request` | 200 subjects × 10 fingers × 12–18 impressions across **15 sensor types**, plus SD302E latents. The best real multi-sensor set still distributed. Source of clean masters for the wear model, and cross-sensor generalisation evidence. |
| **SD300** | `https://nigos.nist.gov/datasets/sd300/request` | 888 subjects × 10 fingers, scanned ink cards, 500/1000/2000 dpi. Arrested US citizens, **some uncooperative captures → naturally poor quality**. This is the free substitute for a degraded real population if the IAB licence stalls. |

Request **both**. They are separate forms and independent reviews, and SD300 is the one
carrying the naturally-poor-quality captures that make it a fallback for the Rural Indian DB.

---

## Filling the forms

Use your institutional email — a `gmail.com` address is the most likely reason for a slow or
rejected review. Expect fields for name, organisation, email, and intended use.

**Intended-use text** (paste into whichever field asks; trim to fit):

> Master's thesis research on the restoration of fingerprint images degraded by
> occupational wear, ageing and skin damage, for use in large-scale civil identity
> systems. The work develops a physiologically grounded degradation model and an
> uncertainty-aware enhancement method, evaluated with open matchers (NBIS BOZORTH3 and
> SourceAFIS) and NFIQ 2 quality scores. SD302 would provide multi-sensor reference imagery
> and clean source prints; SD300 would provide real captures across a wide quality range,
> including uncooperative captures, as an evaluation set. Data would be stored on
> institutional machines, used only for this research, and not redistributed. Published
> outputs consist of evaluation code and dataset split files, not imagery.

If a form asks about redistribution or publication of images, answer **no** — and note that
our public release is scripts and split files only.

---

## Notes worth having

- **SD302 includes latent imagery (SD302E).** Latent enhancement is explicitly out of scope
  (`STRATEGY.md` §6), and per ADR 0001 any latent-only experiment needs an ADR first.
  Requesting the set is fine; drifting into it is the documented trap.
- **SD300 is 500/1000/2000 dpi.** Do not mix resolutions into a 500-dpi-trained model, and
  never compare NFIQ 2 across resolutions. The manifest records declared dpi per image for
  exactly this reason.
- **SD4, SD9, SD10 and SD27 are gone** — discontinued or withdrawn. Do not request them, and
  cite their unavailability as motivation for the alternative protocol (Contribution 3).
- **SD14** status is uncertain; treat as unavailable. It was the 27k–30k training source for
  DNUNets, FingerGAN and SFP, which is part of why those results cannot be reproduced.

---

## Follow-up

- Log both send dates in `REGISTRY.md`.
- These are forms, so there is no thread to reply into — if there is no response in ~3 weeks,
  re-submit and note the second attempt.
- On arrival: save the terms to `docs/datasets/licences/`, download, manifest, update
  `REGISTRY.md`.
