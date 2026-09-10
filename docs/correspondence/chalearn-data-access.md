# Draft — ChaLearn LAP Track 3 data access

**To:** Sergio Escalera — `sergio.escalera.guerrero@gmail.com`
(listed contact on `chalearnlap.cvc.uab.cat/dataset/32/description/`)
**Cc:** your supervisor — a supervisor in the thread makes this read as a lab request
rather than a stranger's
**Send from:** your institutional address, not a personal one
**Status:** not sent

> **Subject:** Registration failing for ChaLearn LAP site — requesting access to the
> fingerprint denoising dataset (Track 3, WCCI'18/ECCV'18)
>
> Dear Prof. Escalera,
>
> I am a master's student working on the restoration of fingerprints degraded by
> occupational wear and ageing, for use in large-scale civil identity systems. The
> ChaLearn LAP Track 3 fingerprint inpainting and denoising dataset is the paired
> benchmark my work needs to compare against, and I would be grateful for access.
>
> I have been unable to register on `chalearnlap.cvc.uab.cat`. Sign-up returns an
> HTTP 500 error, and retrying with the same address then reports that the email
> address is not unique, which suggests the account is created before a later step
> fails. I have tried several addresses with the same result, and the dataset's file
> lists at `/dataset/32/data/55/files/` require a logged-in session.
>
> Could you either point me to a working download route for the training,
> validation and test archives, or let me know whether the registration issue is
> something that can be fixed on your side? I am happy to agree to any terms of use
> and to cite the dataset as requested.
>
> For context on the intended use: my thesis argues that the synthetic degradation
> used to train restoration models — blur, scratches, occlusion and background
> compositing — models crime-scene latent prints rather than occupational ridge wear,
> and proposes a physiologically grounded alternative. The ChaLearn set is the
> baseline condition that argument is measured against, so having the original data
> rather than a regenerated approximation matters for the comparison to be fair to
> the existing literature.
>
> Thank you for your time, and for maintaining the dataset.
>
> With best regards,
> [name]
> [programme, institution]
> [institutional email]

## Notes before sending

- Try the two self-service routes first; they may make this unnecessary:
  1. Log in at `/login/` with the **username** (not the email) from the attempt that
     errored, plus the password set then — the account was likely created before the 500.
  2. The reset page at `/password_reset/` works. Caveat: if their mail backend is what is
     failing, the reset mail will not arrive either.
- If a reply grants access, download the three archives and run
  `python scripts/download_group_b.py chalearn --chalearn-url ...` (or drop the files in
  `data/raw/_archives/`), then `python scripts/make_manifest.py chalearn`.
- If there is no reply by **end of Phase 1 (week 8)**, ADR 0003 triggers and the baseline
  arm is regenerated with Anguli. Record the date the deadline passed.
- Update `docs/datasets/REGISTRY.md` with the send date so the week-4 chase is visible.
